"""High-level DeepStream pipeline integrating YOLO11 detection and pose estimation."""

# =============================================================================
# Imports
# =============================================================================

import csv
import os
import platform
import sys
import time
import uuid
from ctypes import sizeof, c_float
from datetime import datetime, timezone, timedelta
from queue import Queue, Empty
from threading import Lock, Thread
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from src.uploader import Uploader
from src.send_helper import SendHelper

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load config/deepstream.yaml") from exc

sys.path.append(
    "/opt/nvidia/deepstream/deepstream/lib"
)  # Allow importing DeepStream Python bindings
import pyds

# =============================================================================
# Configuration Constants
# =============================================================================
CONFIG_FILE = "config/deepstream.yaml"

if not os.path.isfile(CONFIG_FILE):
    raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

with open(CONFIG_FILE, "r", encoding="utf-8") as cfg_file:
    raw_config = yaml.safe_load(cfg_file) or {}

if not isinstance(raw_config, dict):
    raise ValueError(f"Config root must be a mapping in {CONFIG_FILE}")


def _require(section: str, key: str):
    if section not in raw_config or key not in raw_config[section]:
        raise KeyError(f"Missing '{key}' in '{section}' section of {CONFIG_FILE}")
    return raw_config[section][key]


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_list(value, label):
    if isinstance(value, (list, tuple)):
        result = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        result = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise TypeError(f"{label} must be a list or comma-separated string")

    if not result:
        raise ValueError(f"{label} must contain at least one value")
    return result


# Source defaults
DEFAULT_SOURCE = _require("arguments", "source")
CAMERA_IDS = _require("arguments", "camera_ids")
DEFAULT_INFER_CONFIG = os.path.abspath(_require("arguments", "config"))
POSE_CONFIG = _require("arguments", "pose_config")
TRACKER_CONFIG = _require("constants", "tracker_config")
TRACKER_LIB = _require("constants", "tracker_lib")

# Output defaults
OUTPUT_MP4 = _require("arguments", "output")
CSV_PATH = _require("constants", "csv_path")

# Streammux and inference sizing
STREAMMUX_WIDTH = int(_require("arguments", "width"))
STREAMMUX_HEIGHT = int(_require("arguments", "height"))
STREAMMUX_BATCH_SIZE = int(_require("constants", "streammux_batch_size"))
GPU_ID = int(_require("arguments", "gpu"))

# Visualization tuning
MAX_DISPLAY_ELEMENTS = int(_require("constants", "max_display_elements"))
NUM_KEYPOINTS = int(_require("constants", "num_keypoints"))
FPS_INTERVAL_SEC = int(_require("constants", "fps_interval_sec"))
RTSP_TIMEOUT_SEC = int(_require("constants", "rtsp_timeout_sec"))
INFER_STRIDE = int(_require("constants", "infer_stride"))
MAX_RECORDING_MINUTES = int(_require("constants", "max_recording_minutes"))
MAX_RECORDING_FRAMES = FPS_INTERVAL_SEC * 60 * MAX_RECORDING_MINUTES
MAX_NO_DET_FRAMES = int(_require("constants", "max_no_det_frames"))

# Event Hub / Database Configuration
EH_MAX_RETRIES = int(_require("constants", "eh_max_retries"))
EH_RETRY_DELAY = float(_require("constants", "eh_retry_delay"))

# Mexico City timezone offset (hours)
MEXICO_TZ_OFFSET = int(_require("constants", "mexico_tz_offset"))
MEXICO_TZ = timezone(timedelta(hours=MEXICO_TZ_OFFSET))

# Parser boolean defaults
DEFAULT_ENABLE_CSV = _as_bool(_require("arguments", "enable_csv"))
DEFAULT_ENABLE_DB = _as_bool(_require("arguments", "enable_db"))

# COCO skeleton connectivity (1-indexed joint pairs)
SKELETON = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
]

# =============================================================================
# Runtime State
# =============================================================================
# Pipeline configuration snapshot shared across helpers
g_config = {
    "width": STREAMMUX_WIDTH,
    "height": STREAMMUX_HEIGHT,
    "gpu_id": GPU_ID,
    "is_jetson": False,
    "enable_csv": DEFAULT_ENABLE_CSV,
    "enable_db": DEFAULT_ENABLE_DB,
}
g_csv_file = None
g_csv_writer = None

# Event Hub / Database state
g_send_helper = None

# Queue buffering batched DB writes
db_queue = Queue(maxsize=50000)

# =============================================================================
# FPS Tracker
# =============================================================================


class FPSTracker:
    """Tracks and reports FPS for a stream."""

    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.start_time = time.time()
        self.frame_count = 0
        self.total_frames = 0
        self.total_time = 0
        self.initialized = False
        self.lock = Lock()

    def update(self):
        with self.lock:
            if not self.initialized:
                self.start_time = time.time()
                self.initialized = True
            else:
                self.frame_count += 1

    def get_fps(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                current_fps = self.frame_count / elapsed
                self.total_time += elapsed
                self.total_frames += self.frame_count
                avg_fps = (
                    self.total_frames / self.total_time if self.total_time > 0 else 0
                )
                self.start_time = time.time()
                self.frame_count = 0
                return current_fps, avg_fps
            return 0.0, 0.0

    def print_callback(self):
        if self.initialized:
            current, avg = self.get_fps()
            print(f"[Stream {self.stream_id}] FPS: {current:.2f} (avg: {avg:.2f})")
        return True


# =============================================================================
# Keypoint Extraction
# =============================================================================


def extract_keypoints(obj_meta):
    """Extract pose keypoints from object metadata."""
    if not hasattr(obj_meta, "mask_params") or obj_meta.mask_params.size <= 0:
        return []

    mask = obj_meta.mask_params
    num_joints = int(mask.size / (sizeof(c_float) * 3))

    gain = min(mask.width / g_config["width"], mask.height / g_config["height"])
    if gain <= 0:
        return []

    pad_x = (mask.width - g_config["width"] * gain) * 0.5
    pad_y = (mask.height - g_config["height"] * gain) * 0.5

    data = mask.get_mask_array()
    keypoints = []
    for i in range(num_joints):
        x = (data[i * 3] - pad_x) / gain
        y = (data[i * 3 + 1] - pad_y) / gain
        conf = data[i * 3 + 2]
        keypoints.append((x, y, conf))

    return keypoints


# =============================================================================
# Visualization
# =============================================================================


def clamp(val, min_val, max_val):
    """Clamp value to range."""
    return int(min(max_val, max(min_val, val)))


def set_bbox_style(obj_meta):
    """Configure bounding box appearance with tracking ID label."""
    width, height = g_config["width"], g_config["height"]
    border_width = 6
    font_size = 18

    rect = obj_meta.rect_params
    rect.border_width = border_width
    rect.border_color.set(0.0, 0.0, 1.0, 1.0)  # Blue

    text = obj_meta.text_params
    text.display_text = f"ID {obj_meta.object_id}"
    text.x_offset = clamp(rect.left - border_width * 0.5, 0, width - 1)
    text.y_offset = clamp(rect.top - font_size * 2, 0, height - 1)

    text.font_params.font_name = "Ubuntu"
    text.font_params.font_size = font_size
    text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # White

    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 1.0, 1.0)  # Blue background


def draw_pose(batch_meta, frame_meta, obj_meta):
    """Draw pose skeleton (keypoints and limbs) on frame."""
    keypoints = extract_keypoints(obj_meta)
    if not keypoints:
        return

    width, height = g_config["width"], g_config["height"]
    display_meta = None

    # Draw keypoint circles
    for x, y, conf in keypoints:
        if conf < 0.5:
            continue

        if display_meta is None or display_meta.num_circles >= MAX_DISPLAY_ELEMENTS:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        circle = display_meta.circle_params[display_meta.num_circles]
        circle.xc = clamp(x, 0, width - 1)
        circle.yc = clamp(y, 0, height - 1)
        circle.radius = 6
        circle.circle_color.set(1.0, 1.0, 1.0, 1.0)  # White
        circle.has_bg_color = 1
        circle.bg_color.set(0.0, 0.0, 1.0, 1.0)  # Blue fill
        display_meta.num_circles += 1

    # Draw skeleton lines
    for joint_a, joint_b in SKELETON:
        idx_a, idx_b = joint_a - 1, joint_b - 1
        if idx_a >= len(keypoints) or idx_b >= len(keypoints):
            continue

        x1, y1, c1 = keypoints[idx_a]
        x2, y2, c2 = keypoints[idx_b]

        if c1 < 0.5 or c2 < 0.5:
            continue

        if display_meta is None or display_meta.num_lines >= MAX_DISPLAY_ELEMENTS:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        line = display_meta.line_params[display_meta.num_lines]
        line.x1, line.y1 = clamp(x1, 0, width - 1), clamp(y1, 0, height - 1)
        line.x2, line.y2 = clamp(x2, 0, width - 1), clamp(y2, 0, height - 1)
        line.line_width = 6
        line.line_color.set(0.0, 0.0, 1.0, 1.0)  # Blue
        display_meta.num_lines += 1


# =============================================================================
# Detection Matching
# =============================================================================


def point_in_bbox(x, y, bx, by, bw, bh):
    """Check if point is inside bounding box."""
    return bx <= x <= bx + bw and by <= y <= by + bh


def find_matching_pose(bbox, pose_detections, threshold=0.85):
    """Find pose detection where majority of keypoints fall inside bbox."""
    bx, by, bw, bh = bbox
    for pose in pose_detections:
        kps = pose["keypoints"]
        inside = sum(1 for x, y, _ in kps if point_in_bbox(x, y, bx, by, bw, bh))
        if inside >= threshold * len(kps):
            return pose
    return None


# =============================================================================
# Buffer Probe (Main Processing)
# =============================================================================


def get_mexico_timestamp():
    """Get current timestamp in Mexico City timezone (ISO format)."""
    return datetime.now(MEXICO_TZ).isoformat()


def format_bbox_for_db(bbox_tuple):
    """
    Convert bbox tuple (x, y, w, h) to dict format for database.
    Values are already in frame coordinate scale.
    """
    x, y, w, h = bbox_tuple
    return {
        "x": float(round(x, 2)),
        "y": float(round(y, 2)),
        "width": float(round(w, 2)),
        "height": float(round(h, 2)),
    }


def format_skeleton_for_db(keypoints):
    """
    Convert keypoints list to dict format for database.
    Keypoints are already in frame coordinate scale (same as width/height).
    """
    if not keypoints:
        return None

    # COCO keypoint names
    kp_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    skeleton = []
    for i, (x, y, conf) in enumerate(keypoints):
        name = kp_names[i] if i < len(kp_names) else f"kp_{i}"
        skeleton.append(
            {
                "name": name,
                "x": float(round(x, 2)),
                "y": float(round(y, 2)),
                "confidence": float(round(conf, 3)),
            }
        )

    return skeleton


def process_detections_for_db(
    frame_num,
    video_id,
    normal_detections,
    pose_detections,
    timestamp,
    camera_id,
    track_to_person_id,
):
    """
    Process detections and send to Event Hub using efficient batched method.
    Uses send_frame_with_detections for optimal throughput.
    """
    global g_send_helper

    if not g_config["enable_db"] or g_send_helper is None:
        return

    if not normal_detections:
        return

    width = g_config["width"]
    height = g_config["height"]
    # Prepare detections with bbox and skeleton in correct format
    prepared_detections = []
    for det in normal_detections:
        # Format bbox
        bbox = format_bbox_for_db(det["bbox"])

        # Match pose and format skeleton
        skeleton = None
        match = find_matching_pose(det["bbox"], pose_detections)
        if match:
            skeleton = format_skeleton_for_db(match["keypoints"])

        prepared_detections.append(
            {
                "track_id": det["track_id"],
                "confidence": float(det.get("confidence", 0.0)),
                "bbox": bbox,
                "skeleton": skeleton,
            }
        )

    # Send all at once using the efficient bulk method
    # This queues all events for async batched sending
    g_send_helper.send_frame_with_detections(
        frame_id=str(uuid.uuid4()),
        video_id=video_id,
        camera_id=camera_id,
        timestamp=timestamp,
        width=width,
        height=height,
        detections=prepared_detections,
        track_to_person_id=track_to_person_id,
    )


def stop_pipeline(pipeline, loop):
    """Send EOS to stop pipeline gracefully."""
    print(f"Stopping pipeline after {RTSP_TIMEOUT_SEC}s timeout")
    pipeline.send_event(Gst.Event.new_eos())
    return False


# =============================================================================
# Pipeline Construction
# =============================================================================


def is_jetson():
    """Detect if running on Jetson platform."""
    return platform.uname()[4] == "aarch64"


def try_set_property(element, key, value):
    """Set property if it exists on element."""
    try:
        if element.find_property(key):
            element.set_property(key, value)
            return True
    except Exception:
        pass
    return False


class CameraPipeline:
    """Encapsulates a single DeepStream pipeline tied to one camera source."""

    def __init__(self, index, source_uri, infer_config, output_path, camera_id, loop):
        self.index = index
        self.source_uri = source_uri
        self.infer_config = infer_config
        self.output_path = output_path
        self.camera_id = camera_id
        self.loop = loop

        # Per-camera runtime state
        self.pipeline = None
        self.streammux = None
        self.splitmuxsink = None
        self.recording = False
        self.frames_without_detections = 0
        self.video_id = None
        self.video_path = None
        self.recording_frames = 0
        self.track_to_person_id = {}
        self.fps_tracker = FPSTracker(index)
        self.bus = None
        self.error = None

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        pipeline = Gst.Pipeline.new(f"deepstream-pose-{self.index}")
        if not pipeline:
            self.error = "Failed to create pipeline"
            return

        streammux = self._create_element("nvstreammux", "streammux")
        source = self._create_source(streammux)
        pgie = self._create_element("nvinfer", "pgie")
        tracker = self._create_element("nvtracker", "tracker")
        sgie = self._create_element("nvinfer", "sgie")
        converter1 = self._create_element("nvvideoconvert", "converter1")
        capsfilter = self._create_element("capsfilter", "capsfilter")
        osd = self._create_element("nvdsosd", "osd")
        converter2 = self._create_element("nvvideoconvert", "converter2")
        encoder = self._create_element("nvv4l2h264enc", "encoder")
        parser = self._create_element("h264parse", "parser")
        sink = self._create_element("splitmuxsink", "sink")

        elements = [
            streammux,
            source,
            pgie,
            tracker,
            sgie,
            converter1,
            capsfilter,
            osd,
            converter2,
            encoder,
            parser,
            sink,
        ]

        if not all(elements):
            self.error = "Failed to create pipeline elements"
            return

        for elem in elements:
            pipeline.add(elem)

        streammux.set_property("batch-size", STREAMMUX_BATCH_SIZE)
        streammux.set_property("batched-push-timeout", 25000)
        streammux.set_property("width", g_config["width"])
        streammux.set_property("height", g_config["height"])
        streammux.set_property(
            "live-source", 0 if self.source_uri.startswith("file://") else 1
        )

        pgie.set_property("config-file-path", self.infer_config)
        pgie.set_property("qos", 0)
        sgie.set_property("config-file-path", POSE_CONFIG)
        sgie.set_property("qos", 0)

        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", g_config["gpu_id"])
        tracker.set_property("ll-lib-file", TRACKER_LIB)
        tracker.set_property("ll-config-file", TRACKER_CONFIG)

        osd.set_property("process-mode", int(pyds.MODE_GPU))
        osd.set_property("qos", 0)
        osd.set_property("display-bbox", 0)
        osd.set_property("display-text", 0)

        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        capsfilter.set_property("caps", caps)

        sink.set_property("location", "null/dummy.mp4")
        sink.set_property("muxer-factory", "qtmux")
        sink.set_property("send-keyframe-requests", True)
        sink.set_property("async-finalize", True)
        sink.set_property("max-size-time", 0)

        if not g_config["is_jetson"]:
            gpu = g_config["gpu_id"]
            mem_type = int(pyds.NVBUF_MEM_CUDA_DEVICE)

            streammux.set_property("nvbuf-memory-type", mem_type)
            streammux.set_property("gpu_id", gpu)
            pgie.set_property("gpu_id", gpu)
            sgie.set_property("gpu_id", gpu)
            tracker.set_property("gpu-id", gpu)
            converter1.set_property("nvbuf-memory-type", mem_type)
            converter1.set_property("gpu_id", gpu)
            osd.set_property("gpu_id", gpu)
            converter2.set_property("nvbuf-memory-type", mem_type)
            converter2.set_property("gpu_id", gpu)
            try_set_property(encoder, "gpu-id", gpu)
            try_set_property(encoder, "bufapi-version", 1)

        try_set_property(encoder, "bitrate", 8000000)
        try_set_property(encoder, "insert-sps-pps", 1)
        try_set_property(encoder, "iframeinterval", 30)
        try_set_property(encoder, "profile", 0)

        links = [
            (streammux, pgie),
            (pgie, tracker),
            (tracker, sgie),
            (sgie, converter1),
            (converter1, capsfilter),
            (capsfilter, osd),
            (osd, converter2),
            (converter2, encoder),
            (encoder, parser),
            (parser, sink),
        ]

        for src, dst in links:
            if not src.link(dst):
                self.error = f"Failed to link {src.get_name()} -> {dst.get_name()}"
                return

        osd_pad = osd.get_static_pad("sink")
        if not osd_pad:
            self.error = "Failed to get OSD sink pad"
            return
        osd_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_buffer_probe, None)

        self.pipeline = pipeline
        self.streammux = streammux
        self.splitmuxsink = sink

        self.bus = pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

    def _create_element(self, factory, name):
        element = Gst.ElementFactory.make(factory, name)
        if not element:
            sys.stderr.write(f"ERROR: Failed to create {factory}\n")
        return element

    def _create_source(self, streammux):
        source = Gst.ElementFactory.make("uridecodebin", f"source-{self.index:04d}")
        if not source:
            return None

        if self.source_uri.startswith("rtsp://"):
            pyds.configure_source_for_ntp_sync(hash(source))

        source.set_property("uri", self.source_uri)

        pad_name = "sink_0"
        sink_pad = streammux.request_pad_simple(pad_name)
        if not sink_pad:
            sys.stderr.write(f"ERROR: Failed to get streammux pad {pad_name}\n")
            return None

        source.connect("pad-added", self._on_pad_added, sink_pad)
        source.connect("child-added", self._on_child_added, None)

        GLib.timeout_add(FPS_INTERVAL_SEC * 2000, self.fps_tracker.print_callback)
        return source

    def _on_child_added(self, child_proxy, obj, name, user_data):
        if "decodebin" in name:
            obj.connect("child-added", self._on_child_added, user_data)
        elif "nvv4l2decoder" in name:
            obj.set_property("drop-frame-interval", 0)
            obj.set_property("num-extra-surfaces", 1)
            obj.set_property("qos", 0)
            if g_config["is_jetson"]:
                obj.set_property("enable-max-performance", 1)
            else:
                obj.set_property("cudadec-memtype", 0)
                obj.set_property("gpu-id", g_config["gpu_id"])

    def _on_pad_added(self, decodebin, pad, streammux_pad):
        caps = pad.get_current_caps() or pad.query_caps()
        struct = caps.get_structure(0)
        features = caps.get_features(0)

        if "video" in struct.get_name():
            if features.contains("memory:NVMM"):
                if pad.link(streammux_pad) != Gst.PadLinkReturn.OK:
                    sys.stderr.write("ERROR: Failed to link source to streammux\n")
            else:
                sys.stderr.write("ERROR: Decoder did not use NVIDIA plugin\n")

    def _on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print(f"Pipeline {self.index} reached EOS")
            self.loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            sys.stderr.write(f"WARNING: {err.message}\n")
        elif msg_type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            sys.stderr.write(f"ERROR: {err.message}\n")
            self.loop.quit()
        return True

    def _osd_buffer_probe(self, pad, info, user_data):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if frame_meta.frame_num % INFER_STRIDE != 0:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue

            normal_detections = []
            pose_detections = []

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                bbox = obj_meta.rect_params
                keypoints = extract_keypoints(obj_meta)

                if keypoints:
                    pose_detections.append(
                        {"keypoints": keypoints, "obj_meta": obj_meta}
                    )
                else:
                    normal_detections.append(
                        {
                            "frame": frame_meta.frame_num,
                            "track_id": obj_meta.object_id,
                            "bbox": (bbox.left, bbox.top, bbox.width, bbox.height),
                            "confidence": obj_meta.confidence,
                            "obj_meta": obj_meta,
                        }
                    )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            has_event = bool(normal_detections)

            if g_config["enable_csv"] and g_csv_writer:
                for det in normal_detections:
                    row = [
                        det["frame"],
                        det["track_id"],
                        round(det["bbox"][0], 2),
                        round(det["bbox"][1], 2),
                        round(det["bbox"][2], 2),
                        round(det["bbox"][3], 2),
                    ]

                    match = find_matching_pose(det["bbox"], pose_detections)
                    if match:
                        for x, y, c in match["keypoints"]:
                            row.extend([round(x, 2), round(y, 2), round(c, 3)])

                    g_csv_writer.writerow(row)

            if frame_meta.frame_num % INFER_STRIDE == 0:
                if has_event:
                    timestamp = get_mexico_timestamp()
                    self.frames_without_detections = 0
                    self.recording_frames += 1
                    if not self.recording:
                        print("new clip")
                        self.video_id = str(uuid.uuid4())
                        self.video_path = f"output/{self.camera_id}/temp_{timestamp}.mp4"
                        self.splitmuxsink.set_property("location", self.video_path)
                        self.splitmuxsink.emit("split-now")
                        self.recording = True
                        self.recording_frames = 0
                    self._enqueue_db_event(
                        frame_meta.frame_num,
                        normal_detections,
                        pose_detections,
                        timestamp,
                    )

                    if self.recording and self.recording_frames > MAX_RECORDING_FRAMES:
                        print("splitting long clip")
                        final_path = self.video_path.replace("temp_", "")
                        os.rename(self.video_path, final_path)
                        self.video_id = str(uuid.uuid4())
                        self.video_path = f"output/{self.camera_id}/temp_{timestamp}.mp4"
                        self.splitmuxsink.set_property("location", self.video_path)
                        self.splitmuxsink.emit("split-now")
                        self.recording_frames = 0
                else:
                    self.frames_without_detections += 1
                    if (
                        self.frames_without_detections >= MAX_NO_DET_FRAMES
                        and self.recording
                    ):
                        print("end clip")
                        self.splitmuxsink.set_property("location", "null/dummy.mp4")
                        self.splitmuxsink.emit("split-now")
                        self.recording = False
                        if self.video_path:
                            final_path = self.video_path.replace("temp_", "")
                            os.rename(self.video_path, final_path)
                        self.video_path = None
                        self.video_id = None

            self.fps_tracker.update()

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _enqueue_db_event(
        self, frame_num, normal_detections, pose_detections, timestamp
    ):
        try:
            db_queue.put_nowait(
                (
                    frame_num,
                    self.video_id,
                    normal_detections,
                    pose_detections,
                    timestamp,
                    self.camera_id,
                    self.track_to_person_id,
                )
            )
        except Exception:
            pass

    def start(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)


def init_csv():
    """Initialize CSV file with header."""
    global g_csv_file, g_csv_writer

    if not g_config["enable_csv"]:
        print("CSV output disabled")
        return

    g_csv_file = open(CSV_PATH, "w", newline="")
    g_csv_writer = csv.writer(g_csv_file)

    header = ["frame", "track_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    for i in range(NUM_KEYPOINTS):
        header.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])

    g_csv_writer.writerow(header)
    print(f"CSV output enabled: {CSV_PATH}")


def init_event_hub():
    """Initialize Event Hub connection for database writes."""
    global g_send_helper

    if not g_config["enable_db"]:
        print("Database output disabled")
        return

    try:
        g_send_helper = SendHelper()
        print(f"Event Hub connection initialized for database writes")
    except Exception as e:
        print(f"ERROR: Failed to initialize Event Hub: {e}")
        print("Database writes disabled.")
        g_config["enable_db"] = False
        g_send_helper = None


def cleanup_event_hub():
    """Close Event Hub connection and print stats."""
    global g_send_helper

    if g_send_helper is not None:
        try:
            # Print stats before closing
            stats = g_send_helper.get_stats()
            print(f"\n[Event Hub Stats]")
            print(f"  Events queued:  {stats['events_queued']}")
            print(f"  Events sent:    {stats['events_sent']}")
            print(f"  Batches sent:   {stats['batches_sent']}")
            print(f"  Errors:         {stats['errors']}")
            print(f"  Queue pending:  {stats['queue_size']}")

            # Flush remaining events (waits indefinitely)
            if stats["queue_size"] > 0:
                print(f"  Flushing {stats['queue_size']} pending events...")
                g_send_helper.flush()

            g_send_helper.close()
        except Exception as e:
            print(f"WARNING: Error closing Event Hub: {e}")
        g_send_helper = None


# =============================================================================
# Main
# =============================================================================


def db_worker():
    """Background worker that processes DB queue."""
    while True:
        try:
            item = db_queue.get(timeout=1.0)
            (
                frame_num,
                video_id,
                normal,
                pose,
                timestamp,
                camera_id,
                track_map,
            ) = item
            process_detections_for_db(
                frame_num, video_id, normal, pose, timestamp, camera_id, track_map
            )
        except Empty:
            continue
        except Exception as e:
            print(f"DB worker error: {e}")
        finally:
            try:
                db_queue.task_done()
            except ValueError:
                pass


def Azure_worker(camera_id):
    """Upload finalized clips to Azure storage in a dedicated thread."""
    uploader = Uploader(camera_id=camera_id)
    uploader.loadProcess()


def main():
    """Main entry point."""
    global g_config

    Gst.init(None)
    loop = GLib.MainLoop()

    width = STREAMMUX_WIDTH
    height = STREAMMUX_HEIGHT
    gpu_id = GPU_ID
    enable_csv = DEFAULT_ENABLE_CSV
    enable_db = DEFAULT_ENABLE_DB
    output_path = OUTPUT_MP4
    infer_config = DEFAULT_INFER_CONFIG
    source_uris = _as_list(DEFAULT_SOURCE, "arguments.source")
    camera_ids = _as_list(CAMERA_IDS, "arguments.camera_ids")
    if len(camera_ids) != len(source_uris):
        raise ValueError("camera_ids must match number of sources")

    g_config["width"] = width
    g_config["height"] = height
    g_config["gpu_id"] = gpu_id
    g_config["is_jetson"] = is_jetson()
    g_config["enable_csv"] = enable_csv
    g_config["enable_db"] = enable_db

    if not source_uris:
        sys.stderr.write("ERROR: No sources provided\n")
        return 1

    pipelines = []
    for idx, src in enumerate(source_uris):
        camera_pipeline = CameraPipeline(
            idx, src, infer_config, output_path, camera_ids[idx], loop
        )
        if camera_pipeline.error:
            sys.stderr.write(f"ERROR: {camera_pipeline.error}\n")
            return 1
        pipelines.append(camera_pipeline)

    # Initialize outputs
    init_csv()
    init_event_hub()

    # Start DB worker thread (daemon so it auto-exits)
    db_thread = Thread(target=db_worker, daemon=True, name="DBWorker")
    db_thread.start()

    # Start uploader worker thread
    for cam_id in camera_ids:
        Thread(
            target=Azure_worker,
            args=(cam_id,),
            daemon=True,
            name=f"UploaderWorker-{cam_id}",
        ).start()

    # Print configuration
    print(f"\n{'='*50}")
    print(f"SOURCES:   {', '.join(source_uris)}")
    print(f"CONFIG:    {infer_config}")
    print(f"OUTPUT:    {output_path}")
    print(f"SIZE:      {width}x{height}")
    print(f"GPU:       {gpu_id}")
    print(f"JETSON:    {g_config['is_jetson']}")
    print(f"CSV:       {'enabled' if g_config['enable_csv'] else 'disabled'}")
    print(f"DATABASE:  {'enabled' if g_config['enable_db'] else 'disabled'}")
    if len(pipelines) == 1:
        print(f"CAMERA_ID: {pipelines[0].camera_id}")
    else:
        camera_ids = ", ".join(str(p.camera_id) for p in pipelines)
        print(f"CAMERA_IDS: {camera_ids}")
    print(f"{'='*50}\n")

    try:
        while True:
            print("Starting pipelines...")
            for pipe in pipelines:
                pipe.start()
            loop.run()
            print("Pipelines stopped, restarting in 3s...")
            for pipe in pipelines:
                pipe.stop()
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nInterrupted")

    # Cleanup
    for pipe in pipelines:
        pipe.stop()

    if g_csv_file:
        g_csv_file.close()

    print(f"\nOutput saved: {output_path}")
    if g_config["enable_csv"]:
        print(f"Metadata saved: {CSV_PATH}")
    if g_config["enable_db"]:
        print(f"Database records sent via Event Hub")
        total_persons = sum(len(p.track_to_person_id) for p in pipelines)
        print(f"Total tracked persons: {total_persons}")
    print()

    print("Waiting for DB queue to flush...")
    db_queue.join()
    print("All DB events sent")
    cleanup_event_hub()

    return 0


if __name__ == "__main__":
    sys.exit(main())

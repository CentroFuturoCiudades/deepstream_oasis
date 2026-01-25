#!/usr/bin/env python3
"""
DeepStream YOLO11 + Pose estimation pipeline.
Processes video sources and outputs annotated MP4 with bounding boxes and pose keypoints.

Usage:
    python3 deepstream_rtsp.py -s rtsp://... -c config.txt -o output.mp4
"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import os
import sys
import csv
import time
import platform
import argparse
import uuid
from datetime import datetime, timezone, timedelta
from threading import Lock, Thread
from ctypes import sizeof, c_float
from queue import Queue, Empty

sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
import pyds


# Import SendHelper for Event Hub integration
try:
    from send_helper import SendHelper
    SEND_HELPER_AVAILABLE = True
except ImportError:
    SEND_HELPER_AVAILABLE = False
    print("WARNING: send_helper not available. Database writes disabled.")

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SOURCE = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
DEFAULT_INFER_CONFIG = os.path.abspath("config_infer_primary_yolo11.txt")
POSE_CONFIG = "config_infer_primary_yolo11_pose.txt"
TRACKER_CONFIG = "config_tracker_NvByteTrack.yml"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_nvmultiobjecttracker.so"

OUTPUT_MP4 = "out_yolo11_pose.mp4"
CSV_PATH = "metadata.csv"

STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
STREAMMUX_BATCH_SIZE = 1
GPU_ID = 0

MAX_DISPLAY_ELEMENTS = 16
NUM_KEYPOINTS = 17
FPS_INTERVAL_SEC = 5
RTSP_TIMEOUT_SEC = 10
INFER_STRIDE = 3

# Event Hub / Database Configuration
CAMERA_ID = "-1"  # Hardcoded camera ID
EH_MAX_RETRIES = 3
EH_RETRY_DELAY = 0.5  # seconds

# Mexico City timezone (UTC-6)
MEXICO_TZ = timezone(timedelta(hours=-6))

# COCO skeleton connectivity (1-indexed joint pairs)
SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
    (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)
]

# =============================================================================
# Global State
# =============================================================================

g_config = {
    "width": STREAMMUX_WIDTH,
    "height": STREAMMUX_HEIGHT,
    "gpu_id": GPU_ID,
    "is_jetson": False,
    "enable_csv": True,
    "enable_db": True,
}
g_fps_trackers = {}
g_csv_file = None
g_csv_writer = None

# Event Hub / Database state
g_send_helper = None
g_track_to_person_id = {}  # Maps track_id -> person_id (persists while script runs)

db_queue = Queue(maxsize=10000)

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
                avg_fps = self.total_frames / self.total_time if self.total_time > 0 else 0
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


def send_to_event_hub_with_retry(func, *args, **kwargs):
    """
    Execute a send function with retry logic.
    Retries up to EH_MAX_RETRIES times, logs errors, and continues.
    """
    for attempt in range(EH_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < EH_MAX_RETRIES - 1:
                time.sleep(EH_RETRY_DELAY)
            else:
                print(f"ERROR: Failed to send to Event Hub after {EH_MAX_RETRIES} attempts: {e}")
    return None


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
        "height": float(round(h, 2))
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
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    skeleton = []
    for i, (x, y, conf) in enumerate(keypoints):
        name = kp_names[i] if i < len(kp_names) else f"kp_{i}"
        skeleton.append({
            "name": name,
            "x": float(round(x, 2)),
            "y": float(round(y, 2)),
            "confidence": float(round(conf, 3))
        })
    
    return skeleton


def process_detections_for_db(frame_num, normal_detections, pose_detections):
    """
    Process detections and send to Event Hub using efficient batched method.
    Uses send_frame_with_detections for optimal throughput.
    """
    global g_track_to_person_id, g_send_helper
    
    if not g_config["enable_db"] or g_send_helper is None:
        return
    
    if not normal_detections:
        return
    
    # Generate frame record
    frame_id = str(uuid.uuid4())
    timestamp = get_mexico_timestamp()
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
        
        prepared_detections.append({
            "track_id": det["track_id"],
            "confidence": float(det.get("confidence", 0.0)),
            "bbox": bbox,
            "skeleton": skeleton,
        })
    
    # Send all at once using the efficient bulk method
    # This queues all events for async batched sending
    g_send_helper.send_frame_with_detections(
        frame_id=frame_id,
        camera_id=CAMERA_ID,
        timestamp=timestamp,
        width=width,
        height=height,
        detections=prepared_detections,
        track_to_person_id=g_track_to_person_id,
    )


def osd_buffer_probe(pad, info, user_data):
    """Process each frame: extract detections, match poses, write CSV/DB, draw overlays."""

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

        print(frame_meta.frame_num)


        # Separate detections by type
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
                pose_detections.append({"keypoints": keypoints, "obj_meta": obj_meta})
            else:
                normal_detections.append({
                    "frame": frame_meta.frame_num,
                    "track_id": obj_meta.object_id,
                    "bbox": (bbox.left, bbox.top, bbox.width, bbox.height),
                    "confidence": obj_meta.confidence,
                    "obj_meta": obj_meta
                })

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Process detections and write to CSV (if enabled)
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
        
        # Send detections to database via Event Hub (non-blocking queue)
        try:
            db_queue.put_nowait((
                frame_meta.frame_num,
                normal_detections,
                pose_detections
            ))
        except:
            pass  # Queue full, drop silently to avoid blocking pipeline

        
        # Set bbox style for visualization
        for det in normal_detections:
            set_bbox_style(det["obj_meta"])

        # Draw pose skeletons
        for pose in pose_detections:
            draw_pose(batch_meta, frame_meta, pose["obj_meta"])

        # Update FPS counter
        if frame_meta.source_id in g_fps_trackers:
            g_fps_trackers[frame_meta.source_id].update()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# =============================================================================
# GStreamer Callbacks
# =============================================================================

def on_child_added(child_proxy, obj, name, user_data):
    """Configure decoder elements as they are added."""
    if "decodebin" in name:
        obj.connect("child-added", on_child_added, user_data)
    elif "nvv4l2decoder" in name:
        obj.set_property("drop-frame-interval", 0)
        obj.set_property("num-extra-surfaces", 1)
        obj.set_property("qos", 0)
        if g_config["is_jetson"]:
            obj.set_property("enable-max-performance", 1)
        else:
            obj.set_property("cudadec-memtype", 0)
            obj.set_property("gpu-id", g_config["gpu_id"])


def on_pad_added(decodebin, pad, streammux_pad):
    """Link decoder output to streammux when pad becomes available."""
    caps = pad.get_current_caps() or pad.query_caps()
    struct = caps.get_structure(0)
    features = caps.get_features(0)

    if "video" in struct.get_name():
        if features.contains("memory:NVMM"):
            if pad.link(streammux_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write("ERROR: Failed to link source to streammux\n")
        else:
            sys.stderr.write("ERROR: Decoder did not use NVIDIA plugin\n")


def on_bus_message(bus, message, loop):
    """Handle pipeline messages."""
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        sys.stderr.write(f"WARNING: {err.message}\n")
    elif msg_type == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        sys.stderr.write(f"ERROR: {err.message}\n")
        loop.quit()
    return True


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


def create_source(stream_id, uri, streammux):
    """Create URI decode bin for video source."""
    source = Gst.ElementFactory.make("uridecodebin", f"source-{stream_id:04d}")
    if not source:
        return None

    if uri.startswith("rtsp://"):
        pyds.configure_source_for_ntp_sync(hash(source))

    source.set_property("uri", uri)

    pad_name = f"sink_{stream_id}"
    sink_pad = streammux.request_pad_simple(pad_name)
    if not sink_pad:
        sys.stderr.write(f"ERROR: Failed to get streammux pad {pad_name}\n")
        return None

    source.connect("pad-added", on_pad_added, sink_pad)
    source.connect("child-added", on_child_added, None)

    # Setup FPS tracking
    g_fps_trackers[stream_id] = FPSTracker(stream_id)
    GLib.timeout_add(FPS_INTERVAL_SEC * 2000, g_fps_trackers[stream_id].print_callback)

    return source


def create_element(factory, name):
    """Create GStreamer element with error checking."""
    element = Gst.ElementFactory.make(factory, name)
    if not element:
        sys.stderr.write(f"ERROR: Failed to create {factory}\n")
    return element


def build_pipeline(source_uri, infer_config, output_path):
    """Construct the complete GStreamer pipeline."""
    pipeline = Gst.Pipeline.new("deepstream-pose")
    if not pipeline:
        return None, "Failed to create pipeline"

    # Create elements
    streammux = create_element("nvstreammux", "streammux")
    source = create_source(0, source_uri, streammux)
    pgie = create_element("nvinfer", "pgie")
    tracker = create_element("nvtracker", "tracker")
    sgie = create_element("nvinfer", "sgie")
    converter1 = create_element("nvvideoconvert", "converter1")
    capsfilter = create_element("capsfilter", "capsfilter")
    osd = create_element("nvdsosd", "osd")
    converter2 = create_element("nvvideoconvert", "converter2")
    encoder = create_element("nvv4l2h264enc", "encoder")
    parser = create_element("h264parse", "parser")
    muxer = create_element("qtmux", "muxer")
    sink = create_element("filesink", "sink")

    elements = [streammux, source, pgie, tracker, sgie, converter1,
                capsfilter, osd, converter2, encoder, parser, muxer, sink]

    if not all(elements):
        return None, "Failed to create pipeline elements"

    # Add elements to pipeline
    for elem in elements:
        pipeline.add(elem)

    # Configure streammux
    streammux.set_property("batch-size", STREAMMUX_BATCH_SIZE)
    streammux.set_property("batched-push-timeout", 25000)
    streammux.set_property("width", g_config["width"])
    streammux.set_property("height", g_config["height"])
    streammux.set_property("live-source", 0 if source_uri.startswith("file://") else 1)

    # Configure inference
    pgie.set_property("config-file-path", infer_config)
    pgie.set_property("qos", 0)
    sgie.set_property("config-file-path", POSE_CONFIG)
    sgie.set_property("qos", 0)

    # Configure tracker
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", g_config["gpu_id"])
    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", TRACKER_CONFIG)

    # Configure OSD
    osd.set_property("process-mode", int(pyds.MODE_GPU))
    osd.set_property("qos", 0)

    # Configure encoder input caps
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    capsfilter.set_property("caps", caps)

    # Configure output
    sink.set_property("location", output_path)
    sink.set_property("sync", 0)
    sink.set_property("async", 0)

    # GPU-specific settings for dGPU
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

    # Encoder settings
    try_set_property(encoder, "bitrate", 8000000)
    try_set_property(encoder, "insert-sps-pps", 1)
    try_set_property(encoder, "iframeinterval", 30)
    try_set_property(encoder, "profile", 0)

    # Link pipeline
    links = [
        (streammux, pgie), (pgie, tracker), (tracker, sgie), (sgie, converter1),
        (converter1, capsfilter), (capsfilter, osd), (osd, converter2),
        (converter2, encoder), (encoder, parser), (parser, muxer), (muxer, sink)
    ]

    for src, dst in links:
        if not src.link(dst):
            return None, f"Failed to link {src.get_name()} -> {dst.get_name()}"

    # Add probe for processing
    osd_pad = osd.get_static_pad("sink")
    if not osd_pad:
        return None, "Failed to get OSD sink pad"
    osd_pad.add_probe(Gst.PadProbeType.BUFFER, osd_buffer_probe, None)

    return pipeline, None


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
    
    if not SEND_HELPER_AVAILABLE:
        print("WARNING: SendHelper not available. Database writes disabled.")
        g_config["enable_db"] = False
        return
    
    try:
        g_send_helper = SendHelper()
        print(f"Event Hub connection initialized (camera_id={CAMERA_ID})")
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
            if stats['queue_size'] > 0:
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
            frame_num, normal, pose = item
            process_detections_for_db(frame_num, normal, pose)
        except Empty:
            continue
        except Exception as e:
            print(f"DB worker error: {e}")
        finally:
            try:
                db_queue.task_done()
            except ValueError:
                pass


def main(source_uri, infer_config, output_path, width, height, gpu_id, enable_csv, enable_db):
    """Main entry point."""
    global g_config

    Gst.init(None)

    g_config["width"] = width
    g_config["height"] = height
    g_config["gpu_id"] = gpu_id
    g_config["is_jetson"] = is_jetson()
    g_config["enable_csv"] = enable_csv
    g_config["enable_db"] = enable_db

    # Build pipeline
    pipeline, error = build_pipeline(source_uri, infer_config, output_path)
    if error:
        sys.stderr.write(f"ERROR: {error}\n")
        return 1

    # Setup bus
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message, loop)

    # Initialize outputs
    init_csv()
    init_event_hub()
    
    # Start DB worker thread (daemon so it auto-exits)
    db_thread = Thread(target=db_worker, daemon=True, name="DBWorker")
    db_thread.start()

    # Print configuration
    print(f"\n{'='*50}")
    print(f"SOURCE:    {source_uri}")
    print(f"CONFIG:    {infer_config}")
    print(f"OUTPUT:    {output_path}")
    print(f"SIZE:      {width}x{height}")
    print(f"GPU:       {gpu_id}")
    print(f"JETSON:    {g_config['is_jetson']}")
    print(f"CSV:       {'enabled' if g_config['enable_csv'] else 'disabled'}")
    print(f"DATABASE:  {'enabled' if g_config['enable_db'] else 'disabled'}")
    print(f"CAMERA_ID: {CAMERA_ID}")
    print(f"{'='*50}\n")

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Auto-stop for RTSP sources
    if source_uri.startswith("rtsp://"):
        GLib.timeout_add_seconds(RTSP_TIMEOUT_SEC, stop_pipeline, pipeline, loop)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted")

    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    
    if g_csv_file:
        g_csv_file.close()
    

    print(f"\nOutput saved: {output_path}")
    if g_config["enable_csv"]:
        print(f"Metadata saved: {CSV_PATH}")
    if g_config["enable_db"]:
        print(f"Database records sent via Event Hub")
        print(f"Total tracked persons: {len(g_track_to_person_id)}")
    print()

    print("Waiting for DB queue to flush...")
    db_queue.join()
    print("All DB events sent")
    cleanup_event_hub()

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepStream YOLO11 + Pose estimation pipeline"
    )
    parser.add_argument("-s", "--source", default=DEFAULT_SOURCE,
                        help="Source URI (file:///... or rtsp://...)")
    parser.add_argument("-c", "--config", default=DEFAULT_INFER_CONFIG,
                        help="Primary inference config path")
    parser.add_argument("-o", "--output", default=OUTPUT_MP4,
                        help="Output MP4 path")
    parser.add_argument("-W", "--width", type=int, default=STREAMMUX_WIDTH,
                        help="Processing width")
    parser.add_argument("-H", "--height", type=int, default=STREAMMUX_HEIGHT,
                        help="Processing height")
    parser.add_argument("-g", "--gpu", type=int, default=GPU_ID,
                        help="GPU device ID")
    parser.add_argument("--enable-csv", action="store_true", default=True,
                        help="Enable CSV metadata output (default: True)")
    parser.add_argument("--disable-csv", action="store_true", default=False,
                        help="Disable CSV metadata output")
    parser.add_argument("--enable-db", action="store_true", default=True,
                        help="Enable database output via Event Hub (default: True)")
    parser.add_argument("--disable-db", action="store_true", default=False,
                        help="Disable database output via Event Hub")

    args = parser.parse_args()

    if not args.source:
        sys.stderr.write("ERROR: Source URI required\n")
        sys.exit(1)

    if not os.path.isfile(args.config):
        sys.stderr.write(f"ERROR: Config not found: {args.config}\n")
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args()
    
    # Handle enable/disable flags (disable takes precedence)
    enable_csv = args.enable_csv and not args.disable_csv
    enable_db = args.enable_db and not args.disable_db
    
    sys.exit(main(
        source_uri=args.source,
        infer_config=args.config,
        output_path=args.output,
        width=args.width,
        height=args.height,
        gpu_id=args.gpu,
        enable_csv=enable_csv,
        enable_db=enable_db,
    ))

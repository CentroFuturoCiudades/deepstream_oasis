"""DeepStream pipeline wiring and buffer probes."""

from __future__ import annotations

import platform
import sys
import uuid
from typing import Any, List, Sequence

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
import pyds

from .analytics import FPSTracker, RecordingConfig, RecordingManager
from .config import AppConfig, RuntimeConfig, current_timestamp
from .metadata import ClipMetadataWriter
from .outputs import OutputManager
from .vision import SKELETON, clamp, extract_keypoints


def is_jetson() -> bool:
    return platform.uname()[4] == "aarch64"


def try_set_property(element: Gst.Element, key: str, value: Any) -> bool:
    try:
        if element.find_property(key):
            element.set_property(key, value)
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    return False


class CameraPipeline:
    """Encapsulates a DeepStream pipeline tied to one camera source."""

    def __init__(
        self,
        index: int,
        source_uri: str,
        camera_id: str,
        app_config: AppConfig,
        runtime: RuntimeConfig,
        outputs: OutputManager,
        db_queue,
        loop: GLib.MainLoop,
    ) -> None:
        self.index = index
        self.source_uri = source_uri
        self.camera_id = camera_id
        self.app_config = app_config
        self.runtime = runtime
        self.outputs = outputs
        self.db_queue = db_queue
        self.loop = loop

        self.pipeline: Gst.Pipeline | None = None
        self.streammux: Gst.Element | None = None
        self.splitmuxsink: Gst.Element | None = None
        self.bus = None
        self.error: str | None = None

        self.recording: RecordingManager | None = None
        self.track_to_person_id: dict[int, str] = {}
        self.fps_tracker = FPSTracker(index)
        self.metadata_writer = ClipMetadataWriter(camera_id=self.camera_id)

        self._initialize_pipeline()

    # ------------------------------------------------------------------
    # Pipeline setup
    # ------------------------------------------------------------------
    def _initialize_pipeline(self) -> None:
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

        constants = self.app_config.constants

        streammux.set_property("batch-size", constants.streammux_batch_size)
        streammux.set_property("batched-push-timeout", 25000)
        streammux.set_property("width", self.runtime.width)
        streammux.set_property("height", self.runtime.height)
        streammux.set_property(
            "live-source", 0 if self.source_uri.startswith("file://") else 1
        )

        pgie.set_property("config-file-path", self.app_config.arguments.infer_config)
        pgie.set_property("qos", 0)
        sgie.set_property("config-file-path", self.app_config.arguments.pose_config)
        sgie.set_property("qos", 0)

        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", self.runtime.gpu_id)
        tracker.set_property("ll-lib-file", constants.tracker_lib)
        tracker.set_property("ll-config-file", constants.tracker_config)

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

        if not self.runtime.is_jetson:
            gpu = self.runtime.gpu_id
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

        try_set_property(encoder, "bitrate", 8_000_000)
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

        for src_elem, dst_elem in links:
            if not src_elem.link(dst_elem):
                self.error = (
                    f"Failed to link {src_elem.get_name()} -> {dst_elem.get_name()}"
                )
                return

        osd_pad = osd.get_static_pad("sink")
        if not osd_pad:
            self.error = "Failed to get OSD sink pad"
            return
        osd_pad.add_probe(Gst.PadProbeType.BUFFER, self._osd_buffer_probe, None)

        self.pipeline = pipeline
        self.streammux = streammux
        self.splitmuxsink = sink
        self.recording = RecordingManager(
            camera_id=self.camera_id,
            splitmuxsink=sink,
            config=RecordingConfig(
                max_recording_frames=constants.max_recording_frames,
                max_no_detection_frames=constants.max_no_det_frames,
            ),
            metadata_writer=self.metadata_writer,
        )

        self.bus = pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_bus_message)

    # ------------------------------------------------------------------
    # Element helpers
    # ------------------------------------------------------------------
    def _create_element(self, factory: str, name: str) -> Gst.Element | None:
        element = Gst.ElementFactory.make(factory, name)
        if not element:
            sys.stderr.write(f"ERROR: Failed to create {factory}\n")
        return element

    def _create_source(self, streammux: Gst.Element) -> Gst.Element | None:
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

        return source

    def _on_child_added(self, child_proxy, obj, name, user_data) -> None:
        if "decodebin" in name:
            obj.connect("child-added", self._on_child_added, user_data)
        elif "nvv4l2decoder" in name:
            obj.set_property("drop-frame-interval", 0)
            obj.set_property("num-extra-surfaces", 1)
            obj.set_property("qos", 0)
            if self.runtime.is_jetson:
                obj.set_property("enable-max-performance", 1)
            else:
                obj.set_property("cudadec-memtype", 0)
                obj.set_property("gpu-id", self.runtime.gpu_id)

    def _on_pad_added(self, decodebin, pad, streammux_pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps()
        structure = caps.get_structure(0)
        features = caps.get_features(0)

        if "video" in structure.get_name():
            if features.contains("memory:NVMM"):
                if pad.link(streammux_pad) != Gst.PadLinkReturn.OK:
                    sys.stderr.write("ERROR: Failed to link source to streammux\n")
            else:
                sys.stderr.write("ERROR: Decoder did not use NVIDIA plugin\n")

    def _on_bus_message(self, bus, message) -> bool:
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print(f"Pipeline {self.index} reached EOS")
            self.loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            err, _dbg = message.parse_warning()
            sys.stderr.write(f"WARNING: {err.message}\n")
        elif msg_type == Gst.MessageType.ERROR:
            err, _dbg = message.parse_error()
            sys.stderr.write(f"ERROR: {err.message}\n")
            self.loop.quit()
        return True

    # ------------------------------------------------------------------
    # Buffer probe
    # ------------------------------------------------------------------
    def _osd_buffer_probe(self, pad, info, user_data) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
            except StopIteration:
                break

            if frame_meta.frame_num % self.app_config.constants.infer_stride != 0:
                try:
                    frame_meta_list = frame_meta_list.next
                except StopIteration:
                    break
                continue

            normal_detections: List[dict] = []
            pose_detections: List[dict] = []

            obj_meta_list = frame_meta.obj_meta_list
            while obj_meta_list:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)
                except StopIteration:
                    break

                bbox = obj_meta.rect_params
                keypoints = extract_keypoints(
                    obj_meta.mask_params,
                    self.runtime.width,
                    self.runtime.height,
                )

                if keypoints:
                    pose_detections.append(
                        {"keypoints": keypoints, "obj_meta": obj_meta}
                    )
                else:
                    track_id = obj_meta.object_id
                    person_id = None
                    if track_id is not None:
                        person_id = self.track_to_person_id.setdefault(
                            track_id, str(uuid.uuid4())
                        )

                    normal_detections.append(
                        {
                            "frame": frame_meta.frame_num,
                            "track_id": track_id,
                            "bbox": (
                                bbox.left,
                                bbox.top,
                                bbox.width,
                                bbox.height,
                            ),
                            "confidence": obj_meta.confidence,
                            "person_id": person_id,
                            "obj_meta": obj_meta,
                        }
                    )

                try:
                    obj_meta_list = obj_meta_list.next
                except StopIteration:
                    break
            """
            for detection in normal_detections:
                set_bbox_style(
                    detection["obj_meta"], self.runtime.width, self.runtime.height
                )

            display_meta = None
            for pose in pose_detections:
                display_meta = draw_pose(
                    batch_meta,
                    frame_meta,
                    pose["obj_meta"],
                    pose["keypoints"],
                    display_meta,
                    self.runtime.width,
                    self.runtime.height,
                )

            self.outputs.write_csv_rows(
                frame_meta.frame_num,
                normal_detections,
                pose_detections,
            )"""

            has_event = bool(normal_detections)
            if has_event:
                timestamp = current_timestamp(self.app_config.constants)
                if self.recording:
                    self.recording.on_detections(timestamp)
                self._enqueue_db_event(
                    frame_meta.frame_num,
                    normal_detections,
                    pose_detections,
                    timestamp,
                )
                if self.recording:
                    self.recording.finalize_detection_window(
                        timestamp
                    )  # Slip long clips
            else:
                if self.recording:
                    self.recording.on_no_detections()

            self.fps_tracker.update()

            try:
                frame_meta_list = frame_meta_list.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _enqueue_db_event(
        self,
        frame_num: int,
        normal_detections: Sequence[dict],
        pose_detections: Sequence[dict],
        timestamp: str,
    ) -> None:

        payload = (
            frame_num,
            self.recording.video_id,
            normal_detections,
            pose_detections,
            timestamp,
            self.camera_id,
            self.track_to_person_id,
        )

        if self.recording and self.recording.video_id:
            for det in normal_detections:
                self.metadata_writer.record_detection(
                    video_id=self.recording.video_id,
                    track_id=det.get("track_id"),
                    person_id=det.get("person_id"),
                    timestamp=timestamp,
                    bbox=det.get("bbox", (0, 0, 0, 0)),
                )

        try:
            self.db_queue.put_nowait(payload)
        except Exception:  # pylint: disable=broad-except
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.pipeline:
            self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self) -> None:
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.recording:
            self.recording.finalize_clip()


def draw_pose(
    batch_meta,
    frame_meta,
    obj_meta,
    keypoints,
    display_meta,
    frame_width: int,
    frame_height: int,
):
    if not keypoints:
        return display_meta

    width, height = frame_width, frame_height

    for x, y, conf in keypoints:
        if conf < 0.5:
            continue

        if display_meta is None:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        circle = display_meta.circle_params[display_meta.num_circles]
        circle.xc = clamp(x, 0, width - 1)
        circle.yc = clamp(y, 0, height - 1)
        circle.radius = 6
        circle.circle_color.set(1.0, 1.0, 1.0, 1.0)
        circle.has_bg_color = 1
        circle.bg_color.set(0.0, 0.0, 1.0, 1.0)
        display_meta.num_circles += 1

    for joint_a, joint_b in SKELETON:
        idx_a, idx_b = joint_a - 1, joint_b - 1
        if idx_a >= len(keypoints) or idx_b >= len(keypoints):
            continue

        x1, y1, c1 = keypoints[idx_a]
        x2, y2, c2 = keypoints[idx_b]

        if c1 < 0.5 or c2 < 0.5:
            continue

        if display_meta is None:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        line = display_meta.line_params[display_meta.num_lines]
        line.x1 = clamp(x1, 0, width - 1)
        line.y1 = clamp(y1, 0, height - 1)
        line.x2 = clamp(x2, 0, width - 1)
        line.y2 = clamp(y2, 0, height - 1)
        line.line_width = 6
        line.line_color.set(0.0, 0.0, 1.0, 1.0)
        display_meta.num_lines += 1

    return display_meta


def set_bbox_style(obj_meta, frame_width: int, frame_height: int) -> None:
    border_width = 6
    font_size = 18

    rect = obj_meta.rect_params
    rect.border_width = border_width
    rect.border_color.set(0.0, 0.0, 1.0, 1.0)

    text = obj_meta.text_params
    text.display_text = f"ID {obj_meta.object_id}"
    text.x_offset = clamp(rect.left - border_width * 0.5, 0, frame_width - 1)
    text.y_offset = clamp(rect.top - font_size * 2, 0, frame_height - 1)

    text.font_params.font_name = "Ubuntu"
    text.font_params.font_size = font_size
    text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 1.0, 1.0)

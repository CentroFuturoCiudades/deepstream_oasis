#!/usr/bin/env python3
# deepstream_pose_mp4.py
#
# DeepStream YOLO11-Pose -> procesa TODO el video (hasta EOS) y guarda MP4 con anotaciones (bbox + pose)
# Sin ventana (headless).
#
# Run:
#   python3 deepstream_pose_mp4.py
#
# Override:
#   python3 deepstream_pose_mp4.py \
#       -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
#       -c ./config_infer_primary_yolo11_pose.txt \
#       -o out.mp4

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from threading import Lock
from ctypes import sizeof, c_float
import csv

sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
import pyds

CSV_PATH = "metadata.csv"
csv_file = None
csv_writer = None

MAX_ELEMENTS_IN_DISPLAY_META = 16

DEFAULT_SOURCE = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
DEFAULT_INFER_CONFIG = os.path.abspath("config_infer_primary_yolo11_pose.txt")

STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0

PERF_MEASUREMENT_INTERVAL_SEC = 5
JETSON = False

OUTPUT_MP4 = "out_yolo11_pose.mp4"

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
]

perf_struct = {}


class GETFPS:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.start_time = time.time()
        self.is_first = True
        self.frame_count = 0
        self.total_fps_time = 0
        self.total_frame_count = 0
        self.fps_lock = Lock()

    def update_fps(self):
        with self.fps_lock:
            if self.is_first:
                self.start_time = time.time()
                self.is_first = False
                self.frame_count = 0
                self.total_fps_time = 0
                self.total_frame_count = 0
            else:
                self.frame_count += 1

    def get_fps(self):
        with self.fps_lock:
            end_time = time.time()
            dt = end_time - self.start_time
            self.total_fps_time += dt
            self.total_frame_count += self.frame_count
            current_fps = float(self.frame_count) / dt if dt > 0 else 0.0
            avg_fps = float(self.total_frame_count) / self.total_fps_time if self.total_fps_time > 0 else 0.0
            self.start_time = end_time
            self.frame_count = 0
        return current_fps, avg_fps

    def perf_print_callback(self):
        if not self.is_first:
            current_fps, avg_fps = self.get_fps()
            sys.stdout.write(f"DEBUG - Stream {self.stream_id + 1} - FPS: {current_fps:.2f} ({avg_fps:.2f})\n")
        return True


def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18

    x_offset = obj_meta.rect_params.left - border_width * 0.5
    y_offset = obj_meta.rect_params.top - font_size * 2 + border_width * 0.5 + 1

    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0

    obj_meta.text_params.display_text = f"ID {obj_meta.object_id}"

    obj_meta.text_params.font_params.font_name = "Ubuntu"
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, x_offset)))
    obj_meta.text_params.y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, y_offset)))

    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0

    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0



def parse_pose_from_meta(batch_meta, frame_meta, obj_meta):
    if not hasattr(obj_meta, "mask_params"):
        return
    if obj_meta.mask_params.size <= 0:
        return

    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))
    if num_joints <= 0:
        return

    gain = min(obj_meta.mask_params.width / STREAMMUX_WIDTH, obj_meta.mask_params.height / STREAMMUX_HEIGHT)
    if gain <= 0:
        return

    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) * 0.5
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) * 0.5

    data = obj_meta.mask_params.get_mask_array()
    display_meta = None

    # joints
    for i in range(num_joints):
        xc = (data[i * 3 + 0] - pad_x) / gain
        yc = (data[i * 3 + 1] - pad_y) / gain
        c = data[i * 3 + 2]
        if c < 0.5:
            continue

        if display_meta is None or display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        cp = display_meta.circle_params[display_meta.num_circles]
        cp.xc = int(min(STREAMMUX_WIDTH - 1, max(0, xc)))
        cp.yc = int(min(STREAMMUX_HEIGHT - 1, max(0, yc)))
        cp.radius = 6
        cp.circle_color.red = 1.0
        cp.circle_color.green = 1.0
        cp.circle_color.blue = 1.0
        cp.circle_color.alpha = 1.0
        cp.has_bg_color = 1
        cp.bg_color.red = 0.0
        cp.bg_color.green = 0.0
        cp.bg_color.blue = 1.0
        cp.bg_color.alpha = 1.0
        display_meta.num_circles += 1

    # lines
    for a, b in SKELETON:
        ia = a - 1
        ib = b - 1
        if ia >= num_joints or ib >= num_joints:
            continue

        x1 = (data[ia * 3 + 0] - pad_x) / gain
        y1 = (data[ia * 3 + 1] - pad_y) / gain
        c1 = data[ia * 3 + 2]

        x2 = (data[ib * 3 + 0] - pad_x) / gain
        y2 = (data[ib * 3 + 1] - pad_y) / gain
        c2 = data[ib * 3 + 2]

        if c1 < 0.5 or c2 < 0.5:
            continue

        if display_meta is None or display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        lp = display_meta.line_params[display_meta.num_lines]
        lp.x1 = int(min(STREAMMUX_WIDTH - 1, max(0, x1)))
        lp.y1 = int(min(STREAMMUX_HEIGHT - 1, max(0, y1)))
        lp.x2 = int(min(STREAMMUX_WIDTH - 1, max(0, x2)))
        lp.y2 = int(min(STREAMMUX_HEIGHT - 1, max(0, y2)))
        lp.line_width = 6
        lp.line_color.red = 0.0
        lp.line_color.green = 0.0
        lp.line_color.blue = 1.0
        lp.line_color.alpha = 1.0
        display_meta.num_lines += 1


def nvosd_sink_pad_buffer_probe(pad, info, user_data):

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

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            print(
                f"frame={frame_meta.frame_num} "
                f"class={obj_meta.class_id} "
                f"id={obj_meta.object_id} "
                f"conf={obj_meta.confidence:.3f} "
                f"w={obj_meta.rect_params.width:.1f} "
                f"h={obj_meta.rect_params.height:.1f}",
                flush=True
            )
            sys.stdout.flush()

            frame = frame_meta.frame_num
            track_id = obj_meta.object_id

            bbox = obj_meta.rect_params
            bx = bbox.left
            by = bbox.top
            bw = bbox.width
            bh = bbox.height

            keypoints = extract_keypoints(obj_meta)

            row = [
                frame,
                track_id,
                round(bx, 2),
                round(by, 2),
                round(bw, 2),
                round(bh, 2),
            ]

            for (x, y, c) in keypoints:
                row += [
                    round(x, 2),
                    round(y, 2),
                    round(c, 3),
                ]

            csv_writer.writerow(row)

            parse_pose_from_meta(batch_meta, frame_meta, obj_meta)
            set_custom_bbox(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        if frame_meta.source_id in perf_struct:
            perf_struct[frame_meta.source_id].update_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def uridecodebin_child_added_callback(child_proxy, Object, name, user_data):
    if "decodebin" in name:
        Object.connect("child-added", uridecodebin_child_added_callback, user_data)
    elif "nvv4l2decoder" in name:
        Object.set_property("drop-frame-interval", 0)
        Object.set_property("num-extra-surfaces", 1)
        Object.set_property("qos", 0)
        if JETSON:
            Object.set_property("enable-max-performance", 1)
        else:
            Object.set_property("cudadec-memtype", 0)
            Object.set_property("gpu-id", GPU_ID)


def uridecodebin_pad_added_callback(decodebin, pad, user_data):
    nvstreammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()

    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)

    if "video" in name:
        if features.contains("memory:NVMM"):
            if pad.link(nvstreammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write("ERROR - Failed to link source to nvstreammux sink pad\n")
        else:
            sys.stderr.write("ERROR - decodebin did not pick NVIDIA decoder plugin\n")


def create_uridecodebin(stream_id, uri, nvstreammux):
    bin_name = f"source-bin-{stream_id:04d}"
    uridecodebin = Gst.ElementFactory.make("uridecodebin", bin_name)

    if "rtsp://" in uri:
        pyds.configure_source_for_ntp_sync(uridecodebin)

    uridecodebin.set_property("uri", uri)

    pad_name = f"sink_{stream_id}"
    nvstreammux_sink_pad = nvstreammux.request_pad_simple(pad_name)
    if not nvstreammux_sink_pad:
        sys.stderr.write(f"ERROR - Failed to get nvstreammux {pad_name} pad\n")
        return None

    uridecodebin.connect("pad-added", uridecodebin_pad_added_callback, nvstreammux_sink_pad)
    uridecodebin.connect("child-added", uridecodebin_child_added_callback, None)

    perf_struct[stream_id] = GETFPS(stream_id)
    GLib.timeout_add(PERF_MEASUREMENT_INTERVAL_SEC * 1000, perf_struct[stream_id].perf_print_callback)

    return uridecodebin


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("DEBUG - EOS\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        sys.stderr.write(f"WARNING - {err.message} - {dbg}\n")
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        sys.stderr.write(f"ERROR - {err.message} - {dbg}\n")
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == "aarch64"


def try_set(enc, key, value):
    """Set a GObject property only if it exists (avoids crashes across versions)."""
    try:
        if enc.find_property(key) is not None:
            enc.set_property(key, value)
            return True
    except Exception:
        pass
    return False

def extract_keypoints(obj_meta):
    if not hasattr(obj_meta, "mask_params"):
        return []

    if obj_meta.mask_params.size <= 0:
        return []

    data = obj_meta.mask_params.get_mask_array()
    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

    gain = min(
        obj_meta.mask_params.width / STREAMMUX_WIDTH,
        obj_meta.mask_params.height / STREAMMUX_HEIGHT
    )
    if gain <= 0:
        return []

    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) * 0.5
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) * 0.5

    keypoints = []
    for i in range(num_joints):
        x = (data[i * 3 + 0] - pad_x) / gain
        y = (data[i * 3 + 1] - pad_y) / gain
        c = data[i * 3 + 2]
        keypoints.append((x, y, c))

    return keypoints

def main(source_uri, infer_config, output_mp4, streammux_w, streammux_h, gpu_id):
    global JETSON, GPU_ID, STREAMMUX_WIDTH, STREAMMUX_HEIGHT

    Gst.init(None)
    loop = GLib.MainLoop()

    GPU_ID = gpu_id
    STREAMMUX_WIDTH = streammux_w
    STREAMMUX_HEIGHT = streammux_h
    JETSON = is_aarch64()

    pipeline = Gst.Pipeline.new("pipeline")
    if not pipeline:
        sys.stderr.write("ERROR - Failed to create pipeline\n")
        return -1

    nvstreammux = Gst.ElementFactory.make("nvstreammux", "nvstreammux")
    if not nvstreammux:
        sys.stderr.write("ERROR - Failed to create nvstreammux\n")
        return -1
    pipeline.add(nvstreammux)

    uridecodebin = create_uridecodebin(0, source_uri, nvstreammux)
    if not uridecodebin:
        sys.stderr.write("ERROR - Failed to create uridecodebin\n")
        return -1
    pipeline.add(uridecodebin)

    nvinfer = Gst.ElementFactory.make("nvinfer", "nvinfer")
    nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not nvtracker:
        sys.stderr.write("ERROR - Failed to create nvtracker\n")
        return -1
    pipeline.add(nvtracker)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nvdsosd")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert2")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264enc")
    h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
    mp4mux = Gst.ElementFactory.make("qtmux", "mp4mux")
    filesink = Gst.ElementFactory.make("filesink", "filesink")

    for e, name in [
        (nvinfer, "nvinfer"), (nvvidconv, "nvvideoconvert"), (capsfilter, "capsfilter"),
        (nvosd, "nvdsosd"), (nvvidconv2, "nvvideoconvert2"), (encoder, "nvv4l2h264enc"),
        (h264parse, "h264parse"), (mp4mux, "qtmux"), (filesink, "filesink")
    ]:
        if not e:
            sys.stderr.write(f"ERROR - Failed to create {name}\n")
            return -1
        pipeline.add(e)

    filesink.set_property("location", output_mp4)
    filesink.set_property("sync", 0)
    filesink.set_property("async", 0)

    # streammux
    nvstreammux.set_property("batch-size", STREAMMUX_BATCH_SIZE)
    nvstreammux.set_property("batched-push-timeout", 25000)
    nvstreammux.set_property("width", STREAMMUX_WIDTH)
    nvstreammux.set_property("height", STREAMMUX_HEIGHT)
    nvstreammux.set_property("live-source", 0 if source_uri.startswith("file://") else 1)

    # nvinfer
    nvinfer.set_property("config-file-path", infer_config)
    nvinfer.set_property("qos", 0)

    # nvtracker
    nvtracker.set_property("tracker-width", 640)
    nvtracker.set_property("tracker-height", 384)
    nvtracker.set_property("gpu-id", GPU_ID)
    nvtracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_nvmultiobjecttracker.so"
    )
    nvtracker.set_property(
        "ll-config-file",
        "config_tracker_NvByteTrack.yml"
    )

    # osd
    nvosd.set_property("process-mode", int(pyds.MODE_GPU))
    nvosd.set_property("qos", 0)

    # caps for encoder
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    capsfilter.set_property("caps", caps)

    # GPU / memtype
    if not JETSON:
        nvstreammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
        nvstreammux.set_property("gpu_id", GPU_ID)

        nvinfer.set_property("gpu_id", GPU_ID)
        nvvidconv.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
        nvvidconv.set_property("gpu_id", GPU_ID)

        nvosd.set_property("gpu_id", GPU_ID)
        nvvidconv2.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
        nvvidconv2.set_property("gpu_id", GPU_ID)

        try_set(encoder, "gpu-id", GPU_ID)
        try_set(encoder, "bufapi-version", 1)

    # encoder settings (solo las que suelen existir)
    try_set(encoder, "bitrate", 8000000)
    try_set(encoder, "insert-sps-pps", 1)
    try_set(encoder, "iframeinterval", 30)
    try_set(encoder, "profile", 0)

    # Link
    if not nvstreammux.link(nvinfer):
        sys.stderr.write("ERROR - link nvstreammux -> nvinfer\n"); return -1
    if not nvinfer.link(nvtracker):
        sys.stderr.write("ERROR - link nvinfer -> nvtracker\n"); return -1
    if not nvtracker.link(nvvidconv):
        sys.stderr.write("ERROR - link nvtracker -> nvvidconv\n"); return -1
    if not nvvidconv.link(capsfilter):
        sys.stderr.write("ERROR - link nvvidconv -> capsfilter\n"); return -1
    if not capsfilter.link(nvosd):
        sys.stderr.write("ERROR - link capsfilter -> nvosd\n"); return -1
    if not nvosd.link(nvvidconv2):
        sys.stderr.write("ERROR - link nvosd -> nvvidconv2\n"); return -1
    if not nvvidconv2.link(encoder):
        sys.stderr.write("ERROR - link nvvidconv2 -> encoder\n"); return -1
    if not encoder.link(h264parse):
        sys.stderr.write("ERROR - link encoder -> h264parse\n"); return -1
    if not h264parse.link(mp4mux):
        sys.stderr.write("ERROR - link h264parse -> mp4mux\n"); return -1
    if not mp4mux.link(filesink):
        sys.stderr.write("ERROR - link mp4mux -> filesink\n"); return -1
    # bus
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # probe to draw
    nvosd_sink_pad = nvosd.get_static_pad("sink")
    if not nvosd_sink_pad:
        sys.stderr.write("ERROR - Failed to get nvosd sink pad\n")
        return -1
    nvosd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, nvosd_sink_pad_buffer_probe, None)

    ("\n")
    sys.stdout.write(f"SOURCE: {source_uri}\n")
    sys.stdout.write(f"INFER_CONFIG: {infer_config}\n")
    sys.stdout.write(f"OUTPUT: {output_mp4}\n")
    sys.stdout.write(f"STREAMMUX: {STREAMMUX_WIDTH}x{STREAMMUX_HEIGHT}\n")
    sys.stdout.write(f"GPU_ID: {GPU_ID}\n")
    sys.stdout.write(f"JETSON: {'TRUE' if JETSON else 'FALSE'}\n\n")

    global csv_file, csv_writer
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)

    # header
    header = [
        "frame",
        "track_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    # joints: kp0_x, kp0_y, kp0_conf, ...
    num_kp = 17  # YOLO11-Pose
    for i in range(num_kp):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"]

    csv_writer.writerow(header)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except Exception:
        pass

    pipeline.set_state(Gst.State.NULL)
    if csv_file:
        csv_file.close()

    sys.stdout.write(f"\nDone. Wrote: {output_mp4}\n\n")
    return 0


def parse_args():
    parser = argparse.ArgumentParser("DeepStream YOLO11-Pose -> MP4 (headless)")
    parser.add_argument("-s", "--source", default=DEFAULT_SOURCE, help="Source URI (file:///... or rtsp://...)")
    parser.add_argument("-c", "--infer-config", default=DEFAULT_INFER_CONFIG, help="nvinfer config path")
    parser.add_argument("-o", "--output", default=OUTPUT_MP4, help="Output MP4 path")
    parser.add_argument("-w", "--streammux-width", type=int, default=STREAMMUX_WIDTH)
    parser.add_argument("-e", "--streammux-height", type=int, default=STREAMMUX_HEIGHT)
    parser.add_argument("-g", "--gpu-id", type=int, default=GPU_ID)
    args = parser.parse_args()

    if not args.source:
        sys.stderr.write("ERROR - empty source\n"); sys.exit(1)
    if not args.infer_config or not os.path.isfile(args.infer_config):
        sys.stderr.write(f"ERROR - infer config not found: {args.infer_config}\n"); sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(
        source_uri=args.source,
        infer_config=args.infer_config,
        output_mp4=args.output,
        streammux_w=args.streammux_width,
        streammux_h=args.streammux_height,
        gpu_id=args.gpu_id,
    ))


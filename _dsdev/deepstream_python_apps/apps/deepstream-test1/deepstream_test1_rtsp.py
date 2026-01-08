#!/usr/bin/env python3
import argparse
import os
import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds

MUXER_BATCH_TIMEOUT_USEC = 33000

# -------------------------
# Global state (para cortar EXACTO N segundos desde el 1er frame)
# -------------------------
STOP_SCHEDULED = False
PIPELINE_REF = None
STOP_SECONDS = 0

# Guardamos el request pad del streammux para NO pedirlo múltiples veces
MUX_SINKPAD = None


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("EOS received -> quitting main loop")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print("ERROR:", err)
        if dbg:
            print("DBG:", dbg)
        loop.quit()
    return True


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    - Programa el STOP (EOS) al primer buffer que llegue (primer frame real).
    - Imprime conteos cada ~30 frames.
    """
    global STOP_SCHEDULED, PIPELINE_REF, STOP_SECONDS

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # ✅ Programa el timer SOLO una vez, al primer frame real
    if STOP_SECONDS and STOP_SECONDS > 0 and (not STOP_SCHEDULED):
        STOP_SCHEDULED = True

        def _stop():
            print(f"Stopping after {STOP_SECONDS}s (counted from FIRST FRAME) -> sending EOS...")
            if PIPELINE_REF is not None:
                PIPELINE_REF.send_event(Gst.Event.new_eos())
            return False  # no repetir

        GLib.timeout_add_seconds(STOP_SECONDS, _stop)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if frame_meta.frame_num % 30 == 0:
            num_obj = frame_meta.num_obj_meta
            ids = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                ids.append(int(obj_meta.object_id))
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            print(f"frame={frame_meta.frame_num} objs={num_obj} ids(sample)={ids[:5]}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, pad, streammux):
    """
    Link dinámico:
      uridecodebin (decoded raw video) -> nvstreammux:sink_0
    """
    global MUX_SINKPAD

    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    name = caps.to_string()

    # Solo video raw
    if "video/x-raw" not in name:
        return

    # Pedimos UNA sola vez sink_0 (batch-size=1)
    if MUX_SINKPAD is None:
        MUX_SINKPAD = streammux.request_pad_simple("sink_0")
        if not MUX_SINKPAD:
            raise RuntimeError("No se pudo pedir sink_0 de nvstreammux")

    if pad.link(MUX_SINKPAD) != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"No se pudo linkear decodebin->streammux. caps={name}")

    print("Linked decodebin video pad -> streammux sink_0")


def ensure_dir_for_file(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="RTSP -> DeepStream PGIE -> MP4")
    parser.add_argument("uri", help='RTSP URI (ej: "rtsp://100.80.43.126:8554/cam7")')
    parser.add_argument("--pgie-config", required=True, help="Path a config de nvinfer (pgie)")
    parser.add_argument("--out", default="out.mp4", help="Archivo de salida .mp4")
    parser.add_argument("--seconds", type=int, default=0, help="Graba N segundos y termina (0 = infinito)")
    parser.add_argument("--osd", action="store_true", help="Dibuja overlay (solo render en archivo)")
    parser.add_argument("--bitrate", type=int, default=6000000, help="Bitrate encoder (bps). default 6Mbps")
    parser.add_argument("--fps", type=int, default=30, help="FPS para caps antes del encoder (default 30)")
    args = parser.parse_args()

    if not os.path.isfile(args.pgie_config):
        print(f"ERROR: no existe --pgie-config: {args.pgie_config}")
        sys.exit(2)

    ensure_dir_for_file(args.out)

    Gst.init(None)

    # Global refs para stop-from-first-frame
    global PIPELINE_REF, STOP_SECONDS
    STOP_SECONDS = args.seconds

    pipeline = Gst.Pipeline.new("ds-rtsp-infer-mp4")
    PIPELINE_REF = pipeline

    # --- Elements ---
    source = Gst.ElementFactory.make("uridecodebin", "source")
    if not source:
        raise RuntimeError("No se pudo crear uridecodebin")
    source.set_property("uri", args.uri)

    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    if not streammux:
        raise RuntimeError("No se pudo crear nvstreammux")
    streammux.set_property("batch-size", 1)
    streammux.set_property("live-source", 1)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)

    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    if not pgie:
        raise RuntimeError("No se pudo crear nvinfer")
    pgie.set_property("config-file-path", args.pgie_config)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
    if not nvvidconv:
        raise RuntimeError("No se pudo crear nvvideoconvert")

    nvosd = None
    nvvidconv_postosd = None
    if args.osd:
        nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")
        if not nvosd:
            raise RuntimeError("No se pudo crear nvdsosd")
        nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_postosd")
        if not nvvidconv_postosd:
            raise RuntimeError("No se pudo crear nvvideoconvert postosd")

    # Convertimos a I420 antes del encoder HW
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter:
        raise RuntimeError("No se pudo crear capsfilter")
    capsfilter.set_property(
        "caps",
        Gst.Caps.from_string(f"video/x-raw(memory:NVMM), format=I420, framerate={args.fps}/1")
    )

    # Encoder + parser + muxer + sink (H264->MP4)
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    if not encoder:
        raise RuntimeError("No se pudo crear nvv4l2h264enc")
    encoder.set_property("bitrate", args.bitrate)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("iframeinterval", args.fps)  # ~1s GOP

    h264parse = Gst.ElementFactory.make("h264parse", "h264parse")
    if not h264parse:
        raise RuntimeError("No se pudo crear h264parse")

    mp4mux = Gst.ElementFactory.make("mp4mux", "mp4mux")
    if not mp4mux:
        raise RuntimeError("No se pudo crear mp4mux")
    mp4mux.set_property("faststart", True)

    filesink = Gst.ElementFactory.make("filesink", "filesink")
    if not filesink:
        raise RuntimeError("No se pudo crear filesink")
    filesink.set_property("location", args.out)
    filesink.set_property("sync", False)
    filesink.set_property("async", False)

    # --- Add to pipeline ---
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    if args.osd:
        pipeline.add(nvosd)
        pipeline.add(nvvidconv_postosd)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(h264parse)
    pipeline.add(mp4mux)
    pipeline.add(filesink)

    # --- Link fixed part ---
    if not streammux.link(pgie):
        raise RuntimeError("No link streammux->pgie")
    if not pgie.link(nvvidconv):
        raise RuntimeError("No link pgie->nvvidconv")

    if args.osd:
        if not nvvidconv.link(nvosd):
            raise RuntimeError("No link nvvidconv->nvosd")
        if not nvosd.link(nvvidconv_postosd):
            raise RuntimeError("No link nvosd->nvvidconv_postosd")
        if not nvvidconv_postosd.link(capsfilter):
            raise RuntimeError("No link postosd->capsfilter")

        # Probe en sink del OSD (ya tienes toda la metadata + OSD listo)
        osd_sink_pad = nvosd.get_static_pad("sink")
        if osd_sink_pad:
            osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)
    else:
        if not nvvidconv.link(capsfilter):
            raise RuntimeError("No link nvvidconv->capsfilter")

        # Probe en sink del nvvidconv (después de pgie)
        pad_probe = nvvidconv.get_static_pad("sink")
        if pad_probe:
            pad_probe.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, None)

    if not capsfilter.link(encoder):
        raise RuntimeError("No link capsfilter->encoder")
    if not encoder.link(h264parse):
        raise RuntimeError("No link encoder->h264parse")
    if not h264parse.link(mp4mux):
        raise RuntimeError("No link h264parse->mp4mux")
    if not mp4mux.link(filesink):
        raise RuntimeError("No link mp4mux->filesink")

    # Dynamic pad from uridecodebin
    source.connect("pad-added", cb_newpad, streammux)

    # Bus + loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print(f"Starting pipeline. Writing to: {args.out}")
    if args.seconds and args.seconds > 0:
        print(f"Will stop after {args.seconds}s (counted from FIRST FRAME).")
    else:
        print("Running until you Ctrl+C (or EOS from source).")

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt -> sending EOS...")
        pipeline.send_event(Gst.Event.new_eos())
        # deja que mp4mux cierre; si quieres, puedes esperar un poco aquí
        try:
            loop.run()
        except:
            pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("Done.")


if __name__ == "__main__":
    main()

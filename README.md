# DeepStream Oasis

DeepStream 8.0 Python pipeline to run **RTSP inference** and save the result as **MP4** using GPU acceleration.

## What it does
- Reads RTSP stream
- Runs DeepStream `nvinfer` (PGIE)
- Optional OSD overlay
- Encodes to H.264 → MP4
- Stops after N seconds (from first frame)

## Run

```bash
'python3 deepstream_test1_rtsp.py \
  "rtsp://<IP>:<PORT>/<stream>" \
  --pgie-config /opt/nvidia/deepstream/deepstream-8.0/samples/configs/deepstream-app/config_infer_primary.txt \
  --out ../../outputs/output.mp4 \
  --seconds 10 \
  --osd'


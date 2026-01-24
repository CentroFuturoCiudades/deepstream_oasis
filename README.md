# DeepStream Oasis

DeepStream 8.0 Python pipeline to run **RTSP inference** and save the result as **MP4** using GPU acceleration.

## What it does
- Reads RTSP stream
- Runs DeepStream `nvinfer` (PGIE)
- Optional OSD overlay
- Encodes to H.264 → MP4
- Stops after N seconds (from first frame)

## setup

```bash
'git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git'

'export CUDA_VER=XY.Z'
'make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo'
'make -C nvdsinfer_custom_impl_Yolo_Pose clean && make -C nvdsinfer_custom_impl_Yolo_Pose'

## Run

```bash
'python3 deepstream_test1_rtsp.py \
  "rtsp://<IP>:<PORT>/<stream>" \
  --pgie-config /opt/nvidia/deepstream/deepstream-8.0/samples/configs/deepstream-app/config_infer_primary.txt \
  --out ../../outputs/output.mp4 \
  --seconds 10 \
  --osd'



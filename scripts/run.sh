#!/usr/bin/env bash
set -euo pipefail

IMAGE="ds8-pyds:oasis"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Opcionales (headless friendly)
DISPLAY="${DISPLAY:-}"
WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}"

DOCKER_ARGS=(
  -it --rm
  --gpus all
  --net=host
  --ipc=host
  -v "${PROJECT_DIR}:/workspace/deepstream_oasis"
  -w /workspace/deepstream_oasis
)

# Si hay X11/Wayland disponibles, pásalos
if [[ -n "${DISPLAY}" ]]; then
  DOCKER_ARGS+=(-e DISPLAY="${DISPLAY}")
  [[ -S /tmp/.X11-unix/X0 || -d /tmp/.X11-unix ]] && DOCKER_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
  DOCKER_ARGS+=(-e QT_X11_NO_MITSHM=1 -e LIBGL_ALWAYS_INDIRECT=1)
fi

if [[ -n "${WAYLAND_DISPLAY}" ]]; then
  DOCKER_ARGS+=(-e WAYLAND_DISPLAY="${WAYLAND_DISPLAY}")
fi

if [[ -n "${XDG_RUNTIME_DIR}" ]]; then
  DOCKER_ARGS+=(-e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}")
fi

# WSLg (si existe)
if [[ -d /mnt/wslg ]]; then
  DOCKER_ARGS+=(-v /mnt/wslg:/mnt/wslg)
fi

# NVIDIA env (no estorba si ya está)
DOCKER_ARGS+=(
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
)

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" bash


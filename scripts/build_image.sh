#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ds8-pyds:oasis"
NO_CACHE="${NO_CACHE:-0}"

EXTRA_ARGS=()
if [ "$NO_CACHE" = "1" ]; then
  EXTRA_ARGS+=(--no-cache)
fi

docker build "${EXTRA_ARGS[@]}" \
  -f docker/Dockerfile.ds8-pyds \
  -t "${IMAGE_NAME}" \
  .

echo "✅ Built image: ${IMAGE_NAME}"


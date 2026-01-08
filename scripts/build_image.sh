#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ds8-pyds:oasis"

docker build \
  -f docker/Dockerfile.ds8-pyds \
  -t "${IMAGE_NAME}" \
  .

echo "✅ Built image: ${IMAGE_NAME}"

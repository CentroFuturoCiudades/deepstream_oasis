#!/usr/bin/env bash
set -e

source /opt/venvs/pyds/bin/activate

export DS_ROOT=/opt/nvidia/deepstream/deepstream-8.0
export LD_LIBRARY_PATH="$DS_ROOT/lib:$DS_ROOT/lib/gst-plugins:${LD_LIBRARY_PATH}"

exec "$@"

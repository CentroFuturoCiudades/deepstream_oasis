#!/usr/bin/env bash
set -euo pipefail

python -c "import sys; import pyds; print('python=', sys.executable); print('pyds=', pyds.__file__)"

#!/usr/bin/env bash
set -e

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_dir/.venv/bin/activate"

cuda_lib_dirs=$(printf "%s:" "$VIRTUAL_ENV"/lib/python3.11/site-packages/nvidia/*/lib)
export LD_LIBRARY_PATH="${cuda_lib_dirs}${LD_LIBRARY_PATH:-}"

exec "$@"
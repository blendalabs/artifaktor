#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/ml_backend/grounding_dino_backend"
PORT="${ML_BACKEND_PORT:-9090}"
HOST="${ML_BACKEND_HOST:-127.0.0.1}"

# Make pip wheels work on NixOS and expose NVIDIA driver libs to torch (libcuda.so)
export LD_LIBRARY_PATH="/run/opengl-driver/lib${NIX_LD_LIBRARY_PATH:+:$NIX_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export LOCAL_FILES_DOCUMENT_ROOT="${LOCAL_FILES_DOCUMENT_ROOT:-$ROOT_DIR}"
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cuda}"

cd "$BACKEND_DIR"
exec "$ROOT_DIR/.venv/bin/python" _wsgi.py --host "$HOST" --port "$PORT"

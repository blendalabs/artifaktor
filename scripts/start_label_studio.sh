#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LS_BIND_HOST="${LABEL_STUDIO_BIND_HOST:-127.0.0.1}"
LS_PORT="${LABEL_STUDIO_PORT:-8080}"
LS_URL="${LABEL_STUDIO_URL:-http://127.0.0.1:8080}"
ML_BACKEND_URL="${ML_BACKEND_URL:-http://127.0.0.1:9090}"
EXTRA_ARGS=("$@")

# NixOS runtime linker path for Label Studio + torch wheels.
export LD_LIBRARY_PATH="/run/opengl-driver/lib${NIX_LD_LIBRARY_PATH:+:$NIX_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Required for Label Studio Local Files storage.
export LOCAL_FILES_SERVING_ENABLED="${LOCAL_FILES_SERVING_ENABLED:-true}"
export LOCAL_FILES_DOCUMENT_ROOT="${LOCAL_FILES_DOCUMENT_ROOT:-$ROOT_DIR}"
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT:-$LOCAL_FILES_DOCUMENT_ROOT}"

# Prefer local GPU for model warmup/predict.
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cuda}"

wait_http() {
  local url="$1"
  local timeout_s="$2"
  local waited=0
  until curl -fsS "$url" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$timeout_s" ]; then
      echo "[ERROR] Timed out waiting for $url" >&2
      return 1
    fi
  done
}

if ! curl -fsS "$ML_BACKEND_URL/health" >/dev/null 2>&1; then
  echo "[INFO] Starting ML backend at $ML_BACKEND_URL ..."
  "$ROOT_DIR/scripts/start_ml_backend.sh" > /tmp/ml_backend_9090.log 2>&1 &
fi

wait_http "$ML_BACKEND_URL/health" 180

echo "[INFO] Warming ML backend model ..."
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/warmup_ml_backend.py" --backend-url "$ML_BACKEND_URL"

if curl -fsS "$LS_URL/api/health" >/dev/null 2>&1; then
  echo "[INFO] Label Studio already running at $LS_URL"
  echo "[INFO] Warmup complete."
  exit 0
fi

echo "[INFO] Starting Label Studio at $LS_URL ..."
exec "$ROOT_DIR/.venv/bin/label-studio" start --no-browser --internal-host "$LS_BIND_HOST" -p "$LS_PORT" --host "$LS_URL" "${EXTRA_ARGS[@]}"

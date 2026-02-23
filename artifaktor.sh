#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Load project environment (venv + runtime libs)
if [ -f "$ROOT_DIR/.envrc" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.envrc"
fi

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/main.py" "$@"

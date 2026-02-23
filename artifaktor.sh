#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec direnv exec . .venv/bin/python main.py "$@"

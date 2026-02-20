#!/usr/bin/env python3
"""Warm up the Label Studio ML backend with a real local frame.

Python: 3.11+
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:9090"
DEFAULT_LABEL_CONFIG = Path("label_studio/artifact_detection_config.xml")
DEFAULT_FRAMES_DIR = Path("sequences")


class WarmupError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up Grounding DINO backend so first LS prediction is fast")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="ML backend URL")
    parser.add_argument("--label-config", type=Path, default=DEFAULT_LABEL_CONFIG, help="Label config XML path")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help="Directory used to auto-pick a warmup image when --image is omitted",
    )
    parser.add_argument("--image", type=Path, default=None, help="Exact image path to use for warmup")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout seconds")
    return parser.parse_args()


def pick_image(image: Path | None, frames_dir: Path) -> Path:
    if image is not None:
        img = image.resolve()
        if not img.exists() or not img.is_file():
            raise WarmupError(f"--image not found: {img}")
        return img

    root = frames_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise WarmupError(f"--frames-dir does not exist: {root}")

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not candidates:
        raise WarmupError(f"No image files found in: {root}")
    return candidates[0].resolve()


def main() -> int:
    args = parse_args()
    backend_url = args.backend_url.rstrip("/")

    try:
        image_path = pick_image(args.image, args.frames_dir)

        config_path = args.label_config.resolve()
        if not config_path.exists():
            raise WarmupError(f"--label-config not found: {config_path}")
        label_config = config_path.read_text(encoding="utf-8")

        payload: dict[str, Any] = {
            "tasks": [{"id": 1, "data": {"image": str(image_path)}}],
            "label_config": label_config,
            "project": "warmup",
            "params": {"context": None},
        }

        resp = requests.post(f"{backend_url}/predict", json=payload, timeout=args.timeout)
        if resp.status_code != 200:
            raise WarmupError(f"POST /predict failed ({resp.status_code}): {resp.text}")

        body = resp.json()
        first = body[0] if isinstance(body, list) and body else {}
        result_count = len(first.get("result", [])) if isinstance(first, dict) else 0
        score = float(first.get("score", 0.0)) if isinstance(first, dict) else 0.0

    except (requests.RequestException, WarmupError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Warmup failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "backend_url": backend_url,
                "image": str(image_path),
                "predictions": result_count,
                "score": score,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

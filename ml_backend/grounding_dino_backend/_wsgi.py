#!/usr/bin/env python3
"""WSGI/Flask entrypoint for Grounding DINO Label Studio ML backend."""

from __future__ import annotations

import argparse
import json
import logging
import logging.config
import os

from label_studio_ml.api import init_app

from model import GroundingDinoArtifactModel

logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": True,
        },
    }
)

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def get_kwargs_from_config(config_path: str = _DEFAULT_CONFIG_PATH) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    return config


def parse_kv_pairs(raw_pairs: list[list[str]] | None) -> dict:
    if not raw_pairs:
        return {}

    def as_scalar(value: str):
        if value.isdigit():
            return int(value)
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return float(value)
        except ValueError:
            return value

    out: dict[str, object] = {}
    for pair in raw_pairs:
        if len(pair) != 2:
            raise ValueError(f"Invalid --kwargs pair: {pair}")
        key, value = pair
        out[key] = as_scalar(value)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounding DINO Label Studio ML backend")
    parser.add_argument("-p", "--port", type=int, default=9090, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    parser.add_argument("--model-dir", default=os.path.dirname(__file__), help="Model directory")
    parser.add_argument(
        "--kwargs",
        "--with",
        metavar="KEY=VAL",
        nargs="+",
        type=lambda kv: kv.split("=", 1),
        help="Additional GroundingDinoArtifactModel kwargs",
    )
    parser.add_argument("--check", action="store_true", help="Validate model init before run")

    args = parser.parse_args()

    if args.log_level:
        logging.getLogger().setLevel(args.log_level)

    kwargs = get_kwargs_from_config()
    kwargs.update(parse_kv_pairs(args.kwargs))

    if args.check:
        print(f"Checking model init: {GroundingDinoArtifactModel.__name__}")
        _ = GroundingDinoArtifactModel(**kwargs)

    app = init_app(
        model_class=GroundingDinoArtifactModel,
        model_dir=os.environ.get("MODEL_DIR", args.model_dir),
        redis_queue=os.environ.get("RQ_QUEUE_NAME", "default"),
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=os.environ.get("REDIS_PORT", 6379),
        **kwargs,
    )

    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    app = init_app(
        model_class=GroundingDinoArtifactModel,
        model_dir=os.environ.get("MODEL_DIR", os.path.dirname(__file__)),
        redis_queue=os.environ.get("RQ_QUEUE_NAME", "default"),
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=os.environ.get("REDIS_PORT", 6379),
    )

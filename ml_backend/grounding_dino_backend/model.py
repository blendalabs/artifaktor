#!/usr/bin/env python3
"""Label Studio ML backend model for artifact prelabeling.

Primary mode:
- Grounding DINO zero-shot object detection via transformers

Fallback mode (when torch/transformers are unavailable):
- Simple OpenCV contour detector that proposes "edge_artifact" boxes

Python: 3.11+
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptDef:
    label: str
    phrase: str


PROMPTS: list[PromptDef] = [
    PromptDef(label="flicker", phrase="flicker artifact"),
    PromptDef(label="morph", phrase="morphing distortion"),
    PromptDef(label="hand_distortion", phrase="distorted hand"),
    PromptDef(label="face_distortion", phrase="distorted face"),
    PromptDef(label="temporal_blur", phrase="motion blur artifact"),
    PromptDef(label="edge_artifact", phrase="edge artifact"),
]


class GroundingDinoArtifactModel(LabelStudioMLBase):
    """Artifact detector for Label Studio preannotations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model_id = os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
        self.box_threshold = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.22"))
        self.text_threshold = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.20"))
        self.device = os.getenv("GROUNDING_DINO_DEVICE", "cpu")
        self.max_detections = int(os.getenv("GROUNDING_DINO_MAX_DETECTIONS", "20"))

        self.from_name, self.to_name, self.image_key, self.allowed_labels = self._resolve_label_mapping()

        self._processor = None
        self._model = None
        self._backend_mode = "lazy-not-initialized"
        self._runtime_initialized = False

        LOGGER.info(
            "Initialized model (lazy) mode=%s model_id=%s labels=%s",
            self._backend_mode,
            self.model_id,
            self.allowed_labels,
        )

    def _resolve_label_mapping(self) -> tuple[str, str, str, list[str]]:
        if not self.parsed_label_config:
            return "label", "image", "image", [p.label for p in PROMPTS]

        first_control = next(iter(self.parsed_label_config.items()))
        from_name, cfg = first_control
        to_name = cfg["to_name"][0]
        image_key = cfg["inputs"][0]["value"]
        labels = list(cfg.get("labels", []))
        if not labels:
            labels = [p.label for p in PROMPTS]
        return from_name, to_name, image_key, labels

    def _init_grounding_dino_if_available(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            LOGGER.warning("Grounding DINO deps unavailable, using OpenCV fallback: %s", exc)
            self._backend_mode = "opencv-fallback"
            return

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self._backend_mode = "grounding-dino"
            LOGGER.info("Grounding DINO loaded: %s on %s", self.model_id, self.device)
        except Exception as exc:  # pragma: no cover - env dependent
            LOGGER.exception("Failed to load Grounding DINO model; using fallback. error=%s", exc)
            self._processor = None
            self._model = None
            self._backend_mode = "opencv-fallback"

    def _ensure_runtime_initialized(self) -> None:
        if self._runtime_initialized:
            return
        self._init_grounding_dino_if_available()
        self._runtime_initialized = True

    def predict(self, tasks: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        self._ensure_runtime_initialized()
        predictions: list[dict[str, Any]] = []
        for task in tasks:
            image_path = self._resolve_image_path(task)
            if not image_path:
                LOGGER.warning("Skipping task with unresolved image path. task_id=%s", task.get("id"))
                predictions.append({"result": [], "score": 0.0})
                continue

            if self._backend_mode == "grounding-dino":
                result_items = self._predict_grounding_dino(image_path)
            else:
                result_items = self._predict_fallback(image_path)

            score = max((item["value"].get("score", 0.0) for item in result_items), default=0.0)
            predictions.append({"result": result_items, "score": float(score)})

        return predictions

    def _resolve_image_path(self, task: dict[str, Any]) -> str | None:
        data = task.get("data", {})
        raw = data.get(self.image_key)
        if not raw:
            return None

        # 1) Direct filesystem path
        candidate = Path(str(raw))
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())

        # 2) Label Studio local-files URL => map to LOCAL_FILES_DOCUMENT_ROOT
        #    Example: http://127.0.0.1:8080/data/local-files/?d=sequences/x.png
        as_str = str(raw)
        try:
            parsed = urlparse(as_str)
            if "/data/local-files/" in parsed.path:
                rel_values = parse_qs(parsed.query).get("d", [])
                if rel_values:
                    rel_path = unquote(rel_values[0]).lstrip("/")
                    root = Path(os.getenv("LOCAL_FILES_DOCUMENT_ROOT", "/"))
                    local_path = (root / rel_path).resolve()
                    if local_path.exists() and local_path.is_file():
                        return str(local_path)
        except Exception:
            LOGGER.exception("Failed parsing local-files URL: %s", as_str)

        # 3) Generic URL/path resolver provided by LS SDK base class
        try:
            local = self.get_local_path(as_str)
            if local and Path(local).exists():
                return str(Path(local).resolve())
        except Exception:
            LOGGER.exception("get_local_path failed for %s", as_str)

        return None

    def _predict_grounding_dino(self, image_path: str) -> list[dict[str, Any]]:
        import torch  # type: ignore

        assert self._processor is not None
        assert self._model is not None

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        prompt_text = ". ".join(p.phrase for p in PROMPTS if p.label in self.allowed_labels) + "."

        inputs = self._processor(images=image, text=prompt_text, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        try:
            # transformers<=4.x
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]
        except TypeError:
            # transformers>=5.x renamed box_threshold -> threshold
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]

        boxes = processed["boxes"].cpu().numpy()
        scores = processed["scores"].cpu().numpy()
        labels = processed["labels"]

        results: list[dict[str, Any]] = []
        for i, (box, score, raw_label) in enumerate(zip(boxes, scores, labels)):
            if i >= self.max_detections:
                break
            mapped = self._map_phrase_to_label(str(raw_label))
            if not mapped:
                continue
            x_min, y_min, x_max, y_max = [float(v) for v in box.tolist()]
            results.append(self._to_ls_result(mapped, x_min, y_min, x_max, y_max, width, height, float(score)))

        return results

    def _map_phrase_to_label(self, phrase: str) -> str | None:
        normalized = phrase.lower().strip()
        for p in PROMPTS:
            if p.label not in self.allowed_labels:
                continue
            if p.phrase in normalized or normalized in p.phrase or p.label.replace("_", " ") in normalized:
                return p.label

        # last-resort keyword mapping
        keyword_map = {
            "flick": "flicker",
            "morph": "morph",
            "hand": "hand_distortion",
            "face": "face_distortion",
            "blur": "temporal_blur",
            "edge": "edge_artifact",
            "artifact": "edge_artifact",
        }
        for key, label in keyword_map.items():
            if key in normalized and label in self.allowed_labels:
                return label
        return None

    def _predict_fallback(self, image_path: str) -> list[dict[str, Any]]:
        """Fallback detector: detect strong edge clusters as edge artifacts."""
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return []
        h, w = bgr.shape[:2]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 80, 160)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        results: list[dict[str, Any]] = []

        max_boxes = min(self.max_detections, 5)
        for cnt in contours[:max_boxes]:
            area = cv2.contourArea(cnt)
            if area < 150.0:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x_min, y_min = float(x), float(y)
            x_max, y_max = float(x + bw), float(y + bh)
            conf = float(min(0.55, max(0.15, area / float(w * h))))
            if "edge_artifact" in self.allowed_labels:
                label = "edge_artifact"
            else:
                label = self.allowed_labels[0]
            results.append(self._to_ls_result(label, x_min, y_min, x_max, y_max, w, h, conf))

        return results

    def _to_ls_result(
        self,
        label: str,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        width_px: int,
        height_px: int,
        score: float,
    ) -> dict[str, Any]:
        x_min = max(0.0, min(x_min, float(width_px - 1)))
        y_min = max(0.0, min(y_min, float(height_px - 1)))
        x_max = max(x_min + 1.0, min(x_max, float(width_px)))
        y_max = max(y_min + 1.0, min(y_max, float(height_px)))

        x_pct = 100.0 * x_min / float(width_px)
        y_pct = 100.0 * y_min / float(height_px)
        w_pct = 100.0 * (x_max - x_min) / float(width_px)
        h_pct = 100.0 * (y_max - y_min) / float(height_px)

        return {
            "from_name": self.from_name,
            "to_name": self.to_name,
            "type": "rectanglelabels",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "rotation": 0,
                "rectanglelabels": [label],
                "score": float(score),
            },
        }

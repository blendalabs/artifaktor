#!/usr/bin/env python3
"""Label Studio ML backend model for body-distortion prelabeling.

Primary mode:
- Grounding DINO zero-shot object detection via transformers

Fallback mode (when torch/transformers are unavailable):
- Simple OpenCV contour detector that proposes generic body_distortion boxes

Python: 3.11+
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase

LOGGER = logging.getLogger(__name__)

PRIMARY_DEFAULT_LABEL = "body_distortion"
DEFAULT_BODY_PART_CHOICES = ["face", "hair", "hand", "arm", "leg", "torso", "full_body", "other"]
DEFAULT_SEVERITY_CHOICES = ["mild", "medium", "severe"]


@dataclass(frozen=True)
class PromptDef:
    part: str
    phrase: str


PROMPTS: list[PromptDef] = [
    PromptDef(part="face", phrase="distorted face"),
    PromptDef(part="hair", phrase="distorted hair"),
    PromptDef(part="hand", phrase="distorted hand"),
    PromptDef(part="arm", phrase="distorted arm"),
    PromptDef(part="leg", phrase="distorted leg"),
    PromptDef(part="torso", phrase="distorted torso"),
    PromptDef(part="full_body", phrase="distorted full body"),
    PromptDef(part="other", phrase="body distortion artifact"),
]


class GroundingDinoArtifactModel(LabelStudioMLBase):
    """Body distortion detector for Label Studio preannotations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model_id = os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
        self.box_threshold = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.22"))
        self.text_threshold = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.20"))
        self.device = os.getenv("GROUNDING_DINO_DEVICE", "cpu")
        self.max_detections = int(os.getenv("GROUNDING_DINO_MAX_DETECTIONS", "20"))

        (
            self.from_name,
            self.to_name,
            self.image_key,
            self.allowed_labels,
            self.body_part_from_name,
            self.body_part_choices,
            self.severity_from_name,
            self.severity_choices,
        ) = self._resolve_label_mapping()

        self.primary_label = self.allowed_labels[0] if self.allowed_labels else PRIMARY_DEFAULT_LABEL

        self._processor = None
        self._model = None
        self._backend_mode = "lazy-not-initialized"
        self._runtime_initialized = False

        LOGGER.info(
            "Initialized model (lazy) mode=%s model_id=%s primary_label=%s body_part_control=%s severity_control=%s",
            self._backend_mode,
            self.model_id,
            self.primary_label,
            self.body_part_from_name,
            self.severity_from_name,
        )

    def _resolve_label_mapping(self) -> tuple[str, str, str, list[str], str | None, list[str], str | None, list[str]]:
        if not self.parsed_label_config:
            return (
                "label",
                "image",
                "image",
                [PRIMARY_DEFAULT_LABEL],
                "body_part",
                DEFAULT_BODY_PART_CHOICES,
                "severity",
                DEFAULT_SEVERITY_CHOICES,
            )

        controls = self.parsed_label_config

        # Main rectangle control
        rect_item = next(
            ((from_name, cfg) for from_name, cfg in controls.items() if str(cfg.get("type", "")).lower() == "rectanglelabels"),
            None,
        )
        if rect_item is None:
            return (
                "label",
                "image",
                "image",
                [PRIMARY_DEFAULT_LABEL],
                None,
                [],
                None,
                [],
            )

        from_name, rect_cfg = rect_item
        to_name = rect_cfg["to_name"][0]
        image_key = rect_cfg["inputs"][0]["value"]
        labels = list(rect_cfg.get("labels", [])) or [PRIMARY_DEFAULT_LABEL]

        body_part_from_name: str | None = None
        body_part_choices: list[str] = []
        severity_from_name: str | None = None
        severity_choices: list[str] = []

        for candidate_name, cfg in controls.items():
            if str(cfg.get("type", "")).lower() != "choices":
                continue

            conditionals = cfg.get("conditionals", {})
            if conditionals.get("type") != "tag" or conditionals.get("name") != from_name:
                continue

            choices = list(cfg.get("labels", []))
            choices_norm = {self._norm_choice(c) for c in choices}

            if not body_part_from_name and any(c in choices_norm for c in DEFAULT_BODY_PART_CHOICES):
                body_part_from_name = candidate_name
                body_part_choices = choices
                continue

            if not severity_from_name and any(c in choices_norm for c in DEFAULT_SEVERITY_CHOICES):
                severity_from_name = candidate_name
                severity_choices = choices

        return (
            from_name,
            to_name,
            image_key,
            labels,
            body_part_from_name,
            body_part_choices,
            severity_from_name,
            severity_choices,
        )

    @staticmethod
    def _norm_choice(value: str) -> str:
        return value.strip().lower().replace(" ", "_")

    def _resolve_choice(self, desired: str, choices: list[str]) -> str | None:
        desired_norm = self._norm_choice(desired)
        for choice in choices:
            if self._norm_choice(choice) == desired_norm:
                return choice
        return None

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

            score = max((item.get("value", {}).get("score", 0.0) for item in result_items), default=0.0)
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

        prompt_text = ". ".join(p.phrase for p in PROMPTS) + "."

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

            mapped_label, mapped_part = self._map_phrase(str(raw_label))
            if not mapped_label:
                continue

            x_min, y_min, x_max, y_max = [float(v) for v in box.tolist()]
            region_id = f"r_{uuid4().hex[:12]}"
            results.append(
                self._to_ls_result(
                    mapped_label,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    width,
                    height,
                    float(score),
                    region_id=region_id,
                )
            )

            if mapped_part and self.body_part_from_name and self.body_part_choices:
                part_choice = self._resolve_choice(mapped_part, self.body_part_choices)
                if part_choice:
                    results.append(self._to_ls_choice_result(region_id, self.body_part_from_name, part_choice))

        return results

    def _map_phrase(self, phrase: str) -> tuple[str | None, str | None]:
        normalized = phrase.lower().strip().replace("_", " ")

        # label is intentionally unified
        mapped_label = self.primary_label if self.primary_label in self.allowed_labels else None
        if not mapped_label:
            return None, None

        body_part: str | None = None
        keyword_map: list[tuple[str, str]] = [
            ("full body", "full_body"),
            ("whole body", "full_body"),
            ("entire body", "full_body"),
            ("hair", "hair"),
            ("face", "face"),
            ("hand", "hand"),
            ("arm", "arm"),
            ("leg", "leg"),
            ("torso", "torso"),
            ("body", "other"),
            ("artifact", "other"),
            ("distort", "other"),
        ]
        for key, candidate in keyword_map:
            if key in normalized:
                body_part = candidate
                break

        return mapped_label, body_part

    def _predict_fallback(self, image_path: str) -> list[dict[str, Any]]:
        """Fallback detector: detect strong edge clusters as generic body distortions."""
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

            region_id = f"r_{uuid4().hex[:12]}"
            results.append(
                self._to_ls_result(
                    self.primary_label,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    w,
                    h,
                    conf,
                    region_id=region_id,
                )
            )

            if self.body_part_from_name and self.body_part_choices:
                other_choice = self._resolve_choice("other", self.body_part_choices)
                if other_choice:
                    results.append(self._to_ls_choice_result(region_id, self.body_part_from_name, other_choice))

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
        region_id: str | None = None,
    ) -> dict[str, Any]:
        x_min = max(0.0, min(x_min, float(width_px - 1)))
        y_min = max(0.0, min(y_min, float(height_px - 1)))
        x_max = max(x_min + 1.0, min(x_max, float(width_px)))
        y_max = max(y_min + 1.0, min(y_max, float(height_px)))

        x_pct = 100.0 * x_min / float(width_px)
        y_pct = 100.0 * y_min / float(height_px)
        w_pct = 100.0 * (x_max - x_min) / float(width_px)
        h_pct = 100.0 * (y_max - y_min) / float(height_px)

        result = {
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
        if region_id:
            result["id"] = region_id
        return result

    def _to_ls_choice_result(self, region_id: str, from_name: str, choice: str) -> dict[str, Any]:
        return {
            "id": region_id,
            "from_name": from_name,
            "to_name": self.to_name,
            "type": "choices",
            "value": {
                "choices": [choice],
            },
        }

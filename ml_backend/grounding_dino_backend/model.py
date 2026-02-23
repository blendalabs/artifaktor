#!/usr/bin/env python3
"""Label Studio ML backend model for body-distortion prelabeling.

Primary mode:
- Grounding DINO zero-shot object detection via transformers

Fallback mode (when torch/transformers are unavailable):
- Simple OpenCV contour detector that proposes generic body_distortion boxes

Learning mode (manual retrain loop in Label Studio):
- Sequence KNN model trained from accepted annotations (frame -> center/size)
- Triggered by Label Studio "Train" button (START_TRAINING webhook)

Python: 3.11+
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import cv2
import numpy as np
import requests
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


@dataclass(frozen=True)
class SequenceSample:
    frame: int
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    img_w: int
    img_h: int


class GroundingDinoArtifactModel(LabelStudioMLBase):
    """Body distortion detector for Label Studio preannotations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model_id = os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
        self.box_threshold = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.22"))
        self.text_threshold = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.20"))
        self.device = os.getenv("GROUNDING_DINO_DEVICE", "cpu")
        self.max_detections = int(os.getenv("GROUNDING_DINO_MAX_DETECTIONS", "20"))

        # Sequence-learning knobs (for quick iterative retraining in LS)
        self.sequence_learning_enabled = self._env_bool("SEQUENCE_LEARNING_ENABLED", True)
        self.sequence_prefer_trained = self._env_bool("SEQUENCE_PREFER_TRAINED", True)
        self.sequence_min_train_samples = int(os.getenv("SEQUENCE_MIN_TRAIN_SAMPLES", "8"))
        self.sequence_k_neighbors = max(1, int(os.getenv("SEQUENCE_K_NEIGHBORS", "4")))
        self.sequence_default_box_w_pct = float(os.getenv("SEQUENCE_DEFAULT_BOX_WIDTH_PCT", "6.0"))
        self.sequence_default_box_h_pct = float(os.getenv("SEQUENCE_DEFAULT_BOX_HEIGHT_PCT", "10.0"))
        # Keep off by default to avoid expensive retrain on every annotation webhook.
        self.sequence_train_on_annotation_events = self._env_bool("SEQUENCE_TRAIN_ON_ANNOTATION_EVENTS", False)

        (
            self.from_name,
            self.to_name,
            self.image_key,
            self.allowed_labels,
            self.body_part_from_name,
            self.body_part_choices,
            self.severity_from_name,
            self.severity_choices,
            self.primary_control_type,
        ) = self._resolve_label_mapping()

        self.primary_label = self.allowed_labels[0] if self.allowed_labels else PRIMARY_DEFAULT_LABEL

        # Unwrap train_output; LS ML manager may pass either raw train_output dict
        # or the whole job_result containing "train_output".
        train_payload = self.train_output if isinstance(self.train_output, dict) else {}
        if isinstance(train_payload.get("train_output"), dict):
            train_payload = train_payload["train_output"]
        self._train_payload = train_payload

        self._sequence_samples = self._load_sequence_samples(train_payload)
        self._sequence_frames = np.array([s.frame for s in self._sequence_samples], dtype=np.float64)

        self._processor = None
        self._model = None
        self._backend_mode = "lazy-not-initialized"
        self._runtime_initialized = False

        LOGGER.info(
            "Initialized model (lazy) mode=%s model_id=%s primary_label=%s control=%s seq_enabled=%s seq_samples=%s",
            self._backend_mode,
            self.model_id,
            self.primary_label,
            self.primary_control_type,
            self.sequence_learning_enabled,
            len(self._sequence_samples),
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_label_mapping(
        self,
    ) -> tuple[str, str, str, list[str], str | None, list[str], str | None, list[str], str]:
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
                "rectanglelabels",
            )

        controls = self.parsed_label_config

        # Main control: prefer RectangleLabels, fallback to KeyPointLabels
        primary_item = next(
            (
                (from_name, cfg)
                for from_name, cfg in controls.items()
                if str(cfg.get("type", "")).lower() == "rectanglelabels"
            ),
            None,
        )
        if primary_item is None:
            primary_item = next(
                (
                    (from_name, cfg)
                    for from_name, cfg in controls.items()
                    if str(cfg.get("type", "")).lower() == "keypointlabels"
                ),
                None,
            )

        if primary_item is None:
            return (
                "label",
                "image",
                "image",
                [PRIMARY_DEFAULT_LABEL],
                None,
                [],
                None,
                [],
                "rectanglelabels",
            )

        from_name, main_cfg = primary_item
        control_type = str(main_cfg.get("type", "rectanglelabels")).lower()
        to_name = main_cfg["to_name"][0]
        image_key = main_cfg["inputs"][0]["value"]
        labels = list(main_cfg.get("labels", [])) or [PRIMARY_DEFAULT_LABEL]

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
            control_type,
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

    # ---------------------------
    # Runtime model init / predict
    # ---------------------------

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

            # 1) Prefer trained sequence model when available.
            trained_result: list[dict[str, Any]] | None = None
            if self.sequence_learning_enabled and self.sequence_prefer_trained:
                trained_result = self._predict_sequence(task, image_path)

            if trained_result is not None:
                result_items = trained_result
            else:
                # 2) Fallback to zero-shot detector.
                if self._backend_mode == "grounding-dino":
                    result_items = self._predict_grounding_dino(image_path)
                else:
                    result_items = self._predict_fallback(image_path)

            score = max((item.get("value", {}).get("score", 0.0) for item in result_items), default=0.0)
            predictions.append({"result": result_items, "score": float(score)})

        return predictions

    # ---------------------------
    # Training hooks (Label Studio)
    # ---------------------------

    def process_event(self, event, data, job_id, additional_params=None):
        additional_params = additional_params or {}

        # Explicit manual retrain from LS button (modern flow)
        if event == "START_TRAINING":
            # Allow direct task payload for non-Label-Studio callers (custom local app).
            direct_tasks = data.get("tasks") if isinstance(data, dict) else None
            if isinstance(direct_tasks, list) and direct_tasks:
                LOGGER.info("START_TRAINING received with direct task payload: tasks=%s", len(direct_tasks))
                return self.fit(direct_tasks, event=event, data=data, job_id=job_id, **additional_params)

            project_id = self._extract_project_id_from_event(data)
            tasks = self._fetch_annotated_tasks_from_ls(project_id)
            LOGGER.info("START_TRAINING received: project_id=%s annotated_tasks=%s", project_id, len(tasks))
            return self.fit(tasks, event=event, data=data, job_id=job_id, **additional_params)

        # Optional webhook-based retraining on each annotation update
        if event in self.TRAIN_EVENTS:
            if not self.sequence_train_on_annotation_events:
                return {
                    "status": "skipped",
                    "reason": "annotation-event-training-disabled",
                    "event": event,
                }
            project_id = self._extract_project_id_from_event(data)
            tasks = self._fetch_annotated_tasks_from_ls(project_id)
            LOGGER.info("%s received: project_id=%s annotated_tasks=%s", event, project_id, len(tasks))
            return self.fit(tasks, event=event, data=data, job_id=job_id, **additional_params)

        return {"status": "skipped", "reason": "event-not-supported", "event": event}

    def fit(self, tasks, workdir=None, **kwargs):
        """Train a light sequence KNN model from accepted annotations.

        This enables the loop:
          label ~15m -> Train in LS -> Retrieve Predictions -> continue.
        """
        task_list = list(tasks or [])
        samples = self._extract_samples_from_tasks(task_list)

        if len(samples) < self.sequence_min_train_samples:
            return {
                "status": "insufficient-data",
                "sample_count": len(samples),
                "required_samples": self.sequence_min_train_samples,
                "message": "Annotate a few more frames, then click Train again.",
            }

        samples = sorted(samples, key=lambda s: s.frame)
        metrics = self._evaluate_samples(samples)

        payload = {
            "status": "ok",
            "trainer": "sequence-knn-v1",
            "sample_count": len(samples),
            "k_neighbors": self.sequence_k_neighbors,
            "metrics": metrics,
            "sequence_samples": [self._sample_to_wire(s) for s in samples],
        }

        LOGGER.info(
            "Training complete: samples=%s mae_pct=%.4f mae_px=%s",
            len(samples),
            float(metrics.get("mae_pct", 0.0)),
            metrics.get("mae_px"),
        )

        return payload

    # ---------------------------
    # Sequence learner internals
    # ---------------------------

    def _sample_to_wire(self, s: SequenceSample) -> dict[str, Any]:
        return {
            "f": int(s.frame),
            "x": float(s.x_pct),
            "y": float(s.y_pct),
            "w": float(s.w_pct),
            "h": float(s.h_pct),
            "iw": int(s.img_w),
            "ih": int(s.img_h),
        }

    def _load_sequence_samples(self, train_payload: dict[str, Any]) -> list[SequenceSample]:
        raw = train_payload.get("sequence_samples", []) if isinstance(train_payload, dict) else []
        out: list[SequenceSample] = []

        for item in raw:
            try:
                out.append(
                    SequenceSample(
                        frame=int(item.get("f")),
                        x_pct=float(item.get("x")),
                        y_pct=float(item.get("y")),
                        w_pct=float(item.get("w", self.sequence_default_box_w_pct)),
                        h_pct=float(item.get("h", self.sequence_default_box_h_pct)),
                        img_w=max(1, int(item.get("iw", 1))),
                        img_h=max(1, int(item.get("ih", 1))),
                    )
                )
            except Exception:
                continue

        # keep one sample per frame (latest wins)
        unique: dict[int, SequenceSample] = {}
        for s in out:
            unique[s.frame] = s

        return [unique[k] for k in sorted(unique.keys())]

    def _predict_sequence(self, task: dict[str, Any], image_path: str) -> list[dict[str, Any]] | None:
        if len(self._sequence_samples) < self.sequence_min_train_samples:
            return None

        frame = self._extract_frame_number(task)
        if frame is None:
            return None

        pred = self._interpolate_for_frame(frame)
        if pred is None:
            return None

        x_pct, y_pct, w_pct, h_pct, confidence = pred

        try:
            image = Image.open(image_path)
            width_px, height_px = image.size
        except Exception:
            return None

        x_min = (x_pct - w_pct / 2.0) / 100.0 * width_px
        y_min = (y_pct - h_pct / 2.0) / 100.0 * height_px
        x_max = (x_pct + w_pct / 2.0) / 100.0 * width_px
        y_max = (y_pct + h_pct / 2.0) / 100.0 * height_px

        region_id = f"r_{uuid4().hex[:12]}"
        result = [
            self._to_ls_region_result(
                label=self.primary_label,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                width_px=width_px,
                height_px=height_px,
                score=confidence,
                region_id=region_id,
                x_center_pct=x_pct,
                y_center_pct=y_pct,
            )
        ]

        if self.body_part_from_name and self.body_part_choices:
            other_choice = self._resolve_choice("other", self.body_part_choices)
            if other_choice:
                result.append(self._to_ls_choice_result(region_id, self.body_part_from_name, other_choice))

        return result

    def _interpolate_for_frame(self, frame: int) -> tuple[float, float, float, float, float] | None:
        if self._sequence_frames.size == 0:
            return None

        diffs = np.abs(self._sequence_frames - float(frame))
        nearest_order = np.argsort(diffs)
        k = min(self.sequence_k_neighbors, len(nearest_order))
        if k <= 0:
            return None

        idx = nearest_order[:k]
        selected = [self._sequence_samples[int(i)] for i in idx]
        selected_diffs = [float(diffs[int(i)]) for i in idx]

        # Exact frame known
        if selected and selected_diffs[0] == 0.0:
            s = selected[0]
            return (s.x_pct, s.y_pct, s.w_pct, s.h_pct, 0.99)

        weights = np.array([1.0 / (d + 1e-6) for d in selected_diffs], dtype=np.float64)
        weights = weights / weights.sum()

        x = float(sum(w * s.x_pct for w, s in zip(weights, selected)))
        y = float(sum(w * s.y_pct for w, s in zip(weights, selected)))
        w_pct = float(sum(wi * s.w_pct for wi, s in zip(weights, selected)))
        h_pct = float(sum(wi * s.h_pct for wi, s in zip(weights, selected)))

        nearest = selected_diffs[0] if selected_diffs else 1e9
        confidence = float(max(0.25, min(0.95, 1.0 / (1.0 + nearest))))

        return (x, y, max(1.0, w_pct), max(1.0, h_pct), confidence)

    def _extract_samples_from_tasks(self, tasks: list[dict[str, Any]]) -> list[SequenceSample]:
        by_frame: dict[int, SequenceSample] = {}

        for task in tasks:
            frame = self._extract_frame_number(task)
            if frame is None:
                continue

            annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
            if not annotations:
                continue

            annotation = sorted(
                annotations,
                key=lambda a: (str(a.get("updated_at") or ""), int(a.get("id") or 0)),
            )[-1]

            region_points = self._extract_region_points(annotation)
            if not region_points:
                continue

            x_pct = float(np.mean([p[0] for p in region_points]))
            y_pct = float(np.mean([p[1] for p in region_points]))
            w_vals = [p[2] for p in region_points if p[2] is not None]
            h_vals = [p[3] for p in region_points if p[3] is not None]
            w_pct = float(np.mean(w_vals)) if w_vals else self.sequence_default_box_w_pct
            h_pct = float(np.mean(h_vals)) if h_vals else self.sequence_default_box_h_pct

            img_w, img_h = self._extract_image_size_from_annotation(annotation)
            by_frame[frame] = SequenceSample(
                frame=frame,
                x_pct=x_pct,
                y_pct=y_pct,
                w_pct=w_pct,
                h_pct=h_pct,
                img_w=img_w,
                img_h=img_h,
            )

        return [by_frame[k] for k in sorted(by_frame.keys())]

    def _extract_region_points(self, annotation: dict[str, Any]) -> list[tuple[float, float, float | None, float | None]]:
        out: list[tuple[float, float, float | None, float | None]] = []

        for item in annotation.get("result", []) or []:
            if item.get("from_name") != self.from_name:
                continue

            value = item.get("value", {})
            item_type = str(item.get("type") or "").lower()

            # Rectangle region
            if item_type == "rectanglelabels" and "x" in value and "y" in value:
                x = float(value.get("x", 0.0))
                y = float(value.get("y", 0.0))
                w = float(value.get("width", self.sequence_default_box_w_pct))
                h = float(value.get("height", self.sequence_default_box_h_pct))
                out.append((x + w / 2.0, y + h / 2.0, w, h))
                continue

            # Keypoint region
            if item_type == "keypointlabels" and "x" in value and "y" in value:
                x = float(value.get("x", 0.0))
                y = float(value.get("y", 0.0))
                out.append((x, y, self.sequence_default_box_w_pct, self.sequence_default_box_h_pct))

        return out

    def _extract_image_size_from_annotation(self, annotation: dict[str, Any]) -> tuple[int, int]:
        for item in annotation.get("result", []) or []:
            try:
                iw = int(item.get("original_width") or 0)
                ih = int(item.get("original_height") or 0)
            except Exception:
                iw, ih = 0, 0
            if iw > 0 and ih > 0:
                return iw, ih
        return (1, 1)

    def _evaluate_samples(self, samples: list[SequenceSample]) -> dict[str, Any]:
        if len(samples) < 3:
            return {"mae_pct": None, "mae_px": None, "eval_count": 0}

        max_eval = min(400, len(samples))
        eval_idx = np.linspace(0, len(samples) - 1, max_eval, dtype=int)

        errors_pct: list[float] = []
        errors_px: list[float] = []

        for idx in eval_idx:
            truth = samples[int(idx)]
            others = samples[: int(idx)] + samples[int(idx) + 1 :]
            if len(others) < 1:
                continue

            pred = self._interpolate_from_samples(truth.frame, others)
            if pred is None:
                continue

            px, py = pred[0], pred[1]
            dx = px - truth.x_pct
            dy = py - truth.y_pct
            dist_pct = math.sqrt(dx * dx + dy * dy)
            errors_pct.append(dist_pct)

            if truth.img_w > 1 and truth.img_h > 1:
                dx_px = dx / 100.0 * truth.img_w
                dy_px = dy / 100.0 * truth.img_h
                errors_px.append(math.sqrt(dx_px * dx_px + dy_px * dy_px))

        mae_pct = float(np.mean(errors_pct)) if errors_pct else None
        mae_px = float(np.mean(errors_px)) if errors_px else None
        return {
            "mae_pct": mae_pct,
            "mae_px": mae_px,
            "eval_count": len(errors_pct),
        }

    def _interpolate_from_samples(
        self,
        frame: int,
        samples: list[SequenceSample],
    ) -> tuple[float, float, float, float] | None:
        if not samples:
            return None

        ordered = sorted(samples, key=lambda s: abs(s.frame - frame))
        use = ordered[: min(self.sequence_k_neighbors, len(ordered))]

        if use and use[0].frame == frame:
            s = use[0]
            return (s.x_pct, s.y_pct, s.w_pct, s.h_pct)

        diffs = np.array([abs(s.frame - frame) for s in use], dtype=np.float64)
        weights = 1.0 / (diffs + 1e-6)
        weights = weights / weights.sum()

        x = float(sum(w * s.x_pct for w, s in zip(weights, use)))
        y = float(sum(w * s.y_pct for w, s in zip(weights, use)))
        w_pct = float(sum(wi * s.w_pct for wi, s in zip(weights, use)))
        h_pct = float(sum(wi * s.h_pct for wi, s in zip(weights, use)))
        return (x, y, w_pct, h_pct)

    # ---------------------------
    # Label Studio API helpers (for START_TRAINING webhook flow)
    # ---------------------------

    def _extract_project_id_from_event(self, data: dict[str, Any]) -> int | None:
        if not isinstance(data, dict):
            return None

        project = data.get("project")
        if isinstance(project, dict):
            pid = project.get("id")
        else:
            pid = project

        if pid is not None:
            try:
                return int(pid)
            except (TypeError, ValueError):
                return None

        return None

    def _fetch_annotated_tasks_from_ls(self, project_id: int | None) -> list[dict[str, Any]]:
        if not project_id:
            LOGGER.warning("No project id in training webhook payload; skipping.")
            return []

        base_url = (self.hostname or os.getenv("LABEL_STUDIO_URL") or "").rstrip("/")
        token = self.access_token or os.getenv("LABEL_STUDIO_API_KEY") or os.getenv("LABEL_STUDIO_API_TOKEN") or ""

        if not base_url:
            LOGGER.warning("Missing Label Studio URL; cannot fetch annotated tasks.")
            return []

        session: requests.Session | None = None
        headers: dict[str, str] = {}

        if token:
            headers["Authorization"] = f"Token {token}"
        else:
            username = os.getenv("LABEL_STUDIO_USERNAME")
            password = os.getenv("LABEL_STUDIO_PASSWORD")
            if not username or not password:
                LOGGER.warning("Missing token and username/password; cannot fetch annotated tasks.")
                return []
            session = self._login_ls_session(base_url, username, password)
            if session is None:
                LOGGER.warning("Session login failed; cannot fetch annotated tasks.")
                return []

        tasks: list[dict[str, Any]] = []
        page = 1
        page_size = 200

        while True:
            params = {
                "project": project_id,
                "fields": "all",
                "page": page,
                "page_size": page_size,
            }

            url = f"{base_url}/api/tasks"
            try:
                if session is not None:
                    resp = session.get(url, params=params, timeout=60)
                else:
                    resp = requests.get(url, params=params, headers=headers, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                LOGGER.exception("Failed to fetch tasks page=%s project=%s: %s", page, project_id, exc)
                break

            if isinstance(payload, list):
                page_tasks = payload
                total = len(payload)
            elif isinstance(payload, dict):
                page_tasks = payload.get("tasks", [])
                total = int(payload.get("total", 0))
            else:
                page_tasks = []
                total = 0

            if not page_tasks:
                break

            tasks.extend(t for t in page_tasks if isinstance(t, dict) and t.get("annotations"))

            if total > 0 and page * page_size >= total:
                break

            page += 1

        return tasks

    def _login_ls_session(self, base_url: str, username: str, password: str) -> requests.Session | None:
        session = requests.Session()
        login_url = f"{base_url}/user/login"

        try:
            first = session.get(login_url, timeout=30)
            first.raise_for_status()
        except Exception as exc:
            LOGGER.exception("Failed opening LS login page: %s", exc)
            return None

        match = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', first.text)
        if not match:
            LOGGER.warning("Could not parse CSRF token on login page")
            return None

        form = {
            "email": username,
            "password": password,
            "csrfmiddlewaretoken": match.group(1),
        }

        try:
            resp = session.post(
                login_url,
                data=form,
                headers={"Referer": login_url},
                allow_redirects=False,
                timeout=30,
            )
            if resp.status_code not in (200, 302):
                LOGGER.warning("Session login failed with status=%s", resp.status_code)
                return None

            who = session.get(f"{base_url}/api/current-user/whoami", timeout=30)
            if who.status_code != 200:
                LOGGER.warning("Session verification failed with status=%s", who.status_code)
                return None
        except Exception as exc:
            LOGGER.exception("Session login request failed: %s", exc)
            return None

        return session

    # ---------------------------
    # Task / image helpers
    # ---------------------------

    def _extract_frame_number(self, task: dict[str, Any]) -> int | None:
        data = task.get("data", {}) if isinstance(task, dict) else {}
        raw = data.get(self.image_key)

        # try explicit image key first
        if raw:
            parsed = self._parse_frame_from_pathlike(str(raw))
            if parsed is not None:
                return parsed

        # fallback: any data string that resembles a file
        for value in data.values():
            if isinstance(value, str):
                parsed = self._parse_frame_from_pathlike(value)
                if parsed is not None:
                    return parsed

        # fallback: LS inner_id then id
        for key in ("inner_id", "id"):
            val = task.get(key)
            try:
                if val is not None:
                    return int(val)
            except (TypeError, ValueError):
                continue

        return None

    def _parse_frame_from_pathlike(self, value: str) -> int | None:
        try:
            parsed = urlparse(value)
            if "/data/local-files/" in parsed.path:
                rel = parse_qs(parsed.query).get("d", [])
                if rel:
                    value = unquote(rel[0])
        except Exception:
            pass

        stem = Path(value).stem
        nums = re.findall(r"(\d+)", stem)
        if not nums:
            return None

        try:
            return int(nums[-1])
        except ValueError:
            return None

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

    # ---------------------------
    # Base detector modes
    # ---------------------------

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
                self._to_ls_region_result(
                    label=mapped_label,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    width_px=width,
                    height_px=height,
                    score=float(score),
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
                self._to_ls_region_result(
                    label=self.primary_label,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    width_px=w,
                    height_px=h,
                    score=conf,
                    region_id=region_id,
                )
            )

            if self.body_part_from_name and self.body_part_choices:
                other_choice = self._resolve_choice("other", self.body_part_choices)
                if other_choice:
                    results.append(self._to_ls_choice_result(region_id, self.body_part_from_name, other_choice))

        return results

    # ---------------------------
    # Label Studio result shaping
    # ---------------------------

    def _to_ls_region_result(
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
        x_center_pct: float | None = None,
        y_center_pct: float | None = None,
    ) -> dict[str, Any]:
        width_px = max(1, int(width_px))
        height_px = max(1, int(height_px))

        x_min = max(0.0, min(x_min, float(width_px - 1)))
        y_min = max(0.0, min(y_min, float(height_px - 1)))
        x_max = max(x_min + 1.0, min(x_max, float(width_px)))
        y_max = max(y_min + 1.0, min(y_max, float(height_px)))

        x_pct = 100.0 * x_min / float(width_px)
        y_pct = 100.0 * y_min / float(height_px)
        w_pct = 100.0 * (x_max - x_min) / float(width_px)
        h_pct = 100.0 * (y_max - y_min) / float(height_px)

        if self.primary_control_type == "keypointlabels":
            if x_center_pct is None:
                x_center_pct = x_pct + w_pct / 2.0
            if y_center_pct is None:
                y_center_pct = y_pct + h_pct / 2.0

            result = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "keypointlabels",
                "value": {
                    "x": float(x_center_pct),
                    "y": float(y_center_pct),
                    "keypointlabels": [label],
                    "score": float(score),
                },
            }
        else:
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

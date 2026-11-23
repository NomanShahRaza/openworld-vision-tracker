from __future__ import annotations

import contextlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.config import Settings
from app.visualization import mask_to_polygon


@dataclass
class Detection:
    box: np.ndarray
    score: float
    label: str
    mask: np.ndarray | None
    embedding: np.ndarray
    semantic_similarity: float
    polygon: list[list[int]] | None = None
    track_id: int | None = None


class InferenceCancelled(Exception):
    """Raised when the user stops a running inference job."""


class OpenWorldEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._job_id = 0
        self._cancelled_job = 0
        self.loaded = False
        self.device = self._resolve_device(settings.device)

        self.detector_processor: Any = None
        self.detector_model: Any = None
        self.sam_processor: Any = None
        self.sam_model: Any = None
        self.clip_processor: Any = None
        self.clip_model: Any = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        import torch

        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def parse_prompts(prompt_text: str) -> list[str]:
        prompts = [
            part.strip().strip(".")
            for part in re.split(r"[\n;]+", prompt_text)
            if part.strip().strip(".")
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for prompt in prompts:
            key = prompt.casefold()
            if key not in seen:
                seen.add(key)
                deduped.append(prompt)
        return deduped[:20]

    def begin_job(self) -> int:
        with self._job_lock:
            self._job_id += 1
            return self._job_id

    def cancel(self) -> None:
        with self._job_lock:
            self._cancelled_job = self._job_id

    def raise_if_cancelled(self, job_id: int | None) -> None:
        if not job_id:
            return
        with self._job_lock:
            cancelled = self._cancelled_job == job_id
        if cancelled:
            raise InferenceCancelled("Stopped")

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return

            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                CLIPModel,
                SamModel,
            )

            self.detector_processor = AutoProcessor.from_pretrained(self.settings.detector_model)
            self.detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.settings.detector_model
            ).to(self.device)
            self.detector_model.eval()

            self.sam_processor = AutoProcessor.from_pretrained(self.settings.sam_model)
            self.sam_model = SamModel.from_pretrained(self.settings.sam_model).to(self.device)
            self.sam_model.eval()

            self.clip_processor = AutoProcessor.from_pretrained(self.settings.clip_model)
            self.clip_model = CLIPModel.from_pretrained(self.settings.clip_model).to(self.device)
            self.clip_model.eval()

            self.loaded = True

    def _autocast(self):
        import torch

        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def infer_image(
        self,
        image: Image.Image,
        prompts: list[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        max_objects: int = 20,
        include_polygon: bool = True,
        job_id: int | None = None,
    ) -> tuple[list[Detection], float]:
        if not prompts:
            raise ValueError("At least one target description is required")
        self.raise_if_cancelled(job_id)
        self.load()
        self.raise_if_cancelled(job_id)
        image = image.convert("RGB")

        start = time.perf_counter()
        with self._infer_lock:
            self.raise_if_cancelled(job_id)
            boxes, scores, labels = self._detect(
                image=image,
                prompts=prompts,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_objects=max_objects,
            )
            if len(boxes) == 0:
                return [], (time.perf_counter() - start) * 1000.0

            self.raise_if_cancelled(job_id)
            masks = self._segment(image, boxes)
            self.raise_if_cancelled(job_id)
            embeddings, semantic_scores = self._embed_and_score(image, boxes, prompts)

        detections: list[Detection] = []
        for index in range(len(boxes)):
            mask = masks[index] if index < len(masks) else None
            polygon = mask_to_polygon(mask) if include_polygon and mask is not None else None
            detections.append(
                Detection(
                    box=boxes[index],
                    score=float(scores[index]),
                    label=str(labels[index]),
                    mask=mask,
                    embedding=embeddings[index],
                    semantic_similarity=float(semantic_scores[index]),
                    polygon=polygon,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return detections, elapsed_ms

    def _detect(
        self,
        image: Image.Image,
        prompts: list[str],
        box_threshold: float,
        text_threshold: float,
        max_objects: int,
    ) -> tuple[list[np.ndarray], list[float], list[str]]:
        import torch

        text_labels = [prompts]
        inputs = self.detector_processor(
            images=image,
            text=text_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode(), self._autocast():
            outputs = self.detector_model(**inputs)

        result = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            target_sizes=[(image.height, image.width)],
        )[0]

        raw_boxes = result.get("boxes", [])
        raw_scores = result.get("scores", [])
        raw_labels = result.get("text_labels", result.get("labels", []))

        candidates: list[tuple[np.ndarray, float, str]] = []
        for box, score, label in zip(raw_boxes, raw_scores, raw_labels):
            box_np = box.detach().float().cpu().numpy().astype(np.float32)
            score_value = float(score.detach().float().cpu().item())
            x1, y1, x2, y2 = box_np.tolist()
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            label_text = self._normalize_detector_label(label, prompts)
            candidates.append((box_np, score_value, label_text))

        candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = candidates[: max(1, min(max_objects, self.settings.max_objects))]
        return (
            [c[0] for c in candidates],
            [c[1] for c in candidates],
            [c[2] for c in candidates],
        )

    @staticmethod
    def _normalize_detector_label(label: Any, prompts: list[str]) -> str:
        if isinstance(label, str):
            return label.strip().strip(".") or "object"
        try:
            index = int(label)
            if 0 <= index < len(prompts):
                return prompts[index]
        except (TypeError, ValueError):
            pass
        return str(label)

    def _segment(self, image: Image.Image, boxes: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        box_list = [box.tolist() for box in boxes]
        inputs = self.sam_processor(
            images=image,
            input_boxes=[box_list],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode(), self._autocast():
            outputs = self.sam_model(**inputs, multimask_output=False)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().float().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]

        masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        return [(mask > 0).astype(bool) for mask in masks_np]

    def _embed_and_score(
        self,
        image: Image.Image,
        boxes: list[np.ndarray],
        prompts: list[str],
    ) -> tuple[list[np.ndarray], list[float]]:
        import torch

        crops: list[Image.Image] = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, int(np.floor(x1)))
            y1 = max(0, int(np.floor(y1)))
            x2 = min(image.width, int(np.ceil(x2)))
            y2 = min(image.height, int(np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32)))
            else:
                crops.append(image.crop((x1, y1, x2, y2)))

        image_inputs = self.clip_processor(images=crops, return_tensors="pt")
        image_pixel_values = image_inputs["pixel_values"].to(self.device)
        text_inputs = self.clip_processor(
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode(), self._autocast():
            image_features = self.clip_model.get_image_features(pixel_values=image_pixel_values)
            text_features = self.clip_model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )

        image_features = self._feature_tensor(image_features).float()
        text_features = self._feature_tensor(text_features).float()
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        similarity = image_features @ text_features.T
        semantic_scores = similarity.max(dim=1).values

        embeddings = [row.cpu().numpy().astype(np.float32) for row in image_features]
        scores = [float(value.cpu().item()) for value in semantic_scores]
        return embeddings, scores

    @staticmethod
    def _feature_tensor(value: Any):
        if hasattr(value, "pooler_output") and value.pooler_output is not None:
            return value.pooler_output
        if hasattr(value, "last_hidden_state") and value.last_hidden_state is not None:
            return value.last_hidden_state[:, 0]
        if isinstance(value, tuple):
            return value[0]
        return value

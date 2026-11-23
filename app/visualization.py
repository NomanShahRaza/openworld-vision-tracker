from __future__ import annotations

import colorsys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def color_for_id(identifier: int) -> tuple[int, int, int]:
    hue = (identifier * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def mask_to_polygon(mask: np.ndarray, max_points: int = 80) -> list[list[int]] | None:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, perimeter * 0.005)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) > max_points:
        step = max(1, len(approx) // max_points)
        approx = approx[::step][:max_points]
    return [[int(x), int(y)] for x, y in approx]


def annotate_image(image: Image.Image, detections: Iterable[object]) -> Image.Image:
    canvas = np.asarray(image.convert("RGB")).copy()
    overlay = canvas.copy()

    detections = list(detections)
    for index, det in enumerate(detections):
        track_id = getattr(det, "track_id", None)
        color = color_for_id(track_id or (index + 1))
        mask = getattr(det, "mask", None)
        if mask is not None and mask.shape[:2] == canvas.shape[:2]:
            overlay[mask.astype(bool)] = color

    canvas = cv2.addWeighted(overlay, 0.34, canvas, 0.66, 0)

    for index, det in enumerate(detections):
        track_id = getattr(det, "track_id", None)
        color = color_for_id(track_id or (index + 1))
        x1, y1, x2, y2 = [int(round(v)) for v in getattr(det, "box")]
        x1 = max(0, min(x1, canvas.shape[1] - 1))
        y1 = max(0, min(y1, canvas.shape[0] - 1))
        x2 = max(0, min(x2, canvas.shape[1] - 1))
        y2 = max(0, min(y2, canvas.shape[0] - 1))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = str(getattr(det, "label", "object"))
        score = float(getattr(det, "score", 0.0))
        semantic = float(getattr(det, "semantic_similarity", 0.0))
        prefix = f"#{track_id} " if track_id is not None else ""
        caption = f"{prefix}{label}  {score:.2f} | sem {semantic:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
        label_y1 = max(0, y1 - text_h - baseline - 8)
        label_y2 = label_y1 + text_h + baseline + 8
        label_x2 = min(canvas.shape[1] - 1, x1 + text_w + 10)
        cv2.rectangle(canvas, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            canvas,
            caption,
            (x1 + 5, label_y2 - baseline - 4),
            font,
            font_scale,
            (8, 12, 18),
            thickness,
            cv2.LINE_AA,
        )

    return Image.fromarray(canvas)

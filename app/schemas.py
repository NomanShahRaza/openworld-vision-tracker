from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    track_id: int | None = None
    label: str
    score: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float
    box: list[float]
    polygon: list[list[int]] | None = None


class ImageResult(BaseModel):
    media_type: Literal["image"] = "image"
    result_url: str
    download_name: str
    prompts: list[str]
    detections: list[DetectionOut]
    object_count: int
    elapsed_ms: float
    device: str


class TrackSummary(BaseModel):
    track_id: int
    label: str
    first_seen_s: float
    last_seen_s: float
    observations: int
    max_confidence: float
    best_semantic_similarity: float


class VideoResult(BaseModel):
    media_type: Literal["video"] = "video"
    result_url: str
    download_name: str
    prompts: list[str]
    tracks: list[TrackSummary]
    unique_tracks: int
    processed_frames: int
    source_duration_s: float
    process_fps: float
    elapsed_ms: float
    device: str


class HealthResult(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    device: str
    models_loaded: bool


class ModelInfo(BaseModel):
    detector: str
    segmenter: str
    embedder: str
    max_video_seconds: int
    max_upload_mb: int
    default_video_process_fps: float

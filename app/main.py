from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.inference import InferenceCancelled, OpenWorldEngine
from app.media import process_video, save_annotated_image
from app.schemas import (
    DetectionOut,
    HealthResult,
    ImageResult,
    ModelInfo,
    TrackSummary,
    VideoResult,
)


settings = get_settings()
engine = OpenWorldEngine(settings)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Zero-shot open-world object discovery using Grounding DINO, SAM, "
        "CLIP-like embeddings, and semantic tracking."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=settings.output_dir), name="results")
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/api/health", response_model=HealthResult)
def health() -> HealthResult:
    return HealthResult(
        app=settings.app_name,
        version=settings.app_version,
        device=engine.device,
        models_loaded=engine.loaded,
    )


@app.get("/api/models", response_model=ModelInfo)
def models() -> ModelInfo:
    return ModelInfo(
        detector=settings.detector_model,
        segmenter=settings.sam_model,
        embedder=settings.clip_model,
        max_video_seconds=settings.max_video_seconds,
        max_upload_mb=settings.max_upload_mb,
        default_video_process_fps=settings.default_video_process_fps,
    )


@app.post("/api/warmup")
async def warmup() -> dict:
    await run_in_threadpool(engine.load)
    return {"status": "ready", "device": engine.device}


@app.post("/api/stop")
def stop_inference() -> dict:
    engine.cancel()
    return {"status": "stopping"}


@app.post("/api/reveal")
def reveal_result(filename: str = Form(...)) -> dict:
    """Open the results folder with this annotated file selected. No browser download."""
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=404, detail="Result not found")
    path = (settings.output_dir / safe_name).resolve()
    if path.parent != settings.output_dir.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    _open_in_folder(path)
    return {"folder": str(settings.output_dir), "filename": safe_name}


@app.post("/api/image", response_model=ImageResult)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    max_objects: int = Form(12),
) -> ImageResult:
    prompts = engine.parse_prompts(prompt)
    _validate_prompts(prompts)
    _validate_thresholds(box_threshold, text_threshold)
    max_objects = _validate_max_objects(max_objects)

    payload = await _read_upload(file)
    job_id = engine.begin_job()
    try:
        from io import BytesIO

        image = Image.open(BytesIO(payload)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from exc

    try:
        detections, elapsed_ms = await run_in_threadpool(
            engine.infer_image,
            image,
            prompts,
            box_threshold,
            text_threshold,
            max_objects,
            True,
            job_id,
        )
        output_path = await run_in_threadpool(
            save_annotated_image,
            settings.output_dir,
            image,
            detections,
            file.filename,
        )
    except InferenceCancelled as exc:
        raise HTTPException(status_code=499, detail="Stopped") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return ImageResult(
        result_url=f"/results/{quote(output_path.name)}",
        download_name=output_path.name,
        prompts=prompts,
        detections=[
            DetectionOut(
                track_id=det.track_id,
                label=det.label,
                score=round(det.score, 4),
                semantic_similarity=round(det.semantic_similarity, 4),
                box=[round(float(v), 2) for v in det.box.tolist()],
                polygon=det.polygon,
            )
            for det in detections
        ],
        object_count=len(detections),
        elapsed_ms=round(elapsed_ms, 2),
        device=engine.device,
    )


@app.post("/api/video", response_model=VideoResult)
async def analyze_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    max_objects: int = Form(10),
    process_fps: float = Form(4.0),
) -> VideoResult:
    prompts = engine.parse_prompts(prompt)
    _validate_prompts(prompts)
    _validate_thresholds(box_threshold, text_threshold)
    max_objects = _validate_max_objects(max_objects)
    if not 0.5 <= process_fps <= 12.0:
        raise HTTPException(status_code=422, detail="process_fps must be between 0.5 and 12")

    payload = await _read_upload(file)
    job_id = engine.begin_job()
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(status_code=400, detail="Unsupported video file extension")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(payload)
            temp_path = Path(temp.name)

        result = await run_in_threadpool(
            process_video,
            engine,
            temp_path,
            settings.output_dir,
            prompts,
            box_threshold,
            text_threshold,
            max_objects,
            process_fps,
            settings.max_video_seconds,
            job_id,
            file.filename,
        )
    except InferenceCancelled as exc:
        raise HTTPException(status_code=499, detail="Stopped") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return VideoResult(
        result_url=f"/results/{quote(result.output_path.name)}",
        download_name=result.output_path.name,
        prompts=prompts,
        tracks=[TrackSummary(**item) for item in result.track_summaries],
        unique_tracks=len(result.track_summaries),
        processed_frames=result.processed_frames,
        source_duration_s=round(result.source_duration_s, 3),
        process_fps=round(result.process_fps, 3),
        elapsed_ms=round(result.elapsed_ms, 2),
        device=engine.device,
    )


async def _read_upload(file: UploadFile) -> bytes:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {settings.max_upload_mb} MB limit",
        )
    return payload


def _validate_prompts(prompts: list[str]) -> None:
    if not prompts:
        raise HTTPException(
            status_code=422,
            detail="Enter at least one target description. Separate multiple targets with new lines or semicolons.",
        )


def _validate_thresholds(box_threshold: float, text_threshold: float) -> None:
    if not 0.05 <= box_threshold <= 0.95:
        raise HTTPException(status_code=422, detail="box_threshold must be between 0.05 and 0.95")
    if not 0.05 <= text_threshold <= 0.95:
        raise HTTPException(status_code=422, detail="text_threshold must be between 0.05 and 0.95")


def _open_in_folder(path: Path) -> None:
    if sys.platform == "win32":
        subprocess.Popen(["explorer.exe", f"/select,{path}"])
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path.parent)])


def _validate_max_objects(value: int) -> int:
    if not 1 <= value <= settings.max_objects:
        raise HTTPException(
            status_code=422,
            detail=f"max_objects must be between 1 and {settings.max_objects}",
        )
    return value

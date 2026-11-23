from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

from app.inference import Detection, InferenceCancelled, OpenWorldEngine
from app.tracking import BotSort
from app.visualization import annotate_image


@dataclass
class VideoProcessingResult:
    output_path: Path
    track_summaries: list[dict]
    processed_frames: int
    source_duration_s: float
    process_fps: float
    elapsed_ms: float


def new_output_path(output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{uuid4().hex}{suffix}"


def result_output_path(output_dir: Path, source_name: str | None, suffix: str) -> Path:
    """Save into the results folder as '<original>_<date>_<time><suffix>'."""
    raw_stem = Path(source_name or "result").stem
    safe = re.sub(r"[^\w.\-]+", "_", raw_stem, flags=re.UNICODE).strip("._") or "result"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = output_dir / f"{safe}_{stamp}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, 100):
        alt = output_dir / f"{safe}_{stamp}_{index}{suffix}"
        if not alt.exists():
            return alt
    return output_dir / f"{safe}_{stamp}_{uuid4().hex[:6]}{suffix}"


def save_annotated_image(
    output_dir: Path,
    image: Image.Image,
    detections: list[Detection],
    source_name: str | None = None,
) -> Path:
    path = result_output_path(output_dir, source_name, ".png")
    annotate_image(image, detections).save(path, format="PNG", optimize=True)
    return path


class _FrameWriter:
    """Write annotated frames. OpenCV mp4v is tried first; ffmpeg covers sizes it rejects."""

    def __init__(self, raw_path: Path, final_path: Path, fps: float, width: int, height: int):
        self.raw_path = raw_path
        self.final_path = final_path
        self.fps = max(0.5, float(fps))
        self.width = width if width % 2 == 0 else width + 1
        self.height = height if height % 2 == 0 else height + 1
        self.direct = False
        self._cv: cv2.VideoWriter | None = None
        self._proc: subprocess.Popen | None = None
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._try_opencv():
            self._start_ffmpeg()
            self.direct = True

    def _try_opencv(self) -> bool:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = (self.width, self.height)
        api = getattr(cv2, "CAP_FFMPEG", None)
        candidates = []
        if api is not None:
            candidates.append(lambda: cv2.VideoWriter(str(self.raw_path), api, fourcc, self.fps, size))
        candidates.append(lambda: cv2.VideoWriter(str(self.raw_path), fourcc, self.fps, size))
        for open_writer in candidates:
            writer = open_writer()
            if writer.isOpened():
                self._cv = writer
                return True
            writer.release()
        return False

    def _start_ffmpeg(self) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.final_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        frame = _fit_frame(frame_bgr, self.width, self.height)
        if self._cv is not None:
            self._cv.write(frame)
            return
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise ValueError(self._failure_message())
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise ValueError(self._failure_message()) from exc

    def finish(self) -> Path:
        if self._cv is not None:
            self._cv.release()
            self._cv = None
            return self.raw_path
        self._close_ffmpeg()
        return self.final_path

    def abort(self) -> None:
        if self._cv is not None:
            self._cv.release()
            self._cv = None
        if self._proc is not None:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except OSError:
                    pass
            self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            self._proc = None
        self.raw_path.unlink(missing_ok=True)
        if self.direct:
            self.final_path.unlink(missing_ok=True)

    def _close_ffmpeg(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        err = ""
        if self._proc.stderr is not None:
            err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
        code = self._proc.wait(timeout=180)
        self._proc = None
        if code != 0 or not self.final_path.exists():
            raise ValueError(f"Could not initialize the video encoder. {err}".strip())

    def _failure_message(self) -> str:
        err = ""
        if self._proc is not None and self._proc.stderr is not None:
            err = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
        return f"Could not initialize the video encoder. {err}".strip()


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    if frame_w == width and frame_h == height:
        return np.ascontiguousarray(frame)
    if frame_w > width or frame_h > height:
        return np.ascontiguousarray(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    return np.ascontiguousarray(
        cv2.copyMakeBorder(frame, 0, height - frame_h, 0, width - frame_w, cv2.BORDER_REPLICATE)
    )


def process_video(
    engine: OpenWorldEngine,
    input_path: Path,
    output_dir: Path,
    prompts: list[str],
    box_threshold: float,
    text_threshold: float,
    max_objects: int,
    process_fps: float,
    max_duration_s: int,
    job_id: int | None = None,
    source_name: str | None = None,
) -> VideoProcessingResult:
    start = time.perf_counter()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video metadata could not be read")

    duration_s = frame_count / src_fps
    if duration_s > max_duration_s + 0.25:
        cap.release()
        raise ValueError(
            f"Video is {duration_s:.1f}s; the configured limit is {max_duration_s}s"
        )

    requested_fps = max(0.5, float(process_fps))
    sample_every = max(1, int(round(src_fps / requested_fps)))
    actual_process_fps = src_fps / sample_every

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise ValueError("Could not read frames from the uploaded video")
    height, width = first_frame.shape[:2]

    raw_path = new_output_path(output_dir, ".raw.mp4")
    final_path = result_output_path(output_dir, source_name, ".mp4")
    writer = _FrameWriter(raw_path, final_path, actual_process_fps, width, height)
    pending: np.ndarray | None = first_frame

    tracker = BotSort(max_age=30, use_cmc=True)
    track_stats: dict[int, dict] = {}
    source_index = 0
    processed_index = 0
    finished = False

    try:
        while True:
            engine.raise_if_cancelled(job_id)
            if pending is not None:
                frame_bgr = pending
                pending = None
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
            if source_index % sample_every != 0:
                source_index += 1
                continue

            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            detections, _ = engine.infer_image(
                image=image,
                prompts=prompts,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_objects=max_objects,
                include_polygon=False,
                job_id=job_id,
            )

            track_ids = tracker.update(
                boxes=[det.box for det in detections],
                embeddings=[det.embedding for det in detections],
                labels=[det.label for det in detections],
                frame_index=processed_index,
                scores=[det.score for det in detections],
                frame_bgr=frame_bgr,
            )

            source_time_s = source_index / src_fps
            for det, track_id in zip(detections, track_ids):
                det.track_id = track_id
                stat = track_stats.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "label": det.label,
                        "first_seen_s": source_time_s,
                        "last_seen_s": source_time_s,
                        "observations": 0,
                        "max_confidence": 0.0,
                        "best_semantic_similarity": -1.0,
                    },
                )
                stat["label"] = det.label
                stat["last_seen_s"] = source_time_s
                stat["observations"] += 1
                stat["max_confidence"] = max(stat["max_confidence"], float(det.score))
                stat["best_semantic_similarity"] = max(
                    stat["best_semantic_similarity"], float(det.semantic_similarity)
                )

            annotated = annotate_image(image, detections)
            writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))

            processed_index += 1
            source_index += 1
        finished = processed_index > 0
    except InferenceCancelled:
        raise
    finally:
        cap.release()
        if not finished:
            writer.abort()

    if processed_index == 0:
        raise ValueError("No frames were processed from the video")

    produced = writer.finish()
    if produced != final_path:
        _transcode_h264(produced, final_path)
        produced.unlink(missing_ok=True)

    summaries = sorted(track_stats.values(), key=lambda item: item["track_id"])
    for summary in summaries:
        summary["first_seen_s"] = round(float(summary["first_seen_s"]), 3)
        summary["last_seen_s"] = round(float(summary["last_seen_s"]), 3)
        summary["max_confidence"] = round(float(summary["max_confidence"]), 4)
        summary["best_semantic_similarity"] = round(
            float(summary["best_semantic_similarity"]), 4
        )

    return VideoProcessingResult(
        output_path=final_path,
        track_summaries=summaries,
        processed_frames=processed_index,
        source_duration_s=duration_s,
        process_fps=actual_process_fps,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )


def _transcode_h264(input_path: Path, output_path: Path) -> None:
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        shutil.copy2(input_path, output_path)

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "OpenWorld Vision"
    app_version: str = "1.0.0"

    device: str = "auto"
    detector_model: str = "IDEA-Research/grounding-dino-tiny"
    sam_model: str = "facebook/sam-vit-base"
    clip_model: str = "openai/clip-vit-base-patch32"

    max_upload_mb: int = 250
    max_video_seconds: int = 30
    default_video_process_fps: float = 4.0
    max_objects: int = 20

    output_dir: Path = BASE_DIR / "results"
    static_dir: Path = BASE_DIR / "app" / "static"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_prefix="OPENWORLD_",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings

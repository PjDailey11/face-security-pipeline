from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    yolo_weights: Path = Path("weights/yolov8_face.pt")
    torch_device: str = "cuda:0"

    yolo_conf: float = 0.35
    yolo_iou: float = 0.45
    max_faces_per_frame: int = 20

    # Dashboard mode: detection-only, face-only
    detection_only: bool = True
    face_only: bool = True

    # Legacy/optional modules (kept for compatibility but unused in detection_only mode)
    antispoof_weights: Path | None = Path("weights/antispoof.pt")
    chroma_dir: Path = Path("data/chroma_faces")
    cosine_similarity_threshold: float = 0.65
    antispoof_live_prob_min: float = 0.55
    mask_yolo_weights: Path | None = None
    attendance_database_url: str | None = None
    redact_unknown: bool = True
    blur_sigma: float = 25.0

    camera_source: str | int = 0

    skip_model_init: bool = False

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.antispoof_cnn import AntiSpoofCNN
from models.embedding_wrapper import FaceEmbeddingModel
from models.yolo_face import YoloFaceDetector
from utils.alignment import align_face_bgr_to_160, fallback_square_crop, make_face_mesh_static, make_face_mesh_video
from utils.mask_detection import MaskDetector, match_mask_label_for_face
from utils.vector_store import EmbeddingIndex

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    yolo_weights: Path
    antispoof_weights: Path | None
    chroma_dir: Path
    device: str
    detection_only: bool = True
    face_only: bool = True
    yolo_conf: float = 0.35
    yolo_iou: float = 0.45
    cosine_similarity_threshold: float = 0.65
    antispoof_live_prob_min: float = 0.55
    max_faces_per_frame: int = 20
    mask_yolo_weights: Path | None = None


class UnifiedFaceSecurityPipeline:
    """YOLOv8 detection → landmark alignment → embedding → Chroma verification → anti-spoof."""

    def __init__(self, cfg: PipelineConfig, *, video_mesh: bool = False) -> None:
        self.cfg = cfg
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — using CPU.")
            self.device = "cpu"
        else:
            self.device = cfg.device

        self.detector = YoloFaceDetector(
            weights_path=cfg.yolo_weights,
            conf=cfg.yolo_conf,
            iou=cfg.yolo_iou,
            device=self.device,
            face_only=cfg.face_only,
        )
        self.embedder = None if cfg.detection_only else FaceEmbeddingModel(device=self.device)
        self.index = None if cfg.detection_only else EmbeddingIndex(cfg.chroma_dir)

        self._mesh_static = make_face_mesh_static()
        self._mesh_video = make_face_mesh_video() if video_mesh else None
        if self._mesh_static is None:
            logger.warning(
                "MediaPipe FaceMesh unavailable in this environment. Face alignment disabled; using bbox crops."
            )

        self.antispoof: AntiSpoofCNN | None = None
        self._as_tf = transforms.Compose([transforms.ToTensor()])
        if not cfg.detection_only:
            if cfg.antispoof_weights and Path(cfg.antispoof_weights).exists():
                self.antispoof = AntiSpoofCNN().eval().to(self.device)
                ckpt = torch.load(cfg.antispoof_weights, map_location=self.device)
                self.antispoof.load_state_dict(ckpt["model"])
            else:
                logger.warning("Anti-spoof weights missing — liveness scores disabled.")

        self.mask_detector = None if cfg.detection_only else MaskDetector(cfg.mask_yolo_weights, device=self.device)

    def close(self) -> None:
        if getattr(self._mesh_static, "close", None):
            self._mesh_static.close()
        if self._mesh_video is not None and getattr(self._mesh_video, "close", None):
            self._mesh_video.close()

    def _crop_face(self, frame_bgr: np.ndarray, bbox: list[float], margin: float = 0.18) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        mx = w * margin
        my = h * margin
        H, W = frame_bgr.shape[:2]
        xa = int(max(0, x1 - mx))
        ya = int(max(0, y1 - my))
        xb = int(min(W, x2 + mx))
        yb = int(min(H, y2 + my))
        return frame_bgr[ya:yb, xa:xb]

    @torch.inference_mode()
    def _liveness(self, face_bgr_224: np.ndarray) -> float | None:
        if self.antispoof is None:
            return None
        rgb = cv2.cvtColor(face_bgr_224, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = self._as_tf(rgb).unsqueeze(0).to(self.device)
        logits = self.antispoof(x)
        prob = F.softmax(logits, dim=1)[0, 1].detach().cpu().item()
        return float(prob)

    def analyze_frame(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        mesh = self._mesh_video if self._mesh_video is not None else self._mesh_static
        detections = self.detector.predict(frame_bgr)[: self.cfg.max_faces_per_frame]
        if self.cfg.detection_only:
            return [
                {
                    "bbox": d["bbox"],
                    "detection_confidence": d["confidence"],
                    "class_name": "face",
                }
                for d in detections
            ]

        mask_map = self.mask_detector.predict_map(frame_bgr) if self.mask_detector is not None else {}

        faces_out: list[dict[str, Any]] = []
        crops_for_emb: list[np.ndarray] = []
        meta: list[dict[str, Any]] = []

        for det in detections:
            bbox = det["bbox"]
            crop = self._crop_face(frame_bgr, bbox)
            if mesh is None:
                aligned = fallback_square_crop(crop)
            else:
                aligned = align_face_bgr_to_160(crop, mesh)
                if aligned is None:
                    aligned = fallback_square_crop(crop)
            face224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

            live_prob = self._liveness(face224)

            crops_for_emb.append(aligned)
            meta.append(
                {
                    "bbox": bbox,
                    "detection_confidence": det["confidence"],
                    "class_name": det.get("class_name", "face"),
                    "liveness_prob": live_prob,
                    "mask_label": match_mask_label_for_face(bbox, mask_map),
                }
            )

        if not crops_for_emb:
            return []

        embs = self.embedder.embed_bgr_np(crops_for_emb, normalize=True)

        for row, emb in zip(meta, embs, strict=True):
            q = self.index.query(emb, top_k=1)
            identity = q["identity"]
            similarity = q["similarity"]

            is_known = (
                identity is not None
                and similarity is not None
                and similarity >= self.cfg.cosine_similarity_threshold
            )

            if row["liveness_prob"] is not None and row["liveness_prob"] < self.cfg.antispoof_live_prob_min:
                is_known = False

            faces_out.append(
                {
                    **row,
                    "identity": identity if is_known else "unknown",
                    "verification_similarity": similarity,
                    "known": bool(is_known),
                    "redact": not bool(is_known),
                }
            )

        return faces_out

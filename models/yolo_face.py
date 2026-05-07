from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class YoloFaceDetector:
    """Thin wrapper around Ultralytics YOLO for face bounding boxes."""

    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.35,
        iou: float = 0.45,
        device: str | None = None,
        face_only: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self._weights = Path(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.face_only = face_only
        self._model = YOLO(str(self._weights))

    def predict(self, bgr_image: np.ndarray) -> list[dict[str, Any]]:
        """Return detections with xyxy boxes in pixel coords (float)."""
        results = self._model.predict(
            source=bgr_image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        out: list[dict[str, Any]] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out if not self.face_only else self._fallback_faces(bgr_image)
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.zeros(len(scores))
        names = r.names or {0: "face"}

        # If this is not a face model (e.g. COCO), the names won't include 'face' and detections will be 'person/bed/clock'.
        # In face_only mode we fall back to a local face detector rather than returning non-face classes.
        names_lower = {str(v).lower() for v in names.values()} if isinstance(names, dict) else set()
        if self.face_only and ("face" not in names_lower) and ("yolov8_face" not in str(self._weights).lower()):
            return self._fallback_faces(bgr_image)

        for i in range(len(scores)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            class_name = names.get(int(cls[i]), "face")
            if self.face_only and str(class_name).lower() != "face":
                continue
            out.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(scores[i]),
                    "class_id": 0,
                    "class_name": "face",
                }
            )
        return out

    def _fallback_faces(self, bgr_image: np.ndarray) -> list[dict[str, Any]]:
        from utils.face_fallback import detect_faces_haar

        return detect_faces_haar(bgr_image, min_conf=self.conf)

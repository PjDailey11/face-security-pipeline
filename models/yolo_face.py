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
    ) -> None:
        from ultralytics import YOLO

        self._weights = Path(weights_path)
        self.conf = conf
        self.iou = iou
        self.device = device
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
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.zeros(len(scores))
        names = r.names or {0: "face"}
        for i in range(len(scores)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            out.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(scores[i]),
                    "class_id": int(cls[i]),
                    "class_name": names.get(int(cls[i]), "face"),
                }
            )
        return out

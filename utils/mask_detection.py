from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MaskDetector:
    """
    Optional secondary Ultralytics model for mask vs no-mask overlay.

    If weights are omitted, `predict` returns empty dict (no overlay).
    """

    def __init__(self, weights_path: str | Path | None, device: str | None = None) -> None:
        self._weights = Path(weights_path) if weights_path else None
        self.device = device
        self._model = None
        if self._weights and self._weights.exists():
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights))

    def predict_map(self, bgr: np.ndarray) -> dict[tuple[int, int, int, int], str]:
        """Map tight bbox tuples (x1,y1,x2,y2 ints) to coarse label strings."""
        if self._model is None:
            return {}
        res = self._model.predict(source=bgr, verbose=False, device=self.device)
        out: dict[tuple[int, int, int, int], str] = {}
        if not res or res[0].boxes is None:
            return out
        r = res[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        names: dict[int, Any] = r.names or {}
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[i].tolist()]
            cid = int(cls[i])
            out[(x1, y1, x2, y2)] = str(names.get(cid, "mask_class"))
        return out


def match_mask_label_for_face(
    face_bbox: list[float],
    mask_boxes: dict[tuple[int, int, int, int], str],
    iou_min: float = 0.05,
) -> str | None:
    """Best-effort association by IoU overlap between face box and mask-model boxes."""
    fx1, fy1, fx2, fy2 = face_bbox

    def iou(a: tuple[int, int, int, int]) -> float:
        x1 = max(fx1, a[0])
        y1 = max(fy1, a[1])
        x2 = min(fx2, a[2])
        y2 = min(fy2, a[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(1.0, (fx2 - fx1) * (fy2 - fy1))
        area_b = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    best = None
    best_score = 0.0
    for box, lab in mask_boxes.items():
        score = iou(box)
        if score > best_score:
            best_score = score
            best = lab
    return best if best_score >= iou_min else None

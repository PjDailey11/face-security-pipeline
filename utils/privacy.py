from __future__ import annotations

import cv2
import numpy as np


def blur_bbox_region(image_bgr: np.ndarray, bbox_xyxy: list[float], sigma: float = 25.0) -> None:
    """In-place Gaussian blur on axis-aligned rectangle."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = image_bgr[y1:y2, x1:x2]
    k = max(3, int(sigma) | 1)
    blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=sigma, sigmaY=sigma)
    image_bgr[y1:y2, x1:x2] = blurred


def redact_unknown_faces(
    frame_bgr: np.ndarray,
    faces: list[dict],
    *,
    blur_sigma: float,
) -> np.ndarray:
    """Return a copy with unknown/low-confidence identities blurred."""
    out = frame_bgr.copy()
    for f in faces:
        if f.get("identity") == "unknown" or f.get("redact"):
            blur_bbox_region(out, f["bbox"], sigma=blur_sigma)
    return out

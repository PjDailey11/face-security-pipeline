from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def iter_camera_frames(source: str | int = 0) -> Iterator[tuple[bool, np.ndarray]]:
    """Yield (ok, frame_bgr) from webcam or video path."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield ok, frame
    finally:
        cap.release()

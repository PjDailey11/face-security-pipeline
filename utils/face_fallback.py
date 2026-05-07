from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_faces_haar(bgr_image: np.ndarray, *, min_conf: float = 0.35) -> list[dict[str, Any]]:
    """
    Offline/local face detection fallback using OpenCV Haar cascades.

    This exists so the app can still detect faces when the YOLO weights are COCO
    (person/bed/clock etc.) or when a face-specific YOLO checkpoint isn't available.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    out: list[dict[str, Any]] = []
    # Haar doesn't output confidence; we provide a heuristic constant for UI display.
    conf = float(min_conf)
    for (x, y, w, h) in rects:
        out.append(
            {
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "confidence": conf,
                "class_id": 0,
                "class_name": "face",
            }
        )
    return out


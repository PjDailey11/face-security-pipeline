from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _nms_rects(rects: list[tuple[int, int, int, int]], iou_thresh: float = 0.35) -> list[tuple[int, int, int, int]]:
    if not rects:
        return []
    boxes = np.array(rects, dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = []
        for j in rest:
            inter_x1 = max(boxes[i, 0], boxes[j, 0])
            inter_y1 = max(boxes[i, 1], boxes[j, 1])
            inter_x2 = min(boxes[i, 2], boxes[j, 2])
            inter_y2 = min(boxes[i, 3], boxes[j, 3])
            iw = max(0.0, inter_x2 - inter_x1)
            ih = max(0.0, inter_y2 - inter_y1)
            inter = iw * ih
            ua = areas[i] + areas[int(j)] - inter + 1e-6
            ious.append(inter / ua)
        ious = np.array(ious, dtype=np.float32)
        order = rest[ious < iou_thresh]
    out = [tuple(int(v) for v in boxes[i]) for i in keep]
    return out


def detect_faces_haar(bgr_image: np.ndarray, *, min_conf: float = 0.35) -> list[dict[str, Any]]:
    """
    Offline/local face detection using OpenCV Haar cascades.

    Tuned to reduce false positives on textured backgrounds (pillows, patterns).
    """
    h, w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # More conservative size gates to reduce background false positives.
    min_side = max(72, int(min(w, h) * 0.12))
    max_side = int(min(w, h) * 0.62)

    rects_xywh = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.06,
        minNeighbors=12,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_side, min_side),
        maxSize=(max_side, max_side),
    )

    candidates: list[tuple[int, int, int, int]] = []
    for (x, y, rw, rh) in rects_xywh:
        ar = rw / float(rh)
        if ar < 0.72 or ar > 1.38:
            continue
        area = rw * rh
        if area < (min_side * min_side * 0.85):
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + rw), int(y + rh)
        candidates.append((x1, y1, x2, y2))

    merged = _nms_rects(candidates, iou_thresh=0.32)

    # Drop tiny detections vs. largest (often background texture).
    if merged:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in merged]
        amax = max(areas)
        merged = [b for b, a in zip(merged, areas, strict=True) if a >= amax * 0.35]

    out: list[dict[str, Any]] = []
    conf = float(min_conf)
    for x1, y1, x2, y2 in merged:
        out.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "class_id": 0,
                "class_name": "face",
            }
        )
    return out

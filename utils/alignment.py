from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


def _eye_centers_from_landmarks(lms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate eye centers from MediaPipe FaceMesh landmark subset."""
    right_idx = [33, 160, 158, 133, 153]
    left_idx = [362, 385, 387, 263, 373]
    rc = lms[right_idx].mean(axis=0)
    lc = lms[left_idx].mean(axis=0)
    return rc.astype(np.float32), lc.astype(np.float32)


def align_face_bgr_to_160(face_bgr: np.ndarray, mesh: Any) -> np.ndarray | None:
    """
    Similarity-align face crop to 160x160 RGB-normalized output for embedding nets.

    Returns BGR 160x160 aligned crop or None if landmarks missing.
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lm.landmark], dtype=np.float32)
    reye, leye = _eye_centers_from_landmarks(pts)

    desired_dist = 70.0
    desired_mid_x = 80.0
    desired_mid_y = 60.0

    dx = float(leye[0] - reye[0])
    dy = float(leye[1] - reye[1])
    dist = float(np.hypot(dx, dy)) + 1e-6
    scale = desired_dist / dist

    angle = np.degrees(np.arctan2(dy, dx))

    mid_x = float(reye[0] + leye[0]) / 2.0
    mid_y = float(reye[1] + leye[1]) / 2.0

    M = cv2.getRotationMatrix2D((mid_x, mid_y), angle, scale)
    M[0, 2] += desired_mid_x - mid_x
    M[1, 2] += desired_mid_y - mid_y

    aligned = cv2.warpAffine(face_bgr, M, (160, 160), flags=cv2.INTER_LINEAR)
    return aligned


def fallback_square_crop(face_bgr: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """BBox crop fallback — resize to 160 square."""
    h, w = face_bgr.shape[:2]
    side = min(h, w)
    cx, cy = w / 2, h / 2
    half = side / 2 * (1.0 + margin)
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))
    crop = face_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)


def make_face_mesh_static() -> Any:
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def make_face_mesh_video() -> Any:
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

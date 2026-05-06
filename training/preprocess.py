"""
Offline preprocessing for robustness to lighting, viewpoint, and partial occlusion.

Uses Albumentations for reproducible augmentation pipelines suitable for detector pre-training
or anti-spoof data synthesis from captured clips.

Privacy: operates on local tensors/images only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np


def build_detector_train_augmentation(img_size: int = 640) -> A.Compose:
    """Strong-but-safe augmentations for face detection (applied to image + bbox)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=12,
                sat_shift_limit=35,
                val_shift_limit=35,
                p=0.5,
            ),
            A.MotionBlur(blur_limit=(3, 7), p=0.12),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.25), p=0.15),
            A.CoarseDropout(
                max_holes=3,
                max_height=int(0.12 * img_size),
                max_width=int(0.12 * img_size),
                min_holes=1,
                fill_value=0,
                p=0.25,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=12,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def build_antispoof_augmentation(img_size: int = 224) -> A.Compose:
    """Micro-texture-sensitive augmentations for spoof/live classifier."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.45),
            A.GaussNoise(var_limit=(5.0, 45.0), p=0.25),
            A.MotionBlur(p=0.12),
            A.Resize(img_size, img_size),
        ]
    )


def load_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def apply_det_aug(
    image_bgr: np.ndarray,
    bboxes_yolo: list[list[float]],
    class_labels: list[str],
    img_size: int = 640,
) -> tuple[np.ndarray, list[list[float]], list[str]]:
    tfm = build_detector_train_augmentation(img_size)
    aug = tfm(image=image_bgr, bboxes=bboxes_yolo, class_labels=class_labels)
    return aug["image"], aug["bboxes"], aug["class_labels"]


def summarize_dataset_folder(images_dir: Path) -> dict[str, Any]:
    """Lightweight stats for QA (mean luminance, resolution histogram omitted for brevity)."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    return {"count": len(paths), "root": str(images_dir.resolve())}

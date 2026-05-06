"""
Train YOLOv8 face detector with transfer learning from COCO weights.

Hyperparameters are documented in configs/hyperparameters.yaml and mirrored below as argparse defaults.

Usage:
  python training/train_yolov8_face.py --data configs/dataset_face.yaml --weights yolov8n.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_hp(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    return doc.get("yolov8_face_detection", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("configs/dataset_face.yaml"))
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--hyp-yaml", type=Path, default=Path("configs/hyperparameters.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers (backbone)")
    parser.add_argument("--project", type=str, default="runs/face_yolov8")
    parser.add_argument("--name", type=str, default="exp")
    args = parser.parse_args()

    hp = load_hp(args.hyp_yaml)
    epochs = args.epochs if args.epochs is not None else int(hp.get("epochs", 100))
    batch = args.batch if args.batch is not None else int(hp.get("batch_size", 16))
    imgsz = args.imgsz if args.imgsz is not None else int(hp.get("imgsz", 640))
    freeze = args.freeze if args.freeze is not None else int(hp.get("freeze_backbone_layers", 10))

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=float(hp.get("lr0", 0.001)),
        lrf=float(hp.get("lrf", 0.01)),
        momentum=float(hp.get("momentum", 0.937)),
        weight_decay=float(hp.get("weight_decay", 0.0005)),
        warmup_epochs=float(hp.get("warmup_epochs", 3.0)),
        warmup_momentum=float(hp.get("warmup_momentum", 0.8)),
        warmup_bias_lr=float(hp.get("warmup_bias_lr", 0.1)),
        mosaic=float(hp.get("mosaic", 1.0)),
        mixup=float(hp.get("mixup", 0.0)),
        copy_paste=float(hp.get("copy_paste", 0.0)),
        hsv_h=float(hp.get("hsv_h", 0.015)),
        hsv_s=float(hp.get("hsv_s", 0.7)),
        hsv_v=float(hp.get("hsv_v", 0.4)),
        degrees=float(hp.get("degrees", 10.0)),
        translate=float(hp.get("translate", 0.1)),
        scale=float(hp.get("scale", 0.5)),
        shear=float(hp.get("shear", 2.0)),
        perspective=float(hp.get("perspective", 0.0001)),
        flipud=float(hp.get("flipud", 0.0)),
        fliplr=float(hp.get("fliplr", 0.5)),
        close_mosaic=int(hp.get("close_mosaic", 10)),
        patience=int(hp.get("patience", 25)),
        workers=int(hp.get("workers", 8)),
        optimizer=str(hp.get("optimizer", "AdamW")),
        amp=bool(hp.get("amp", True)),
        freeze=freeze,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

"""
Dataset ingestion helpers for diverse facial imagery.

Supports folder layouts compatible with Ultralytics YOLO (`datasets/faces/`).
Optional normalization into train/val splits from a flat directory of images + JSON sidecars.

Privacy: this module only organizes local files; it does not upload data.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def stratified_split_flat_images(
    images_dir: Path,
    output_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    copy: bool = True,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> None:
    """
    Split flat `images_dir` into:
      output_root/images/{train,val}/...
    Labels are NOT generated (use an annotator or pseudo-label step separately).
    """
    rng = random.Random(seed)
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in extensions]
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio)) if files else 0
    val_set = set(files[:n_val])
    for p in files:
        split = "val" if p in val_set else "train"
        _link_or_copy(p, output_root / "images" / split / p.name, copy=copy)


def write_manifest(output_root: Path, extra: dict | None = None) -> None:
    manifest = {"root": str(output_root.resolve()), **(extra or {})}
    out = output_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest/split face image folders for YOLO training prep.")
    parser.add_argument("--flat-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", help="Try hardlink/symlink behavior via hardlink first.")
    args = parser.parse_args()
    stratified_split_flat_images(
        args.flat_dir,
        args.out,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy=not args.symlink,
    )
    write_manifest(args.out, {"val_ratio": args.val_ratio})
    print(f"Wrote split under {args.out}")


if __name__ == "__main__":
    main()

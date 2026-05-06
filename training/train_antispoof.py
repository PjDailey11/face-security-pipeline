"""Train anti-spoofing CNN (see module docstring in repo README patterns)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.antispoof_cnn import AntiSpoofCNN


class FolderBinaryDataset(Dataset):
    def __init__(self, root: Path, img_size: int = 224) -> None:
        self.samples: list[tuple[Path, int]] = []
        live = root / "live"
        spoof = root / "spoof"
        for p in live.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                self.samples.append((p, 1))
        for p in spoof.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                self.samples.append((p, 0))
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(label, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/antispoof"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("weights/antispoof.pt"))
    args = parser.parse_args()

    ds = FolderBinaryDataset(args.data)
    if len(ds) < 8:
        raise SystemExit(
            f"Need more samples under {args.data}/live and {args.data}/spoof (found {len(ds)})."
        )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device(args.device)
    model = AntiSpoofCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.size(0)
        sched.step()
        print(f"epoch {epoch+1} loss={total / len(ds):.4f}")

    torch.save({"model": model.state_dict()}, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

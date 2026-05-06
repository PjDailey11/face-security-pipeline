from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MicroTextureStem(nn.Module):
    """Emphasize high-frequency texture prior to standard conv trunk."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.highpass = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False, groups=in_ch)
        with torch.no_grad():
            k = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            k = k.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)
            self.highpass.weight.copy_(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hp = self.highpass(x)
        return torch.cat([x, hp], dim=1)


class AntiSpoofCNN(nn.Module):
    """
    Lightweight CNN for live vs print/screen spoofing.
    Expects 224x224 RGB input; outputs logits for [spoof, live].
    """

    def __init__(self, dropout: float = 0.35) -> None:
        super().__init__()
        stem_in = 6  # RGB + high-pass RGB
        self.stem = _MicroTextureStem(3)
        self.conv1 = nn.Conv2d(stem_in, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

# -*- coding: utf-8 -*-
"""src/model.py — LiteTBNet architecture."""
import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, C: int):
        super().__init__()
        t = int(abs((math.log2(C) + 1) / 2))
        k = t if t % 2 else t + 1
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        w = self.gap(x).squeeze(-1).transpose(-1, -2)   # (B,1,C)
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1) # (B,C,1,1)
        return x * self.sig(w)


class IRB(nn.Module):
    """Inverted Residual Block with optional ECA on expanded channels."""
    def __init__(self, cin: int, c: int, t: int, s: int, use_eca: bool = True):
        super().__init__()
        hidden = cin * t
        self.use_res = (s == 1 and cin == c)
        layers = [
            # expand
            nn.Conv2d(cin, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            # depthwise
            nn.Conv2d(hidden, hidden, 3, stride=s, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ]
        if use_eca:
            layers.append(ECA(hidden))
        layers += [
            # project
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.BatchNorm2d(c),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


def _conv_bn_silu(cin, cout, k, s, p):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class LiteTBNet(nn.Module):
    def __init__(self, use_eca: bool = True):
        super().__init__()
        e = use_eca
        # Stem: 224 -> 112
        self.stem = nn.Sequential(
            _conv_bn_silu(3,  16, 3, 2, 1),
            _conv_bn_silu(16, 16, 3, 1, 1),
        )
        # Stage1: 112 -> 56
        self.stage1 = nn.Sequential(
            IRB(16, 24, t=2, s=2, use_eca=e),
            IRB(24, 24, t=2, s=1, use_eca=e),
        )
        # Stage2: 56 -> 28
        self.stage2 = nn.Sequential(
            IRB(24, 40, t=3, s=2, use_eca=e),
            IRB(40, 40, t=3, s=1, use_eca=e),
            IRB(40, 40, t=3, s=1, use_eca=e),
        )
        # Stage3: 28 -> 14
        self.stage3 = nn.Sequential(
            IRB(40, 80, t=4, s=2, use_eca=e),
            IRB(80, 80, t=4, s=1, use_eca=e),
            IRB(80, 80, t=4, s=1, use_eca=e),
        )
        # Stage4: 14 -> 14
        self.stage4 = nn.Sequential(
            IRB(80, 112, t=4, s=1, use_eca=e),
            _conv_bn_silu(112, 160, 1, 1, 0),
        )
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(160, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

    def feature_maps(self, x: torch.Tensor):
        """Return (stem_out, s1, s2, s3, s4) for spatial size checks."""
        s  = self.stem(x)
        s1 = self.stage1(s)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s, s1, s2, s3, s4

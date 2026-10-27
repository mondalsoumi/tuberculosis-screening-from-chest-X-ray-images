# -*- coding: utf-8 -*-
"""src/gpu_data.py — device-agnostic GPUBatcher (no DataLoader / num_workers)."""
import math
import numpy as np
import torch
import torch.nn.functional as F


class GPUBatcher:
    """
    Keeps the whole uint8 array on `device`.
    Yields (x_float, y_float) where:
        x_float : (B, 3, 224, 224)  float32  in [0, 1]
        y_float : (B, 1)            float32
    Train augmentation (per-sample random):
        rotation     ±10 deg
        translation  ±5 % of image size  (±0.1 in affine_grid normalised units)
        brightness   factor in [0.9, 1.1]
        contrast     factor in [0.9, 1.1]
    No flips.  Val / test: no augmentation.
    Implemented via F.affine_grid + F.grid_sample.
    """

    def __init__(self, X_uint8: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool, augment: bool,
                 device, seed: int = 42,
                 drop_last: bool = False, mask_border: int = 0):
        self.device      = torch.device(device)
        self.X           = torch.from_numpy(X_uint8).to(self.device)   # (N,H,W) uint8
        self.y           = torch.from_numpy(np.asarray(y, dtype=np.float32)
                                            ).unsqueeze(1).to(self.device)  # (N,1)
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.augment     = augment
        self.drop_last   = drop_last
        self.mask_border = mask_border
        self.n           = X_uint8.shape[0]
        self._gen        = torch.Generator(device='cpu')   # cpu generator is portable
        self._gen.manual_seed(seed)

    # ── iteration ────────────────────────────────────────────────────────────
    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, generator=self._gen)
        else:
            idx = torch.arange(self.n)
        idx = idx.to(self.device)

        total = (self.n // self.batch_size) * self.batch_size if self.drop_last else self.n
        for start in range(0, total, self.batch_size):
            b_idx = idx[start: start + self.batch_size]
            x = self.X[b_idx].float() / 255.0          # (B, H, W)  float [0,1]
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)   # (B, 3, H, W)
            if self.augment:
                x = self._augment(x)
            if self.mask_border > 0:
                mb = self.mask_border
                x[:, :, :mb, :]  = 0.0
                x[:, :, -mb:, :] = 0.0
                x[:, :, :, :mb]  = 0.0
                x[:, :, :, -mb:] = 0.0
            yield x, self.y[b_idx]

    # ── augmentation ─────────────────────────────────────────────────────────
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        dev = x.device

        # Per-sample random scalars on CPU then move
        angle = (torch.rand(B, generator=self._gen) * 2 - 1) * 10.0        # ±10 deg
        tx    = (torch.rand(B, generator=self._gen) * 2 - 1) * 0.1          # ±0.1 (=±5%)
        ty    = (torch.rand(B, generator=self._gen) * 2 - 1) * 0.1
        brt   = (torch.rand(B, generator=self._gen) * 0.2 + 0.9)            # [0.9,1.1]
        con   = (torch.rand(B, generator=self._gen) * 0.2 + 0.9)

        angle = angle.to(dev); tx = tx.to(dev); ty = ty.to(dev)
        brt   = brt.to(dev);   con = con.to(dev)

        # Affine matrix (B,2,3): rotation + translation
        rad   = angle * (math.pi / 180.0)
        cos_a = torch.cos(rad)
        sin_a = torch.sin(rad)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] =  cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] =  tx
        theta[:, 1, 0] =  sin_a
        theta[:, 1, 1] =  cos_a
        theta[:, 1, 2] =  ty

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear',
                          padding_mode='reflection', align_corners=False)

        # Brightness × contrast, then clamp
        x = x * brt.view(B, 1, 1, 1)
        x = (x - 0.5) * con.view(B, 1, 1, 1) + 0.5
        return x.clamp(0.0, 1.0)

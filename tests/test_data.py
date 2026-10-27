# -*- coding: utf-8 -*-
"""tests/test_data.py — CPU tests for GPUBatcher shape, dtype, range, augmentation."""
import sys, os
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gpu_data import GPUBatcher

IMG_SIZE   = 224
BATCH_SIZE = 4
SEED       = 42


def _make_batcher(augment: bool) -> GPUBatcher:
    rng = np.random.default_rng(SEED)
    X = rng.integers(0, 256, size=(8, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return GPUBatcher(X, y, batch_size=BATCH_SIZE,
                      shuffle=False, augment=augment,
                      device="cpu", seed=SEED)


def _first_batch(batcher: GPUBatcher):
    return next(iter(batcher))


# ── shape & dtype tests ────────────────────────────────────────────────────────

def test_x_shape():
    x, y = _first_batch(_make_batcher(augment=False))
    assert x.shape == (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), f"bad x shape {x.shape}"


def test_y_shape():
    x, y = _first_batch(_make_batcher(augment=False))
    assert y.shape == (BATCH_SIZE, 1), f"bad y shape {y.shape}"


def test_x_dtype():
    x, _ = _first_batch(_make_batcher(augment=False))
    assert x.dtype == torch.float32, f"bad x dtype {x.dtype}"


def test_y_dtype():
    _, y = _first_batch(_make_batcher(augment=False))
    assert y.dtype == torch.float32, f"bad y dtype {y.dtype}"


def test_x_range():
    x, _ = _first_batch(_make_batcher(augment=False))
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, \
        f"x out of [0,1]: min={x.min():.4f} max={x.max():.4f}"


# ── augmentation tests ─────────────────────────────────────────────────────────

def test_augment_changes_pixels():
    """Train-mode augmentation must change pixels."""
    no_aug_x, _ = _first_batch(_make_batcher(augment=False))
    aug_x,    _ = _first_batch(_make_batcher(augment=True))
    assert not torch.allclose(no_aug_x, aug_x, atol=1e-4), \
        "Augmented batch identical to original — augmentation has no effect"


def test_no_augment_stable():
    """Eval-mode: two identical batchers must yield identical tensors."""
    x1, _ = _first_batch(_make_batcher(augment=False))
    x2, _ = _first_batch(_make_batcher(augment=False))
    assert torch.allclose(x1, x2), "Non-augmented batches differ"


def test_augmented_range():
    """Augmented pixels must still be in [0,1]."""
    x, _ = _first_batch(_make_batcher(augment=True))
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, \
        f"Augmented x out of [0,1]: min={x.min():.4f} max={x.max():.4f}"


# ── aug_check.png ─────────────────────────────────────────────────────────────

def test_save_aug_check_png():
    """Save outputs/aug_check.png: one row of 4 images (orig / aug / orig / aug)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    X = rng.integers(0, 256, size=(8, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    no_aug = GPUBatcher(X, y, 4, False, False, "cpu", SEED)
    aug    = GPUBatcher(X, y, 4, False, True,  "cpu", SEED)

    x_orig, _ = _first_batch(no_aug)   # (4,3,224,224)
    x_aug,  _ = _first_batch(aug)

    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Original", "Augmented", "Original", "Augmented"]
    imgs   = [x_orig[0], x_aug[0], x_orig[1], x_aug[1]]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/aug_check.png", dpi=100)
    plt.close()
    assert os.path.exists("outputs/aug_check.png")

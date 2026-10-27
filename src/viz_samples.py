# -*- coding: utf-8 -*-
"""src/viz_samples.py — outputs/samples.png: 3 rows x 8 grid.
Row 0: real TB train images (first 8).
Row 1: SMOTE-synthetic TB images (indices >= original train count, first 8 TB).
Row 2: ADASYN-synthetic TB images (same rule).
"""
import os, sys, argparse
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_tb(X: np.ndarray, y: np.ndarray, n_orig: int, synthetic: bool, k: int = 8):
    """Return up to k images. synthetic=True → indices >= n_orig with label 1."""
    if synthetic:
        mask = (np.arange(len(y)) >= n_orig) & (y == 1)
    else:
        mask = y == 1
    imgs = X[mask][:k]
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    args = ap.parse_args()

    # Load originals
    X_none = np.load(os.path.join(args.cache, "train_none_X.npy"))
    y_none = np.load(os.path.join(args.cache, "train_none_y.npy"))
    n_orig = int((y_none == 1).sum())   # how many real TB images

    # Smote / adasyn arrays
    X_smote = np.load(os.path.join(args.cache, "train_smote_X.npy"))
    y_smote = np.load(os.path.join(args.cache, "train_smote_y.npy"))

    X_ada   = np.load(os.path.join(args.cache, "train_adasyn_X.npy"))
    y_ada   = np.load(os.path.join(args.cache, "train_adasyn_y.npy"))

    rows = [
        ("Real TB (train)",       pick_tb(X_none,  y_none,  n_orig, synthetic=False)),
        ("SMOTE-synthetic TB",    pick_tb(X_smote, y_smote, len(X_none), synthetic=True)),
        ("ADASYN-synthetic TB",   pick_tb(X_ada,   y_ada,   len(X_none), synthetic=True)),
    ]

    COLS = 8
    fig, axes = plt.subplots(3, COLS, figsize=(COLS * 2, 7))
    fig.suptitle("TB image samples", fontsize=13, y=1.01)

    for r, (title, imgs) in enumerate(rows):
        for c in range(COLS):
            ax = axes[r, c]
            if c < len(imgs):
                ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=255)
            else:
                ax.set_facecolor("#111")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(title, fontsize=9, labelpad=4)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/samples.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved outputs/samples.png")


if __name__ == "__main__":
    main()

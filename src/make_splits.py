# -*- coding: utf-8 -*-
"""
src/make_splits.py  –  LiteTBNet Stratified Dataset Splits
===========================================================
Creates a reproducible train/val/test split with EXACT target counts:

  Normal:       train=2531  val=698   test=271
  Tuberculosis: train=502   val=134   test=64

Before splitting:
  1. Removes exact MD5 duplicates (keeps one representative per group).
  2. Logs removed files to logs/dedup_removed.log.
  3. If dedup changes class totals, adjusts train counts only and reports.

Outputs:
  outputs/splits.csv  (path, label, split)

Assertions:
  - No path appears in more than one split.
  - No MD5 hash appears in more than one split.

Prints a final count table.
"""

import os
import sys

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import hashlib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_global_seed, print_gpu_info, SEED, DATA_ROOT, OUTPUTS_DIR, LOGS_DIR, get_class_dirs

# ── constants ────────────────────────────────────────────────────────────────
# Target counts (when no dedup changes occur)
TARGET = {
    "Normal":       {"train": 2531, "val": 698,  "test": 271},
    "Tuberculosis": {"train": 502,  "val": 134,  "test": 64},
}
LABEL_MAP = {"Normal": 0, "Tuberculosis": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_logging(logs_dir: str) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "dedup_removed.log")
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger("dedup")
    # also print to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(f"# Dedup log — {datetime.now().isoformat()}")
    logger.info(f"# Seed = {SEED}")
    logger.info("")
    return logger


def collect_images(class_dirs: dict) -> pd.DataFrame:
    """Collect all image paths with md5 per class."""
    rows = []
    for cls, dpath in class_dirs.items():
        for f in sorted(Path(dpath).iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "path" : str(f),
                    "label": LABEL_MAP[cls],
                    "class": cls,
                })
    return pd.DataFrame(rows)


def compute_md5s(df: pd.DataFrame) -> pd.DataFrame:
    """Add md5 column to dataframe."""
    from tqdm import tqdm
    md5s = []
    for path in tqdm(df["path"], desc="  Computing MD5s", unit="file"):
        try:
            md5s.append(md5_file(path))
        except Exception:
            md5s.append(None)
    df = df.copy()
    df["md5"] = md5s
    return df


def remove_exact_duplicates(df: pd.DataFrame, logger: logging.Logger):
    """
    Remove exact MD5 duplicates, keeping the first occurrence (sorted path).
    Returns (deduped_df, removed_df).
    """
    before = len(df)
    # Sort for determinism: keep lexicographically first path per md5
    df_sorted = df.sort_values("path").copy()
    removed_mask = df_sorted.duplicated(subset=["md5"], keep="first")
    removed = df_sorted[removed_mask].copy()
    deduped = df_sorted[~removed_mask].copy()

    if len(removed) == 0:
        logger.info("No exact duplicates found. Nothing removed.")
    else:
        logger.info(f"Removed {len(removed)} exact duplicate(s) (MD5 collision):")
        for _, row in removed.iterrows():
            logger.info(f"  REMOVED  {row['path']}  (class={row['class']}, md5={row['md5']})")

    logger.info(f"\nBefore dedup: {before}  |  After dedup: {len(deduped)}  |  Removed: {len(removed)}")
    return deduped, removed


# ════════════════════════════════════════════════════════════════════════════
# Splitting logic
# ════════════════════════════════════════════════════════════════════════════

def compute_adjusted_targets(class_counts: dict) -> dict:
    """
    Given actual counts after dedup, compute train/val/test split counts.
    Val and test are kept at their target sizes (or max available).
    Train gets the remainder.
    Logs a warning if adjustment was needed.
    """
    adjusted = {}
    changed = False
    for cls, total in class_counts.items():
        tgt = TARGET[cls]
        test_n = min(tgt["test"], total)
        val_n  = min(tgt["val"],  total - test_n)
        train_n = total - test_n - val_n

        if train_n != tgt["train"] or val_n != tgt["val"] or test_n != tgt["test"]:
            changed = True
            print(f"  ⚠️  {cls}: dedup changed totals. "
                  f"Original target train={tgt['train']}, actual train={train_n}.")

        adjusted[cls] = {"train": train_n, "val": val_n, "test": test_n}
    return adjusted, changed


def make_split(df_cls: pd.DataFrame, counts: dict, seed: int) -> pd.DataFrame:
    """
    Split a single-class dataframe into train/val/test using the given counts.
    Returns df with added 'split' column.
    """
    n_total = len(df_cls)
    n_test  = counts["test"]
    n_val   = counts["val"]
    n_train = counts["train"]

    assert n_train + n_val + n_test == n_total, (
        f"Count mismatch: {n_train}+{n_val}+{n_test} = {n_train+n_val+n_test} ≠ {n_total}"
    )

    # First split off test
    idx_all = df_cls.index.tolist()
    idx_trainval, idx_test = train_test_split(
        idx_all,
        test_size=n_test,
        random_state=seed,
        shuffle=True,
    )
    # Then split val from trainval
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=n_val,
        random_state=seed,
        shuffle=True,
    )

    df_out = df_cls.copy()
    df_out["split"] = "train"
    df_out.loc[idx_val,  "split"] = "val"
    df_out.loc[idx_test, "split"] = "test"
    return df_out


# ════════════════════════════════════════════════════════════════════════════
# Assertions
# ════════════════════════════════════════════════════════════════════════════

def assert_no_leakage(splits_df: pd.DataFrame):
    """Assert no path or md5 appears in more than one split."""
    for col in ["path", "md5"]:
        grp = splits_df.groupby(col)["split"].nunique()
        leakers = grp[grp > 1]
        assert len(leakers) == 0, (
            f"DATA LEAKAGE DETECTED: {len(leakers)} {col}(s) appear in multiple splits!\n"
            f"{leakers}"
        )
    print("  ✅ Leakage check passed: no path or md5 appears in more than one split.")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    set_global_seed(SEED)
    print_gpu_info()

    class_dirs = get_class_dirs()
    if not class_dirs:
        print("\n❌  ERROR: No class directories found in data/")
        print("   Expected: data/Normal/ and data/Tuberculosis/")
        print("   Please download the dataset first (see README).")
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    logger = setup_logging(LOGS_DIR)

    # ── Collect ─────────────────────────────────────────────────────────────
    print("\n[1/5] Collecting image files …")
    df = collect_images(class_dirs)
    print(f"  Total collected: {len(df)}")
    for cls in ["Normal", "Tuberculosis"]:
        print(f"  {cls}: {(df['class']==cls).sum()} files")

    # ── Compute MD5s ─────────────────────────────────────────────────────────
    print("\n[2/5] Computing MD5 hashes …")
    df = compute_md5s(df)

    # ── Dedup ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Removing exact duplicates …")
    df_deduped, df_removed = remove_exact_duplicates(df, logger)
    print(f"  Files after dedup: {len(df_deduped)}")
    for cls in ["Normal", "Tuberculosis"]:
        n = (df_deduped["class"] == cls).sum()
        r = (df_removed["class"] == cls).sum()
        print(f"  {cls}: {n} kept  ({r} removed)")

    # ── Compute split counts ──────────────────────────────────────────────────
    print("\n[4/5] Computing split sizes …")
    class_counts = {cls: (df_deduped["class"] == cls).sum()
                    for cls in ["Normal", "Tuberculosis"]}
    adjusted_targets, changed = compute_adjusted_targets(class_counts)
    if not changed:
        print("  Exact target counts will be used (no adjustment needed).")
    print(f"\n  {'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*47}")
    for cls in ["Normal", "Tuberculosis"]:
        t = adjusted_targets[cls]
        print(f"  {cls:<15} {t['train']:>8} {t['val']:>8} {t['test']:>8} {sum(t.values()):>8}")

    # ── Split per class ───────────────────────────────────────────────────────
    print("\n[5/5] Creating stratified splits …")
    split_dfs = []
    for cls in ["Normal", "Tuberculosis"]:
        df_cls = df_deduped[df_deduped["class"] == cls].reset_index(drop=True)
        counts = adjusted_targets[cls]
        df_split = make_split(df_cls, counts, seed=SEED)
        split_dfs.append(df_split)

    splits_df = pd.concat(split_dfs, ignore_index=True)

    # ── Assertions ─────────────────────────────────────────────────────────
    assert_no_leakage(splits_df)

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_csv = os.path.join(OUTPUTS_DIR, "splits.csv")
    splits_df[["path", "label", "split", "md5"]].to_csv(out_csv, index=False)
    print(f"  ✅ Saved splits.csv → {out_csv}")

    # ── Final count table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SPLIT COUNT TABLE")
    print("=" * 60)
    ct = splits_df.groupby(["class", "split"]).size().unstack(fill_value=0)
    # reorder columns
    for col in ["train", "val", "test"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["train", "val", "test"]]
    ct["total"] = ct.sum(axis=1)
    print(ct.to_string())
    print("-" * 60)
    totals = ct.sum()
    print(f"  {'TOTAL':<15} {totals['train']:>6}  {totals['val']:>6}  {totals['test']:>6}  {totals['total']:>6}")
    print("=" * 60)

    # Verify against targets
    print("\nTarget vs Actual:")
    for cls in ["Normal", "Tuberculosis"]:
        for split in ["train", "val", "test"]:
            actual = int(ct.loc[cls, split])
            target = TARGET[cls][split]
            match  = "✅" if actual == target else "⚠️ "
            print(f"  {match} {cls:<15} {split:<5}: target={target:>4}  actual={actual:>4}")


if __name__ == "__main__":
    main()

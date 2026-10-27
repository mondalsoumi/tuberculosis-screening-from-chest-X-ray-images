# -*- coding: utf-8 -*-
"""
src/audit.py  –  LiteTBNet Dataset Audit
=========================================
Produces outputs/audit_report.md with:
  1. File counts per class, image sizes/modes/formats
  2. Corrupt file detection
  3. Exact-duplicate detection (MD5) and near-duplicate detection (pHash, Hamming ≤4)
  4. Per-class pixel intensity mean/std and mean file size
  5. Shortcut probes (5-fold CV logistic regression):
       a) 32x32 grayscale thumbnails
       b) Outer 10% border strips only
       c) mean/std/percentile intensity features only
"""

import os
import sys

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import hashlib
import collections
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ── project root on sys.path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_global_seed, print_gpu_info, SEED, OUTPUTS_DIR, get_class_dirs

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────
THUMB_SIZE  = (32, 32)
BORDER_PCT  = 0.10        # outer 10 % border
PHASH_THRESH = 4          # Hamming distance ≤ 4 → near-duplicate
PERCENTILES  = [5, 25, 50, 75, 95]
CV_FOLDS     = 5
PROBE_MAX_ITER = 500   # saga solver converges fast; lbfgs was timing out


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_safe(path: str):
    """Return (PIL.Image, error_str_or_None)."""
    try:
        img = Image.open(path)
        img.verify()          # catches truncated files
        img = Image.open(path)  # re-open after verify
        img.load()
        return img, None
    except Exception as e:
        return None, str(e)


def extract_border_strip(arr: np.ndarray, pct: float) -> np.ndarray:
    """Return the outer border strip (top/bottom/left/right pct rows/cols) flattened."""
    h, w = arr.shape
    bh = max(1, int(h * pct))
    bw = max(1, int(w * pct))
    top    = arr[:bh, :]
    bottom = arr[h - bh:, :]
    left   = arr[bh:h - bh, :bw]
    right  = arr[bh:h - bh, w - bw:]
    return np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])


def phash_str(path: str) -> str | None:
    try:
        img = Image.open(path).convert("L")
        return str(imagehash.phash(img))
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# Step 1 – collect metadata per image
# ════════════════════════════════════════════════════════════════════════════

def collect_metadata(class_dirs: dict) -> pd.DataFrame:
    """Walk each class directory and collect per-image metadata."""
    rows = []
    for cls, dpath in class_dirs.items():
        files = sorted(
            f for f in Path(dpath).iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
        )
        for fpath in tqdm(files, desc=f"  Scanning {cls}", unit="img"):
            img, err = load_image_safe(str(fpath))
            row = {
                "path"   : str(fpath),
                "class"  : cls,
                "filename": fpath.name,
                "size_bytes": fpath.stat().st_size,
                "corrupt": err is not None,
                "error"  : err,
                "width"  : None,
                "height" : None,
                "mode"   : None,
                "format" : None,
                "md5"    : None,
                "phash"  : None,
            }
            if img is not None:
                row["width"]  = img.width
                row["height"] = img.height
                row["mode"]   = img.mode
                row["format"] = img.format
                row["md5"]    = md5(str(fpath))
                row["phash"]  = phash_str(str(fpath))
            rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 – pixel statistics  (only on valid images)
# ════════════════════════════════════════════════════════════════════════════

def pixel_stats(class_dirs: dict, meta: pd.DataFrame) -> dict:
    """Compute per-class mean & std of pixel intensity across all valid images."""
    stats = {}
    for cls, dpath in class_dirs.items():
        valid = meta[(meta["class"] == cls) & (~meta["corrupt"])]
        all_pixels = []
        for path in tqdm(valid["path"], desc=f"  Pixel stats {cls}", unit="img"):
            try:
                img = Image.open(path).convert("L")
                all_pixels.append(np.array(img, dtype=np.float32).ravel())
            except Exception:
                pass
        if all_pixels:
            arr = np.concatenate(all_pixels)
            stats[cls] = {
                "mean" : float(arr.mean()),
                "std"  : float(arr.std()),
                "n_images": len(all_pixels),
            }
        else:
            stats[cls] = {"mean": None, "std": None, "n_images": 0}
    return stats


# ════════════════════════════════════════════════════════════════════════════
# Step 3 – duplicate detection
# ════════════════════════════════════════════════════════════════════════════

def find_exact_duplicates(meta: pd.DataFrame):
    """Return list of (md5, [paths]) for md5 with >1 occurrence."""
    valid = meta[meta["md5"].notna()]
    groups = valid.groupby("md5")["path"].apply(list)
    dups = groups[groups.map(len) > 1]
    return dups.reset_index().to_dict("records")  # list of dicts with md5, path


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def find_near_duplicates(meta: pd.DataFrame, thresh: int = PHASH_THRESH):
    """Find pairs with pHash Hamming distance <= thresh (excluding exact MD5 dups)."""
    valid = meta[meta["phash"].notna() & ~meta["corrupt"]].copy()
    near_dup_pairs = []
    paths  = valid["path"].tolist()
    hashes = valid["phash"].tolist()
    md5s   = valid["md5"].tolist()
    n = len(paths)
    for i in tqdm(range(n), desc="  Near-dup scan", unit="img"):
        for j in range(i + 1, n):
            if md5s[i] == md5s[j]:
                continue  # already an exact dup
            try:
                d = hamming(hashes[i], hashes[j])
            except Exception:
                continue
            if d <= thresh:
                near_dup_pairs.append({
                    "path_a" : paths[i],
                    "path_b" : paths[j],
                    "class_a": valid.iloc[i]["class"],
                    "class_b": valid.iloc[j]["class"],
                    "hamming": d,
                })
    return near_dup_pairs


# ════════════════════════════════════════════════════════════════════════════
# Step 4 – Shortcut probes
# ════════════════════════════════════════════════════════════════════════════

def build_feature_sets(meta: pd.DataFrame):
    """Build X arrays and y labels for the three probe types."""
    valid = meta[~meta["corrupt"]].copy()
    print(f"  Building features for {len(valid)} valid images …")

    X_thumb   = []
    X_border  = []
    X_stats   = []
    y         = []
    label_map = {"Normal": 0, "Tuberculosis": 1}

    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="  Feature extraction", unit="img"):
        try:
            img = Image.open(row["path"]).convert("L")
            arr = np.array(img, dtype=np.float32)

            # a) 32×32 thumbnail
            thumb = np.array(img.resize(THUMB_SIZE, Image.LANCZOS), dtype=np.float32).ravel() / 255.0

            # b) border strip
            border = extract_border_strip(arr, BORDER_PCT) / 255.0

            # c) global stats: mean, std, percentiles
            pct_vals = np.percentile(arr, PERCENTILES)
            stat_feat = np.array([arr.mean(), arr.std()] + list(pct_vals)) / 255.0

            X_thumb.append(thumb)
            X_border.append(border)
            X_stats.append(stat_feat)
            y.append(label_map[row["class"]])
        except Exception as e:
            continue  # skip problematic files silently

    return (
        np.array(X_thumb,  dtype=np.float32),
        np.array(X_border, dtype=np.float32),
        np.array(X_stats,  dtype=np.float32),
        np.array(y,        dtype=np.int32),
    )


def run_probe(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Run 5-fold stratified CV with logistic regression.
    For high-dimensional X (e.g. border strips), PCA to 64 components first.
    """
    print(f"  Running probe: {name}  (X shape={X.shape}) …", flush=True)
    n_feat = X.shape[1]
    # Use PCA when features >> samples to keep saga feasible
    if n_feat > 512:
        print(f"    High-dim input ({n_feat} features) → PCA(64) first", flush=True)
        steps = [
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=64, random_state=SEED)),
            ("lr",     LogisticRegression(
                max_iter=PROBE_MAX_ITER,
                random_state=SEED,
                class_weight="balanced",
                solver="saga",
                tol=1e-3,
            )),
        ]
    else:
        steps = [
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(
                max_iter=PROBE_MAX_ITER,
                random_state=SEED,
                class_weight="balanced",
                solver="saga",
                tol=1e-3,
            )),
        ]
    pipe = Pipeline(steps)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    # n_jobs=1: avoid Windows multiprocessing deadlock with joblib
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    result = {
        "name"   : name,
        "cv_mean": float(scores.mean()),
        "cv_std" : float(scores.std()),
        "cv_folds": scores.tolist(),
    }
    print(f"  [{name}]  acc = {scores.mean():.4f} ± {scores.std():.4f}", flush=True)
    return result


def shortcut_interpretation(probes: list[dict]) -> str:
    flagged = [p["name"] for p in probes if p["cv_mean"] > 0.90]
    if not flagged:
        return (
            "**No shortcut risk detected.** None of the three shortcut probes exceeded "
            "90 % cross-validated accuracy.  The dataset does not appear to contain "
            "trivially-learnable acquisition or border artefacts that would inflate model scores."
        )
    lines = [
        "⚠️  **SHORTCUT RISK FLAGGED** for probe(s): " + ", ".join(f"`{n}`" for n in flagged) + ".",
        "",
        "| Probe | CV Acc | Risk |",
        "|-------|--------|------|",
    ]
    for p in probes:
        flag = "🔴 >90%" if p["cv_mean"] > 0.90 else "🟢 ≤90%"
        lines.append(f"| {p['name']} | {p['cv_mean']:.4f} ± {p['cv_std']:.4f} | {flag} |")
    lines += [
        "",
        "**Interpretation:**",
        "- A probe accuracy >90 % on **thumbnail** (a) means coarse pixel patterns "
          "alone separate classes, suggesting strong acquisition-level differences "
          "(e.g. different scanners, exposure settings) rather than purely pathological signal.",
        "- A probe accuracy >90 % on **border strips** (b) strongly suggests image-level "
          "artefacts such as metadata text overlays, padding colour or scanner borders "
          "that correlate with label — a source-domain shortcut.",
        "- A probe accuracy >90 % on **global intensity stats** (c) indicates systematic "
          "brightness/contrast differences between classes at the dataset level, pointing "
          "to acquisition protocol differences.  Models trained without normalisation "
          "may exploit this rather than learning anatomy.",
        "",
        "**Recommended mitigations:** histogram equalisation or CLAHE, aggressive border "
          "cropping, mixed-source data collection, and dataset-unbiased evaluation.",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Report builder
# ════════════════════════════════════════════════════════════════════════════

def build_report(meta: pd.DataFrame,
                 pix_stats: dict,
                 exact_dups: list,
                 near_dups: list,
                 probes: list[dict]) -> str:
    lines = []

    def h(level, text):
        lines.append(f"{'#' * level} {text}\n")

    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        lines.append("")

    h(1, "LiteTBNet Dataset Audit Report")
    lines.append(f"**Generated by** `src/audit.py`  |  **Seed** = {SEED}\n")

    # ── 1. File counts ──────────────────────────────────────────────────────
    h(2, "1. File Counts per Class")
    for cls in ["Normal", "Tuberculosis"]:
        sub = meta[meta["class"] == cls]
        valid = sub[~sub["corrupt"]]
        corrupt = sub[sub["corrupt"]]
        lines.append(f"- **{cls}**: {len(sub)} files total  |  {len(valid)} valid  |  {len(corrupt)} corrupt\n")

    # ── 2. Image sizes / modes / formats ────────────────────────────────────
    h(2, "2. Image Sizes, Modes and Formats")
    for cls in ["Normal", "Tuberculosis"]:
        h(3, cls)
        sub = meta[(meta["class"] == cls) & (~meta["corrupt"])]
        if len(sub) == 0:
            lines.append("_No valid images._\n")
            continue

        # size counts
        size_counts = sub.apply(lambda r: f"{int(r['width'])}×{int(r['height'])}", axis=1).value_counts()
        lines.append(f"**Distinct sizes ({len(size_counts)} unique):**\n")
        size_rows = [(sz, cnt) for sz, cnt in size_counts.head(20).items()]
        table(["Size (WxH)", "Count"], size_rows)
        if len(size_counts) > 20:
            lines.append(f"_… and {len(size_counts)-20} more distinct sizes._\n")

        # modes
        mode_counts = sub["mode"].value_counts()
        lines.append("**Colour modes:**\n")
        table(["Mode", "Count"], mode_counts.items())

        # formats
        fmt_counts = sub["format"].value_counts()
        lines.append("**File formats:**\n")
        table(["Format", "Count"], fmt_counts.items())

    # ── 3. Corrupt files ─────────────────────────────────────────────────────
    h(2, "3. Corrupt Files")
    corrupt = meta[meta["corrupt"]]
    if len(corrupt) == 0:
        lines.append("✅ No corrupt files detected.\n")
    else:
        lines.append(f"❌ **{len(corrupt)} corrupt file(s) detected:**\n")
        table(["Class", "Filename", "Error"], corrupt[["class", "filename", "error"]].values.tolist())

    # ── 4. Duplicates ─────────────────────────────────────────────────────────
    h(2, "4. Duplicate Detection")

    h(3, "4a. Exact Duplicates (MD5)")
    if not exact_dups:
        lines.append("✅ No exact duplicates found.\n")
    else:
        lines.append(f"⚠️ **{len(exact_dups)} MD5 group(s)** with duplicates:\n")
        for rec in exact_dups[:50]:  # cap display
            paths_str = ", ".join(Path(p).name for p in rec["path"])
            lines.append(f"- `{rec['md5'][:12]}…`  →  {paths_str}\n")
        if len(exact_dups) > 50:
            lines.append(f"_… {len(exact_dups)-50} more groups omitted._\n")

    h(3, "4b. Near-Duplicates (pHash, Hamming ≤ 4)")
    if not near_dups:
        lines.append("✅ No near-duplicates found.\n")
    else:
        lines.append(f"⚠️ **{len(near_dups)} near-duplicate pair(s):**\n")
        nd_rows = [
            (Path(p["path_a"]).name, p["class_a"],
             Path(p["path_b"]).name, p["class_b"], p["hamming"])
            for p in near_dups[:50]
        ]
        table(["File A", "Class A", "File B", "Class B", "Hamming"], nd_rows)
        if len(near_dups) > 50:
            lines.append(f"_… {len(near_dups)-50} more pairs omitted._\n")

    # breakdown: within-class vs cross-class
    if near_dups:
        within = sum(1 for p in near_dups if p["class_a"] == p["class_b"])
        cross  = len(near_dups) - within
        lines.append(f"- Within-class pairs : {within}\n")
        lines.append(f"- Cross-class pairs  : {cross}\n")

    # ── 5. Pixel statistics ───────────────────────────────────────────────────
    h(2, "5. Per-Class Pixel Intensity and File Size Statistics")
    stat_rows = []
    for cls in ["Normal", "Tuberculosis"]:
        sub = meta[(meta["class"] == cls) & (~meta["corrupt"])]
        mean_size_kb = sub["size_bytes"].mean() / 1024
        ps = pix_stats.get(cls, {})
        stat_rows.append([
            cls,
            ps.get("n_images", 0),
            f"{ps['mean']:.3f}" if ps.get("mean") is not None else "N/A",
            f"{ps['std']:.3f}"  if ps.get("std")  is not None else "N/A",
            f"{mean_size_kb:.1f} KB",
        ])
    table(["Class", "N Images", "Mean Intensity", "Std Intensity", "Mean File Size"], stat_rows)

    # ── 6. Shortcut probes ────────────────────────────────────────────────────
    h(2, "6. Shortcut Probes (5-Fold Stratified CV — Logistic Regression)")
    lines.append("Each probe uses **only** the feature set described to test whether "
                 "trivial patterns separate the classes.\n")
    probe_rows = [
        (p["name"],
         f"{p['cv_mean']:.4f}",
         f"{p['cv_std']:.4f}",
         ", ".join(f"{s:.4f}" for s in p["cv_folds"]))
        for p in probes
    ]
    table(["Probe", "CV Mean Acc", "CV Std", "Per-Fold Accuracies"], probe_rows)

    h(3, "Interpretation")
    lines.append(shortcut_interpretation(probes) + "\n")

    return "\n".join(lines)


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

    # ── Step 1: Metadata ────────────────────────────────────────────────────
    print("\n[1/5] Collecting image metadata …")
    meta = collect_metadata(class_dirs)
    print(f"  Total files found: {len(meta)}")
    for cls in ["Normal", "Tuberculosis"]:
        n = (meta["class"] == cls).sum()
        nc = (meta[meta["class"] == cls]["corrupt"]).sum()
        print(f"  {cls}: {n} files  ({nc} corrupt)")

    # ── Step 2: Pixel statistics ────────────────────────────────────────────
    print("\n[2/5] Computing per-class pixel statistics …")
    pix_st = pixel_stats(class_dirs, meta)
    for cls, s in pix_st.items():
        print(f"  {cls}: mean={s['mean']:.3f}  std={s['std']:.3f}  n={s['n_images']}")

    # ── Step 3: Duplicate detection ─────────────────────────────────────────
    print("\n[3/5] Detecting duplicates …")
    exact_dups = find_exact_duplicates(meta)
    print(f"  Exact duplicate MD5 groups: {len(exact_dups)}")
    near_dups  = find_near_duplicates(meta, thresh=PHASH_THRESH)
    print(f"  Near-duplicate pairs (Hamming ≤ {PHASH_THRESH}): {len(near_dups)}")

    # ── Step 4: Shortcut probes ─────────────────────────────────────────────
    print("\n[4/5] Building feature sets for shortcut probes …")
    X_thumb, X_border, X_stats_feat, y = build_feature_sets(meta)
    print(f"  Samples: {len(y)}  |  Class 0 (Normal)={int((y==0).sum())}  |  Class 1 (TB)={int((y==1).sum())}")

    probes = []
    print("\n  Running 5-fold CV probes (n_jobs=1, saga solver) …", flush=True)
    if len(X_thumb) > 0:
        probes.append(run_probe(X_thumb,      y, "a) 32x32 thumbnail"))
        sys.stdout.flush()
        probes.append(run_probe(X_border,     y, "b) Border strips only"))
        sys.stdout.flush()
        probes.append(run_probe(X_stats_feat, y, "c) Intensity stats only"))
        sys.stdout.flush()
    else:
        print("  ⚠️  No valid images for probes.")

    # ── Step 5: Write report ────────────────────────────────────────────────
    print("\n[5/5] Writing audit report …")
    report_md = build_report(meta, pix_st, exact_dups, near_dups, probes)
    report_path = os.path.join(OUTPUTS_DIR, "audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  Saved: {report_path}")

    # ── Save raw metadata ───────────────────────────────────────────────────
    meta_csv = os.path.join(OUTPUTS_DIR, "audit_metadata.csv")
    meta.to_csv(meta_csv, index=False)
    print(f"  Metadata CSV: {meta_csv}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    for cls in ["Normal", "Tuberculosis"]:
        sub = meta[meta["class"] == cls]
        print(f"  {cls}: {len(sub)} files")
    print(f"  Corrupt: {meta['corrupt'].sum()}")
    print(f"  Exact dup groups: {len(exact_dups)}")
    print(f"  Near-dup pairs: {len(near_dups)}")
    if probes:
        for p in probes:
            flag = " ⚠️ >90% – SHORTCUT RISK" if p["cv_mean"] > 0.90 else ""
            print(f"  {p['name']}: {p['cv_mean']:.4f} ± {p['cv_std']:.4f}{flag}")
    print("=" * 60)

    print("\n── Full audit_report.md ──────────────────────────────────────")
    print(report_md)


if __name__ == "__main__":
    main()

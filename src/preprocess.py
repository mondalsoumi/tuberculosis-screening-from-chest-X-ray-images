# -*- coding: utf-8 -*-
"""src/preprocess.py — load, resize, cache per split as .npy arrays."""
import argparse, os, sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import print_gpu_info, SEED

SMOKE_TRAIN = 120
SMOKE_VAL   = 30
SMOKE_TEST  = 30
SMOKE_MIN_TB = 20
IMG_SIZE = 224


def build_path_map(root: str) -> dict:
    """Recursively map basename -> full_path for all files under root."""
    m = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            m[f] = os.path.join(dirpath, f)
    return m


def resolve_paths(df: pd.DataFrame, path_map: dict) -> pd.Series:
    def _resolve(p):
        bn = os.path.basename(p.replace("\\", "/"))
        assert bn in path_map, f"Cannot resolve: {bn}"
        return path_map[bn]
    return df["path"].apply(_resolve)


def smoke_subset(df_split: pd.DataFrame, max_n: int, min_tb: int = 0) -> pd.DataFrame:
    tb   = df_split[df_split["label"] == 1]
    norm = df_split[df_split["label"] == 0]
    if min_tb > 0:
        n_tb   = min(min_tb, len(tb))
        n_norm = min(max_n - n_tb, len(norm))
    else:
        half   = max_n // 2
        n_tb   = min(half, len(tb))
        n_norm = min(max_n - n_tb, len(norm))
    out = pd.concat([
        tb.sample(n=n_tb, random_state=SEED),
        norm.sample(n=n_norm, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out


def process_split(rows: pd.DataFrame, split: str, out_dir: str):
    X, y = [], []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {split}", unit="img"):
        img = Image.open(row["real"]).convert("L").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        X.append(np.asarray(img, dtype=np.uint8))
        y.append(int(row["label"]))
    X_arr = np.stack(X, dtype=np.uint8)
    y_arr = np.array(y, dtype=np.int64)
    np.save(os.path.join(out_dir, f"{split}_X.npy"), X_arr)
    np.save(os.path.join(out_dir, f"{split}_y.npy"), y_arr)
    tb_n = int((y_arr == 1).sum())
    print(f"  {split}: {len(y_arr)} samples  (TB={tb_n}, Normal={len(y_arr)-tb_n})")
    return X_arr, y_arr


def main():
    print_gpu_info()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv("outputs/splits.csv")
    path_map = build_path_map(args.data_root)
    df["real"] = resolve_paths(df, path_map)

    for split, max_n, min_tb in [("train", SMOKE_TRAIN, SMOKE_MIN_TB),
                                   ("val",   SMOKE_VAL,   0),
                                   ("test",  SMOKE_TEST,  0)]:
        rows = df[df["split"] == split].copy()
        if args.smoke:
            rows = smoke_subset(rows, max_n, min_tb)
        assert rows["label"].nunique() == 2, f"{split}: only one class present!"
        process_split(rows, split, args.out)


if __name__ == "__main__":
    main()

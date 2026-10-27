# -*- coding: utf-8 -*-
"""src/oversample.py — SMOTE / ADASYN / BorderlineSMOTE on train split cache."""
import argparse, os, sys, gc
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import print_gpu_info, set_global_seed, SEED

METHODS = ["none", "smote", "adasyn", "bsmote"]


def run_method(method: str, X_flat: np.ndarray, y: np.ndarray,
               k: int, out_dir: str, records: list):
    """Run one oversampling method and save results."""
    if method == "none":
        X_out, y_out = X_flat.copy(), y.copy()
    else:
        try:
            if method == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(sampling_strategy="auto", k_neighbors=k, random_state=SEED)
            elif method == "adasyn":
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(sampling_strategy="auto", n_neighbors=k, random_state=SEED)
            elif method == "bsmote":
                from imblearn.over_sampling import BorderlineSMOTE
                sampler = BorderlineSMOTE(sampling_strategy="auto", k_neighbors=k,
                                          kind="borderline-1", random_state=SEED)
            X_out, y_out = sampler.fit_resample(X_flat, y)
            X_out = np.clip(np.round(X_out), 0, 255).astype(np.uint8)
            y_out = y_out.astype(np.int64)
        except Exception as e:
            print(f"  [{method}] ERROR: {e} — skipping")
            records.append({"method": method, "n_total": None,
                            "n_tb": None, "n_normal": None, "status": f"ERROR: {e}"})
            return

    # Save flat → reshape back to (N,224,224)
    n_side = int(np.sqrt(X_flat.shape[1]))
    X_out_img = X_out.reshape(-1, n_side, n_side).astype(np.uint8)
    y_out = np.array(y_out, dtype=np.int64)

    np.save(os.path.join(out_dir, f"train_{method}_X.npy"), X_out_img)
    np.save(os.path.join(out_dir, f"train_{method}_y.npy"), y_out)

    n_tb     = int((y_out == 1).sum())
    n_normal = int((y_out == 0).sum())
    print(f"  [{method}] total={len(y_out)}  TB={n_tb}  Normal={n_normal}")
    records.append({"method": method, "n_total": len(y_out),
                    "n_tb": n_tb, "n_normal": n_normal, "status": "OK"})


def main():
    print_gpu_info()
    set_global_seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--method", default="smote",
                    choices=METHODS + ["all"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    k = 3 if args.smoke else 5
    methods = METHODS if args.method == "all" else [args.method]

    X = np.load(os.path.join(args.cache, "train_X.npy"))   # (N,224,224) uint8
    y = np.load(os.path.join(args.cache, "train_y.npy"))   # (N,) int64
    print(f"  Loaded train: {X.shape}  TB={int((y==1).sum())}  Normal={int((y==0).sum())}")

    N = X.shape[0]
    X_flat = X.reshape(N, -1).astype(np.float32)   # flatten for imblearn

    records = []
    for method in methods:
        run_method(method, X_flat, y, k, args.cache, records)
        gc.collect()   # free memory between methods

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(records).to_csv("outputs/oversample_counts.csv", index=False)
    print(f"  Saved outputs/oversample_counts.csv")


if __name__ == "__main__":
    main()

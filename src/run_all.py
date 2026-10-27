# -*- coding: utf-8 -*-
"""src/run_all.py — orchestrate all training runs and collect results."""
import argparse, csv, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import run as train_run


TIER1 = [
    ("litetbnet",      "none"),
    ("litetbnet",      "weighted"),
    ("litetbnet",      "smote"),
    ("litetbnet",      "adasyn"),
    ("litetbnet",      "bsmote"),
    ("litetbnet_noeca","adasyn"),
]
TIER2 = [
    ("swin",      "none"),
    ("swin",      "adasyn"),
    ("inception", "none"),
    ("inception", "adasyn"),
]
TIER3 = [
    ("swin",      "weighted"),
    ("swin",      "smote"),
    ("swin",      "bsmote"),
    ("inception", "weighted"),
    ("inception", "smote"),
    ("inception", "bsmote"),
]
TIER_MAP = {1: TIER1, 2: TIER2, 3: TIER3}


def collect(out_root):
    out_root = Path(out_root)
    rows = []
    for d in sorted(out_root.iterdir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        t = m.get("test", {})
        rows.append(dict(
            run=d.name,
            model=m.get("model"), method=m.get("method"),
            best_epoch=m.get("best_epoch"),
            params=m.get("params"),
            test_tb_p=t.get("tb_precision"),
            test_tb_r=t.get("tb_recall"),
            test_tb_f1=t.get("tb_f1"),
            test_acc=t.get("accuracy"),
            test_auc=t.get("roc_auc"),
        ))
    if not rows:
        return 0
    fields = list(rows[0].keys())
    out_csv = out_root / "results_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache",  required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--tiers",  default="1,2,3")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--smoke",  action="store_true")
    a = p.parse_args()

    tiers = [int(t) for t in a.tiers.split(",")]
    epochs = 1 if a.smoke else a.epochs
    bs = 8 if a.smoke else 32

    queue = []
    for t in tiers:
        queue.extend(TIER_MAP[t])

    for model_name, method in queue:
        t0 = time.time()
        try:
            train_run(model_name, method, a.cache, a.out,
                      epochs=epochs, bs=bs, device="auto")
            elapsed = time.time() - t0
            print(f"OK  {model_name}_{method}  {elapsed:.1f}s", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAIL {model_name}_{method}  {elapsed:.1f}s  {e}", flush=True)

    n = collect(a.out)
    print(f"results_table.csv rows={n}", flush=True)


if __name__ == "__main__":
    main()

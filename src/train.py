# -*- coding: utf-8 -*-
"""src/train.py — training loop for TB classification experiments."""
import argparse, csv, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from gpu_data import GPUBatcher
from model import LiteTBNet
from baselines import build_swin_tiny, build_inception_v3


# ── helpers ────────────────────────────────────────────────────────────────────

def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(name):
    if name == "litetbnet":
        return LiteTBNet(use_eca=True)
    if name == "litetbnet_noeca":
        return LiteTBNet(use_eca=False)
    if name == "swin":
        return build_swin_tiny(pretrained=False)
    if name == "inception":
        return build_inception_v3(pretrained=False)
    raise ValueError(f"Unknown model: {name}")


def _count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _load_split(cache, prefix):
    X = np.load(Path(cache) / f"{prefix}_X.npy")
    y = np.load(Path(cache) / f"{prefix}_y.npy")
    return X, y


def _compute_metrics(probs, labels):
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, balanced_accuracy_score,
        roc_auc_score, precision_recall_fscore_support,
    )
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds).tolist()
    acc = float(accuracy_score(labels, preds))
    bal = float(balanced_accuracy_score(labels, preds))
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")
    p_tb, r_tb, f1_tb, _ = precision_recall_fscore_support(
        labels, preds, labels=[1], zero_division=0)
    pm, rm, f1m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    return dict(
        confusion_matrix=cm, accuracy=acc, balanced_accuracy=bal, roc_auc=auc,
        tb_precision=float(p_tb[0]), tb_recall=float(r_tb[0]), tb_f1=float(f1_tb[0]),
        macro_precision=float(pm), macro_recall=float(rm), macro_f1=float(f1m),
        weighted_precision=float(pw), weighted_recall=float(rw), weighted_f1=float(f1w),
    )


@torch.no_grad()
def _evaluate(model, batcher, device, use_amp):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0; total_n = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for xb, yb in batcher:
        xb = xb.to(device); yb = yb.to(device)
        if use_amp:
            with autocast():
                logits = model(xb)
        else:
            logits = model(xb)
        loss = bce(logits, yb).mean().item()
        total_loss += loss * xb.shape[0]; total_n += xb.shape[0]
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.append(probs); all_labels.append(yb.squeeze(1).cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    avg_loss = total_loss / total_n
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels.astype(int)).mean())
    return avg_loss, acc, probs, labels


# ── main run ──────────────────────────────────────────────────────────────────

def run(model_name, method, cache, out_root,
        epochs=50, bs=32, seed=42, mask_border=0,
        stop_after=None, device="auto"):

    device = _auto_device() if device == "auto" else device
    print(f"[train] device={device}", flush=True)

    suffix = f"_mb{mask_border}" if mask_border else ""
    run_dir = Path(out_root) / f"{model_name}_{method}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"SKIP {run_dir.name}", flush=True)
        return

    _seed(seed)

    # ── data ──────────────────────────────────────────────────────────────────
    train_prefix = "train_none" if method in ("none", "weighted") else f"train_{method}"
    Xtr, ytr = _load_split(cache, train_prefix)
    Xvl, yvl = _load_split(cache, "val")
    Xts, yts = _load_split(cache, "test")

    use_amp = (device == "cuda")

    def make_batcher(X, y, shuffle, augment, drop_last=False):
        return GPUBatcher(X, y, batch_size=bs, shuffle=shuffle, augment=augment,
                          device=device, seed=seed,
                          drop_last=drop_last, mask_border=mask_border)

    tr_batcher  = make_batcher(Xtr, ytr, shuffle=True,  augment=True,  drop_last=True)
    val_batcher = make_batcher(Xvl, yvl, shuffle=False, augment=False)
    tst_batcher = make_batcher(Xts, yts, shuffle=False, augment=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = _build_model(model_name).to(device)
    if use_amp:
        model = model.to(memory_format=torch.channels_last)

    n_params = _count_params(model)

    # ── loss weights ──────────────────────────────────────────────────────────
    if method == "weighted":
        n_total = len(ytr)
        n_pos = int(ytr.sum())
        n_neg = n_total - n_pos
        w_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
        w_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
    else:
        w_pos = w_neg = None

    bce_elem = nn.BCEWithLogitsLoss(reduction="none")

    def train_loss_fn(logits, targets):
        elem = bce_elem(logits, targets)
        if w_pos is not None:
            w = torch.where(targets == 1,
                            torch.tensor(w_pos, device=device, dtype=torch.float32),
                            torch.tensor(w_neg, device=device, dtype=torch.float32))
            return (w * elem).mean()
        return elem.mean()

    # ── optimizer / scheduler / scaler ────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=5, min_lr=1e-6)
    scaler = GradScaler(enabled=use_amp)

    # ── resume ────────────────────────────────────────────────────────────────
    last_pt = run_dir / "last.pt"
    start_epoch = 0
    history = []
    best_val_loss = math.inf
    best_state = None
    best_epoch = 0

    if last_pt.exists():
        ckpt = torch.load(last_pt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch    = ckpt["epoch"] + 1
        history        = ckpt["history"]
        best_val_loss  = ckpt["best_val_loss"]
        best_state     = ckpt["best_state"]
        best_epoch     = ckpt.get("best_epoch", 0)
        print(f"[train] resumed from epoch {ckpt['epoch']}", flush=True)

    # ── history csv ───────────────────────────────────────────────────────────
    hist_csv = run_dir / "history.csv"
    csv_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                  "lr", "seconds"]
    if not hist_csv.exists():
        with open(hist_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── epoch loop ────────────────────────────────────────────────────────────
    peak_gpu_mb = 0
    t_train_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        run_loss = run_correct = run_n = 0

        for xb, yb in tr_batcher:
            xb = xb.to(device); yb = yb.to(device)
            if use_amp:
                xb = xb.to(memory_format=torch.channels_last)
            opt.zero_grad()
            if use_amp:
                with autocast():
                    logits = model(xb)
                    loss = train_loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                logits = model(xb)
                loss = train_loss_fn(logits, yb)
                loss.backward(); opt.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                run_correct += (preds == yb).sum().item()
            run_loss += loss.item() * xb.shape[0]
            run_n    += xb.shape[0]

        train_loss = run_loss / run_n if run_n else 0.0
        train_acc  = run_correct / run_n if run_n else 0.0

        val_loss, val_acc, _, _ = _evaluate(model, val_batcher, device, use_amp)
        sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            torch.save(best_state, run_dir / "best.pt")

        cur_lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_loss=round(train_loss, 6),
                   train_acc=round(train_acc, 6), val_loss=round(val_loss, 6),
                   val_acc=round(val_acc, 6), lr=cur_lr, seconds=round(elapsed, 2))
        history.append(row)
        with open(hist_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        if use_amp and torch.cuda.is_available():
            peak_gpu_mb = max(peak_gpu_mb,
                              torch.cuda.max_memory_allocated(device) // (1024 * 1024))

        # ── checkpoint ────────────────────────────────────────────────────────
        tmp_pt = run_dir / "_last_tmp.pt"
        if tmp_pt.exists():
            try:
                tmp_pt.unlink()
            except OSError:
                pass
        torch.save(dict(
            model=model.state_dict(), optimizer=opt.state_dict(),
            scheduler=sched.state_dict(), scaler=scaler.state_dict(),
            epoch=epoch, history=history,
            best_val_loss=best_val_loss, best_state=best_state,
            best_epoch=best_epoch,
        ), last_pt)

        if stop_after is not None and epoch >= stop_after:
            print(f"[train] stop_after={stop_after} at epoch {epoch}", flush=True)
            return

    # ── final evaluation ──────────────────────────────────────────────────────
    train_seconds = time.time() - t_train_start

    model.load_state_dict(best_state)

    _, _, val_probs, val_labels = _evaluate(model, val_batcher, device, use_amp)
    np.save(run_dir / "val_probs.npy",
            np.stack([val_probs, val_labels], axis=1))

    _, _, tst_probs, tst_labels = _evaluate(model, tst_batcher, device, use_amp)
    np.save(run_dir / "test_probs.npy",
            np.stack([tst_probs, tst_labels], axis=1))

    val_metrics  = _compute_metrics(val_probs, val_labels)
    test_metrics = _compute_metrics(tst_probs, tst_labels)

    out = dict(
        model=model_name, method=method, seed=seed,
        epochs_run=len(history), best_epoch=best_epoch,
        params=n_params, train_seconds=round(train_seconds, 2),
        peak_gpu_mb=peak_gpu_mb,
        val=val_metrics, test=test_metrics,
    )
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[train] done best_epoch={best_epoch} val_loss={best_val_loss:.4f}",
          flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True,
                   choices=["litetbnet","litetbnet_noeca","swin","inception"])
    p.add_argument("--method", required=True,
                   choices=["none","weighted","smote","adasyn","bsmote"])
    p.add_argument("--cache",  required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs",     type=int, default=32)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--mask_border", type=int, default=0)
    p.add_argument("--stop_after",  type=int, default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    run(a.model, a.method, a.cache, a.out,
        epochs=a.epochs, bs=a.bs, seed=a.seed,
        mask_border=a.mask_border, stop_after=a.stop_after,
        device=a.device)


if __name__ == "__main__":
    _cli()

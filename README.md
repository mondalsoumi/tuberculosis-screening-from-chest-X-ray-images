# LiteTBNet — TB Chest X-Ray Classification

Reproducible research codebase for classifying Tuberculosis (TB) vs Normal chest X-rays.

## Project Structure

```
litetbnet/
├── data/
│   ├── Normal/          ← 3500 images (from Kaggle)
│   └── Tuberculosis/    ← 700 images (from Kaggle)
├── src/
│   ├── utils.py             (shared: seed, GPU info, paths)
│   ├── download_dataset.py  (Step 0: get dataset)
│   ├── audit.py             (Step 1: dataset audit)
│   └── make_splits.py       (Step 2: create splits)
├── outputs/
│   ├── audit_report.md      (generated)
│   ├── audit_metadata.csv   (generated)
│   └── splits.csv           (generated)
├── logs/
│   └── dedup_removed.log    (generated)
└── requirements.txt
```

## Setup

```powershell
pip install -r requirements.txt
```

## Step 0 — Download Dataset

```powershell
python src/download_dataset.py
```

**If Kaggle credentials are missing**, the script will stop and print exact instructions.
Short version:
1. Go to https://www.kaggle.com → Settings → API → **Create New Token** → downloads `kaggle.json`
2. Place it at `C:\Users\<YourName>\.kaggle\kaggle.json`
3. Re-run the download script.

**Manual alternative:**
1. Download from https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
2. Extract so that `data/Normal/` (3500 files) and `data/Tuberculosis/` (700 files) exist.

## Step 1 — Dataset Audit

```powershell
python src/audit.py
```

Produces `outputs/audit_report.md` with:
- File counts per class
- Image size / mode / format distributions
- Corrupt file detection
- Exact duplicate detection (MD5)
- Near-duplicate detection (pHash, Hamming ≤ 4)
- Per-class pixel intensity statistics
- **Shortcut probes** (5-fold CV logistic regression on thumbnail / border / intensity features)

## Step 2 — Create Splits

```powershell
python src/make_splits.py
```

Creates `outputs/splits.csv` with stratified splits:

| Class         | Train | Val | Test |
|---------------|------:|----:|-----:|
| Normal        |  2531 | 698 |  271 |
| Tuberculosis  |   502 | 134 |   64 |

Exact duplicates are removed before splitting. Leakage is asserted.

## Reproducibility

- Global seed = **42** (Python, NumPy, PyTorch, cuDNN)
- All scripts print GPU/CPU info at start.
- All results are real — no fabricated numbers.

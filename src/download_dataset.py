# -*- coding: utf-8 -*-
"""
src/download_dataset.py  –  Download TB Chest X-Ray dataset from Kaggle
========================================================================
Uses kagglehub (preferred) or kaggle CLI.

Usage:
    python src/download_dataset.py

Requirements (one of):
  Option A – kagglehub API token (recommended):
    Place ~/.kaggle/kaggle.json  containing:
        {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}

  Option B – environment variables:
    set KAGGLE_USERNAME=your_username
    set KAGGLE_KEY=your_api_key

Get your API key at: https://www.kaggle.com/settings  → "API" → "Create New Token"
"""

import os
import sys
import shutil
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
from utils import print_gpu_info, set_global_seed, SEED, DATA_ROOT

DATASET = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
EXPECTED_CLASSES = {"Normal": 3500, "Tuberculosis": 700}


def check_credentials() -> bool:
    """Return True if Kaggle credentials are discoverable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_with_kagglehub(data_root: str) -> str:
    """Download dataset using kagglehub and symlink into data/."""
    import kagglehub
    print(f"  Downloading via kagglehub: {DATASET}")
    path = kagglehub.dataset_download(DATASET)
    print(f"  kagglehub downloaded to: {path}")
    return path


def find_class_dirs(base: str) -> dict:
    """Recursively find Normal/ and Tuberculosis/ under base."""
    found = {}
    for cls in EXPECTED_CLASSES:
        for p in Path(base).rglob(cls):
            if p.is_dir():
                found[cls] = str(p)
                break
    return found


def copy_to_data_root(src_dirs: dict, data_root: str):
    """Copy/link class folders into data_root."""
    os.makedirs(data_root, exist_ok=True)
    for cls, src in src_dirs.items():
        dst = os.path.join(data_root, cls)
        if os.path.exists(dst):
            print(f"  {cls} already exists at {dst}, skipping copy.")
            continue
        print(f"  Copying {cls}: {src} → {dst}")
        shutil.copytree(src, dst)
        count = len(list(Path(dst).glob("*")))
        print(f"    Copied {count} files.")


def verify(data_root: str):
    """Count files and warn if counts are unexpected."""
    print("\n  Verification:")
    all_ok = True
    for cls, expected in EXPECTED_CLASSES.items():
        dpath = os.path.join(data_root, cls)
        if not os.path.isdir(dpath):
            print(f"  ❌ Missing: {dpath}")
            all_ok = False
            continue
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        count = sum(1 for f in Path(dpath).iterdir() if f.suffix.lower() in exts)
        status = "✅" if count == expected else f"⚠️  (expected {expected})"
        print(f"  {status} {cls}: {count} images")
    return all_ok


def main():
    set_global_seed(SEED)
    print_gpu_info()

    # Check if dataset already present
    existing = find_class_dirs(DATA_ROOT)
    if len(existing) == len(EXPECTED_CLASSES):
        print(f"\n✅ Dataset already present in {DATA_ROOT}")
        verify(DATA_ROOT)
        return

    # Check credentials
    if not check_credentials():
        print("\n" + "=" * 70)
        print("❌  KAGGLE CREDENTIALS NOT FOUND")
        print("=" * 70)
        print("""
To download the dataset, you need a Kaggle API token.

STEP-BY-STEP INSTRUCTIONS:
──────────────────────────
1. Go to https://www.kaggle.com  and log in (create account if needed).

2. Click your profile picture (top right) → Settings

3. Scroll to the "API" section → click "Create New Token"
   This downloads a file named  kaggle.json

4. Move that file to:
     Windows:  C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json
   (Create the .kaggle folder if it doesn't exist)

5. The file should look like:
     {"username":"your_kaggle_username","key":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

6. Re-run this script:
     python src/download_dataset.py

ALTERNATIVE – Manual Download:
───────────────────────────────
1. Go to: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
2. Click "Download" → download the ZIP
3. Extract it so you get:
     litetbnet/data/Normal/       (3500 JPEG files)
     litetbnet/data/Tuberculosis/ (700 JPEG files)
4. Run the audit:
     python src/audit.py
""")
        sys.exit(1)

    # Download
    print(f"\n  Credentials found. Downloading {DATASET} …")
    try:
        dl_path = download_with_kagglehub(DATA_ROOT)
        src_dirs = find_class_dirs(dl_path)
        if not src_dirs:
            print(f"  ❌ Could not find Normal/ or Tuberculosis/ under {dl_path}")
            sys.exit(1)
        copy_to_data_root(src_dirs, DATA_ROOT)
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        sys.exit(1)

    verify(DATA_ROOT)
    print("\n✅ Dataset ready. Run: python src/audit.py")


if __name__ == "__main__":
    main()

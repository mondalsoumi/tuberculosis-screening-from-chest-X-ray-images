"""
Shared utilities for LiteTBNet project.
Sets global seed=42 and provides GPU info printer used by all scripts.
"""

import os
import random
import numpy as np
import torch


SEED = 42
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TB_Chest_Radiography_Database")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
CLASSES = ["Normal", "Tuberculosis"]


def set_global_seed(seed: int = SEED) -> None:
    """Set seed for Python, NumPy, PyTorch (CPU+GPU) and make cuDNN deterministic."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_gpu_info() -> None:
    """Print GPU/CPU info at script start (called by every script)."""
    print("=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)
    print(f"  PyTorch version : {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  CUDA available  : True  ({n} device(s))")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name}  "
                  f"({props.total_memory // 1024**2} MB)")
    else:
        print("  CUDA available  : False  (running on CPU)")
    print(f"  Global seed     : {SEED}")
    print("=" * 60)


def get_class_dirs(data_root: str = DATA_ROOT):
    """Return dict {class_name: dir_path} for existing class directories."""
    dirs = {}
    for cls in CLASSES:
        p = os.path.join(data_root, cls)
        if os.path.isdir(p):
            dirs[cls] = p
    return dirs

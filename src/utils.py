"""Utility functions for logging, reproducibility, and dataset checks."""

import logging
import os
import random
import numpy as np
import torch

from .config import CFG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOG = logging.getLogger("ml-pipeline")


def labels_with_pred1(out):
    """Return label names where prediction bit equals 1."""
    return [lbl for lbl, bit in zip(out["labels"], out["preds"]) if int(bit) == 1]


def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dataset_exists():
    """Verify that the configured dataset file exists."""
    if not os.path.exists(CFG.data_csv):
        raise FileNotFoundError(f"Dataset not found: {CFG.data_csv}")

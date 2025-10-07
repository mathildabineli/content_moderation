import os
import random
import logging
import numpy as np
import torch
from .config import CFG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOG = logging.getLogger("ml-pipeline")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dataset_exists():
    if not os.path.exists(CFG.data_csv):
        raise FileNotFoundError(f"Dataset not found: {CFG.data_csv}")

"""Global configuration module for the Content Moderation API.

Defines immutable dataclass-based settings for model training,
API behavior, and filesystem paths.
"""
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class RunFlags:
    """Runtime flags controlling training and inference modes."""
    train_teacher: bool = True
    export_teacher_onnx: bool = True
    infer_demo: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration for model training and storage."""

    data_csv: str = "data/merged_final.csv"
    out_dir: str = "artifacts"
    teacher_model: str = "microsoft/mdeberta-v3-base"
    max_len: int = 192
    seed: int = 42

    # Teacher hyperparameters
    t_epochs: int = 1
    t_lr: float = 3e-5
    t_wd: float = 0.01
    t_bsz_train: int = 16
    t_bsz_eval: int = 32
    gamma_focal: float = 2.0
    early_stop: int = 2

    # Risk constraints
    high_risk: tuple[str, ...] = (
        "harassment",
        "violence",
        "sexual",
        "exploitation",
        "harm",
        "illicit",
    )
    recall_floor: float = 0.90

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# NEW: API-related config kept with dataclasses too
@dataclass(frozen=True)
class ApiConfig:
    """Configuration for API-specific artifacts and request limits."""
    artifacts_subdir: str = "teacher/best"
    onnx_filename: str = "teacher_fp32.onnx"
    metadata_filename: str = "metadata.json"
    thresholds_filename: str = "teacher_thresholds.json"

    # Request payload limits
    max_texts: int = 64
    max_text_len: int = 10_000


CFG = AppConfig()
API = ApiConfig()
FLAGS = RunFlags()

# Ensure base dirs exist
Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

# ---------- Derived paths the API will import ----------
ARTIFACTS_DIR = Path(CFG.out_dir) / API.artifacts_subdir
ONNX_PATH = ARTIFACTS_DIR / API.onnx_filename
TOKENIZER_PATH = ARTIFACTS_DIR                     # tokenizer lives in the same dir
METADATA_PATH = ARTIFACTS_DIR / API.metadata_filename
THRESHOLDS_PATH = ARTIFACTS_DIR / API.thresholds_filename

# Limits the API needs
MAX_TEXTS = API.max_texts
MAX_TEXT_LEN = API.max_text_len

from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class RunFlags:
    train_teacher: bool = True
    export_teacher_onnx: bool = True
    infer_demo: bool = True

@dataclass(frozen=True)
class AppConfig:
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
    high_risk: tuple[str, ...] = ("harassment", "violence", "sexual", "exploitation", "harm", "illicit")
    recall_floor: float = 0.90

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = AppConfig()
FLAGS = RunFlags()

# Ensure base dirs exist
Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

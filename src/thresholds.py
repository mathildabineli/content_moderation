from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve

from .config import CFG


def _per_label_thresholds(
    y_true_col: np.ndarray, y_prob_col: np.ndarray, floor: float
) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true_col.astype(int), y_prob_col
    )
    if thresholds.size == 0:
        return 0.5, 0.5
    p1, r1 = precision[1:], recall[1:]
    f1 = np.divide(2 * p1 * r1, (p1 + r1), out=np.zeros_like(p1), where=(p1 + r1) != 0)
    best_idx = int(np.argmax(f1))
    best_f1_thr = float(thresholds[best_idx])
    ok = np.where(r1 >= floor)[0]
    floor_idx = int(ok[np.argmax(f1[ok])]) if ok.size else int(np.argmax(r1))
    floor_thr = float(thresholds[floor_idx])
    return best_f1_thr, floor_thr


def fit_thresholds_with_floor(
    probs: np.ndarray,
    gold: np.ndarray,
    labels: List[str],
    floor: float = 0.90,
    high_risk: Iterable[str] | None = None,
) -> Dict[str, float]:
    risky = set(CFG.high_risk if high_risk is None else high_risk)
    cols = range(len(labels))
    pairs = list(
        map(lambda j: _per_label_thresholds(gold[:, j], probs[:, j], floor), cols)
    )
    chosen = list(
        map(lambda j: pairs[j][1] if labels[j] in risky else pairs[j][0], cols)
    )
    return dict(zip(labels, chosen))

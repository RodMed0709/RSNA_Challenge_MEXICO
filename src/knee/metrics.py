import numpy as np
from sklearn.metrics import roc_auc_score

from .config import LABELS


def mean_auc(targets, scores, labels=None):
    targets = np.asarray(targets, dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = labels or LABELS
    per_label = {}
    for index, label in enumerate(labels):
        valid = np.isfinite(targets[:, index]) & np.isfinite(scores[:, index])
        valid &= np.isin(targets[:, index], (0.0, 1.0))
        y_true = targets[valid, index]
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            per_label[label] = float("nan")
        else:
            per_label[label] = float(roc_auc_score(y_true, scores[valid, index]))
    valid_aucs = [value for value in per_label.values() if np.isfinite(value)]
    return {"mean": float(np.mean(valid_aucs)) if valid_aucs else float("nan"), "per_label": per_label}

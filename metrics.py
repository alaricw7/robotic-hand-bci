import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    """Return acc, kappa, per-class recall (shape (n_classes,))."""
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    row_sums = cm.sum(axis=1).clip(min=1)
    per_class = (cm.diagonal() / row_sums).astype(float)
    return acc, kappa, per_class

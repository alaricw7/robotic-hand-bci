import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/home/wong/eeg/preprocessing/pythondata1"
SUBJECTS = [f"S{i}" for i in range(1, 11)]
SFREQ = 250
TMIN = -0.5
MI_START_SEC = 0.0
MI_END_SEC = 4.0


def _npz_path(subject: str) -> str:
    return os.path.join(DATA_ROOT, subject, f"{subject}_EEGNet_NoICA_uV.npz")


def load_subject(subject: str, crop_mi: bool = True):
    """Return (X, y) for one subject in original chronological order.

    X: float32 (n_trials, n_channels, n_times)
    y: int64   (n_trials,)
    """
    d = np.load(_npz_path(subject), allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    if crop_mi:
        start = int(round((MI_START_SEC - TMIN) * SFREQ))
        end = int(round((MI_END_SEC - TMIN) * SFREQ))
        X = X[:, :, start:end]
    return X, y


def load_all(subjects=SUBJECTS, crop_mi: bool = True):
    return {s: load_subject(s, crop_mi=crop_mi) for s in subjects}


def chronological_per_class_val_split(y: np.ndarray, val_size: float = 0.1):
    """Within a single subject's trials (assumed chronological), for each class
    take the LAST ``val_size`` fraction as val, the rest as train.

    Returns (train_idx, val_idx) — both sorted index arrays into the original
    trial axis.
    """
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]  # ascending — chronological within class
        n_val = max(1, int(round(len(idx_c) * val_size)))
        if n_val >= len(idx_c):
            n_val = len(idx_c) - 1
        train_idx.extend(idx_c[:-n_val].tolist())
        val_idx.extend(idx_c[-n_val:].tolist())
    return np.sort(np.asarray(train_idx, dtype=np.int64)), \
           np.sort(np.asarray(val_idx, dtype=np.int64))


def chronological_per_class_train_val_test_split(y: np.ndarray, test_size: float, val_size: float):
    """Per-class chronological split: last ``test_size`` -> test, the previous
    ``val_size`` (of the full per-class count) -> val, rest -> train.
    """
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        n = len(idx_c)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 10)
        test_idx.extend(idx_c[-n_test:].tolist())
        val_idx.extend(idx_c[-(n_test + n_val):-n_test].tolist())
        train_idx.extend(idx_c[:-(n_test + n_val)].tolist())
    return (np.sort(np.asarray(train_idx, dtype=np.int64)),
            np.sort(np.asarray(val_idx, dtype=np.int64)),
            np.sort(np.asarray(test_idx, dtype=np.int64)))


def stratified_kfold_train_val_test_splits(
    y: np.ndarray,
    n_splits: int = 5,
    val_size: float = 0.1,
    seed: int = 42,
):
    """Yield stratified train/val/test index splits for subject-dependent CV.

    The outer fold supplies TEST. VAL is a stratified random split from the
    remaining train_val trials, so no test trial contributes to validation.
    """
    indices = np.arange(len(y), dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, y), start=1):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=y[train_val_idx],
            random_state=seed + fold_idx,
        )
        yield (
            fold_idx,
            np.sort(train_idx.astype(np.int64)),
            np.sort(val_idx.astype(np.int64)),
            np.sort(test_idx.astype(np.int64)),
        )


def standardize_per_channel(X_train: np.ndarray, *X_eval_sets):
    """Z-score per channel using train statistics only."""
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train_std = (X_train - mean) / std
    others = tuple((X - mean) / std for X in X_eval_sets)
    return (X_train_std,) + others


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size: int, shuffle: bool, num_workers: int = 2):
    return DataLoader(
        EEGDataset(X, y), batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

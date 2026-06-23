"""Data loading + CV utilities shared by all stages.

Reuses the same loader/protocol that produced the 0.3444 / 0.5783 anchors
(see selfmodel/test/probe_utils.py). Keep this file the single point of truth
for paths / subjects / CV split so all stages stay comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = Path("/home/wong/eeg/preprocessing/pythondata1")
SUBJECTS = [f"S{i}" for i in range(1, 11)]
SEED = 42
WINDOW_START = 0.0
WINDOW_END = 2.0
TIME_DECIMATION = 2  # 250 Hz -> 125 Hz, matches the anchor pipeline
N_FOLDS = 5
N_CLASSES = 6
CHANCE = 1.0 / N_CLASSES


def subject_number(subject: str) -> int:
    return int(subject.upper().removeprefix("S"))


def normalize_subjects(subjects):
    return sorted(
        [f"S{subject_number(s)}" for s in subjects], key=subject_number
    )


def load_subject(subject: str, data_root: Path = DATA_ROOT) -> dict:
    path = Path(data_root) / subject / f"{subject}_EEGNet_NoICA_uV.npz"
    with np.load(path, allow_pickle=True) as data:
        return {
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "times": data["times"].astype(np.float64),
            "sfreq": float(data["sfreq"]),
            "ch_names": [str(v) for v in data["ch_names"]],
        }


def crop(X: np.ndarray, times: np.ndarray, start: float, end: float):
    mask = (times >= start) & (times <= end)
    return X[..., mask], times[mask]


def channel_standardize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )


def prepare_subject(subject: str, data_root: Path = DATA_ROOT):
    """Load -> crop to [0, 2] s -> decimate to 125 Hz.

    Returns X[trials, channels, time], y[trials], sfreq_effective, ch_names.
    """
    data = load_subject(subject, data_root)
    X, _ = crop(data["X"], data["times"], WINDOW_START, WINDOW_END)
    X = X[..., ::TIME_DECIMATION]
    sfreq_effective = data["sfreq"] / TIME_DECIMATION
    return X.astype(np.float32), data["y"], sfreq_effective, data["ch_names"]


def stratified_folds(
    X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y):
        # Hard guard against leakage.
        assert np.intersect1d(train_idx, test_idx).size == 0
        yield train_idx, test_idx

#!/usr/bin/env python3
"""
Multi-feature RSA for EEGNet representations, comparing NoICA vs ICA.

For each subject and condition:
  - Recreate StratifiedKFold(k=5, seed=42), use fold 0 test samples only.
  - Build Model-RDM from EEGNet penultimate embeddings.
  - Build Neuro-RDMs from ERP, band power, spatial variance, and z-scored concat.
  - Run Spearman RSA with class-label permutation tests.

Outputs are written under artifacts/analysis/representation/rsa by default.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from mne.time_frequency import psd_array_multitaper
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, wilcoxon
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from robotic_hand_bci.project import PROJECT_ROOT

DEFAULT_OUT_DIR = str(PROJECT_ROOT / "artifacts" / "analysis" / "representation" / "rsa")
DEFAULT_MONTAGE = str(
    PROJECT_ROOT / "assets" / "montages" / "Standard-10-5-Cap385_witheog.elp"
)
DATA_TEMPLATES = {
    "noica": str(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pythondata1"
        / "{subject}"
        / "{subject}_EEGNet_NoICA_uV.npz"
    ),
    "ica": str(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pythondata1_ICA"
        / "{subject}"
        / "{subject}_EEGNet_ICA_uV.npz"
    ),
}
CKPT_TEMPLATES = {
    "noica": (
        str(PROJECT_ROOT / "model" / "10fold_npz" / "experiments")
        + "/pythondata1_npz_repr/representation/checkpoints/{subject}_eegnet_noica.pt"
    ),
    "ica": (
        str(PROJECT_ROOT / "model" / "10fold_npz" / "experiments")
        + "/pythondata1_ica_npz_repr/representation/checkpoints/{subject}_eegnet_ica.pt"
    ),
}

EVENT_CODES = [11, 12, 13, 14, 15, 16]
CLASS_NAMES = [
    "11 manipulation",
    "12 color class",
    "13 color tracking",
    "14 face",
    "15 RPS",
    "16 gesture",
]
FAMILY_BY_CLASS = {
    0: "MOTOR",
    1: "VISUAL",
    2: "VISUAL",
    3: "VISUAL",
    4: "MOTOR",
    5: "VISUAL",
}
FAMILY_MARKERS = {"VISUAL": "o", "MOTOR": "s"}
FAMILY_COLORS = {"VISUAL": "#0072B2", "MOTOR": "#D55E00"}
ERP_CHANNELS = [
    "O1",
    "Oz",
    "O2",
    "POz",
    "PO7",
    "PO8",
    "P7",
    "P8",
    "C3",
    "Cz",
    "C4",
    "CP3",
    "CP4",
]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
FEATURES = ["A", "B", "C", "ALL"]
CONDS = ["noica", "ica"]


class EEGNetBaseline(nn.Module):
    """Local fallback matching the repository EEGNet baseline."""

    def __init__(
        self,
        n_classes: int = 6,
        n_channels: int = 59,
        n_samples: int = 1126,
        n_temporal_filters: int = 16,
        temporal_kernel: int = 64,
        depth_multiplier: int = 2,
        separable_kernel: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        f1 = n_temporal_filters
        f2 = f1 * depth_multiplier
        self.temporal_conv = nn.Conv2d(
            1,
            f1,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial_conv = nn.Conv2d(
            f1,
            f2,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_depthwise = nn.Conv2d(
            f2,
            f2,
            kernel_size=(1, separable_kernel),
            padding=(0, separable_kernel // 2),
            groups=f2,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat_dim = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        in_t = x.shape[-1]
        x = self.temporal_conv(x)
        if x.shape[-1] > in_t:
            x = x[..., :in_t]
        x = self.bn1(x)
        x = self.bn2(self.spatial_conv(x))
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        pre_sep_t = x.shape[-1]
        x = self.sep_depthwise(x)
        if x.shape[-1] > pre_sep_t:
            x = x[..., :pre_sep_t]
        x = self.bn3(self.sep_pointwise(x))
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self._forward_features(x))


EEGNet = EEGNetBaseline


@dataclass
class ConditionResult:
    subject: str
    condition: str
    rdm_model: np.ndarray
    rdm_neuro: Dict[str, np.ndarray]
    rsa: Dict[str, Dict[str, float]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-feature EEGNet RSA: ICA vs NoICA.")
    parser.add_argument("--subjects", nargs="+", default=[f"S{i}" for i in range(1, 11)])
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--montage", default=DEFAULT_MONTAGE)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--n-channels", type=int, default=59)
    parser.add_argument("--n-samples", type=int, default=1126)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--epoch-tmin", type=float, default=-0.5)
    parser.add_argument("--model-module", default=None)
    parser.add_argument("--model-class", default="EEGNet")
    parser.add_argument("--embedding-layer", default="classifier")
    parser.add_argument("--n-temporal-filters", type=int, default=16)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--temporal-kernel", type=int, default=64)
    parser.add_argument("--separable-kernel", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--min-trials-per-class", type=int, default=5)
    parser.add_argument("--psd-bandwidth", type=float, default=2.0)
    parser.add_argument("--no-channel-normalize", action="store_true")
    for cond in CONDS:
        parser.add_argument(f"--{cond}-data-template", default=DATA_TEMPLATES[cond])
        parser.add_argument(f"--{cond}-checkpoint-template", default=CKPT_TEMPLATES[cond])
    return parser.parse_args(argv)


def resolve_template(template: str, subject: str) -> Path:
    return Path(template.format(subject=subject, Subject=subject, si=subject, Si=subject))


def log_message(out_dir: Path, message: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")
    print(message)


def load_npz(npz_path: Path, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    ch_names = [str(ch) for ch in data["ch_names"].tolist()]
    if X.shape[1] != args.n_channels or X.shape[2] != args.n_samples:
        raise ValueError(
            f"{npz_path} has X shape {X.shape}; expected (*,{args.n_channels},{args.n_samples})"
        )
    if len(ch_names) != args.n_channels:
        raise ValueError(f"{npz_path} has {len(ch_names)} ch_names; expected {args.n_channels}")
    return X, y, ch_names


def get_fold_split(X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X, y))
    if args.fold < 0 or args.fold >= len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {args.fold}")
    return splits[args.fold]


def fit_channel_standardizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_channel_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def import_model_class(model_module: Optional[str], model_class: str):
    if not model_module:
        return EEGNetBaseline
    module = importlib.import_module(model_module)
    return getattr(module, model_class)


def build_model(args: argparse.Namespace) -> nn.Module:
    model_cls = import_model_class(args.model_module, args.model_class)
    attempts = [
        dict(
            n_classes=args.n_classes,
            n_channels=args.n_channels,
            n_samples=args.n_samples,
            n_temporal_filters=args.n_temporal_filters,
            depth_multiplier=args.depth_multiplier,
            temporal_kernel=args.temporal_kernel,
            separable_kernel=args.separable_kernel,
            dropout=args.dropout,
        ),
        dict(n_classes=args.n_classes, n_chans=args.n_channels, n_times=args.n_samples),
        dict(n_outputs=args.n_classes, n_chans=args.n_channels, n_times=args.n_samples),
        dict(),
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return model_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise TypeError(f"Could not instantiate {model_cls}") from last_error


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "state"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not look like a state_dict container.")


def strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(strip_common_prefixes(extract_state_dict(checkpoint)), strict=True)


def named_module(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        available = ", ".join(key for key in modules if key)
        raise KeyError(f"Module {name!r} not found. Available modules: {available}")
    return modules[name]


def infer_embeddings(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    embedding_layer: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    captured: List[np.ndarray] = []

    def hook(_module, module_inputs, _module_output):
        captured.append(module_inputs[0].detach().cpu().numpy())

    handle = named_module(model, embedding_layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
                _ = model(xb)
    finally:
        handle.remove()
    return np.concatenate(captured, axis=0)


def sample_indices(start_sec: float, end_sec: float, args: argparse.Namespace) -> Tuple[int, int]:
    start = int(round((start_sec - args.epoch_tmin) * args.sfreq))
    end = int(round((end_sec - args.epoch_tmin) * args.sfreq))
    start = int(np.clip(start, 0, args.n_samples))
    end = int(np.clip(end, start + 1, args.n_samples))
    return start, end


def read_montage_allow_count_header(montage_path: Path) -> mne.channels.DigMontage:
    try:
        return mne.channels.read_custom_montage(str(montage_path))
    except ValueError:
        lines = montage_path.read_text(encoding="utf-8").splitlines()
        if not lines or not lines[0].strip().isdigit():
            raise
        with tempfile.NamedTemporaryFile("w", suffix=montage_path.suffix, delete=False) as tmp:
            tmp.write("\n".join(lines[1:]))
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        try:
            return mne.channels.read_custom_montage(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)


def resolve_channel_indices(
    requested: Sequence[str], ch_names: Sequence[str], montage_path: Path
) -> Tuple[List[int], Dict[str, str]]:
    by_upper = {ch.upper(): idx for idx, ch in enumerate(ch_names)}
    mapping: Dict[str, str] = {}
    indices: List[int] = []
    montage = None
    pos_by_upper = None
    available_positions = None

    for req in requested:
        key = req.upper()
        if key in by_upper:
            indices.append(by_upper[key])
            mapping[req] = ch_names[by_upper[key]]
            continue

        if montage is None:
            montage = read_montage_allow_count_header(montage_path)
            pos_by_upper = {name.upper(): pos for name, pos in montage.get_positions()["ch_pos"].items()}
            available_positions = [
                (ch, by_upper[ch.upper()], pos_by_upper[ch.upper()])
                for ch in ch_names
                if ch.upper() in pos_by_upper
            ]
        if pos_by_upper is None or key not in pos_by_upper:
            raise ValueError(f"Requested channel {req} not found and has no montage position.")
        target = pos_by_upper[key]
        nearest_ch, nearest_idx, _ = min(
            available_positions,
            key=lambda item: float(np.linalg.norm(item[2] - target)),
        )
        indices.append(nearest_idx)
        mapping[req] = nearest_ch

    return indices, mapping


def erp_features(X: np.ndarray, ch_names: Sequence[str], args: argparse.Namespace) -> np.ndarray:
    ch_idx, _mapping = resolve_channel_indices(ERP_CHANNELS, ch_names, Path(args.montage))
    start, end = sample_indices(0.0, 1.0, args)
    edges = np.linspace(start, end, 21).round().astype(int)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(lo + 1, hi)
        parts.append(X[:, ch_idx, lo:hi].mean(axis=-1))
    return np.stack(parts, axis=-1).reshape(X.shape[0], -1).astype(np.float32)


def band_power_features(X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    start, end = sample_indices(0.5, 3.5, args)
    data = X[:, :, start:end]
    psd, freqs = psd_array_multitaper(
        data,
        sfreq=args.sfreq,
        fmin=0.5,
        fmax=30.0,
        bandwidth=args.psd_bandwidth,
        adaptive=False,
        normalization="full",
        verbose=False,
    )
    df = np.gradient(freqs)
    features = []
    for band_idx, (_name, (lo, hi)) in enumerate(BANDS.items()):
        if band_idx == len(BANDS) - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        power = np.sum(psd[:, :, mask] * df[mask], axis=-1)
        features.append(power)
    return np.stack(features, axis=-1).reshape(X.shape[0], -1).astype(np.float32)


def spatial_variance_features(X: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    start, end = sample_indices(0.0, 1.5, args)
    return np.var(X[:, :, start:end], axis=-1).astype(np.float32)


def zscore_features(features: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(features).astype(np.float32)


def class_prototype_rdm(features: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    prototypes = []
    for cls in range(n_classes):
        if np.sum(y == cls) == 0:
            prototypes.append(np.full(features.shape[1], np.nan, dtype=np.float64))
        else:
            prototypes.append(features[y == cls].mean(axis=0))
    prototypes_arr = np.vstack(prototypes)
    if np.isnan(prototypes_arr).any():
        rdm = np.full((n_classes, n_classes), np.nan, dtype=np.float64)
        valid = ~np.isnan(prototypes_arr).any(axis=1)
        rdm[np.ix_(valid, valid)] = squareform(pdist(prototypes_arr[valid], metric="cosine"))
        np.fill_diagonal(rdm, 0.0)
        return rdm
    rdm = squareform(pdist(prototypes_arr, metric="cosine"))
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float64)


def upper_tri(rdm: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(rdm, k=1)
    return rdm[idx]


def spearman_rdm(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    a = upper_tri(rdm_a)
    b = upper_tri(rdm_b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    stat = spearmanr(a[mask], b[mask])
    return float(stat.correlation)


def permutation_p(
    rdm_model: np.ndarray,
    rdm_neuro: np.ndarray,
    observed: float,
    n_perm: int,
    seed: int,
) -> float:
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(rdm_neuro.shape[0])
        permuted = rdm_neuro[np.ix_(perm, perm)]
        null.append(spearman_rdm(rdm_model, permuted))
    null_arr = np.asarray(null, dtype=np.float64)
    null_arr = null_arr[np.isfinite(null_arr)]
    if len(null_arr) == 0:
        return float("nan")
    return float((np.sum(np.abs(null_arr) >= abs(observed)) + 1) / (len(null_arr) + 1))


def plot_rdm(rdm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=160)
    sns.heatmap(
        rdm,
        ax=ax,
        cmap="mako",
        square=True,
        annot=True,
        fmt=".2f",
        xticklabels=[str(code) for code in EVENT_CODES],
        yticklabels=[str(code) for code in EVENT_CODES],
        cbar_kws={"shrink": 0.82},
    )
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def finite_json(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def stable_perm_seed(subject: str, condition: str, feature: str, base_seed: int) -> int:
    text = f"{subject}:{condition}:{feature}"
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text))
    return int(base_seed + offset)


def save_subject_rdms(
    out_dir: Path,
    subject: str,
    condition: str,
    rdm_model: np.ndarray,
    rdm_neuro: Dict[str, np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"rdm_model_{condition}.npy", rdm_model)
    plot_rdm(rdm_model, out_dir / f"rdm_model_{condition}.png", f"{subject} model RDM ({condition})")
    for feature, rdm in rdm_neuro.items():
        np.save(out_dir / f"rdm_neuro_{feature}_{condition}.npy", rdm)
        plot_rdm(
            rdm,
            out_dir / f"rdm_neuro_{feature}_{condition}.png",
            f"{subject} neuro RDM {feature} ({condition})",
        )


def process_condition(
    subject: str,
    condition: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Optional[ConditionResult]:
    data_template = getattr(args, f"{condition}_data_template")
    ckpt_template = getattr(args, f"{condition}_checkpoint_template")
    npz_path = resolve_template(data_template, subject)
    ckpt_path = resolve_template(ckpt_template, subject)

    try:
        X, y, ch_names = load_npz(npz_path, args)
        train_idx, test_idx = get_fold_split(X, y, args)
        counts = np.bincount(y[test_idx], minlength=args.n_classes)
        if np.any(counts < args.min_trials_per_class):
            log_message(
                out_dir,
                f"[SKIP] {subject} {condition}: test class counts {counts.tolist()} below min {args.min_trials_per_class}",
            )
            return None
        X_train, X_test, y_test = X[train_idx], X[test_idx], y[test_idx]
        if not args.no_channel_normalize:
            mean, std = fit_channel_standardizer(X_train)
            X_test_model = apply_channel_standardize(X_test, mean, std)
        else:
            X_test_model = X_test

        model = build_model(args)
        device = torch.device(args.device)
        load_checkpoint(model, ckpt_path, device)
        embeddings = infer_embeddings(
            model, X_test_model, device, args.batch_size, args.embedding_layer
        )
        rdm_model = class_prototype_rdm(embeddings, y_test, args.n_classes)

        feat_a = erp_features(X_test, ch_names, args)
        feat_b = band_power_features(X_test, args)
        feat_c = spatial_variance_features(X_test, args)
        feat_all = np.concatenate(
            [zscore_features(feat_a), zscore_features(feat_b), zscore_features(feat_c)], axis=1
        )
        neuro_features = {"A": feat_a, "B": feat_b, "C": feat_c, "ALL": feat_all}
        rdm_neuro = {
            name: class_prototype_rdm(features, y_test, args.n_classes)
            for name, features in neuro_features.items()
        }
        rsa = {}
        for feature, rdm in rdm_neuro.items():
            r = spearman_rdm(rdm_model, rdm)
            p = permutation_p(
                rdm_model,
                rdm,
                r,
                args.n_perm,
                stable_perm_seed(subject, condition, feature, args.seed),
            )
            rsa[feature] = {"r": r, "p_perm": p}

        save_subject_rdms(out_dir / subject, subject, condition, rdm_model, rdm_neuro)
        return ConditionResult(subject, condition, rdm_model, rdm_neuro, rsa)
    except Exception as exc:
        log_message(out_dir, f"[SKIP] {subject} {condition}: {type(exc).__name__}: {exc}")
        return None


def write_subject_summary(subject_dir: Path, subject: str, results: Dict[str, ConditionResult]) -> None:
    payload = {}
    for condition, result in results.items():
        payload[condition] = {
            feature: {"r": finite_json(vals["r"]), "p_perm": finite_json(vals["p_perm"])}
            for feature, vals in result.rsa.items()
        }
    if payload:
        with (subject_dir / "rsa_summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def mean_rdm(results: Sequence[ConditionResult], rdm_getter) -> Optional[np.ndarray]:
    rdms = [rdm_getter(result) for result in results]
    if not rdms:
        return None
    return np.nanmean(np.stack(rdms, axis=0), axis=0)


def group_rsa_and_rdms(
    all_results: Dict[Tuple[str, str], ConditionResult], args: argparse.Namespace, out_dir: Path
) -> Dict[str, Dict[str, Dict[str, float]]]:
    group_dir = out_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    group_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    group_rdms: Dict[Tuple[str, str], np.ndarray] = {}

    for condition in CONDS:
        cond_results = [res for (sub, cond), res in all_results.items() if cond == condition]
        model_rdm = mean_rdm(cond_results, lambda res: res.rdm_model)
        if model_rdm is None:
            continue
        group_rdms[(condition, "model")] = model_rdm
        np.save(group_dir / f"group_rdm_model_{condition}.npy", model_rdm)
        plot_rdm(model_rdm, group_dir / f"group_rdm_model_{condition}.png", f"Group model RDM ({condition})")
        group_json[condition] = {}
        for feature in FEATURES:
            neuro_rdm = mean_rdm(cond_results, lambda res, feat=feature: res.rdm_neuro[feat])
            if neuro_rdm is None:
                continue
            group_rdms[(condition, feature)] = neuro_rdm
            np.save(group_dir / f"group_rdm_neuro_{feature}_{condition}.npy", neuro_rdm)
            plot_rdm(
                neuro_rdm,
                group_dir / f"group_rdm_neuro_{feature}_{condition}.png",
                f"Group neuro RDM {feature} ({condition})",
            )
            r = spearman_rdm(model_rdm, neuro_rdm)
            p = permutation_p(model_rdm, neuro_rdm, r, args.n_perm, args.seed + 991 * (FEATURES.index(feature) + 1))
            group_json[condition][feature] = {"r": finite_json(r), "p_perm": finite_json(p)}

    with (group_dir / "group_rsa.json").open("w", encoding="utf-8") as f:
        json.dump(group_json, f, indent=2, ensure_ascii=False)
    plot_mds_comparison(group_rdms, group_dir / "mds_comparison.png")
    return group_json


def paired_ica_vs_noica(all_results: Dict[Tuple[str, str], ConditionResult], out_dir: Path) -> pd.DataFrame:
    rows = []
    subjects = sorted({sub for sub, _cond in all_results})
    p_by_feature: Dict[str, float] = {}
    for feature in FEATURES:
        deltas = []
        for subject in subjects:
            noica = all_results.get((subject, "noica"))
            ica = all_results.get((subject, "ica"))
            if noica is None or ica is None:
                continue
            deltas.append(ica.rsa[feature]["r"] - noica.rsa[feature]["r"])
        valid = np.asarray([d for d in deltas if np.isfinite(d)], dtype=np.float64)
        if len(valid) >= 2 and np.any(np.abs(valid) > 0):
            p_by_feature[feature] = float(wilcoxon(valid).pvalue)
        else:
            p_by_feature[feature] = float("nan")

    for subject in subjects:
        row = {"subject": subject}
        noica = all_results.get((subject, "noica"))
        ica = all_results.get((subject, "ica"))
        for feature in FEATURES:
            if noica is not None and ica is not None:
                delta = ica.rsa[feature]["r"] - noica.rsa[feature]["r"]
            else:
                delta = float("nan")
            row[f"delta_r_{feature}"] = delta
            row[f"p_paired_{feature}"] = p_by_feature[feature]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "group" / "ica_vs_noica.csv", index=False)
    return df


def plot_mds_panel(ax, rdm: np.ndarray, title: str) -> None:
    try:
        kwargs = dict(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
        if "normalized_stress" in inspect.signature(MDS).parameters:
            kwargs["normalized_stress"] = "auto"
        xy = MDS(**kwargs).fit_transform(rdm)
    except Exception as exc:
        ax.text(0.5, 0.5, f"MDS failed\n{exc}", ha="center", va="center")
        ax.set_axis_off()
        return
    for cls, code in enumerate(EVENT_CODES):
        family = FAMILY_BY_CLASS[cls]
        ax.scatter(
            xy[cls, 0],
            xy[cls, 1],
            s=70,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.7,
        )
        ax.text(xy[cls, 0], xy[cls, 1], str(code), fontsize=9, ha="center", va="center")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color="0.85", linewidth=0.6)
    ax.axvline(0, color="0.85", linewidth=0.6)


def plot_mds_comparison(group_rdms: Dict[Tuple[str, str], np.ndarray], out_path: Path) -> None:
    available_conds = [cond for cond in CONDS if (cond, "model") in group_rdms]
    if not available_conds:
        return
    n_rows = len(available_conds)
    labels = ["model", "A", "B", "C", "ALL"]
    fig, axes = plt.subplots(n_rows, len(labels), figsize=(3.0 * len(labels), 3.0 * n_rows), dpi=160)
    axes_arr = np.asarray(axes).reshape(n_rows, len(labels))
    for r, condition in enumerate(available_conds):
        for c, label in enumerate(labels):
            ax = axes_arr[r, c]
            rdm = group_rdms.get((condition, label))
            if rdm is None:
                ax.set_axis_off()
                continue
            plot_mds_panel(ax, rdm, f"{condition} {label}")
    handles = [
        plt.Line2D([0], [0], marker=FAMILY_MARKERS[fam], color="w", label=fam,
                   markerfacecolor=FAMILY_COLORS[fam], markersize=8)
        for fam in ("VISUAL", "MOTOR")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"
    if log_path.exists():
        log_path.unlink()

    all_results: Dict[Tuple[str, str], ConditionResult] = {}
    all_rows = []
    for subject in args.subjects:
        subject_results: Dict[str, ConditionResult] = {}
        for condition in CONDS:
            print(f"[INFO] Processing {subject} {condition}")
            result = process_condition(subject, condition, args, out_dir)
            if result is None:
                continue
            all_results[(subject, condition)] = result
            subject_results[condition] = result
            for feature in FEATURES:
                all_rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "feature": feature,
                        "r": result.rsa[feature]["r"],
                        "p_perm": result.rsa[feature]["p_perm"],
                    }
                )
        write_subject_summary(out_dir / subject, subject, subject_results)

    pd.DataFrame(all_rows).to_csv(out_dir / "all_subjects_rsa.csv", index=False)
    group_rsa_and_rdms(all_results, args, out_dir)
    paired_ica_vs_noica(all_results, out_dir)
    print(f"[INFO] Saved RSA outputs under {out_dir}")


if __name__ == "__main__":
    main()

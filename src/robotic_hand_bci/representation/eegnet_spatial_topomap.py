#!/usr/bin/env python3
"""
Convert EEGNet spatial filters into class-conditional topomaps and score them
against task-specific neuroanatomical priors.

Default conventions:
  NPZ:        data/processed/pythondata2/{S}/...
  checkpoint: model/10fold_npz/experiments/pythondata2_npz_repr/representation/checkpoints/{S}_eegnet_ica.pt
  fold:       StratifiedKFold(k=5, seed=42), fold 0 test split only
  EEGNet:     parameters are inferred from checkpoint when possible.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tempfile
import warnings
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
from sklearn.model_selection import StratifiedKFold

from robotic_hand_bci.project import PROJECT_ROOT

DEFAULT_DATA_TEMPLATE = (
    str(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pythondata2"
        / "{subject}"
        / "{subject}_EEGNet_ICA_uV.npz"
    )
)
DEFAULT_CKPT_TEMPLATE = (
    str(PROJECT_ROOT / "model" / "10fold_npz" / "experiments")
    + "/pythondata2_npz_repr/representation/checkpoints/{subject}_eegnet_ica.pt"
)
DEFAULT_MONTAGE = str(
    PROJECT_ROOT / "assets" / "montages" / "Standard-10-5-Cap385_witheog.elp"
)
DEFAULT_OUT_DIR = str(
    PROJECT_ROOT / "artifacts" / "analysis" / "representation" / "spatial_topomap_pythondata2"
)

EVENT_CODES = [11, 12, 13, 14, 15, 16]
CLASS_NAMES = [
    "MOTOR manipulation",
    "VISUAL color classification",
    "ATTENTION color tracking",
    "FACE recognition",
    "RPS motor+visual",
    "BIO-MOTION gesture",
]
EXPECTED_CHANNELS = {
    0: ["C3", "C4", "CZ", "FC3", "FC4"],
    1: ["O1", "O2", "OZ", "POZ"],
    2: ["P3", "P4", "PO3", "PO4", "POZ"],
    3: ["P8", "PO8", "TP8", "P10"],
    4: ["C3", "C4", "CZ", "FC3", "FC4", "O1", "O2", "OZ", "POZ"],
    5: ["P7", "P8", "CP5", "CP6"],
}


class EEGNetBaseline(nn.Module):
    """Local EEGNet fallback; spatial_conv is the depthwise spatial layer."""

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
class SubjectOutput:
    subject: str
    ch_names: List[str]
    class_patterns: np.ndarray
    anatomical_rows: List[Dict[str, object]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEGNet spatial filter topomap and anatomical-prior scoring."
    )
    parser.add_argument("--subjects", nargs="+", default=[f"S{i}" for i in range(1, 11)])
    parser.add_argument("--data-template", default=DEFAULT_DATA_TEMPLATE)
    parser.add_argument("--checkpoint-template", default=DEFAULT_CKPT_TEMPLATE)
    parser.add_argument("--montage", default=DEFAULT_MONTAGE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--n-channels", type=int, default=59)
    parser.add_argument("--n-samples", type=int, default=1126)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--epoch-tmin", type=float, default=-0.5)
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-end", type=float, default=1.5)
    parser.add_argument("--block1-pool-factor", type=float, default=4.0)
    parser.add_argument("--n-temporal-filters", type=int, default=16)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--temporal-kernel", type=int, default=64)
    parser.add_argument("--separable-kernel", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--model-module", default=None)
    parser.add_argument("--model-class", default="EEGNet")
    parser.add_argument(
        "--spatial-layer",
        default="auto",
        help="Layer to use for spatial filters and activation hook; auto tries block1.depthwise then spatial_conv.",
    )
    parser.add_argument("--no-channel-normalize", action="store_true")
    return parser.parse_args(argv)


def resolve_template(template: str, subject: str) -> Path:
    return Path(template.format(subject=subject, Subject=subject, si=subject, Si=subject))


def load_npz(npz_path: Path, n_channels: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    ch_names = [str(ch) for ch in data["ch_names"].tolist()]
    if X.shape[1] != n_channels or X.shape[2] != n_samples:
        raise ValueError(f"{npz_path} has X shape {X.shape}; expected (*,{n_channels},{n_samples})")
    if len(ch_names) != n_channels:
        raise ValueError(f"{npz_path} has {len(ch_names)} ch_names; expected {n_channels}")
    return X, y, ch_names


def get_fold_split(
    X: np.ndarray, y: np.ndarray, n_splits: int, fold: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {fold}")
    return splits[fold]


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


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
    return strip_common_prefixes(extract_state_dict(checkpoint))


def infer_eegnet_hparams_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """Update local fallback EEGNet args from checkpoint tensor shapes."""
    temporal = state_dict.get("temporal_conv.weight")
    spatial = state_dict.get("spatial_conv.weight")
    sep_depthwise = state_dict.get("sep_depthwise.weight")

    inferred = []
    if temporal is not None and temporal.ndim == 4:
        n_temporal_filters = int(temporal.shape[0])
        temporal_kernel = int(temporal.shape[-1])
        if args.n_temporal_filters != n_temporal_filters:
            inferred.append(f"F1 {args.n_temporal_filters}->{n_temporal_filters}")
        if args.temporal_kernel != temporal_kernel:
            inferred.append(f"temporal_kernel {args.temporal_kernel}->{temporal_kernel}")
        args.n_temporal_filters = n_temporal_filters
        args.temporal_kernel = temporal_kernel

    if spatial is not None and spatial.ndim == 4:
        spatial_filters = int(spatial.shape[0])
        n_channels = int(spatial.shape[2])
        if args.n_channels != n_channels:
            inferred.append(f"n_channels {args.n_channels}->{n_channels}")
        args.n_channels = n_channels
        if args.n_temporal_filters > 0 and spatial_filters % args.n_temporal_filters == 0:
            depth_multiplier = spatial_filters // args.n_temporal_filters
            if args.depth_multiplier != depth_multiplier:
                inferred.append(f"D {args.depth_multiplier}->{depth_multiplier}")
            args.depth_multiplier = depth_multiplier

    if sep_depthwise is not None and sep_depthwise.ndim == 4:
        separable_kernel = int(sep_depthwise.shape[-1])
        if args.separable_kernel != separable_kernel:
            inferred.append(f"separable_kernel {args.separable_kernel}->{separable_kernel}")
        args.separable_kernel = separable_kernel

    if inferred:
        print(f"[INFO] Inferred EEGNet checkpoint architecture: {', '.join(inferred)}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    if state_dict is None:
        state_dict = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)


def named_module(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        available = ", ".join(key for key in modules if key)
        raise KeyError(f"Module {name!r} not found. Available modules: {available}")
    return modules[name]


def resolve_spatial_layer_name(model: nn.Module, requested: str) -> str:
    if requested != "auto":
        named_module(model, requested)
        return requested
    modules = dict(model.named_modules())
    for candidate in ("block1.depthwise", "spatial_conv"):
        if candidate in modules:
            return candidate
    for name, module in modules.items():
        if "depthwise" in name.lower() and isinstance(module, nn.Conv2d):
            return name
    raise KeyError("Could not find block1.depthwise, spatial_conv, or any depthwise Conv2d.")


def spatial_filters_from_layer(layer: nn.Module, n_channels: int) -> np.ndarray:
    if not hasattr(layer, "weight"):
        raise TypeError(f"Layer {layer} has no weight.")
    weight = layer.weight.detach().cpu().numpy()
    n_filters = weight.shape[0]
    filters = weight.reshape(n_filters, -1)
    if filters.shape[1] != n_channels:
        squeezed = np.squeeze(weight)
        filters = squeezed.reshape(n_filters, -1)
    if filters.shape[1] != n_channels:
        raise ValueError(
            f"Cannot reshape spatial layer weight {weight.shape} to (n_filters, {n_channels})."
        )
    return filters.astype(np.float64)


def infer_spatial_activations(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    layer_name: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    captured: List[np.ndarray] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().cpu().numpy())

    handle = named_module(model, layer_name).register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
                _ = model(xb)
    finally:
        handle.remove()

    acts = np.concatenate(captured, axis=0)
    acts = np.squeeze(acts, axis=2) if acts.ndim == 4 and acts.shape[2] == 1 else acts
    if acts.ndim != 3:
        raise ValueError(f"Expected activations shaped (N, filters, time), got {acts.shape}")
    return acts


def epoch_to_block1_idx(
    time_sec: float,
    epoch_tmin: float,
    sfreq: float,
    pool_factor: float,
    n_time: int,
) -> int:
    epoch_idx = int(round((time_sec - epoch_tmin) * sfreq))
    block_idx = int(round(epoch_idx / pool_factor))
    return int(np.clip(block_idx, 0, n_time))


def activation_window_indices(acts: np.ndarray, args: argparse.Namespace) -> Tuple[int, int]:
    n_time = acts.shape[-1]
    start = epoch_to_block1_idx(
        args.window_start, args.epoch_tmin, args.sfreq, args.block1_pool_factor, n_time
    )
    end = epoch_to_block1_idx(
        args.window_end, args.epoch_tmin, args.sfreq, args.block1_pool_factor, n_time
    )
    end = max(start + 1, end)
    return start, min(end, n_time)


def class_conditional_patterns(
    spatial_filters: np.ndarray,
    activations: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    start, end = activation_window_indices(activations, args)
    energy = np.mean(activations[:, :, start:end] ** 2, axis=-1)
    patterns = []
    for cls in range(args.n_classes):
        in_cls = y == cls
        if not np.any(in_cls) or not np.any(~in_cls):
            score = np.zeros(spatial_filters.shape[0], dtype=np.float64)
        else:
            score = energy[in_cls].mean(axis=0) - energy[~in_cls].mean(axis=0)
        pattern = score @ spatial_filters
        denom = np.max(np.abs(pattern))
        if denom > 0:
            pattern = pattern / denom
        patterns.append(pattern)
    return np.vstack(patterns).astype(np.float64)


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


def make_info(ch_names: Sequence[str], montage_path: Path, sfreq: float) -> mne.Info:
    montage = read_montage_allow_count_header(montage_path)
    ch_pos_src = montage.get_positions()["ch_pos"]
    pos_by_upper = {name.upper(): pos for name, pos in ch_pos_src.items()}
    ch_pos = {}
    missing = []
    for ch in ch_names:
        key = ch.upper()
        if key in pos_by_upper:
            ch_pos[ch] = pos_by_upper[key]
        else:
            missing.append(ch)
    if missing:
        raise ValueError(f"Montage is missing positions for channels: {missing}")
    info = mne.create_info(list(ch_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"))
    return info


def topomap_limit(values: np.ndarray) -> float:
    limit = float(np.nanpercentile(np.abs(values), 99))
    return limit if limit > 0 else 1.0


def plot_topomap_grid(
    values: np.ndarray,
    info: mne.Info,
    titles: Sequence[str],
    out_path: Path,
    n_cols: int,
    suptitle: str,
    annotate: Optional[Sequence[str]] = None,
) -> None:
    n_maps = values.shape[0]
    n_rows = int(math.ceil(n_maps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.7 * n_rows), dpi=160)
    axes_arr = np.asarray(axes).reshape(-1)
    limit = topomap_limit(values)

    im = None
    for idx, ax in enumerate(axes_arr):
        if idx >= n_maps:
            ax.axis("off")
            continue
        im, _ = mne.viz.plot_topomap(
            values[idx],
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=0,
            vlim=(-limit, limit),
            sensors=True,
        )
        title = titles[idx]
        if annotate is not None:
            title = f"{title}\n{annotate[idx]}"
        ax.set_title(title, fontsize=9)
    if im is not None:
        fig.colorbar(im, ax=axes_arr.tolist(), shrink=0.72, fraction=0.03, pad=0.02)
    fig.suptitle(suptitle, y=0.995, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def channel_indices(ch_names: Sequence[str], expected: Sequence[str]) -> Tuple[List[int], List[str], List[str]]:
    by_upper = {ch.upper(): idx for idx, ch in enumerate(ch_names)}
    idx = []
    used = []
    missing = []
    for ch in expected:
        key = ch.upper()
        if key in by_upper:
            idx.append(by_upper[key])
            used.append(ch)
        else:
            missing.append(ch)
    return idx, used, missing


def anatomical_match_rows(patterns: np.ndarray, ch_names: Sequence[str], subject: str) -> List[Dict[str, object]]:
    rows = []
    all_idx = np.arange(len(ch_names))
    for cls, pattern in enumerate(patterns):
        exp_idx, used, missing = channel_indices(ch_names, EXPECTED_CHANNELS[cls])
        rest_idx = np.setdiff1d(all_idx, exp_idx)
        abs_pattern = np.abs(pattern)
        score_expected = float(np.mean(abs_pattern[exp_idx])) if exp_idx else float("nan")
        score_rest = float(np.mean(abs_pattern[rest_idx])) if len(rest_idx) else float("nan")
        denom = score_expected + score_rest
        anatomical_match = float(score_expected / denom) if denom > 0 else float("nan")
        top_idx = np.argsort(abs_pattern)[::-1][:5]
        rows.append(
            {
                "subject": subject,
                "class_index": cls,
                "event_code": EVENT_CODES[cls],
                "class_name": CLASS_NAMES[cls],
                "anatomical_match": anatomical_match,
                "score_expected": score_expected,
                "score_rest": score_rest,
                "expected_channels_used": ",".join(used),
                "expected_channels_missing": ",".join(missing),
                "top5_channels": ",".join(ch_names[i] for i in top_idx),
            }
        )
    return rows


def topo_similarity(patterns: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(patterns, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    sim = (patterns / safe) @ (patterns / safe).T
    return np.clip(sim, -1.0, 1.0)


def plot_similarity_matrix(matrix: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.3), dpi=160)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
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


def save_subject_outputs(
    subject: str,
    out_dir: Path,
    info: mne.Info,
    spatial_filters: np.ndarray,
    patterns: np.ndarray,
    ch_names: Sequence[str],
) -> List[Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_titles = [f"Filter {idx + 1}" for idx in range(spatial_filters.shape[0])]
    plot_topomap_grid(
        spatial_filters,
        info,
        raw_titles,
        out_dir / "raw_filters.png",
        n_cols=4,
        suptitle=f"{subject} raw EEGNet spatial filters",
    )

    rows = anatomical_match_rows(patterns, ch_names, subject)
    annotations = [f"match={row['anatomical_match']:.2f}" for row in rows]
    plot_topomap_grid(
        patterns,
        info,
        [f"{EVENT_CODES[i]} {CLASS_NAMES[i]}" for i in range(len(CLASS_NAMES))],
        out_dir / "class_patterns.png",
        n_cols=3,
        suptitle=f"{subject} class-conditional spatial patterns",
        annotate=annotations,
    )

    np.save(out_dir / "class_patterns.npy", patterns)
    sim = topo_similarity(patterns)
    np.save(out_dir / "topo_similarity_matrix.npy", sim)
    plot_similarity_matrix(sim, out_dir / "topo_similarity_matrix.png", f"{subject} topology similarity")
    pd.DataFrame(rows).drop(columns=["subject"]).to_csv(out_dir / "anatomical_match.csv", index=False)
    return rows


def process_subject(subject: str, args: argparse.Namespace) -> SubjectOutput:
    npz_path = resolve_template(args.data_template, subject)
    ckpt_path = resolve_template(args.checkpoint_template, subject)
    X, y, ch_names = load_npz(npz_path, args.n_channels, args.n_samples)
    train_idx, test_idx = get_fold_split(X, y, args.n_splits, args.fold, args.seed)
    X_train, X_test, y_test = X[train_idx], X[test_idx], y[test_idx]
    if not args.no_channel_normalize:
        mean, std = fit_channel_standardizer(X_train)
        X_test = apply_channel_standardize(X_test, mean, std)

    device = torch.device(args.device)
    checkpoint_state = load_checkpoint_state(ckpt_path, device)
    if args.model_module is None:
        infer_eegnet_hparams_from_state_dict(checkpoint_state, args)
    model = build_model(args)
    load_checkpoint(model, ckpt_path, device, checkpoint_state)
    spatial_layer_name = resolve_spatial_layer_name(model, args.spatial_layer)
    spatial_layer = named_module(model, spatial_layer_name)
    spatial_filters = spatial_filters_from_layer(spatial_layer, args.n_channels)
    activations = infer_spatial_activations(
        model, X_test, device, args.batch_size, spatial_layer_name
    )
    patterns = class_conditional_patterns(spatial_filters, activations, y_test, args)
    info = make_info(ch_names, Path(args.montage), args.sfreq)
    rows = save_subject_outputs(
        subject,
        Path(args.out_dir) / subject,
        info,
        spatial_filters,
        patterns,
        ch_names,
    )
    return SubjectOutput(subject=subject, ch_names=ch_names, class_patterns=patterns, anatomical_rows=rows)


def save_grand_outputs(outputs: Sequence[SubjectOutput], args: argparse.Namespace) -> None:
    if not outputs:
        return
    base_ch = [ch.upper() for ch in outputs[0].ch_names]
    for output in outputs[1:]:
        if [ch.upper() for ch in output.ch_names] != base_ch:
            raise ValueError("Cannot grand-average: subject channel orders differ.")

    normalized = []
    for output in outputs:
        denom = np.max(np.abs(output.class_patterns), axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        normalized.append(output.class_patterns / denom)
    grand_patterns = np.mean(np.stack(normalized, axis=0), axis=0)

    grand_dir = Path(args.out_dir) / "grand"
    grand_dir.mkdir(parents=True, exist_ok=True)
    info = make_info(outputs[0].ch_names, Path(args.montage), args.sfreq)
    rows = anatomical_match_rows(grand_patterns, outputs[0].ch_names, "grand")
    annotations = [f"match={row['anatomical_match']:.2f}" for row in rows]
    plot_topomap_grid(
        grand_patterns,
        info,
        [f"{EVENT_CODES[i]} {CLASS_NAMES[i]}" for i in range(len(CLASS_NAMES))],
        grand_dir / "grand_class_patterns.png",
        n_cols=3,
        suptitle="Grand-average class-conditional spatial patterns",
        annotate=annotations,
    )
    sim = topo_similarity(grand_patterns)
    np.save(grand_dir / "grand_topo_similarity.npy", sim)
    plot_similarity_matrix(sim, grand_dir / "grand_topo_similarity.png", "Grand topology similarity")
    pd.DataFrame(rows).drop(columns=["subject"]).to_csv(
        grand_dir / "grand_anatomical_match.csv", index=False
    )

    all_rows = []
    for output in outputs:
        all_rows.extend(output.anatomical_rows)
    pd.DataFrame(all_rows).to_csv(grand_dir / "summary_across_subjects.csv", index=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    outputs = []
    for subject in args.subjects:
        print(f"[INFO] Processing {subject}")
        output = process_subject(subject, args)
        outputs.append(output)
        mean_match = np.nanmean([row["anatomical_match"] for row in output.anatomical_rows])
        print(f"[INFO] {subject}: mean anatomical_match={mean_match:.3f}")
    save_grand_outputs(outputs, args)
    print(f"[INFO] Saved outputs under {args.out_dir}")


if __name__ == "__main__":
    main()

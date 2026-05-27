#!/usr/bin/env python3
"""
EEGNet representation visualization for cognitive-task-family block structure.

For each subject and for a pooled analysis, this script uses only fold-0 test
samples from StratifiedKFold(k=5, seed=42), captures the penultimate EEGNet
embedding with a forward hook, and saves t-SNE plots, confusion/prototype RDMs,
block-structure scores, and family-level silhouette scores.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from robotic_hand_bci.project import PROJECT_ROOT

DEFAULT_DATA_TEMPLATE = (
    str(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "pythondata1"
        / "{subject}"
        / "{subject}_EEGNet_NoICA_uV.npz"
    )
)
DEFAULT_CKPT_TEMPLATE = (
    str(PROJECT_ROOT / "model" / "10fold_npz" / "experiments")
    + "/pythondata1_npz_repr/representation/checkpoints/{subject}_eegnet_noica.pt"
)
DEFAULT_OUT_DIR = str(
    PROJECT_ROOT / "artifacts" / "analysis" / "representation" / "tsne_rdm_block"
)

EVENT_CODES = [11, 12, 13, 14, 15, 16]
CLASS_NAMES = [
    "11 manipulation",
    "12 color class",
    "13 color track",
    "14 face",
    "15 RPS",
    "16 gesture",
]

# y is assumed to be 0..5 in EVENT_CODES order. Edit here if your mapping differs.
FAMILY_BY_CLASS = {
    0: "MOTOR",   # code 11: manipulation/control
    1: "VISUAL",  # code 12: color classification
    2: "VISUAL",  # code 13: color tracking
    3: "VISUAL",  # code 14: face recognition
    4: "MOTOR",   # code 15: RPS action
    5: "VISUAL",  # code 16: gesture / biological motion
}
FAMILY_TO_ID = {"VISUAL": 0, "MOTOR": 1}
FAMILY_MARKERS = {"VISUAL": "o", "MOTOR": "s"}
FAMILY_COLORS = {"VISUAL": "#0072B2", "MOTOR": "#D55E00"}


class EEGNetBaseline(nn.Module):
    """Local EEGNet fallback matching the repository baseline implementation."""

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
        feat = self._forward_features(x)
        return self.classifier(feat)


EEGNet = EEGNetBaseline


@dataclass
class SubjectResult:
    subject: str
    y_true: np.ndarray
    y_pred: np.ndarray
    embeddings_raw: np.ndarray
    embeddings_scaled: np.ndarray
    tsne_xy: np.ndarray
    test_acc: float
    block_score: float
    silhouette_2d: float
    silhouette_highdim: float
    top1_confused_pair: str
    top1_confused_acc_drop: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEGNet t-SNE/RDM analysis focused on cognitive-family block structure."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"S{i}" for i in range(1, 11)],
        help="Subject ids, e.g. S1 S2 ... S10.",
    )
    parser.add_argument("--data-template", default=DEFAULT_DATA_TEMPLATE)
    parser.add_argument("--checkpoint-template", default=DEFAULT_CKPT_TEMPLATE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--n-channels", type=int, default=59)
    parser.add_argument("--n-samples", type=int, default=1126)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iter", type=int, default=1000)
    parser.add_argument(
        "--model-module",
        default=None,
        help="Optional import path containing the EEGNet class.",
    )
    parser.add_argument(
        "--model-class",
        default="EEGNet",
        help="Model class name inside --model-module. Ignored when module is absent.",
    )
    parser.add_argument(
        "--embedding-layer",
        default="classifier",
        help="Module name whose forward input is captured as the penultimate embedding.",
    )
    parser.add_argument(
        "--no-channel-normalize",
        action="store_true",
        help="Disable train-fold channel standardization before model inference.",
    )
    return parser.parse_args(argv)


def resolve_template(template: str, subject: str) -> Path:
    return Path(template.format(subject=subject, Subject=subject, si=subject, Si=subject))


def load_npz(npz_path: Path, n_channels: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if X.ndim != 3:
        raise ValueError(f"{npz_path} X must be (trials, channels, samples), got {X.shape}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"{npz_path} y shape {y.shape} does not match X shape {X.shape}")
    if X.shape[1] != n_channels:
        raise ValueError(f"{npz_path} has {X.shape[1]} channels, expected {n_channels}")
    if X.shape[2] != n_samples:
        raise ValueError(
            f"{npz_path} has {X.shape[2]} samples, expected {n_samples}. "
            "Use --n-samples if your checkpoint was trained on a cropped length."
        )
    return X, y


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


def apply_channel_standardize(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def import_model_class(model_module: Optional[str], model_class: str):
    if not model_module:
        return EEGNetBaseline
    module = importlib.import_module(model_module)
    return getattr(module, model_class)


def build_model(args: argparse.Namespace) -> nn.Module:
    model_cls = import_model_class(args.model_module, args.model_class)
    attempts = [
        dict(n_classes=args.n_classes, n_channels=args.n_channels, n_samples=args.n_samples),
        dict(n_classes=args.n_classes, n_chans=args.n_channels, n_times=args.n_samples),
        dict(n_outputs=args.n_classes, n_chans=args.n_channels, n_times=args.n_samples),
        dict(in_chans=args.n_channels, n_classes=args.n_classes, input_window_samples=args.n_samples),
        dict(classes=args.n_classes, channels=args.n_channels, samples=args.n_samples),
        dict(),
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return model_cls(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise TypeError(f"Could not instantiate {model_cls} with known EEGNet kwargs") from last_error


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
    raise ValueError("Checkpoint does not look like a model state_dict container.")


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
    state_dict = strip_common_prefixes(extract_state_dict(checkpoint))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load checkpoint {checkpoint_path}. "
            "Check --model-module/--model-class and --n-samples."
        ) from exc


def get_named_module(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        available = ", ".join(name for name in modules if name)
        raise KeyError(f"Module {module_name!r} not found. Available modules: {available}")
    return modules[module_name]


def infer_logits_and_embeddings(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    embedding_layer: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    captured: List[np.ndarray] = []

    def hook(_module, module_inputs, _module_output):
        captured.append(module_inputs[0].detach().cpu().numpy())

    target = get_named_module(model, embedding_layer)
    handle = target.register_forward_hook(hook)
    logits_batches: List[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start : start + batch_size]).float().to(device)
                logits = model(xb)
                logits_batches.append(logits.detach().cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(logits_batches, axis=0), np.concatenate(captured, axis=0)


def scale_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(embeddings)


def make_tsne_from_scaled(embeddings_scaled: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(embeddings_scaled) <= 2:
        raise ValueError("Need at least 3 samples for t-SNE.")
    perplexity = min(args.perplexity, max(1.0, (len(embeddings_scaled) - 1) / 3.0))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    signature = inspect.signature(TSNE)
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = args.tsne_iter
    else:
        kwargs["n_iter"] = args.tsne_iter
    return TSNE(**kwargs).fit_transform(embeddings_scaled)


def row_normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        normalize="true",
    )


def rdm_from_confusion(conf: np.ndarray) -> np.ndarray:
    rdm = 1.0 - (conf + conf.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def cosine_distance_matrix(prototypes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    normalized = prototypes / safe
    sim = normalized @ normalized.T
    rdm = 1.0 - np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_from_prototypes(
    embeddings: np.ndarray, y: np.ndarray, n_classes: int
) -> np.ndarray:
    prototypes = []
    for cls in range(n_classes):
        cls_emb = embeddings[y == cls]
        if len(cls_emb) == 0:
            prototypes.append(np.full(embeddings.shape[1], np.nan, dtype=np.float64))
        else:
            prototypes.append(cls_emb.mean(axis=0))
    prototypes_arr = np.vstack(prototypes)
    if np.isnan(prototypes_arr).any():
        rdm = np.full((n_classes, n_classes), np.nan, dtype=np.float64)
        valid = ~np.isnan(prototypes_arr).any(axis=1)
        rdm[np.ix_(valid, valid)] = cosine_distance_matrix(prototypes_arr[valid])
        np.fill_diagonal(rdm, 0.0)
        return rdm
    return cosine_distance_matrix(prototypes_arr)


def block_structure_metrics(rdm: np.ndarray) -> Dict[str, float]:
    visual = [cls for cls, fam in FAMILY_BY_CLASS.items() if fam == "VISUAL"]
    motor = [cls for cls, fam in FAMILY_BY_CLASS.items() if fam == "MOTOR"]

    within_visual = [rdm[i, j] for idx, i in enumerate(visual) for j in visual[idx + 1 :]]
    within_motor = [rdm[i, j] for idx, i in enumerate(motor) for j in motor[idx + 1 :]]
    between = [rdm[i, j] for i in visual for j in motor]

    within_visual_mean = float(np.nanmean(within_visual)) if within_visual else float("nan")
    within_motor_mean = float(np.nanmean(within_motor)) if within_motor else float("nan")
    between_mean = float(np.nanmean(between)) if between else float("nan")
    block_score = between_mean - (within_visual_mean + within_motor_mean) / 2.0
    return {
        "within_visual": within_visual_mean,
        "within_motor": within_motor_mean,
        "between_family": between_mean,
        "block_score": float(block_score),
    }


def balanced_indices_by_class(y: np.ndarray, n_classes: int, seed: int) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    per_class = [np.where(y == cls)[0] for cls in range(n_classes)]
    min_count = min(len(idx) for idx in per_class)
    if min_count <= 0:
        return np.array([], dtype=np.int64), int(min_count)
    chosen = [rng.choice(idx, size=min_count, replace=False) for idx in per_class]
    return np.sort(np.concatenate(chosen)).astype(np.int64), int(min_count)


def family_labels_from_class(y: np.ndarray) -> np.ndarray:
    return np.asarray([FAMILY_TO_ID[FAMILY_BY_CLASS[int(cls)]] for cls in y], dtype=np.int64)


def balanced_family_silhouette(
    features: np.ndarray, y: np.ndarray, n_classes: int, seed: int
) -> Tuple[float, int]:
    idx, min_count = balanced_indices_by_class(y, n_classes, seed)
    if len(idx) < 3:
        return float("nan"), min_count
    family_y = family_labels_from_class(y[idx])
    if len(np.unique(family_y)) < 2:
        return float("nan"), min_count
    return float(silhouette_score(features[idx], family_y, metric="euclidean")), min_count


def top_confused_pair_and_drop(conf: np.ndarray) -> Tuple[str, float]:
    sym = (conf + conf.T) / 2.0
    np.fill_diagonal(sym, -np.inf)
    if not np.isfinite(sym).any() or float(np.nanmax(sym)) <= 0.0:
        return "none", 0.0
    i, j = np.unravel_index(np.nanargmax(sym), sym.shape)
    a, b = sorted((int(i), int(j)))
    pair = f"{EVENT_CODES[a]}-{EVENT_CODES[b]}"
    return pair, float(sym[i, j])


def finite_json_value(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def plot_tsne_by_class(tsne_xy: np.ndarray, y: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.9), dpi=160)
    palette = sns.color_palette("tab10", n_colors=len(CLASS_NAMES))
    for cls, label in enumerate(CLASS_NAMES):
        mask = y == cls
        if not np.any(mask):
            continue
        family = FAMILY_BY_CLASS[cls]
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            s=22,
            alpha=0.78,
            marker=FAMILY_MARKERS[family],
            color=palette[cls],
            label=f"{label} ({family})",
            linewidths=0.35,
            edgecolors="white",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=7, markerscale=1.25, ncol=2)
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tsne_by_family(tsne_xy: np.ndarray, y: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=160)
    family_y = np.asarray([FAMILY_BY_CLASS[int(cls)] for cls in y])
    for family in ("VISUAL", "MOTOR"):
        mask = family_y == family
        if not np.any(mask):
            continue
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            s=24,
            alpha=0.78,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            label=family,
            linewidths=0.35,
            edgecolors="white",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=9, markerscale=1.25)
    ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_matrix(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=160)
    ticklabels = [str(code) for code in EVENT_CODES]
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        annot=True,
        fmt=".2f",
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"shrink": 0.82},
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted" if "Confusion" in title else "Class")
    ax.set_ylabel("True" if "Confusion" in title else "Class")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_analysis_outputs(
    out_dir: Path,
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings_for_rdm: np.ndarray,
    embeddings_scaled: np.ndarray,
    tsne_xy: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    conf = row_normalized_confusion(y_true, y_pred, args.n_classes)
    rdm_conf = rdm_from_confusion(conf)
    rdm_proto = rdm_from_prototypes(embeddings_for_rdm, y_true, args.n_classes)
    np.save(out_dir / "rdm_confusion.npy", rdm_conf)
    np.save(out_dir / "rdm_prototype.npy", rdm_proto)

    plot_tsne_by_class(tsne_xy, y_true, out_dir / "tsne_by_class.png", f"{subject} classes")
    plot_tsne_by_family(tsne_xy, y_true, out_dir / "tsne_by_family.png", f"{subject} families")
    plot_matrix(conf, out_dir / "confusion_matrix.png", f"{subject} Confusion", "Blues")
    plot_matrix(rdm_conf, out_dir / "rdm_confusion.png", f"{subject} Confusion RDM", "mako")
    plot_matrix(rdm_proto, out_dir / "rdm_prototype.png", f"{subject} Prototype RDM", "rocket")

    block_conf = block_structure_metrics(rdm_conf)
    block_proto = block_structure_metrics(rdm_proto)
    silhouette_2d, min_count_2d = balanced_family_silhouette(
        tsne_xy, y_true, args.n_classes, args.seed
    )
    silhouette_highdim, min_count_highdim = balanced_family_silhouette(
        embeddings_scaled, y_true, args.n_classes, args.seed
    )
    pair, acc_drop = top_confused_pair_and_drop(conf)
    test_acc = float(accuracy_score(y_true, y_pred))

    block_payload = {
        "subject": subject,
        "block_score": finite_json_value(block_proto["block_score"]),
        "block_score_source": "prototype_cosine_rdm",
        "block_score_confusion": finite_json_value(block_conf["block_score"]),
        "block_score_prototype": finite_json_value(block_proto["block_score"]),
        "prototype_rdm_components": {
            key: finite_json_value(value) for key, value in block_proto.items()
        },
        "confusion_rdm_components": {
            key: finite_json_value(value) for key, value in block_conf.items()
        },
        "silhouette_2d": finite_json_value(silhouette_2d),
        "silhouette_highdim": finite_json_value(silhouette_highdim),
        "silhouette_metric": "euclidean",
        "silhouette_balanced_min_count_per_class_2d": min_count_2d,
        "silhouette_balanced_min_count_per_class_highdim": min_count_highdim,
        "family_map": {str(EVENT_CODES[cls]): FAMILY_BY_CLASS[cls] for cls in range(args.n_classes)},
    }
    with (out_dir / "block_structure.json").open("w", encoding="utf-8") as f:
        json.dump(block_payload, f, indent=2, ensure_ascii=False)

    return {
        "subject": subject,
        "test_acc": test_acc,
        "block_score": float(block_proto["block_score"]),
        "silhouette_2d": float(silhouette_2d),
        "silhouette_highdim": float(silhouette_highdim),
        "top1_confused_pair": pair,
        "top1_confused_acc_drop": acc_drop,
    }


def process_subject(
    subject: str,
    args: argparse.Namespace,
    device: torch.device,
) -> SubjectResult:
    npz_path = resolve_template(args.data_template, subject)
    ckpt_path = resolve_template(args.checkpoint_template, subject)
    X, y = load_npz(npz_path, args.n_channels, args.n_samples)
    train_idx, test_idx = get_fold_split(X, y, args.n_splits, args.fold, args.seed)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    if not args.no_channel_normalize:
        mean, std = fit_channel_standardizer(X_train)
        X_test = apply_channel_standardize(X_test, mean, std)

    model = build_model(args)
    load_checkpoint(model, ckpt_path, device)
    logits, embeddings = infer_logits_and_embeddings(
        model,
        X_test,
        device=device,
        batch_size=args.batch_size,
        embedding_layer=args.embedding_layer,
    )
    embeddings_scaled = scale_embeddings(embeddings)
    tsne_xy = make_tsne_from_scaled(embeddings_scaled, args)
    y_pred = logits.argmax(axis=1).astype(np.int64)

    summary_row = save_analysis_outputs(
        Path(args.out_dir) / subject,
        subject,
        y_test,
        y_pred,
        embeddings,
        embeddings_scaled,
        tsne_xy,
        args,
    )
    return SubjectResult(
        subject=subject,
        y_true=y_test,
        y_pred=y_pred,
        embeddings_raw=embeddings,
        embeddings_scaled=embeddings_scaled,
        tsne_xy=tsne_xy,
        test_acc=float(summary_row["test_acc"]),
        block_score=float(summary_row["block_score"]),
        silhouette_2d=float(summary_row["silhouette_2d"]),
        silhouette_highdim=float(summary_row["silhouette_highdim"]),
        top1_confused_pair=str(summary_row["top1_confused_pair"]),
        top1_confused_acc_drop=float(summary_row["top1_confused_acc_drop"]),
    )


def zscore_embeddings_per_subject(
    results: Sequence[SubjectResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = []
    y_true = []
    y_pred = []
    for result in results:
        embeddings.append(scale_embeddings(result.embeddings_raw))
        y_true.append(result.y_true)
        y_pred.append(result.y_pred)
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(y_true, axis=0),
        np.concatenate(y_pred, axis=0),
    )


def save_pooled_outputs(results: Sequence[SubjectResult], args: argparse.Namespace) -> Dict[str, object]:
    embeddings_pooled, y_true, y_pred = zscore_embeddings_per_subject(results)
    embeddings_pooled_scaled = scale_embeddings(embeddings_pooled)
    tsne_xy = make_tsne_from_scaled(embeddings_pooled_scaled, args)
    return save_analysis_outputs(
        Path(args.out_dir) / "pooled",
        "pooled",
        y_true,
        y_pred,
        embeddings_pooled,
        embeddings_pooled_scaled,
        tsne_xy,
        args,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    results: List[SubjectResult] = []
    rows: List[Dict[str, object]] = []
    for subject in args.subjects:
        print(f"[INFO] Processing {subject}")
        result = process_subject(subject, args, device)
        results.append(result)
        rows.append(
            {
                "subject": result.subject,
                "test_acc": result.test_acc,
                "block_score": result.block_score,
                "silhouette_2d": result.silhouette_2d,
                "silhouette_highdim": result.silhouette_highdim,
                "top1_confused_pair": result.top1_confused_pair,
                "top1_confused_acc_drop": result.top1_confused_acc_drop,
            }
        )
        print(
            f"[INFO] {subject}: acc={result.test_acc:.4f}, "
            f"block={result.block_score:.4f}, "
            f"sil2d={result.silhouette_2d:.4f}, "
            f"silHD={result.silhouette_highdim:.4f}, "
            f"confused={result.top1_confused_pair}"
        )

    if results:
        print("[INFO] Processing pooled")
        rows.append(save_pooled_outputs(results, args))

    columns = [
        "subject",
        "test_acc",
        "block_score",
        "silhouette_2d",
        "silhouette_highdim",
        "top1_confused_pair",
        "top1_confused_acc_drop",
    ]
    summary = pd.DataFrame(rows, columns=columns)
    summary_path = out_dir / "all_subjects_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

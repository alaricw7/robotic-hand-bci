#!/usr/bin/env python3
"""
EEGNet test-fold representation visualization and confusion/prototype RDMs.

For each subject, this script:
  1. Loads pythondata1 NoICA NPZ and the trained checkpoint.
  2. Recreates StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
  3. Uses fold 0 test samples only.
  4. Captures classifier-preceding embeddings with a forward hook.
  5. Saves t-SNE, confusion matrix, confusion RDM, prototype RDM, and summary.csv.

The pooled analysis uses per-subject z-scored embeddings before concatenating, so
cross-subject scale/offset does not dominate t-SNE.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
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
from sklearn.metrics import accuracy_score, confusion_matrix
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
    PROJECT_ROOT / "artifacts" / "analysis" / "representation" / "tsne_rdm"
)
CLASS_NAMES = ["11", "12", "13", "14", "15", "16"]


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
    embeddings: np.ndarray
    tsne_xy: np.ndarray
    test_acc: float
    top1_confused_pair: str
    mean_intercluster_distance_in_tsne: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize EEGNet NoICA test-fold embeddings and RDMs."
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
        help="Module name whose forward input is captured as embedding.",
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


def fold0_test_split(
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
        emb = module_inputs[0].detach().cpu().numpy()
        captured.append(emb)

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

    logits_all = np.concatenate(logits_batches, axis=0)
    embeddings = np.concatenate(captured, axis=0)
    return logits_all, embeddings


def make_tsne(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(embeddings) <= 2:
        raise ValueError("Need at least 3 samples for t-SNE.")
    scaled = StandardScaler().fit_transform(embeddings)
    perplexity = min(args.perplexity, max(1.0, (len(scaled) - 1) / 3.0))
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
    return TSNE(**kwargs).fit_transform(scaled)


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


def top_confused_pair(conf: np.ndarray, class_names: Sequence[str]) -> str:
    sym = (conf + conf.T) / 2.0
    np.fill_diagonal(sym, -np.inf)
    if not np.isfinite(sym).any() or float(np.nanmax(sym)) <= 0.0:
        return "none"
    i, j = np.unravel_index(np.nanargmax(sym), sym.shape)
    a, b = sorted((int(i), int(j)))
    return f"{class_names[a]}-{class_names[b]}"


def mean_intercluster_distance(tsne_xy: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    centroids = []
    for cls in range(n_classes):
        pts = tsne_xy[y == cls]
        if len(pts) > 0:
            centroids.append(pts.mean(axis=0))
    if len(centroids) < 2:
        return float("nan")
    centroids = np.vstack(centroids)
    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return float(np.mean(dists))


def plot_tsne(tsne_xy: np.ndarray, y: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=160)
    palette = sns.color_palette("tab10", n_colors=len(CLASS_NAMES))
    for cls, name in enumerate(CLASS_NAMES):
        mask = y == cls
        if not np.any(mask):
            continue
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            s=18,
            alpha=0.78,
            color=palette[cls],
            label=f"code {name}",
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=8, markerscale=1.2)
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
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=160)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        annot=True,
        fmt=".2f",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"shrink": 0.82},
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted" if "Confusion" in title else "Class")
    ax.set_ylabel("True" if "Confusion" in title else "Class")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_subject_outputs(
    out_dir: Path,
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    tsne_xy: np.ndarray,
    n_classes: int,
) -> Tuple[float, str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conf = row_normalized_confusion(y_true, y_pred, n_classes)
    rdm_conf = rdm_from_confusion(conf)
    rdm_proto = rdm_from_prototypes(embeddings, y_true, n_classes)
    np.save(out_dir / "rdm_confusion.npy", rdm_conf)
    np.save(out_dir / "rdm_prototype.npy", rdm_proto)

    plot_tsne(tsne_xy, y_true, out_dir / "tsne.png", f"{subject} test embeddings")
    plot_matrix(conf, out_dir / "confusion_matrix.png", f"{subject} Confusion", "Blues")
    plot_matrix(rdm_conf, out_dir / "rdm_confusion.png", f"{subject} Confusion RDM", "mako")
    plot_matrix(rdm_proto, out_dir / "rdm_prototype.png", f"{subject} Prototype RDM", "rocket")

    acc = float(accuracy_score(y_true, y_pred))
    pair = top_confused_pair(conf, CLASS_NAMES)
    tsne_dist = mean_intercluster_distance(tsne_xy, y_true, n_classes)
    return acc, pair, tsne_dist


def process_subject(
    subject: str,
    args: argparse.Namespace,
    device: torch.device,
) -> SubjectResult:
    npz_path = resolve_template(args.data_template, subject)
    ckpt_path = resolve_template(args.checkpoint_template, subject)
    X, y = load_npz(npz_path, args.n_channels, args.n_samples)
    train_idx, test_idx = fold0_test_split(X, y, args.n_splits, args.fold, args.seed)

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
    y_pred = logits.argmax(axis=1).astype(np.int64)
    tsne_xy = make_tsne(embeddings, args)
    subject_out = Path(args.out_dir) / subject
    acc, pair, tsne_dist = save_subject_outputs(
        subject_out,
        subject,
        y_test,
        y_pred,
        embeddings,
        tsne_xy,
        args.n_classes,
    )
    return SubjectResult(
        subject=subject,
        y_true=y_test,
        y_pred=y_pred,
        embeddings=embeddings,
        tsne_xy=tsne_xy,
        test_acc=acc,
        top1_confused_pair=pair,
        mean_intercluster_distance_in_tsne=tsne_dist,
    )


def zscore_embeddings_per_subject(results: Sequence[SubjectResult]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_embeddings = []
    all_true = []
    all_pred = []
    for result in results:
        emb_z = StandardScaler().fit_transform(result.embeddings)
        all_embeddings.append(emb_z)
        all_true.append(result.y_true)
        all_pred.append(result.y_pred)
    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_true, axis=0),
        np.concatenate(all_pred, axis=0),
    )


def save_pooled_outputs(
    results: Sequence[SubjectResult], args: argparse.Namespace
) -> Dict[str, object]:
    embeddings, y_true, y_pred = zscore_embeddings_per_subject(results)
    pooled_args = argparse.Namespace(**vars(args))
    tsne_xy = make_tsne(embeddings, pooled_args)
    out_dir = Path(args.out_dir) / "pooled"
    acc, pair, tsne_dist = save_subject_outputs(
        out_dir,
        "pooled",
        y_true,
        y_pred,
        embeddings,
        tsne_xy,
        args.n_classes,
    )
    return {
        "subject": "pooled",
        "test_acc": acc,
        "top1_confused_pair": pair,
        "mean_intercluster_distance_in_tsne": tsne_dist,
    }


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
                "top1_confused_pair": result.top1_confused_pair,
                "mean_intercluster_distance_in_tsne": result.mean_intercluster_distance_in_tsne,
            }
        )
        print(
            f"[INFO] {subject}: acc={result.test_acc:.4f}, "
            f"top_confused={result.top1_confused_pair}, "
            f"tsne_intercluster={result.mean_intercluster_distance_in_tsne:.4f}"
        )

    if results:
        print("[INFO] Processing pooled")
        pooled_row = save_pooled_outputs(results, args)
        rows.append(pooled_row)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

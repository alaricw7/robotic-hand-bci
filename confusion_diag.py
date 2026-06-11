"""Task 5 A2 — confusion matrix diagnostics on a single joint run's ckpts.

Aggregates all (subject, fold) test predictions, builds 6x6 confusion matrix
normalised per true class, reports top-3 confused class pairs and per-class
recall. Also dumps the excl-S2/S3 variant.

Outputs:
  diagnostics/confusion/<exp>/{confusion.csv, confusion_excl.csv, top_pairs.txt}
"""

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from ablation_config import ABLATION_PRESETS, TRI_DEFAULTS
from data import (
    SUBJECTS,
    load_subject,
    standardize_per_channel,
    stratified_kfold_train_val_test_splits,
)
from model import build_model


def build_tri_cfg(n_channels, n_times, n_classes, ablation, tri_overrides, seed):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation])
    if tri_overrides:
        values.update(tri_overrides)
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes})
    return SimpleNamespace(**values)


@torch.no_grad()
def _predict_argmax(model, X_np, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, X_np.shape[0], batch_size):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device, non_blocking=True)
        out.append(model(xb).argmax(dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def collect_predictions(exp, results_root, subjects, n_folds, val_size, seed, device):
    y_true_all, y_pred_all, subj_all = [], [], []
    for subject in subjects:
        X, y = load_subject(subject)
        n_classes = int(y.max() + 1)
        n_channels, n_times = X.shape[1], X.shape[2]
        for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
            y, n_splits=n_folds, val_size=val_size, seed=seed,
        ):
            cp = Path(results_root) / exp / "ckpts" / subject / f"fold{fold_idx}.pt"
            blob = torch.load(cp, map_location="cpu", weights_only=False)
            meta = blob.get("meta") or {}
            cfg = build_tri_cfg(n_channels, n_times, n_classes,
                                meta.get("ablation", "full_std_coords"),
                                meta.get("tri_overrides") or {}, seed)
            model = build_model(cfg, model_name="tridomain")
            model.load_state_dict(blob["state_dict"])
            model.eval().to(device)
            X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
            _, _, X_te_std = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)
            yp = _predict_argmax(model, X_te_std, device)
            y_true_all.append(y[te_idx])
            y_pred_all.append(yp)
            subj_all.extend([subject] * len(yp))
            del model
        print(f"  {subject} collected")
    return (np.concatenate(y_true_all), np.concatenate(y_pred_all),
            np.asarray(subj_all), n_classes)


def confusion_normalised(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1.0
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_n = cm / np.maximum(row_sum, 1)
    return cm, cm_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="abl_signalfit_T7_full_stack_taps251")
    p.add_argument("--results-root", default=str(THIS_DIR / "results" / "aug_hpo"))
    p.add_argument("--out-root", default=str(THIS_DIR / "diagnostics" / "confusion"))
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    args = p.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_root) / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[A2] exp={args.exp} device={device}")
    y_true, y_pred, subj, n_classes = collect_predictions(
        args.exp, args.results_root, args.subjects,
        args.n_folds, args.val_size, args.seed, device,
    )

    def write_cm(tag, mask):
        cm, cm_n = confusion_normalised(y_true[mask], y_pred[mask], n_classes)
        # Write normalised matrix
        with (out_dir / f"confusion_{tag}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + [f"T{i+1}" for i in range(n_classes)])
            for i in range(n_classes):
                w.writerow([f"T{i+1}"] + [f"{cm_n[i,j]:.4f}" for j in range(n_classes)])
        per_class_recall = np.diag(cm_n)
        # off-diagonal pairs
        pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    continue
                pairs.append((cm_n[i, j], i, j))
        pairs.sort(reverse=True)
        return cm, cm_n, per_class_recall, pairs

    cm_all, cm_n_all, recall_all, pairs_all = write_cm("all", np.ones_like(y_true, dtype=bool))
    excl_mask = ~np.isin(subj, ("S2", "S3"))
    cm_e, cm_n_e, recall_e, pairs_e = write_cm("excl_S2S3", excl_mask)

    lines = []
    lines.append(f"# Confusion diagnostics — {args.exp}\n")
    lines.append("## All 10 subjects")
    lines.append(f"  per-class recall: " + "  ".join(
        f"T{i+1}={recall_all[i]:.3f}" for i in range(n_classes)))
    lines.append(f"  top-3 confused (true → pred, rate):")
    for rate, i, j in pairs_all[:3]:
        lines.append(f"    T{i+1} → T{j+1}: {rate:.3f}")
    lines.append("")
    lines.append("## Excl S2/S3")
    lines.append(f"  per-class recall: " + "  ".join(
        f"T{i+1}={recall_e[i]:.3f}" for i in range(n_classes)))
    lines.append(f"  top-3 confused (true → pred, rate):")
    for rate, i, j in pairs_e[:3]:
        lines.append(f"    T{i+1} → T{j+1}: {rate:.3f}")
    txt = "\n".join(lines) + "\n"
    (out_dir / "top_pairs.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"saved: {out_dir/'confusion_all.csv'} / confusion_excl_S2S3.csv / top_pairs.txt")


if __name__ == "__main__":
    main()

"""Task 6 Step 0 — channel importance ranking.

Two rankings on the T7 (or any joint exp) checkpoints:

  (a) SpaceBranch importance:
        For each (subject, fold), forward TRAIN trials through the loaded
        model, pull aux['channel_importance'] (B, 59) → mean over trials.
        Average over 100 (subject, fold).
  (b) Permutation importance:
        For each (subject, fold) × each channel c: zero out channel c on
        VAL, forward, compute val acc. Drop = baseline_val_acc - permuted_acc.
        Average drops over 100 (subject, fold).

Outputs (under diagnostics/):
  channel_importance.csv       — both rankings + Spearman correlation
  channel_importance_plot.txt  — top-K side-by-side
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

from ablation_config import ABLATION_PRESETS, TRI_DEFAULTS, TRI_CHANNEL_NAMES
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
def _imp_mean(model, X_np, device, batch_size=64):
    """Average SpaceBranch.channel_importance over given trials."""
    model.eval()
    accum = None
    n_total = 0
    for i in range(0, X_np.shape[0], batch_size):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device, non_blocking=True)
        _, aux = model(xb, return_aux=True)
        imp = aux.get("channel_importance")
        if imp is None:
            return None
        v = imp.detach().cpu().numpy()
        accum = v.sum(axis=0) if accum is None else accum + v.sum(axis=0)
        n_total += v.shape[0]
    return accum / max(n_total, 1)


@torch.no_grad()
def _val_acc_with_mask(model, X_np, y, device, mask_idx=None, batch_size=64):
    model.eval()
    correct, total = 0, 0
    for i in range(0, X_np.shape[0], batch_size):
        xb_np = X_np[i:i + batch_size].copy()
        if mask_idx is not None:
            xb_np[:, mask_idx, :] = 0.0
        xb = torch.from_numpy(xb_np).float().to(device, non_blocking=True)
        pred = model(xb).argmax(dim=-1).cpu().numpy()
        correct += int((pred == y[i:i + batch_size]).sum())
        total += pred.size
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="abl_signalfit_T7_full_stack_taps251")
    p.add_argument("--results-root", default=str(THIS_DIR / "results" / "aug_hpo"))
    p.add_argument("--out-dir", default=str(THIS_DIR / "diagnostics"))
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--skip-perm", action="store_true",
                   help="Skip permutation importance (saves time, leaves only SpaceBranch ranking).")
    args = p.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[chimp] exp={args.exp} device={device}  skip_perm={args.skip_perm}")

    n_ch = len(TRI_CHANNEL_NAMES)
    imp_accum = np.zeros(n_ch, dtype=float)
    imp_count = 0
    perm_drop_accum = np.zeros(n_ch, dtype=float)
    perm_count = 0

    for subject in args.subjects:
        X, y = load_subject(subject)
        n_classes = int(y.max() + 1)
        n_channels, n_times = X.shape[1], X.shape[2]
        for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
            y, n_splits=args.n_folds, val_size=args.val_size, seed=args.seed,
        ):
            cp = Path(args.results_root) / args.exp / "ckpts" / subject / f"fold{fold_idx}.pt"
            blob = torch.load(cp, map_location="cpu", weights_only=False)
            meta = blob.get("meta") or {}
            cfg = build_tri_cfg(n_channels, n_times, n_classes,
                                meta.get("ablation", "full_std_coords"),
                                meta.get("tri_overrides") or {}, args.seed)
            model = build_model(cfg, model_name="tridomain")
            model.load_state_dict(blob["state_dict"])
            model.eval().to(device)

            X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
            y_va = y[va_idx]
            X_tr_std, X_va_std, X_te_std = standardize_per_channel(
                X_tr_raw, X_va_raw, X_te_raw)

            v = _imp_mean(model, X_tr_std, device)
            if v is not None:
                imp_accum += v
                imp_count += 1

            if not args.skip_perm:
                base_acc = _val_acc_with_mask(model, X_va_std, y_va, device, None)
                for c in range(n_ch):
                    a = _val_acc_with_mask(model, X_va_std, y_va, device, mask_idx=c)
                    perm_drop_accum[c] += (base_acc - a)
                perm_count += 1
            del model
        print(f"  {subject} done (n_folds processed)")

    imp_mean = imp_accum / max(imp_count, 1)
    perm_mean = perm_drop_accum / max(perm_count, 1) if perm_count else np.full(n_ch, np.nan)

    # rankings (descending — more important = higher)
    order_imp = np.argsort(-imp_mean)
    order_perm = np.argsort(-perm_mean) if perm_count else None

    # Spearman correlation between rankings
    spear = None
    if perm_count > 0:
        # use rank arrays
        rank_imp = np.empty_like(imp_mean)
        rank_imp[order_imp] = np.arange(n_ch)
        rank_perm = np.empty_like(perm_mean)
        rank_perm[order_perm] = np.arange(n_ch)
        diff2 = ((rank_imp - rank_perm) ** 2).sum()
        spear = 1 - (6 * diff2) / (n_ch * (n_ch ** 2 - 1))

    csv_path = out_dir / "channel_importance.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ch_idx", "ch_name", "space_imp_mean",
                    "perm_drop_mean", "rank_space", "rank_perm"])
        rank_imp_arr = np.empty_like(imp_mean); rank_imp_arr[order_imp] = np.arange(n_ch)
        if order_perm is not None:
            rank_perm_arr = np.empty_like(perm_mean); rank_perm_arr[order_perm] = np.arange(n_ch)
        else:
            rank_perm_arr = np.full(n_ch, -1)
        for c in range(n_ch):
            w.writerow([c, TRI_CHANNEL_NAMES[c],
                        f"{imp_mean[c]:.6f}",
                        f"{perm_mean[c]:.6f}",
                        int(rank_imp_arr[c]),
                        int(rank_perm_arr[c])])

    txt_lines = []
    txt_lines.append(f"# Channel importance for {args.exp}")
    txt_lines.append(f"  folds aggregated (space-imp): {imp_count}")
    txt_lines.append(f"  folds aggregated (perm)     : {perm_count}")
    if spear is not None:
        txt_lines.append(f"  Spearman corr (space-imp ↔ perm) = {spear:.3f}")
    K = 20
    txt_lines.append(f"\n## top-{K} by SpaceBranch importance")
    for i in range(K):
        c = order_imp[i]
        txt_lines.append(f"  {i+1:>2}. {TRI_CHANNEL_NAMES[c]:<6}  imp={imp_mean[c]:.4f}")
    if order_perm is not None:
        txt_lines.append(f"\n## top-{K} by permutation drop")
        for i in range(K):
            c = order_perm[i]
            txt_lines.append(f"  {i+1:>2}. {TRI_CHANNEL_NAMES[c]:<6}  drop={perm_mean[c]:.4f}")
    txt = "\n".join(txt_lines) + "\n"
    (out_dir / "channel_importance_plot.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()

"""Task 6 Step 2 — 4-branch (with cov) training runner.

Mirrors run_aug.py's protocol exactly but threads a per-fold, train-only
Riemannian tangent-space cov_feat through the model. Augmentations on the
raw signal are kept (mixup, etc.) but cov_feat is computed ONCE per fold
(from z-scored train signal) and used as-is for both clean training and
the mixed sample, since cov stats are per-trial and the cov vector is a
property of the trial identity.

This file ONLY runs when --tri-override cov_branch_enabled=True.
For cov_branch_enabled=False, use run_aug.py.
"""

import argparse
import ast
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from ablation_config import ABLATION_PRESETS, TRI_DEFAULTS  # noqa: E402
from augment import (  # noqa: E402
    DEFAULT_AUG, MixUp, build_sample_aug,
)
from configs import get_config  # noqa: E402
from cov_features import compute_cov_features, load_topk_channels  # noqa: E402
from data import (  # noqa: E402
    DATA_ROOT, SUBJECTS,
    load_subject, standardize_per_channel,
    stratified_kfold_train_val_test_splits,
)
from formatter import format_summary  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from model import build_model  # noqa: E402
from run_aug import (  # noqa: E402
    _make_optim, load_aug_config, parse_subjects, parse_tri_overrides,
)
from train import get_branch_decorr_loss, get_device, sample_branch_drop_mask, set_seed  # noqa: E402

MODEL_NAME = "tridomain_cov"


# -------- Cov-aware dataset --------
class CovTrainDataset(torch.utils.data.Dataset):
    """Holds (X, cov_feat, y) and applies per-sample aug to X only."""

    def __init__(self, X_np, cov_np, y_np, sample_aug=None):
        self.X = torch.from_numpy(X_np).float()
        self.cov = torch.from_numpy(cov_np).float()
        self.y = torch.from_numpy(y_np).long()
        self.sample_aug = sample_aug or []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        for aug in self.sample_aug:
            x = aug(x)
        return x, self.cov[idx], self.y[idx]


class CovEvalDataset(torch.utils.data.Dataset):
    def __init__(self, X_np, cov_np, y_np):
        self.X = torch.from_numpy(X_np).float()
        self.cov = torch.from_numpy(cov_np).float()
        self.y = torch.from_numpy(y_np).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.cov[idx], self.y[idx]


def make_train_loader_cov(X, cov, y, batch_size, aug_cfg, num_workers=2):
    aug = build_sample_aug(aug_cfg)
    ds = CovTrainDataset(X, cov, y, sample_aug=aug)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


def make_eval_loader_cov(X, cov, y, batch_size, num_workers=2):
    return torch.utils.data.DataLoader(
        CovEvalDataset(X, cov, y), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


# -------- Train / eval helpers (4-branch aware) --------
def _train_epoch_cov(model, loader, optim, n_classes, device, mixup,
                     aux_weights):
    model.train()
    total, loss_sum = 0, 0.0
    log_sm = nn.LogSoftmax(dim=-1)
    use_aux = (aux_weights is not None) and getattr(model, "aux_loss_enabled", False)
    for X, C, y in loader:
        X = X.to(device, non_blocking=True)
        C = C.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup is not None:
            # mixup on signals AND cov_feat together (per-trial linear blend
            # of cov_feat is sound in tangent space — Euclidean).
            lam = float(np.random.beta(mixup.alpha, mixup.alpha)) if mixup.alpha > 0 else 1.0
            perm = torch.randperm(X.size(0), device=X.device)
            X = lam * X + (1 - lam) * X[perm]
            C = lam * C + (1 - lam) * C[perm]
            yh = torch.zeros(y.size(0), n_classes, device=device).scatter_(1, y.view(-1, 1), 1.0)
            y_soft = lam * yh + (1 - lam) * yh[perm]
        else:
            y_soft = torch.zeros(y.size(0), n_classes, device=device).scatter_(1, y.view(-1, 1), 1.0)
        optim.zero_grad()
        branch_drop_mask = sample_branch_drop_mask(model, X.size(0), X.device, X.dtype)
        if use_aux:
            if branch_drop_mask is not None:
                logits, aux = model(X, return_aux=True, cov_feat=C,
                                    branch_drop_mask=branch_drop_mask)
            else:
                logits, aux = model(X, return_aux=True, cov_feat=C)
        else:
            if branch_drop_mask is not None:
                logits = model(X, cov_feat=C, branch_drop_mask=branch_drop_mask)
            else:
                logits = model(X, cov_feat=C)
            aux = None
        loss = -(y_soft * log_sm(logits)).sum(dim=1).mean()
        if use_aux and aux is not None and aux.get("aux_logits") is not None:
            for name, aux_logit in aux["aux_logits"].items():
                w = float(aux_weights.get(name, 0.0))
                if w > 0:
                    loss = loss + w * (
                        -(y_soft * log_sm(aux_logit)).sum(dim=1).mean()
                    )
        branch_decorr_loss = get_branch_decorr_loss(model, aux)
        if branch_decorr_loss is not None:
            loss = loss + branch_decorr_loss
        loss.backward()
        optim.step()
        loss_sum += loss.item() * y.size(0)
        total += y.size(0)
    return loss_sum / max(total, 1)


@torch.no_grad()
def _predict_cov(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, C, y in loader:
        X = X.to(device, non_blocking=True); C = C.to(device, non_blocking=True)
        logits = model(X, cov_feat=C)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fit_one_fold_cov(model, train_loader, val_loader, n_classes, train_cfg, device,
                     mixup, aux_weights, save_ckpt_path=None, save_meta=None,
                     swa_enabled=False, swa_start_frac=0.75):
    optim = _make_optim(model, train_cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg["epochs"])
    best_val_kappa = -float("inf")
    best_state = None
    best_epoch = -1
    epochs = int(train_cfg["epochs"])
    swa_model = None
    swa_start = epochs + 1
    if swa_enabled:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = max(1, int(epochs * swa_start_frac))
    for epoch in range(1, epochs + 1):
        _train_epoch_cov(model, train_loader, optim, n_classes, device, mixup, aux_weights)
        y_true, y_pred = _predict_cov(model, val_loader, device)
        _, val_kappa, _ = compute_metrics(y_true, y_pred, n_classes)
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if swa_enabled and epoch >= swa_start:
            swa_model.update_parameters(model)
        sched.step()
    if swa_enabled and swa_model is not None:
        from torch.optim.swa_utils import update_bn
        # Need a (x,) loader to update BN — workaround: wrap loader.
        # We do it manually since DataLoader emits (X,C,y).
        model.train()
        with torch.no_grad():
            momenta = {}
            for mod in swa_model.module.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    momenta[mod] = mod.momentum
                    mod.reset_running_stats()
                    mod.momentum = None
            for X, C, _ in train_loader:
                X = X.to(device); C = C.to(device)
                swa_model(X, cov_feat=C)
            for mod, m in momenta.items():
                mod.momentum = m
        model.load_state_dict(swa_model.module.state_dict())
        y_true, y_pred = _predict_cov(model, val_loader, device)
        _, swa_val_kappa, _ = compute_metrics(y_true, y_pred, n_classes)
        best_val_kappa = float(swa_val_kappa)
        best_epoch = -1
    else:
        model.load_state_dict(best_state)
    if save_ckpt_path is not None:
        os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "best_val_kappa": float(best_val_kappa),
            "best_epoch": int(best_epoch),
            "meta": dict(save_meta or {}),
            "swa_enabled": bool(swa_enabled),
        }, save_ckpt_path)
    return float(best_val_kappa), int(best_epoch)


def build_tri_cfg(n_channels, n_times, n_classes, ablation_name, seed,
                  tri_overrides=None, cov_dim=None):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation_name])
    if tri_overrides:
        values.update(tri_overrides)
    values["cov_dim"] = cov_dim
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes})
    return SimpleNamespace(**values)


def parse_aux_weights(tri_overrides):
    """Resolve aux loss weight(s). Accepts:
       - aux_loss_weight=0.3        (single float, all branches)
       - aux_loss_weight={"time":0.3,"freq":0.1,"space":0.4,"cov":0.4}
    """
    if not tri_overrides:
        return {}
    w = tri_overrides.get("aux_loss_weight", None)
    if w is None:
        return {}
    if isinstance(w, dict):
        return {k: float(v) for k, v in w.items()}
    # scalar -> uniform across (time, freq, space, cov)
    return {b: float(w) for b in ("time", "freq", "space", "cov")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--ablation", default="full_std_coords")
    p.add_argument("--aug-config", default=None)
    p.add_argument("--override", nargs="*", default=None)
    p.add_argument("--tri-override", nargs="*", default=None)
    p.add_argument("--subject", default="all")
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--results-root", default=str(THIS_DIR / "results" / "cov"))
    p.add_argument("--save-ckpts", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="if a per-fold ckpt already exists, skip training and "
                        "just load + test forward (cov_feat is still computed).")
    p.add_argument("--cov-topk", type=int, default=20)
    p.add_argument("--cov-band", type=float, nargs=2, default=(0.5, 8.0))
    p.add_argument("--cov-crop", type=float, nargs=2, default=(0.0, 2.0))
    p.add_argument("--cov-taps", type=int, default=251)
    p.add_argument("--channel-rank-csv",
                   default=str(THIS_DIR / "diagnostics" / "channel_importance.csv"))
    p.add_argument("--cov-cache-dir",
                   default=str(THIS_DIR / "results" / "cov" / "_cache"),
                   help="Shared cov-feature cache. Set to '' to disable.")
    args = p.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    train_cfg = get_config("tridomain")
    if args.epochs is not None: train_cfg["epochs"] = args.epochs
    if args.batch_size is not None: train_cfg["batch_size"] = args.batch_size
    if args.lr is not None: train_cfg["lr"] = args.lr
    if args.weight_decay is not None: train_cfg["weight_decay"] = args.weight_decay

    aug_cfg = load_aug_config(args.aug_config, args.override)
    tri_overrides = parse_tri_overrides(args.tri_override)
    # cov_branch_enabled must be True for this runner
    tri_overrides.setdefault("cov_branch_enabled", True)
    aux_weights = parse_aux_weights(tri_overrides)
    device = get_device(args.device)
    subjects = parse_subjects(args.subject)
    topk = int(args.cov_topk)
    topk_channels = load_topk_channels(args.channel_rank_csv, topk)

    cov_dim = (topk * (topk + 1)) // 2

    print(f"device: {device}\nexperiment: {args.exp_name}\nablation: {args.ablation}")
    print(f"subjects: {subjects}\ntrain_cfg: {train_cfg}\naug_cfg: {aug_cfg}")
    print(f"tri_overrides: {tri_overrides}\naux_weights: {aux_weights}")
    print(f"cov_topk={topk}  cov_dim={cov_dim}  topk_channels[:10]={topk_channels[:10]}")

    ckpt_dir = None
    if args.save_ckpts:
        ckpt_dir = Path(args.results_root) / args.exp_name / "ckpts"

    per_subj = {}
    n_classes_seen = None
    for subject in subjects:
        X, y = load_subject(subject)
        n_classes = int(y.max() + 1)
        n_channels, n_times = X.shape[1], X.shape[2]
        fold_results = []
        for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
            y, n_splits=args.n_folds, val_size=args.val_size, seed=args.seed,
        ):
            X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
            y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
            X_tr, X_va, X_te = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)

            cache_dir = args.cov_cache_dir if args.cov_cache_dir else None
            cache_id = f"{subject}_fold{fold_idx}" if cache_dir else None
            cov_tr, cov_va, cov_te = compute_cov_features(
                X_tr, X_va, X_te,
                fs=250, band=tuple(args.cov_band),
                crop_sec=tuple(args.cov_crop),
                topk_channels=topk_channels, taps=args.cov_taps, verbose=True,
                cache_dir=cache_dir, cache_id=cache_id,
            )

            train_loader = make_train_loader_cov(X_tr, cov_tr, y_tr,
                                                 train_cfg["batch_size"], aug_cfg,
                                                 num_workers=args.num_workers)
            val_loader = make_eval_loader_cov(X_va, cov_va, y_va,
                                              train_cfg["batch_size"],
                                              num_workers=args.num_workers)
            test_loader = make_eval_loader_cov(X_te, cov_te, y_te,
                                               train_cfg["batch_size"],
                                               num_workers=args.num_workers)

            set_seed(args.seed + fold_idx - 1)
            cfg_tri = build_tri_cfg(n_channels, n_times, n_classes, args.ablation,
                                    args.seed, tri_overrides=tri_overrides,
                                    cov_dim=cov_dim)
            model = build_model(cfg_tri, model_name="tridomain").to(device)
            mixup = MixUp(aug_cfg["mixup_alpha"]) if aug_cfg.get("mixup_enabled", False) else None
            swa_enabled = bool(tri_overrides.get("swa_enabled", False))
            swa_start_frac = float(tri_overrides.get("swa_start_frac", 0.75))

            save_ckpt_path = None
            if ckpt_dir is not None:
                save_ckpt_path = str(ckpt_dir / subject / f"fold{fold_idx}.pt")
            save_meta = {"ablation": args.ablation, "tri_overrides": tri_overrides,
                         "seed": args.seed, "cov_topk": topk, "cov_dim": cov_dim,
                         "topk_channels": topk_channels,
                         "cov_band": list(args.cov_band),
                         "cov_crop": list(args.cov_crop)}

            t0 = time.time()
            if args.resume and save_ckpt_path and Path(save_ckpt_path).exists():
                blob = torch.load(save_ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(blob["state_dict"])
                best_val_kappa = float(blob.get("best_val_kappa", 0.0))
                best_epoch = int(blob.get("best_epoch", -1))
                print(f"  {subject} fold{fold_idx}: [RESUME] loaded ckpt "
                      f"(val_k={best_val_kappa:.4f}@ep{best_epoch})")
            else:
                best_val_kappa, best_epoch = fit_one_fold_cov(
                    model, train_loader, val_loader, n_classes, train_cfg, device,
                    mixup, aux_weights, save_ckpt_path=save_ckpt_path,
                    save_meta=save_meta, swa_enabled=swa_enabled,
                    swa_start_frac=swa_start_frac,
                )
            y_true, y_pred = _predict_cov(model, test_loader, device)
            test_acc, test_kappa, test_pc = compute_metrics(y_true, y_pred, n_classes)
            elapsed = time.time() - t0
            fold_results.append({
                "fold": fold_idx,
                "acc": float(test_acc), "kappa": float(test_kappa),
                "per_class": test_pc.tolist(),
                "best_val_kappa": best_val_kappa, "best_epoch": best_epoch,
                "counts": {"train": int(len(tr_idx)), "val": int(len(va_idx)),
                           "test": int(len(te_idx))},
            })
            print(f"  {subject} f{fold_idx}/{args.n_folds}: "
                  f"acc={test_acc:.4f} k={test_kappa:.4f} "
                  f"val_k={best_val_kappa:.4f}@ep{best_epoch} ({elapsed:.0f}s)")
        accs = np.asarray([r["acc"] for r in fold_results])
        ks = np.asarray([r["kappa"] for r in fold_results])
        pcs = np.stack([np.asarray(r["per_class"]) for r in fold_results])
        per_subj[subject] = {
            "acc": float(accs.mean()), "acc_std": float(accs.std()),
            "kappa": float(ks.mean()), "kappa_std": float(ks.std()),
            "per_class": pcs.mean(axis=0).tolist(),
            "folds": fold_results,
        }
        print(f"  {subject} AVG: acc={accs.mean():.4f}±{accs.std():.4f}")
        n_classes_seen = n_classes

    extras = [
        f"Ablation: {args.ablation}",
        f"N folds: {args.n_folds}",
        f"cov_topk={topk}  cov_dim={cov_dim}  band={args.cov_band}  crop={args.cov_crop}s",
        f"aux_weights={aux_weights}",
    ]
    summary = format_summary(
        experiment=args.exp_name, model_name=MODEL_NAME,
        cv=f"{args.n_folds}fold",
        validation_desc=f"outer stratified {args.n_folds}-fold; train-only z-score + cov fit",
        metric_desc="fold test set, val-kappa ckpt",
        val_size=args.val_size, data_root=DATA_ROOT,
        n_classes=n_classes_seen, per_subject_results=per_subj,
        extra_lines=extras,
    )
    out_dir = Path(args.results_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    with (out_dir / "results.json").open("w") as f:
        json.dump({"experiment": args.exp_name, "model": MODEL_NAME,
                   "ablation": args.ablation, "n_folds": args.n_folds,
                   "val_size": args.val_size, "seed": args.seed,
                   "train_config": train_cfg, "aug_config": aug_cfg,
                   "tri_overrides": tri_overrides,
                   "cov_topk": topk, "cov_dim": cov_dim,
                   "topk_channels": topk_channels,
                   "per_subject": per_subj}, f, indent=2)
    print(summary)
    print(f"saved: {out_dir/'summary.txt'}")


if __name__ == "__main__":
    main()

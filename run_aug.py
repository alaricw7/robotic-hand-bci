"""Aug-aware n-fold runner for TriDomain.

Same protocol as run_ablation_5fold.py (stratified KFold, val_size,
val-kappa checkpoint selection, per-fold seed) plus:
  - configurable training augmentations (augment.py)
  - optional mixup at batch level
  - optional crop_voting test protocol

Inputs:
  --aug-config path/to/aug.json     (merges into augment.DEFAULT_AUG)
  --override key=value [...]        (key from augment.DEFAULT_AUG, parsed as python literal)

Outputs:
  results_root/<exp_name>/summary.txt + results.json
  Two AVG rows in summary: "AVG" (all subjects) and "AVG_excl_S2S3" via extras.
"""

import argparse
import ast
import copy
import json
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
    DEFAULT_AUG,
    MixUp,
    crop_vote_logits,
    make_eval_loader,
    make_train_loader,
)
from configs import get_config  # noqa: E402
from data import (  # noqa: E402
    DATA_ROOT,
    SUBJECTS,
    load_subject,
    standardize_per_channel,
    stratified_kfold_train_val_test_splits,
)
from formatter import format_summary  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from model import build_model  # noqa: E402
from train import get_branch_decorr_loss, get_device, sample_branch_drop_mask, set_seed  # noqa: E402


MODEL_NAME = "tridomain_aug"
DEFAULT_ABLATION = "full_std_coords"


def parse_subjects(value):
    if value == "all":
        return SUBJECTS
    subjects = []
    for item in value.replace(",", " ").split():
        item = item.strip()
        if not item:
            continue
        if item.lower().startswith("s") and item[1:].isdigit():
            subjects.append(f"S{int(item[1:])}")
        elif item.isdigit():
            subjects.append(f"S{int(item)}")
        else:
            subjects.append(item)
    return subjects


def load_aug_config(path, overrides):
    cfg = dict(DEFAULT_AUG)
    if path:
        with open(path) as f:
            user = json.load(f)
        cfg.update(user)
    for kv in (overrides or []):
        if "=" not in kv:
            raise ValueError(f"bad --override {kv}, expected key=value")
        k, v = kv.split("=", 1)
        cfg[k] = ast.literal_eval(v)
    if cfg["crop_enabled"]:
        if cfg["crop_len"] % TRI_DEFAULTS["tri_freq_windows"] != 0:
            raise ValueError(
                f"crop_len={cfg['crop_len']} must be divisible by "
                f"tri_freq_windows={TRI_DEFAULTS['tri_freq_windows']}"
            )
    return cfg


def build_tri_cfg(n_channels, n_times, n_classes, ablation_name, seed, tri_overrides=None):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation_name])
    if tri_overrides:
        values.update(tri_overrides)
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes})
    return SimpleNamespace(**values)


def parse_tri_overrides(items):
    """Parse ['key=value', ...] (python literals) for tri_* / dropout fields."""
    out = {}
    for kv in (items or []):
        if "=" not in kv:
            raise ValueError(f"bad --tri-override {kv}, expected key=value")
        k, v = kv.split("=", 1)
        out[k] = ast.literal_eval(v)
    return out


def _make_optim(model, cfg):
    name = cfg["optimizer"].lower()
    kwargs = dict(lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0),
                  betas=cfg.get("betas", (0.9, 0.999)))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(name)


def _train_epoch(model, loader, optim, n_classes, device, mixup, aux_loss_weight=0.0):
    """Soft-label cross-entropy so mixup is integrated naturally
    (one-hot when mixup disabled). If the model exposes per-branch aux
    heads (aux_logits is not None) and aux_loss_weight>0, add a
    soft-label CE for each aux head against the SAME mixed target.
    Aux heads do not affect inference or ckpt selection.
    """
    model.train()
    total, loss_sum = 0, 0.0
    log_softmax = nn.LogSoftmax(dim=-1)
    use_aux = aux_loss_weight > 0.0 and getattr(model, "aux_loss_enabled", False)
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup is not None:
            X, y_soft = mixup(X, y, n_classes)
        else:
            y_soft = torch.zeros(y.size(0), n_classes, device=device).scatter_(1, y.view(-1, 1), 1.0)
        optim.zero_grad()
        branch_drop_mask = sample_branch_drop_mask(model, X.size(0), X.device, X.dtype)
        if use_aux:
            if branch_drop_mask is not None:
                logits, aux = model(X, return_aux=True, branch_drop_mask=branch_drop_mask)
            else:
                logits, aux = model(X, return_aux=True)
        else:
            if branch_drop_mask is not None:
                logits = model(X, branch_drop_mask=branch_drop_mask)
            else:
                logits = model(X)
            aux = None
        loss = -(y_soft * log_softmax(logits)).sum(dim=1).mean()
        if use_aux and aux is not None and aux.get("aux_logits") is not None:
            for _name, aux_logit in aux["aux_logits"].items():
                loss = loss + aux_loss_weight * (
                    -(y_soft * log_softmax(aux_logit)).sum(dim=1).mean()
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
def _predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fit_one_fold(model, train_loader, val_loader, n_classes, train_cfg, device, mixup,
                 aux_loss_weight=0.0, save_ckpt_path=None, save_meta=None,
                 swa_enabled=False, swa_start_frac=0.75):
    optim = _make_optim(model, train_cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg["epochs"])
    best_val_kappa = -float("inf")
    best_state = None
    best_epoch = -1
    epochs = int(train_cfg["epochs"])

    # SWA setup (Task 5 B1). Default disabled = original path.
    swa_model = None
    swa_start = epochs + 1
    if swa_enabled:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start = max(1, int(epochs * float(swa_start_frac)))

    for epoch in range(1, epochs + 1):
        _train_epoch(model, train_loader, optim, n_classes, device, mixup,
                     aux_loss_weight=aux_loss_weight)
        # ckpt selection by main-head val-kappa (unchanged)
        y_true, y_pred = _predict(model, val_loader, device)
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
        # Recompute BN running stats over the train loader before evaluating.
        update_bn(train_loader, swa_model, device=device)
        # Copy SWA-averaged weights back into the user-facing model so the
        # subsequent ckpt-save and test eval use the averaged model.
        model.load_state_dict(swa_model.module.state_dict())
        # Recompute val-kappa on the SWA model purely for logging (NOT used
        # for selection — there is nothing to select with weight averaging).
        y_true, y_pred = _predict(model, val_loader, device)
        _, swa_val_kappa, _ = compute_metrics(y_true, y_pred, n_classes)
        best_val_kappa = float(swa_val_kappa)
        best_epoch = -1  # signal: SWA-averaged, no single best-epoch
    else:
        model.load_state_dict(best_state)

    if save_ckpt_path is not None:
        import os
        os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "best_val_kappa": float(best_val_kappa),
            "best_epoch": int(best_epoch),
            "meta": dict(save_meta or {}),
            "swa_enabled": bool(swa_enabled),
        }, save_ckpt_path)
    return float(best_val_kappa), int(best_epoch)


def run_one_subject(subject, train_cfg, aug_cfg, args, device):
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

        train_loader = make_train_loader(X_tr, y_tr, train_cfg["batch_size"], aug_cfg,
                                         num_workers=args.num_workers)
        val_loader = make_eval_loader(X_va, y_va, train_cfg["batch_size"], num_workers=args.num_workers)
        test_loader = make_eval_loader(X_te, y_te, train_cfg["batch_size"], num_workers=args.num_workers)

        set_seed(args.seed + fold_idx - 1)
        cfg_tri = build_tri_cfg(n_channels, n_times, n_classes, args.ablation, args.seed,
                                tri_overrides=args._tri_overrides)
        model = build_model(cfg_tri, model_name="tridomain").to(device)
        mixup = MixUp(aug_cfg["mixup_alpha"]) if aug_cfg.get("mixup_enabled", False) else None

        aux_loss_weight = float(
            args._tri_overrides.get("aux_loss_weight", 0.0)
            if args._tri_overrides else 0.0
        )
        swa_enabled = bool(
            args._tri_overrides.get("swa_enabled", False)
            if args._tri_overrides else False
        )
        swa_start_frac = float(
            args._tri_overrides.get("swa_start_frac", 0.75)
            if args._tri_overrides else 0.75
        )
        save_ckpt_path = None
        if getattr(args, "_ckpt_dir", None) is not None:
            save_ckpt_path = str(Path(args._ckpt_dir) / subject / f"fold{fold_idx}.pt")
        save_meta = {
            "ablation": args.ablation,
            "tri_overrides": dict(args._tri_overrides or {}),
            "seed": args.seed,
        }
        t0 = time.time()
        if getattr(args, "resume", False) and save_ckpt_path \
                and Path(save_ckpt_path).exists():
            blob = torch.load(save_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"])
            best_val_kappa = float(blob.get("best_val_kappa", 0.0))
            best_epoch = int(blob.get("best_epoch", -1))
            print(f"  {subject} fold{fold_idx}: [RESUME] loaded ckpt "
                  f"(val_k={best_val_kappa:.4f}@ep{best_epoch})")
        else:
            best_val_kappa, best_epoch = fit_one_fold(
                model, train_loader, val_loader, n_classes, train_cfg, device, mixup,
                aux_loss_weight=aux_loss_weight,
                save_ckpt_path=save_ckpt_path,
                save_meta=save_meta,
                swa_enabled=swa_enabled,
                swa_start_frac=swa_start_frac,
            )

        # Test under chosen protocol
        proto = aug_cfg.get("test_protocol", "full_trial")
        if proto == "crop_voting" and aug_cfg.get("crop_enabled", False):
            y_pred, _ = crop_vote_logits(model, X_te, aug_cfg, device,
                                         batch_size=train_cfg["batch_size"])
            y_true = y_te
        else:
            y_true, y_pred = _predict(model, test_loader, device)
        test_acc, test_kappa, test_per_class = compute_metrics(y_true, y_pred, n_classes)
        elapsed = time.time() - t0

        fold_results.append({
            "fold": fold_idx,
            "acc": float(test_acc),
            "kappa": float(test_kappa),
            "per_class": test_per_class.tolist(),
            "best_val_kappa": best_val_kappa,
            "best_epoch": best_epoch,
            "counts": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        })
        print(f"  {subject} fold{fold_idx}/{args.n_folds}: "
              f"acc={test_acc:.4f} kappa={test_kappa:.4f} "
              f"val_kappa={best_val_kappa:.4f}@ep{best_epoch} (t={elapsed:.1f}s)")

    accs = np.asarray([r["acc"] for r in fold_results])
    kappas = np.asarray([r["kappa"] for r in fold_results])
    per_class = np.stack([np.asarray(r["per_class"]) for r in fold_results])
    summary = {
        "acc": float(accs.mean()), "acc_std": float(accs.std()),
        "kappa": float(kappas.mean()), "kappa_std": float(kappas.std()),
        "per_class": per_class.mean(axis=0).tolist(),
        "folds": fold_results,
    }
    print(f"  {subject} {args.n_folds}-fold mean: "
          f"acc={summary['acc']:.4f}±{summary['acc_std']:.4f} "
          f"kappa={summary['kappa']:.4f}±{summary['kappa_std']:.4f}")
    return summary, n_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="aug_run")
    parser.add_argument("--ablation", default=DEFAULT_ABLATION,
                        choices=sorted(ABLATION_PRESETS))
    parser.add_argument("--aug-config", default=None,
                        help="JSON file overriding augment.DEFAULT_AUG")
    parser.add_argument("--override", nargs="*", default=None,
                        help="key=value pairs for aug cfg (python literals)")
    parser.add_argument("--tri-override", nargs="*", default=None,
                        help="key=value pairs for TRI_DEFAULTS / dropout (python literals)")
    parser.add_argument("--save-ckpts", action="store_true",
                        help="save best-val-kappa state per fold to "
                             "results_root/<exp>/ckpts/<subject>/foldK.pt")
    parser.add_argument("--resume", action="store_true",
                        help="if a per-fold ckpt already exists, skip training "
                             "for that fold; just load and run test forward.")
    parser.add_argument("--subject", default="all")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--results-root", default=str(THIS_DIR / "results" / "aug_hpo"))
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))

    train_cfg = get_config("tridomain")
    if args.epochs is not None: train_cfg["epochs"] = args.epochs
    if args.batch_size is not None: train_cfg["batch_size"] = args.batch_size
    if args.lr is not None: train_cfg["lr"] = args.lr
    if args.weight_decay is not None: train_cfg["weight_decay"] = args.weight_decay

    aug_cfg = load_aug_config(args.aug_config, args.override)
    args._tri_overrides = parse_tri_overrides(args.tri_override)
    args._ckpt_dir = (
        str(Path(args.results_root) / args.exp_name / "ckpts")
        if args.save_ckpts else None
    )
    device = get_device(args.device)
    subjects = parse_subjects(args.subject)
    print(f"device: {device}\nexperiment: {args.exp_name}\nablation: {args.ablation}")
    print(f"subjects: {subjects}\ntrain_cfg: {train_cfg}\naug_cfg: {aug_cfg}")

    per_subj = {}
    n_classes_seen = None
    for s in subjects:
        out, n_classes = run_one_subject(s, train_cfg, aug_cfg, args, device)
        n_classes_seen = n_classes
        per_subj[s] = out

    extra = [
        f"Ablation: {args.ablation}",
        f"N folds: {args.n_folds}",
        f"Aug config: {aug_cfg}",
    ]

    summary = format_summary(
        experiment=args.exp_name, model_name=MODEL_NAME,
        cv=f"{args.n_folds}fold",
        validation_desc=f"outer stratified {args.n_folds}-fold test; "
                        f"stratified val split from train_val",
        metric_desc=f"fold test set, checkpoint selected by val kappa "
                    f"(test protocol: {aug_cfg.get('test_protocol', 'full_trial')})",
        val_size=args.val_size, data_root=DATA_ROOT,
        n_classes=n_classes_seen, per_subject_results=per_subj,
        extra_lines=extra,
    )

    out_dir = Path(args.results_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    with (out_dir / "results.json").open("w") as f:
        json.dump({
            "experiment": args.exp_name, "model": MODEL_NAME,
            "ablation": args.ablation,
            "cv": f"{args.n_folds}fold", "n_folds": args.n_folds,
            "val_size": args.val_size, "seed": args.seed,
            "train_config": train_cfg, "aug_config": aug_cfg,
            "tri_overrides": dict(args._tri_overrides or {}),
            "per_subject": per_subj,
        }, f, indent=2)
    print(summary)
    print(f"saved: {out_dir / 'summary.txt'}")
    print(f"saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

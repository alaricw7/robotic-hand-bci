"""Random-search HPO over augmentation + optimizer knobs.

Runs on a subset of folds (default: fold_idx 1,2,3 over a subset of subjects)
for speed; selects by VAL kappa (test untouched). Best config is then meant
to be run with run_aug.py on the full 10-fold protocol.

Outputs (under results_root/<study_name>/):
  hpo_trials.csv       -- one row per (trial, fold, subject) with cfg + val/test
  hpo_trials_agg.csv   -- one row per trial: mean val_kappa across (subject, fold)
  best_config.json     -- selected aug + train cfg
"""

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

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
    SUBJECTS,
    load_subject,
    standardize_per_channel,
    stratified_kfold_train_val_test_splits,
)
from metrics import compute_metrics  # noqa: E402
from model import build_model  # noqa: E402
from run_aug import _predict, fit_one_fold  # noqa: E402
from train import get_device, set_seed  # noqa: E402


SEARCH_SPACE = {
    "lr":              [3e-4, 5e-4, 1e-3, 2e-3],
    "weight_decay":    [0.0, 1e-5, 1e-4, 5e-4],
    "dropout":         [0.2, 0.3, 0.4, 0.5],
    "crop_enabled":    [False, True],
    "crop_len":        [500, 750, 1000],   # 4-divisible
    "crop_stride":     [125, 250, 500],
    "freqmask_enabled":[False, True],
    "freqmask_p":      [0.3, 0.5],
    "freqmask_bw_hz":  [4.0, 6.0, 8.0],
    "chdrop_enabled":  [False, True],
    "chdrop_p":        [0.1, 0.2, 0.3],
    "noise_enabled":   [False, True],
    "noise_std":       [0.02, 0.05, 0.1],
    "mixup_enabled":   [False, True],
    "mixup_alpha":     [0.1, 0.2, 0.4],
}


def sample_trial(rng):
    cfg = dict(DEFAULT_AUG)
    train_overrides = {}
    train_overrides["lr"] = rng.choice(SEARCH_SPACE["lr"])
    train_overrides["weight_decay"] = rng.choice(SEARCH_SPACE["weight_decay"])
    dropout = rng.choice(SEARCH_SPACE["dropout"])

    cfg["crop_enabled"] = rng.choice(SEARCH_SPACE["crop_enabled"])
    if cfg["crop_enabled"]:
        cfg["crop_len"] = int(rng.choice(SEARCH_SPACE["crop_len"]))
        cfg["crop_stride"] = int(rng.choice(SEARCH_SPACE["crop_stride"]))

    cfg["freqmask_enabled"] = rng.choice(SEARCH_SPACE["freqmask_enabled"])
    if cfg["freqmask_enabled"]:
        cfg["freqmask_p"] = float(rng.choice(SEARCH_SPACE["freqmask_p"]))
        cfg["freqmask_bw_hz"] = float(rng.choice(SEARCH_SPACE["freqmask_bw_hz"]))

    cfg["chdrop_enabled"] = rng.choice(SEARCH_SPACE["chdrop_enabled"])
    if cfg["chdrop_enabled"]:
        cfg["chdrop_p"] = float(rng.choice(SEARCH_SPACE["chdrop_p"]))

    cfg["noise_enabled"] = rng.choice(SEARCH_SPACE["noise_enabled"])
    if cfg["noise_enabled"]:
        cfg["noise_std"] = float(rng.choice(SEARCH_SPACE["noise_std"]))

    cfg["mixup_enabled"] = rng.choice(SEARCH_SPACE["mixup_enabled"])
    if cfg["mixup_enabled"]:
        cfg["mixup_alpha"] = float(rng.choice(SEARCH_SPACE["mixup_alpha"]))

    return cfg, train_overrides, float(dropout)


def build_tri_cfg(n_channels, n_times, n_classes, ablation_name, seed, dropout):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation_name])
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes,
                   "dropout": dropout})
    return SimpleNamespace(**values)


def evaluate_trial(trial_id, aug_cfg, train_cfg, dropout, args, device, target_folds, subjects):
    rows = []
    val_kappas = []
    test_accs = []
    test_kappas = []
    for subject in subjects:
        X, y = load_subject(subject)
        n_classes = int(y.max() + 1)
        n_channels, n_times = X.shape[1], X.shape[2]
        for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
            y, n_splits=args.n_folds, val_size=args.val_size, seed=args.seed,
        ):
            if fold_idx not in target_folds:
                continue
            X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
            y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
            X_tr, X_va, X_te = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)
            train_loader = make_train_loader(X_tr, y_tr, train_cfg["batch_size"], aug_cfg,
                                             num_workers=args.num_workers)
            val_loader = make_eval_loader(X_va, y_va, train_cfg["batch_size"],
                                          num_workers=args.num_workers)
            test_loader = make_eval_loader(X_te, y_te, train_cfg["batch_size"],
                                           num_workers=args.num_workers)

            set_seed(args.seed + fold_idx - 1)
            cfg_tri = build_tri_cfg(n_channels, n_times, n_classes,
                                    args.ablation, args.seed, dropout)
            model = build_model(cfg_tri, model_name="tridomain").to(device)
            mixup = MixUp(aug_cfg["mixup_alpha"]) if aug_cfg.get("mixup_enabled", False) else None

            t0 = time.time()
            best_val_kappa, best_epoch = fit_one_fold(
                model, train_loader, val_loader, n_classes, train_cfg, device, mixup,
            )

            if aug_cfg.get("test_protocol", "full_trial") == "crop_voting" \
                    and aug_cfg.get("crop_enabled", False):
                y_pred, _ = crop_vote_logits(model, X_te, aug_cfg, device,
                                             batch_size=train_cfg["batch_size"])
                y_true = y_te
            else:
                y_true, y_pred = _predict(model, test_loader, device)
            test_acc, test_kappa, _ = compute_metrics(y_true, y_pred, n_classes)
            elapsed = time.time() - t0

            val_kappas.append(best_val_kappa)
            test_accs.append(test_acc)
            test_kappas.append(test_kappa)
            rows.append({
                "trial": trial_id, "subject": subject, "fold": fold_idx,
                "val_kappa": best_val_kappa, "test_acc": test_acc, "test_kappa": test_kappa,
                "best_epoch": best_epoch, "elapsed_s": round(elapsed, 1),
                **{f"aug_{k}": aug_cfg[k] for k in aug_cfg},
                "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"],
                "dropout": dropout,
            })
            print(f"  [t{trial_id:03d}] {subject} f{fold_idx}: "
                  f"val_k={best_val_kappa:.3f} test_acc={test_acc:.3f} "
                  f"test_k={test_kappa:.3f} ({elapsed:.0f}s)")
    return rows, (np.mean(val_kappas), np.mean(test_accs), np.mean(test_kappas))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="hpo_v1")
    parser.add_argument("--ablation", default="full_std_coords")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--target-folds", type=int, nargs="+", default=[1, 2, 3],
                        help="fold ids (1-indexed) used per trial")
    parser.add_argument("--subjects", nargs="+",
                        default=["S6", "S8", "S10"],
                        help="HPO uses easy + mid-easy subjects so signal is visible.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="reduced epochs for HPO; final run uses 200")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--results-root", default=str(THIS_DIR / "results" / "aug_hpo"))
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))

    out_dir = Path(args.results_root) / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    rng = random.Random(args.rng_seed)
    base_cfg = get_config("tridomain")
    base_cfg["epochs"] = args.epochs
    base_cfg["batch_size"] = args.batch_size
    print(f"device={device} study={args.study_name} trials={args.n_trials} "
          f"folds={args.target_folds} subjects={args.subjects} epochs={args.epochs}")

    all_rows = []
    agg_rows = []
    for trial_id in range(1, args.n_trials + 1):
        aug_cfg, train_overrides, dropout = sample_trial(rng)
        train_cfg = copy.deepcopy(base_cfg)
        train_cfg.update(train_overrides)
        if aug_cfg["crop_enabled"] and \
                aug_cfg["crop_len"] % TRI_DEFAULTS["tri_freq_windows"] != 0:
            aug_cfg["crop_len"] = (
                aug_cfg["crop_len"] // TRI_DEFAULTS["tri_freq_windows"]
            ) * TRI_DEFAULTS["tri_freq_windows"]
        print(f"\n=== trial {trial_id}/{args.n_trials}  train_cfg={train_cfg}  "
              f"dropout={dropout}  aug_cfg={aug_cfg}")
        rows, (vk, ta, tk) = evaluate_trial(trial_id, aug_cfg, train_cfg, dropout,
                                            args, device, set(args.target_folds),
                                            args.subjects)
        all_rows.extend(rows)
        agg_rows.append({
            "trial": trial_id, "val_kappa_mean": vk,
            "test_acc_mean": ta, "test_kappa_mean": tk,
            "lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"],
            "dropout": dropout, **{f"aug_{k}": aug_cfg[k] for k in aug_cfg},
        })
        print(f"=== trial {trial_id} mean: val_k={vk:.3f} test_acc={ta:.3f} test_k={tk:.3f}")

        # Incrementally save in case of interruption
        with (out_dir / "hpo_trials.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wr.writeheader(); wr.writerows(all_rows)
        with (out_dir / "hpo_trials_agg.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            wr.writeheader(); wr.writerows(agg_rows)

    # Pick best by val_kappa_mean
    best = max(agg_rows, key=lambda r: r["val_kappa_mean"])
    best_aug = {k.removeprefix("aug_"): v for k, v in best.items() if k.startswith("aug_")}
    best_train = {"lr": best["lr"], "weight_decay": best["weight_decay"],
                  "dropout": best["dropout"]}
    with (out_dir / "best_config.json").open("w") as f:
        json.dump({
            "selected_by": "val_kappa_mean",
            "best_trial": best["trial"],
            "val_kappa_mean": best["val_kappa_mean"],
            "test_acc_mean": best["test_acc_mean"],
            "test_kappa_mean": best["test_kappa_mean"],
            "aug_cfg": best_aug,
            "train_overrides": best_train,
        }, f, indent=2)
    print("\n=== BEST trial", best["trial"], "===")
    print(json.dumps({"aug_cfg": best_aug, "train_overrides": best_train}, indent=2))
    print(f"saved: {out_dir / 'hpo_trials.csv'}")
    print(f"saved: {out_dir / 'hpo_trials_agg.csv'}")
    print(f"saved: {out_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()

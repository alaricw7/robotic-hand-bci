"""Run one TriDomain ablation with the selfmodel n-fold protocol.

Self-contained inside ``selfmodel/tri_domain``: data loading, split policy,
standardization, optimizer defaults, cosine schedule, and val-kappa
checkpoint selection are all local — no parent-directory imports.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

try:
    from .ablation_config import ABLATION_PRESETS, TRI_DEFAULTS
    from .configs import get_config
    from .data import (
        DATA_ROOT,
        SUBJECTS,
        load_subject,
        make_loader,
        standardize_per_channel,
        stratified_kfold_train_val_test_splits,
    )
    from .formatter import format_summary
    from .model import build_model
    from .train import fit_select_test, get_device, set_seed
except ImportError:
    from ablation_config import ABLATION_PRESETS, TRI_DEFAULTS
    from configs import get_config
    from data import (
        DATA_ROOT,
        SUBJECTS,
        load_subject,
        make_loader,
        standardize_per_channel,
        stratified_kfold_train_val_test_splits,
    )
    from formatter import format_summary
    from model import build_model
    from train import fit_select_test, get_device, set_seed


MODEL_NAME = "tridomain_ablation"
VALIDATION_DESC = "outer stratified n-fold test; stratified val split from train_val"
METRIC_DESC = "fold test set, checkpoint selected by val kappa"


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


def build_tri_cfg(n_channels, n_times, n_classes, ablation_name, overrides):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation_name])
    values.update(overrides)
    values.update(
        {
            "n_channels": n_channels,
            "n_samples": n_times,
            "n_classes": n_classes,
        }
    )
    return SimpleNamespace(**values)


def summarize_folds(fold_results):
    accs = np.asarray([r["acc"] for r in fold_results], dtype=float)
    kappas = np.asarray([r["kappa"] for r in fold_results], dtype=float)
    per_class = np.stack([np.asarray(r["per_class"], dtype=float) for r in fold_results], axis=0)
    return {
        "acc": float(accs.mean()),
        "acc_std": float(accs.std()),
        "kappa": float(kappas.mean()),
        "kappa_std": float(kappas.std()),
        "per_class": per_class.mean(axis=0).tolist(),
        "folds": fold_results,
    }


def run_one_subject(
    subject,
    ablation_name,
    train_cfg,
    n_folds,
    val_size,
    num_workers,
    seed,
    device,
    verbose,
    tri_overrides,
    ckpt_dir=None,
):
    X, y = load_subject(subject)
    n_classes = int(y.max() + 1)
    n_channels, n_times = X.shape[1], X.shape[2]
    fold_results = []

    for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
        y,
        n_splits=n_folds,
        val_size=val_size,
        seed=seed,
    ):
        X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
        y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
        X_tr, X_va, X_te = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)

        train_loader = make_loader(
            X_tr,
            y_tr,
            train_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = make_loader(
            X_va,
            y_va,
            train_cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = make_loader(
            X_te,
            y_te,
            train_cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )

        set_seed(seed + fold_idx - 1)
        tri_cfg = build_tri_cfg(n_channels, n_times, n_classes, ablation_name, tri_overrides)
        model = build_model(tri_cfg, model_name="tridomain")
        save_ckpt_path = None
        if ckpt_dir is not None:
            save_ckpt_path = str(Path(ckpt_dir) / subject / f"fold{fold_idx}.pt")
        out = fit_select_test(
            model,
            train_loader,
            val_loader,
            test_loader,
            n_classes=n_classes,
            cfg=train_cfg,
            device=device,
            verbose=verbose,
            tag=f"{MODEL_NAME}-{ablation_name}-{subject}-fold{fold_idx}",
            save_ckpt_path=save_ckpt_path,
        )
        result = {
            "fold": fold_idx,
            "acc": out["test_acc"],
            "kappa": out["test_kappa"],
            "per_class": out["test_per_class"].tolist(),
            "best_val_kappa": out["best_val_kappa"],
            "best_epoch": out["best_epoch"],
            "counts": {
                "train": int(len(tr_idx)),
                "val": int(len(va_idx)),
                "test": int(len(te_idx)),
            },
        }
        fold_results.append(result)
        print(
            f"  {subject} fold {fold_idx}/{n_folds}: "
            f"acc={out['test_acc']:.4f} kappa={out['test_kappa']:.4f} "
            f"best_val_kappa={out['best_val_kappa']:.4f}@ep{out['best_epoch']} "
            f"(t={out['elapsed']:.1f}s)"
        )

    summary = summarize_folds(fold_results)
    print(
        f"  {subject} {n_folds}-fold mean: "
        f"acc={summary['acc']:.4f}±{summary['acc_std']:.4f} "
        f"kappa={summary['kappa']:.4f}±{summary['kappa_std']:.4f}"
    )
    return summary, n_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=sorted(ABLATION_PRESETS), required=True)
    parser.add_argument("--subject", default="all", help="'all', 'S1', '1', or comma/space list")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results-root", default=str(THIS_DIR / "results"))
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--tri-elp-path", default=None)
    parser.add_argument("--no-normalize-coords", action="store_true")
    parser.add_argument("--save-ckpts", action="store_true",
                        help="save best-val-kappa state per fold under results/<exp>/ckpts/<subject>/foldK.pt")
    args = parser.parse_args()

    if args.torch_threads is not None:
        import torch

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))

    train_cfg = get_config("tridomain")
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr

    tri_overrides = {"random_seed": args.seed}
    if args.tri_elp_path is not None:
        tri_overrides["tri_elp_path"] = args.tri_elp_path
    if args.no_normalize_coords:
        tri_overrides["tri_normalize_coords"] = False

    exp_name = args.exp_name or f"abl_{args.ablation}"
    ckpt_dir = None
    if args.save_ckpts:
        ckpt_dir = str(Path(args.results_root) / exp_name / "ckpts")
    device = get_device(args.device)
    subjects = parse_subjects(args.subject)
    print(f"device: {device}")
    print(f"experiment: {exp_name}")
    print(f"ablation: {args.ablation}")
    print(f"subjects: {subjects}")
    print(f"train_cfg: {train_cfg}")
    print(
        "protocol: 0-4s crop, stratified n-fold test, "
        "stratified val from train_val, train-only z-score, val-kappa checkpoint"
    )

    per_subj = {}
    n_classes_seen = None
    for subject in subjects:
        out, n_classes = run_one_subject(
            subject,
            args.ablation,
            train_cfg,
            n_folds=args.n_folds,
            val_size=args.val_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            verbose=args.verbose,
            tri_overrides=tri_overrides,
            ckpt_dir=ckpt_dir,
        )
        n_classes_seen = n_classes
        per_subj[subject] = out

    summary = format_summary(
        experiment=exp_name,
        model_name=MODEL_NAME,
        cv=f"{args.n_folds}fold",
        validation_desc=VALIDATION_DESC.replace("n-fold", f"{args.n_folds}-fold"),
        metric_desc=METRIC_DESC,
        val_size=args.val_size,
        data_root=DATA_ROOT,
        n_classes=n_classes_seen,
        per_subject_results=per_subj,
        extra_lines=[
            f"Ablation: {args.ablation}",
            f"N folds: {args.n_folds}",
            "Training config source: selfmodel/tri_domain/configs.py:tridomain",
        ],
    )

    out_dir = Path(args.results_root) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    json_path = out_dir / "results.json"
    summary_path.write_text(summary, encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": exp_name,
                "model": MODEL_NAME,
                "ablation": args.ablation,
                "cv": f"{args.n_folds}fold",
                "n_folds": args.n_folds,
                "val_size": args.val_size,
                "seed": args.seed,
                "train_config": train_cfg,
                "tri_overrides": tri_overrides,
                "per_subject": per_subj,
            },
            f,
            indent=2,
        )
    print(summary)
    print(f"saved: {summary_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()

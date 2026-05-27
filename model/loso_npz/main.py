"""
main.py - LOSO experiment entry point for preprocessing NPZ data.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from config import Config
from data import get_loso_loaders, load_subject
from model import build_model
from train import evaluate, per_class_accuracy, train_one_fold


def config_to_dict(cfg):
    cfg_dict = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            cfg_dict[name] = value
        else:
            cfg_dict[name] = str(value)
    return cfg_dict


def _nan_to_none(obj):
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _history_values(history, key):
    return [np.nan if value is None else float(value) for value in history.get(key, [])]


def save_training_history_csv(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys = [
        "train_loss",
        "val_acc",
        "test_acc",
        "val_kappa",
        "test_kappa",
        "val_macro_f1",
        "val_score",
    ]
    n_epochs = max((len(history.get(key, [])) for key in keys), default=0)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for epoch in range(n_epochs):
            row = [epoch + 1]
            for key in keys:
                values = history.get(key, [])
                row.append(values[epoch] if epoch < len(values) else "")
            writer.writerow(row)


def save_training_curves_png(history, out_path, title, best_epoch=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return

    panels = [
        ("train_loss", "Train loss", "Loss"),
        ("val_acc", "Validation accuracy", "Accuracy"),
        ("test_acc", "Test accuracy", "Accuracy"),
        ("val_kappa", "Validation kappa", "Kappa"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
    for ax, (key, label, ylabel) in zip(axes.ravel(), panels):
        values = _history_values(history, key)
        ax.plot(epochs, values, linewidth=1.8)
        if best_epoch is not None and best_epoch > 0:
            ax.axvline(best_epoch, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_loso(heldout_subjects, all_subjects, cfg, model_name, exp_dir, device):
    """Run leave-one-subject-out evaluation."""
    print("\n===== LOSO =====")

    subject_data = {}
    for sid in all_subjects:
        X, y = load_subject(sid, cfg.data_root, cfg)
        subject_data[sid] = (X, y)
        print(f"  Loaded S{sid}: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    summaries = []
    all_preds, all_labels = [], []

    for heldout in heldout_subjects:
        print(f"\n  --- Held-out S{heldout} ---")
        train_subjects = [sid for sid in all_subjects if sid != heldout]
        train_subject_data = [
            (sid, subject_data[sid][0], subject_data[sid][1])
            for sid in train_subjects
        ]
        n_train_val_trials = sum(len(subject_data[sid][1]) for sid in train_subjects)
        X_test, y_test = subject_data[heldout]

        print(
            f"  Train subjects={train_subjects}, "
            f"train_val_trials={n_train_val_trials}, X_test={X_test.shape}"
        )

        train_loader, val_loader, test_loader, split_info = get_loso_loaders(
            train_subject_data,
            X_test,
            y_test,
            cfg,
        )
        counts = split_info["counts"]
        class_counts = split_info["class_counts"]
        per_subject_split_counts = {
            str(sid): {
                "counts": subject_split["counts"],
                "class_counts": subject_split["class_counts"],
            }
            for sid, subject_split in split_info.get("per_subject_splits", {}).items()
        }
        print(
            f"  Split: train={counts['train']}, val={counts['val']}, test={counts['test']} "
            f"| test class counts={class_counts['test']}"
        )
        print(f"  Val per train subject={ {sid: item['counts']['val'] for sid, item in per_subject_split_counts.items()} }")

        model = build_model(cfg, model_name=model_name)
        checkpoint_path = os.path.join(
            exp_dir,
            "checkpoints",
            f"loso_heldout_s{heldout}.pt",
        )
        checkpoint_meta = {
            "cv": "loso",
            "heldout_subject": heldout,
            "train_subjects": train_subjects,
            "model_name": model_name,
            "config": config_to_dict(cfg),
            "split_info": split_info,
        }

        result = train_one_fold(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            checkpoint_path=checkpoint_path,
            checkpoint_meta=checkpoint_meta,
            test_loader=test_loader,
        )

        test_acc, test_kappa, test_preds, test_labels = evaluate(model, test_loader, device)
        per_cls = per_class_accuracy(test_preds, test_labels, cfg.n_classes)
        val_per_cls = per_class_accuracy(result["preds"], result["labels"], cfg.n_classes)
        curve_png = os.path.join(
            exp_dir,
            "training_curves",
            f"loso_heldout_s{heldout}.png",
        )
        curve_csv = os.path.join(
            exp_dir,
            "training_curves",
            f"loso_heldout_s{heldout}.csv",
        )
        save_training_curves_png(
            result["history"],
            curve_png,
            title=f"LOSO held-out S{heldout} training curves",
            best_epoch=result["best_epoch"],
        )
        save_training_history_csv(result["history"], curve_csv)

        summary = {
            "subject": heldout,
            "heldout_subject": heldout,
            "train_subjects": train_subjects,
            "mean_acc": float(test_acc),
            "mean_kappa": float(test_kappa),
            "std_acc": 0.0,
            "std_kappa": 0.0,
            "per_class_acc": per_cls.tolist(),
            "folds": [
                {
                    "fold": "loso",
                    "test_acc": float(test_acc),
                    "test_kappa": float(test_kappa),
                    "best_val_acc": float(result["best_acc"]),
                    "best_val_kappa": float(result["best_kappa"]),
                    "best_val_macro_f1": float(result["best_macro_f1"]),
                    "best_val_score": float(result["best_score"]),
                    "early_stop_metric": result["early_stop_metric"],
                    "best_epoch": int(result["best_epoch"]),
                    "per_class_acc": per_cls.tolist(),
                    "val_per_class_acc": val_per_cls.tolist(),
                    "split_info": split_info,
                    "split_counts": counts,
                    "split_class_counts": class_counts,
                    "per_subject_split_counts": per_subject_split_counts,
                    "checkpoint": result["checkpoint_path"],
                    "history": result["history"],
                    "training_curve_png": curve_png,
                    "training_curve_csv": curve_csv,
                }
            ],
        }

        summaries.append(summary)
        all_preds.append(test_preds)
        all_labels.append(test_labels)
        print(
            f"  >>> Held-out S{heldout}: best_val_{result['early_stop_metric']}={result['best_score']:.4f}, "
            f"best_val_acc={result['best_acc']:.4f}, "
            f"test_acc={test_acc:.4f}, test_kappa={test_kappa:.4f}, "
            f"curves={curve_png}"
        )

    cm = confusion_matrix(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        labels=list(range(cfg.n_classes)),
    )
    np.save(os.path.join(exp_dir, "confusion_loso.npy"), cm)
    return summaries


def append_summary(summary_path, exp_name, model_name, cfg, all_summaries):
    """Append one run summary to summary.txt instead of overwriting it."""
    summary_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0

    with open(summary_path, "a", encoding="utf-8") as f:
        if summary_exists:
            f.write("\n" + "=" * 80 + "\n\n")

        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write("CV: loso\n")
        f.write("Validation: per-class chronological last val_size within each train subject\n")
        f.write(f"Metric: held-out subject test set, checkpoint selected by val {getattr(cfg, 'early_stop_metric', 'kappa')}\n")
        f.write(f"Val size: {getattr(cfg, 'val_size', 0.1)}\n")
        f.write(f"Data root: {cfg.data_root}\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Subject':<10}{'Acc':<18}{'Kappa':<18}"
            + "".join([f"T{i + 1:<7}" for i in range(cfg.n_classes)])
            + "\n"
        )
        f.write("-" * 80 + "\n")

        for summary in all_summaries:
            f.write(
                f"S{summary['subject']:<9}"
                f"{summary['mean_acc']:.4f}±{summary.get('std_acc', 0.0):.4f}  "
                f"{summary['mean_kappa']:.4f}±{summary.get('std_kappa', 0.0):.4f}  "
                + "".join([f"{acc:.4f}  " for acc in summary["per_class_acc"]])
                + "\n"
            )

        if len(all_summaries) > 1:
            accs = np.array([summary["mean_acc"] for summary in all_summaries])
            kappas = np.array([summary["mean_kappa"] for summary in all_summaries])
            mean_pc = np.nanmean([summary["per_class_acc"] for summary in all_summaries], axis=0)
            f.write("-" * 80 + "\n")
            f.write(
                f"{'AVG':<10}"
                f"{accs.mean():.4f}±{accs.std():.4f}  "
                f"{kappas.mean():.4f}±{kappas.std():.4f}  "
                + "".join([f"{acc:.4f}  " for acc in mean_pc])
                + "\n"
            )


def parse_subjects(subject_arg):
    if subject_arg == "all":
        return list(range(1, 11))
    return [int(s) for s in subject_arg.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="all", help="Held-out subject id, comma list, or 'all'")
    parser.add_argument("--exp_name", type=str, default="pythondata1_loso_npz")
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--npz_variant", type=str, default=None, help="NPZ variant, e.g. NoICA or ICA.")
    parser.add_argument("--cv", type=str, default="loso", choices=["loso"])
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_metric", type=str, default=None, choices=["kappa", "macro_f1"])
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["none", "cosine", "plateau"])
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--spatial_max_norm", type=float, default=None)
    parser.add_argument("--classifier_max_norm", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_size", type=float, default=None)
    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--skip_bandpass", action="store_true")
    parser.add_argument("--preload_gpu", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save best checkpoint per fold (default off; turn on for final runs).",
    )
    parser.add_argument("--torch_threads", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.npz_variant is not None:
        cfg.npz_variant = args.npz_variant
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.early_stop_metric is not None:
        cfg.early_stop_metric = args.early_stop_metric
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.lr is not None:
        cfg.lr = args.lr
    if args.lr_scheduler is not None:
        cfg.lr_scheduler = args.lr_scheduler
    if args.grad_clip_norm is not None:
        cfg.grad_clip_norm = args.grad_clip_norm
    if args.spatial_max_norm is not None:
        cfg.spatial_max_norm = args.spatial_max_norm
    if args.classifier_max_norm is not None:
        cfg.classifier_max_norm = args.classifier_max_norm
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.val_size is not None:
        cfg.val_size = args.val_size
    if args.apply_bandpass and args.skip_bandpass:
        raise ValueError("Use only one of --apply_bandpass or --skip_bandpass.")
    if args.apply_bandpass:
        cfg.skip_bandpass = False
    if args.skip_bandpass:
        cfg.skip_bandpass = True
    if args.preload_gpu:
        cfg.preload_gpu = True
    if args.save_model:
        cfg.save_model = True
    if args.no_save_model:
        cfg.save_model = False
    if not 0.0 < cfg.val_size < 1.0:
        raise ValueError(f"--val_size must be between 0 and 1, got {cfg.val_size}")

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(4, args.torch_threads)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.preload_gpu and device == "cuda":
        cfg.preload_device = device
        cfg.num_workers = 0
    else:
        cfg.preload_device = None

    print(f"Device: {device}")
    print(f"Data root: {cfg.data_root}")
    print(
        f"Runtime: batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, "
        f"n_samples={cfg.n_samples}, pin_memory={cfg.pin_memory}, "
        f"bandpass={cfg.bandpass_low}-{cfg.bandpass_high}Hz, "
        f"skip_bandpass={cfg.skip_bandpass}, preload_gpu={cfg.preload_gpu}, "
        f"val_size={cfg.val_size}, early_stop_metric={cfg.early_stop_metric}, "
        f"optimizer={cfg.optimizer}, lr={cfg.lr}, weight_decay={cfg.weight_decay}, "
        f"lr_scheduler={cfg.lr_scheduler}, grad_clip_norm={cfg.grad_clip_norm}, "
        f"plateau_factor={cfg.plateau_factor}, plateau_patience={cfg.plateau_patience}, "
        f"dropout={cfg.dropout}, "
        f"spatial_max_norm={cfg.spatial_max_norm}, classifier_max_norm={cfg.classifier_max_norm}, "
        f"save_model={cfg.save_model}"
    )

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)

    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = cfg.log_dir if os.path.isabs(cfg.log_dir) else os.path.join(project_root, cfg.log_dir)
    exp_dir = os.path.join(log_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    heldout_subjects = parse_subjects(args.subject)
    all_subjects = list(range(1, 11))
    unknown_subjects = sorted(set(heldout_subjects) - set(all_subjects))
    if unknown_subjects:
        raise ValueError(f"Unknown subject ids for this dataset: {unknown_subjects}")

    all_summaries = run_loso(
        heldout_subjects=heldout_subjects,
        all_subjects=all_subjects,
        cfg=cfg,
        model_name=args.model,
        exp_dir=exp_dir,
        device=device,
    )

    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(all_summaries), f, indent=2)

    append_summary(
        summary_path=os.path.join(project_root, "summary.txt"),
        exp_name=args.exp_name,
        model_name=args.model,
        cfg=cfg,
        all_summaries=all_summaries,
    )

    print(f"\nResults saved to {exp_dir}/")
    if cfg.save_model:
        print(f"Checkpoints saved to {os.path.join(exp_dir, 'checkpoints')}/")


if __name__ == "__main__":
    main()

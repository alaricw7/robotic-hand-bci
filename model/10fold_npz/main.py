"""
main.py - Subject-dependent 10-fold EEGNet experiment entry point.
"""

import argparse
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from config import Config
from data import discover_subject_ids, get_fold_loaders, load_subject
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


def _export_checkpoint_for_representation(source_path, subject_id, tag, export_dir):
    if not tag:
        return None
    if source_path is None or not os.path.exists(source_path):
        print(
            f"  [WARN] Cannot export representation checkpoint for S{subject_id}: "
            f"source checkpoint missing ({source_path})"
        )
        return None
    os.makedirs(export_dir, exist_ok=True)
    target_path = os.path.join(export_dir, f"S{subject_id}_eegnet_{tag}.pt")
    shutil.copy2(source_path, target_path)
    print(f"  Exported representation checkpoint: {target_path}")
    return target_path


def save_confusion_matrix_png(cm, out_path, title, class_labels=None):
    """Save a readable heatmap PNG next to a confusion-matrix NPY file."""
    cm = np.asarray(cm)
    if class_labels is None:
        class_labels = [str(i) for i in range(cm.shape[0])]

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160)
    matrices = [(cm, "Counts", "d"), (cm_norm, "Row-normalized", ".2f")]
    for ax, (matrix, subtitle, fmt) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap="Blues", aspect="equal")
        ax.set_title(subtitle)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        threshold = float(np.nanmax(matrix)) / 2.0 if np.size(matrix) else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                color = "white" if value > threshold else "black"
                ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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


def run_subject(subject_id, cfg, model_name, exp_dir, device):
    """Run subject-dependent 10-fold cross-validation for one subject."""
    print(f"\n===== Subject {subject_id} =====")
    X, y = load_subject(subject_id, cfg.data_root, cfg)
    print(f"  Data shape: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    fold_results = []
    all_preds, all_labels = [], []

    for fold in range(cfg.n_folds):
        print(f"  --- Fold {fold + 1}/{cfg.n_folds} ---")

        model = build_model(cfg, model_name=model_name)
        train_loader, val_loader, test_loader, split_info = get_fold_loaders(X, y, fold, cfg)

        counts = split_info["counts"]
        class_counts = split_info["class_counts"]
        print(
            f"  Split: train={counts['train']}, val={counts['val']}, test={counts['test']} "
            f"| test class counts={class_counts['test']}"
        )

        checkpoint_path = os.path.join(
            exp_dir,
            "checkpoints",
            f"subj{subject_id}_fold{fold + 1}.pt",
        )
        checkpoint_meta = {
            "cv": "subject_dependent",
            "subject": subject_id,
            "fold": fold + 1,
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
            f"subj{subject_id}_fold{fold + 1}.png",
        )
        curve_csv = os.path.join(
            exp_dir,
            "training_curves",
            f"subj{subject_id}_fold{fold + 1}.csv",
        )
        save_training_curves_png(
            result["history"],
            curve_png,
            title=f"S{subject_id} fold {fold + 1} training curves",
            best_epoch=result["best_epoch"],
        )
        save_training_history_csv(result["history"], curve_csv)

        exported_checkpoint = None
        if fold == getattr(cfg, "representation_checkpoint_fold", 0):
            exported_checkpoint = _export_checkpoint_for_representation(
                result["checkpoint_path"],
                subject_id,
                getattr(cfg, "representation_checkpoint_tag", None),
                getattr(cfg, "representation_checkpoint_dir", None),
            )

        fold_results.append(
            {
                "fold": fold + 1,
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
                "checkpoint": result["checkpoint_path"],
                "exported_checkpoint": exported_checkpoint,
                "history": result["history"],
                "training_curve_png": curve_png,
                "training_curve_csv": curve_csv,
            }
        )
        all_preds.append(test_preds)
        all_labels.append(test_labels)
        print(
            f"  Fold {fold + 1}: best_val_{result['early_stop_metric']}={result['best_score']:.4f}, "
            f"best_val_acc={result['best_acc']:.4f}, "
            f"test_acc={test_acc:.4f}, test_kappa={test_kappa:.4f}, "
            f"curves={curve_png}"
        )

    mean_acc = np.mean([r["test_acc"] for r in fold_results])
    mean_kappa = np.mean([r["test_kappa"] for r in fold_results])
    mean_per_class = np.nanmean([r["per_class_acc"] for r in fold_results], axis=0)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.n_classes)))
    cm_path = os.path.join(exp_dir, f"confusion_subj{subject_id}.npy")
    np.save(cm_path, cm)
    save_confusion_matrix_png(
        cm,
        os.path.join(exp_dir, f"confusion_subj{subject_id}.png"),
        title=f"S{subject_id} confusion matrix",
        class_labels=[str(i) for i in range(cfg.n_classes)],
    )

    summary = {
        "subject": subject_id,
        "mean_acc": float(mean_acc),
        "mean_kappa": float(mean_kappa),
        "std_acc": float(np.std([r["test_acc"] for r in fold_results])),
        "std_kappa": float(np.std([r["test_kappa"] for r in fold_results])),
        "per_class_acc": mean_per_class.tolist(),
        "folds": fold_results,
    }

    print(f"  >>> Subject {subject_id}: acc={mean_acc:.4f}, kappa={mean_kappa:.4f}")
    return summary


def append_summary(summary_path, exp_name, model_name, cfg, all_summaries):
    """Append one run summary to summary.txt instead of overwriting it."""
    summary_exists = os.path.exists(summary_path) and os.path.getsize(summary_path) > 0

    with open(summary_path, "a", encoding="utf-8") as f:
        if summary_exists:
            f.write("\n" + "=" * 80 + "\n\n")

        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"CV: subject-dependent stratified {cfg.n_folds}-fold\n")
        f.write("Validation: stratified random val_size from train_val\n")
        f.write(f"Metric: final test set, checkpoint selected by val {getattr(cfg, 'early_stop_metric', 'kappa')}\n")
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


def parse_subject_ids(subject_arg):
    subject_ids = []
    for item in subject_arg.split(","):
        item = item.strip()
        if item.lower().startswith("s") and item[1:].isdigit():
            item = item[1:]
        subject_ids.append(int(item))
    return subject_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="all", help="Subject id, comma list, or 'all'")
    parser.add_argument("--exp_name", type=str, default="pythondata1_npz_repr")
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--data_root", type=str, default=None)
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
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--apply_bandpass", action="store_true")
    parser.add_argument("--preload_gpu", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save best checkpoint per fold (default off; turn on for final runs).",
    )
    parser.add_argument(
        "--representation_checkpoint_tag",
        type=str,
        default=None,
        choices=["noica", "ica"],
        help=(
            "Also export the selected fold checkpoint to the experiment-local "
            "representation/checkpoints/S{subject}_eegnet_{tag}.pt directory by default. "
            "Use 'noica' for pythondata1 and 'ica' for ICA data."
        ),
    )
    parser.add_argument(
        "--representation_checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory for exported representation checkpoints. "
            "Defaults to <exp_dir>/representation/checkpoints."
        ),
    )
    parser.add_argument(
        "--representation_checkpoint_fold",
        type=int,
        default=0,
        help="0-indexed fold to export. Use 0 to match representation scripts' fold 0 test split.",
    )
    parser.add_argument("--torch_threads", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
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
    if args.n_folds is not None:
        cfg.n_folds = args.n_folds
    if args.apply_bandpass:
        cfg.skip_bandpass = False
    if args.preload_gpu:
        cfg.preload_gpu = True
    if args.save_model:
        cfg.save_model = True
    if args.no_save_model:
        cfg.save_model = False
    if args.representation_checkpoint_tag is not None:
        cfg.save_model = True
        cfg.representation_checkpoint_tag = args.representation_checkpoint_tag
        cfg.representation_checkpoint_dir = args.representation_checkpoint_dir
        cfg.representation_checkpoint_fold = args.representation_checkpoint_fold
    else:
        cfg.representation_checkpoint_tag = None
        cfg.representation_checkpoint_dir = args.representation_checkpoint_dir
        cfg.representation_checkpoint_fold = args.representation_checkpoint_fold
    if not 0.0 < cfg.val_size < 1.0:
        raise ValueError(f"--val_size must be between 0 and 1, got {cfg.val_size}")
    if cfg.n_folds < 2:
        raise ValueError(f"--n_folds must be >= 2, got {cfg.n_folds}")
    if not 0 <= cfg.representation_checkpoint_fold < cfg.n_folds:
        raise ValueError(
            f"--representation_checkpoint_fold must be in [0, {cfg.n_folds - 1}], "
            f"got {cfg.representation_checkpoint_fold}"
        )

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
        f"skip_bandpass={cfg.skip_bandpass}, preload_gpu={cfg.preload_gpu}, "
        f"val_size={cfg.val_size}, early_stop_metric={cfg.early_stop_metric}, "
        f"n_folds={cfg.n_folds}, "
        f"optimizer={cfg.optimizer}, lr={cfg.lr}, weight_decay={cfg.weight_decay}, "
        f"lr_scheduler={cfg.lr_scheduler}, grad_clip_norm={cfg.grad_clip_norm}, "
        f"spatial_max_norm={cfg.spatial_max_norm}, classifier_max_norm={cfg.classifier_max_norm}, "
        f"save_model={cfg.save_model}, "
        f"representation_checkpoint_tag={cfg.representation_checkpoint_tag}"
    )

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)

    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = cfg.log_dir if os.path.isabs(cfg.log_dir) else os.path.join(project_root, cfg.log_dir)
    exp_dir = os.path.join(log_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    if cfg.representation_checkpoint_tag is not None and cfg.representation_checkpoint_dir is None:
        cfg.representation_checkpoint_dir = os.path.join(
            exp_dir,
            "representation",
            "checkpoints",
        )

    if args.subject == "all":
        subjects = discover_subject_ids(cfg.data_root)
        if not subjects:
            raise FileNotFoundError(f"No subject files found under {cfg.data_root}")
    else:
        subjects = parse_subject_ids(args.subject)
    print(f"Subjects: {subjects}")
    if cfg.representation_checkpoint_tag is not None:
        print(f"Representation checkpoints saved to {cfg.representation_checkpoint_dir}/")

    all_summaries = []
    for sid in subjects:
        summary = run_subject(sid, cfg, args.model, exp_dir, device)
        all_summaries.append(summary)

    subject_cms = []
    for sid in subjects:
        cm_path = os.path.join(exp_dir, f"confusion_subj{sid}.npy")
        if os.path.exists(cm_path):
            subject_cms.append(np.load(cm_path))
    if subject_cms:
        cm_all = np.sum(np.stack(subject_cms, axis=0), axis=0)
        np.save(os.path.join(exp_dir, "confusion_all.npy"), cm_all)
        save_confusion_matrix_png(
            cm_all,
            os.path.join(exp_dir, "confusion_all.png"),
            title="All subjects confusion matrix",
            class_labels=[str(i) for i in range(cfg.n_classes)],
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
    if cfg.representation_checkpoint_tag is not None:
        print(f"Representation checkpoints saved to {cfg.representation_checkpoint_dir}/")


if __name__ == "__main__":
    main()

"""
train.py - 训练循环 + 评估工具。
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def _cpu_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_checkpoint(
    path,
    model_state_dict,
    optimizer,
    epoch,
    best_acc,
    best_kappa,
    best_macro_f1,
    best_score,
    early_stop_metric,
    history,
    meta,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_acc,
            "best_val_kappa": best_kappa,
            "best_val_macro_f1": best_macro_f1,
            "best_val_score": best_score,
            "early_stop_metric": early_stop_metric,
            "history": history,
            "meta": meta or {},
        },
        path,
    )


def _macro_f1(preds, labels, n_classes):
    return f1_score(
        labels,
        preds,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )


def _early_stop_score(metric_name, val_acc, val_kappa, val_macro_f1):
    if metric_name == "kappa":
        return val_kappa
    if metric_name == "macro_f1":
        return val_macro_f1
    raise ValueError(f"Unknown early_stop_metric={metric_name!r}; use 'kappa' or 'macro_f1'.")


def _apply_max_norm_constraints(model, cfg):
    if not hasattr(model, "apply_max_norm_constraints"):
        return
    model.apply_max_norm_constraints(
        spatial_max_norm=getattr(cfg, "spatial_max_norm", 1.0),
        classifier_max_norm=getattr(cfg, "classifier_max_norm", 0.25),
    )


def train_one_fold(
    model,
    train_loader,
    val_loader,
    cfg,
    device="cuda",
    checkpoint_path=None,
    checkpoint_meta=None,
    test_loader=None,
):
    """
    训练一折。checkpoint/early stopping 使用 cfg.early_stop_metric，而不是 accuracy。
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    opt_name = getattr(cfg, "optimizer", "adam").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name!r}")

    scheduler_name = getattr(cfg, "lr_scheduler", "none").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_epochs,
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=getattr(cfg, "plateau_factor", 0.5),
            patience=getattr(cfg, "plateau_patience", 10),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name!r}")

    early_stop_metric = getattr(cfg, "early_stop_metric", "kappa")
    best_score = -np.inf
    best_acc = 0.0
    best_kappa = 0.0
    best_macro_f1 = 0.0
    best_epoch = -1
    best_preds = None
    best_labels = None
    best_state_dict = None
    patience = 0

    history = {
        "train_loss": [],
        "val_acc": [],
        "val_kappa": [],
        "val_macro_f1": [],
        "val_score": [],
        "test_acc": [],
        "test_kappa": [],
    }

    for epoch in range(cfg.n_epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            clip_value = getattr(cfg, "grad_clip_norm", 0.0)
            if clip_value and clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            _apply_max_norm_constraints(model, cfg)
            train_losses.append(loss.item())

        if scheduler is not None and scheduler_name != "plateau":
            scheduler.step()

        val_acc, val_kappa, preds, labels = evaluate(model, val_loader, device)
        val_macro_f1 = _macro_f1(preds, labels, cfg.n_classes)
        val_score = _early_stop_score(early_stop_metric, val_acc, val_kappa, val_macro_f1)
        test_acc, test_kappa = None, None
        if test_loader is not None:
            test_acc, test_kappa, _, _ = evaluate(model, test_loader, device)
        if scheduler is not None and scheduler_name == "plateau":
            scheduler.step(val_score)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_acc"].append(float(val_acc))
        history["val_kappa"].append(float(val_kappa))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["val_score"].append(float(val_score))
        history["test_acc"].append(float(test_acc) if test_acc is not None else None)
        history["test_kappa"].append(float(test_kappa) if test_kappa is not None else None)

        if val_score > best_score:
            best_score = val_score
            best_acc = val_acc
            best_kappa = val_kappa
            best_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            best_preds = preds
            best_labels = labels
            best_state_dict = _cpu_state_dict(model)
            patience = 0
            if checkpoint_path is not None and getattr(cfg, "save_model", True):
                _save_checkpoint(
                    checkpoint_path,
                    best_state_dict,
                    optimizer,
                    best_epoch,
                    best_acc,
                    best_kappa,
                    best_macro_f1,
                    best_score,
                    early_stop_metric,
                    history,
                    checkpoint_meta,
                )
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(
                    f"  Early stop at epoch {epoch + 1}, "
                    f"best val {early_stop_metric}={best_score:.4f}"
                )
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:3d} | loss={np.mean(train_losses):.4f} "
                f"| val_acc={val_acc:.4f} | val_kappa={val_kappa:.4f} "
                f"| val_macro_f1={val_macro_f1:.4f} | val_{early_stop_metric}={val_score:.4f}"
                + (f" | test_acc={test_acc:.4f}" if test_acc is not None else "")
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "best_acc": float(best_acc),
        "best_kappa": float(best_kappa),
        "best_macro_f1": float(best_macro_f1),
        "best_score": float(best_score),
        "early_stop_metric": early_stop_metric,
        "best_epoch": best_epoch,
        "preds": best_preds,
        "labels": best_labels,
        "history": history,
        "checkpoint_path": checkpoint_path if checkpoint_path is not None and getattr(cfg, "save_model", True) else None,
    }


def evaluate(model, loader, device):
    """在一个 loader 上评估，返回 acc, kappa, preds, labels。"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    return acc, kappa, preds, labels


def per_class_accuracy(preds, labels, n_classes):
    """
    每类的准确率。某类在 labels 中完全不存在时返回 np.nan,
    避免把"该类不存在"和"该类全错"混为一谈。
    """
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    row_sums = cm.sum(axis=1)
    per_class = np.full(n_classes, np.nan, dtype=np.float64)
    nonzero = row_sums > 0
    per_class[nonzero] = cm.diagonal()[nonzero] / row_sums[nonzero]
    return per_class

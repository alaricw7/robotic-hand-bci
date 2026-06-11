import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn

try:
    from .metrics import compute_metrics
except ImportError:
    from metrics import compute_metrics


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_optimizer(model, cfg):
    name = cfg["optimizer"].lower()
    kwargs = dict(lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0),
                  betas=cfg.get("betas", (0.9, 0.999)))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"unknown optimizer: {name}")


def sample_branch_drop_mask(model, batch_size, device, dtype=torch.float32):
    """Sample train-time modality-dropout keep mask, or None when disabled."""
    if (not getattr(model, "modality_dropout_enabled", False)) or (not model.training):
        return None

    p = float(getattr(model, "modality_dropout_p", 0.2))
    if p <= 0.0:
        return None
    if p > 1.0:
        raise ValueError(f"modality_dropout_p must be <= 1.0, got {p}")

    k = len(getattr(model, "active_branches", ()))
    if k <= 0:
        return None

    keep_prob = 1.0 - p
    mask = torch.empty(batch_size, k, device=device, dtype=dtype).bernoulli_(keep_prob)
    dead_rows = mask.sum(dim=1) == 0
    if dead_rows.any():
        rescue_rows = dead_rows.nonzero(as_tuple=False).flatten()
        rescue_cols = torch.randint(0, k, (rescue_rows.numel(),), device=device)
        mask[rescue_rows, rescue_cols] = 1.0
    return mask


def get_branch_decorr_loss(model, aux=None):
    if aux is not None and aux.get("branch_decorr_loss") is not None:
        return aux["branch_decorr_loss"]
    last_aux = getattr(model, "last_aux", None)
    if isinstance(last_aux, dict):
        return last_aux.get("branch_decorr_loss")
    return None


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        branch_drop_mask = sample_branch_drop_mask(model, X.size(0), X.device, X.dtype)
        if branch_drop_mask is not None:
            logits = model(X, branch_drop_mask=branch_drop_mask)
        else:
            logits = model(X)
        loss = criterion(logits, y)
        branch_decorr_loss = get_branch_decorr_loss(model)
        if branch_decorr_loss is not None:
            loss = loss + branch_decorr_loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fit_select_test(
    model,
    train_loader,
    val_loader,
    test_loader,
    n_classes: int,
    cfg: dict,
    device,
    verbose: bool = False,
    tag: str = "",
    save_ckpt_path: str = None,
):
    """Train ``cfg['epochs']`` epochs; at the epoch with best val kappa,
    snapshot weights and report metrics on ``test_loader``.

    Returns dict with test_acc, test_kappa, test_per_class (np.ndarray
    shape (n_classes,)), best_val_kappa, best_epoch, elapsed.
    """
    model = model.to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_val_kappa = -float("inf")
    best_state = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = predict(model, val_loader, device)
        _, val_kappa, _ = compute_metrics(y_true, y_pred, n_classes)
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == cfg["epochs"]):
            print(f"[{tag}] ep{epoch:03d} tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} "
                  f"val_kappa={val_kappa:.3f} best={best_val_kappa:.3f}@{best_epoch}")

    model.load_state_dict(best_state)
    if save_ckpt_path is not None:
        os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "best_val_kappa": float(best_val_kappa),
                "best_epoch": int(best_epoch),
                "tag": tag,
            },
            save_ckpt_path,
        )
    y_true, y_pred = predict(model, test_loader, device)
    test_acc, test_kappa, test_per_class = compute_metrics(y_true, y_pred, n_classes)
    return {
        "test_acc": test_acc,
        "test_kappa": test_kappa,
        "test_per_class": test_per_class,
        "best_val_kappa": float(best_val_kappa),
        "best_epoch": int(best_epoch),
        "elapsed": time.time() - t0,
    }

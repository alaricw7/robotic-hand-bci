"""Train-time augmentation for TriDomain. Test/val: identity.

All transforms operate on torch tensors of shape (C, T) (per-sample) or
(B, C, T) (mixup, applied at batch level). Each transform is a class with
torch.no_grad forward.

Augmentation config keys (defaults disable everything; baseline = no augmentation):

  crop_enabled        bool
  crop_len            int       # samples; must be divisible by freq_windows (=4 by default)
  crop_stride         int       # samples; stride between training crops
  freqmask_enabled    bool
  freqmask_p          float     # per-sample apply prob
  freqmask_bw_hz      float     # bandwidth of the masked band
  fs                  int       # sample rate (for freqmask)
  chdrop_enabled      bool
  chdrop_p            float     # per-channel zero-out probability
  noise_enabled       bool
  noise_std           float     # additive gaussian std (post-standardize)
  mixup_enabled       bool
  mixup_alpha         float
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- per-sample CPU augmentations ---------- #

class FreqBandMask:
    """Mask a random frequency band by FFT-zero-and-inverse. Per-sample."""

    def __init__(self, p: float, bw_hz: float, fs: int):
        self.p = float(p)
        self.bw_hz = float(bw_hz)
        self.fs = int(fs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T) float
        if np.random.rand() >= self.p:
            return x
        C, T = x.shape
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fs)  # (T//2+1,)
        f_lo = float(np.random.uniform(0.5, max(0.5, self.fs / 2.0 - self.bw_hz)))
        f_hi = f_lo + self.bw_hz
        mask = ~((freqs >= f_lo) & (freqs <= f_hi))
        X = torch.fft.rfft(x, dim=-1)
        X = X * torch.from_numpy(mask.astype(np.float32)).to(X.real.dtype).to(X.device)
        return torch.fft.irfft(X, n=T, dim=-1)


class ChannelDropout:
    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[0]
        keep = (torch.rand(C, device=x.device) >= self.p).float().unsqueeze(-1)
        return x * keep


class GaussianNoise:
    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std


# ---------- sliding-crop dataset ---------- #

class SlidingCropTrainDataset(Dataset):
    """Expands each trial into overlapping crops of length crop_len with stride
    crop_stride. Labels are inherited from the parent trial.
    Sample augmentations (freqmask/chdrop/noise) are applied on top per __getitem__.
    """

    def __init__(self, X_np: np.ndarray, y_np: np.ndarray,
                 crop_len: int, crop_stride: int, sample_transforms=None):
        assert X_np.ndim == 3, f"X shape {X_np.shape}"
        n, C, T = X_np.shape
        assert crop_len <= T, f"crop_len={crop_len} > T={T}"
        starts = list(range(0, T - crop_len + 1, crop_stride))
        if starts[-1] != T - crop_len:
            starts.append(T - crop_len)
        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(y_np).long()
        self.crop_len = int(crop_len)
        self.starts = starts
        self.sample_transforms = sample_transforms or []
        self.index = [(i, s) for i in range(n) for s in starts]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        x = self.X[i, :, s:s + self.crop_len].clone()
        for transform in self.sample_transforms:
            x = transform(x)
        return x, self.y[i]


class PlainTrainDataset(Dataset):
    """Full-trial training dataset with per-sample augmentations only."""

    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, sample_transforms=None):
        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(y_np).long()
        self.sample_transforms = sample_transforms or []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        for transform in self.sample_transforms:
            x = transform(x)
        return x, self.y[idx]


class EvalDataset(Dataset):
    """Plain trial-level eval dataset. No augmentation."""

    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
        self.X = torch.from_numpy(X_np).float()
        self.y = torch.from_numpy(y_np).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- batch-level mixup ---------- #

class MixUp:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def __call__(self, X: torch.Tensor, y: torch.Tensor, n_classes: int):
        # X: (B,C,T)  y: (B,)
        if self.alpha <= 0:
            return X, _one_hot(y, n_classes)
        lam = float(np.random.beta(self.alpha, self.alpha))
        perm = torch.randperm(X.size(0), device=X.device)
        X_mix = lam * X + (1.0 - lam) * X[perm]
        y_oh = _one_hot(y, n_classes)
        y_mix = lam * y_oh + (1.0 - lam) * y_oh[perm]
        return X_mix, y_mix


def _one_hot(y: torch.Tensor, n_classes: int):
    oh = torch.zeros(y.size(0), n_classes, device=y.device, dtype=torch.float32)
    return oh.scatter_(1, y.view(-1, 1), 1.0)


# ---------- factory ---------- #

DEFAULT_AUGMENTATION = {
    "crop_enabled": False,
    "crop_len": 500,
    "crop_stride": 125,
    "freqmask_enabled": False,
    "freqmask_p": 0.5,
    "freqmask_bw_hz": 6.0,
    "fs": 250,
    "chdrop_enabled": False,
    "chdrop_p": 0.2,
    "noise_enabled": False,
    "noise_std": 0.05,
    "mixup_enabled": False,
    "mixup_alpha": 0.2,
    "test_protocol": "full_trial",  # or "crop_voting"
}


def build_sample_transforms(cfg: dict):
    ops = []
    if cfg.get("freqmask_enabled", False):
        ops.append(FreqBandMask(p=cfg["freqmask_p"], bw_hz=cfg["freqmask_bw_hz"], fs=cfg["fs"]))
    if cfg.get("chdrop_enabled", False):
        ops.append(ChannelDropout(p=cfg["chdrop_p"]))
    if cfg.get("noise_enabled", False):
        ops.append(GaussianNoise(std=cfg["noise_std"]))
    return ops


def make_train_loader(X_tr_std, y_tr, batch_size, cfg: dict, num_workers=2):
    transforms = build_sample_transforms(cfg)
    if cfg.get("crop_enabled", False):
        ds = SlidingCropTrainDataset(
            X_tr_std, y_tr,
            crop_len=cfg["crop_len"], crop_stride=cfg["crop_stride"],
            sample_transforms=transforms,
        )
    else:
        ds = PlainTrainDataset(X_tr_std, y_tr, sample_transforms=transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def make_eval_loader(X_std, y, batch_size, num_workers=2):
    return DataLoader(EvalDataset(X_std, y), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def crop_vote_logits(model, X_te_std, cfg, device, batch_size=64):
    """Average softmax over overlapping crops, then argmax.

    Returns (y_pred,) as numpy int64.
    """
    n, C, T = X_te_std.shape
    crop_len = int(cfg["crop_len"])
    stride = int(cfg["crop_stride"])
    starts = list(range(0, T - crop_len + 1, stride))
    if starts[-1] != T - crop_len:
        starts.append(T - crop_len)
    prob_sum = None
    model.eval()
    X = torch.from_numpy(X_te_std).float()
    with torch.no_grad():
        for s in starts:
            xb = X[:, :, s:s + crop_len]
            probs_chunks = []
            for i in range(0, n, batch_size):
                logits = model(xb[i:i + batch_size].to(device, non_blocking=True))
                probs_chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
            p = np.concatenate(probs_chunks, axis=0)
            prob_sum = p if prob_sum is None else (prob_sum + p)
    prob_sum = prob_sum / len(starts)
    return prob_sum.argmax(axis=1).astype(np.int64), prob_sum

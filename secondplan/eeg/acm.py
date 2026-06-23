"""Augmented covariance matrix (ACM) utilities for DAFG-Net.

The delay embedding follows the channel-stacked Takens construction used by
Carrara & Papadopoulo (2023): each channel contributes `p` delayed copies over a
common cropped time axis.
"""
from __future__ import annotations

import numpy as np
import torch


def delay_embed(sig, p: int, tau: int):
    """Delay-embed signals from (B, ch, T) to (B, ch*p, T-(p-1)*tau).

    The stacked order is channel-major:
        out[:, c*p + j, :] = sig[:, c, (p-1-j)*tau : T-j*tau]

    For p=1 this returns `sig` unchanged, preserving the baseline path exactly.
    """
    if p < 1:
        raise ValueError(f"p/embed_dim must be >= 1, got {p}")
    if tau < 1:
        raise ValueError(f"tau/embed_delay must be >= 1, got {tau}")
    if sig.ndim != 3:
        raise ValueError(f"delay_embed expects (B, ch, T), got shape {tuple(sig.shape)}")
    if p == 1:
        return sig

    B, ch, T = sig.shape
    Tprime = T - (p - 1) * tau
    if Tprime <= 0:
        raise ValueError(f"delay embedding has non-positive length: T={T}, p={p}, tau={tau}")

    pieces = []
    for c in range(ch):
        for j in range(p):
            pieces.append(sig[:, c:c + 1, (p - 1 - j) * tau:T - j * tau])
    if torch.is_tensor(sig):
        return torch.cat(pieces, dim=1)
    if isinstance(sig, np.ndarray):
        return np.concatenate(pieces, axis=1)
    raise TypeError(f"delay_embed supports torch.Tensor or np.ndarray, got {type(sig)!r}")


def acm_is_full_rank_feasible(n_channels: int, T: int, p: int, tau: int) -> tuple[bool, int, int]:
    """Return (ok, embedded_channels, embedded_time) for the ACM rank guard."""
    Tprime = T - (p - 1) * tau
    d = n_channels * p
    return (Tprime > 0 and d < Tprime), d, Tprime

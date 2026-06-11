"""Task 6 Step 1 — covariance + Riemannian tangent-space features.

PIPELINE (train-only fits, applied identically to val/test):
  1) bandpass to delta/theta (default 0.5-8 Hz)
  2) crop to [0, 2s] (default first 500 samples at 250 Hz)
  3) keep top-K channels by importance ranking (from channel_importance.csv)
  4) per-trial spatial covariance with shrinkage (LedoitWolf or +epsI)
  5) Riemannian geometric mean on TRAIN covariances → reference SPD matrix
  6) log-map all cov matrices to tangent space at that reference → upper-tri
     vectors of length D = K*(K+1)/2

PRINTS one assertion line per fold proving train/val/test use the SAME
train-fit reference (Frobenius norm hash), so no leakage.

Returns three arrays (cov_feat_tr, cov_feat_va, cov_feat_te) ready to
concat with the rest of the model.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


# ----------------------------- Band-pass FIR ----------------------------- #
def _sinc_bandpass_kernel(num_taps: int, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    flo, fhi = low_hz / fs, high_hz / fs

    def lp(fc):
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc = np.where(n == 0, 1.0, np.sin(2 * np.pi * fc * n) / (np.pi * n))
            return 2 * fc * np.where(n == 0, 1.0, sinc / (2 * fc))

    h = lp(fhi) - lp(flo)
    w = np.hamming(num_taps)
    h = h * w
    norm = np.sqrt((h ** 2).sum()) + 1e-8
    return (h / norm).astype(np.float32)


def _filtfilt_same(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Single-pass 'same' FIR. X: (N, C, T). Returns same shape.

    Vectorised via scipy.signal.fftconvolve along the last axis so the cost
    scales like one FFT of length (T + taps - 1) per (trial × channel) batch,
    instead of N*C nested Python convolutions.
    """
    try:
        from scipy.signal import fftconvolve
        # broadcast h over (1,1,taps); mode='same' gives length T output
        return fftconvolve(X, h.reshape(1, 1, -1), mode="same", axes=-1)
    except (ImportError, TypeError):
        # Fallback: per-row fftconvolve (older SciPy without `axes`)
        try:
            from scipy.signal import fftconvolve as _fc
            N, C, T = X.shape
            out = np.empty_like(X)
            for n in range(N):
                for c in range(C):
                    y = _fc(X[n, c], h, mode="same")
                    out[n, c] = y
            return out
        except ImportError:
            N, C, T = X.shape
            out = np.empty_like(X)
            pad = (len(h) - 1) // 2
            for n in range(N):
                for c in range(C):
                    y = np.convolve(X[n, c], h, mode="full")
                    out[n, c] = y[pad:pad + T]
            return out


# ----------------------------- SPD ops ----------------------------- #
def _shrink_cov(X_trials: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """X_trials: (N, K, T). Returns covs (N, K, K) using Ledoit-Wolf-style
    shrinkage if sklearn available, else sample cov + epsI."""
    N, K, T = X_trials.shape
    out = np.empty((N, K, K), dtype=np.float64)
    try:
        from sklearn.covariance import LedoitWolf
        for i in range(N):
            lw = LedoitWolf(store_precision=False, assume_centered=False)
            lw.fit(X_trials[i].T)  # samples = T, features = K
            out[i] = lw.covariance_
    except Exception:
        for i in range(N):
            x = X_trials[i] - X_trials[i].mean(axis=1, keepdims=True)
            S = (x @ x.T) / max(T - 1, 1)
            out[i] = S + eps * np.eye(K)
    # Force SPD
    for i in range(N):
        out[i] = (out[i] + out[i].T) / 2.0 + eps * np.eye(K)
    return out


def _spd_logm(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)) @ V.T


def _spd_expm(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    return (V * np.exp(w)) @ V.T


def _sqrt_inv_sqrt(S: np.ndarray):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    sw = np.sqrt(w)
    sqrt = (V * sw) @ V.T
    inv_sqrt = (V * (1.0 / sw)) @ V.T
    return sqrt, inv_sqrt


def _riemann_mean(Cs: np.ndarray, n_iter: int = 50, tol: float = 1e-7) -> np.ndarray:
    """Karcher / Frechet mean on SPD via Log-Euclidean iteration."""
    try:
        from pyriemann.utils.mean import mean_riemann
        return mean_riemann(Cs)
    except Exception:
        pass
    # Fallback
    M = Cs.mean(axis=0)
    M = (M + M.T) / 2.0
    for _ in range(n_iter):
        sqrt, inv_sqrt = _sqrt_inv_sqrt(M)
        logs = np.stack([_spd_logm(inv_sqrt @ Ck @ inv_sqrt) for Ck in Cs], axis=0)
        T_mean = logs.mean(axis=0)
        if np.linalg.norm(T_mean) < tol:
            break
        M = sqrt @ _spd_expm(T_mean) @ sqrt
        M = (M + M.T) / 2.0
    return M


def _tangent_vec(C: np.ndarray, M: np.ndarray, sqrt_inv_M: np.ndarray) -> np.ndarray:
    L = _spd_logm(sqrt_inv_M @ C @ sqrt_inv_M)
    # upper triangular (k=0) including diagonal
    iu = np.triu_indices(L.shape[0], k=0)
    # Standard tangent embedding: include sqrt(2) on off-diagonal terms to
    # preserve Frobenius inner product on the tangent space.
    out = L[iu].copy()
    diag_mask = (iu[0] == iu[1])
    out[~diag_mask] *= np.sqrt(2.0)
    return out


def _matrix_hash(M: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(M).tobytes()).hexdigest()[:10]


# ----------------------------- Public API ----------------------------- #
def _cache_key(subject, fold_idx, band, crop_sec, topk_channels, taps):
    sig = f"{subject}|f{fold_idx}|b{band[0]}-{band[1]}|c{crop_sec[0]}-{crop_sec[1]}|t{taps}|k{len(topk_channels)}|" \
          + ",".join(str(c) for c in topk_channels)
    return hashlib.md5(sig.encode()).hexdigest()[:16]


def compute_cov_features(
    X_tr_std: np.ndarray,
    X_va_std: np.ndarray,
    X_te_std: np.ndarray,
    *,
    fs: int = 250,
    band: tuple = (0.5, 8.0),
    crop_sec: tuple = (0.0, 2.0),
    topk_channels: list,        # list of channel indices into original 59
    taps: int = 251,
    verbose: bool = True,
    cache_dir: str = None,      # if set, key by (subject,fold,...) and reuse across runs
    cache_id: str = None,       # caller-provided identity, e.g. f"{subject}_fold{fold_idx}"
) -> tuple:
    """Apply the band-pass → crop → top-K → cov → Riemannian tangent pipeline.

    train-only fits: only X_tr_std is used to compute the reference
    Riemannian mean M. The same M is then applied to val and test.
    """
    # ---- disk cache (shared between runs that use the same K, band, etc.) ----
    cache_path = None
    if cache_dir is not None and cache_id is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = _cache_key(cache_id, "auto", band, crop_sec, topk_channels, taps)
        cache_path = Path(cache_dir) / f"{cache_id}_{key}.npz"
        if cache_path.exists():
            try:
                npz = np.load(cache_path, allow_pickle=False)
                if verbose:
                    print(f"    [cov] CACHE HIT  {cache_path.name}  D={npz['cov_tr'].shape[1]}")
                return (npz["cov_tr"].astype(np.float32),
                        npz["cov_va"].astype(np.float32),
                        npz["cov_te"].astype(np.float32))
            except Exception as e:
                print(f"    [cov] cache load failed ({e}), recomputing")

    h = _sinc_bandpass_kernel(taps, band[0], band[1], fs)
    start = int(round(crop_sec[0] * fs))
    end = int(round(crop_sec[1] * fs))
    K = len(topk_channels)
    idx = np.asarray(topk_channels, dtype=np.int64)

    def crop_filter_select(X):
        Xf = _filtfilt_same(X.astype(np.float32, copy=False), h)
        Xc = Xf[:, idx, start:end]
        return Xc

    Xtr = crop_filter_select(X_tr_std)
    Xva = crop_filter_select(X_va_std)
    Xte = crop_filter_select(X_te_std)

    C_tr = _shrink_cov(Xtr)
    C_va = _shrink_cov(Xva)
    C_te = _shrink_cov(Xte)

    # train-only Riemannian reference
    M = _riemann_mean(C_tr)
    _, sqrt_inv_M = _sqrt_inv_sqrt(M)
    h_train_ref = _matrix_hash(M)

    def tan(Cs):
        return np.stack([_tangent_vec(C, M, sqrt_inv_M) for C in Cs], axis=0)

    feat_tr = tan(C_tr).astype(np.float32)
    feat_va = tan(C_va).astype(np.float32)
    feat_te = tan(C_te).astype(np.float32)

    if verbose:
        print(f"    [cov] band={band} crop={crop_sec}s K={K} D={feat_tr.shape[1]}  "
              f"train-fit M hash={h_train_ref}  "
              f"(val/test mapped through THIS reference — no leakage)")
    if cache_path is not None:
        try:
            np.savez_compressed(cache_path, cov_tr=feat_tr, cov_va=feat_va,
                                cov_te=feat_te, M_hash=h_train_ref)
            if verbose:
                print(f"    [cov] cached -> {cache_path.name}")
        except Exception as e:
            print(f"    [cov] cache save failed: {e}")
    return feat_tr, feat_va, feat_te


def load_topk_channels(csv_path: str, k: int, n_ch: int = 59):
    """Read diagnostics/channel_importance.csv, return top-k channel indices
    sorted by space_imp_mean descending. Falls back to [0..k-1] if file
    missing (and prints a warning)."""
    p = Path(csv_path)
    if not p.exists():
        print(f"    [cov] WARN: {csv_path} not found, using first {k} channels.")
        return list(range(min(k, n_ch)))
    rows = []
    import csv as _csv
    with p.open() as f:
        for r in _csv.DictReader(f):
            rows.append((int(r["ch_idx"]), float(r["space_imp_mean"])))
    rows.sort(key=lambda t: -t[1])
    return [int(rows[i][0]) for i in range(min(k, len(rows)))]

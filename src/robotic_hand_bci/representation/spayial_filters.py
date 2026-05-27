#!/usr/bin/env python3
"""
EEGNet spatial-filter visualization with class-conditional weighting.

Per subject (S1..S10):
  1. Load checkpoint, extract `spatial_conv.weight` -> (16, 59)
  2. Plot 16 raw spatial filters as topomaps (4x4)            -> fig_per_filter.png
  3. Forward hook on `spatial_conv` (pre-BN, pre-ELU) to get
     activations of shape (N, 16, 1, 1126). Compute per-trial
     per-filter energy over MI window [0.5, 3.5]s = sample idx [250, 1000].
  4. score[k, f] = mean(energy[y==k, f]) - mean(energy[y!=k, f])
  5. class_pattern[k] = sum_f score[k,f] * spatial_filter[f]   -> class_patterns.png
                                                              -> class_patterns.npy

Grand average:
  6. Normalize each (subject, class) pattern by max|.|, then mean across subjects.
     Plot 6 topomaps                                          -> grand_class_patterns.png
                                                              -> grand_class_patterns.npy
  7. Region-specificity sanity check (task-specific, NOT left/right hand):
       0 Manipulation -> central motor (C3..C4, FC3, FC4)
       1 ColorClass   -> occipital (O1, Oz, O2, PO3, POz, PO4)
       2 ColorTrack   -> occipital + parietal
       3 Face         -> bilateral occipitotemporal; also check P8+PO8 > P7+PO7 (N170)
       4 RPS          -> motor + occipital
       5 Gesture      -> central motor
                                                              -> region_specificity.csv

Notes:
- spatial_conv hook is taken BEFORE bn2 (more stable scale-wise).
- All topomaps in the same figure share a symmetric vlim = (-V, V), V = max|.|.
- Model is reconstructed locally from state_dict (independent of project's model.py).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from robotic_hand_bci.project import DEFAULT_MONTAGE_PATH, PROJECT_ROOT

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "pythondata1"
CKPT_DIR = (
    PROJECT_ROOT
    / "model"
    / "10fold_npz"
    / "experiments"
    / "pythondata1_npz_repr"
    / "representation"
    / "checkpoints"
)
MONTAGE_PATH = DEFAULT_MONTAGE_PATH
OUT_ROOT = PROJECT_ROOT / "analysis" / "repr" / "spatial_filters2"

SFREQ = 250.0
TMIN, TMAX = -0.5, 4.0
N_SAMPLES = 1126
N_CHANNELS = 59
F1, D = 16, 2

# MI window -> sample index in spatial_conv output (T = N_SAMPLES, pre-pool)
T_MI_START = int(round((0.5 - TMIN) * SFREQ))   # 250
T_MI_END   = int(round((3.5 - TMIN) * SFREQ))   # 1000

CLASS_NAMES = {
    0: "Manipulation",   # 操控控制
    1: "ColorClass",     # 颜色分类
    2: "ColorTrack",     # 颜色追踪
    3: "Face",           # 人脸识别
    4: "RPS",            # 猜拳动作
    5: "Gesture",        # 手势识别
}

# Task-specific ROIs for sanity check (replaces hand-MI contralateral test)
REGION_OF_INTEREST = {
    0: ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4"],
    1: ["O1", "Oz", "O2", "PO3", "POz", "PO4"],
    2: ["O1", "Oz", "O2", "PO3", "POz", "PO4", "P3", "Pz", "P4"],
    3: ["P8", "PO8", "P7", "PO7"],
    4: ["C3", "C1", "Cz", "C2", "C4", "O1", "Oz", "O2"],
    5: ["C3", "C1", "Cz", "C2", "C4", "FC3", "FC4"],
}
EOG_CHS = {"Fp1", "Fpz", "Fp2", "AF3", "AF4", "AF7", "AF8"}


# -----------------------------------------------------------------------------
# Minimal front of EEGNet: temporal_conv -> bn1 -> spatial_conv (hook target)
# -----------------------------------------------------------------------------
class MiniEEGNetFront(nn.Module):
    def __init__(self, kernel_length, n_channels=N_CHANNELS, F1=F1, D=D):
        super().__init__()
        # Pad to preserve T for odd kernels (kl=125 -> pad=62 -> T_out = T_in)
        pad_t = kernel_length // 2
        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length),
                                       padding=(0, pad_t), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                      groups=F1, bias=False)

    def forward(self, x):                       # (B, 1, C, T)
        x = self.temporal_conv(x)
        # For even kernel_length, conv may add one sample; trim to be safe.
        x = x[..., :N_SAMPLES]
        x = self.bn1(x)
        x = self.spatial_conv(x)                # (B, F1*D, 1, T)
        return x


def load_front_from_ckpt(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj

    required = ["temporal_conv.weight", "bn1.weight", "bn1.bias",
                "bn1.running_mean", "bn1.running_var", "spatial_conv.weight"]
    missing = [k for k in required if k not in sd]
    if missing:
        raise KeyError(f"checkpoint {ckpt_path} missing keys: {missing}\n"
                       f"available: {list(sd.keys())[:10]}...")

    kl   = sd["temporal_conv.weight"].shape[-1]
    f1   = sd["temporal_conv.weight"].shape[0]
    fd   = sd["spatial_conv.weight"].shape[0]
    n_ch = sd["spatial_conv.weight"].shape[-2]
    d    = fd // f1
    if (n_ch, f1, d) != (N_CHANNELS, F1, D):
        print(f"  WARNING: unexpected dims n_ch={n_ch} F1={f1} D={d} "
              f"(expected {N_CHANNELS}/{F1}/{D})")

    front = MiniEEGNetFront(kernel_length=kl, n_channels=n_ch, F1=f1, D=d)
    front_sd = {
        "temporal_conv.weight": sd["temporal_conv.weight"],
        "bn1.weight":           sd["bn1.weight"],
        "bn1.bias":             sd["bn1.bias"],
        "bn1.running_mean":     sd["bn1.running_mean"],
        "bn1.running_var":      sd["bn1.running_var"],
        "spatial_conv.weight":  sd["spatial_conv.weight"],
    }
    if "bn1.num_batches_tracked" in sd:
        front_sd["bn1.num_batches_tracked"] = sd["bn1.num_batches_tracked"]
    front.load_state_dict(front_sd, strict=False)
    front.eval()

    W = sd["spatial_conv.weight"].squeeze().cpu().numpy()  # (16, 59)
    return front, W


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_subject_npz(subj_id):
    p = DATA_DIR / f"S{subj_id}" / f"S{subj_id}_EEGNet_NoICA_uV.npz"
    z = np.load(p, allow_pickle=True)

    x_key = "X" if "X" in z.files else ("data" if "data" in z.files else None)
    y_key = "y" if "y" in z.files else ("labels" if "labels" in z.files else None)
    if x_key is None or y_key is None:
        raise KeyError(f"{p}: cannot find X/y in keys {z.files}")
    X = z[x_key]; y = z[y_key].astype(int)
    ch_names = list(z["ch_names"])

    if X.ndim == 3:                              # (N, C, T) -> (N, 1, C, T)
        X = X[:, None, :, :]
    if not (X.shape[1] == 1 and X.shape[2] == N_CHANNELS and X.shape[3] == N_SAMPLES):
        raise ValueError(f"{p}: unexpected X shape {X.shape}")
    return X.astype(np.float32), y, ch_names


def make_info(ch_names):
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    try:
        montage = mne.channels.read_custom_montage(MONTAGE_PATH)
    except ValueError as exc:
        print(f"  WARNING: failed to read custom montage ({exc}); using standard_1005.")
        montage = mne.channels.make_standard_montage("standard_1005")
    # match_case=False handles minor case discrepancies; on_missing='warn' will
    # warn if any of the 59 channels has no position. We expect all to match.
    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


# -----------------------------------------------------------------------------
# Raw EEG class power topography
# -----------------------------------------------------------------------------
def compute_raw_class_power(X, y, ch_names):
    """
    X: (N, 1, C, T) or (N, C, T).
    Return (6, C) per-class mean log-power, channel-wise z-scored.
    """
    Xc = X[:, 0] if X.ndim == 4 else X
    power = (Xc ** 2).mean(axis=-1)
    logp = np.log(power + 1e-12)
    non_eog = [i for i, ch in enumerate(ch_names) if ch not in EOG_CHS]
    if not non_eog:
        non_eog = list(range(Xc.shape[1]))

    topo = np.zeros((6, Xc.shape[1]))
    for k in range(6):
        m_in = y == k
        if m_in.sum() == 0:
            continue
        p = logp[m_in].mean(axis=0)
        mu = p[non_eog].mean()
        sd = p[non_eog].std()
        if sd < 1e-12:
            sd = 1.0
        topo[k] = (p - mu) / sd
    return topo


# -----------------------------------------------------------------------------
# Plotting helpers (uniform symmetric vlim per figure)
# -----------------------------------------------------------------------------
def _sym_vlim(arr):
    v = float(np.max(np.abs(arr)))
    if v == 0:
        v = 1e-12
    return -v, v


def plot_filter_grid(W, info, path, title=""):
    vlim = _sym_vlim(W)
    n_filters = W.shape[0]
    n_cols = 8 if n_filters > 16 else 4
    n_rows = int(np.ceil(n_filters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.2 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    im = None
    for f in range(n_filters):
        ax = axes[f // n_cols, f % n_cols]
        im, _ = mne.viz.plot_topomap(
            W[f], info, axes=ax, show=False,
            vlim=vlim, cmap="RdBu_r", sensors=True, contours=6,
            extrapolate="head",
            sphere=(0.0, 0.0, 0.0, 0.11),
        )
        ax.set_title(f"filter {f}", fontsize=9)
    for f in range(n_filters, n_rows * n_cols):
        axes[f // n_cols, f % n_cols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_patterns(patterns, info, path, suptitle=""):
    """patterns: (6, 59)."""
    vlim = _sym_vlim(patterns)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.6))
    im = None
    for k in range(6):
        ax = axes[k]
        im, _ = mne.viz.plot_topomap(
            patterns[k], info, axes=ax, show=False,
            vlim=vlim, cmap="RdBu_r", sensors=True, contours=6,
            extrapolate="head",
            sphere=(0.0, 0.0, 0.0, 0.11),
        )
        ax.set_title(f"{k}: {CLASS_NAMES[k]}", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-subject pipeline
# -----------------------------------------------------------------------------
def run_subject(subj_id, batch_size=64):
    print(f"\n=== S{subj_id} ===")
    out_dir = OUT_ROOT / f"S{subj_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / f"S{subj_id}_eegnet_noica.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: missing checkpoint {ckpt_path}")
        return None

    front, W = load_front_from_ckpt(ckpt_path)       # W: (n_filters, 59)
    X, y, ch_names = load_subject_npz(subj_id)
    info = make_info(ch_names)
    print(f"  X: {X.shape}, y: {y.shape}, classes: {sorted(np.unique(y).tolist())}")
    print(f"  spatial_filters W: {W.shape}, MI window samples: [{T_MI_START}, {T_MI_END})")

    # 1) raw filter grid
    plot_filter_grid(W, info, out_dir / "fig_per_filter.png",
                     title=f"S{subj_id} — {W.shape[0]} spatial filters (raw)")
    np.save(out_dir / "spatial_filters.npy", W)

    # 2) hook spatial_conv output (pre-BN)
    acts = []
    def hook(_m, _inp, out):
        acts.append(out.detach().cpu().numpy())
    h = front.spatial_conv.register_forward_hook(hook)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            front(torch.from_numpy(X[i:i + batch_size]))
    h.remove()
    A = np.concatenate(acts, axis=0)[:, :, 0, :]      # (N, n_filters, T=1126)
    del acts

    # 3) per-trial per-filter energy over MI window
    energy = (A[:, :, T_MI_START:T_MI_END] ** 2).mean(axis=-1)   # (N, n_filters)

    # 4) class scores: in-class vs out-of-class mean energy
    score = np.zeros((6, W.shape[0]), dtype=np.float64)
    for k in range(6):
        m_in = (y == k)
        if m_in.sum() == 0 or (~m_in).sum() == 0:
            print(f"  WARNING: class {k} empty for S{subj_id}")
            continue
        score[k] = energy[m_in].mean(0) - energy[~m_in].mean(0)

    # 5) class-conditional spatial patterns: (6, n_filters) @ (n_filters, 59) -> (6, 59)
    class_patterns = score @ W
    np.save(out_dir / "class_patterns.npy", class_patterns)
    np.save(out_dir / "scores.npy", score)
    plot_class_patterns(class_patterns, info, out_dir / "class_patterns.png",
                        suptitle=f"S{subj_id} — class-conditional patterns")

    # 6) Raw EEG per-class log-power z-score topography for top-row comparison.
    raw_topo = compute_raw_class_power(X, y, ch_names)
    np.save(out_dir / "raw_class_topo.npy", raw_topo)
    plot_class_patterns(
        raw_topo,
        info,
        out_dir / "raw_class_topo.png",
        suptitle=f"S{subj_id} — raw EEG class log-power z-score topography",
    )

    return class_patterns, raw_topo, ch_names


# -----------------------------------------------------------------------------
# Grand average + region-specificity check
# -----------------------------------------------------------------------------
def grand_average(per_subject, ch_names, tag="class_patterns"):
    info = make_info(ch_names)
    grand_dir = OUT_ROOT / "grand"
    grand_dir.mkdir(parents=True, exist_ok=True)

    # Per (subject, class) max-abs normalization
    normed = []
    for p in per_subject:
        n = p.copy().astype(np.float64)
        for k in range(6):
            m = np.max(np.abs(n[k]))
            if m > 0:
                n[k] /= m
        normed.append(n)
    arr = np.stack(normed, axis=0)                # (S, 6, 59)
    grand = arr.mean(axis=0)                      # (6, 59)
    np.save(grand_dir / f"grand_{tag}.npy", grand)
    plot_class_patterns(
        grand, info, grand_dir / f"grand_{tag}.png",
        suptitle=(
            f"Grand average {tag} across {len(per_subject)} subjects "
            f"(per-(subj,class) max-abs normed)"
        ),
    )

    # Region specificity per class
    ch_idx = {c: i for i, c in enumerate(ch_names)}
    rows = []
    for k in range(6):
        absp = np.abs(grand[k])
        roi_chs = [c for c in REGION_OF_INTEREST[k] if c in ch_idx]
        roi_idx = [ch_idx[c] for c in roi_chs]
        rest_idx = [i for i in range(len(ch_names)) if i not in roi_idx]

        roi_mean = float(absp[roi_idx].mean()) if roi_idx else float("nan")
        rest_mean = float(absp[rest_idx].mean()) if rest_idx else float("nan")
        ratio = roi_mean / rest_mean if rest_mean > 0 else float("nan")

        row = {
            "class": k,
            "name": CLASS_NAMES[k],
            "roi_channels": ",".join(roi_chs),
            "mean_abs_in_roi": roi_mean,
            "mean_abs_outside": rest_mean,
            "roi_vs_rest_ratio": ratio,
        }
        # Face: right-vs-left N170-style lateralization
        if k == 3:
            r_chs = [c for c in ["P8", "PO8"] if c in ch_idx]
            l_chs = [c for c in ["P7", "PO7"] if c in ch_idx]
            r_mean = float(absp[[ch_idx[c] for c in r_chs]].mean()) if r_chs else float("nan")
            l_mean = float(absp[[ch_idx[c] for c in l_chs]].mean()) if l_chs else float("nan")
            row["face_right_mean"] = r_mean
            row["face_left_mean"]  = l_mean
            row["face_right_vs_left_ratio"] = r_mean / l_mean if l_mean > 0 else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_name = "region_specificity.csv" if tag == "class_patterns" else f"region_specificity_{tag}.csv"
    df.to_csv(grand_dir / csv_name, index=False)
    print(f"\n=== Region specificity ({tag}, grand average) ===")
    print(df.to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None):
    global DATA_DIR, CKPT_DIR, MONTAGE_PATH, OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--montage", type=Path, default=MONTAGE_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)),
                    help="subject IDs to process (default: 1..10)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-grand", action="store_true",
                    help="skip grand average step")
    args = ap.parse_args(argv)

    DATA_DIR = args.data_dir
    CKPT_DIR = args.checkpoint_dir
    MONTAGE_PATH = args.montage
    OUT_ROOT = args.out_dir

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    per_subj_patterns = []
    per_subj_raw = []
    ref_ch_names = None
    for s in args.subjects:
        try:
            result = run_subject(s, batch_size=args.batch_size)
        except Exception as e:
            print(f"  ERROR S{s}: {type(e).__name__}: {e}")
            continue
        if result is None:
            continue
        cp, raw_topo, ch_names = result
        per_subj_patterns.append(cp)
        per_subj_raw.append(raw_topo)
        if ref_ch_names is None:
            ref_ch_names = ch_names

    if not args.skip_grand and per_subj_patterns:
        grand_average(per_subj_patterns, ref_ch_names, tag="class_patterns")
        grand_average(per_subj_raw, ref_ch_names, tag="raw_logpower_zscore")
    elif not per_subj_patterns:
        print("\nNo subject completed successfully; nothing to grand-average.")


if __name__ == "__main__":
    main()

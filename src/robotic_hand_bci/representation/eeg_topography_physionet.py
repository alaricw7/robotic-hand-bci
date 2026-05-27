#!/usr/bin/env python3
"""
Multi-band EEG topography for the 6-class cognitive-MI experiment.

Why this version:
  * Single band-pass (e.g. 8-30 Hz) often blanks out cognitive-imagery
    differences because the discriminative info lives in OTHER bands.
  * This script computes log-power topographies in multiple bands
    (theta / alpha / beta / wide), and for each band produces BOTH
    the absolute view and the class-contrast view.
  * The contrast view uses RAW (class - grand-mean) values with a
    symmetric vlim — no per-row z-score that erases magnitude info.

Run:
    python eeg_topography_multiband.py \
        --data-dir data/processed/pythondata1 \
        --montage  assets/montages/Standard-10-5-Cap385_witheog.elp \
        --out-dir  ./out_multiband \
        --sfreq 250
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, filtfilt

from robotic_hand_bci.project import DEFAULT_MONTAGE_PATH, PROJECT_ROOT

mne.set_log_level("WARNING")

DEFAULT_CLASS_NAMES = ["Manipulation", "ColorClass", "ColorTrack",
                       "Face", "RPS", "Gesture"]
EOG_CHS = {"Fp1", "Fpz", "Fp2", "AF3", "AF4", "AF7", "AF8"}
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "pythondata1"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "analysis" / "representation" / "topography_physionet"

# Frequency bands to scan — for cognitive-imagery tasks θ and α are
# usually MORE informative than β
BANDS = {
    "theta_4_7":   (4.0, 7.0),
    "alpha_8_13":  (8.0, 13.0),
    "beta_13_30":  (13.0, 30.0),
    "wide_4_30":   (4.0, 30.0),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--montage",  type=Path, default=DEFAULT_MONTAGE_PATH)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--class-names", nargs="+", default=None)
    p.add_argument("--sfreq", type=float, default=250.0)
    p.add_argument("--drop-frontal", action="store_true",
                   help="set Fp/AF channels to NaN before plotting "
                        "(use when ICA has not been applied)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


# ---------- IO ---------------------------------------------------------------
def discover_subjects(data_dir: Path, subjects) -> List[str]:
    if subjects:
        return [s if s.upper().startswith("S") else f"S{s}" for s in subjects]
    found = [p.name for p in sorted(data_dir.glob("S*"))
             if p.is_dir() and any(p.glob("*_EEGNet_NoICA_uV.npz"))]
    if not found:
        raise FileNotFoundError(f"no subject NPZ under {data_dir}")
    return found


def load_subject(data_dir: Path, subject: str):
    matches = sorted((data_dir / subject).glob("*_EEGNet_NoICA_uV.npz"))
    if not matches:
        raise FileNotFoundError(f"npz missing for {subject}")
    z = np.load(matches[0], allow_pickle=True)
    X = np.asarray(z["X" if "X" in z.files else "data"], dtype=np.float32)
    y = np.asarray(z["y" if "y" in z.files else "labels"], dtype=np.int64)
    ch_names = [str(c) for c in z["ch_names"].tolist()]
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    return X, y, ch_names


def build_info(ch_names, sfreq, montage_path):
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    montage = None
    if montage_path and Path(montage_path).exists():
        try:
            montage = mne.channels.read_custom_montage(montage_path)
        except Exception as e:
            print(f"[warn] custom montage failed: {e}; fall back to 10-05")
    if montage is None:
        montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


# ---------- DSP --------------------------------------------------------------
def bandpass(X, sfreq, lo, hi):
    nyq = 0.5 * sfreq
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1).astype(np.float32)


def class_log_power(X, y, labels):
    """(n_classes, n_channels) of log10(mean power per class)."""
    pwr = np.mean(X ** 2, axis=2)
    out = np.zeros((len(labels), X.shape[1]))
    for r, lab in enumerate(labels):
        out[r] = np.log10(pwr[y == lab].mean(axis=0) + 1e-12)
    return out


# ---------- plotting ---------------------------------------------------------
def sym_vlim(arr, percentile=99):
    """Robust symmetric vlim — uses high percentile so an outlier
    channel does not wash out the rest."""
    a = arr[~np.isnan(arr)]
    if a.size == 0:
        return -1.0, 1.0
    v = float(np.percentile(np.abs(a), percentile))
    return (-v, v) if v > 0 else (-1.0, 1.0)


def mask_frontal(topo, ch_names):
    topo = topo.copy()
    for i, ch in enumerate(ch_names):
        if ch in EOG_CHS:
            topo[:, i] = np.nan
    return topo


def plot_row(topo, info, class_names, title, out_path,
             dpi=200, vlim=None):
    if vlim is None:
        vlim = sym_vlim(topo)
    n = topo.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(max(3.4 * n, 10), 3.6), squeeze=False)
    axes = axes.ravel()
    im = None
    for ax, vals, name in zip(axes, topo, class_names):
        im, _ = mne.viz.plot_topomap(
            vals, info, axes=ax, show=False,
            cmap="RdBu_r", vlim=vlim, contours=6,
            sensors=True, outlines="head",
            # let MNE auto-pick sphere from montage — don't hardcode
        )
        ax.set_title(name, fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.75, pad=0.02)
    fig.suptitle(title, y=0.02, fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ---------- main -------------------------------------------------------------
def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(discover_subjects(args.data_dir, args.subjects))

    # accumulate per-band grand topographies
    band_topos = {b: [] for b in BANDS}
    ref_chs, labels = None, None

    for subj in subjects:
        try:
            X_raw, y, ch_names = load_subject(args.data_dir, subj)
            sl = sorted(int(v) for v in np.unique(y))
            if labels is None:
                labels, ref_chs = sl, ch_names
            elif sl != labels or ch_names != ref_chs:
                print(f"[skip] {subj}: label/channel mismatch")
                continue
            for bname, (lo, hi) in BANDS.items():
                Xf = bandpass(X_raw, args.sfreq, lo, hi)
                band_topos[bname].append(class_log_power(Xf, y, labels))
            print(f"[ok] {subj}")
        except Exception as e:
            print(f"[skip] {subj}: {type(e).__name__}: {e}")

    if not band_topos[list(BANDS.keys())[0]]:
        raise RuntimeError("no subjects processed")

    info = build_info(ref_chs, args.sfreq, args.montage)
    names = args.class_names or (DEFAULT_CLASS_NAMES
                                 if len(labels) == 6 else
                                 [f"class {l}" for l in labels])

    for bname, (lo, hi) in BANDS.items():
        grand = np.mean(np.stack(band_topos[bname], 0), axis=0)
        contrast = grand - grand.mean(axis=0, keepdims=True)  # RAW contrast, no z-score

        if args.drop_frontal:
            grand_plot    = mask_frontal(grand, ref_chs)
            contrast_plot = mask_frontal(contrast, ref_chs)
        else:
            grand_plot, contrast_plot = grand, contrast

        # (a) absolute log-power — center per row to remove DC offset between classes
        abs_centered = grand_plot - np.nanmean(grand_plot, axis=1, keepdims=True)
        plot_row(abs_centered, info, names,
                 f"(a) Log-power topography  [{lo:.0f}-{hi:.0f} Hz]  (per-class DC removed)",
                 args.out_dir / f"topo_abs_{bname}.png",
                 dpi=args.dpi, vlim=sym_vlim(abs_centered))

        # (b) class - grand mean: highlights class-specific deviations
        plot_row(contrast_plot, info, names,
                 f"(b) Class - grand mean   [{lo:.0f}-{hi:.0f} Hz]",
                 args.out_dir / f"topo_contrast_{bname}.png",
                 dpi=args.dpi, vlim=sym_vlim(contrast_plot))

    print("\nDone. Inspect outputs; the band with the strongest, most "
          "class-distinct pattern is your answer for visualization.")


if __name__ == "__main__":
    main()

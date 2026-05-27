#!/usr/bin/env python3
"""
Re-render fig_per_filter.png from already-saved spatial_filters.npy.
Grid size is auto-chosen from the actual number of filters (32 here, not 16).

Run after spayial_filters.py. It reuses the saved (n_filters, 59) array,
so we do not have to reload checkpoints or do another forward pass.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from robotic_hand_bci.project import DEFAULT_MONTAGE_PATH, PROJECT_ROOT

OUT_ROOT = PROJECT_ROOT / "artifacts" / "analysis" / "representation" / "spatial_filters_grid"
MONTAGE_PATH = DEFAULT_MONTAGE_PATH
SFREQ = 250.0

CH_NAMES = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
    "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
    "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
    "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz",
    "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
]


def robust_montage_info(ch_names):
    """Try .elp via multiple readers; fall back to standard_1005."""
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    montage = None
    last_err = None

    try:
        montage = mne.channels.read_custom_montage(MONTAGE_PATH)
    except Exception as exc:
        last_err = exc

    if montage is None:
        try:
            montage = mne.channels.read_polhemus_fastscan(MONTAGE_PATH, unit="mm")
        except Exception as exc:
            last_err = exc

    if montage is None:
        print(
            f"  .elp not parseable ({type(last_err).__name__}); "
            "falling back to standard_1005."
        )
        montage = mne.channels.make_standard_montage("standard_1005")

    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


def _sym_vlim(arr):
    v = float(np.max(np.abs(arr)))
    return (-v, v) if v > 0 else (-1e-12, 1e-12)


def _grid_shape(n):
    """Square-ish grid that fits all n filters: (rows, cols)."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_filter_grid(W, info, path, title=""):
    n_filters = W.shape[0]
    rows, cols = _grid_shape(n_filters)
    vlim = _sym_vlim(W)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.atleast_2d(axes)
    im = None

    for f in range(rows * cols):
        ax = axes[f // cols, f % cols]
        if f < n_filters:
            im, _ = mne.viz.plot_topomap(
                W[f],
                info,
                axes=ax,
                show=False,
                vlim=vlim,
                cmap="RdBu_r",
                sensors=True,
                contours=6,
                extrapolate="head",
                sphere="auto",
            )
            ax.set_title(f"filter {f}", fontsize=9)
        else:
            ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    global OUT_ROOT, MONTAGE_PATH
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--montage", type=Path, default=MONTAGE_PATH)
    args = parser.parse_args(argv)
    OUT_ROOT = args.out_root
    MONTAGE_PATH = args.montage

    info = robust_montage_info(CH_NAMES)
    subj_dirs = sorted(
        [p for p in OUT_ROOT.glob("S*") if p.is_dir()],
        key=lambda p: int(p.name[1:]),
    )
    if not subj_dirs:
        print(f"No S* dirs in {OUT_ROOT}")
        return

    for sd in subj_dirs:
        npy = sd / "spatial_filters.npy"
        if not npy.exists():
            print(f"  skip {sd.name}: missing spatial_filters.npy")
            continue
        W = np.load(npy)
        out = sd / "fig_per_filter.png"
        plot_filter_grid(
            W,
            info,
            out,
            title=f"{sd.name} - {W.shape[0]} spatial filters (raw)",
        )
        print(f"  {sd.name}: W={tuple(W.shape)} -> {out.name}")


if __name__ == "__main__":
    main()

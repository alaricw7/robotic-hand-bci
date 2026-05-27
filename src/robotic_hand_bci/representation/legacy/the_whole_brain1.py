"""Whole-brain topomap visualization from consolidated EEGNet NPZ data.

Reads ``S*_EEGNet_NoICA_uV.npz`` files, slices trials by event code 11..16,
concatenates trials across subjects, band-pass filters into standard bands,
and plots scalp topographies at fixed time points.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import plot_topomap


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45),
    "All":   (0.5, 45),
}

EVENT_CODES_INT: tuple[int, ...] = (11, 12, 13, 14, 15, 16)

PLOT_TIMES_MS: tuple[int, ...] = (
    -500, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500,
)

VMIN, VMAX = -3.0, 3.0  # µV

TOPOMAP_KW = dict(
    sensors=True,
    contours=6,
    cmap="RdBu_r",
    extrapolate="head",
    sphere=(0.0, 0.0, 0.0, 0.11),
    outlines="head",
)


def _configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    subjects: tuple[str, ...]
    apply_baseline: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/wong/robotic-hand-bci/data/processed/pythondata1"),
        help="Directory containing S*/S*_EEGNet_NoICA_uV.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wong/robotic-hand-bci/representation/oldcode/results/"
                     "pythondata1_whole_brain_topomap_npz"),
        help="Output directory for topomap figures.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to include (e.g. S1 S2). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not apply the NPZ baseline interval before filtering.",
    )
    ns = parser.parse_args()
    subjects = normalize_subjects(ns.subjects) if ns.subjects else discover_subjects(ns.data_dir)
    return Config(ns.data_dir, ns.output_dir, subjects, not ns.no_baseline)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

_SUBJECT_RE = re.compile(r"^(S\d+)_EEGNet_NoICA_uV\.npz$")


def normalize_subjects(subjects: Sequence[str]) -> tuple[str, ...]:
    return tuple(s if str(s).upper().startswith("S") else f"S{s}" for s in subjects)


def discover_subjects(data_dir: Path) -> tuple[str, ...]:
    """Find all subjects by looking for ``S*_EEGNet_NoICA_uV.npz`` files."""
    matches = (
        _SUBJECT_RE.match(p.name)
        for p in data_dir.rglob("*_EEGNet_NoICA_uV.npz")
    )
    subjects = sorted(
        {m.group(1) for m in matches if m},
        key=lambda s: int(s[1:]),
    )
    if not subjects:
        raise FileNotFoundError(
            f"No *_EEGNet_NoICA_uV.npz files found under {data_dir}"
        )
    return tuple(subjects)


def resolve_npz_path(data_dir: Path, subject: str) -> Path | None:
    """Find ``{subject}_EEGNet_NoICA_uV.npz`` either flat or under a subj folder."""
    candidates = [
        data_dir / f"{subject}_EEGNet_NoICA_uV.npz",
        data_dir / subject / f"{subject}_EEGNet_NoICA_uV.npz",
    ]
    return next((p for p in candidates if p.exists()), None)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def build_info(ch_names: Sequence[str], sfreq: float) -> mne.Info:
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


def load_subject_npz(npz_path: Path) -> dict[str, object]:
    """Load one subject's consolidated NPZ once.

    ``X`` is stored in microvolts and converted to volts only when building
    the MNE EpochsArray.
    """
    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["X" if "X" in z.files else "data"], dtype=np.float64)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    if X.ndim != 3:
        raise ValueError(f"{npz_path}: expected X shape (trials, channels, times), got {X.shape}")

    y_key = "event_codes" if "event_codes" in z.files else ("y" if "y" in z.files else "labels")
    event_codes = np.asarray(z[y_key], dtype=np.int64)
    if y_key in {"y", "labels"} and event_codes.size and event_codes.min() == 0:
        event_codes = event_codes + 11

    ch_names = [str(ch) for ch in z["ch_names"].tolist()]
    if len(ch_names) != X.shape[1]:
        raise ValueError(f"{npz_path}: {len(ch_names)} channel names for X shape {X.shape}")
    if event_codes.shape[0] != X.shape[0]:
        raise ValueError(f"{npz_path}: event_codes shape {event_codes.shape} for X shape {X.shape}")

    baseline = None
    if "config" in z.files:
        # The pythondata1 config records baseline [-0.5, 0.0]. Keep the parser
        # deliberately conservative so malformed config never blocks loading.
        import json
        try:
            cfg = json.loads(str(z["config"].item()))
            if "baseline" in cfg and cfg["baseline"] is not None:
                baseline = tuple(float(v) for v in cfg["baseline"])
        except Exception:
            baseline = None
    if baseline is None:
        baseline = (-0.5, 0.0)

    return {
        "path": npz_path,
        "X": X,
        "event_codes": event_codes,
        "ch_names": ch_names,
        "sfreq": float(np.asarray(z["sfreq"]).squeeze()),
        "tmin": float(np.asarray(z["tmin"]).squeeze()),
        "baseline": baseline,
    }


def load_condition_epochs(
    subject_npzs: Iterable[dict[str, object]],
    event_code: int,
    apply_baseline: bool,
) -> mne.EpochsArray | None:
    """Concatenate trials across subjects for one event code."""
    trials: list[np.ndarray] = []
    info: mne.Info | None = None
    tmin: float | None = None
    baseline: tuple[float, float] | None = None
    ref_ch_names: list[str] | None = None
    ref_sfreq: float | None = None

    for subject_data in subject_npzs:
        X = subject_data["X"]
        event_codes = subject_data["event_codes"]
        ch_names = subject_data["ch_names"]
        sfreq = subject_data["sfreq"]
        mask = event_codes == event_code
        if not mask.any():
            continue

        if info is None:
            info = build_info(ch_names, sfreq)
            tmin = float(subject_data["tmin"])
            baseline = subject_data["baseline"]
            ref_ch_names = list(ch_names)
            ref_sfreq = float(sfreq)
        elif list(ch_names) != ref_ch_names or float(sfreq) != ref_sfreq:
            log.warning("Skipping %s: channel names or sfreq differ from reference",
                        subject_data["path"])
            continue

        trials.append(X[mask] * 1e-6)  # µV → V

    if not trials:
        return None

    epochs = mne.EpochsArray(
        np.concatenate(trials, axis=0),
        info,
        tmin=tmin,
        verbose="ERROR",
    )
    if apply_baseline and baseline is not None:
        epochs.apply_baseline(baseline, verbose="ERROR")
    return epochs


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_topomap_grid(
    evoked_uv: np.ndarray,
    times_ms: np.ndarray,
    info: mne.Info,
    plot_times_ms: Iterable[float],
    title: str,
    save_path: Path,
    vmin: float = VMIN,
    vmax: float = VMAX,
) -> None:
    """Plot one topomap per time point in a single horizontal row.

    Parameters
    ----------
    evoked_uv : (n_channels, n_times) array in µV
    times_ms : (n_times,) time axis in ms
    """
    times_to_plot = tuple(plot_times_ms)
    n = len(times_to_plot)

    fig = plt.figure(figsize=(1.7 * n + 0.9, 2.4), facecolor="white")
    gs = fig.add_gridspec(
        1, n + 1,
        width_ratios=[1] * n + [0.06],
        wspace=0.15, left=0.02, right=0.96, top=0.82, bottom=0.06,
    )

    im = None
    for col, t_ms in enumerate(times_to_plot):
        ax = fig.add_subplot(gs[0, col])
        sample_idx = int(np.argmin(np.abs(times_ms - t_ms)))
        im, _ = plot_topomap(
            evoked_uv[:, sample_idx], info,
            axes=ax, show=False, vlim=(vmin, vmax),
            **TOPOMAP_KW,
        )
        ax.set_title(f"{int(t_ms)} ms", pad=4)

    cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("µV", rotation=0, labelpad=8, va="center")
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.95, title, ha="center", va="top",
             fontsize=12, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def process_band(
    epochs: mne.EpochsArray,
    band_name: str,
    f_low: float,
    f_high: float,
    event_code: str,
    output_dir: Path,
) -> None:
    log.info("Event %s · %s (%g–%g Hz)", event_code, band_name, f_low, f_high)

    filtered = epochs.copy().filter(l_freq=f_low, h_freq=f_high, verbose="ERROR")
    evoked_uv = filtered.get_data().mean(axis=0) * 1e6  # V → µV
    times_ms = filtered.times * 1000

    save_path = output_dir / event_code / f"Event{event_code}_{band_name}_Topo.png"
    title = f"Event {event_code}  ·  {band_name} ({f_low}–{f_high} Hz)"

    plot_topomap_grid(
        evoked_uv, times_ms, filtered.info,
        PLOT_TIMES_MS, title, save_path,
    )
    log.info("Saved %s", save_path)


def run(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    subject_npzs: list[dict[str, object]] = []
    for subject in cfg.subjects:
        npz_path = resolve_npz_path(cfg.data_dir, subject)
        if npz_path is None:
            log.warning("Missing NPZ for %s under %s", subject, cfg.data_dir)
            continue
        log.info("Loading %s", npz_path)
        subject_npzs.append(load_subject_npz(npz_path))

    if not subject_npzs:
        raise RuntimeError(f"No subject NPZ files loaded from {cfg.data_dir}")

    for event_code in EVENT_CODES_INT:
        epochs = load_condition_epochs(subject_npzs, event_code, cfg.apply_baseline)
        if epochs is None:
            log.warning("No data for event %d — skipped", event_code)
            continue

        for band_name, (f_low, f_high) in FREQ_BANDS.items():
            process_band(epochs, band_name, f_low, f_high,
                         str(event_code), cfg.output_dir)

    log.info("All processing complete.")


# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _configure_matplotlib()
    run(parse_args())


if __name__ == "__main__":
    main()

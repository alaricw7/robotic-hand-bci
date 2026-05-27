"""EEG Spectral Analysis - All Channels (NPZ version)
=====================================================

Reads consolidated ``S*_EEGNet_NoICA_uV.npz`` files (data in µV), slices
trials by event code 11..16, averages within subjects, and computes a
multi-panel ERSP figure for every EEG channel. A one-way ANOVA across
tasks at each channel's peak-activation time provides a per-channel
p-value displayed in the figure title.

Pipeline
--------
1. Load each subject's consolidated NPZ once.
2. For every task (event code 11..16):
   - Slice trials by event code, optionally apply time-domain baseline
     subtraction, then average across trials per subject.
   - Stack subjects along a new axis  -> (n_channels, n_times, n_subjects).
3. For each channel:
   - Locate the peak (max |amplitude|) within the ANOVA search window.
   - Run a one-way ANOVA across tasks on subject-level peak values.
   - Compute Morlet-wavelet power, apply dB baseline correction over
     [-500, 0] ms, and plot one ERSP panel per task.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet
from scipy import stats

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EVENT_CODES_INT: tuple[int, ...] = (11, 12, 13, 14, 15, 16)
TASKS: tuple[str, ...] = tuple(f"Task{i}" for i in range(1, 7))

# Frequency grid for Morlet time-frequency decomposition
FREQS = np.linspace(0.5, 45.0, 50)
N_CYCLES = FREQS / 2.0  # adaptive cycles - common ERSP choice

# Peak-search window for the ANOVA (ms). The original MATLAB code searched
# from the first sample to 1000 ms; here we make the lower bound explicit.
ANOVA_TMIN_MS, ANOVA_TMAX_MS = -500.0, 1000.0

# dB baseline interval (ms) - matches the original MATLAB code
DB_BASELINE_MS: tuple[float, float] = (-500.0, 0.0)

# ERSP color scale (dB)
ERSP_VMIN, ERSP_VMAX = -15.0, 15.0


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
                     "pythondata1_All_Tasks_ERSP"),
        help="Output directory for per-channel ERSP figures.",
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
        help="Do not apply the NPZ baseline interval before averaging.",
    )
    ns = parser.parse_args()
    subjects = (normalize_subjects(ns.subjects)
                if ns.subjects else discover_subjects(ns.data_dir))
    return Config(ns.data_dir, ns.output_dir, subjects, not ns.no_baseline)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

_SUBJECT_RE = re.compile(r"^(S\d+)_EEGNet_NoICA_uV\.npz$")


def normalize_subjects(subjects: Sequence[str]) -> tuple[str, ...]:
    return tuple(s if str(s).upper().startswith("S") else f"S{s}" for s in subjects)


def discover_subjects(data_dir: Path) -> tuple[str, ...]:
    """Find subjects by scanning for ``S*_EEGNet_NoICA_uV.npz`` files."""
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
# NPZ loading
# --------------------------------------------------------------------------- #

def load_subject_npz(npz_path: Path) -> dict:
    """Load one subject's consolidated NPZ once.

    Returns a dict with the trial tensor still in µV. For ERSP we keep the
    data in µV throughout: the dB baseline correction is a ratio, so unit
    scaling cancels out.
    """
    z = np.load(npz_path, allow_pickle=True)

    X = np.asarray(z["X" if "X" in z.files else "data"], dtype=np.float64)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    if X.ndim != 3:
        raise ValueError(
            f"{npz_path}: expected X shape (trials, channels, times), got {X.shape}"
        )

    # Event codes default to original 11..16 range; fall back to y+11 if the
    # NPZ stores zero-based labels instead.
    y_key = ("event_codes" if "event_codes" in z.files
             else "y" if "y" in z.files else "labels")
    event_codes = np.asarray(z[y_key], dtype=np.int64)
    if y_key in {"y", "labels"} and event_codes.size and event_codes.min() == 0:
        event_codes = event_codes + 11

    ch_names = [str(c) for c in z["ch_names"].tolist()]
    if len(ch_names) != X.shape[1]:
        raise ValueError(
            f"{npz_path}: {len(ch_names)} channel names for X shape {X.shape}"
        )
    if event_codes.shape[0] != X.shape[0]:
        raise ValueError(
            f"{npz_path}: event_codes shape {event_codes.shape} for X shape {X.shape}"
        )

    sfreq = float(np.asarray(z["sfreq"]).squeeze())
    tmin = float(np.asarray(z["tmin"]).squeeze())
    times_s = tmin + np.arange(X.shape[-1]) / sfreq

    # Default baseline matches the original pipeline; override via config json.
    baseline: tuple[float, float] = (-0.5, 0.0)
    if "config" in z.files:
        try:
            cfg = json.loads(str(z["config"].item()))
            bl = cfg.get("baseline")
            if bl is not None:
                baseline = (float(bl[0]), float(bl[1]))
        except Exception:
            pass

    return {
        "path": npz_path,
        "X": X,
        "event_codes": event_codes,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "tmin": tmin,
        "times_s": times_s,
        "baseline": baseline,
    }


def baseline_correct(trials: np.ndarray,
                     times_s: np.ndarray,
                     baseline: tuple[float, float]) -> np.ndarray:
    """Per-trial baseline subtraction (equivalent to MNE apply_baseline)."""
    t0, t1 = baseline
    mask = (times_s >= t0) & (times_s <= t1)
    if not mask.any():
        return trials
    return trials - trials[..., mask].mean(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Build per-task data structure
# --------------------------------------------------------------------------- #

def build_data_struct(
    subject_npzs: list[dict],
    apply_baseline: bool,
) -> tuple[dict[str, np.ndarray], list[str], float, np.ndarray]:
    """
    For each task: average trials within subject, then stack subjects.

    Returns
    -------
    data_struct : dict[str, np.ndarray]
        Task name -> array of shape (n_channels, n_times, n_subjects).
    ch_names, sfreq, times_ms : metadata pulled from the first subject.
    """
    ref = subject_npzs[0]
    ref_ch_names = ref["ch_names"]
    ref_sfreq = ref["sfreq"]
    ref_times_s = ref["times_s"]

    data_struct: dict[str, np.ndarray] = {}

    for code, task in zip(EVENT_CODES_INT, TASKS):
        per_subject_means: list[np.ndarray] = []
        for s in subject_npzs:
            if s["ch_names"] != ref_ch_names or s["sfreq"] != ref_sfreq:
                log.warning("Skipping %s for %s: metadata mismatch with %s",
                            s["path"], task, ref["path"])
                continue
            mask = s["event_codes"] == code
            if not mask.any():
                log.warning("%s has no trials for event %d", s["path"], code)
                continue

            trials = s["X"][mask]                       # (n_trials, n_ch, n_times) µV
            if apply_baseline:
                trials = baseline_correct(trials, s["times_s"], s["baseline"])
            per_subject_means.append(trials.mean(axis=0))   # (n_ch, n_times)

        if not per_subject_means:
            log.warning("No subjects contributed for event %d (%s)", code, task)
            continue

        data_struct[f"epoch_mean_{task}"] = np.stack(per_subject_means, axis=-1)

    return data_struct, ref_ch_names, ref_sfreq, ref_times_s * 1000.0


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_channel_ersp(
    ch_name: str,
    all_avg: list[np.ndarray | None],
    p_value: float,
    times_ms: np.ndarray,
    sfreq: float,
    save_path: Path,
) -> None:
    """One column of stacked ERSP panels (one per task)."""
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(6, 10),
                             constrained_layout=True)
    fig.patch.set_facecolor("w")

    baseline_mask = ((times_ms >= DB_BASELINE_MS[0]) &
                     (times_ms <= DB_BASELINE_MS[1]))
    if not baseline_mask.any():
        log.warning("dB baseline window %s falls outside times_ms; using full window",
                    DB_BASELINE_MS)
        baseline_mask = np.ones_like(times_ms, dtype=bool)

    for task_idx, task in enumerate(TASKS):
        ax = axes[task_idx]
        task_data = all_avg[task_idx]   # (n_times, n_subjects) or None
        if task_data is None:
            ax.set_title(f"{task} ERSP (no data)", fontsize=11)
            ax.set_axis_off()
            continue

        # MNE expects (n_epochs, n_channels, n_times); treat subjects as epochs.
        tf_input = task_data.T[:, np.newaxis, :]

        power = tfr_array_morlet(
            tf_input,
            sfreq=sfreq,
            freqs=FREQS,
            n_cycles=N_CYCLES,
            output="power",
            verbose=False,
        )                                              # (n_subj, 1, n_freqs, n_times)
        power_mean = power.mean(axis=0).squeeze(axis=0)  # (n_freqs, n_times)

        baseline_power = power_mean[:, baseline_mask].mean(axis=1, keepdims=True)
        baseline_power = np.where(baseline_power > 0,
                                  baseline_power, np.finfo(float).eps)
        ersp_db = 10.0 * np.log10(power_mean / baseline_power)

        im = ax.imshow(
            ersp_db,
            aspect="auto",
            origin="lower",
            extent=[times_ms[0], times_ms[-1], FREQS[0], FREQS[-1]],
            cmap="jet",
            vmin=ERSP_VMIN, vmax=ERSP_VMAX,
        )
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("dB", rotation=0, labelpad=12, va="center")

        ax.set_yticks(np.arange(0, 46, 5))
        ax.set_xticks(np.arange(-500, times_ms[-1] + 1, 500))
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel("Frequency (Hz)", fontsize=11)
        ax.axvline(x=0, color="k", linewidth=1)
        ax.set_title(f"{task} ERSP", fontsize=11)

    fig.suptitle(f"{ch_name} (p = {p_value:.4f})",
                 fontsize=14, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    subject_npzs: list[dict] = []
    for subj in cfg.subjects:
        npz_path = resolve_npz_path(cfg.data_dir, subj)
        if npz_path is None:
            log.warning("Missing NPZ for %s under %s", subj, cfg.data_dir)
            continue
        log.info("Loading %s", npz_path)
        subject_npzs.append(load_subject_npz(npz_path))

    if not subject_npzs:
        raise RuntimeError(f"No NPZ files loaded from {cfg.data_dir}")

    log.info("Building per-task data struct from %d subjects", len(subject_npzs))
    data_struct, ch_names, sfreq, times_ms = build_data_struct(
        subject_npzs, cfg.apply_baseline,
    )

    n_channels = len(ch_names)
    log.info("Loaded: %d channels, %d timepoints, %d subjects",
             n_channels, len(times_ms), len(subject_npzs))
    log.info("Sampling rate: %g Hz, time range: %.0f to %.0f ms",
             sfreq, times_ms[0], times_ms[-1])
    log.info("Subjects per task: %s",
             {t: data_struct[f"epoch_mean_{t}"].shape[-1]
              for t in TASKS if f"epoch_mean_{t}" in data_struct})

    # ANOVA peak-search window indices, derived from times_ms
    anova_start = max(int(np.searchsorted(times_ms, ANOVA_TMIN_MS, side="left")), 0)
    anova_end = min(int(np.searchsorted(times_ms, ANOVA_TMAX_MS, side="right")),
                    len(times_ms))
    log.info("ANOVA peak-search window: samples [%d, %d) = [%g, %g] ms",
             anova_start, anova_end,
             times_ms[anova_start], times_ms[anova_end - 1])

    for ch_idx, ch_name in enumerate(ch_names):
        all_avg: list[np.ndarray | None] = []
        peak_values_by_task: list[np.ndarray | None] = []

        for task in TASKS:
            key = f"epoch_mean_{task}"
            if key not in data_struct:
                all_avg.append(None)
                peak_values_by_task.append(None)
                continue

            avg = data_struct[key][ch_idx, :, :]   # (n_times, n_subjects)
            row_avg = avg.mean(axis=1)             # mean across subjects

            # Locate peak (max |amplitude|) inside the ANOVA window
            sub = row_avg[anova_start:anova_end]
            max_idx_subset = int(np.argmax(np.abs(sub)))
            target_idx = anova_start + max_idx_subset

            all_avg.append(avg)
            peak_values_by_task.append(avg[target_idx, :])   # (n_subjects,)

        valid = [v for v in peak_values_by_task if v is not None]
        if len(valid) < 2:
            log.warning("Channel %s: not enough tasks for ANOVA, skipping",
                        ch_name)
            continue

        _, p_value = stats.f_oneway(*valid)

        save_path = cfg.output_dir / f"{ch_name}_AllTasks_ERSP.png"
        plot_channel_ersp(ch_name, all_avg, p_value, times_ms, sfreq, save_path)
        log.info("[%3d/%d] %-8s processed  p = %.4f",
                 ch_idx + 1, n_channels, ch_name, p_value)

    log.info("All channels processed successfully!")


# --------------------------------------------------------------------------- #

def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
    })


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
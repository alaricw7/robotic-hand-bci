"""EEG 多频段、多任务、按通道时程绘图（NPZ 版本）。

读取合并后的 ``S*_EEGNet_NoICA_uV.npz``，按 event code 11..16 切分 trial，
在每个频段下进行零相位 FIR 带通滤波，对所有被试做被试内平均后跨被试堆叠，
然后逐通道绘制时程曲线并附带单因素 ANOVA 的 p 值。

与原 ``polt2(2).m`` Python 等价版的差异
--------------------------------------
- 数据源由 ``S*_EEG_{1..6}.set`` 改为单一 NPZ（µV 单位）。
- 滤波改用 ``mne.filter.filter_data`` 直接作用在 numpy 数组上，
  避免每次构造 Epochs 对象；FIR 参数（firwin/hamming/zero-phase/auto 带宽）
  与原版完全一致。
- 峰值搜索窗口由硬编码 ``n_end=374`` 改为按时间显式的 [-500, 1000] ms，
  通过 ``searchsorted`` 推算索引；在原数据集上结果完全相同。
- ANOVA 行为与原版严格一致：保留 ``one_way_anova_by_columns``
  （即 MATLAB ``anova1`` 默认的“列为组”语义，等同于以被试为组做方差分析）。
- 文件名前缀 ``6_`` 同样为复刻 MATLAB 脚本中
  ``epochs_index`` 循环外取末值（=6）的遗留行为，未做修改。
"""

from __future__ import annotations

import argparse
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
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EVENT_CODES_INT: tuple[int, ...] = (11, 12, 13, 14, 15, 16)
TASKS: tuple[str, ...] = tuple(f"Task{i}" for i in range(1, 7))

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 45.0),
    "All":   (0.5, 45.0),
}

# 峰值搜索时间窗，对应原版 m_start=0, n_end=374（sfreq=250 Hz 下 ~ -500..996 ms）
ANOVA_TMIN_MS, ANOVA_TMAX_MS = -500.0, 1000.0

# 画图时的 baseline 校正区间（ms）
PLOT_BASELINE_MS: tuple[float, float] = (-500.0, 0.0)

COLORS = np.array([
    [0.8, 0.0, 0.8],   # Task1 - Purple
    [0.4, 0.4, 0.4],   # Task2 - Gray
    [0.2, 0.8, 0.2],   # Task3 - Green
    [0.0, 0.4, 1.0],   # Task4 - Blue
    [1.0, 0.6, 0.0],   # Task5 - Orange
    [0.5, 0.0, 0.5],   # Task6 - Dark Purple
])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    subjects: tuple[str, ...]


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
                     "pythondata1_the_brain_channels"),
        help="Output directory for per-band per-channel time-course PNGs.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to include (e.g. S1 S2). Defaults to auto-detect.",
    )
    ns = parser.parse_args()
    subjects = (normalize_subjects(ns.subjects)
                if ns.subjects else discover_subjects(ns.data_dir))
    return Config(ns.data_dir, ns.output_dir, subjects)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

_SUBJECT_RE = re.compile(r"^(S\d+)_EEGNet_NoICA_uV\.npz$")


def normalize_subjects(subjects: Sequence[str]) -> tuple[str, ...]:
    return tuple(s if str(s).upper().startswith("S") else f"S{s}" for s in subjects)


def discover_subjects(data_dir: Path) -> tuple[str, ...]:
    """通过 ``S*_EEGNet_NoICA_uV.npz`` 自动检测被试。"""
    matches = (_SUBJECT_RE.match(p.name)
               for p in data_dir.rglob("*_EEGNet_NoICA_uV.npz"))
    subjects = sorted(
        {m.group(1) for m in matches if m},
        key=lambda s: int(s[1:]),
    )
    if not subjects:
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到任何 *_EEGNet_NoICA_uV.npz 文件"
        )
    return tuple(subjects)


def resolve_npz_path(data_dir: Path, subject: str) -> Path | None:
    """支持扁平或 ``{subject}/`` 子目录两种摆放方式。"""
    for cand in (
        data_dir / f"{subject}_EEGNet_NoICA_uV.npz",
        data_dir / subject / f"{subject}_EEGNet_NoICA_uV.npz",
    ):
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------------------- #
# NPZ loading
# --------------------------------------------------------------------------- #

def load_subject_npz(npz_path: Path) -> dict:
    """加载单个被试的合并 NPZ。

    返回字典含 ``X``（µV，shape=(trials, channels, times)）以及元数据。
    本脚本全程使用 µV，不向 Volts 换算 —— 输出图直接显示 µV 数值。
    """
    z = np.load(npz_path, allow_pickle=True)

    X = np.asarray(z["X" if "X" in z.files else "data"], dtype=np.float64)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    if X.ndim != 3:
        raise ValueError(
            f"{npz_path}: expected X shape (trials, channels, times), got {X.shape}"
        )

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

    return {
        "path": npz_path,
        "X": X,                         # µV, (trials, channels, times)
        "event_codes": event_codes,
        "ch_names": ch_names,
        "sfreq": sfreq,
        "tmin": tmin,
        "times_s": times_s,
    }


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def filter_trials(X: np.ndarray, sfreq: float,
                  l_freq: float, h_freq: float) -> np.ndarray:
    """带通滤波，沿最后一维。

    参数与原版 ``epochs.filter(...)`` 一致：
    FIR + firwin design + Hamming 窗 + 零相位 + auto 过渡带。
    """
    return mne.filter.filter_data(
        X.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        fir_window="hamming",
        phase="zero",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        verbose=False,
        copy=True,
    )


# --------------------------------------------------------------------------- #
# Build per-task data for one band
# --------------------------------------------------------------------------- #

def build_band_data_struct(
    subject_npzs: list[dict],
    l_freq: float,
    h_freq: float,
) -> dict[str, np.ndarray]:
    """对每个被试在指定频段下滤波后，按 event code 切分 trial，按 trial 求平均，
    再跨被试堆叠。

    Returns
    -------
    dict[str, np.ndarray]
        task name -> (n_channels, n_times, n_subjects)
    """
    ref = subject_npzs[0]
    ref_ch_names = ref["ch_names"]
    ref_sfreq = ref["sfreq"]

    per_task_subj_means: dict[str, list[np.ndarray]] = {t: [] for t in TASKS}

    for s in subject_npzs:
        if s["ch_names"] != ref_ch_names or s["sfreq"] != ref_sfreq:
            log.warning("Skipping %s: metadata mismatch with reference", s["path"])
            continue

        log.info("  Filtering %s @ %g–%g Hz", s["path"].name, l_freq, h_freq)
        X_filt = filter_trials(s["X"], s["sfreq"], l_freq, h_freq)

        for code, task in zip(EVENT_CODES_INT, TASKS):
            mask = s["event_codes"] == code
            if not mask.any():
                log.warning("    %s has no trials for event %d", s["path"].name, code)
                continue
            # (n_trials, n_ch, n_times) -> mean over trials -> (n_ch, n_times)
            per_task_subj_means[task].append(X_filt[mask].mean(axis=0))

        del X_filt   # 立即释放，避免下一个被试时叠加占用

    out: dict[str, np.ndarray] = {}
    for task in TASKS:
        means = per_task_subj_means[task]
        if means:
            out[task] = np.stack(means, axis=-1)   # (n_ch, n_times, n_subj)
    return out


# --------------------------------------------------------------------------- #
# ANOVA
# --------------------------------------------------------------------------- #

def one_way_anova_by_columns(X: np.ndarray) -> float:
    """与 MATLAB ``anova1(X)`` 默认行为一致：把 X 的每一列当作一组。

    输入约定：``X`` shape=(n_obs_per_group, n_groups)。返回 p 值。

    注意
    ----
    本脚本调用时传入 ``p_matrixdata`` 的 shape 是 (n_tasks, n_subjects)，
    因此“列为组”等同于以**被试**为组做方差分析 —— 这是原 MATLAB
    脚本的行为，特意保留。如果想改为以任务为组比较 task 差异，
    传入 ``X.T`` 或换用 ``f_oneway(*X)``。
    """
    if X.ndim != 2 or X.shape[1] < 2:
        return float("nan")
    groups = [X[:, c] for c in range(X.shape[1])]
    try:
        _, p = f_oneway(*groups)
        return float(p)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_channel_timecourse(
    ch_name: str,
    band_name: str,
    times_ms: np.ndarray,
    mean_value: np.ndarray,   # (n_times, n_tasks)
    std_value: np.ndarray,    # (n_times, n_tasks)
    p_value: float,
    save_path: Path,
) -> None:
    """单通道、单频段下，6 个 task 的时程曲线（被试均值 ± std）。"""
    # Post-hoc plot baseline correction（仅作用在被试均值上）
    bl_idx = (times_ms >= PLOT_BASELINE_MS[0]) & (times_ms <= PLOT_BASELINE_MS[1])
    if bl_idx.any():
        baseline_mean = mean_value[bl_idx, :].mean(axis=0)
        mean_bc = mean_value - baseline_mean
    else:
        mean_bc = mean_value

    fig, ax = plt.subplots(figsize=(8, 5))
    handles = []
    for i, task in enumerate(TASKS):
        upper = mean_bc[:, i] + std_value[:, i]
        lower = mean_bc[:, i] - std_value[:, i]
        h, = ax.plot(times_ms, mean_bc[:, i],
                     color=COLORS[i], linewidth=2, label=task)
        ax.fill_between(times_ms, lower, upper,
                        color=COLORS[i], alpha=0.1, edgecolor="none")
        handles.append(h)

    # 锁定 y 轴范围后再画 x=0 竖线
    y_lim = ax.get_ylim()
    ax.plot([0, 0], y_lim, "k-", linewidth=1.5)
    ax.set_ylim(y_lim)

    ax.set_xlim([times_ms[0], times_ms[-1]])
    ax.set_xticks(np.arange(-500, times_ms[-1] + 1, 500))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude [μV]")
    ax.set_title(ch_name)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    p_text = f"p = {p_value:.4f}" if not np.isnan(p_value) else "p = NaN"
    ax.text(np.mean(x_lim), y_lim[1], p_text,
            ha="center", va="top", fontsize=12, fontweight="bold")

    info_str = f"Freq: {band_name}, Channel: {ch_name}"
    ax.text(x_lim[0] + 0.05 * (x_lim[1] - x_lim[0]),
            y_lim[1] - 0.05 * (y_lim[1] - y_lim[0]),
            info_str, va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="black"))

    ax.legend(handles=handles, loc="best")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 避免通道名里有非法字符
    safe_name = re.sub(r"[^\w\-\.]", "_", save_path.name)
    fig.savefig(save_path.with_name(safe_name), dpi=100, bbox_inches="tight")
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

    print(f"Found {len(subject_npzs)} subjects: "
          f"{[s['path'].stem.split('_')[0] for s in subject_npzs]}")

    ref = subject_npzs[0]
    ch_names = ref["ch_names"]
    sfreq = ref["sfreq"]
    times_ms = ref["times_s"] * 1000.0
    n_channels = len(ch_names)

    log.info("Channels: %d, time points: %d, sfreq: %g Hz, range: %g..%g ms",
             n_channels, len(times_ms), sfreq, times_ms[0], times_ms[-1])

    # ANOVA 峰值搜索区间索引
    anova_start = max(int(np.searchsorted(times_ms, ANOVA_TMIN_MS, side="left")), 0)
    anova_end = min(int(np.searchsorted(times_ms, ANOVA_TMAX_MS, side="right")),
                    len(times_ms))
    log.info("ANOVA peak-search window: samples [%d, %d) = [%g, %g] ms",
             anova_start, anova_end,
             times_ms[anova_start], times_ms[anova_end - 1])

    for band_name, (l_freq, h_freq) in FREQ_BANDS.items():
        print(f"\n=== Band: {band_name} ({l_freq}-{h_freq} Hz) ===")

        data_struct = build_band_data_struct(subject_npzs, l_freq, h_freq)
        missing = set(TASKS) - set(data_struct.keys())
        if missing:
            log.warning("Band %s missing tasks %s — skipped", band_name, missing)
            continue

        for task in TASKS:
            print(f"  {task} done: shape = {data_struct[task].shape}")

        save_dir = cfg.output_dir / band_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for j in range(n_channels):
            mean_value_cols = []
            std_value_cols = []
            peak_values_per_task = []

            for task in TASKS:
                avg = data_struct[task][j, :, :]      # (n_times, n_subjects)
                if avg.ndim == 1:
                    avg = avg[:, np.newaxis]

                row_avg = avg.mean(axis=1)
                row_std = (avg.std(axis=1, ddof=1)
                           if avg.shape[1] > 1 else np.zeros_like(row_avg))

                # 峰值定位：在 [ANOVA_TMIN_MS, ANOVA_TMAX_MS] 内找 |amplitude| 极大
                sub = np.abs(row_avg[anova_start:anova_end])
                target_idx = anova_start + int(np.argmax(sub))
                peak_vals = avg[target_idx, :]         # (n_subjects,)

                mean_value_cols.append(row_avg)
                std_value_cols.append(row_std)
                peak_values_per_task.append(peak_vals)

            mean_value = np.stack(mean_value_cols, axis=1)   # (n_times, n_tasks)
            std_value = np.stack(std_value_cols, axis=1)
            peak_matrix = np.stack(peak_values_per_task, axis=0)  # (n_tasks, n_subjects)

            # ANOVA：保持与原 MATLAB anova1 默认一致（列为组）。
            # 此处列 = 被试，因此实际比较的是被试间差异，而非任务间差异。
            p_value = one_way_anova_by_columns(peak_matrix)

            # 文件名前缀 "6_"：复刻原 MATLAB 脚本中 epochs_index 残留为最后值的行为
            file_name = f"6_{ch_names[j]}.png"
            plot_channel_timecourse(
                ch_names[j], band_name, times_ms,
                mean_value, std_value, p_value,
                save_dir / file_name,
            )

        print(f"  → Saved {n_channels} plots → {save_dir}")

    print("\nAll processing complete.")


# --------------------------------------------------------------------------- #

def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
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
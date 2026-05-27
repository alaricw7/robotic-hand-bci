"""
detect_marker_and_rest.py
对单个被试做完整的 marker 位置 + Rest 时长检测。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.signal as sps

from robotic_hand_bci.preprocessing.pipeline import (
    drop_non_eeg_channels,
    get_events,
    load_config,
    read_subject_bdf_files,
    set_montage_if_available,
)
from robotic_hand_bci.project import find_project_root


def split_s1_into_blocks_and_analyze(raw, events, pause_threshold=20.0):
    """把 S1 按长暂停切分成 block, 分别分析每个 block."""
    sfreq = raw.info["sfreq"]
    event_times = events[:, 0] / sfreq
    intervals = np.diff(event_times)

    pause_indices = np.where(intervals > pause_threshold)[0]
    print(f"\nS1 长暂停位置: {len(pause_indices)} 个")

    blocks = []
    start = 0
    for p in pause_indices:
        blocks.append((start, p + 1))
        start = p + 1
    blocks.append((start, len(events)))

    print(f"\nS1 切分成 {len(blocks)} 个 block:")
    print(f"{'Block':<8} {'#trials':<10} {'period_median':<15} {'codes':<40}")
    for i, (s, e) in enumerate(blocks):
        block_events = events[s:e]
        block_times = event_times[s:e]
        if len(block_times) < 2:
            print(f"Block {i:<3} {e - s:<10} (太短)")
            continue
        block_intervals = np.diff(block_times)
        block_intervals_clean = block_intervals[block_intervals < 15]
        if len(block_intervals_clean) == 0:
            period = float("nan")
        else:
            period = np.median(block_intervals_clean)
        codes = sorted(set(block_events[:, 2].tolist()))
        codes_str = str(codes)[:38]
        print(f"Block {i:<3} {e - s:<10} {period:<15.3f} {codes_str}")

    return blocks


def analyze_marker_and_rest(raw, events, subject_name, output_dir="."):
    """
    完整分析一个被试的 marker 位置和 trial 时序结构。

    输出:
      1. inter-marker 间隔的细致分布 (直方图)
      2. marker 锁定的 ERP, 标记理论时间点
      3. ERP envelope 自动峰检测
      4. 判断 marker 位置 + 推断 Rest 时长
    """
    sfreq = raw.info["sfreq"]
    event_times = events[:, 0] / sfreq
    intervals = np.diff(event_times)

    normal_intervals = intervals[intervals < 15.0]
    if len(normal_intervals) == 0:
        raise ValueError("No inter-marker intervals below 15 s.")

    print(f"\n{'=' * 70}")
    print(f"被试 {subject_name} marker 时序分析")
    print(f"{'=' * 70}")

    print(f"\n[Step 1] Inter-marker intervals (filtered, < 15s, n={len(normal_intervals)})")
    print(f"  median  = {np.median(normal_intervals):.4f} s")
    print(f"  mean    = {np.mean(normal_intervals):.4f} s")
    print(f"  std     = {np.std(normal_intervals):.4f} s")
    print(f"  min     = {np.min(normal_intervals):.4f} s")
    print(f"  max     = {np.max(normal_intervals):.4f} s")
    print(f"  范围 (max-min) = {np.max(normal_intervals) - np.min(normal_intervals):.4f} s")

    is_tight = (np.max(normal_intervals) - np.min(normal_intervals)) < 0.5
    print(
        f"  -> 分布{'非常集中' if is_tight else '有变化'} "
        f"(范围 {'<0.5s' if is_tight else '>=0.5s'})"
    )

    period = np.median(normal_intervals)
    print(f"\n[Step 2] Trial 周期 (用 median 估计) = {period:.4f} s")
    print("\n  可能的 trial 结构 (假设 marker = MI onset):")
    print(f"    假设 Fixation=2s, MI=4s -> Rest = {period - 6:.3f} s")
    print(f"    假设 Fixation=1s, MI=4s -> Rest = {period - 5:.3f} s")
    print(f"    假设 Fixation=0s, MI=4s -> Rest = {period - 4:.3f} s")
    print("  可能的 trial 结构 (假设 marker = Fixation onset):")
    print(
        f"    假设 Fixation=2s, MI=4s, Rest=?s -> 周期 = 6 + Rest, "
        f"  Rest = {period - 6:.3f} s"
    )

    print("\n[Step 3] Marker 锁定的 ERP 分析")
    occip_chans = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO7", "PO8"]
    by_upper = {ch.upper(): ch for ch in raw.ch_names}
    picks = [by_upper[ch.upper()] for ch in occip_chans if ch.upper() in by_upper]
    if len(picks) < 3:
        picks = [by_upper[ch.upper()] for ch in ["P3", "P4", "Pz", "POz"] if ch.upper() in by_upper]
    if len(picks) == 0:
        raise ValueError("No usable occipital/parietal channels found.")
    print(f"  使用电极: {picks}")

    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=30.0, picks=picks, verbose="ERROR")

    tmin = -period - 0.5
    tmax = period + 0.5
    present_codes = set(events[:, 2].astype(int).tolist())
    event_id_present = {
        f"task_{c}": int(c)
        for c in [11, 12, 13, 14, 15, 16]
        if int(c) in present_codes
    }

    diag_epochs = mne.Epochs(
        raw_filt,
        events,
        event_id=event_id_present,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, tmin + 0.3),
        picks=picks,
        preload=True,
        verbose="ERROR",
        reject_by_annotation=False,
    )
    if len(diag_epochs) == 0:
        raise ValueError("All diagnostic epochs were dropped.")

    data = diag_epochs.get_data()
    erp = np.mean(data, axis=0).mean(axis=0)
    times = diag_epochs.times

    envelope = np.abs(erp)
    sw = max(1, int(0.05 * sfreq))
    envelope_smooth = np.convolve(envelope, np.ones(sw) / sw, mode="same")

    min_distance = int(0.3 * sfreq)
    prominence_threshold = np.std(envelope_smooth) * 0.4
    peaks, props = sps.find_peaks(
        envelope_smooth,
        distance=min_distance,
        prominence=prominence_threshold,
    )
    peak_times = times[peaks]
    peak_amps = envelope_smooth[peaks]

    if len(props["prominences"]) > 0:
        sorted_idx = np.argsort(props["prominences"])[::-1][:8]
        top_peaks = sorted(
            [
                (peak_times[i], peak_amps[i], props["prominences"][i])
                for i in sorted_idx
            ]
        )
    else:
        top_peaks = []

    print(f"\n  自动检测到的 ERP 峰 (top {len(top_peaks)}):")
    for t, amp, prom in top_peaks:
        print(f"    t = {t:+.3f} s,  amp = {amp:.3e},  prominence = {prom:.3e}")

    print("\n[Step 5] 关键时间点的 ERP 验证")
    peak_t_array = np.array([t for t, _, _ in top_peaks]) if top_peaks else np.array([])

    def has_peak_near(target_t, tolerance=0.4):
        if len(peak_t_array) == 0:
            return False, None
        diffs = np.abs(peak_t_array - target_t)
        idx = np.argmin(diffs)
        if diffs[idx] < tolerance:
            return True, peak_t_array[idx]
        return False, None

    checks = {
        "0s (marker 自身)": 0.0,
        "+4s (若 MI=4s, Rest 开始)": 4.0,
        f"+{period:.2f}s (下一 trial 周期)": period,
        "-2s (若 Fix=2s, 当前 Fix)": -2.0,
        f"+{period - 2:.2f}s (若 Fix=2s, 下个 Fix)": period - 2,
        f"-{period:.2f}s (上一 trial 周期)": -period,
    }

    erp_signature = {}
    for label, t in checks.items():
        found, actual_t = has_peak_near(t, tolerance=0.4)
        erp_signature[label] = found
        marker_str = "YES" if found else "NO"
        actual_str = f"(峰在 {actual_t:+.3f})" if found else ""
        print(f"  {marker_str:<3s}  {label:<40s}  {actual_str}")

    print("\n[Step 6] 综合判断")
    has_4s = erp_signature.get("+4s (若 MI=4s, Rest 开始)", False)
    has_0s = erp_signature.get("0s (marker 自身)", False)
    has_minus2 = erp_signature.get("-2s (若 Fix=2s, 当前 Fix)", False)

    verdict_lines = []
    if has_4s and has_minus2:
        verdict = "MI_onset (high confidence)"
        verdict_lines.append("  +4s 和 -2s 都有 ERP -> Fixation 在 marker 前 2s, Rest 在 +4s 开始")
        rest_estimate = period - 6.0
        verdict_lines.append(f"  推断: Fixation = 2s, MI = 4s, Rest = {rest_estimate:.3f} s")
    elif has_4s and not has_0s:
        verdict = "MI_onset (medium confidence)"
        verdict_lines.append("  +4s 有 ERP 但 -2s 不明显 -> 可能 Fixation 很短或 ERP 弱")
        rest_estimate = period - 6.0
        verdict_lines.append(f"  推断 Rest ≈ {rest_estimate:.3f} s (假设 Fixation = 2s)")
    elif has_0s and not has_4s:
        verdict = "Trial_onset / Fixation_onset"
        verdict_lines.append("  0s 有 ERP -> marker 是 Fixation onset")
        verdict_lines.append(f"  Trial 周期 {period:.3f}s, MI 推断在 +2s 到 +{period - 2:.2f}s")
    else:
        verdict = "UNCLEAR"
        verdict_lines.append("  ERP 信号不够清晰, 需要看图判断")

    print(f"  >>> Verdict: {verdict}")
    for line in verdict_lines:
        print(line)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    ax.hist(normal_intervals, bins=60, color="steelblue", edgecolor="white")
    ax.axvline(period, color="red", linestyle="--", label=f"median = {period:.3f}s")
    ax.set_xlabel("Inter-marker interval (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{subject_name} - Inter-marker interval distribution (excl. >15s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, erp, "b-", linewidth=1.0, label="ERP (mean)")
    ax.plot(times, envelope_smooth, "r-", linewidth=0.8, alpha=0.5, label="|ERP| smoothed")

    annotations = [
        (0, "red", "marker"),
        (4, "green", "+4s (Rest if MI_onset)"),
        (-2, "orange", "-2s (Fix if MI_onset)"),
        (period, "purple", f"+{period:.2f}s (next trial)"),
        (-period, "purple", f"-{period:.2f}s (prev trial)"),
    ]
    for t_anno, color, label in annotations:
        if tmin < t_anno < tmax:
            ax.axvline(t_anno, color=color, linestyle="--", alpha=0.6, label=label)
    for t_peak, _, _ in top_peaks:
        ax.axvline(t_peak, color="black", linestyle=":", alpha=0.3)

    ax.set_xlabel("Time relative to marker (s)")
    ax.set_ylabel("ERP amplitude (occipital mean)")
    ax.set_title(f"{subject_name} - Marker-locked ERP (1-30Hz filtered)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(times, envelope_smooth, "k-", linewidth=1.2)
    for t_peak, amp, _ in top_peaks:
        ax.plot(t_peak, amp, "r*", markersize=15)
        ax.annotate(f"{t_peak:+.2f}s", xy=(t_peak, amp), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for t_anno, color, _label in annotations:
        if tmin < t_anno < tmax:
            ax.axvline(t_anno, color=color, linestyle="--", alpha=0.4)

    ax.set_xlabel("Time relative to marker (s)")
    ax.set_ylabel("|ERP| smoothed")
    ax.set_title(f"{subject_name} - Envelope with detected peaks")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{subject_name} - Verdict: {verdict}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(output_dir) / f"{subject_name}_marker_analysis.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  图保存到: {out_path}")

    return {
        "subject": subject_name,
        "trial_period": float(period),
        "interval_std": float(np.std(normal_intervals)),
        "verdict": verdict,
        "peak_times": [float(t) for t, _, _ in top_peaks],
        "has_peak_at_+4s": has_4s,
        "has_peak_at_0s": has_0s,
        "has_peak_at_-2s": has_minus2,
        "estimated_rest_sec_if_MI_onset": float(period - 6.0),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Marker/rest timing diagnostics, with S1 block split analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional preprocessing TOML config.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["S1"],
        help="Subjects to process, e.g. S1 S3. Use 'all' for S1-S10.",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=20.0,
        help="Long-pause threshold in seconds for S1 block splitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    output_dir = find_project_root() / "artifacts" / "marker_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.subjects) == 1 and args.subjects[0].lower() == "all":
        subjects = [f"S{i}" for i in range(1, 11)]
    else:
        subjects = args.subjects

    all_results = []
    for sub in subjects:
        print(f"\n\n{'#' * 70}\n# Processing {sub}\n{'#' * 70}")
        sub_dir = Path(cfg.base_path) / sub
        try:
            raw = read_subject_bdf_files(sub_dir)
            set_montage_if_available(raw, cfg.montage_path)
            drop_non_eeg_channels(raw, cfg.drop_channels)
            events = get_events(raw, cfg)

            if sub.upper() == "S1":
                split_s1_into_blocks_and_analyze(
                    raw, events, pause_threshold=args.pause_threshold
                )

            result = analyze_marker_and_rest(raw, events, sub, output_dir)
            all_results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback

            traceback.print_exc()

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "marker_verdict.csv", index=False)
    print("\n\n汇总结果:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

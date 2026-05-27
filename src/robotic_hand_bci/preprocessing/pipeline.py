"""
Preprocess 59-channel MI EEG data for an EEGNet NoICA baseline.

Pipeline (= MATLAB pipeline 减去 ICA):
    raw BDF -> 加载 montage -> 移除 ECG/EOG -> 在 resample 之前提取事件
    -> resample 到 250 Hz (事件样点一起同步缩放)
    -> 0.5-45 Hz 带通滤波 -> CAR
    -> epoch [-0.5, 4] s, baseline [-0.5, 0] s (不做 crop,EEGNet 训练时自行裁剪)
    -> (可选) 坏段剔除  -> 保存 NPZ / FIF / classwise NPY (X 单位为 μV)

数据约定:
- 6 类 MI 事件码: 11, 12, 13, 14, 15, 16
- 输出 epoch 范围: -0.5 ~ 4.0 s, 在 250 Hz 下共 1126 个时间点
- 输出 X 单位: μV (与 MATLAB EEG.data 一致)

依赖:
    pip install mne numpy pandas

运行:
    uv run rhbci run prepare-eegnet-noica
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd

from robotic_hand_bci.project import (
    DEFAULT_MONTAGE_PATH,
    DEFAULT_PREPROCESS_CONFIG,
    PROCESSED_DIR,
    RAW_EEG_DIR,
    find_project_root,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    # Paths
    base_path: str
    save_path: str
    montage_path: Optional[str]

    # Subjects
    subjects: Tuple[str, ...] = (
    "S1", "S2", "S3", "S4", "S5",
    "S6", "S7", "S8", "S9", "S10",
    )

    # Channel handling
    drop_channels: Tuple[str, ...] = ("ECG", "HEOR", "HEOL", "VEOU", "VEOL")

    # 六类 MI 事件码
    stim_codes: Tuple[int, ...] = (11, 12, 13, 14, 15, 16)

    # True  -> marker 11-16 就是 MI cue onset (MATLAB pipeline 默认假设)
    # False -> marker 是试次/注视点 onset, 向后偏移 trial_to_mi_offset_sec 到 MI cue
    event_is_mi_cue: bool = True
    trial_to_mi_offset_sec: float = 2.0

    # Sampling and filtering (与 MATLAB pipeline 对齐)
    sfreq: float = 250.0
    l_freq: float = 0.5
    h_freq: float = 45.0

    # Epoch (与 MATLAB 完全一致, 不 crop)
    epoch_tmin: float = -0.5
    epoch_tmax: float = 4.0
    baseline: Tuple[Optional[float], Optional[float]] = (-0.5, 0.0)

    # 坏段剔除 (MATLAB pipeline 没做, 默认关闭以做严格的 "有 ICA vs 无 ICA" 对照)
    # 想开启就改成 True; 阈值已经按 EEG NoICA 经验给了一个温和的设置
    do_bad_epoch_rejection: bool = False
    abs_threshold_uv: float = 150.0
    ptp_threshold_uv: float = 250.0
    flat_std_threshold_uv: float = 0.5
    max_bad_channel_fraction: float = 0.10

    # Saving
    save_fif: bool = True
    save_npz: bool = True
    save_eeglab_set: bool = True  # 需要 `pip install eeglabio`


def default_config() -> PreprocessConfig:
    return PreprocessConfig(
        base_path=str(RAW_EEG_DIR),
        save_path=str(PROCESSED_DIR / "pythondata1"),
        montage_path=str(DEFAULT_MONTAGE_PATH),
        subjects=("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"),
    )


def _resolve_path(value: Optional[str], root: Path) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def load_config(config_path: Optional[Path] = None) -> PreprocessConfig:
    cfg = default_config()
    config_file = config_path or DEFAULT_PREPROCESS_CONFIG
    if not config_file.exists():
        return cfg

    project_root = find_project_root(config_file.resolve())
    payload = tomllib.loads(config_file.read_text(encoding="utf-8"))
    paths = payload.get("paths", {})
    subjects = payload.get("subjects", {})
    channel_handling = payload.get("channel_handling", {})
    events = payload.get("events", {})
    filtering = payload.get("filtering", {})
    epoch = payload.get("epoch", {})
    rejection = payload.get("rejection", {})
    saving = payload.get("saving", {})

    baseline = epoch.get("baseline", list(cfg.baseline))
    baseline_tuple = tuple(baseline) if baseline is not None else (None, None)

    return PreprocessConfig(
        base_path=_resolve_path(paths.get("base_path", cfg.base_path), project_root),
        save_path=_resolve_path(paths.get("save_path", cfg.save_path), project_root),
        montage_path=_resolve_path(
            paths.get("montage_path", cfg.montage_path), project_root
        ),
        subjects=tuple(subjects.get("ids", cfg.subjects)),
        drop_channels=tuple(
            channel_handling.get("drop_channels", cfg.drop_channels)
        ),
        stim_codes=tuple(events.get("stim_codes", cfg.stim_codes)),
        event_is_mi_cue=events.get("event_is_mi_cue", cfg.event_is_mi_cue),
        trial_to_mi_offset_sec=events.get(
            "trial_to_mi_offset_sec", cfg.trial_to_mi_offset_sec
        ),
        sfreq=filtering.get("sfreq", cfg.sfreq),
        l_freq=filtering.get("l_freq", cfg.l_freq),
        h_freq=filtering.get("h_freq", cfg.h_freq),
        epoch_tmin=epoch.get("tmin", cfg.epoch_tmin),
        epoch_tmax=epoch.get("tmax", cfg.epoch_tmax),
        baseline=baseline_tuple,
        do_bad_epoch_rejection=rejection.get(
            "enabled", cfg.do_bad_epoch_rejection
        ),
        abs_threshold_uv=rejection.get("abs_threshold_uv", cfg.abs_threshold_uv),
        ptp_threshold_uv=rejection.get("ptp_threshold_uv", cfg.ptp_threshold_uv),
        flat_std_threshold_uv=rejection.get(
            "flat_std_threshold_uv", cfg.flat_std_threshold_uv
        ),
        max_bad_channel_fraction=rejection.get(
            "max_bad_channel_fraction", cfg.max_bad_channel_fraction
        ),
        save_fif=saving.get("save_fif", cfg.save_fif),
        save_npz=saving.get("save_npz", cfg.save_npz),
        save_eeglab_set=saving.get("save_eeglab_set", cfg.save_eeglab_set),
    )


CONFIG = load_config()


# ----------------------------------------------------------------------
# Raw I/O
# ----------------------------------------------------------------------


def read_subject_bdf_files(subject_dir: Path) -> mne.io.BaseRaw:
    """读取被试目录下的 BDF 文件.

    支持两种格式:
    1. Neuracle 格式: ``data.bdf`` (EEG) + ``evt.bdf`` (事件标记)
       -> 读 data.bdf 作为 raw, 从 evt.bdf 提取事件注入到 raw.annotations
    2. 普通格式: 一个或多个同结构 EEG BDF 文件
       -> 按文件名排序后用 mne.concatenate_raws 拼接
    """
    bdf_files = sorted(subject_dir.glob("*.bdf"))
    if len(bdf_files) == 0:
        raise FileNotFoundError(f"No .bdf files found in {subject_dir}")

    name_to_file = {f.name.lower(): f for f in bdf_files}
    if "data.bdf" in name_to_file and "evt.bdf" in name_to_file:
        return _read_neuracle_bdf(name_to_file["data.bdf"], name_to_file["evt.bdf"])

    raws = []
    for bdf_file in bdf_files:
        print(f"Reading: {bdf_file}")
        raw = mne.io.read_raw_bdf(str(bdf_file), preload=True, verbose="ERROR")
        raws.append(raw)
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws, preload=True, verbose="ERROR")


def _read_neuracle_bdf(data_file: Path, evt_file: Path) -> mne.io.BaseRaw:
    """Neuracle 格式: 把 evt.bdf 的事件作为 annotations 注入 data.bdf 的 raw.

    重要: MNE 的 read_raw_bdf 会把超出 evt.bdf 数据通道时长的 annotations
    丢掉 (典型表现: evt.bdf 只有 641 个数据样点 = 641s,但 BDF annotation
    区里有 ~641 个事件 onset 覆盖到 5698s, MNE 只保留 onset<=641s 的部分).
    因此这里优先用 pyedflib 直接读底层 BDF annotations.
    """
    print(f"Detected Neuracle format:")
    print(f"  Data: {data_file}")
    print(f"  Evt : {evt_file}")

    raw = mne.io.read_raw_bdf(str(data_file), preload=True, verbose="ERROR")
    print(
        f"  data.bdf: {len(raw.ch_names)} channels, {raw.n_times} samples, "
        f"sfreq={raw.info['sfreq']:.1f} Hz, duration={raw.times[-1]:.1f} s"
    )

    annotations = _read_full_annotations_from_evt(evt_file, raw)
    if annotations is None or len(annotations) == 0:
        # 回退到 MNE 默认解析 (会被截断,但至少不空)
        print("  ⚠️  pyedflib 不可用或失败,回退到 MNE 默认解析 (可能丢事件)")
        raw_evt = mne.io.read_raw_bdf(str(evt_file), preload=True, verbose="ERROR")
        annotations = raw_evt.annotations

    if annotations is None or len(annotations) == 0:
        print("  ⚠️  WARNING: 未能从 evt.bdf 提取任何事件 (后面 epoch 会失败)")
    else:
        print(
            f"  从 evt.bdf 提取到 {len(annotations)} 个事件 -> 写入 raw.annotations"
        )
        for i in range(min(5, len(annotations))):
            print(
                f"    [{i}] onset={annotations.onset[i]:.3f}s  "
                f"desc='{annotations.description[i]}'"
            )
        # 时间范围校验: 提示数据文件 duration 是否覆盖事件
        if len(annotations) > 0:
            max_onset = float(np.max(annotations.onset))
            raw_dur = raw.times[-1]
            if max_onset > raw_dur:
                print(
                    f"  ⚠️  最晚事件 onset={max_onset:.1f}s 超过 data.bdf "
                    f"时长 {raw_dur:.1f}s, 这些事件 epoch 时会被自动丢弃"
                )
        raw.set_annotations(annotations)

    return raw


def _read_full_annotations_from_evt(
    evt_file: Path, raw_data: mne.io.BaseRaw
) -> Optional[mne.Annotations]:
    """用 pyedflib 读 evt.bdf 的全部 BDF annotations (不被截断)."""
    try:
        import pyedflib  # type: ignore
    except ImportError:
        print("  pyedflib 未安装, 无法读取完整事件. 建议: pip install pyedflib")
        return None

    try:
        f = pyedflib.EdfReader(str(evt_file))
        onsets, durations, descriptions = f.readAnnotations()
        f.close()
    except Exception as exc:
        print(f"  pyedflib 读取失败: {exc}")
        return None

    if len(onsets) == 0:
        return None

    print(
        f"  pyedflib 读到完整 annotations: {len(onsets)} 个, "
        f"onset 范围 {float(np.min(onsets)):.1f} ~ {float(np.max(onsets)):.1f} s"
    )

    return mne.Annotations(
        onset=np.asarray(onsets, dtype=float),
        duration=np.asarray(durations, dtype=float),
        description=[str(d) for d in descriptions],
        orig_time=raw_data.info.get("meas_date"),
    )


def _extract_annotations_from_evt_bdf(
    raw_evt: mne.io.BaseRaw, raw_data: mne.io.BaseRaw
) -> Optional[mne.Annotations]:
    """从 Neuracle evt.bdf 提取事件 -> mne.Annotations.

    Neuracle 的 evt.bdf 通常把事件码存在某个通道里:
    事件发生的样点写入触发码值, 其它样点为 0.
    优先尝试 evt.bdf 自带的 annotations, 否则扫描通道数据找非零阶跃.
    """
    # 路径 1: evt.bdf 自带 annotations
    if raw_evt.annotations is not None and len(raw_evt.annotations) > 0:
        print(f"  evt.bdf 含 {len(raw_evt.annotations)} 个 annotations, 直接复用")
        return mne.Annotations(
            onset=np.asarray(raw_evt.annotations.onset, dtype=float),
            duration=np.asarray(raw_evt.annotations.duration, dtype=float),
            description=list(raw_evt.annotations.description),
            orig_time=raw_data.info.get("meas_date"),
        )

    # 路径 2: 扫描通道数据
    print("  evt.bdf 无 annotations, 扫描通道数据中的非零事件值...")
    evt_data = raw_evt.get_data()  # n_channels x n_times, 单位 V
    sfreq_evt = raw_evt.info["sfreq"]

    # Neuracle 通常用整数事件码, 但 read_raw_bdf 会按物理单位 (V) 还原
    # 这里我们对每个通道单独做缩放探测: 找一个使非零值变成整数的因子
    for ch_idx in range(evt_data.shape[0]):
        ch_data = evt_data[ch_idx]
        if not np.any(ch_data != 0):
            continue

        # 先尝试直接 round
        candidates = [
            np.round(ch_data).astype(np.int64),
            np.round(ch_data * 1e6).astype(np.int64),  # μV scale
            np.round(ch_data * 1e3).astype(np.int64),  # mV scale
        ]

        for ch_int in candidates:
            nonzero_mask = ch_int != 0
            if not nonzero_mask.any():
                continue
            prev = np.concatenate([[False], nonzero_mask[:-1]])
            rising = np.where(nonzero_mask & ~prev)[0]
            if len(rising) == 0:
                continue

            codes_seen = ch_int[rising]
            # 合理性检查: Neuracle 事件码一般是小整数 (<= 几百)
            if np.max(np.abs(codes_seen)) > 1e5:
                continue

            print(
                f"    通道 [{ch_idx}] '{raw_evt.ch_names[ch_idx]}': "
                f"找到 {len(rising)} 个事件, 唯一码 = "
                f"{sorted(set(codes_seen.tolist()))}"
            )

            onsets = (rising / sfreq_evt).tolist()
            descriptions = [str(int(c)) for c in codes_seen]
            return mne.Annotations(
                onset=onsets,
                duration=[0.0] * len(onsets),
                description=descriptions,
                orig_time=raw_data.info.get("meas_date"),
            )

    return None


def set_montage_if_available(raw: mne.io.BaseRaw, montage_path: Optional[str]) -> None:
    """尝试加载电极位置文件; 找不到就跳过."""
    if not montage_path:
        return
    montage_file = Path(montage_path)
    if not montage_file.exists():
        print(f"Montage file not found, skipping: {montage_file}")
        return
    try:
        montage = mne.channels.read_custom_montage(str(montage_file))
        raw.set_montage(montage, match_case=False, on_missing="ignore")
        print(f"Montage loaded: {montage_file}")
    except Exception as exc:
        print(f"Could not load montage from {montage_file}: {exc}")


def drop_non_eeg_channels(raw: mne.io.BaseRaw, drop_channels: Sequence[str]) -> None:
    """删除 ECG/EOG 等非 EEG 通道, 并把剩下的通道标为 EEG / stim."""
    existing = [ch for ch in drop_channels if ch in raw.ch_names]
    if existing:
        print(f"Dropping non-EEG channels: {existing}")
        raw.drop_channels(existing)
    else:
        print("No configured non-EEG channels found to drop.")

    mapping: Dict[str, str] = {}
    for ch in raw.ch_names:
        if ch.lower() in {"status", "stim", "trigger", "sti 014"}:
            mapping[ch] = "stim"
        else:
            mapping[ch] = "eeg"
    raw.set_channel_types(mapping, on_unit_change="ignore")


# ----------------------------------------------------------------------
# Event extraction (必须在 resample 之前调用)
# ----------------------------------------------------------------------


def extract_events_from_annotations(
    raw: mne.io.BaseRaw, stim_codes: Iterable[int]
) -> Optional[np.ndarray]:
    """优先用 mne.events_from_annotations: 它会正确处理 first_samp."""
    if raw.annotations is None or len(raw.annotations) == 0:
        return None

    stim_codes_set = {int(c) for c in stim_codes}

    # 扫描所有 description, 找出含有目标 stim code 数字的并建立映射
    desc_to_code: Dict[str, int] = {}
    for desc in set(raw.annotations.description):
        numbers = re.findall(r"\d+", str(desc))
        for num in numbers:
            code = int(num)
            if code in stim_codes_set:
                desc_to_code[str(desc)] = code
                break

    if not desc_to_code:
        return None

    try:
        events, _ = mne.events_from_annotations(
            raw, event_id=desc_to_code, verbose="ERROR"
        )
    except Exception as exc:
        print(f"events_from_annotations failed: {exc}")
        return None

    if len(events) == 0:
        return None

    print(f"Found {len(events)} events from annotations.")
    return events.astype(int)


def extract_events_from_stim_channel(
    raw: mne.io.BaseRaw, stim_codes: Iterable[int]
) -> Optional[np.ndarray]:
    """回退路径: 从 Status / stim 通道找事件 (带低字节掩码 fallback)."""
    stim_codes_set = {int(c) for c in stim_codes}
    candidate_names = ["Status", "STATUS", "STI 014", "Trigger", "TRIGGER", "stim", "Stim"]
    stim_channels = [ch for ch in candidate_names if ch in raw.ch_names]
    if not stim_channels:
        stim_channels = [
            raw.ch_names[i]
            for i, typ in enumerate(raw.get_channel_types())
            if typ == "stim"
        ]
    if not stim_channels:
        return None

    for stim_ch in stim_channels:
        print(f"Trying stim channel: {stim_ch}")
        try:
            events = mne.find_events(
                raw, stim_channel=stim_ch, shortest_event=1, verbose="ERROR"
            )
        except Exception as exc:
            print(f"Could not read stim channel {stim_ch}: {exc}")
            continue
        if len(events) == 0:
            continue

        direct = events[np.isin(events[:, 2], list(stim_codes_set))]
        if len(direct) > 0:
            print(f"Found {len(direct)} direct trigger events from {stim_ch}.")
            return direct.astype(int)

        masked = events.copy()
        masked[:, 2] = masked[:, 2] & 255
        masked = masked[np.isin(masked[:, 2], list(stim_codes_set))]
        if len(masked) > 0:
            print(f"Found {len(masked)} low-byte masked trigger events from {stim_ch}.")
            return masked.astype(int)
    return None


def get_events(raw: mne.io.BaseRaw, cfg: PreprocessConfig) -> np.ndarray:
    """统一入口: annotations 优先, stim 通道兜底, 必要时偏移到 MI cue."""
    events = extract_events_from_annotations(raw, cfg.stim_codes)
    if events is None:
        events = extract_events_from_stim_channel(raw, cfg.stim_codes)
    if events is None or len(events) == 0:
        raise RuntimeError(
            "无法找到事件码 11-16. 请确认触发存储在 annotations 还是 Status/stim 通道."
        )

    if not cfg.event_is_mi_cue:
        shift = int(round(cfg.trial_to_mi_offset_sec * raw.info["sfreq"]))
        events = events.copy()
        events[:, 0] += shift
        print(f"Shifted events by +{cfg.trial_to_mi_offset_sec:.3f} s to MI cue onset.")

    return events


def print_event_summary(events: np.ndarray, label: str) -> None:
    """打印每类事件数, 方便与 MATLAB 输出做核对."""
    print(f"\n=== Event counts [{label}] ===")
    unique, counts = np.unique(events[:, 2], return_counts=True)
    for code, n in zip(unique.tolist(), counts.tolist()):
        print(f"  code {code:>3}: {n:>4} trials")
    print(f"  total   : {len(events):>4} events\n")


# ----------------------------------------------------------------------
# Epoching
# ----------------------------------------------------------------------


def make_epochs(
    raw: mne.io.BaseRaw, events: np.ndarray, cfg: PreprocessConfig
) -> mne.Epochs:
    """根据事件切 epoch, 同时做 baseline correction."""
    event_id = {f"task_{code}": int(code) for code in cfg.stim_codes}
    # 只保留 events 中实际出现的 code, 免得 MNE 报警
    present_codes = set(events[:, 2].tolist())
    event_id = {k: v for k, v in event_id.items() if v in present_codes}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=cfg.epoch_tmin,
        tmax=cfg.epoch_tmax,
        baseline=cfg.baseline,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        detrend=None,
        verbose="ERROR",
    )

    print(
        f"Epochs created: {len(epochs)} trials, "
        f"{len(epochs.ch_names)} channels, {len(epochs.times)} time points, "
        f"time range {epochs.tmin:.3f} ~ {epochs.tmax:.3f} s"
    )
    return epochs


def _detect_microvolt_scale(data: np.ndarray) -> Tuple[float, str]:
    """探测 EEG 数据当前单位, 返回 (转 μV 的乘子, 单位描述).

    EEG 信号典型 std 在 5-50 μV. 用 channel-wise std 的中位数判断:
    - std < 1e-3   -> 单位是 V (典型 std ~5e-6 到 5e-5)
    - 1e-3 ~ 1e3   -> 单位已经是 μV
    - >= 1e3       -> 未知, 数据异常大
    """
    median_std = float(np.median(np.std(data, axis=-1)))
    if median_std < 1e-3:
        return 1e6, f"V (median channel std = {median_std:.3e} V)"
    elif median_std < 1e3:
        return 1.0, f"μV (median channel std = {median_std:.3f} μV)"
    else:
        return 1.0, f"UNKNOWN, 异常大 (median channel std = {median_std:.3e})"


def reject_bad_epochs(
    epochs: mne.Epochs, cfg: PreprocessConfig
) -> Tuple[mne.Epochs, pd.DataFrame]:
    """按"坏通道占比"判定整 epoch (而非任一通道超阈值)."""
    raw_data = epochs.get_data()
    scale, unit_str = _detect_microvolt_scale(raw_data)
    print(f"  数据单位探测: {unit_str}  ->  缩放因子 = {scale}")
    data_uv = raw_data * scale  # 统一到 μV

    abs_bad = np.max(np.abs(data_uv), axis=2) > cfg.abs_threshold_uv
    ptp_bad = np.ptp(data_uv, axis=2) > cfg.ptp_threshold_uv
    flat_bad = np.std(data_uv, axis=2) < cfg.flat_std_threshold_uv

    bad_ch = abs_bad | ptp_bad | flat_bad
    bad_frac = bad_ch.mean(axis=1)
    reject_mask = bad_frac > cfg.max_bad_channel_fraction

    log = pd.DataFrame(
        {
            "epoch_index": np.arange(len(epochs)),
            "event_code": epochs.events[:, 2],
            "bad_channel_fraction": bad_frac,
            "n_abs_bad_channels": abs_bad.sum(axis=1),
            "n_ptp_bad_channels": ptp_bad.sum(axis=1),
            "n_flat_bad_channels": flat_bad.sum(axis=1),
            "reject": reject_mask,
        }
    )

    keep_indices = np.where(~reject_mask)[0]
    cleaned = epochs[keep_indices]

    print(
        f"Bad epoch rejection: rejected {reject_mask.sum()} / {len(epochs)} trials "
        f"({100 * reject_mask.mean():.2f}%)."
    )
    return cleaned, log


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def make_X_y(epochs: mne.Epochs, cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray]:
    """返回 X: trials x channels x time (单位 μV), y: 0-5 标签."""
    raw_data = epochs.get_data()
    scale, unit_str = _detect_microvolt_scale(raw_data)
    print(f"X 单位探测: {unit_str}  ->  缩放因子 = {scale}")
    X = (raw_data * scale).astype(np.float32)
    code_to_label = {int(code): idx for idx, code in enumerate(cfg.stim_codes)}
    y = np.asarray(
        [code_to_label[int(code)] for code in epochs.events[:, 2]], dtype=np.int64
    )
    return X, y


def save_outputs(
    subject: str,
    epochs: mne.Epochs,
    rejection_log: Optional[pd.DataFrame],
    cfg: PreprocessConfig,
) -> None:
    out_dir = Path(cfg.save_path) / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_X_y(epochs, cfg)

    if cfg.save_npz:
        npz_path = out_dir / f"{subject}_EEGNet_NoICA_uV.npz"
        np.savez_compressed(
            npz_path,
            X=X,  # trials x channels x time, 单位 μV
            y=y,  # 0..5
            ch_names=np.asarray(epochs.ch_names, dtype=object),
            sfreq=float(epochs.info["sfreq"]),
            times=epochs.times.astype(np.float64),  # 时间轴, 方便下游 crop
            event_codes=epochs.events[:, 2].astype(np.int64),
            stim_codes=np.asarray(cfg.stim_codes, dtype=np.int64),
            unit="uV",
            tmin=float(epochs.tmin),
            tmax=float(epochs.tmax),
            config=json.dumps(asdict(cfg), ensure_ascii=False),
        )
        print(f"Saved NPZ: {npz_path}   (X 单位 μV, shape={X.shape})")

        # 同时按类别保存 .npy, 与你之前 EEGNet 加载器兼容
        class_dir = out_dir / "classwise"
        class_dir.mkdir(exist_ok=True)
        for label_idx, code in enumerate(cfg.stim_codes):
            X_class = X[y == label_idx]
            np.save(
                class_dir / f"{subject}_EEG_{label_idx + 1}_code{code}.npy", X_class
            )

    if cfg.save_fif:
        fif_path = out_dir / f"{subject}_noica-epo.fif"
        epochs.save(fif_path, overwrite=True, verbose="ERROR")
        print(f"Saved FIF: {fif_path}   (FIF 内部仍是 V, 是 MNE 标准)")

    if cfg.save_eeglab_set:
        try:
            import eeglabio  # 仅做依赖检查
        except ImportError:
            print("⚠️ 未安装 eeglabio, 跳过 .set 导出。  pip install eeglabio")
        else:
            # 按类别分别保存 .set, 命名与 MATLAB 完全一致: S1_EEG_1.set ~ S1_EEG_6.set
            for label_idx, code in enumerate(cfg.stim_codes):
                name = f"task_{code}"
                if name not in epochs.event_id:
                    print(f"  跳过 code {code}: 0 trials")
                    continue
                ep_class = epochs[name]
                set_path = out_dir / f"{subject}_EEG_{label_idx + 1}.set"
                try:
                    ep_class.export(set_path, fmt="eeglab", overwrite=True)
                    print(
                        f"Saved EEGLAB SET: {set_path.name}  "
                        f"(code {code}, {len(ep_class)} trials)"
                    )
                except Exception as exc:
                    print(f"  导出 {set_path.name} 失败: {exc}")

    if rejection_log is not None:
        log_path = out_dir / f"{subject}_bad_epoch_rejection_log.csv"
        rejection_log.to_csv(log_path, index=False)
        print(f"Saved rejection log: {log_path}")

    counts = pd.DataFrame(
        {
            "label_0_to_5": np.arange(len(cfg.stim_codes)),
            "event_code": list(cfg.stim_codes),
            "n_trials_kept": [int(np.sum(y == i)) for i in range(len(cfg.stim_codes))],
        }
    )
    counts_path = out_dir / f"{subject}_class_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"\nSaved class counts: {counts_path}")
    print(counts.to_string(index=False))
    print(f"\nFinal X shape: {X.shape}   (trials x channels x time, μV)")
    print(f"Final y shape: {y.shape}")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def preprocess_subject(subject: str, cfg: PreprocessConfig) -> None:
    subject_dir = Path(cfg.base_path) / subject
    print("=" * 80)
    print(f"Preprocessing subject: {subject}")
    print(f"Subject folder: {subject_dir}")

    raw = read_subject_bdf_files(subject_dir)

    # montage 放在 drop 通道之前, 因为 montage 文件可能含 EOG/ECG 名
    set_montage_if_available(raw, cfg.montage_path)
    drop_non_eeg_channels(raw, cfg.drop_channels)

    # === 关键: 在 resample 之前提取事件, 避免 Status 通道被插值破坏 ===
    print("\nExtracting events BEFORE resample (at original sampling rate)...")
    events = get_events(raw, cfg)
    print_event_summary(events, "raw / before resample")

    # === Resample + 同步缩放事件样点索引 ===
    if abs(raw.info["sfreq"] - cfg.sfreq) > 1e-6:
        orig_sfreq = raw.info["sfreq"]
        print(
            f"Resampling from {orig_sfreq:.3f} Hz to {cfg.sfreq:.3f} Hz "
            f"(events 一起同步)"
        )
        raw, events = raw.resample(
            cfg.sfreq, events=events, npad="auto", verbose="ERROR"
        )
        print_event_summary(events, "after resample")

    # === 带通滤波 (与 MATLAB 一致: 0.5-45 Hz) ===
    print(f"Band-pass filter: {cfg.l_freq}-{cfg.h_freq} Hz")
    raw.filter(
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )

    # === CAR ===
    print("Applying common average reference (CAR)")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    # === Epoch [-0.5, 4] s + baseline [-0.5, 0] s (不 crop) ===
    epochs = make_epochs(raw, events, cfg)
    print_event_summary(epochs.events, "epoch (kept)")

    # === 数据诊断: 打印数据范围 / 单位, 方便对照 MATLAB ===
    _data_for_diag = epochs.get_data()
    _scale, _unit = _detect_microvolt_scale(_data_for_diag)
    print(
        f"\n[Data diagnostics]\n"
        f"  shape       = {_data_for_diag.shape}\n"
        f"  min / max   = {_data_for_diag.min():.4e}  /  {_data_for_diag.max():.4e}\n"
        f"  global std  = {_data_for_diag.std():.4e}\n"
        f"  detected    = {_unit}\n"
        f"  -> 输出会用 scale = {_scale} 统一到 μV\n"
    )

    # === 可选: 坏段剔除 ===
    if cfg.do_bad_epoch_rejection:
        print("Bad-epoch rejection: ENABLED")
        epochs_final, rejection_log = reject_bad_epochs(epochs, cfg)
        print_event_summary(epochs_final.events, "after bad-epoch rejection")
    else:
        print("Bad-epoch rejection: DISABLED (与 MATLAB pipeline 对齐)")
        epochs_final = epochs
        rejection_log = None

    save_outputs(subject, epochs_final, rejection_log, cfg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess raw EEG BDF files into pythondata1-style outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PREPROCESS_CONFIG,
        help="Path to preprocessing TOML config.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subset of subjects, e.g. S1 S3.",
    )
    parser.add_argument(
        "--trial-onset-markers",
        action="store_true",
        help="Treat event markers as trial/fixation onset, then shift to MI cue.",
    )
    parser.add_argument(
        "--enable-bad-epoch-rejection",
        action="store_true",
        help="Enable bad epoch rejection regardless of config file.",
    )
    parser.add_argument(
        "--disable-eeglab-set",
        action="store_true",
        help="Skip EEGLAB .set export for this run.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    mne.set_log_level("WARNING")

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.subjects:
        cfg = replace(cfg, subjects=tuple(args.subjects))
    if args.trial_onset_markers:
        cfg = replace(cfg, event_is_mi_cue=False)
    if args.enable_bad_epoch_rejection:
        cfg = replace(cfg, do_bad_epoch_rejection=True)
    if args.disable_eeglab_set:
        cfg = replace(cfg, save_eeglab_set=False)

    for sub in cfg.subjects:
        preprocess_subject(sub, cfg)


if __name__ == "__main__":
    main()

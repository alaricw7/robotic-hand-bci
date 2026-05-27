"""
data.py - 数据加载、预处理、subject-dependent 交叉验证划分。
"""

import os
import re
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


def bandpass_filter(X, low, high, fs, order=4):
    """Butterworth 带通滤波。X: (n_trials, n_channels, n_samples)。"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1).astype(np.float32)


def fit_channel_standardizer(X_train):
    """只在训练集上计算逐通道标准化参数。"""
    mean = X_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_channel_standardize(X, mean, std):
    """使用训练集 mean/std 标准化任意 split。"""
    return ((X - mean) / std).astype(np.float32)


class MIDataset(Dataset):
    def __init__(self, X, y, device=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _match_subject_samples(X, cfg, source_path, epochs=None):
    """
    把读取到的 epoch 长度对齐到 cfg.n_samples。
    长则保留最后 n_samples；只短 1 个点时复制最后一个点补齐。
    """
    n_actual = X.shape[2]
    n_target = cfg.n_samples

    if n_actual == n_target:
        return X

    if n_actual > n_target:
        extra = n_actual - n_target
        print(
            f"[INFO] {source_path} has {n_actual} samples, "
            f"cropping first {extra} samples -> keep last {n_target}."
        )
        return X[..., -n_target:]

    if n_target - n_actual == 1:
        print(f"[INFO] {source_path} has {n_actual} samples, padding last sample -> {n_target}.")
        return np.pad(X, ((0, 0), (0, 0), (0, 1)), mode="edge").astype(np.float32)

    raise ValueError(
        f"{source_path} has only {n_actual} samples, but cfg.n_samples={n_target}. "
        "Cannot pad/crop safely."
    )


def _validate_subject_shape(X, cfg, source_path):
    if X.ndim != 3:
        raise ValueError(f"{source_path} is not a 3D array shaped as (trials, channels, samples).")
    if X.shape[1] != cfg.n_channels:
        raise ValueError(f"{source_path} has {X.shape[1]} channels, but cfg.n_channels={cfg.n_channels}.")
    if X.shape[2] != cfg.n_samples:
        raise ValueError(f"{source_path} has {X.shape[2]} samples, but cfg.n_samples={cfg.n_samples}.")


def _find_preprocessing_npz_file(subject_id, data_root, cfg):
    root = Path(data_root)
    subject_name = f"S{subject_id}"
    variant = getattr(cfg, "npz_variant", "NoICA")
    subject_dir = root / subject_name

    candidates = [
        subject_dir / f"{subject_name}_EEGNet_{variant}_uV.npz",
        subject_dir / f"{subject_name}_EEGNet_NoICA_uV.npz",
    ]
    for path in candidates:
        if path.exists():
            return path

    if subject_dir.exists():
        matches = sorted(subject_dir.glob(f"{subject_name}_EEGNet_*_uV.npz"))
        if matches:
            return matches[0]

    return None


def _load_preprocessing_npz(npz_path, cfg):
    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError(f"{npz_path} must contain X and y arrays.")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if X.ndim != 3:
        raise ValueError(f"{npz_path} X must be (trials, channels, samples), got {X.shape}.")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"{npz_path} y shape {y.shape} does not match X trials {X.shape[0]}.")

    print(f"[INFO] {npz_path}: shape={X.shape}, y={y.shape}")
    X = _match_subject_samples(X, cfg, npz_path)
    _validate_subject_shape(X, cfg, npz_path)
    return X, y


def _resolve_fdt_path(set_path):
    sibling_fdt = set_path.with_suffix(".fdt")
    if sibling_fdt.exists():
        return sibling_fdt
    upper_fdt = set_path.with_suffix(".FDT")
    if upper_fdt.exists():
        return upper_fdt
    return None


def _load_eeglab_mat_epochs(set_path, cfg, label_override=None):
    mat = loadmat(set_path, squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise ValueError(f"{set_path} does not contain an EEGLAB data matrix.")
    if isinstance(mat["data"], str):
        raise ValueError(f"{set_path} points to external data file {mat['data']}.")

    X = np.asarray(mat["data"], dtype=np.float32)
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    if X.ndim != 3:
        raise ValueError(f"{set_path} data is not 3D: got shape {X.shape}.")

    if X.shape[0] == cfg.n_channels and X.shape[1] == cfg.n_channels:
        raise ValueError(
            f"{set_path} has ambiguous shape {X.shape}: "
            f"both dim 0 and dim 1 equal n_channels={cfg.n_channels}. "
            "Cannot tell channels-first from trials-first. "
            "Please reshape this file explicitly before loading."
        )

    # EEGLAB stores epoched data as (channels, samples, trials).
    if X.shape[0] == cfg.n_channels:
        X = np.transpose(X, (2, 0, 1))
    elif X.shape[1] == cfg.n_channels:
        pass
    else:
        raise ValueError(f"{set_path} data shape {X.shape} does not match cfg.n_channels={cfg.n_channels}.")

    sfreq = float(np.asarray(mat.get("srate", cfg.sample_rate)).squeeze())
    print(f"[INFO] {set_path}: sfreq={sfreq}, shape={X.shape}")

    X = _match_subject_samples(X, cfg, set_path)
    _validate_subject_shape(X, cfg, set_path)

    if label_override is None:
        raise ValueError(f"{set_path} needs event labels when label_override is not provided.")
    y = np.full(X.shape[0], label_override, dtype=np.int64)
    return X, y


def _load_eeglab_epochs(set_path, cfg, label_override=None):
    try:
        return _load_eeglab_mat_epochs(set_path, cfg, label_override=label_override)
    except ValueError as mat_exc:
        print(f"[INFO] Falling back to MNE for {set_path}: {mat_exc}")

    try:
        import mne
    except ImportError as exc:
        raise ImportError("Reading .set/.fdt requires the mne package to be installed.") from exc

    _resolve_fdt_path(set_path)
    epochs = mne.read_epochs_eeglab(str(set_path), verbose="ERROR")
    epochs.pick("eeg")
    X = epochs.get_data().astype(np.float32)
    print(f"[INFO] {set_path}: sfreq={epochs.info['sfreq']}, shape={X.shape}")

    X = _match_subject_samples(X, cfg, set_path, epochs)
    _validate_subject_shape(X, cfg, set_path)

    if label_override is not None:
        y = np.full(X.shape[0], label_override, dtype=np.int64)
    elif epochs.events is not None and len(epochs.events) == X.shape[0]:
        event_ids = epochs.events[:, -1]
        unique_ids = np.unique(event_ids)
        id_to_label = {event_id: idx for idx, event_id in enumerate(unique_ids)}
        y = np.array([id_to_label[event_id] for event_id in event_ids], dtype=np.int64)
    else:
        raise ValueError(f"{set_path} does not contain usable event labels.")

    print(f"[INFO] {set_path}: sfreq={epochs.info['sfreq']}, shape={X.shape}")
    return X, y


def _load_mne_epoch_file(epoch_path, cfg, label_override=None):
    try:
        import mne
    except ImportError as exc:
        raise ImportError("Reading .fif epochs requires the mne package to be installed.") from exc

    epochs = mne.read_epochs(str(epoch_path), preload=True, verbose="ERROR")
    epochs.pick("eeg")
    X = epochs.get_data().astype(np.float32)
    print(f"[INFO] {epoch_path}: sfreq={epochs.info['sfreq']}, shape={X.shape}")

    X = _match_subject_samples(X, cfg, epoch_path, epochs)
    _validate_subject_shape(X, cfg, epoch_path)

    if label_override is not None:
        y = np.full(X.shape[0], label_override, dtype=np.int64)
    elif epochs.events is not None and len(epochs.events) == X.shape[0]:
        event_ids = epochs.events[:, -1]
        unique_ids = np.unique(event_ids)
        id_to_label = {event_id: idx for idx, event_id in enumerate(unique_ids)}
        y = np.array([id_to_label[event_id] for event_id in event_ids], dtype=np.int64)
    else:
        raise ValueError(f"{epoch_path} does not contain usable event labels.")
    return X, y


def _find_subject_fif_epoch_files(subject_id, data_root):
    root = Path(data_root)
    class_files = []
    for epoch_path in sorted(root.rglob("*-epo.fif")):
        match = re.match(
            r"^s(?P<subject>\d+)_eeg_(?P<class>\d+)(?:_(?:bl|nobl))?-epo\.fif$",
            epoch_path.name.lower(),
        )
        if match is None:
            continue
        if int(match.group("subject")) != subject_id:
            continue
        class_files.append((int(match.group("class")), epoch_path))
    if class_files:
        return sorted(class_files, key=lambda item: item[0])
    return None


def _find_subject_set_files(subject_id, data_root):
    root = Path(data_root)
    single_file_candidates = [
        root / f"subject_{subject_id:02d}.set",
        root / f"subject_{subject_id}.set",
    ]
    for set_path in single_file_candidates:
        if set_path.exists():
            return {"mode": "single", "files": [set_path]}

    class_files = []
    for set_path in sorted(root.rglob("*.set")):
        name_parts = set_path.stem.lower().split("_")
        if len(name_parts) != 3:
            continue
        subject_token, eeg_token, class_suffix = name_parts
        if eeg_token != "eeg" or not subject_token.startswith("s"):
            continue
        if not subject_token[1:].isdigit() or int(subject_token[1:]) != subject_id:
            continue
        if not class_suffix.isdigit():
            continue
        class_files.append((int(class_suffix), set_path))

    if class_files:
        return {"mode": "per_class", "files": class_files}
    return None


def discover_subject_ids(data_root):
    """从 data_root 中自动发现可加载的 subject id。"""
    root = Path(data_root)
    subject_ids = set()

    for npz_path in root.glob("S*/S*_EEGNet_*_uV.npz"):
        subject_dir = npz_path.parent.name
        if subject_dir.lower().startswith("s") and subject_dir[1:].isdigit():
            subject_ids.add(int(subject_dir[1:]))

    for npz_path in root.glob("subject_*.npz"):
        match = re.match(r"^subject_(?P<subject>\d+)\.npz$", npz_path.name.lower())
        if match is not None:
            subject_ids.add(int(match.group("subject")))

    for set_path in root.rglob("*.set"):
        stem = set_path.stem.lower()
        match = re.match(r"^subject_(?P<subject>\d+)$", stem)
        if match is not None:
            subject_ids.add(int(match.group("subject")))
            continue

        name_parts = stem.split("_")
        if len(name_parts) != 3:
            continue
        subject_token, eeg_token, class_suffix = name_parts
        if (
            eeg_token == "eeg"
            and subject_token.startswith("s")
            and subject_token[1:].isdigit()
            and class_suffix.isdigit()
        ):
            subject_ids.add(int(subject_token[1:]))

    for epoch_path in root.rglob("*-epo.fif"):
        match = re.match(
            r"^s(?P<subject>\d+)_eeg_(?P<class>\d+)(?:_(?:bl|nobl))?-epo\.fif$",
            epoch_path.name.lower(),
        )
        if match is not None:
            subject_ids.add(int(match.group("subject")))

    return sorted(subject_ids)


def load_subject(subject_id, data_root, cfg):
    """加载一个受试者的数据，做预处理，返回 X, y。"""
    preprocessing_npz = _find_preprocessing_npz_file(subject_id, data_root, cfg)
    fif_epoch_files = _find_subject_fif_epoch_files(subject_id, data_root)
    eeglab_files = _find_subject_set_files(subject_id, data_root)
    if preprocessing_npz is not None:
        X, y = _load_preprocessing_npz(preprocessing_npz, cfg)
    elif fif_epoch_files is not None:
        all_X, all_y = [], []
        for class_id, epoch_path in fif_epoch_files:
            class_X, class_y = _load_mne_epoch_file(epoch_path, cfg, label_override=class_id - 1)
            all_X.append(class_X)
            all_y.append(class_y)
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
    elif eeglab_files is not None:
        if eeglab_files["mode"] == "single":
            X, y = _load_eeglab_epochs(eeglab_files["files"][0], cfg)
        else:
            all_X, all_y = [], []
            for class_id, set_path in eeglab_files["files"]:
                class_X, class_y = _load_eeglab_epochs(set_path, cfg, label_override=class_id - 1)
                all_X.append(class_X)
                all_y.append(class_y)
            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)
    else:
        path = Path(data_root) / f"subject_{subject_id:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find subject_{subject_id:02d}.npz or matching .set/.fdt/.fif epoch files under {data_root}."
            )
        data = np.load(path)
        X, y = data["X"], data["y"]
        _validate_subject_shape(X, cfg, path)

    if getattr(cfg, "skip_bandpass", False):
        print("[INFO] skip bandpass_filter (cfg.skip_bandpass=True)")
    else:
        X = bandpass_filter(X, cfg.bandpass_low, cfg.bandpass_high, cfg.sample_rate)

    return X.astype(np.float32), y.astype(np.int64)


def _build_loader(X, y, cfg, shuffle=False, drop_last=False):
    preload_device = getattr(cfg, "preload_device", None)
    dataset = MIDataset(X, y, device=preload_device)
    pin_memory = bool(getattr(cfg, "pin_memory", True)) and preload_device is None

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def _label_counts(labels, n_classes):
    labels = np.asarray(labels, dtype=np.int64)
    return np.bincount(labels, minlength=n_classes).astype(int).tolist()


def _split_train_val_time_ordered(indices, y, cfg):
    """按类别分别使用原始 trial 顺序切 validation。LOSO 目录还会用这个逻辑。"""
    val_size = getattr(cfg, "val_size", 0.1)
    indices = np.asarray(indices, dtype=np.int64)
    train_parts, val_parts = [], []

    for class_id in range(cfg.n_classes):
        class_idx = np.sort(indices[y[indices] == class_id])
        if len(class_idx) < 2:
            raise ValueError(
                f"Class {class_id} has only {len(class_idx)} train_val trials; "
                "cannot create a time-ordered validation split."
            )
        n_val = int(round(len(class_idx) * val_size))
        n_val = max(1, min(n_val, len(class_idx) - 1))
        train_parts.append(class_idx[:-n_val])
        val_parts.append(class_idx[-n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _split_train_val_stratified(train_val_idx, y, cfg, random_state):
    """
    在 train_val 子集上做 stratified random 切分，validation 比例 = cfg.val_size。
    返回 (train_idx, val_idx)，已升序排列方便 debug。
    """
    train_val_idx = np.asarray(train_val_idx, dtype=np.int64)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=cfg.val_size,
        stratify=y[train_val_idx],
        random_state=random_state,
    )
    return np.sort(train_idx).astype(np.int64), np.sort(val_idx).astype(np.int64)


def get_fold_loaders(X, y, fold_idx, cfg):
    """
    返回第 fold_idx 折的 train/val/test DataLoader。
    外层 trial-level StratifiedKFold 留出的 fold 作为 test。
    剩余 trial 分层随机抽取 val_size 作为 validation。
    """
    skf = StratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.random_seed,
    )
    splits = list(skf.split(X, y))
    train_val_idx, test_idx = splits[fold_idx]
    train_idx, val_idx = _split_train_val_stratified(
        train_val_idx,
        y,
        cfg,
        random_state=cfg.random_seed + fold_idx,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    mean, std = None, None
    if cfg.channel_normalize:
        mean, std = fit_channel_standardizer(X_train)
        X_train = apply_channel_standardize(X_train, mean, std)
        X_val = apply_channel_standardize(X_val, mean, std)
        X_test = apply_channel_standardize(X_test, mean, std)

    train_loader = _build_loader(X_train, y[train_idx], cfg, shuffle=True, drop_last=True)
    val_loader = _build_loader(X_val, y[val_idx], cfg, shuffle=False)
    test_loader = _build_loader(X_test, y[test_idx], cfg, shuffle=False)

    split_info = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": np.asarray(test_idx).tolist(),
        "split_policy": f"outer stratified {cfg.n_folds}-fold test; stratified random val_size validation from train_val",
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_counts": {
            "train": _label_counts(y[train_idx], cfg.n_classes),
            "val": _label_counts(y[val_idx], cfg.n_classes),
            "test": _label_counts(y[test_idx], cfg.n_classes),
        },
        "standardizer": None,
    }
    if cfg.channel_normalize:
        split_info["standardizer"] = {
            "fit_on": "train",
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        }

    return train_loader, val_loader, test_loader, split_info

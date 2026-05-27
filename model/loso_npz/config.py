"""
config.py - LOSO NPZ 实验超参数。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    # ---------- 数据相关 ----------
    n_channels = 59
    n_samples = 1126
    n_classes = 6
    sample_rate = 250

    data_root = str(PROJECT_ROOT / "data" / "processed" / "pythondata1")
    npz_variant = "NoICA"

    # ---------- 预处理 ----------
    bandpass_low = 4.0
    bandpass_high = 40.0
    channel_normalize = True
    # LOSO 默认显式跑 4-40Hz，避免 bandpass_low/high 只是摆设。
    # 如果输入 NPZ 已经在上游严格做过同样滤波，可改回 True。
    skip_bandpass = False

    # ---------- 训练 ----------
    batch_size = 64
    num_workers = 0
    pin_memory = True
    preload_gpu = False
    preload_device = None
    n_epochs = 200
    lr = 1e-3
    optimizer = "adamw"
    weight_decay = 1e-4
    lr_scheduler = "plateau"
    grad_clip_norm = 1.0
    plateau_factor = 0.5
    plateau_patience = 10

    # 早停
    early_stop_patience = 30
    early_stop_metric = "kappa"

    # ---------- LOSO validation ----------
    n_folds = 10
    # LOSO 中每个训练 subject 内部按类别取时间末尾 val_size 做 validation。
    val_size = 0.1
    random_seed = 42

    # ---------- 模型相关 ----------
    temporal_kernel = 125
    n_temporal_filters = 16
    depth_multiplier = 2
    separable_kernel = 16
    dropout = 0.5
    spatial_max_norm = 1.0
    classifier_max_norm = 0.25

    # ---------- 日志 ----------
    log_dir = "./experiments"
    save_model = False

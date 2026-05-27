"""
config.py - 所有超参数集中在这里。
改超参只在这一个文件改，不要散落在代码各处。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    # ---------- 数据相关（基本不动） ----------
    n_channels = 59
    n_samples = 1126
    n_classes = 6
    sample_rate = 250

    # 数据路径：直接读取 preprocessing 输出的 NPZ:
    #   data_root/S1/S1_EEGNet_NoICA_uV.npz
    data_root = str(PROJECT_ROOT / "data" / "processed" / "pythondata1")
    npz_variant = "NoICA"

    # ---------- 预处理 ----------
    bandpass_low = 0.5
    bandpass_high = 40.0

    channel_normalize = True
    # 这里的数据已经在预处理阶段滤波/降采样过；命令行加 --apply_bandpass 可额外做 0.5-40Hz。
    skip_bandpass = True

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
    lr_scheduler = "cosine"
    grad_clip_norm = 1.0
    plateau_factor = 0.5
    plateau_patience = 10

    # 早停
    early_stop_patience = 30
    # checkpoint/early stopping 选择指标：可选 "kappa" 或 "macro_f1"，不要用 accuracy。
    early_stop_metric = "kappa"

    # ---------- Subject-dependent 交叉验证 ----------
    n_folds = 10
    # 外层分层 K-fold 留出 test 后，从剩余 train_val 中分层随机抽取 validation。
    val_size = 0.15
    random_seed = 42

    # ---------- 模型相关 ----------
    temporal_kernel = 125
    n_temporal_filters = 16
    depth_multiplier = 2
    separable_kernel = 16
    dropout = 0.25
    spatial_max_norm = 1.0
    classifier_max_norm = 0.25

    # ---------- 日志 ----------
    log_dir = "./experiments"
    save_model = False

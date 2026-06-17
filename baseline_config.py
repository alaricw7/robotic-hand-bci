"""Defaults for the TriDomain baseline.

The runner intentionally reuses the parent check_experiment data/training
protocol and EEGNet optimizer defaults. These values describe the baseline
TriDomain architecture switches.
"""

from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent

TRI_CHANNEL_NAMES = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6",
    "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "TP7", "TP8",
    "Pz", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]


TRI_DEFAULTS = {
    "n_channels": 59,
    "n_classes": 6,
    "sample_rate": 250,
    "random_seed": 42,
    "tri_elp_path": str(THIS_DIR / "Standard-10-5-Cap385_witheog.elp"),
    "tri_channel_names": TRI_CHANNEL_NAMES,
    "tri_normalize_coords": True,
    "tri_variant": "full",
    "tri_coords_mode": "std",
    "tri_use_time_attn": True,
    "tri_use_band_attn": True,
    "tri_use_space_graph": True,
    "tri_use_channel_importance": True,
    "tri_d": 64,
    "tri_time_d_model": 64,
    "tri_time_pool": 8,
    "tri_freq_windows": 4,
    "tri_freq_taps": 0,
    "tri_space_hidden": 32,
    "tri_classifier_hidden": 128,
    "dropout": 0.3,
    "modality_dropout_enabled": False,
    "modality_dropout_p": 0.2,
    "freq_pool_mode": "flatten",
    "freq_var_shrinkage": 0.0,
    "freq_tensor_rank_band": None,
    "freq_tensor_rank_chan": 16,
    "freq_tensor_rank_win": None,
    "freq_pac_enabled": False,
    "freq_pac_amp_norm": True,
    "freq_learnable_bands": False,
    "freq_window_dynamics": "none",
    "space_func_adj_enabled": False,
    "space_geom_mode": "innerproduct",
    "space_laplacian_pe": False,
    "space_lap_pe_k": 8,
    "space_sphere_sigma_init": 0.5,
    "space_lap_pe_sign_flip": False,
    "branch_decorr_enabled": False,
    "branch_decorr_weight": 0.01,
    "cross_branch_attn_enabled": False,
    "cross_branch_attn_heads": 4,
    "cross_branch_attn_layers": 1,
    "cross_branch_attn_ff_mult": 2,
    "cross_branch_attn_dropout": None,
    "cross_branch_attn_mode": "branch_token",
    "cross_branch_fusion_tokens": 8,
    "cp_rank": 32,
}


BASELINE_PRESET_NAME = "standard_coords"

BASELINE_PRESETS = {
    BASELINE_PRESET_NAME: {"tri_variant": "full", "tri_coords_mode": "std"},
}


BASELINE_PRESET_ORDER = [
    BASELINE_PRESET_NAME,
]

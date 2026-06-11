"""Defaults for the TriDomain 12-ablation experiments.

The runner intentionally reuses the parent check_experiment data/training
protocol and EEGNet optimizer defaults. These values only describe the
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
    "tri_ablation": "full",
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
    # --- Phase-1 orthogonal freq features (status as of 2026-06-11) ---
    # freq_pac_enabled: VERIFIED NEGATIVE. Implementation is correct (synthetic
    #   PAC sanity passes: coupling isolated to the true band-pair ~48870x;
    #   bit-exact off, zero-init residual => on starts from baseline at step 0),
    #   but it negative-transfers on abl_S5_S_swa + LOO (excl-kappa 0.5428 ->
    #   ~0.40, far beyond noise +-0.007). Default OFF, code KEPT (do not delete).
    "freq_pac_enabled": False,
    "freq_pac_amp_norm": True,   # only used when PAC on; default-on is better-behaved
    # freq_learnable_bands: VERIFIED NEGATIVE (3-seed excl-kappa 0.5142 +-0.0058
    #   vs baseline 0.5428 +-0.0073; learned boundaries drift off and underperform
    #   the hand-tuned lowfreq_dense prior). Default OFF, code KEPT.
    "freq_learnable_bands": False,
    # freq_window_dynamics: NEUTRAL / harmless (gru: 3-seed 0.5490 +-0.0028 vs
    #   baseline 0.5428, +0.006, not significant t~1.7). Default 'none'.
    #   KEPT as a Phase-2 combination candidate.
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
    # Phase-2 #4: bottleneck/Perceiver fusion (cross_branch_attn_mode='bottleneck').
    "cross_branch_attn_mode": "branch_token",
    "cross_branch_fusion_tokens": 8,
    # Phase-2 #5: low-rank CP trilinear fusion (fusion_head_mode='lowrank_cp').
    "cp_rank": 32,
}


ABLATION_PRESETS = {
    "time_only": {"tri_ablation": "time_only"},
    "freq_only": {"tri_ablation": "freq_only"},
    "space_only": {"tri_ablation": "space_only", "tri_coords_mode": "std"},
    "time_freq": {"tri_ablation": "time_freq"},
    "time_space": {"tri_ablation": "time_space", "tri_coords_mode": "std"},
    "full_zero_coords": {"tri_ablation": "full", "tri_coords_mode": "zero"},
    "full_std_coords": {"tri_ablation": "full", "tri_coords_mode": "std"},
    "full_no_time_attn": {
        "tri_ablation": "full",
        "tri_coords_mode": "std",
        "tri_use_time_attn": False,
    },
    "full_no_band_attn": {
        "tri_ablation": "full",
        "tri_coords_mode": "std",
        "tri_use_band_attn": False,
    },
    "full_no_graph": {
        "tri_ablation": "full",
        "tri_coords_mode": "std",
        "tri_use_space_graph": False,
    },
    "full_no_channel_imp": {
        "tri_ablation": "full",
        "tri_coords_mode": "std",
        "tri_use_channel_importance": False,
    },
    "full_random_coords": {"tri_ablation": "full", "tri_coords_mode": "random"},
    "full_freq_tensor": {
        "tri_ablation": "full_freq_tensor",
        "tri_coords_mode": "std",
        "freq_pool_mode": "tensor",
    },
    "full_cross_attn": {
        "tri_ablation": "full_cross_attn",
        "tri_coords_mode": "std",
        "cross_branch_attn_enabled": True,
    },
    "full_sphere_rbf": {
        "tri_ablation": "full_sphere_rbf",
        "tri_coords_mode": "std",
        "space_geom_mode": "sphere_rbf",
    },
    "full_lap_pe": {
        "tri_ablation": "full_lap_pe",
        "tri_coords_mode": "std",
        "space_laplacian_pe": True,
    },
    "full_sphere_lap": {
        "tri_ablation": "full_sphere_lap",
        "tri_coords_mode": "std",
        "space_geom_mode": "sphere_rbf",
        "space_laplacian_pe": True,
    },
}


LAUNCH_ORDER = [
    "time_only",
    "freq_only",
    "space_only",
    "time_freq",
    "time_space",
    "full_zero_coords",
    "full_std_coords",
    "full_no_time_attn",
    "full_no_band_attn",
    "full_no_graph",
    "full_no_channel_imp",
    "full_random_coords",
]

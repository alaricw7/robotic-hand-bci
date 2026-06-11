"""
model.py - Tri-domain (Time / Frequency / Space) EEG classifier.

TriDomainClassifier wraps a tri-domain encoder with a classifier head so the
existing EEGNet train/evaluate code can be reused unchanged.

Key point:
- If cfg.tri_elp_path and cfg.tri_channel_names are provided, SpaceBranch uses
  standard 10-5 electrode coordinates.
- If cfg.tri_elp_path is empty/None, SpaceBranch falls back to zero coordinates.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Speedup: let cuDNN auto-tune the fastest conv algorithm for the (fixed)
# input shapes used in this project. Tiny numerical drift (1e-7 range)
# but no protocol change. Disable by setting env TRI_NO_BENCHMARK=1.
if os.environ.get("TRI_NO_BENCHMARK", "0") != "1" and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def parse_active_branches(ablation):
    mapping = {
        "full": ("time", "freq", "space"),
        "time_only": ("time",),
        "freq_only": ("freq",),
        "space_only": ("space",),
        "time_freq": ("time", "freq"),
        "time_space": ("time", "space"),
        "freq_space": ("freq", "space"),
        "no_time": ("freq", "space"),
        "no_freq": ("time", "space"),
        "no_space": ("time", "freq"),
        "full_zero_coords": ("time", "freq", "space"),
        "full_std_coords": ("time", "freq", "space"),
        "full_no_time_attn": ("time", "freq", "space"),
        "full_no_band_attn": ("time", "freq", "space"),
        "full_no_graph": ("time", "freq", "space"),
        "full_no_channel_imp": ("time", "freq", "space"),
        "full_random_coords": ("time", "freq", "space"),
        "full_freq_tensor": ("time", "freq", "space"),
        "full_cross_attn": ("time", "freq", "space"),
        "full_sphere_rbf": ("time", "freq", "space"),
        "full_lap_pe": ("time", "freq", "space"),
        "full_sphere_lap": ("time", "freq", "space"),
    }
    key = (ablation or "full").lower()
    if key not in mapping:
        raise ValueError(f"Unknown tri_ablation={ablation!r}; choices={sorted(mapping)}")
    return mapping[key]


def resolve_ablation_config(cfg):
    ablation = getattr(cfg, "tri_ablation", "full")
    key = (ablation or "full").lower()
    coords_mode = getattr(cfg, "tri_coords_mode", "std")
    use_time_attn = bool(getattr(cfg, "tri_use_time_attn", True))
    use_band_attn = bool(getattr(cfg, "tri_use_band_attn", True))
    use_space_graph = bool(getattr(cfg, "tri_use_space_graph", True))
    use_channel_importance = bool(getattr(cfg, "tri_use_channel_importance", True))

    # New orthogonal toggles (default to whatever cfg already carries).
    freq_pool_mode = getattr(cfg, "freq_pool_mode", "flatten")
    cross_branch_attn_enabled = bool(getattr(cfg, "cross_branch_attn_enabled", False))
    space_geom_mode = getattr(cfg, "space_geom_mode", "innerproduct")
    space_laplacian_pe = bool(getattr(cfg, "space_laplacian_pe", False))

    if key == "full_freq_tensor":
        freq_pool_mode = "tensor"
    elif key == "full_cross_attn":
        cross_branch_attn_enabled = True
    elif key == "full_sphere_rbf":
        space_geom_mode = "sphere_rbf"
    elif key == "full_lap_pe":
        space_laplacian_pe = True
    elif key == "full_sphere_lap":
        space_geom_mode = "sphere_rbf"
        space_laplacian_pe = True

    if key == "full_zero_coords":
        coords_mode = "zero"
    elif key == "full_std_coords":
        coords_mode = "std"
    elif key == "full_random_coords":
        coords_mode = "random"
    elif key == "full_no_time_attn":
        use_time_attn = False
    elif key == "full_no_band_attn":
        use_band_attn = False
    elif key == "full_no_graph":
        use_space_graph = False
    elif key == "full_no_channel_imp":
        use_channel_importance = False

    active_branches = parse_active_branches(key)
    return {
        "ablation": key,
        "active_branches": active_branches,
        "coords_mode": coords_mode,
        "use_time_attn": use_time_attn,
        "use_band_attn": use_band_attn,
        "use_space_graph": use_space_graph,
        "use_channel_importance": use_channel_importance,
        "freq_pool_mode": freq_pool_mode,
        "cross_branch_attn_enabled": cross_branch_attn_enabled,
        "space_geom_mode": space_geom_mode,
        "space_laplacian_pe": space_laplacian_pe,
    }


# --------------------------------------------------------------------------- #
# Helper: windowed-sinc band-pass FIR
# --------------------------------------------------------------------------- #
def sinc_bandpass(num_taps: int, low_hz: float, high_hz: float, fs: float) -> torch.Tensor:
    """Hamming-windowed sinc band-pass FIR. Returns a (num_taps,) kernel."""
    if num_taps % 2 == 0:
        num_taps += 1

    n = torch.arange(num_taps) - (num_taps - 1) / 2.0
    f_low, f_high = low_hz / fs, high_hz / fs

    def lowpass(fc):
        return 2 * fc * torch.sinc(2 * fc * n)

    h = lowpass(f_high) - lowpass(f_low)
    h = h * torch.hamming_window(num_taps, periodic=False)
    h = h / (h.pow(2).sum().sqrt() + 1e-8)
    return h


# --------------------------------------------------------------------------- #
# Helper: load 2D electrode coordinates from .elp
# --------------------------------------------------------------------------- #
def load_elp_2d_coords(elp_path, ch_names, n_ch=59, normalize=True):
    """
    Load 2D EEG coordinates from a Standard-10-5 .elp file.

    Important:
    ch_names must match the exact channel order of input X[:, C, T].
    For your data, ch_names should be the 59 names stored in the original NPZ:
    ['Fpz', 'Fp1', 'Fp2', ..., 'Oz', 'O1', 'O2'].
    """
    if elp_path is None or str(elp_path).strip() == "":
        raise ValueError("elp_path is empty. Cannot load electrode coordinates.")

    if not os.path.exists(elp_path):
        raise FileNotFoundError(f".elp file not found: {elp_path}")

    if ch_names is None:
        raise ValueError(
            "cfg.tri_channel_names is None. "
            "You must provide the exact 59-channel order used by X[:, C, T]."
        )

    ch_names = list(ch_names)

    if len(ch_names) != n_ch:
        raise ValueError(
            f"tri_channel_names has {len(ch_names)} channels, but n_ch={n_ch}."
        )

    pos = {}

    with open(elp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            kind = parts[0].upper()
            name = parts[1].upper()

            # Only use EEG coordinates. Ignore FID / EOG.
            if kind != "EEG":
                continue

            try:
                x = float(parts[2])
                y = float(parts[3])
            except ValueError:
                continue

            pos[name] = [x, y]

    missing = [ch for ch in ch_names if ch.upper() not in pos]
    if missing:
        available_preview = sorted(list(pos.keys()))[:30]
        raise ValueError(
            "Some channels are missing in the .elp file:\n"
            f"{missing}\n\n"
            "Please check channel naming and channel order.\n"
            f"Available EEG names preview: {available_preview}"
        )

    coords = torch.tensor(
        [pos[ch.upper()] for ch in ch_names],
        dtype=torch.float32,
    )

    if normalize:
        coords = (coords - coords.mean(dim=0, keepdim=True)) / (
            coords.std(dim=0, keepdim=True) + 1e-6
        )

    return coords


# --------------------------------------------------------------------------- #
# Helper: load 3D electrode coordinates from .elp (Cartesian XYZ)
# --------------------------------------------------------------------------- #
def load_elp_3d_coords(elp_path, ch_names, n_ch=59, normalize=True):
    """
    Load 3D EEG coordinates (parts[2:5] = x, y, z) from a .elp file.

    Same structure / validation as load_elp_2d_coords, but reads a z column.
    When normalize=True each electrode is projected onto the unit sphere
    (do NOT z-score per axis here: that breaks the spherical geometry that
    sphere_rbf relies on).
    """
    if elp_path is None or str(elp_path).strip() == "":
        raise ValueError("elp_path is empty. Cannot load electrode coordinates.")

    if not os.path.exists(elp_path):
        raise FileNotFoundError(f".elp file not found: {elp_path}")

    if ch_names is None:
        raise ValueError(
            "cfg.tri_channel_names is None. "
            "You must provide the exact 59-channel order used by X[:, C, T]."
        )

    ch_names = list(ch_names)

    if len(ch_names) != n_ch:
        raise ValueError(
            f"tri_channel_names has {len(ch_names)} channels, but n_ch={n_ch}."
        )

    pos = {}

    with open(elp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            kind = parts[0].upper()
            name = parts[1].upper()

            if kind != "EEG":
                continue

            try:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError:
                continue

            pos[name] = [x, y, z]

    missing = [ch for ch in ch_names if ch.upper() not in pos]
    if missing:
        available_preview = sorted(list(pos.keys()))[:30]
        raise ValueError(
            "Some channels are missing in the .elp file:\n"
            f"{missing}\n\n"
            "Please check channel naming and channel order.\n"
            f"Available EEG names preview: {available_preview}"
        )

    coords = torch.tensor(
        [pos[ch.upper()] for ch in ch_names],
        dtype=torch.float32,
    )

    if normalize:
        coords = coords / (coords.norm(dim=1, keepdim=True) + 1e-8)

    return coords


# --------------------------------------------------------------------------- #
# Time branch: multi-scale temporal conv + temporal self-attention
# --------------------------------------------------------------------------- #
class TimeBranch(nn.Module):
    def __init__(
        self,
        n_ch=59,
        fs=250,
        d_model=64,
        d_out=64,
        pool=8,
        dropout=0.2,
        use_attn=True,
        pool_mode="mean",
    ):
        super().__init__()
        self.use_attn = use_attn
        self.pool_mode = str(pool_mode).lower()
        if self.pool_mode not in ("mean", "attn"):
            raise ValueError(f"time_pool_mode must be 'mean' or 'attn', got {pool_mode!r}")
        self.ks = [15, 31, 63, 125]

        self.scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_ch, n_ch, k, padding=k // 2, groups=n_ch, bias=False),
                    nn.BatchNorm1d(n_ch),
                    nn.ELU(),
                )
                for k in self.ks
            ]
        )

        self.mix = nn.Sequential(
            nn.Conv1d(len(self.ks) * n_ch, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.AvgPool1d(pool),
        )

        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)
        # Attention pooling query — only built when pool_mode='attn' so
        # default ('mean') graph stays bit-exact with current model.
        if self.pool_mode == "attn":
            self.pool_query = nn.Parameter(torch.zeros(d_model))
            nn.init.normal_(self.pool_query, std=0.02)
        else:
            self.pool_query = None
        self.out = nn.Linear(d_model, d_out)

    def _pos_slice(self, n_tokens):
        if n_tokens <= self.pos.size(1):
            return self.pos[:, :n_tokens]

        return F.interpolate(
            self.pos.transpose(1, 2),
            size=n_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x):
        # x: (B, C, T)
        h = torch.cat([scale(x) for scale in self.scales], dim=1)
        h = self.mix(h).transpose(1, 2)  # (B, T', d_model)
        h = h + self._pos_slice(h.size(1))

        if self.use_attn:
            a, _ = self.attn(h, h, h)
            h = self.norm(h + a)
        else:
            h = self.norm(h)

        if self.pool_mode == "attn":
            # h: (B, T', d_model)  query: (d_model,)
            score = (h @ self.pool_query) / math.sqrt(h.size(-1))   # (B, T')
            w = torch.softmax(score, dim=-1)                        # (B, T')
            pooled = (h * w.unsqueeze(-1)).sum(dim=1)               # (B, d_model)
        else:
            pooled = h.mean(dim=1)
        e = self.out(pooled)
        return e, h


# --------------------------------------------------------------------------- #
# Frequency branch: sinc filter bank + time-resolved DE + band attention
# --------------------------------------------------------------------------- #
FREQ_BAND_PRESETS = {
    "standard": [(0.5, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)],
    # Densify the low-frequency region (motor-related slow transients), keep 6 bands
    # so DE flatten dim (n_bands * n_ch * n_win = 1416) and proj head are unchanged.
    "lowfreq_dense": [(0.5, 2), (2, 4), (4, 6), (6, 8), (8, 13), (13, 30)],
}


def _resolve_bands(spec):
    if spec is None:
        return FREQ_BAND_PRESETS["standard"]
    if isinstance(spec, str):
        if spec not in FREQ_BAND_PRESETS:
            raise ValueError(
                f"unknown freq_bands preset {spec!r}; choices={list(FREQ_BAND_PRESETS)}"
            )
        return FREQ_BAND_PRESETS[spec]
    # Treat as a custom list of (lo, hi) tuples
    bands = [tuple(b) for b in spec]
    if any(len(b) != 2 for b in bands):
        raise ValueError("freq_bands list entries must be (lo_hz, hi_hz) pairs")
    return bands


class FreqBranch(nn.Module):
    def __init__(
        self,
        n_ch=59,
        fs=250,
        d_out=64,
        n_win=4,
        taps=None,
        dropout=0.3,
        use_band_attn=True,
        bands=None,
        freq_pool_mode="flatten",
        freq_var_shrinkage=0.0,
        freq_tensor_rank_band=None,
        freq_tensor_rank_chan=16,
        freq_tensor_rank_win=None,
        freq_pac_enabled=False,
        freq_pac_amp_norm=True,
        freq_learnable_bands=False,
        freq_window_dynamics="none",
    ):
        super().__init__()
        self.fs = fs
        self.n_win = n_win
        self.use_band_attn = use_band_attn
        self.freq_pac_enabled = bool(freq_pac_enabled)
        self.freq_pac_amp_norm = bool(freq_pac_amp_norm)
        self.freq_learnable_bands = bool(freq_learnable_bands)
        self.freq_window_dynamics = str(freq_window_dynamics).lower()
        if self.freq_window_dynamics not in ("none", "dwconv", "gru", "attn"):
            raise ValueError(
                "freq_window_dynamics must be one of none/dwconv/gru/attn, "
                f"got {freq_window_dynamics!r}"
            )
        self.freq_pool_mode = str(freq_pool_mode).lower()
        if self.freq_pool_mode not in ("flatten", "structured", "tensor"):
            raise ValueError(
                "freq_pool_mode must be 'flatten', 'structured' or 'tensor', "
                f"got {freq_pool_mode!r}"
            )
        self.freq_var_shrinkage = float(freq_var_shrinkage)
        if not (0.0 <= self.freq_var_shrinkage <= 1.0):
            raise ValueError(
                f"freq_var_shrinkage must be in [0, 1], got {self.freq_var_shrinkage}"
            )

        self.bands = _resolve_bands(bands)
        self.n_bands = len(self.bands)

        taps = taps or (fs // 2)
        if taps % 2 == 0:
            taps += 1
        self.taps = taps
        self.n_ch = n_ch

        if not self.freq_learnable_bands:
            # Default path: fixed sinc filters (bit-exact with current model).
            self.filters = nn.ModuleList()
            for lo, hi in self.bands:
                conv = nn.Conv1d(
                    n_ch,
                    n_ch,
                    taps,
                    padding=taps // 2,
                    groups=n_ch,
                    bias=False,
                )
                k = sinc_bandpass(taps, lo, hi, fs)
                with torch.no_grad():
                    conv.weight.copy_(k.view(1, 1, -1).repeat(n_ch, 1, 1))
                self.filters.append(conv)
        else:
            # Learnable (center, bandwidth) per band, init = preset so the
            # initial reconstructed kernel matches the fixed one.
            self.filters = None
            self.band_f_min = 0.5
            self.band_bw_min = 0.5

            def _inv_softplus(y):
                y = max(float(y), 1e-4)
                return math.log(math.expm1(y))

            p_lo, p_bw = [], []
            for lo, hi in self.bands:
                p_lo.append(_inv_softplus(lo - self.band_f_min))
                p_bw.append(_inv_softplus((hi - lo) - self.band_bw_min))
            self.band_p_lo = nn.Parameter(torch.tensor(p_lo, dtype=torch.float32))
            self.band_p_bw = nn.Parameter(torch.tensor(p_bw, dtype=torch.float32))

        feat_dim = self.n_bands * n_ch * n_win

        if self.freq_pool_mode == "flatten":
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(256, d_out),
                nn.ELU(),
            )
            self.structured_channel_pool = None
            self.structured_win_pool = None
            self.structured_out = None
            self.U_band = None
            self.U_chan = None
            self.U_win = None
            self.tensor_out = None
        elif self.freq_pool_mode == "structured":
            self.proj = None
            self.structured_channel_pool = nn.Conv2d(
                self.n_bands,
                self.n_bands,
                kernel_size=(n_ch, 1),
                groups=self.n_bands,
                bias=True,
            )
            self.structured_win_pool = nn.Linear(n_win, 1)
            self.structured_out = nn.Sequential(
                nn.Linear(self.n_bands, d_out),
                nn.ELU(),
                nn.Dropout(dropout),
            )
            self.U_band = None
            self.U_chan = None
            self.U_win = None
            self.tensor_out = None
        else:  # tensor: Tucker-style multilinear decomposition
            self.proj = None
            self.structured_channel_pool = None
            self.structured_win_pool = None
            self.structured_out = None

            rb = freq_tensor_rank_band if freq_tensor_rank_band is not None else min(self.n_bands, 4)
            rc = freq_tensor_rank_chan if freq_tensor_rank_chan is not None else 16
            rw = freq_tensor_rank_win if freq_tensor_rank_win is not None else min(n_win, 2)
            rb = int(max(1, min(rb, self.n_bands)))
            rc = int(max(1, min(rc, n_ch)))
            rw = int(max(1, min(rw, n_win)))
            self.tensor_rank_band = rb
            self.tensor_rank_chan = rc
            self.tensor_rank_win = rw

            self.U_band = nn.Parameter(torch.empty(self.n_bands, rb))
            self.U_chan = nn.Parameter(torch.empty(n_ch, rc))
            self.U_win = nn.Parameter(torch.empty(n_win, rw))
            nn.init.orthogonal_(self.U_band)
            nn.init.orthogonal_(self.U_chan)
            nn.init.orthogonal_(self.U_win)
            self.tensor_out = nn.Sequential(
                nn.Linear(rb * rc * rw, d_out),
                nn.ELU(),
                nn.Dropout(dropout),
            )

        self.band_att = nn.Sequential(
            nn.Linear(self.n_bands, self.n_bands),
            nn.Tanh(),
            nn.Linear(self.n_bands, self.n_bands),
        )

        flatten_params = feat_dim * 256 + 256 + 256 * d_out + d_out
        structured_params = (
            self.n_bands * n_ch + self.n_bands
            + n_win + 1
            + self.n_bands * d_out + d_out
        )
        if self.proj is not None:
            active_params = sum(p.numel() for p in self.proj.parameters())
        elif self.structured_out is not None:
            active_params = sum(
                p.numel()
                for module in (
                    self.structured_channel_pool,
                    self.structured_win_pool,
                    self.structured_out,
                )
                for p in module.parameters()
            )
        else:
            active_params = (
                self.U_band.numel() + self.U_chan.numel() + self.U_win.numel()
                + sum(p.numel() for p in self.tensor_out.parameters())
            )

        if self.freq_pool_mode == "tensor":
            rb, rc, rw = self.tensor_rank_band, self.tensor_rank_chan, self.tensor_rank_win
            tensor_params = (
                self.n_bands * rb + n_ch * rc + n_win * rw
                + (rb * rc * rw) * d_out + d_out
            )
        else:
            tensor_params = "n/a"
        print(
            "[FreqBranch] "
            f"freq_pool_mode={self.freq_pool_mode}, "
            f"active_pool_params={active_params}, "
            f"flatten_pool_params={flatten_params}, "
            f"structured_pool_params={structured_params}, "
            f"tensor_pool_params={tensor_params}"
        )

        # --- new orthogonal modules. Built under an RNG save/restore so that
        #     enabling them does NOT perturb the init of modules constructed
        #     AFTER this branch (SpaceBranch / head / aux_heads). Combined with
        #     the zero-init PAC residual, an enabled PAC model starts from the
        #     bit-exact baseline state. ---
        _rng_state = torch.get_rng_state()

        # (3) window-axis dynamics: shape-preserving residual on de.
        if self.freq_window_dynamics == "dwconv":
            self.win_dyn = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)
        elif self.freq_window_dynamics == "gru":
            self.win_dyn = nn.GRU(input_size=1, hidden_size=8, batch_first=True)
            self.win_dyn_out = nn.Linear(8, 1)
        elif self.freq_window_dynamics == "attn":
            self.win_dyn_in = nn.Linear(1, 8)
            self.win_dyn = nn.MultiheadAttention(8, num_heads=1, batch_first=True)
            self.win_dyn_out = nn.Linear(8, 1)
        else:
            self.win_dyn = None

        # (1) PAC / Canolty MVL head.
        if self.freq_pac_enabled:
            self.pac_n_pairs = self.n_bands * (self.n_bands - 1) // 2
            pac_in = n_ch * 2 * self.pac_n_pairs
            # LayerNorm on the 30-dim (2*n_pairs) MVL descriptor per channel,
            # so PAC features enter the MLP at a controlled scale.
            self.pac_norm = nn.LayerNorm(2 * self.pac_n_pairs)
            self.pac_proj = nn.Sequential(
                nn.Linear(pac_in, 256),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(256, d_out),
            )
            # Zero-init the residual's last layer => at step 0 the PAC term is
            # exactly 0, so training starts from the bit-exact baseline state.
            nn.init.zeros_(self.pac_proj[-1].weight)
            nn.init.zeros_(self.pac_proj[-1].bias)
            pac_params = sum(p.numel() for p in self.pac_proj.parameters())
            print(
                "[FreqBranch][PAC] "
                f"enabled=True, amp_norm={self.freq_pac_amp_norm}, "
                f"n_pairs={self.pac_n_pairs}, pac_proj_params={pac_params}, "
                "pac_norm=LayerNorm, residual_last_layer=zero-init"
            )
        else:
            self.pac_n_pairs = 0
            self.pac_norm = None
            self.pac_proj = None

        if self.freq_learnable_bands:
            print(
                "[FreqBranch][learnable_bands] init (f_lo, f_hi) Hz: "
                + ", ".join(f"({lo:.2f},{hi:.2f})" for lo, hi in self.bands)
            )
        if self.freq_window_dynamics != "none":
            wd_params = sum(
                p.numel() for m in (self.win_dyn,
                                    getattr(self, "win_dyn_in", None),
                                    getattr(self, "win_dyn_out", None))
                if m is not None for p in m.parameters()
            )
            print(f"[FreqBranch][window_dynamics] mode={self.freq_window_dynamics}, params={wd_params}")

        # Restore the RNG so downstream module init is identical to baseline.
        torch.set_rng_state(_rng_state)

    # ----- helpers for the new orthogonal features ----- #
    def _sinc_kernel_t(self, lo, hi, device):
        """Differentiable Hamming-windowed sinc band-pass (lo, hi are tensors)."""
        taps = self.taps
        n = torch.arange(taps, device=device, dtype=torch.float32) - (taps - 1) / 2.0

        def lowpass(fc):
            return 2 * fc * torch.sinc(2 * fc * n)

        h = lowpass(hi / self.fs) - lowpass(lo / self.fs)
        h = h * torch.hamming_window(taps, periodic=False, device=device)
        h = h / (h.pow(2).sum().sqrt() + 1e-8)
        return h

    @staticmethod
    def _hilbert(sig):
        """Analytic signal via FFT (differentiable). sig: (..., T) real."""
        T = sig.size(-1)
        Xf = torch.fft.fft(sig, dim=-1)
        h = torch.zeros(T, device=sig.device, dtype=sig.dtype)
        if T % 2 == 0:
            h[0] = 1.0
            h[T // 2] = 1.0
            h[1:T // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(T + 1) // 2] = 2.0
        return torch.fft.ifft(Xf * h, dim=-1)

    def _compute_pac(self, band_sigs):
        """band_sigs: list of n_bands tensors (B, C, Tc). Returns (B, C, 2*n_pairs).
        Phase from low band i, amplitude from high band j (i<j); keep re & im."""
        re, im, amp = [], [], []
        for s in band_sigs:
            z = self._hilbert(s)
            r, i = z.real, z.imag
            a = torch.sqrt(r * r + i * i)
            re.append(r); im.append(i); amp.append(a)
        feats = []
        n = len(band_sigs)
        for i in range(n):
            cos_i = re[i] / (amp[i] + 1e-8)
            sin_i = im[i] / (amp[i] + 1e-8)
            for j in range(i + 1, n):
                A = amp[j]
                if self.freq_pac_amp_norm:
                    A = (A - A.mean(dim=-1, keepdim=True)) / (
                        A.std(dim=-1, keepdim=True) + 1e-6
                    )
                real_ij = (A * cos_i).mean(dim=-1)   # (B, C)
                imag_ij = (A * sin_i).mean(dim=-1)   # (B, C)
                feats.append(real_ij)
                feats.append(imag_ij)
        return torch.stack(feats, dim=-1)            # (B, C, 2*n_pairs)

    def _apply_window_dynamics(self, de):
        """de: (B, N, C, W) -> same shape, residual dynamics along W."""
        B, N, C, W = de.shape
        if self.freq_window_dynamics == "dwconv":
            h = de.reshape(B * N * C, 1, W)
            h = self.win_dyn(h).reshape(B, N, C, W)
            return de + h
        if self.freq_window_dynamics == "gru":
            h = de.reshape(B * N * C, W, 1)
            h, _ = self.win_dyn(h)
            h = self.win_dyn_out(h).reshape(B, N, C, W)
            return de + h
        # attn
        h = de.reshape(B * N * C, W, 1)
        h = self.win_dyn_in(h)
        a, _ = self.win_dyn(h, h, h)
        a = self.win_dyn_out(a).reshape(B, N, C, W)
        return de + a

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        win = T // self.n_win

        if win < 1:
            raise ValueError(f"Input has too few samples for n_win={self.n_win}: T={T}")

        de_list = []
        energy = []
        band_sigs = [] if self.freq_pac_enabled else None

        for bi in range(self.n_bands):
            if self.freq_learnable_bands:
                lo = self.band_f_min + F.softplus(self.band_p_lo[bi])
                bw = self.band_bw_min + F.softplus(self.band_p_bw[bi])
                hi = torch.clamp(lo + bw, max=self.fs / 2.0 - 1.0)
                k = self._sinc_kernel_t(lo, hi, x.device)
                weight = k.view(1, 1, -1).repeat(self.n_ch, 1, 1)
                xb = F.conv1d(x, weight, padding=self.taps // 2, groups=self.n_ch)
                xb = xb[..., : win * self.n_win]
            else:
                xb = self.filters[bi](x)[..., : win * self.n_win]

            if self.freq_pac_enabled:
                band_sigs.append(xb)            # (B, C, Tc) full-time band signal
            xb = xb.reshape(B, C, self.n_win, win)

            if self.freq_var_shrinkage == 0.0:
                var = xb.var(dim=-1, unbiased=False) + 1e-6
            else:
                var_raw = xb.var(dim=-1, unbiased=False)
                var_bar = var_raw.mean(dim=(1, 2), keepdim=True)
                lam = self.freq_var_shrinkage
                var = ((1.0 - lam) * var_raw + lam * var_bar) + 1e-6
            logp = 0.5 * torch.log(2 * math.pi * math.e * var)

            de_list.append(logp)
            energy.append(logp.mean(dim=(1, 2)))

        de = torch.stack(de_list, dim=1)      # (B, n_bands, C, n_win)
        be = torch.stack(energy, dim=1)       # (B, n_bands)

        if self.use_band_attn:
            w = torch.softmax(self.band_att(be), dim=-1)
        else:
            w = torch.full(
                (B, self.n_bands),
                1.0 / self.n_bands,
                device=x.device,
                dtype=x.dtype,
            )
        de = de * w.view(B, self.n_bands, 1, 1)

        # (3) window-axis dynamics, shape-preserving, before any pooling.
        if self.win_dyn is not None:
            de = self._apply_window_dynamics(de)

        if self.freq_pool_mode == "flatten":
            e = self.proj(de.reshape(B, -1))
        elif self.freq_pool_mode == "structured":
            h = self.structured_channel_pool(de).squeeze(2)  # (B, n_bands, n_win)
            h = self.structured_win_pool(h).squeeze(-1)      # (B, n_bands)
            e = self.structured_out(h)
        else:  # tensor: mode-product contraction (de is (B, N, C, W))
            g = torch.einsum('bncw,nr->brcw', de, self.U_band)
            g = torch.einsum('brcw,cs->brsw', g, self.U_chan)
            g = torch.einsum('brsw,wt->brst', g, self.U_win)  # (B, Rb, Rc, Rw)
            e = self.tensor_out(g.reshape(B, -1))

        # (1) PAC residual: e <- e + pac_proj(LN(MVL features))
        if self.pac_proj is not None:
            pac = self._compute_pac(band_sigs)           # (B, C, 2*n_pairs)
            pac = self.pac_norm(pac)                      # LN over last dim
            e = e + self.pac_proj(pac.reshape(B, -1))

        # Per-band raw tokens for bottleneck fusion (view; no params, no cost
        # when unused). de is the band-attn-weighted (+ window-dyn) tensor.
        freq_tokens = de.reshape(B, self.n_bands, C * self.n_win)
        return e, w, freq_tokens


# --------------------------------------------------------------------------- #
# Space branch: coordinate-based graph attention + channel importance
# --------------------------------------------------------------------------- #
class SpaceBranch(nn.Module):
    def __init__(
        self,
        n_ch=59,
        coords=None,
        d_hidden=32,
        d_out=64,
        use_graph=True,
        use_channel_importance=True,
        space_func_adj_enabled=False,
        space_geom_mode="innerproduct",
        space_laplacian_pe=False,
        space_lap_pe_k=8,
        space_sphere_sigma_init=0.5,
        space_lap_pe_sign_flip=False,
    ):
        super().__init__()
        self.n_ch = n_ch
        self.d_hidden = d_hidden
        self.use_graph = use_graph
        self.use_channel_importance = use_channel_importance
        self.space_func_adj_enabled = bool(space_func_adj_enabled)
        self.space_geom_mode = str(space_geom_mode).lower()
        if self.space_geom_mode not in ("innerproduct", "sphere_rbf"):
            raise ValueError(
                "space_geom_mode must be 'innerproduct' or 'sphere_rbf', "
                f"got {space_geom_mode!r}"
            )
        self.space_laplacian_pe = bool(space_laplacian_pe)
        self.space_lap_pe_sign_flip = bool(space_lap_pe_sign_flip)

        if coords is None:
            coords = torch.zeros(n_ch, 2)

        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32)

        if coords.shape[0] != n_ch or coords.shape[1] not in (2, 3):
            raise ValueError(
                f"coords must have shape ({n_ch}, 2) or ({n_ch}, 3), "
                f"got {tuple(coords.shape)}"
            )
        self.coords_dim = int(coords.shape[1])

        self.register_buffer("coords", coords.float())

        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, n_ch, 25, padding=12, groups=n_ch, bias=False),
            nn.BatchNorm1d(n_ch),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.node = nn.Linear(16, d_hidden)

        # Geometry of the adjacency. innerproduct keeps the exact construction
        # order of the current model (Wq then Wk) so default RNG is unchanged.
        if self.space_geom_mode == "innerproduct":
            self.Wq = nn.Linear(self.coords_dim, d_hidden, bias=False)
            self.Wk = nn.Linear(self.coords_dim, d_hidden, bias=False)
            self.log_sigma = None
        else:  # sphere_rbf
            if self.coords_dim != 3:
                raise ValueError(
                    "space_geom_mode='sphere_rbf' requires 3D coords "
                    f"(n_ch, 3); got coords_dim={self.coords_dim}"
                )
            self.Wq = None
            self.Wk = None
            with torch.no_grad():
                u = self.coords / (self.coords.norm(dim=1, keepdim=True) + 1e-8)
                G = torch.arccos(torch.clamp(u @ u.t(), -1 + 1e-7, 1 - 1e-7))
            self.register_buffer("geo_dist", G.float())  # (C, C), in [0, pi]
            self.log_sigma = nn.Parameter(
                torch.tensor(float(math.log(space_sphere_sigma_init)))
            )

        if self.space_func_adj_enabled:
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.beta = None
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.imp = nn.Linear(d_hidden, 1)
        self.out = nn.Sequential(
            nn.Linear(d_hidden, d_out),
            nn.ELU(),
        )

        # Graph-Laplacian positional encoding (computed once on CPU, no_grad).
        if self.space_laplacian_pe:
            k = int(max(1, min(int(space_lap_pe_k), n_ch - 1)))
            self.lap_pe_k = k
            with torch.no_grad():
                c = self.coords.float()
                D = torch.cdist(c, c)  # (C, C) euclidean
                off = D[~torch.eye(n_ch, dtype=torch.bool)]
                med = off.median()
                W = torch.exp(-(D ** 2) / (2 * med ** 2 + 1e-8))
                W = W * (1.0 - torch.eye(n_ch))
                deg = W.sum(dim=1)
                dinv = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
                L = torch.eye(n_ch) - dinv @ W @ dinv
                L = 0.5 * (L + L.t())
                evals, evecs = torch.linalg.eigh(L)
                pe = evecs[:, 1:1 + k]  # skip trivial eigvec 0
                # deterministic sign fix: largest-|.| component positive
                for j in range(pe.size(1)):
                    idx = torch.argmax(pe[:, j].abs())
                    if pe[idx, j] < 0:
                        pe[:, j] = -pe[:, j]
                self._lap_evals = evals.float()
            self.register_buffer("lap_pe", pe.float())  # (C, k)
            self.pe_proj = nn.Linear(k, d_hidden)
        else:
            self.lap_pe_k = None
            self.pe_proj = None

        log_sigma_params = 0 if self.log_sigma is None else self.log_sigma.numel()
        pe_proj_params = (
            0 if self.pe_proj is None
            else sum(p.numel() for p in self.pe_proj.parameters())
        )
        print(
            "[TriDomain][SpaceBranch] "
            f"space_geom_mode={self.space_geom_mode}, "
            f"space_laplacian_pe={self.space_laplacian_pe}, "
            f"lap_pe_k={self.lap_pe_k}, "
            f"log_sigma_params={log_sigma_params}, "
            f"pe_proj_params={pe_proj_params}"
        )
        if self.space_geom_mode == "sphere_rbf":
            print(
                "[TriDomain][SpaceBranch] geo_dist min/max: "
                f"{float(self.geo_dist.min()):.4f} / {float(self.geo_dist.max()):.4f} "
                "(expected within [0, pi])"
            )
        if self.space_laplacian_pe:
            ev = self._lap_evals
            ortho = self.lap_pe.t() @ self.lap_pe
            offdiag = (ortho - torch.diag(torch.diag(ortho))).abs().max()
            print(
                "[TriDomain][SpaceBranch] lap eigvals[:3]="
                f"{ev[:3].tolist()}, min={float(ev.min()):.4e}, "
                f"PE^T@PE max off-diag={float(offdiag):.4e}"
            )

    def forward(self, x):
        # x: (B, C, T)
        nf = self.node(self.stem(x))  # (B, C, d_hidden)

        if self.pe_proj is not None:
            pe = self.lap_pe
            if self.space_lap_pe_sign_flip and self.training:
                signs = torch.randint(
                    0, 2, (pe.size(1),), device=pe.device, dtype=pe.dtype
                ) * 2 - 1
                pe = pe * signs.unsqueeze(0)
            nf = nf + self.pe_proj(pe).unsqueeze(0)  # (1, C, d_hidden) broadcast

        if self.use_graph:
            if self.space_geom_mode == "innerproduct":
                Q = self.Wq(self.coords)      # (C, d_hidden)
                K = self.Wk(self.coords)      # (C, d_hidden)
                geo_logits = Q @ K.t() / math.sqrt(self.d_hidden)
            else:  # sphere_rbf
                sigma2 = torch.exp(2 * self.log_sigma)
                geo_logits = -(self.geo_dist ** 2) / (2 * sigma2 + 1e-8)

            if self.space_func_adj_enabled:
                z = F.normalize(nf, p=2, dim=-1, eps=1e-6)
                A_func = torch.bmm(z, z.transpose(1, 2)).mean(dim=0)
                A = torch.softmax(geo_logits + self.beta * A_func, dim=-1)
            else:
                A = torch.softmax(geo_logits, dim=-1)
            nf = nf + self.alpha * (A.unsqueeze(0) @ nf)

        if self.use_channel_importance:
            imp = torch.softmax(self.imp(nf).squeeze(-1), dim=-1)
        else:
            imp = torch.full(
                (x.size(0), self.n_ch),
                1.0 / self.n_ch,
                device=x.device,
                dtype=x.dtype,
            )
        pooled = (nf * imp.unsqueeze(-1)).sum(dim=1)

        e = self.out(pooled)
        # nf (B, C, d_hidden) exposed as node tokens for bottleneck fusion.
        return e, imp, nf


# --------------------------------------------------------------------------- #
# Cov branch: Riemannian tangent-space features → small MLP
# --------------------------------------------------------------------------- #
class CovBranch(nn.Module):
    def __init__(self, cov_dim: int, d_out: int = 64, dropout: float = 0.5):
        super().__init__()
        self.cov_dim = int(cov_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.cov_dim),
            nn.Dropout(dropout),
            nn.Linear(self.cov_dim, d_out),
            nn.ELU(),
        )

    def forward(self, cov_feat):
        # cov_feat: (B, cov_dim)
        return self.net(cov_feat)


# --------------------------------------------------------------------------- #
# Tri-domain encoder
# --------------------------------------------------------------------------- #
class TriDomainEncoder(nn.Module):
    def __init__(
        self,
        n_ch=59,
        fs=250,
        d=64,
        coords=None,
        time_d_model=64,
        time_pool=8,
        freq_windows=4,
        freq_taps=None,
        space_hidden=32,
        dropout=0.3,
        active_branches=("time", "freq", "space"),
        use_time_attn=True,
        use_band_attn=True,
        use_space_graph=True,
        use_channel_importance=True,
        time_pool_mode="mean",
        freq_bands=None,
        cov_dim=None,
        freq_pool_mode="flatten",
        freq_var_shrinkage=0.0,
        freq_tensor_rank_band=None,
        freq_tensor_rank_chan=16,
        freq_tensor_rank_win=None,
        freq_pac_enabled=False,
        freq_pac_amp_norm=True,
        freq_learnable_bands=False,
        freq_window_dynamics="none",
        space_func_adj_enabled=False,
        space_geom_mode="innerproduct",
        space_laplacian_pe=False,
        space_lap_pe_k=8,
        space_sphere_sigma_init=0.5,
        space_lap_pe_sign_flip=False,
    ):
        super().__init__()
        self.active_branches = tuple(active_branches)

        if "cov" in self.active_branches:
            if cov_dim is None:
                raise ValueError("cov_dim is required when 'cov' is in active_branches")
            self.cov = CovBranch(cov_dim=cov_dim, d_out=d, dropout=dropout)

        if "time" in self.active_branches:
            self.time = TimeBranch(
                n_ch=n_ch,
                fs=fs,
                d_model=time_d_model,
                d_out=d,
                pool=time_pool,
                dropout=min(dropout, 0.5),
                use_attn=use_time_attn,
                pool_mode=time_pool_mode,
            )

        if "freq" in self.active_branches:
            self.freq = FreqBranch(
                n_ch=n_ch,
                fs=fs,
                d_out=d,
                n_win=freq_windows,
                taps=freq_taps,
                dropout=dropout,
                use_band_attn=use_band_attn,
                bands=freq_bands,
                freq_pool_mode=freq_pool_mode,
                freq_var_shrinkage=freq_var_shrinkage,
                freq_tensor_rank_band=freq_tensor_rank_band,
                freq_tensor_rank_chan=freq_tensor_rank_chan,
                freq_tensor_rank_win=freq_tensor_rank_win,
                freq_pac_enabled=freq_pac_enabled,
                freq_pac_amp_norm=freq_pac_amp_norm,
                freq_learnable_bands=freq_learnable_bands,
                freq_window_dynamics=freq_window_dynamics,
            )

        if "space" in self.active_branches:
            self.space = SpaceBranch(
                n_ch=n_ch,
                coords=coords,
                d_hidden=space_hidden,
                d_out=d,
                use_graph=use_space_graph,
                use_channel_importance=use_channel_importance,
                space_func_adj_enabled=space_func_adj_enabled,
                space_geom_mode=space_geom_mode,
                space_laplacian_pe=space_laplacian_pe,
                space_lap_pe_k=space_lap_pe_k,
                space_sphere_sigma_init=space_sphere_sigma_init,
                space_lap_pe_sign_flip=space_lap_pe_sign_flip,
            )

    def forward(self, x, cov_feat=None):
        out = {}
        if "time" in self.active_branches:
            out["time"], out["time_tokens"] = self.time(x)
        else:
            out["time_tokens"] = None

        if "freq" in self.active_branches:
            out["freq"], out["band_weights"], out["freq_tokens"] = self.freq(x)
        else:
            out["band_weights"] = None
            out["freq_tokens"] = None

        if "space" in self.active_branches:
            out["space"], out["channel_importance"], out["space_tokens"] = self.space(x)
        else:
            out["channel_importance"] = None
            out["space_tokens"] = None

        if "cov" in self.active_branches:
            if cov_feat is None:
                raise ValueError("cov_feat is required when 'cov' is in active_branches")
            out["cov"] = self.cov(cov_feat)

        return out


# --------------------------------------------------------------------------- #
# Tri-domain classifier
# --------------------------------------------------------------------------- #
class TriDomainClassifier(nn.Module):
    def __init__(
        self,
        n_channels=59,
        n_classes=6,
        sample_rate=250,
        d=64,
        time_d_model=64,
        time_pool=8,
        freq_windows=4,
        freq_taps=None,
        space_hidden=32,
        classifier_hidden=128,
        dropout=0.3,
        coords=None,
        active_branches=("time", "freq", "space"),
        use_time_attn=True,
        use_band_attn=True,
        use_space_graph=True,
        use_channel_importance=True,
        per_branch_norm=False,
        aux_loss_enabled=False,
        time_pool_mode="mean",
        freq_bands=None,
        fusion_head_mode="mlp",
        cov_branch_enabled=False,
        cov_dim=None,
        modality_dropout_enabled=False,
        modality_dropout_p=0.2,
        freq_pool_mode="flatten",
        freq_var_shrinkage=0.0,
        freq_tensor_rank_band=None,
        freq_tensor_rank_chan=16,
        freq_tensor_rank_win=None,
        freq_pac_enabled=False,
        freq_pac_amp_norm=True,
        freq_learnable_bands=False,
        freq_window_dynamics="none",
        space_func_adj_enabled=False,
        space_geom_mode="innerproduct",
        space_laplacian_pe=False,
        space_lap_pe_k=8,
        space_sphere_sigma_init=0.5,
        space_lap_pe_sign_flip=False,
        branch_decorr_enabled=False,
        branch_decorr_weight=0.01,
        cross_branch_attn_enabled=False,
        cross_branch_attn_heads=4,
        cross_branch_attn_layers=1,
        cross_branch_attn_ff_mult=2,
        cross_branch_attn_dropout=None,
        cross_branch_attn_mode="branch_token",
        cross_branch_fusion_tokens=8,
        cp_rank=32,
    ):
        super().__init__()
        # Extend active_branches with 'cov' when enabled.
        if cov_branch_enabled and "cov" not in active_branches:
            active_branches = tuple(active_branches) + ("cov",)
        self.active_branches = tuple(active_branches)
        self.per_branch_norm = bool(per_branch_norm)
        self.aux_loss_enabled = bool(aux_loss_enabled)
        self.modality_dropout_enabled = bool(modality_dropout_enabled)
        self.modality_dropout_p = float(modality_dropout_p)
        self.branch_decorr_enabled = bool(branch_decorr_enabled)
        self.branch_decorr_weight = float(branch_decorr_weight)
        self.cp_rank = int(cp_rank)
        self.fusion_head_mode = str(fusion_head_mode).lower()
        if self.fusion_head_mode not in (
            "mlp", "gate", "gate_mlp", "poe", "poe_mlp", "lowrank_cp"
        ):
            raise ValueError(
                "fusion_head_mode must be one of "
                "mlp/gate/gate_mlp/poe/poe_mlp/lowrank_cp, "
                f"got {fusion_head_mode!r}"
            )
        if self.fusion_head_mode in ("gate", "gate_mlp", "poe", "poe_mlp") and not self.aux_loss_enabled:
            raise ValueError(
                "fusion_head_mode='gate'/'gate_mlp'/'poe'/'poe_mlp' requires "
                "aux_loss_enabled=True (fusion composes the aux head logits)"
            )

        self.encoder = TriDomainEncoder(
            n_ch=n_channels,
            fs=sample_rate,
            d=d,
            coords=coords,
            time_d_model=time_d_model,
            time_pool=time_pool,
            freq_windows=freq_windows,
            freq_taps=freq_taps,
            space_hidden=space_hidden,
            dropout=dropout,
            active_branches=self.active_branches,
            use_time_attn=use_time_attn,
            use_band_attn=use_band_attn,
            use_space_graph=use_space_graph,
            use_channel_importance=use_channel_importance,
            time_pool_mode=time_pool_mode,
            freq_bands=freq_bands,
            cov_dim=cov_dim,
            freq_pool_mode=freq_pool_mode,
            freq_var_shrinkage=freq_var_shrinkage,
            freq_tensor_rank_band=freq_tensor_rank_band,
            freq_tensor_rank_chan=freq_tensor_rank_chan,
            freq_tensor_rank_win=freq_tensor_rank_win,
            freq_pac_enabled=freq_pac_enabled,
            freq_pac_amp_norm=freq_pac_amp_norm,
            freq_learnable_bands=freq_learnable_bands,
            freq_window_dynamics=freq_window_dynamics,
            space_func_adj_enabled=space_func_adj_enabled,
            space_geom_mode=space_geom_mode,
            space_laplacian_pe=space_laplacian_pe,
            space_lap_pe_k=space_lap_pe_k,
            space_sphere_sigma_init=space_sphere_sigma_init,
            space_lap_pe_sign_flip=space_lap_pe_sign_flip,
        )

        head_in_dim = len(self.active_branches) * d

        # per-branch LayerNorm (Task 1.1). When enabled, drop the cat-level
        # LN to avoid double normalization. Disabled = bit-exact with current
        # behaviour (cat-level LN as before).
        if self.per_branch_norm:
            self.branch_norms = nn.ModuleDict({
                name: nn.LayerNorm(d) for name in self.active_branches
            })
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(head_in_dim, classifier_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, n_classes),
            )
        else:
            self.branch_norms = None
            self.head = nn.Sequential(
                nn.LayerNorm(head_in_dim),
                nn.Dropout(dropout),
                nn.Linear(head_in_dim, classifier_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, n_classes),
            )

        # per-branch auxiliary heads (Task 1.2). Only built when enabled;
        # at inference time the aux heads are simply ignored so the
        # deployment graph stays identical to the current model.
        if self.aux_loss_enabled:
            self.aux_heads = nn.ModuleDict({
                name: nn.Linear(d, n_classes) for name in self.active_branches
            })
        else:
            self.aux_heads = None

        # Per-sample confidence gate (Task 5 B2). Input = [max_softmax, entropy]
        # per active branch, so 2*K dims; output K weights via softmax.
        if self.fusion_head_mode in ("gate", "gate_mlp"):
            n_b = len(self.active_branches)
            self.gate_net = nn.Sequential(
                nn.Linear(2 * n_b, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, n_b),
            )
        else:
            self.gate_net = None

        if self.fusion_head_mode in ("poe", "poe_mlp"):
            self.poe_log_tau = nn.Parameter(torch.zeros(len(self.active_branches)))
        else:
            self.poe_log_tau = None

        # Cross-branch attention fusion (orthogonal to fusion_head_mode).
        # Built only when enabled AND there are >=2 branches to attend over.
        self.cross_branch_attn_enabled = bool(cross_branch_attn_enabled)
        self.cross_branch_attn_mode = str(cross_branch_attn_mode).lower()
        if self.cross_branch_attn_mode not in ("branch_token", "bottleneck"):
            raise ValueError(
                "cross_branch_attn_mode must be 'branch_token' or 'bottleneck', "
                f"got {cross_branch_attn_mode!r}"
            )
        self.cross_branch_fusion_tokens = int(cross_branch_fusion_tokens)
        K = len(self.active_branches)
        cb_dropout = (
            dropout if cross_branch_attn_dropout is None
            else float(cross_branch_attn_dropout)
        )
        # defaults (overwritten below per mode)
        self.branch_type_embed = None
        self.cross_attn_layers = None
        self.bottleneck = None

        if not (self.cross_branch_attn_enabled and K >= 2):
            self.branch_type_embed = None
            self.cross_attn_layers = None
            if self.cross_branch_attn_enabled and K < 2:
                print(f"[TriDomain] cross_branch_attn requested but K={K} < 2; disabled.")
            else:
                print("[TriDomain] cross_branch_attn disabled")
        elif self.cross_branch_attn_mode == "branch_token":
            assert d % int(cross_branch_attn_heads) == 0, (
                f"cross_branch_attn_heads ({cross_branch_attn_heads}) must divide d ({d})"
            )
            self.branch_type_embed = nn.Parameter(torch.zeros(1, K, d))
            nn.init.normal_(self.branch_type_embed, std=0.02)
            self.cross_attn_layers = nn.ModuleList()
            for _ in range(int(cross_branch_attn_layers)):
                layer = nn.Module()
                layer.attn = nn.MultiheadAttention(
                    d, int(cross_branch_attn_heads),
                    batch_first=True, dropout=cb_dropout,
                )
                layer.norm1 = nn.LayerNorm(d)
                layer.norm2 = nn.LayerNorm(d)
                layer.ff = nn.Sequential(
                    nn.Linear(d, int(cross_branch_attn_ff_mult) * d),
                    nn.ELU(),
                    nn.Dropout(cb_dropout),
                    nn.Linear(int(cross_branch_attn_ff_mult) * d, d),
                )
                self.cross_attn_layers.append(layer)
            cb_params = (
                self.branch_type_embed.numel()
                + sum(p.numel() for p in self.cross_attn_layers.parameters())
            )
            print(
                "[TriDomain] cross_branch_attn enabled (branch_token): "
                f"heads={cross_branch_attn_heads}, layers={cross_branch_attn_layers}, "
                f"ff_mult={cross_branch_attn_ff_mult}, K={K}, params={cb_params}"
            )
        else:  # bottleneck / Perceiver fusion on un-pooled tokens
            assert d % int(cross_branch_attn_heads) == 0, (
                f"cross_branch_attn_heads ({cross_branch_attn_heads}) must divide d ({d})"
            )
            bn = nn.Module()
            # per-branch input projections to d (raw token feature dims differ)
            tok_in = {
                "time": time_d_model,
                "freq": n_channels * freq_windows,
                "space": space_hidden,
            }
            bn.in_proj = nn.ModuleDict()
            bn.type_embed = nn.ParameterDict()
            self._bottleneck_branches = [
                b for b in self.active_branches if b in tok_in
            ]
            for b in self._bottleneck_branches:
                bn.in_proj[b] = nn.Linear(tok_in[b], d)
                p = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(p, std=0.02)
                bn.type_embed[b] = p
            M = self.cross_branch_fusion_tokens
            bn.fusion_query = nn.Parameter(torch.zeros(1, M, d))
            nn.init.normal_(bn.fusion_query, std=0.02)
            bn.layers = nn.ModuleList()
            for _ in range(int(cross_branch_attn_layers)):
                layer = nn.Module()
                layer.attn = nn.MultiheadAttention(
                    d, int(cross_branch_attn_heads),
                    batch_first=True, dropout=cb_dropout,
                )
                layer.norm_q = nn.LayerNorm(d)
                layer.norm_ff = nn.LayerNorm(d)
                layer.ff = nn.Sequential(
                    nn.Linear(d, int(cross_branch_attn_ff_mult) * d),
                    nn.ELU(),
                    nn.Dropout(cb_dropout),
                    nn.Linear(int(cross_branch_attn_ff_mult) * d, d),
                )
                bn.layers.append(layer)
            bn.head = nn.Sequential(
                nn.Linear(d, classifier_hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, n_classes),
            )
            self.bottleneck = bn
            self._bottleneck_logged = False
            bn_params = sum(p.numel() for p in bn.parameters())
            print(
                "[TriDomain] cross_branch_attn enabled (bottleneck): "
                f"fusion_tokens={M}, branches={self._bottleneck_branches}, "
                f"layers={cross_branch_attn_layers}, heads={cross_branch_attn_heads}, "
                f"params={bn_params}"
            )

        # Low-rank CP (PARAFAC) trilinear fusion (fusion_head_mode='lowrank_cp').
        # Project each branch embedding to rank R, elementwise-multiply across
        # branches => multiplicative cross-domain interaction, then a small head.
        # Same multilinear algebra as the freq Tucker contraction, lifted to the
        # time x freq x space domain level. Built only in this mode so the other
        # fusion modes stay bit-exact.
        if self.fusion_head_mode == "lowrank_cp":
            if K < 2:
                # Degenerate: fall back to the plain MLP head.
                self.fusion_head_mode = "mlp"
                self.cp_proj = None
                self.cp_head = None
                print(
                    f"[TriDomain] lowrank_cp requested but K={K} < 2; "
                    "falling back to mlp."
                )
            else:
                R = self.cp_rank
                self.cp_proj = nn.ModuleDict({
                    name: nn.Linear(d, R) for name in self.active_branches
                })
                self.cp_head = nn.Sequential(
                    nn.Linear(R, classifier_hidden),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(classifier_hidden, n_classes),
                )
                cp_params = (
                    sum(p.numel() for p in self.cp_proj.parameters())
                    + sum(p.numel() for p in self.cp_head.parameters())
                )
                print(
                    "[TriDomain] lowrank_cp fusion enabled: "
                    f"cp_rank={R}, K={K}, params={cp_params}"
                )
        else:
            self.cp_proj = None
            self.cp_head = None

        self.last_aux = None

    def _bottleneck_forward(self, out, branch_drop_mask, B):
        """Perceiver-style fusion: M learnable queries attend over the
        concatenated un-pooled internal tokens of all token-bearing branches."""
        bn = self.bottleneck
        branch_idx = {name: i for i, name in enumerate(self.active_branches)}
        toks, kpm_parts = [], []
        for b in self._bottleneck_branches:
            raw = out[b + "_tokens"]                       # branch internal tokens
            t = bn.in_proj[b](raw) + bn.type_embed[b]      # (B, Lb, d)
            if branch_drop_mask is not None:
                m = branch_drop_mask[:, branch_idx[b]]
                t = t * m.view(B, 1, 1)
                kpm_parts.append((m == 0).view(B, 1).expand(B, t.size(1)))
            toks.append(t)
        seq = torch.cat(toks, dim=1)                       # (B, Ltot, d)
        kpm = None
        if branch_drop_mask is not None:
            kpm = torch.cat(kpm_parts, dim=1)              # (B, Ltot) True=masked
            allm = kpm.all(dim=1)
            if allm.any():
                kpm = kpm.clone(); kpm[allm] = False
        q = bn.fusion_query.expand(B, -1, -1)              # (B, M, d)
        for layer in bn.layers:
            a, _ = layer.attn(q, seq, seq, key_padding_mask=kpm)
            q = layer.norm_q(q + a)
            q = layer.norm_ff(q + layer.ff(q))
        if not self._bottleneck_logged:
            print(f"[TriDomain][bottleneck] exposed token total length={seq.size(1)}, "
                  f"fusion_queries={q.size(1)}")
            self._bottleneck_logged = True
        return bn.head(q.mean(dim=1))                       # (B, n_classes)

    def forward(self, x, return_aux=False, cov_feat=None, branch_drop_mask=None):
        # Accept both (B, C, T) and EEGNet-style (B, 1, C, T)
        if x.dim() == 4:
            if x.size(1) != 1:
                raise ValueError(f"Expected x shape (B, 1, C, T), got {tuple(x.shape)}")
            x = x.squeeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}")

        out = self.encoder(x, cov_feat=cov_feat)

        if self.branch_norms is not None:
            branch_feats = [self.branch_norms[name](out[name])
                            for name in self.active_branches]
        else:
            branch_feats = [out[name] for name in self.active_branches]

        branch_decorr_loss = None
        if self.branch_decorr_enabled and self.training:
            if len(branch_feats) < 2:
                branch_decorr_loss = branch_feats[0].new_zeros(())
            else:
                z_feats = []
                for feat_i in branch_feats:
                    z = feat_i - feat_i.mean(dim=0, keepdim=True)
                    z = z / (feat_i.std(dim=0, unbiased=False, keepdim=True) + 1e-5)
                    z_feats.append(z)
                loss = branch_feats[0].new_zeros(())
                batch = max(int(branch_feats[0].size(0)), 1)
                for i in range(len(z_feats)):
                    for j in range(i + 1, len(z_feats)):
                        corr = z_feats[i].transpose(0, 1) @ z_feats[j] / batch
                        loss = loss + corr.pow(2).sum()
                branch_decorr_loss = loss * self.branch_decorr_weight

        if branch_drop_mask is not None:
            if branch_drop_mask.shape != (x.size(0), len(self.active_branches)):
                raise ValueError(
                    "branch_drop_mask must have shape "
                    f"(B, K)={(x.size(0), len(self.active_branches))}, "
                    f"got {tuple(branch_drop_mask.shape)}"
                )
            branch_drop_mask = branch_drop_mask.to(
                device=branch_feats[0].device,
                dtype=branch_feats[0].dtype,
            )
            head_branch_feats = [
                feat_i * branch_drop_mask[:, i:i + 1]
                for i, feat_i in enumerate(branch_feats)
            ]
        else:
            head_branch_feats = branch_feats

        # Cross-branch attention: only on the concat path fed to the main head.
        # aux_logits / gate / poe / decorr keep using the original branch_feats.
        if self.cross_attn_layers is not None:
            K = len(head_branch_feats)
            tokens = torch.stack(head_branch_feats, dim=1)        # (B, K, d)
            tokens = tokens + self.branch_type_embed[:, :K, :]
            kpm = None
            if branch_drop_mask is not None:
                kpm = (branch_drop_mask == 0)                     # (B, K) True=masked
                # rows fully masked -> unmask to avoid softmax NaN
                all_masked = kpm.all(dim=1)
                if all_masked.any():
                    kpm = kpm.clone()
                    kpm[all_masked] = False
            for layer in self.cross_attn_layers:
                a, _ = layer.attn(tokens, tokens, tokens, key_padding_mask=kpm)
                tokens = layer.norm1(tokens + a)
                tokens = layer.norm2(tokens + layer.ff(tokens))
            head_branch_feats = [tokens[:, i, :] for i in range(K)]

        feat = torch.cat(head_branch_feats, dim=1)

        if self.bottleneck is None:
            mlp_logits = self.head(feat)
        else:
            # Bottleneck / Perceiver fusion over un-pooled internal tokens.
            mlp_logits = self._bottleneck_forward(out, branch_drop_mask, x.size(0))

        aux_logits = None
        if self.aux_heads is not None:
            # Same input as the branch_feats used by the main head: if
            # per_branch_norm is on, aux heads sit after the LN as well,
            # so the aux objective normalises gradients consistently with
            # the main head's view of each branch.
            aux_logits = {
                name: self.aux_heads[name](branch_feats[i])
                for i, name in enumerate(self.active_branches)
            }

        gate_logits = None
        gate_weights = None
        if self.gate_net is not None:
            # Build per-branch [max_softmax_prob, entropy] descriptors from aux logits.
            descs = []
            aux_stack = []
            for name in self.active_branches:
                a = aux_logits[name]
                probs = torch.softmax(a, dim=-1)
                max_p = probs.max(dim=-1).values
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                descs.append(torch.stack([max_p, ent], dim=-1))  # (B,2)
                aux_stack.append(a)
            desc = torch.cat(descs, dim=-1)                  # (B, 2K)
            g_logits = self.gate_net(desc)                   # (B, K)
            gate_weights = torch.softmax(g_logits, dim=-1)   # (B, K)
            if branch_drop_mask is not None:
                gate_weights = gate_weights * branch_drop_mask
            aux_stack = torch.stack(aux_stack, dim=1)        # (B, K, n_classes)
            gate_logits = (aux_stack * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, n_classes)

        poe_logits = None
        if self.poe_log_tau is not None:
            scale = torch.exp(-self.poe_log_tau).to(
                device=mlp_logits.device,
                dtype=mlp_logits.dtype,
            )
            poe_terms = []
            for i, name in enumerate(self.active_branches):
                term = aux_logits[name] * scale[i]
                if branch_drop_mask is not None:
                    term = term * branch_drop_mask[:, i:i + 1]
                poe_terms.append(term)
            poe_logits = torch.stack(poe_terms, dim=0).sum(dim=0)
            # Optional class-prior correction for imbalanced classes would be:
            # poe_logits -= (K - 1) * log_prior. Classes are treated as roughly
            # balanced here, so the constant correction is intentionally omitted.

        cp_logits = None
        if self.cp_proj is not None:
            # Multiplicative CP fusion on the SAME branch features fed to the
            # main head (post per_branch_norm / branch_drop / cross_attn).
            us = [self.cp_proj[name](head_branch_feats[i])
                  for i, name in enumerate(self.active_branches)]
            cp = us[0]
            for u in us[1:]:
                cp = cp * u                      # (B, R) rank-wise product
            cp_logits = self.cp_head(cp)

        if self.fusion_head_mode == "mlp":
            logits = mlp_logits
        elif self.fusion_head_mode == "gate":
            logits = gate_logits
        elif self.fusion_head_mode == "gate_mlp":
            logits = 0.5 * mlp_logits + 0.5 * gate_logits
        elif self.fusion_head_mode == "poe":
            logits = poe_logits
        elif self.fusion_head_mode == "poe_mlp":
            logits = 0.5 * mlp_logits + 0.5 * poe_logits
        elif self.fusion_head_mode == "lowrank_cp":
            logits = cp_logits
        else:
            logits = mlp_logits

        aux = {
            "band_weights": out["band_weights"],
            "channel_importance": out["channel_importance"],
            "branch_embeds": {name: out[name] for name in self.active_branches},
            "aux_logits": aux_logits,
            "gate_weights": gate_weights,
            "mlp_logits": mlp_logits,
            "gate_logits": gate_logits,
            "poe_logits": poe_logits,
            "cp_logits": cp_logits,
            "branch_decorr_loss": branch_decorr_loss,
        }

        self.last_aux = aux

        if return_aux:
            return logits, aux

        return logits


# --------------------------------------------------------------------------- #
# Factory used by main.py
# --------------------------------------------------------------------------- #
def build_model(cfg, model_name="tridomain"):
    """
    Build TriDomainClassifier.

    Required for real spatial topology:
    - cfg.tri_elp_path
    - cfg.tri_channel_names
    - cfg.tri_normalize_coords
    """
    if model_name not in {"tridomain", "tri_domain", "tri-domain"}:
        raise ValueError(f"Unknown model: {model_name}")

    freq_taps = getattr(cfg, "tri_freq_taps", None)
    if freq_taps is not None and freq_taps <= 0:
        freq_taps = None

    ablation_cfg = resolve_ablation_config(cfg)
    active_branches = ablation_cfg["active_branches"]
    coords_mode = ablation_cfg["coords_mode"]
    use_time_attn = ablation_cfg["use_time_attn"]
    use_band_attn = ablation_cfg["use_band_attn"]
    use_space_graph = ablation_cfg["use_space_graph"]
    use_channel_importance = ablation_cfg["use_channel_importance"]

    space_geom_mode = ablation_cfg["space_geom_mode"]
    space_laplacian_pe = bool(ablation_cfg["space_laplacian_pe"])
    space_lap_pe_k = int(getattr(cfg, "space_lap_pe_k", 8))
    space_sphere_sigma_init = float(getattr(cfg, "space_sphere_sigma_init", 0.5))
    space_lap_pe_sign_flip = bool(getattr(cfg, "space_lap_pe_sign_flip", False))
    need_3d = (space_geom_mode == "sphere_rbf")

    coords = None
    if "space" in active_branches:
        if coords_mode == "std":
            if need_3d:
                coords = load_elp_3d_coords(
                    elp_path=getattr(cfg, "tri_elp_path", None),
                    ch_names=getattr(cfg, "tri_channel_names", None),
                    n_ch=cfg.n_channels,
                    normalize=getattr(cfg, "tri_normalize_coords", True),
                )
            else:
                coords = load_elp_2d_coords(
                    elp_path=getattr(cfg, "tri_elp_path", None),
                    ch_names=getattr(cfg, "tri_channel_names", None),
                    n_ch=cfg.n_channels,
                    normalize=getattr(cfg, "tri_normalize_coords", True),
                )
        elif coords_mode == "zero":
            if need_3d:
                raise ValueError(
                    "sphere_rbf requires non-degenerate coords; "
                    "coords_mode='zero' is invalid"
                )
            coords = torch.zeros(cfg.n_channels, 2)
        elif coords_mode == "random":
            dim = 3 if need_3d else 2
            generator = torch.Generator().manual_seed(getattr(cfg, "random_seed", 42))
            coords = torch.randn(cfg.n_channels, dim, generator=generator)
            if need_3d:
                coords = coords / (coords.norm(dim=1, keepdim=True) + 1e-8)
        else:
            raise ValueError("--tri_coords_mode must be one of: std, zero, random")

    print(
        "[TriDomain] "
        f"ablation={ablation_cfg['ablation']}, "
        f"active_branches={active_branches}, "
        f"coords_mode={coords_mode}, "
        f"use_time_attn={use_time_attn}, "
        f"use_band_attn={use_band_attn}, "
        f"use_space_graph={use_space_graph}, "
        f"use_channel_importance={use_channel_importance}"
    )
    if coords is not None:
        print("[TriDomain] coords shape:", tuple(coords.shape))
        print("[TriDomain] coords abs_sum:", float(coords.abs().sum()))
        print("[TriDomain] coords std:", coords.std(dim=0))

    return TriDomainClassifier(
        n_channels=cfg.n_channels,
        n_classes=cfg.n_classes,
        sample_rate=cfg.sample_rate,
        d=getattr(cfg, "tri_d", 64),
        time_d_model=getattr(cfg, "tri_time_d_model", 64),
        time_pool=getattr(cfg, "tri_time_pool", 8),
        freq_windows=getattr(cfg, "tri_freq_windows", 4),
        freq_taps=freq_taps,
        space_hidden=getattr(cfg, "tri_space_hidden", 32),
        classifier_hidden=getattr(cfg, "tri_classifier_hidden", 128),
        dropout=getattr(cfg, "dropout", 0.3),
        coords=coords,
        active_branches=active_branches,
        use_time_attn=use_time_attn,
        use_band_attn=use_band_attn,
        use_space_graph=use_space_graph,
        use_channel_importance=use_channel_importance,
        per_branch_norm=bool(getattr(cfg, "per_branch_norm", False)),
        aux_loss_enabled=bool(getattr(cfg, "aux_loss_enabled", False)),
        time_pool_mode=getattr(cfg, "time_pool_mode", "mean"),
        freq_bands=getattr(cfg, "freq_bands", None),
        fusion_head_mode=getattr(cfg, "fusion_head_mode", "mlp"),
        cov_branch_enabled=bool(getattr(cfg, "cov_branch_enabled", False)),
        cov_dim=getattr(cfg, "cov_dim", None),
        modality_dropout_enabled=bool(getattr(cfg, "modality_dropout_enabled", False)),
        modality_dropout_p=float(getattr(cfg, "modality_dropout_p", 0.2)),
        freq_pool_mode=ablation_cfg["freq_pool_mode"],
        freq_var_shrinkage=float(getattr(cfg, "freq_var_shrinkage", 0.0)),
        freq_tensor_rank_band=getattr(cfg, "freq_tensor_rank_band", None),
        freq_tensor_rank_chan=getattr(cfg, "freq_tensor_rank_chan", 16),
        freq_tensor_rank_win=getattr(cfg, "freq_tensor_rank_win", None),
        freq_pac_enabled=bool(getattr(cfg, "freq_pac_enabled", False)),
        freq_pac_amp_norm=bool(getattr(cfg, "freq_pac_amp_norm", True)),
        freq_learnable_bands=bool(getattr(cfg, "freq_learnable_bands", False)),
        freq_window_dynamics=getattr(cfg, "freq_window_dynamics", "none"),
        space_func_adj_enabled=bool(getattr(cfg, "space_func_adj_enabled", False)),
        space_geom_mode=space_geom_mode,
        space_laplacian_pe=space_laplacian_pe,
        space_lap_pe_k=space_lap_pe_k,
        space_sphere_sigma_init=space_sphere_sigma_init,
        space_lap_pe_sign_flip=space_lap_pe_sign_flip,
        branch_decorr_enabled=bool(getattr(cfg, "branch_decorr_enabled", False)),
        branch_decorr_weight=float(getattr(cfg, "branch_decorr_weight", 0.01)),
        cross_branch_attn_enabled=bool(ablation_cfg["cross_branch_attn_enabled"]),
        cross_branch_attn_heads=int(getattr(cfg, "cross_branch_attn_heads", 4)),
        cross_branch_attn_layers=int(getattr(cfg, "cross_branch_attn_layers", 1)),
        cross_branch_attn_ff_mult=int(getattr(cfg, "cross_branch_attn_ff_mult", 2)),
        cross_branch_attn_dropout=getattr(cfg, "cross_branch_attn_dropout", None),
        cross_branch_attn_mode=getattr(cfg, "cross_branch_attn_mode", "branch_token"),
        cross_branch_fusion_tokens=int(getattr(cfg, "cross_branch_fusion_tokens", 8)),
        cp_rank=int(getattr(cfg, "cp_rank", 32)),
    )

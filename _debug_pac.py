"""PAC debug: (0) bit-exact off, (1) synthetic coupling sanity, (2) scale check."""
import sys; sys.path.insert(0, ".")
import math
import torch
from types import SimpleNamespace
from ablation_config import TRI_DEFAULTS
from model import build_model, FreqBranch

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, C, T, ncls, fs = 2, 59, 1000, 6, 250

RECIPE = dict(
    tri_ablation="full_std_coords", tri_coords_mode="std",
    time_pool_mode="attn", aux_loss_enabled=True, aux_loss_weight=0.3,
    per_branch_norm=True, freq_bands="lowfreq_dense", tri_freq_taps=251,
)
def cfg(**o):
    v = dict(TRI_DEFAULTS); v.update(n_channels=C, n_classes=ncls, sample_rate=fs,
                                     n_samples=T, cov_dim=None); v.update(RECIPE); v.update(o)
    return SimpleNamespace(**v)
def build(seed, **o):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    return build_model(cfg(**o)).to(dev).eval()

# ----------------------------------------------------------------- #
# (0) bit-exact: PAC-off identical to baseline; PAC-on (zero-init) == off at step0
# ----------------------------------------------------------------- #
print("=== (0) bit-exact ===")
torch.set_grad_enabled(False)
x = torch.randn(B, 1, C, T, device=dev)
m_off = build(123, freq_pac_enabled=False)
m_on  = build(123, freq_pac_enabled=True)
# shared (non-PAC) params must be identical => PAC construction didn't move RNG
off_sd = {k: v for k, v in m_off.state_dict().items()}
shared_ok = all(torch.equal(off_sd[k], v) for k, v in m_on.state_dict().items()
                if k in off_sd)
print("  shared params identical (RNG not moved):", shared_ok)
# zero-init residual => identical logits at init
lo, ln = m_off(x), m_on(x)
print("  logits off==on at init (atol=0):", bool(torch.equal(lo, ln)))
assert shared_ok and torch.equal(lo, ln), "PAC-off not bit-exact / zero-init broken"

# ----------------------------------------------------------------- #
# (1) synthetic sanity: low-freq phase modulates high-freq amplitude
# ----------------------------------------------------------------- #
print("\n=== (1) synthetic PAC sanity (3 bands) ===")
fb = FreqBranch(n_ch=1, fs=fs, n_win=4, bands=[(1, 2), (2, 3), (3, 4)],
                freq_pac_enabled=True).to(dev).eval()
fb.freq_pac_amp_norm = False  # raw MVL for specificity test
t = torch.arange(T, device=dev, dtype=torch.float32) / fs
fp, fc = 6.0, 40.0
phase_band = torch.cos(2 * math.pi * fp * t)                       # carries phase fp
env = 1.0 + torch.cos(2 * math.pi * fp * t)                        # amp modulated at fp
amp_band = env * torch.cos(2 * math.pi * fc * t)                   # coupled high band
indep_band = torch.cos(2 * math.pi * fc * t)                       # constant-amp high band
sigs = [phase_band, amp_band, indep_band]
band_sigs = [s.view(1, 1, T) for s in sigs]
pac = fb._compute_pac(band_sigs)[0, 0]                             # (2*n_pairs,)
pairs = [(0, 1), (0, 2), (1, 2)]
print("  pair (phase_i, amp_j) -> |MVL|:")
mags = {}
for p, (i, j) in enumerate(pairs):
    z = (pac[2 * p] ** 2 + pac[2 * p + 1] ** 2).sqrt().item()
    mags[(i, j)] = z
    print(f"    {(i,j)}: |z|={z:.4f}")
coupled = mags[(0, 1)]
others = max(mags[(0, 2)], mags[(1, 2)])
print(f"  coupled (0,1)={coupled:.4f}  vs  max other={others:.4f}  ratio={coupled/(others+1e-9):.1f}x")
assert coupled > 5 * others, "expected coupling concentrated on (0,1) band-pair"
print("  [OK] coupling concentrated on the true band-pair")

# ----------------------------------------------------------------- #
# (2) scale check: raw PAC feature std vs freq embedding std
# ----------------------------------------------------------------- #
print("\n=== (2) scale check (real branch, lowfreq_dense) ===")
fbr = FreqBranch(n_ch=C, fs=fs, n_win=4, taps=251, bands="lowfreq_dense",
                 freq_pac_enabled=True, freq_pac_amp_norm=True).to(dev).eval()
xr = torch.randn(B, C, T, device=dev)
win = T // 4
bs = [fbr.filters[bi](xr)[..., : win * 4] for bi in range(fbr.n_bands)]
pac_raw = fbr._compute_pac(bs)                # (B,C,30)
pac_ln = fbr.pac_norm(pac_raw)
e_final, _ = fbr(xr)
print(f"  pac_raw per-dim std: min={pac_raw.std(dim=(0,1)).min():.4f} "
      f"max={pac_raw.std(dim=(0,1)).max():.4f} mean={pac_raw.std(dim=(0,1)).mean():.4f}")
print(f"  pac_ln  std (overall): {pac_ln.std():.4f}")
print(f"  freq embedding e std : {e_final.std():.4f}  (residual term is zero-init => 0 at step0)")

print("\nPAC DEBUG ALL PASSED")

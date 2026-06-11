"""Phase-1 smoke + bit-exact tests for PAC / learnable-sinc / window-dynamics."""
import sys; sys.path.insert(0, ".")
import torch
from types import SimpleNamespace
from ablation_config import TRI_DEFAULTS
from model import build_model

torch.set_grad_enabled(False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, C, T, ncls = 2, 59, 1000, 6
x = torch.randn(B, 1, C, T, device=dev)

# baseline recipe = abl_S5_S_swa (T7) so tests match the real baseline
RECIPE = dict(
    tri_ablation="full_std_coords", tri_coords_mode="std",
    time_pool_mode="attn", aux_loss_enabled=True, aux_loss_weight=0.3,
    per_branch_norm=True, freq_bands="lowfreq_dense", tri_freq_taps=251,
)

def cfg(**over):
    v = dict(TRI_DEFAULTS); v.update(n_channels=C, n_classes=ncls, sample_rate=250,
                                     n_samples=T, cov_dim=None)
    v.update(RECIPE); v.update(over)
    return SimpleNamespace(**v)

def make(seed=0, **over):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return build_model(cfg(**over)).to(dev).eval()

def chk(name, **over):
    m = make(**over); out = m(x)
    assert out.shape == (B, ncls) and torch.isfinite(out).all(), name
    print(f"  [OK] {name}: logits {tuple(out.shape)} finite")

print("=== smoke ===")
chk("baseline (all off)")
chk("PAC", freq_pac_enabled=True)
chk("PAC+amp_norm", freq_pac_enabled=True, freq_pac_amp_norm=True)
chk("learnable_bands", freq_learnable_bands=True)
chk("wdyn=dwconv", freq_window_dynamics="dwconv")
chk("wdyn=gru", freq_window_dynamics="gru")
chk("wdyn=attn", freq_window_dynamics="attn")
chk("PAC+learnable+gru", freq_pac_enabled=True, freq_learnable_bands=True,
    freq_window_dynamics="gru")

print("=== bit-exact (default off) ===")
a = make(seed=7); b = make(seed=7)
assert torch.allclose(a(x), b(x), atol=0), "default not bit-exact"
print("  [OK] baseline determinism atol=0")

print("=== gradients flow (train, learnable+pac) ===")
torch.set_grad_enabled(True)
m = make(freq_learnable_bands=True, freq_pac_enabled=True,
         freq_window_dynamics="gru").train()
loss = m(x).sum(); loss.backward()
g_lo = m.encoder.freq.band_p_lo.grad
g_pac = m.encoder.freq.pac_proj[0].weight.grad
assert g_lo is not None and torch.isfinite(g_lo).all(), "no grad to band_p_lo"
assert g_pac is not None and torch.isfinite(g_pac).all(), "no grad to pac_proj"
print("  [OK] grads finite to band_p_lo and pac_proj")

print("\nPHASE-1 ALL PASSED")

import torch

from ablation_config import ABLATION_PRESETS, TRI_DEFAULTS
from model import build_model
from types import SimpleNamespace


def main():
    x = torch.randn(2, 59, 1000)
    for name, preset in ABLATION_PRESETS.items():
        cfg = dict(TRI_DEFAULTS)
        cfg.update(preset)
        cfg.update({"n_channels": 59, "n_samples": 1000, "n_classes": 6})
        model = build_model(SimpleNamespace(**cfg), model_name="tridomain")
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (2, 6), (name, tuple(y.shape))
        print(f"[quick forward ok] {name}: {tuple(y.shape)}")
    print(f"ALL {len(ABLATION_PRESETS)} QUICK FORWARD TESTS PASSED")


if __name__ == "__main__":
    main()

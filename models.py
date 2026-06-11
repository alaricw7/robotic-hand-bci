try:
    from .model import build_model as _build_tridomain
except ImportError:
    from model import build_model as _build_tridomain


MODEL_NAMES = ["tridomain"]


def build_model(name: str, cfg):
    """Build a TriDomain classifier from a SimpleNamespace-style cfg.

    The TriDomain encoder needs more than (n_channels, n_times, n_classes)
    because it consumes electrode coordinates, ablation switches, etc.,
    so we pass through the full cfg.
    """
    if name.lower() not in {"tridomain", "tri_domain", "tri-domain"}:
        raise ValueError(f"unknown model: {name}")
    return _build_tridomain(cfg, model_name="tridomain")

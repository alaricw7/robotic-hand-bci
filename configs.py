"""Default training configuration for the TriDomain classifier.

Mirrors the EEGNet training defaults used by the parent check_experiment
framework so the baseline runs stay comparable.
"""

TRIDOMAIN_CONFIG = {
    "optimizer": "adam",
    "betas": (0.9, 0.999),
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 200,
}


def get_config(name: str = "tridomain") -> dict:
    if name.lower() not in {"tridomain", "tri_domain", "tri-domain"}:
        raise ValueError(f"unknown model: {name}")
    return dict(TRIDOMAIN_CONFIG)

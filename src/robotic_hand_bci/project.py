from __future__ import annotations

from pathlib import Path


PROJECT_MARKERS = ("pyproject.toml", "README.md")


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    raise FileNotFoundError("Could not locate project root from current path.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_EEG_DIR = DATA_DIR / "raw" / "EEG"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"
MONTAGE_DIR = ASSETS_DIR / "montages"
DEFAULT_MONTAGE_PATH = MONTAGE_DIR / "Standard-10-5-Cap385_witheog.elp"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_PREPROCESS_CONFIG = CONFIGS_DIR / "preprocessing" / "eegnet_noica.toml"

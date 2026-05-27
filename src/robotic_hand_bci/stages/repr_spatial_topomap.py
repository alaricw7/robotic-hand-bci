from __future__ import annotations

from robotic_hand_bci.representation.eegnet_spatial_topomap import main as target_main
from robotic_hand_bci.stage_config import run_configured_stage


DEFAULT_CONFIG = "configs/representation/spatial_topomap.toml"


def main(argv: list[str] | None = None) -> None:
    run_configured_stage(default_config=DEFAULT_CONFIG, target=target_main, argv=argv)


if __name__ == "__main__":
    main()

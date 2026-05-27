from __future__ import annotations

from robotic_hand_bci.representation.refix_filter_grid import main as target_main
from robotic_hand_bci.stage_config import run_configured_stage


DEFAULT_CONFIG = "configs/representation/refix_filter_grid.toml"


def main(argv: list[str] | None = None) -> None:
    run_configured_stage(default_config=DEFAULT_CONFIG, target=target_main, argv=argv)


if __name__ == "__main__":
    main()

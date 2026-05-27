from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any, Callable

from robotic_hand_bci.project import PROJECT_ROOT


def _config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_stage_args(config_path: str | Path, section: str = "args") -> list[str]:
    path = _config_path(config_path)
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    args = payload.get(section, {})
    cli_args: list[str] = []
    for key, value in args.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            if value:
                cli_args.append(flag)
                cli_args.extend(str(item) for item in value)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def run_configured_stage(
    *,
    default_config: str | Path,
    target: Callable[[list[str] | None], None],
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=default_config)
    known, remainder = parser.parse_known_args(argv)
    target(load_stage_args(known.config) + remainder)

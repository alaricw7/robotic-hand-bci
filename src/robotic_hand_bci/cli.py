from __future__ import annotations

import argparse

from robotic_hand_bci.stages import (
    detect_marker_and_rest,
    prepare_eegnet_noica,
    repr_multifeature_rsa,
    repr_refix_filter_grid,
    repr_spatial_filters,
    repr_spatial_topomap,
    repr_topography_physionet,
    repr_tsne_rdm,
    repr_tsne_rdm_block,
)


STAGE_RUNNERS = {
    "prepare-eegnet-noica": prepare_eegnet_noica.main,
    "detect-marker-and-rest": detect_marker_and_rest.main,
    "repr-tsne-rdm": repr_tsne_rdm.main,
    "repr-tsne-rdm-block": repr_tsne_rdm_block.main,
    "repr-spatial-topomap": repr_spatial_topomap.main,
    "repr-multifeature-rsa": repr_multifeature_rsa.main,
    "repr-topography-physionet": repr_topography_physionet.main,
    "repr-spatial-filters": repr_spatial_filters.main,
    "repr-refix-filter-grid": repr_refix_filter_grid.main,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhbci",
        description="Stage-first entrypoint for robotic-hand-bci workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a configured project stage.")
    run_parser.add_argument(
        "stage",
        choices=sorted(STAGE_RUNNERS),
        help="Stage name to execute.",
    )
    run_parser.add_argument(
        "stage_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the stage entrypoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        stage_args = list(args.stage_args)
        if stage_args and stage_args[0] == "--":
            stage_args = stage_args[1:]
        STAGE_RUNNERS[args.stage](stage_args)


if __name__ == "__main__":
    main()

"""
Run the existing 10-fold pipeline on data/processed/pythondata1 NPZ outputs.

This wrapper keeps the original 10fold code unchanged. It converts files like
S1/S1_EEGNet_NoICA_uV.npz into subject_01.npz files that data.py already knows
how to load, preserving the NPZ uV scale instead of reading the EEGLAB .set
exports.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = str(PROJECT_ROOT / "data" / "processed" / "pythondata1")
DEFAULT_N_SAMPLES = 1126


def _has_option(passthrough, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in passthrough)


def _default_prepared_root(input_root):
    input_root = Path(input_root)
    return input_root.parent / f"_{input_root.name}_10fold_npz"


def _subject_id_from_dir(subject_dir):
    name = subject_dir.name
    if not name.lower().startswith("s") or not name[1:].isdigit():
        return None
    return int(name[1:])


def _normalize_subject_value(value):
    if value == "all":
        return value

    normalized = []
    for item in value.split(","):
        item = item.strip()
        if item.lower().startswith("s") and item[1:].isdigit():
            item = item[1:]
        normalized.append(item)
    return ",".join(normalized)


def _normalize_passthrough_subject(passthrough):
    normalized = []
    i = 0
    while i < len(passthrough):
        arg = passthrough[i]
        if arg == "--subject" and i + 1 < len(passthrough):
            normalized.extend([arg, _normalize_subject_value(passthrough[i + 1])])
            i += 2
            continue
        if arg.startswith("--subject="):
            key, value = arg.split("=", 1)
            normalized.append(f"{key}={_normalize_subject_value(value)}")
            i += 1
            continue
        normalized.append(arg)
        i += 1
    return normalized


def _subject_ids_from_passthrough(passthrough):
    for i, arg in enumerate(passthrough):
        value = None
        if arg == "--subject" and i + 1 < len(passthrough):
            value = passthrough[i + 1]
        elif arg.startswith("--subject="):
            value = arg.split("=", 1)[1]

        if value is None or value == "all":
            continue

        return {int(item) for item in value.split(",") if item}

    return None


def _match_samples(X, n_samples):
    if X.shape[2] == n_samples:
        return X
    if X.shape[2] > n_samples:
        extra = X.shape[2] - n_samples
        print(f"[prepare] crop first {extra} samples: {X.shape[2]} -> {n_samples}")
        return X[..., -n_samples:]
    if n_samples - X.shape[2] == 1:
        print(f"[prepare] pad last sample: {X.shape[2]} -> {n_samples}")
        return np.pad(X, ((0, 0), (0, 0), (0, 1)), mode="edge")
    raise ValueError(f"Cannot safely convert {X.shape[2]} samples to {n_samples}.")


def prepare_npz(input_root, prepared_root, n_samples, subject_ids=None, npz_variant="NoICA"):
    input_root = Path(input_root)
    prepared_root = Path(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    written = []
    for subject_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        subject_id = _subject_id_from_dir(subject_dir)
        if subject_id is None:
            continue
        if subject_ids is not None and subject_id not in subject_ids:
            continue

        npz_path = subject_dir / f"{subject_dir.name}_EEGNet_{npz_variant}_uV.npz"
        if not npz_path.exists():
            matches = sorted(subject_dir.glob(f"*_EEGNet_{npz_variant}_uV.npz"))
            if not matches:
                print(f"[prepare] skip {subject_dir}: no *_EEGNet_{npz_variant}_uV.npz")
                continue
            npz_path = matches[0]

        data = np.load(npz_path, allow_pickle=True)
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)

        if X.ndim != 3:
            raise ValueError(f"{npz_path} X must be (trials, channels, samples), got {X.shape}.")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError(f"{npz_path} y shape {y.shape} does not match X trials {X.shape[0]}.")

        X = _match_samples(X, n_samples).astype(np.float32, copy=False)

        out_path = prepared_root / f"subject_{subject_id:02d}.npz"
        np.savez_compressed(out_path, X=X, y=y)
        counts = dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))
        print(f"[prepare] {npz_path} -> {out_path}")
        print(f"[prepare]   X={X.shape} {X.dtype}, y={y.shape} {y.dtype}, class_counts={counts}")
        print(f"[prepare]   min={float(np.nanmin(X)):.6g}, max={float(np.nanmax(X)):.6g}")
        written.append(out_path)

    if not written:
        raise FileNotFoundError(f"No pythondata1 NPZ subjects found under {input_root}.")
    return prepared_root, written


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pythondata1 NPZ files and run the existing 10fold main.py."
    )
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--prepared_root",
        default=None,
        help="Intermediate subject_XX.npz output root. Defaults to _<input_root_name>_10fold_npz.",
    )
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--npz_variant",
        default="NoICA",
        help="NPZ filename variant, e.g. NoICA or ICA for S*_EEGNet_<variant>_uV.npz.",
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument(
        "--checkpoint_tag",
        default="noica",
        choices=["noica", "ica"],
        help=(
            "Tag for exported representation checkpoint. Defaults to noica, producing "
            "<exp_dir>/representation/checkpoints/S*_eegnet_noica.pt."
        ),
    )
    parser.add_argument(
        "--no_export_repr_checkpoint",
        action="store_true",
        help="Do not auto-export fold-0 checkpoint for representation scripts.",
    )
    args, passthrough = parser.parse_known_args()

    passthrough = _normalize_passthrough_subject(passthrough)
    subject_ids = _subject_ids_from_passthrough(passthrough)
    prepared_root_arg = args.prepared_root or _default_prepared_root(args.input_root)
    print(f"[prepare] input_root    = {args.input_root}")
    print(f"[prepare] prepared_root = {prepared_root_arg}")
    if subject_ids is not None:
        print(f"[prepare] subjects      = {sorted(subject_ids)}")
    prepared_root, _ = prepare_npz(
        args.input_root,
        prepared_root_arg,
        args.n_samples,
        subject_ids=subject_ids,
        npz_variant=args.npz_variant,
    )
    if args.prepare_only:
        print(f"[prepare] done. Prepared data root: {prepared_root}")
        return

    if not _has_option(passthrough, "--n_folds"):
        passthrough.extend(["--n_folds", "10"])

    if not _has_option(passthrough, "--exp_name"):
        passthrough.extend(["--exp_name", "pythondata1_npz_repr"])

    if not args.no_export_repr_checkpoint and not _has_option(
        passthrough, "--representation_checkpoint_tag"
    ):
        passthrough.extend(
            [
                "--representation_checkpoint_tag",
                args.checkpoint_tag,
                "--representation_checkpoint_fold",
                "0",
            ]
        )

    script_dir = Path(__file__).resolve().parent
    main_py = script_dir / "main.py"
    cmd = [
        sys.executable,
        str(main_py),
        "--data_root",
        str(prepared_root),
        "--n_samples",
        str(args.n_samples),
        *passthrough,
    ]
    print("[run]", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MNE_DONTWRITE_HOME", "true")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    raise SystemExit(subprocess.call(cmd, cwd=str(script_dir), env=env))


if __name__ == "__main__":
    main()

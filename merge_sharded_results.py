import argparse
import json
import os
from pathlib import Path

import numpy as np

from data import DATA_ROOT, SUBJECTS
from formatter import format_summary


def _avg_excl(per_subj, excl=("S2", "S3")):
    keep = [s for s in per_subj if s not in excl]
    if not keep:
        return None
    accs = np.asarray([per_subj[s]["acc"] for s in keep], dtype=float)
    ks = np.asarray([per_subj[s]["kappa"] for s in keep], dtype=float)
    return float(accs.mean()), float(accs.std()), float(ks.mean()), float(ks.std()), keep


def _avg_reliability(per_subj):
    keys = sorted({
        key
        for result in per_subj.values()
        for key in result.get("reliability", {})
    })
    out = {}
    for key in keys:
        vals = [
            result["reliability"][key]
            for result in per_subj.values()
            if key in result.get("reliability", {})
        ]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _link_ckpts(results_root, exp_name, parts):
    merged_ckpt = Path(results_root) / exp_name / "ckpts"
    merged_ckpt.mkdir(parents=True, exist_ok=True)
    for part in parts:
        part_ckpt = Path(results_root) / part / "ckpts"
        if not part_ckpt.exists():
            continue
        for subject_dir in part_ckpt.iterdir():
            if not subject_dir.is_dir():
                continue
            target = merged_ckpt / subject_dir.name
            if target.exists() or target.is_symlink():
                continue
            os.symlink(subject_dir.resolve(), target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/aug_hpo")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--parts", nargs="+", required=True)
    args = p.parse_args()

    merged = None
    per_subject = {}
    for part in args.parts:
        path = Path(args.results_root) / part / "results.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if merged is None:
            merged = dict(data)
        for subject, result in data["per_subject"].items():
            per_subject[subject] = result

    ordered = {s: per_subject[s] for s in SUBJECTS if s in per_subject}
    missing = [s for s in SUBJECTS if s not in ordered]
    if missing:
        raise RuntimeError(f"missing subjects for {args.exp_name}: {missing}")

    excl = _avg_excl(ordered)
    reliability_avg = _avg_reliability(ordered)
    n_classes = len(next(iter(ordered.values()))["per_class"])
    extra = [
        f"Ablation: {merged['ablation']}",
        f"N folds: {merged['n_folds']}",
        f"Aug config: {merged['aug_config']}",
    ]
    if reliability_avg:
        extra.append("Reliability means: " + " ".join(
            f"{k}={v:.4f}" for k, v in reliability_avg.items()
        ))
    if excl is not None:
        a_m, a_s, k_m, k_s, keep = excl
        extra.append(f"AVG excl S2/S3 ({','.join(keep)}): "
                     f"acc={a_m:.4f}+-{a_s:.4f}  kappa={k_m:.4f}+-{k_s:.4f}")

    summary = format_summary(
        experiment=args.exp_name,
        model_name=merged["model"],
        cv=merged["cv"],
        validation_desc=f"outer stratified {merged['n_folds']}-fold test; "
                        f"stratified val split from train_val",
        metric_desc="fold test set, checkpoint selected by val kappa "
                    f"(test protocol: {merged['aug_config'].get('test_protocol', 'full_trial')})",
        val_size=merged["val_size"],
        data_root=DATA_ROOT,
        n_classes=n_classes,
        per_subject_results=ordered,
        extra_lines=extra,
    )

    out_dir = Path(args.results_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    merged.update({
        "experiment": args.exp_name,
        "per_subject": ordered,
        "source_parts": args.parts,
        "reliability_means": reliability_avg,
        "avg_excl_s2_s3": (None if excl is None else {
            "subjects": excl[4],
            "acc_mean": excl[0],
            "acc_std": excl[1],
            "kappa_mean": excl[2],
            "kappa_std": excl[3],
        }),
    })
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    _link_ckpts(args.results_root, args.exp_name, args.parts)
    print(summary)
    print(f"saved: {out_dir / 'summary.txt'}")
    print(f"saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

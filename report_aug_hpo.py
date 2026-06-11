"""Compose results/aug_hpo/AUG_HPO_REPORT.md from a chain of aug runs.

Reads:
  results/abl_full_std_coords/results.json     (baseline anchor)
  results/aug_hpo/<each subdir>/results.json   (each aug step)

Usage:
  python report_aug_hpo.py \
    --baseline results/abl_full_std_coords \
    --steps results/aug_hpo/aug_repro \
            results/aug_hpo/aug_crops \
            results/aug_hpo/aug_crops_freqmask \
            results/aug_hpo/aug_crops_freqmask_chdrop \
            results/aug_hpo/aug_crops_freqmask_chdrop_noise \
            results/aug_hpo/aug_full_best \
    --final results/aug_hpo/aug_final_10fold
"""

import argparse
import json
from pathlib import Path


def _load(path):
    p = Path(path) / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _avg(results):
    if results is None:
        return None
    accs = [s["acc"] for s in results["per_subject"].values()]
    kappas = [s["kappa"] for s in results["per_subject"].values()]
    import numpy as np
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "kappa_mean": float(np.mean(kappas)),
        "kappa_std": float(np.std(kappas)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/abl_full_std_coords")
    p.add_argument("--steps", nargs="+", required=True)
    p.add_argument("--final", default=None)
    p.add_argument("--out", default="results/aug_hpo/AUG_HPO_REPORT.md")
    args = p.parse_args()

    lines = []
    lines.append("# TriDomain augmentation + HPO report")
    lines.append("")

    base = _load(args.baseline)
    base_agg = _avg(base)
    lines.append("## Baseline anchor (abl_full_std_coords, 10fold, val=0.1, seed=42)")
    if base_agg:
        lines.append(f"- acc = {base_agg['acc_mean']:.4f} ± {base_agg['acc_std']:.4f}")
        lines.append(f"- kappa = {base_agg['kappa_mean']:.4f} ± {base_agg['kappa_std']:.4f}")
    lines.append("")

    lines.append("## Incremental augmentation (single seed)")
    lines.append("")
    lines.append("| step | exp_name | acc | Δ acc | kappa | Δ kappa | n_folds | test_protocol |")
    lines.append("|------|----------|-----|-------|-------|---------|---------|---------------|")
    prev_acc = base_agg["acc_mean"] if base_agg else None
    prev_k = base_agg["kappa_mean"] if base_agg else None
    for step in args.steps:
        r = _load(step)
        if r is None:
            lines.append(f"| ? | {step} | (missing) |  |  |  |  |  |")
            continue
        agg = _avg(r)
        d_acc = agg["acc_mean"] - prev_acc if prev_acc is not None else 0.0
        d_k = agg["kappa_mean"] - prev_k if prev_k is not None else 0.0
        proto = r.get("aug_config", {}).get("test_protocol", "full_trial")
        lines.append(f"| | {Path(step).name} | {agg['acc_mean']:.4f}±{agg['acc_std']:.4f} "
                     f"| {d_acc:+.4f} | {agg['kappa_mean']:.4f}±{agg['kappa_std']:.4f} "
                     f"| {d_k:+.4f} | {r.get('n_folds','?')} | {proto} |")
        prev_acc = agg["acc_mean"]
        prev_k = agg["kappa_mean"]
    lines.append("")

    if args.final:
        f = _load(args.final)
        if f is not None:
            fagg = _avg(f)
            d_acc = (fagg["acc_mean"] - base_agg["acc_mean"]) if base_agg else 0.0
            d_k = (fagg["kappa_mean"] - base_agg["kappa_mean"]) if base_agg else 0.0
            lines.append("## Final selected config — full 10-fold")
            lines.append(f"- acc = {fagg['acc_mean']:.4f} ± {fagg['acc_std']:.4f}  "
                         f"(Δ vs baseline = {d_acc:+.4f})")
            lines.append(f"- kappa = {fagg['kappa_mean']:.4f} ± {fagg['kappa_std']:.4f}  "
                         f"(Δ vs baseline = {d_k:+.4f})")
            if f.get("avg_excl_s2_s3"):
                e = f["avg_excl_s2_s3"]
                lines.append(f"- AVG excl S2/S3: acc = {e['acc_mean']:.4f} ± {e['acc_std']:.4f}, "
                             f"kappa = {e['kappa_mean']:.4f} ± {e['kappa_std']:.4f}")
            lines.append("")
            lines.append("Aug config used:")
            lines.append("```json")
            lines.append(json.dumps(f.get("aug_config", {}), indent=2))
            lines.append("```")
            lines.append("Train overrides:")
            lines.append("```json")
            lines.append(json.dumps(f.get("train_config", {}), indent=2))
            lines.append("```")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

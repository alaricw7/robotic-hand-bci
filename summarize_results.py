import argparse
import json
from pathlib import Path

import numpy as np


def _fmt_pm(mean, std):
    return f"{float(mean):.4f}+-{float(std):.4f}"


def summarize_run(results_root, run):
    path = Path(results_root) / run / "results.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_subject = data["per_subject"]
    subjects = list(per_subject)
    accs = np.asarray([per_subject[s]["acc"] for s in subjects], dtype=float)
    kappas = np.asarray([per_subject[s]["kappa"] for s in subjects], dtype=float)
    excl = data.get("avg_excl_s2_s3") or {}

    row = {
        "run": run,
        "acc_all10": _fmt_pm(accs.mean(), accs.std()),
        "kappa_all10": _fmt_pm(kappas.mean(), kappas.std()),
        "acc_excl_s2_s3": _fmt_pm(excl.get("acc_mean", np.nan), excl.get("acc_std", np.nan)),
        "kappa_excl_s2_s3": _fmt_pm(excl.get("kappa_mean", np.nan), excl.get("kappa_std", np.nan)),
    }
    row.update(data.get("reliability_means") or {})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/aug_hpo")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = [summarize_run(args.results_root, run) for run in args.runs]
    reliability_keys = sorted({
        key
        for row in rows
        for key in row
        if key.startswith("mean_w_") or key.startswith("mean_u_")
    })
    columns = [
        "run",
        "acc_all10",
        "kappa_all10",
        "acc_excl_s2_s3",
        "kappa_excl_s2_s3",
    ] + reliability_keys

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}"
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()

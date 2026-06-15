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
    if "POOLED" in per_subject:
        result = per_subject["POOLED"]
        row = {
            "run": run,
            "acc": float(result["acc"]),
            "kappa": float(result["kappa"]),
            "best_val_kappa": float(result.get("best_val_kappa", np.nan)),
            "train": int(result.get("counts", {}).get("train", 0)),
            "val": int(result.get("counts", {}).get("val", 0)),
            "test": int(result.get("counts", {}).get("test", 0)),
        }
        return row

    subjects = list(per_subject)
    accs = np.asarray([per_subject[s]["acc"] for s in subjects], dtype=float)
    kappas = np.asarray([per_subject[s]["kappa"] for s in subjects], dtype=float)
    row = {
        "run": run,
        "acc": float(accs.mean()),
        "kappa": float(kappas.mean()),
        "best_val_kappa": np.nan,
        "train": 0,
        "val": 0,
        "test": 0,
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/baseline")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = [summarize_run(args.results_root, run) for run in args.runs]
    columns = ["run", "acc", "kappa", "best_val_kappa", "train", "val", "test"]

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
    if len(rows) > 1:
        accs = np.asarray([r["acc"] for r in rows], dtype=float)
        kappas = np.asarray([r["kappa"] for r in rows], dtype=float)
        bests = np.asarray([r["best_val_kappa"] for r in rows], dtype=float)
        avg_row = {
            "run": "AVG",
            "acc": _fmt_pm(accs.mean(), accs.std()),
            "kappa": _fmt_pm(kappas.mean(), kappas.std()),
            "best_val_kappa": _fmt_pm(np.nanmean(bests), np.nanstd(bests)),
            "train": rows[0].get("train", 0),
            "val": rows[0].get("val", 0),
            "test": rows[0].get("test", 0),
        }
        vals = [str(avg_row.get(col, "")) for col in columns]
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

import argparse
import json
from pathlib import Path

import numpy as np


def _fmt(value):
    return f"{float(value):.4f}"


def _fmt_pm(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return _fmt(values[0])
    return f"{float(values.mean()):.4f}+-{float(values.std()):.4f}"


def _resolve_result(results_root, item):
    path = Path(item)
    if path.is_file():
        return path
    return Path(results_root) / item / "results.json"


def _load_pooled(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pooled = data.get("per_subject", {}).get("POOLED")
    if not pooled:
        raise ValueError(f"{path} is not a pooled result")
    by_subject = pooled.get("by_subject")
    if not by_subject:
        raise ValueError(
            f"{path} has no POOLED.by_subject. Rerun with the updated run_baseline.py."
        )
    return data, pooled, by_subject


def _weakest_class(per_class):
    arr = np.asarray(per_class, dtype=float)
    idx = int(np.nanargmin(arr))
    return idx, float(arr[idx])


def build_report(results_root, runs):
    subject_rows = {}
    subject_class = {}
    pooled_class = []
    loaded = []

    for item in runs:
        path = _resolve_result(results_root, item)
        data, pooled, by_subject = _load_pooled(path)
        loaded.append(data.get("experiment", path.parent.name))
        pooled_class.append(np.asarray(pooled["per_class"], dtype=float))
        for subject, row in by_subject.items():
            subject_rows.setdefault(subject, {"acc": [], "kappa": [], "test": []})
            subject_rows[subject]["acc"].append(float(row["acc"]))
            subject_rows[subject]["kappa"].append(float(row["kappa"]))
            subject_rows[subject]["test"].append(int(row.get("counts", {}).get("test", 0)))
            subject_class.setdefault(subject, []).append(
                np.asarray(row["per_class"], dtype=float)
            )

    lines = []
    lines.append("# Pooled Subject Analysis")
    lines.append("")
    lines.append("Runs: " + ", ".join(loaded))
    lines.append("")

    ranked_subjects = sorted(
        subject_rows,
        key=lambda s: (
            float(np.mean(subject_rows[s]["kappa"])),
            float(np.mean(subject_rows[s]["acc"])),
        ),
    )
    lines.append("## Weak Subjects")
    lines.append("")
    lines.append("| rank | subject | acc | kappa | weakest_class | weakest_class_acc | test |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rank, subject in enumerate(ranked_subjects, start=1):
        class_mean = np.stack(subject_class[subject], axis=0).mean(axis=0)
        cls_idx, cls_value = _weakest_class(class_mean)
        test_count = int(round(float(np.mean(subject_rows[subject]["test"]))))
        lines.append(
            f"| {rank} | {subject} | {_fmt_pm(subject_rows[subject]['acc'])} | "
            f"{_fmt_pm(subject_rows[subject]['kappa'])} | T{cls_idx + 1} | "
            f"{_fmt(cls_value)} | {test_count} |"
        )
    lines.append("")

    pooled_class_mean = np.stack(pooled_class, axis=0).mean(axis=0)
    ranked_classes = sorted(range(len(pooled_class_mean)), key=lambda i: pooled_class_mean[i])
    lines.append("## Weak Classes")
    lines.append("")
    lines.append("| rank | class | pooled_recall |")
    lines.append("| --- | --- | --- |")
    for rank, cls_idx in enumerate(ranked_classes, start=1):
        values = [pc[cls_idx] for pc in pooled_class]
        lines.append(f"| {rank} | T{cls_idx + 1} | {_fmt_pm(values)} |")
    lines.append("")

    n_classes = len(pooled_class_mean)
    lines.append("## Per-Subject Class Recall")
    lines.append("")
    header = ["subject"] + [f"T{i + 1}" for i in range(n_classes)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for subject in ranked_subjects:
        class_mean = np.stack(subject_class[subject], axis=0).mean(axis=0)
        vals = [_fmt(v) for v in class_mean]
        lines.append("| " + " | ".join([subject] + vals) + " |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/baseline")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    text = build_report(args.results_root, args.runs)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()

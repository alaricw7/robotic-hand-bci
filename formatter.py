import numpy as np


def _fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def format_summary(
    *,
    experiment: str,
    model_name: str,
    cv: str,
    validation_desc: str,
    metric_desc: str,
    val_size: float,
    data_root: str,
    n_classes: int,
    per_subject_results: dict,
    extra_lines: list = None,
) -> str:
    """``per_subject_results`` maps subject_id -> dict with keys
    ``acc``, ``kappa``, ``per_class`` (length n_classes).

    Each subject is one run (std = 0), so per-subject ± is 0.0000.
    The AVG row computes mean ± std across subjects.
    """
    lines = []
    lines.append(f"Experiment: {experiment}")
    lines.append(f"Model: {model_name}")
    lines.append(f"CV: {cv}")
    lines.append(f"Validation: {validation_desc}")
    lines.append(f"Metric: {metric_desc}")
    lines.append(f"Val size: {val_size}")
    lines.append(f"Data root: {data_root}")
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("=" * 80)

    header_classes = "".join(f"T{i + 1:<7}" for i in range(n_classes))
    lines.append(f"{'Subject':<10}{'Acc':<18}{'Kappa':<18}{header_classes}")
    lines.append("-" * 80)

    accs, kappas = [], []
    per_class_stack = []
    for sid, r in per_subject_results.items():
        acc = float(r["acc"])
        kappa = float(r["kappa"])
        acc_std = float(r.get("acc_std", 0.0))
        kappa_std = float(r.get("kappa_std", 0.0))
        pc = np.asarray(r["per_class"], dtype=float)
        accs.append(acc)
        kappas.append(kappa)
        per_class_stack.append(pc)
        pc_str = "".join(f"{v:<8.4f}" for v in pc)
        lines.append(
            f"{sid:<10}{_fmt_pm(acc, acc_std):<15}{_fmt_pm(kappa, kappa_std):<15}{pc_str}"
        )

    lines.append("-" * 80)
    accs = np.asarray(accs)
    kappas = np.asarray(kappas)
    per_class_stack = np.stack(per_class_stack, axis=0)
    pc_mean = per_class_stack.mean(axis=0)
    avg_pc_str = "".join(f"{v:<8.4f}" for v in pc_mean)
    lines.append(
        f"{'AVG':<10}"
        f"{_fmt_pm(accs.mean(), accs.std()):<15}"
        f"{_fmt_pm(kappas.mean(), kappas.std()):<15}"
        f"{avg_pc_str}"
    )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"

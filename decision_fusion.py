"""Decision fusion ceiling probe.

Loads per-(subject, fold) checkpoints from three independently-trained
single-branch ablations (time_only / freq_only / space_only) and combines
their probabilities at inference time. By construction these encoders are
NOT joint-trained, so each one runs at its solo capacity ceiling — fusion
on top gives an upper bound on what naive (non-jointly-trained) fusion
can reach.

Inputs (must already exist):
  results/abl_time_only/ckpts/<subj>/foldK.pt
  results/abl_freq_only/ckpts/<subj>/foldK.pt
  results/abl_space_only/ckpts/<subj>/foldK.pt

Outputs (results/decision_fusion/<exp>/):
  summary.txt + results.json (formatter-style, one row per fusion strategy)
  per_fold_weights.csv  (val-optimal simplex weights per (subj, fold))
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from ablation_config import ABLATION_PRESETS, TRI_DEFAULTS
from data import (
    DATA_ROOT,
    SUBJECTS,
    load_subject,
    standardize_per_channel,
    stratified_kfold_train_val_test_splits,
)
from formatter import format_summary
from metrics import compute_metrics
from model import build_model


BRANCHES = ("time", "freq", "space")
PRESETS = {"time": "time_only", "freq": "freq_only", "space": "space_only"}


def build_tri_cfg(n_channels, n_times, n_classes, ablation_name, seed):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation_name])
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes})
    return SimpleNamespace(**values)


def ckpt_path(results_root, exp_name, subject, fold_idx):
    return Path(results_root) / exp_name / "ckpts" / subject / f"fold{fold_idx}.pt"


@torch.no_grad()
def _forward_probs(model, X_np, device, batch_size=64):
    model.eval()
    n = X_np.shape[0]
    out = []
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device, non_blocking=True)
        logits = model(xb)
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _kappa(y_true, y_pred, n_classes):
    _, k, _ = compute_metrics(y_true, y_pred, n_classes)
    return float(k)


def _simplex_grid(step=0.05):
    """Generate all (a,b,c) with a+b+c=1, a,b,c in {step·k}.
    Returns list of np.array shape (3,)."""
    out = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append(np.array([i, j, k], dtype=float) / n)
    return out


def fuse_subject(
    subject,
    *,
    exp_names,
    results_root,
    n_folds,
    val_size,
    seed,
    device,
):
    X, y = load_subject(subject)
    n_classes = int(y.max() + 1)
    n_channels, n_times = X.shape[1], X.shape[2]

    fold_results = []
    grid = _simplex_grid(step=0.05)

    for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
        y, n_splits=n_folds, val_size=val_size, seed=seed,
    ):
        X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
        y_va, y_te = y[va_idx], y[te_idx]
        X_tr_std, X_va_std, X_te_std = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)

        # Per-branch probabilities on val + test
        val_p, test_p = {}, {}
        val_solo_acc = {}
        for branch in BRANCHES:
            ablation = PRESETS[branch]
            preset_name = exp_names.get(branch, f"abl_{ablation}")
            cfg = build_tri_cfg(n_channels, n_times, n_classes, ablation, seed)
            model = build_model(cfg, model_name="tridomain").to(device)
            path = ckpt_path(results_root, preset_name, subject, fold_idx)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing single-branch ckpt: {path}. "
                    f"Run: python run_ablation_5fold.py --ablation {ablation} --save-ckpts ..."
                )
            blob = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["state_dict"])
            val_p[branch] = _forward_probs(model, X_va_std, device)
            test_p[branch] = _forward_probs(model, X_te_std, device)
            val_solo_acc[branch] = float((val_p[branch].argmax(axis=1) == y_va).mean())

        # --- strategies ---
        # (a) equal-weight average
        eq = (val_p["time"] + val_p["freq"] + val_p["space"]) / 3.0
        eq_te = (test_p["time"] + test_p["freq"] + test_p["space"]) / 3.0

        # (b) val-acc weighted
        accs = np.asarray([val_solo_acc[b] for b in BRANCHES], dtype=float)
        wac = accs / max(accs.sum(), 1e-9)
        vac_te = sum(wac[i] * test_p[b] for i, b in enumerate(BRANCHES))

        # (c) simplex grid search maximising val-kappa
        best_w, best_kappa = None, -float("inf")
        for w in grid:
            p = w[0] * val_p["time"] + w[1] * val_p["freq"] + w[2] * val_p["space"]
            k = _kappa(y_va, p.argmax(axis=1), n_classes)
            if k > best_kappa:
                best_kappa = k
                best_w = w
        simp_te = (best_w[0] * test_p["time"]
                   + best_w[1] * test_p["freq"]
                   + best_w[2] * test_p["space"])

        # (d) stacking: multinomial LR on concatenated val probabilities
        from sklearn.linear_model import LogisticRegression
        X_val_stack = np.concatenate([val_p[b] for b in BRANCHES], axis=1)
        X_te_stack = np.concatenate([test_p[b] for b in BRANCHES], axis=1)
        try:
            clf = LogisticRegression(solver="lbfgs", multi_class="multinomial",
                                     max_iter=2000, n_jobs=1)
            clf.fit(X_val_stack, y_va)
            stack_pred = clf.predict(X_te_stack)
            stack_acc, stack_kappa, stack_pc = compute_metrics(y_te, stack_pred, n_classes)
        except Exception as e:
            stack_acc, stack_kappa = float("nan"), float("nan")
            stack_pc = np.full(n_classes, np.nan)

        # Tabulate
        per_strat = {
            "equal": (y_te, eq_te.argmax(axis=1)),
            "val_acc_weighted": (y_te, vac_te.argmax(axis=1)),
            "simplex_val_kappa": (y_te, simp_te.argmax(axis=1)),
        }
        out_row = {"fold": fold_idx, "w_simplex": best_w.tolist(),
                   "val_solo_acc": val_solo_acc}
        for name, (yt, yp) in per_strat.items():
            acc, k, pc = compute_metrics(yt, yp, n_classes)
            out_row[name] = {"acc": float(acc), "kappa": float(k),
                             "per_class": pc.tolist()}
        out_row["stacking_lr"] = {"acc": float(stack_acc), "kappa": float(stack_kappa),
                                  "per_class": stack_pc.tolist()}
        out_row["counts"] = {"train": int(len(tr_idx)),
                             "val": int(len(va_idx)),
                             "test": int(len(te_idx))}
        fold_results.append(out_row)
        print(f"  {subject} f{fold_idx}: "
              f"eq={out_row['equal']['acc']:.3f} "
              f"vac={out_row['val_acc_weighted']['acc']:.3f} "
              f"simp={out_row['simplex_val_kappa']['acc']:.3f} "
              f"(w={best_w.round(2).tolist()}) "
              f"stack={out_row['stacking_lr']['acc']:.3f}")
    return fold_results, n_classes


def aggregate(per_subj, strategy):
    """Build {subject: {acc, kappa, per_class}} for the formatter."""
    table = {}
    for s, folds in per_subj.items():
        accs = np.asarray([f[strategy]["acc"] for f in folds])
        ks = np.asarray([f[strategy]["kappa"] for f in folds])
        pcs = np.stack([np.asarray(f[strategy]["per_class"]) for f in folds])
        table[s] = {"acc": float(accs.mean()),
                    "acc_std": float(accs.std()),
                    "kappa": float(ks.mean()),
                    "kappa_std": float(ks.std()),
                    "per_class": pcs.mean(axis=0).tolist()}
    return table


def avg_excl(per_subj_table, excl=("S2", "S3")):
    keep = [s for s in per_subj_table if s not in excl]
    accs = np.asarray([per_subj_table[s]["acc"] for s in keep])
    ks = np.asarray([per_subj_table[s]["kappa"] for s in keep])
    return float(accs.mean()), float(accs.std()), float(ks.mean()), float(ks.std()), keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="decision_fusion_v1")
    p.add_argument("--time-exp", default="abl_time_only")
    p.add_argument("--freq-exp", default="abl_freq_only")
    p.add_argument("--space-exp", default="abl_space_only")
    p.add_argument("--results-root", default=str(THIS_DIR / "results"))
    p.add_argument("--out-root", default=str(THIS_DIR / "results" / "decision_fusion"))
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    args = p.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_names = {"time": args.time_exp, "freq": args.freq_exp, "space": args.space_exp}
    print(f"device={device} exp={args.exp_name} ckpts={exp_names}")

    per_subj_folds = {}
    n_classes_seen = None
    for subject in args.subjects:
        folds, n_classes = fuse_subject(
            subject,
            exp_names=exp_names,
            results_root=args.results_root,
            n_folds=args.n_folds,
            val_size=args.val_size,
            seed=args.seed,
            device=device,
        )
        per_subj_folds[subject] = folds
        n_classes_seen = n_classes

    # Persist
    out_dir = Path(args.out_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold weight CSV
    w_csv = out_dir / "per_fold_weights.csv"
    with w_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["subject", "fold", "w_time", "w_freq", "w_space",
                     "val_acc_time", "val_acc_freq", "val_acc_space"])
        for s, folds in per_subj_folds.items():
            for fr in folds:
                w = fr["w_simplex"]
                v = fr["val_solo_acc"]
                wr.writerow([s, fr["fold"],
                             f"{w[0]:.4f}", f"{w[1]:.4f}", f"{w[2]:.4f}",
                             f"{v['time']:.4f}", f"{v['freq']:.4f}", f"{v['space']:.4f}"])

    summary_lines = []
    json_payload = {"experiment": args.exp_name, "ckpts": exp_names,
                    "n_folds": args.n_folds, "val_size": args.val_size,
                    "seed": args.seed, "strategies": {}}

    for strategy in ("equal", "val_acc_weighted", "simplex_val_kappa", "stacking_lr"):
        per_subj_tbl = aggregate(per_subj_folds, strategy)
        extra = []
        excl = avg_excl(per_subj_tbl)
        if excl is not None:
            a_m, a_s, k_m, k_s, keep = excl
            extra.append(f"AVG excl S2/S3 ({','.join(keep)}): "
                         f"acc={a_m:.4f}±{a_s:.4f}  kappa={k_m:.4f}±{k_s:.4f}")
        s = format_summary(
            experiment=f"{args.exp_name}::{strategy}",
            model_name="decision_fusion",
            cv=f"{args.n_folds}fold",
            validation_desc=f"outer stratified {args.n_folds}-fold; val from train_val",
            metric_desc=f"test set, strategy={strategy}, weights fit on val",
            val_size=args.val_size, data_root=DATA_ROOT,
            n_classes=n_classes_seen, per_subject_results=per_subj_tbl,
            extra_lines=extra,
        )
        summary_lines.append(s)
        json_payload["strategies"][strategy] = {
            "per_subject": per_subj_tbl,
            "avg_excl_s2_s3": (None if excl is None else {
                "subjects": keep, "acc_mean": excl[0], "acc_std": excl[1],
                "kappa_mean": excl[2], "kappa_std": excl[3],
            }),
        }
        print(s)

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    with (out_dir / "results.json").open("w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"saved: {out_dir/'summary.txt'}")
    print(f"saved: {out_dir/'results.json'}")
    print(f"saved: {w_csv}")


if __name__ == "__main__":
    main()

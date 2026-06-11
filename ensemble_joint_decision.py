"""Task 5 A1 — test-time ensemble of joint (T7) and decision fusion.

Zero training. Per (subject, fold):
  P_joint(val/test)  : softmax of T7 main-head logits on val / test
  P_dec(val/test)    : weighted blend of solo time/freq/space probabilities,
                       weights = simplex grid argmax_val_kappa (same as
                       decision_fusion.py strategy 'simplex_val_kappa')
  final = alpha * P_joint + (1 - alpha) * P_dec
  alpha* = argmax_{alpha in {0, 0.05, ..., 1.0}} val_kappa(final_val)
  test prediction uses alpha*

Outputs:
  results/ensemble/joint_decision/{summary.txt,results.json,alpha_per_fold.csv}
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


SOLO_BRANCHES = ("time", "freq", "space")
SOLO_PRESETS = {"time": "time_only", "freq": "freq_only", "space": "space_only"}


def build_tri_cfg(n_channels, n_times, n_classes, ablation, tri_overrides, seed):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation])
    if tri_overrides:
        values.update(tri_overrides)
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes})
    return SimpleNamespace(**values)


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


def _simplex_grid(step=0.05):
    out = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append(np.array([i, j, k], dtype=float) / n)
    return out


def _alpha_grid(step=0.05):
    n = int(round(1.0 / step))
    return [round(i / n, 4) for i in range(n + 1)]


def _kappa(y_true, y_pred, n_classes):
    _, k, _ = compute_metrics(y_true, y_pred, n_classes)
    return float(k)


def run_one_subject(subject, args, device, simplex, alphas):
    X, y = load_subject(subject)
    n_classes = int(y.max() + 1)
    n_channels, n_times = X.shape[1], X.shape[2]

    fold_rows = []
    for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
        y, n_splits=args.n_folds, val_size=args.val_size, seed=args.seed,
    ):
        X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
        y_va, y_te = y[va_idx], y[te_idx]
        X_tr_std, X_va_std, X_te_std = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)

        # === Joint (T7) probabilities ===
        joint_ckpt = (Path(args.results_root) / args.joint_exp / "ckpts"
                      / subject / f"fold{fold_idx}.pt")
        if not joint_ckpt.exists():
            raise FileNotFoundError(f"missing joint ckpt: {joint_ckpt}")
        blob = torch.load(joint_ckpt, map_location="cpu", weights_only=False)
        meta = blob.get("meta") or {}
        cfg = build_tri_cfg(n_channels, n_times, n_classes,
                            meta.get("ablation", "full_std_coords"),
                            meta.get("tri_overrides") or {}, args.seed)
        joint_model = build_model(cfg, model_name="tridomain")
        joint_model.load_state_dict(blob["state_dict"])
        joint_model.eval().to(device)
        P_joint_val = _forward_probs(joint_model, X_va_std, device)
        P_joint_test = _forward_probs(joint_model, X_te_std, device)
        del joint_model

        # === Solo branch probabilities ===
        solo_val, solo_test = {}, {}
        for branch in SOLO_BRANCHES:
            ab = SOLO_PRESETS[branch]
            cp = (Path(args.solo_results_root) / f"abl_{ab}" / "ckpts"
                  / subject / f"fold{fold_idx}.pt")
            if not cp.exists():
                raise FileNotFoundError(f"missing solo ckpt: {cp}")
            blob_s = torch.load(cp, map_location="cpu", weights_only=False)
            cfg_s = build_tri_cfg(n_channels, n_times, n_classes, ab, {}, args.seed)
            model_s = build_model(cfg_s, model_name="tridomain")
            model_s.load_state_dict(blob_s["state_dict"])
            model_s.eval().to(device)
            solo_val[branch] = _forward_probs(model_s, X_va_std, device)
            solo_test[branch] = _forward_probs(model_s, X_te_std, device)
            del model_s

        # simplex search for decision-fusion weights on val
        best_w, best_kappa = None, -float("inf")
        for w in simplex:
            p = (w[0] * solo_val["time"] + w[1] * solo_val["freq"]
                 + w[2] * solo_val["space"])
            k = _kappa(y_va, p.argmax(axis=1), n_classes)
            if k > best_kappa:
                best_kappa = k
                best_w = w
        P_dec_val = (best_w[0] * solo_val["time"] + best_w[1] * solo_val["freq"]
                     + best_w[2] * solo_val["space"])
        P_dec_test = (best_w[0] * solo_test["time"] + best_w[1] * solo_test["freq"]
                      + best_w[2] * solo_test["space"])

        # alpha search on val
        best_alpha, best_alpha_kappa = None, -float("inf")
        for a in alphas:
            blend = a * P_joint_val + (1 - a) * P_dec_val
            k = _kappa(y_va, blend.argmax(axis=1), n_classes)
            if k > best_alpha_kappa:
                best_alpha_kappa = k
                best_alpha = a

        # test prediction with alpha*
        blend_te = best_alpha * P_joint_test + (1 - best_alpha) * P_dec_test
        y_pred = blend_te.argmax(axis=1)
        test_acc, test_kappa, test_pc = compute_metrics(y_te, y_pred, n_classes)

        # also test joint-only and dec-only for reference (for log lines)
        y_joint = P_joint_test.argmax(axis=1)
        y_dec = P_dec_test.argmax(axis=1)
        acc_j, k_j, _ = compute_metrics(y_te, y_joint, n_classes)
        acc_d, k_d, _ = compute_metrics(y_te, y_dec, n_classes)

        fold_rows.append({
            "fold": fold_idx,
            "alpha": float(best_alpha),
            "w_simplex": best_w.tolist(),
            "acc": float(test_acc), "kappa": float(test_kappa),
            "per_class": test_pc.tolist(),
            "acc_joint": float(acc_j), "kappa_joint": float(k_j),
            "acc_dec": float(acc_d), "kappa_dec": float(k_d),
            "counts": {"train": int(len(tr_idx)), "val": int(len(va_idx)),
                       "test": int(len(te_idx))},
        })
        print(f"  {subject} f{fold_idx}: alpha*={best_alpha:.2f} "
              f"ens={test_acc:.3f}/{test_kappa:.3f} "
              f"joint={acc_j:.3f} dec={acc_d:.3f} "
              f"w={best_w.round(2).tolist()}")

    # subject-level summary
    accs = np.asarray([r["acc"] for r in fold_rows])
    ks = np.asarray([r["kappa"] for r in fold_rows])
    pcs = np.stack([np.asarray(r["per_class"]) for r in fold_rows])
    summary = {
        "acc": float(accs.mean()), "acc_std": float(accs.std()),
        "kappa": float(ks.mean()), "kappa_std": float(ks.std()),
        "per_class": pcs.mean(axis=0).tolist(),
        "folds": fold_rows,
    }
    print(f"  {subject} AVG: acc={summary['acc']:.4f}±{summary['acc_std']:.4f} "
          f"kappa={summary['kappa']:.4f}")
    return summary, n_classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint-exp", default="abl_signalfit_T7_full_stack_taps251")
    p.add_argument("--results-root", default=str(THIS_DIR / "results" / "aug_hpo"),
                   help="Where the joint (T7) exp lives.")
    p.add_argument("--solo-results-root", default=str(THIS_DIR / "results"),
                   help="Where abl_time_only / abl_freq_only / abl_space_only live.")
    p.add_argument("--out-name", default="joint_decision")
    p.add_argument("--out-root", default=str(THIS_DIR / "results" / "ensemble"))
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

    simplex = _simplex_grid(0.05)
    alphas = _alpha_grid(0.05)
    print(f"device={device} joint_exp={args.joint_exp} n_alpha={len(alphas)} n_simplex={len(simplex)}")

    per_subj = {}
    n_classes_seen = None
    for s in args.subjects:
        out, n_classes = run_one_subject(s, args, device, simplex, alphas)
        per_subj[s] = out
        n_classes_seen = n_classes

    out_dir = Path(args.out_root) / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # alpha distribution CSV
    with (out_dir / "alpha_per_fold.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["subject", "fold", "alpha", "w_time", "w_freq", "w_space",
                     "acc_ensemble", "acc_joint", "acc_dec"])
        for s, info in per_subj.items():
            for fr in info["folds"]:
                w = fr["w_simplex"]
                wr.writerow([s, fr["fold"], f"{fr['alpha']:.2f}",
                             f"{w[0]:.2f}", f"{w[1]:.2f}", f"{w[2]:.2f}",
                             f"{fr['acc']:.4f}", f"{fr['acc_joint']:.4f}",
                             f"{fr['acc_dec']:.4f}"])

    # excl S2/S3 AVG
    keep = [s for s in args.subjects if s not in ("S2", "S3")]
    a_excl = np.asarray([per_subj[s]["acc"] for s in keep])
    k_excl = np.asarray([per_subj[s]["kappa"] for s in keep])
    extras = [
        f"Joint exp:  {args.joint_exp}",
        f"Solo root:  {args.solo_results_root}  (time_only / freq_only / space_only)",
        f"AVG excl S2/S3 ({','.join(keep)}): "
        f"acc={a_excl.mean():.4f}±{a_excl.std():.4f}  kappa={k_excl.mean():.4f}±{k_excl.std():.4f}",
    ]
    # alpha distribution stats
    all_alphas = [fr["alpha"] for s in per_subj for fr in per_subj[s]["folds"]]
    extras.append(f"alpha*: mean={np.mean(all_alphas):.3f}  std={np.std(all_alphas):.3f}  "
                  f"n_in_interior(0<a<1)={sum(1 for a in all_alphas if 0 < a < 1)}/{len(all_alphas)}  "
                  f"n_alpha=1 (joint only)={all_alphas.count(1.0)}  "
                  f"n_alpha=0 (dec only)={all_alphas.count(0.0)}")

    summary_txt = format_summary(
        experiment=f"ensemble::{args.out_name}",
        model_name="ensemble_joint_decision",
        cv=f"{args.n_folds}fold",
        validation_desc=f"outer stratified {args.n_folds}-fold; val for alpha+simplex",
        metric_desc="test set, alpha* tuned on val by val-kappa",
        val_size=args.val_size, data_root=DATA_ROOT,
        n_classes=n_classes_seen, per_subject_results=per_subj,
        extra_lines=extras,
    )
    (out_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")
    with (out_dir / "results.json").open("w") as f:
        json.dump({"per_subject": per_subj,
                   "joint_exp": args.joint_exp,
                   "alpha_grid": alphas,
                   "avg_excl_s2_s3": {
                       "subjects": keep,
                       "acc_mean": float(a_excl.mean()), "acc_std": float(a_excl.std()),
                       "kappa_mean": float(k_excl.mean()), "kappa_std": float(k_excl.std()),
                   }}, f, indent=2)
    print(summary_txt)
    print(f"saved: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()

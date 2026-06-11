"""Cross-run per-branch linear probe aggregator.

For each (run, subject, fold) it loads the saved ckpt, encodes train + test
through the frozen encoder, fits a per-branch StandardScaler + multinomial LR
on the train embeddings, and reports test accuracy / Cohen kappa. Subject
aggregation mirrors formatter.py (subject mean over folds → AVG mean±std).

The ckpt blob is expected to carry meta={"ablation", "tri_overrides", "seed"}
(as written by run_aug.py --save-ckpts). The model is rebuilt from those so
custom flags (per_branch_norm / aux_loss_enabled / time_pool_mode / freq_bands)
load correctly.

Output: a single markdown table merging
  - main metrics from results/<exp>/results.json
  - probe metrics computed here
"""

import argparse
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
    SUBJECTS,
    load_subject,
    standardize_per_channel,
    stratified_kfold_train_val_test_splits,
)
from model import build_model


BRANCHES = ("time", "freq", "space")
BASELINE_ACC = 0.4797


def build_tri_cfg(n_channels, n_times, n_classes, ablation, tri_overrides, seed,
                  cov_dim=None):
    values = dict(TRI_DEFAULTS)
    values.update(ABLATION_PRESETS[ablation])
    if tri_overrides:
        values.update(tri_overrides)
    if cov_dim is not None:
        values["cov_dim"] = cov_dim
    values.update({"random_seed": seed, "n_channels": n_channels,
                   "n_samples": n_times, "n_classes": n_classes})
    return SimpleNamespace(**values)


@torch.no_grad()
def _encode_batches(model, X_np, device, batch_size=64):
    """Probe time / freq / space sub-encoders directly.

    Going through model.encoder.forward() would require cov_feat when the
    cov branch is active. The probe only needs time/freq/space embeddings,
    so we invoke each sub-encoder individually — sidesteps the cov_feat
    requirement and works for both 3- and 4-branch ckpts.
    """
    n = X_np.shape[0]
    outs = {b: [] for b in BRANCHES}
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device, non_blocking=True)
        if hasattr(model.encoder, "time"):
            e_t, _ = model.encoder.time(xb)
            outs["time"].append(e_t.detach().cpu().numpy())
        if hasattr(model.encoder, "freq"):
            e_f, _ = model.encoder.freq(xb)
            outs["freq"].append(e_f.detach().cpu().numpy())
        if hasattr(model.encoder, "space"):
            e_s, _ = model.encoder.space(xb)
            outs["space"].append(e_s.detach().cpu().numpy())
    return {b: np.concatenate(outs[b], axis=0) for b in BRANCHES if outs[b]}


def probe_run(
    exp_name,
    *,
    results_root,
    subjects,
    n_folds,
    val_size,
    seed,
    device,
    batch_size,
    max_iter,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    ckpt_root = Path(results_root) / exp_name / "ckpts"
    if not ckpt_root.exists():
        return None

    per_subject = {s: {b: [] for b in BRANCHES} for s in subjects}

    for subject in subjects:
        X, y = load_subject(subject)
        n_classes = int(y.max() + 1)
        n_channels, n_times = X.shape[1], X.shape[2]

        for fold_idx, tr_idx, va_idx, te_idx in stratified_kfold_train_val_test_splits(
            y, n_splits=n_folds, val_size=val_size, seed=seed,
        ):
            ckpt_p = ckpt_root / subject / f"fold{fold_idx}.pt"
            if not ckpt_p.exists():
                print(f"  [probe:{exp_name}] missing {ckpt_p}, skip fold")
                continue
            X_tr_raw, X_va_raw, X_te_raw = X[tr_idx], X[va_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            X_tr_std, _, X_te_std = standardize_per_channel(X_tr_raw, X_va_raw, X_te_raw)

            blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            meta = blob.get("meta") or {}
            ablation = meta.get("ablation", "full_std_coords")
            tri_overrides = meta.get("tri_overrides") or {}
            cov_dim = meta.get("cov_dim")
            cfg = build_tri_cfg(n_channels, n_times, n_classes,
                                ablation, tri_overrides, seed, cov_dim=cov_dim)
            model = build_model(cfg, model_name="tridomain")
            model.load_state_dict(blob["state_dict"])
            model.eval().to(device)

            emb_tr = _encode_batches(model, X_tr_std, device, batch_size)
            emb_te = _encode_batches(model, X_te_std, device, batch_size)

            for b in BRANCHES:
                if b not in emb_tr:
                    continue
                scaler = StandardScaler().fit(emb_tr[b])
                Z_tr = scaler.transform(emb_tr[b])
                Z_te = scaler.transform(emb_te[b])
                clf = LogisticRegression(solver="lbfgs", multi_class="multinomial",
                                         max_iter=max_iter, n_jobs=1)
                clf.fit(Z_tr, y_tr)
                yp = clf.predict(Z_te)
                acc = accuracy_score(y_te, yp)
                kappa = cohen_kappa_score(y_te, yp)
                per_subject[subject][b].append((acc, kappa))

    # subject means → AVG (all + excl S2/S3)
    table = {}
    for b in BRANCHES:
        subj_means = []
        for s in subjects:
            arr = np.asarray(per_subject[s][b], dtype=float)
            if arr.size == 0:
                continue
            subj_means.append((s, arr[:, 0].mean(), arr[:, 1].mean()))
        if not subj_means:
            table[b] = None
            continue
        arr_all = np.asarray([(a, k) for _, a, k in subj_means])
        kept_excl = [(s, a, k) for s, a, k in subj_means if s not in ("S2", "S3")]
        arr_excl = np.asarray([(a, k) for _, a, k in kept_excl]) if kept_excl else None
        table[b] = {
            "acc_mean_all": float(arr_all[:, 0].mean()),
            "acc_std_all": float(arr_all[:, 0].std()),
            "kappa_mean_all": float(arr_all[:, 1].mean()),
            "acc_mean_excl": float(arr_excl[:, 0].mean()) if arr_excl is not None else None,
        }
    return table


def read_main_metrics(exp_name, results_root):
    rj = Path(results_root) / exp_name / "results.json"
    if not rj.exists():
        return None
    j = json.loads(rj.read_text())
    accs, kappas = [], []
    for s, r in j["per_subject"].items():
        accs.append(float(r["acc"]))
        kappas.append(float(r["kappa"]))
    excl = [s for s in j["per_subject"] if s not in ("S2", "S3")]
    accs_excl = [float(j["per_subject"][s]["acc"]) for s in excl]
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "kappa_mean": float(np.mean(kappas)),
        "acc_mean_excl": float(np.mean(accs_excl)) if accs_excl else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="List of exp_name (each must have ckpts/ subdir)")
    p.add_argument("--results-root", default=str(THIS_DIR / "results"))
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--out", default=str(THIS_DIR / "results" / "probe_table.md"))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for exp in args.runs:
        print(f"[probe] === {exp} ===")
        main_metrics = read_main_metrics(exp, args.results_root)
        probe_metrics = probe_run(
            exp,
            results_root=args.results_root, subjects=args.subjects,
            n_folds=args.n_folds, val_size=args.val_size, seed=args.seed,
            device=device, batch_size=args.batch_size, max_iter=args.max_iter,
        )
        rows.append({"exp": exp, "main": main_metrics, "probe": probe_metrics})
        if main_metrics:
            print(f"  main: acc={main_metrics['acc_mean']:.4f} "
                  f"κ={main_metrics['kappa_mean']:.4f} "
                  f"excl={main_metrics['acc_mean_excl']:.4f}")
        if probe_metrics:
            for b in BRANCHES:
                m = probe_metrics.get(b)
                if m:
                    print(f"  probe {b}: {m['acc_mean_all']:.4f} "
                          f"(excl {m['acc_mean_excl']:.4f})")

    def cell_main(m, key):
        return f"{m[key]:.4f}" if m and m.get(key) is not None else "-"

    def cell_delta(m):
        if not m or m.get("acc_mean") is None:
            return "-"
        d = m["acc_mean"] - BASELINE_ACC
        return f"{d:+.4f}"

    def cell_probe(p, b):
        if not p or not p.get(b):
            return "-"
        return f"{p[b]['acc_mean_all']:.4f}"

    md = ["# Cross-run probe table (Task 1 + Task 3)\n"]
    md.append(
        "| run | Acc(all10) ± std | Δ vs 0.4797 | κ | Acc(excl S2/S3) | "
        "probe time | probe freq | probe space |"
    )
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        m = r["main"]; pr = r["probe"]
        acc_pm = (f"{m['acc_mean']:.4f}±{m['acc_std']:.4f}"
                  if m and m.get("acc_mean") is not None else "-")
        md.append(
            f"| {r['exp']} | {acc_pm} | {cell_delta(m)} | "
            f"{cell_main(m, 'kappa_mean')} | {cell_main(m, 'acc_mean_excl')} | "
            f"{cell_probe(pr, 'time')} | {cell_probe(pr, 'freq')} | "
            f"{cell_probe(pr, 'space')} |"
        )

    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[probe] saved: {args.out}")


if __name__ == "__main__":
    main()

"""Evidential training + inference for the DAFG decoder (component A).

The decoder front-end is DAFGNetV0 (ERP-augmented covariance -> log-Eucl tangent
-> distance evidence head). The head emits Dirichlet evidence, trained with the
EDL loss (Sensoy et al. 2018). Predicted prob p = alpha/S is calibrated by
construction; uncertainty (vacuity) u = K/S drives selective abstention.

Within-subject, StratifiedKFold(5), seed=42. Saves per-subject npz with the
evidential probabilities and vacuity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import prepare_subject, stratified_folds, channel_standardize, SUBJECTS
from .model import DAFGNetV0, evoked_prototypes, init_prototypes, DEV

OUT = Path(__file__).resolve().parent / "evi"
OUT.mkdir(exist_ok=True)
N_CLASSES = 6


# --------------------------------------------------------------------------- #
# evidential deep learning (Sensoy et al. 2018) on the DAFG decoder
# --------------------------------------------------------------------------- #
def kl_to_uniform(alpha):
    """KL( Dir(alpha) || Dir(1) ), per-sample. alpha: (B, K)."""
    K = alpha.shape[1]
    beta = torch.ones_like(alpha)
    S_a = alpha.sum(1, keepdim=True)
    lnB = torch.lgamma(S_a) - torch.lgamma(alpha).sum(1, keepdim=True)
    lnB_uni = torch.lgamma(beta).sum(1, keepdim=True) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    dg = torch.digamma(alpha) - torch.digamma(S_a)
    return (lnB + lnB_uni + ((alpha - beta) * dg).sum(1, keepdim=True)).squeeze(1)


def edl_loss(evidence, y, epoch, n_classes, anneal=30, kl_weight=1.0):
    """EDL loss given non-negative evidence. kl_weight scales the final
    KL-to-uniform coefficient (lower => less collapse pressure for the bounded
    distance-evidence head)."""
    alpha = evidence + 1.0
    S = alpha.sum(1, keepdim=True)
    y1 = F.one_hot(y, n_classes).float()
    # expected cross-entropy under the Dirichlet (digamma form)
    ce = (y1 * (torch.digamma(S) - torch.digamma(alpha))).sum(1).mean()
    # penalize evidence on the WRONG classes, annealed
    alpha_tilde = y1 + (1.0 - y1) * alpha
    lam = 1.0 if anneal <= 0 else min(1.0, epoch / anneal)
    return ce + kl_weight * lam * kl_to_uniform(alpha_tilde).mean()


def evidence_of(model, X):
    """Non-negative Dirichlet evidence = exp(log-evidence). Accepts a device
    tensor (training, grad kept) or a numpy array (inference, eval + no_grad)."""
    if torch.is_tensor(X):
        out = model(X)
    else:
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X).to(DEV))
    return out.exp()


def train_edl(Xtr, ytr, args):
    P = evoked_prototypes(Xtr, ytr, svd=args.svd)
    C, T = Xtr.shape[1], Xtr.shape[2]
    m = DAFGNetV0(C, T, N_CLASSES, P, rank=args.rank, dropout=args.dropout,
                  evidence_scale_init=args.evidence_scale,
                  embed_dim=args.embed_dim, embed_delay=args.embed_delay).to(DEV)
    init_prototypes(m, Xtr, ytr, evidence_scale_init=args.evidence_scale)
    # prototypes/scale/tau are large-norm anchors -> exclude from weight decay
    # (AdamW decay would pull protos to 0 and collapse them -> chance).
    nodecay = {"head.proto", "head.raw_scale", "head.raw_tau"}
    opt = torch.optim.AdamW([
        {"params": [p for n, p in m.named_parameters() if n not in nodecay],
         "weight_decay": args.wd},
        {"params": [p for n, p in m.named_parameters() if n in nodecay],
         "weight_decay": 0.0}], lr=args.lr)
    X = torch.tensor(Xtr).to(DEV); y = torch.tensor(ytr).long().to(DEV); n = len(X)
    for ep in range(args.epochs):
        m.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            evidence = evidence_of(m, X[idx])           # tensor -> grad path
            edl_loss(evidence, y[idx], ep, N_CLASSES,
                     anneal=args.kl_anneal, kl_weight=args.kl_weight).backward()
            opt.step()
    return m


def evidential_predict(m, X):
    evidence = evidence_of(m, X)                        # numpy X -> eval + no_grad
    alpha = evidence + 1.0
    S = alpha.sum(1, keepdim=True)
    p = (alpha / S).cpu().numpy()
    unc = (evidence.shape[1] / S.squeeze(1)).cpu().numpy()   # vacuity in [0,1]
    return p, unc


def run_subject(subj, args):
    X, y, _, _ = prepare_subject(subj)
    n = len(y)
    p_ev = np.zeros((n, 6)); unc = np.zeros(n)
    for tr, te in stratified_folds(X, y):
        Xtr, Xte = channel_standardize(X[tr], X[te])
        m_ev = train_edl(Xtr, y[tr], args)
        p_ev[te], unc[te] = evidential_predict(m_ev, Xte)
    np.savez(OUT / f"{subj}.npz", y=y, p_evi=p_ev, unc=unc)
    acc_ev = float((p_ev.argmax(1) == y).mean())
    print(f"[{subj}] acc EDL={acc_ev:.4f}  mean vacuity={unc.mean():.3f}", flush=True)
    return acc_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--evidence-scale", type=float, default=20.0,
                    help="distance head: per-class evidence cap init (confidence ceiling)")
    ap.add_argument("--kl-weight", type=float, default=1.0,
                    help="EDL KL-to-uniform coefficient (lower stabilizes the distance head)")
    ap.add_argument("--kl-anneal", type=int, default=30,
                    help="epochs to ramp KL from 0 to kl_weight")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--svd", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--embed-dim", type=int, default=2,
                    help="ACM Takens embedding dimension p")
    ap.add_argument("--embed-delay", type=int, default=1,
                    help="ACM delay tau in samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    for s in args.subjects:
        run_subject(s, args)


if __name__ == "__main__":
    main()

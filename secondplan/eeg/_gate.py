"""Throwaway gate: confirm the cosine distance head recovers accuracy with usable
vacuity before committing the full sharded run. Prints to stdout (redirect to log)."""
import numpy as np, torch, time
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score
from eeg.train_evi import train_edl, evidential_predict
from eeg.data import prepare_subject, stratified_folds, channel_standardize

X, y, _, _ = prepare_subject('S1')
tr, te = next(stratified_folds(X, y))
Xtr, Xte = channel_standardize(X[tr], X[te])
for emb, klw, ep in [(2, 0.3, 150), (1, 0.3, 150)]:
    torch.manual_seed(42); np.random.seed(42)
    args = SimpleNamespace(evidence_scale=20.0, kl_weight=klw, kl_anneal=30, rank=8, svd=4,
                           epochs=ep, batch=128, lr=1e-3, wd=5e-3, dropout=0.3,
                           embed_dim=emb, embed_delay=1)
    t0 = time.time(); m = train_edl(Xtr, y[tr], args); p, u = evidential_predict(m, Xte)
    top1 = p.argmax(1); correct = (top1 == y[te])
    auroc = roc_auc_score((~correct).astype(int), u) if 0 < correct.mean() < 1 else float('nan')
    print(f'embed={emb} kl_w={klw} ep={ep}: acc={correct.mean():.3f}  '
          f'u[{u.min():.2f},{u.max():.2f}]m{u.mean():.2f}  '
          f'uC={u[correct].mean():.3f} uW={u[~correct].mean():.3f}  '
          f'AUROC={auroc:.3f}  {time.time()-t0:.0f}s', flush=True)

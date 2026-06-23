"""DAFG decoder (component A of EOGCs-DAFG) — model definition.

Single canonical path, matching the architecture figure:

    EEG [B, C, T]
      1. spatial graph mix    G = I + alpha * (U Vᵀ)      (low-rank, learnable)
      2. ERP prototype concat  fixed per-fold class templates P  -> [B, m+C, T]
      3. Takens delay embed    delay_embed(p, tau)               (ACM)
      4. covariance + trace-normalize                            (well-conditioned SPD)
      5. Log-Euclidean tangent vectorization        z = bn(logm(C))
      6. distance evidence head   score_k = log(scale_k) - ||z-proto_k||^2 / (2 tau^2)
      7. Dirichlet output (in dafg_evi.py): alpha = e+1, p = alpha/S, vacuity u = K/S

Trained end-to-end with the EDL loss (see dafg_evi.py). Within-subject protocol
(StratifiedKFold(5), seed=42, [0,2]s @125Hz, per-fold channel-standardize).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .acm import delay_embed

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def evoked_prototypes(Xtr, ytr, svd=4):
    """Per-class evoked template reduced to `svd` spatial comps -> P[K*svd, T]."""
    blocks = []
    for k in sorted(np.unique(ytr)):
        Mk = Xtr[ytr == k].mean(0)            # [C, T]
        U, _, _ = np.linalg.svd(Mk, full_matrices=False)
        blocks.append(U[:, :svd].T @ Mk)      # [svd, T]
    return np.concatenate(blocks, 0).astype(np.float32)   # [K*svd, T]


def cov(h, eps=1e-2, trace_norm=True):
    h = h - h.mean(-1, keepdim=True)
    d = h.shape[1]
    C = h @ h.transpose(-1, -2) / (h.shape[-1] - 1)
    C = 0.5 * (C + C.transpose(-1, -2))                      # symmetrize
    if trace_norm:
        tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)[:, None, None]
        C = C / (tr / d + 1e-6)                              # trace-normalize -> well-conditioned
    return C + eps * torch.eye(d, device=C.device, dtype=C.dtype)


def logeig_vec(C):
    """Log-Euclidean tangent vectorization of an SPD matrix (off-diag scaled by sqrt(2))."""
    Cd = C.double()
    Cd = 0.5 * (Cd + Cd.transpose(-1, -2))
    w, V = torch.linalg.eigh(Cd)                            # double precision -> stable
    w = torch.clamp(w, min=1e-6)
    fC = (V @ torch.diag_embed(torch.log(w)) @ V.transpose(-1, -2)).float()
    d = C.shape[-1]
    iu = torch.triu_indices(d, d, offset=1, device=C.device)
    diag = torch.diagonal(fC, dim1=-2, dim2=-1)
    off = fC[:, iu[0], iu[1]] * (2 ** 0.5)
    return torch.cat([diag, off], 1)


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y, for y > 0."""
    return float(math.log(math.expm1(y)))


class DistanceEvidenceHead(nn.Module):
    """Cosine distance-kernel evidence head. Emits LOG-evidence (score):

        score_k = log(softplus(raw_scale_k)) - 0.5 * ||ẑ - p̂_k||^2 / tau^2

    where ẑ and p̂_k are L2-normalized, so the squared distance lies in [0, 4] and
    the evidence cannot underflow to 0 -- which would collapse vacuity to u≡1 (the
    bare-kernel collapse). evidence_k = exp(score_k) = scale_k * exp(-0.5*d2/tau^2)
    in (0, scale_k]. proto/raw_scale/raw_tau are learnable and are (re)initialized
    from data by init_prototypes(). Param count = n_classes*D + n_classes + 1.
    """
    def __init__(self, D, n_classes, evidence_scale_init=20.0):
        super().__init__()
        self.proto = nn.Parameter(torch.zeros(n_classes, D))           # learnable class means
        self.raw_scale = nn.Parameter(                                  # per-class evidence cap
            torch.full((n_classes,), _inv_softplus(evidence_scale_init)))
        self.raw_tau = nn.Parameter(torch.tensor(_inv_softplus(1.0)))  # shared bandwidth tau

    def forward(self, z):                                # z: [B, D] (normalized tangent feats)
        zp = F.normalize(z, dim=-1)                      # L2-normalize -> cosine geometry
        proto = F.normalize(self.proto, dim=-1)
        d2 = torch.cdist(zp, proto) ** 2                 # [B, n_classes], in [0, 4]
        tau = F.softplus(self.raw_tau) + 1e-4            # scalar
        log_scale = torch.log(F.softplus(self.raw_scale))             # [n_classes]
        inv_2tau2 = 0.5 / (tau ** 2)
        return log_scale[None, :] - inv_2tau2 * d2       # log-evidence (score) [B, K]


class DAFGNetV0(nn.Module):
    def __init__(self, C, T, n_classes, P, rank=8, dropout=0.3,
                 evidence_scale_init=20.0, embed_dim=2, embed_delay=1,
                 trace_norm=True):
        super().__init__()
        self.C = C
        self.T = T
        self.trace_norm = bool(trace_norm)
        self.n_classes = n_classes
        self.embed_dim = int(embed_dim)
        self.embed_delay = int(embed_delay)
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if self.embed_delay < 1:
            raise ValueError(f"embed_delay must be >= 1, got {embed_delay}")
        self.register_buffer("P", torch.tensor(P))      # [m, T] fixed prototypes
        m = P.shape[0]
        self.base_channels = m + C
        self.acm_channels = self.base_channels * self.embed_dim
        self.acm_time = T - (self.embed_dim - 1) * self.embed_delay
        if self.acm_time <= 0:
            raise ValueError(
                f"ACM embedded time must be positive, got T={T}, "
                f"embed_dim={self.embed_dim}, embed_delay={self.embed_delay}"
            )
        d = self.acm_channels
        self.D = d * (d + 1) // 2
        # low-rank learnable spatial graph G = I + alpha * (U Vᵀ)
        self.U = nn.Parameter(0.01 * torch.randn(C, rank))
        self.V = nn.Parameter(0.01 * torch.randn(C, rank))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(self.D, affine=False)
        self.drop = nn.Dropout(dropout)
        self.head = DistanceEvidenceHead(self.D, n_classes,
                                         evidence_scale_init=evidence_scale_init)

    def mix(self, x):
        I = torch.eye(self.C, device=x.device)
        G = I + self.alpha * (self.U @ self.V.t())
        return torch.einsum("cd,bdt->bct", G, x)

    def _bnf(self, x):
        """Normalized SPD feature z = bn(logm(C))."""
        h = self.mix(x)                                  # [B, C, T]
        P = self.P.unsqueeze(0).expand(x.shape[0], -1, -1)
        sup = torch.cat([P, h], 1)                       # [B, m+C, T]
        sup = delay_embed(sup, self.embed_dim, self.embed_delay)
        f = logeig_vec(cov(sup, trace_norm=self.trace_norm))
        return self.bn(f)

    def forward(self, x):
        # distance head -> log-evidence (score), shape [B, n_classes].
        return self.head(self.drop(self._bnf(x)))


@torch.no_grad()
def init_prototypes(model, Xtr, ytr, evidence_scale_init=20.0, batch=128):
    """Data-initialize the distance head AFTER BN running stats are populated:
    proto[k] = mean of normalized tangent feats z=bn(f) for class k; tau = median
    within-class distance; scale = evidence_scale_init. All stay requires_grad."""
    X = torch.as_tensor(Xtr).to(DEV)
    y = torch.as_tensor(ytr).long().to(DEV)
    # 1) warmup: populate BN running stats with a train-mode pass over Xtr
    model.train()
    for i in range(0, len(X), batch):
        model._bnf(X[i:i + batch])
    # 2) eval-mode z = bn(f) (dropout off, running stats)
    model.eval()
    z = torch.cat([model._bnf(X[i:i + batch]) for i in range(0, len(X), batch)], 0)  # [N, D]
    head = model.head
    zp = F.normalize(z, dim=-1)                                       # cosine geometry (matches forward)
    for k in range(head.proto.shape[0]):
        mask = y == k
        if mask.any():
            head.proto.data[k] = zp[mask].mean(0)
    protoc = F.normalize(head.proto.data, dim=-1)
    d = (zp - protoc[y]).norm(dim=1)                                  # matches forward's distance
    med = d.median().clamp_min(1e-3)
    head.raw_tau.data.fill_(float(torch.log(torch.expm1((med - 1e-4).clamp_min(1e-3)))))
    head.raw_scale.data.fill_(_inv_softplus(evidence_scale_init))
    return model

"""
setpred_center_head.py  —  self-nominated-center set-prediction head for GDEGAN.

NEW FILE (non-destructive): consumes the frozen encoder interface `(h, X)` from
models/gda_hegnn.py and turns per-node features into a SET of binding pockets WITHOUT
DETR queries, WITHOUT Hungarian matching, and WITHOUT clustering.

Design (as discussed):
  Each pocket is a self-nominated CENTER RESIDUE.
  Three tiny per-node heads on features the encoder already produces:
    objectness  p[N]  = "am I in a pocket?"           (MLP on h)          -> reuses foreground label
    centerness  q[N]  = "am I my pocket's center?"    (MLP on h)          -> keypoint heatmap
    vote        v[N,3]= "offset to my pocket center"  (equivariant, from X[:, :3, :])
  Training: all per-node, NO matching (geometry assigns targets in the converter).
  Inference (decode_setpred): centers = threshold(q) + graph local-max (NOT clustering);
    each foreground residue is assigned to the nearest center by its voted point -> masks.

The l=1 block X[:, :3, :] was verified to be in (x,y,z) order (identity SH), so
`pred_center = pos + vote` is a valid Cartesian center prediction.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from equisite.utils.utils import MLP
from equisite.utils.losses import FocalLoss2, DiceLoss


class SetPredCenterHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, mlp_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.obj_head = MLP([hidden_dim, mlp_hidden, 1], activation=F.silu,
                            last_activation=None, dropout=dropout)
        self.center_head = MLP([hidden_dim, mlp_hidden, 1], activation=F.silu,
                               last_activation=None, dropout=dropout)
        # equivariant vote: linear MIX over the C channels of the l=1 vector block
        # X[:, :3, :] is [N, 3, C] -> Linear(C,1) on the last dim -> [N, 3] (bias=False keeps it equivariant)
        self.vote_lin = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, X: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        p_logit = self.obj_head(h).squeeze(-1)          # [N]
        q_logit = self.center_head(h).squeeze(-1)       # [N]
        l1 = X[:, :3, :]                                 # [N, 3, C]  (x,y,z-ordered equivariant vectors)
        vote = self.vote_lin(l1).squeeze(-1)            # [N, 3]  equivariant offset
        pred_center = pos + vote                        # [N, 3]
        return {"p_logit": p_logit, "q_logit": q_logit, "vote": vote, "pred_center": pred_center}


class SetPredCenterLoss(nn.Module):
    """All-per-node loss; no bipartite matching. Reuses the repo's FocalLoss2 / DiceLoss."""

    def __init__(self, obj_alpha: float = 0.25, obj_gamma: float = 2.0,
                 center_alpha: float = 0.95, center_gamma: float = 2.0,
                 w_obj: float = 1.0, w_dice: float = 2.0, w_center: float = 5.0, w_vote: float = 1.0,
                 dice_smooth: float = 1.0):
        super().__init__()
        self.obj_focal = FocalLoss2(alpha=obj_alpha, gamma=obj_gamma, logits=True)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.center_focal = FocalLoss2(alpha=center_alpha, gamma=center_gamma, logits=True)
        self.w = dict(obj=w_obj, dice=w_dice, center=w_center, vote=w_vote)

    def forward(self, out: Dict[str, torch.Tensor], batch) -> (torch.Tensor, Dict[str, float]):
        y = batch.y.float()                              # [N] foreground
        q = batch.q_centerness.float()                   # [N] center target
        vt = batch.vote_target                           # [N, 3]
        fg = y > 0.5

        l_obj = self.obj_focal(out["p_logit"], y)
        l_dice = self.dice(out["p_logit"], y)
        l_center = self.center_focal(out["q_logit"], q)
        if fg.any():
            l_vote = F.smooth_l1_loss(out["vote"][fg], vt[fg])
        else:
            l_vote = out["vote"].sum() * 0.0

        total = (self.w["obj"] * l_obj + self.w["dice"] * l_dice
                 + self.w["center"] * l_center + self.w["vote"] * l_vote)
        parts = {"obj": float(l_obj.detach()), "dice": float(l_dice.detach()),
                 "center": float(l_center.detach()), "vote": float(l_vote.detach())}
        return total, parts


def _graph_local_max(q: torch.Tensor, pos: torch.Tensor, cand: torch.Tensor, radius: float) -> torch.Tensor:
    """Keep a candidate center only if its q is >= q of every neighbor within `radius` (deterministic NMS,
    NOT clustering: single pass, no bandwidth search)."""
    if cand.sum() <= 1:
        return cand
    d = torch.cdist(pos, pos)                             # [Ng, Ng]
    neigh = (d <= radius)
    neigh.fill_diagonal_(False)
    qq = q.unsqueeze(0).expand_as(d).clone()
    qq[~neigh] = -1e9
    max_neigh = qq.max(dim=1).values
    return cand & (q >= max_neigh)


@torch.no_grad()
def decode_setpred(out: Dict[str, torch.Tensor], batch,
                   q_thresh: float = 0.5, p_thresh: float = 0.5,
                   local_max: bool = True, lm_radius: float = 8.0,
                   fallback_topk: int = 1) -> List[Dict[str, np.ndarray]]:
    """Per-graph decode -> list of dicts {labels[n], pocket_masks[n,Ng], scores[n], centers[n,3]}.
    No clustering: centers are a thresholded + local-max read of the centerness head; each foreground
    residue is assigned by argmin distance from its VOTED point to a (discrete) center residue."""
    p = torch.sigmoid(out["p_logit"])
    q = torch.sigmoid(out["q_logit"])
    cvote = out["pred_center"]                            # [N, 3] voted points
    pos = batch.pos
    results = []
    for g in range(batch.num_graphs):
        idx = (batch.batch == g).nonzero(as_tuple=True)[0]
        pg, qg, cg, posg = p[idx], q[idx], cvote[idx], pos[idx]
        Ng = idx.numel()

        cand = qg > q_thresh
        if local_max:
            cand = _graph_local_max(qg, posg, cand, lm_radius)
        center_ids = cand.nonzero(as_tuple=True)[0]
        if center_ids.numel() == 0:                       # always emit something rankable
            k = min(fallback_topk, Ng)
            center_ids = torch.topk(qg, k).indices
        n = center_ids.numel()
        centers_pos = posg[center_ids]                    # [n, 3]

        masks = torch.zeros((n, Ng), dtype=torch.bool, device=pos.device)
        fg_ids = (pg > p_thresh).nonzero(as_tuple=True)[0]
        if fg_ids.numel() > 0:
            d = torch.cdist(cg[fg_ids], centers_pos)      # voted point -> center residue
            assign = d.argmin(dim=1)
            masks[assign, fg_ids] = True
        for j, cid in enumerate(center_ids):              # a center always belongs to its own pocket
            masks[j, cid] = True

        scores = torch.empty(n, device=pos.device)
        centers_coord = torch.empty((n, 3), device=pos.device)
        for j in range(n):
            m = masks[j]
            scores[j] = pg[m].mean() if m.any() else qg[center_ids[j]]
            centers_coord[j] = cg[m].mean(0) if m.any() else centers_pos[j]

        results.append({
            "labels": np.zeros(n, dtype=np.int64),
            "pocket_masks": masks.cpu().numpy().astype(np.float32),   # [n, Ng]
            "scores": scores.cpu().numpy().astype(np.float32),
            "centers": centers_coord.cpu().numpy().astype(np.float32),
        })
    return results


# =====================================================================================
# exp-2: center-conditioned mask head (adds the missing PER-SITE mask supervision).
# Diagnosis (exp-1): merged-foreground BCE/Dice trained, per-site IoU evaluated -> masks
# were the bottleneck (oracle GT-centers+GT-fg AP=0.89 but learned-fg AP=0.27). This head
# lets each center paint its OWN mask, supervised per-site with Dice+BCE.
# =====================================================================================
class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int = 16, cutoff: float = 20.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = 1.0 / (2.0 * (cutoff / n_rbf) ** 2)

    def forward(self, d):                                    # [...] -> [..., n_rbf]
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class SetPredMaskHead(nn.Module):
    """Center-conditioned mask. For center residue c and node j:
        logit(c,j) = query(h_c)·key(h_j)/sqrt(D)  +  s·<X_c, X_j>  +  dist_bias(||pos_c-pos_j||)
    The equivariant inner product <X_c,X_j> is rotation-invariant (safe) and is our differentiator;
    `equiv_scale` starts at 0 (pure dot-product, UniSite-like) and is learned up. The distance bias
    injects the pocket-locality prior."""

    def __init__(self, hidden_dim: int = 128, n_rbf: int = 16, cutoff: float = 20.0, use_equiv: bool = True):
        super().__init__()
        self.use_equiv = use_equiv   # False -> pure-invariant mask head (equivariance ablation)
        self.query = MLP([hidden_dim, hidden_dim, hidden_dim], activation=F.silu, last_activation=None)
        self.key = MLP([hidden_dim, hidden_dim, hidden_dim], activation=F.silu, last_activation=None)
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.dist_mlp = MLP([n_rbf, hidden_dim // 2, 1], activation=F.silu, last_activation=None)
        self.equiv_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = hidden_dim ** -0.5

    def forward(self, h_g, X_g, pos_g, center_ids):
        """h_g[Ng,C], X_g[Ng,equi,C], pos_g[Ng,3], center_ids[n] (local) -> mask_logits[n, Ng]."""
        qc = self.query(h_g[center_ids])                     # [n, C]
        kj = self.key(h_g)                                   # [Ng, C]
        dot = (qc @ kj.t()) * self.scale                     # [n, Ng]
        equiv = torch.einsum("nec,mec->nm", X_g[center_ids], X_g) / X_g.shape[-1]   # [n, Ng] invariant
        d = torch.cdist(pos_g[center_ids], pos_g)            # [n, Ng]
        db = self.dist_mlp(self.rbf(d)).squeeze(-1)          # [n, Ng]
        eq = self.equiv_scale * equiv if self.use_equiv else 0.0
        return dot + eq + db + self.bias


def batched_dice_loss(logits, targets, smooth: float = 1.0):
    """Per-row (per-site) soft-Dice on [K, Ng] logits vs [K, Ng] targets, averaged over sites."""
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=1)
    union = p.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def _mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Greedy NMS over predicted masks (drop high-IoU duplicates). masks[n,Ng] bool -> keep idx."""
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        inter = (masks[i] & masks[rest]).sum(1).float()
        union = (masks[i] | masks[rest]).sum(1).float().clamp(min=1)
        order = rest[(inter / union) < iou_thr]
    return torch.tensor(keep, device=masks.device, dtype=torch.long)


@torch.no_grad()
def decode_setpred_mask(mask_head, h, X, pos, batch, q_logit,
                        q_thresh: float = 0.5, lm_radius: float = 8.0,
                        mask_thresh: float = 0.5, nms_iou: float = 0.5,
                        fallback_topk: int = 1):
    """exp-2 decode: centers = threshold(q)+local-max (NO clustering); each center paints its mask
    via the conditioned head; mask-NMS drops duplicates. -> per-graph list of AP-ready dicts."""
    q = torch.sigmoid(q_logit)
    results = []
    for g in range(batch.num_graphs):
        idx = (batch.batch == g).nonzero(as_tuple=True)[0]
        qg, hg, Xg, posg = q[idx], h[idx], X[idx], pos[idx]
        Ng = idx.numel()
        cand = _graph_local_max(qg, posg, qg > q_thresh, lm_radius)
        cids = cand.nonzero(as_tuple=True)[0]
        if cids.numel() == 0:
            cids = torch.topk(qg, min(fallback_topk, Ng)).indices
        ml = mask_head(hg, Xg, posg, cids)                   # [n, Ng]
        mp = torch.sigmoid(ml)
        masks = mp > mask_thresh
        masks[torch.arange(cids.numel(), device=pos.device), cids] = True
        denom = masks.sum(1).clamp(min=1).float()
        scores = qg[cids] * (mp * masks).sum(1) / denom
        centers = torch.stack([posg[masks[j]].mean(0) if masks[j].any() else posg[cids[j]]
                               for j in range(cids.numel())])
        keep = _mask_nms(masks, scores, nms_iou)
        masks, scores, centers = masks[keep], scores[keep], centers[keep]
        results.append({
            "labels": np.zeros(masks.shape[0], dtype=np.int64),
            "pocket_masks": masks.cpu().numpy().astype(np.float32),
            "scores": scores.cpu().numpy().astype(np.float32),
            "centers": centers.cpu().numpy().astype(np.float32),
        })
    return results

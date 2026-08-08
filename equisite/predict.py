"""High-level prediction API: checkpoint in, binding sites out.

    from equisite.predict import EquiSitePredictor
    p = EquiSitePredictor("checkpoints/equisite_split0.9.ckpt")
    sites = p.predict_pdb("examples/pdb/1fldA.pdb", chain="A")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import yaml

from equisite.featurize import Chain, ESMEmbedder, build_input, list_chains, parse_chain
from equisite.method.center_head import decode_setpred_mask
from equisite.method.lightning import SetPredLightningModule

__all__ = ["Site", "EquiSitePredictor", "DEFAULTS"]

#: Decode constants committed in the paper (validation-selected, never tuned on test).
DEFAULTS = dict(q_thresh=0.3, lm_radius=8.0, mask_thresh=0.4, nms_iou=0.5, fallback_topk=1)


@dataclass
class Site:
    """One predicted binding site."""
    score: float
    center: np.ndarray                     # [3] centroid of the predicted mask
    residue_indices: np.ndarray            # 0-based indices into the chain
    residue_ids: List[str] = field(default_factory=list)   # author numbering, e.g. "A:123"

    @property
    def n_residues(self) -> int:
        return int(len(self.residue_indices))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = ", ".join(f"{v:.1f}" for v in self.center)
        return f"Site(score={self.score:.3f}, {self.n_residues} residues, center=({c}))"


class EquiSitePredictor:
    """Loads a trained EquiSite checkpoint and predicts sites for a chain."""

    def __init__(self, checkpoint: str, config: Optional[str] = None,
                 device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = SetPredLightningModule.load_from_checkpoint(
            checkpoint, map_location=device)
        self.model.to(device).eval()
        self.decode_kw = dict(DEFAULTS)
        if config:
            cfg = yaml.safe_load(open(config))
            self.decode_kw.update(cfg.get("eval", {}).get("decode_mask", {}))
        self._esm: Optional[ESMEmbedder] = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def chains(pdb_path: str) -> List[str]:
        return list_chains(pdb_path)

    def _embedder(self, esm_device: Optional[str] = None) -> ESMEmbedder:
        if self._esm is None:
            self._esm = ESMEmbedder(device=esm_device or self.device)
        return self._esm

    def effective_decode(self, **overrides) -> dict:
        """Decode constants actually used, after applying non-None overrides."""
        return {**self.decode_kw, **{k: v for k, v in overrides.items() if v is not None}}

    # ------------------------------------------------------------------- predict
    def predict(self, chain: Chain, esm_features: np.ndarray, **overrides) -> List[Site]:
        """Run the model on an already-featurised chain."""
        kw = self.effective_decode(**overrides)
        batch = build_input(chain, esm_features).to(self.device)
        with torch.no_grad():
            h, X = self.model.model(batch)
            out = self.model.head(h, X, batch.pos)
            det = decode_setpred_mask(self.model.mask_head, h, X, batch.pos, batch,
                                      out["q_logit"], **kw)[0]
        masks = np.asarray(det["pocket_masks"], dtype=bool)
        order = np.argsort(-np.asarray(det["scores"]))          # rank by confidence
        sites = []
        for j in order:
            idx = np.where(masks[j])[0]
            sites.append(Site(score=float(det["scores"][j]),
                              center=np.asarray(det["centers"][j], dtype=np.float32),
                              residue_indices=idx,
                              residue_ids=[chain.res_ids[i] for i in idx]))
        return sites

    def predict_pdb(self, pdb_path: str, chain: Optional[str] = None,
                    esm_device: Optional[str] = None, **overrides) -> List[Site]:
        """Parse a structure file, embed with ESM-2, and predict. Requires ``fair-esm``."""
        ch = parse_chain(pdb_path, chain)
        feats = self._embedder(esm_device).embed(ch.sequence)
        return self.predict(ch, feats, **overrides)

"""Turn a raw structure file into the exact inputs EquiSite was trained on.

The model consumes, per chain:
  * C-alpha coordinates            [N, 3]
  * frozen ESM-2 650M embeddings   [N, 1280]   (layer 33, one row per residue)
  * residue-type indices           [N]

This mirrors ``equisite/data/process_coach_holo.py``, which is how the transfer
benchmarks (COACH420, HOLO4K-sc) were featurised for the paper, so predictions on a
user-supplied PDB follow the same path as the reported transfer numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from equisite.data.process_unisite import THREE_TO_ONE, ONE_TO_IDX

__all__ = ["Chain", "list_chains", "parse_chain", "ESMEmbedder", "build_input"]


# ----------------------------------------------------------------------------- parsing
@dataclass
class Chain:
    """One parsed polypeptide chain."""
    chain_id: str
    coords: np.ndarray          # [N, 3] C-alpha, float32
    sequence: str               # length N, one-letter ('X' for non-standard)
    res_ids: List[str]          # author residue numbers, e.g. "A:123" — for reporting
    n_residues: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Chain({self.chain_id!r}, {self.n_residues} residues)"


def _iter_residues(path: str):
    """Yield residues in file order with their C-alpha, honouring altloc and skipping H."""
    res: Dict[tuple, dict] = {}
    order: List[tuple] = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[16] not in (" ", "A"):                       # altloc: keep blank or 'A'
                continue
            elem = (line[76:78].strip() or line[12:16].strip()[:1])
            if elem == "H":
                continue
            key = (line[21], line[22:26].strip(), line[26].strip())
            if key not in res:
                res[key] = {"ca": None, "resname": line[17:20].strip(), "chain": line[21],
                            "resseq": line[22:26].strip(), "icode": line[26].strip()}
                order.append(key)
            if line[12:16].strip() == "CA":
                res[key]["ca"] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    for k in order:
        if res[k]["ca"] is not None:
            yield res[k]


def list_chains(path: str) -> List[str]:
    """Chain identifiers present in the file, in order of first appearance."""
    seen: List[str] = []
    for r in _iter_residues(path):
        if r["chain"] not in seen:
            seen.append(r["chain"])
    return seen


def parse_chain(path: str, chain_id: Optional[str] = None) -> Chain:
    """Extract one chain. With ``chain_id=None`` the first chain in the file is used."""
    residues = [r for r in _iter_residues(path)
                if chain_id is None or r["chain"] == chain_id]
    if not residues:
        avail = list_chains(path)
        raise ValueError(f"chain {chain_id!r} not found in {path}; available: {avail}")
    cid = residues[0]["chain"]
    coords = np.asarray([r["ca"] for r in residues], dtype=np.float32)
    sequence = "".join(THREE_TO_ONE.get(r["resname"], "X") for r in residues)
    res_ids = [f"{r['chain']}:{r['resseq']}{r['icode']}" for r in residues]
    return Chain(cid, coords, sequence, res_ids, len(residues))


# -------------------------------------------------------------------------------- ESM
class ESMEmbedder:
    """Frozen ESM-2 650M, layer 33 — identical to the encoder used to build the datasets.

    Requires ``fair-esm``; weights download once to the torch hub cache (~2.5 GB).
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = "cuda:0"):
        import torch
        import esm
        self.torch = torch
        self.device = device
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = 33 if "t33" in model_name else 6
        self.dim = 1280 if "t33" in model_name else 320
        self.model.to(device).eval()

    def embed(self, sequence: str) -> np.ndarray:
        """[len(sequence), dim] float32, BOS/EOS stripped."""
        _, _, toks = self.batch_converter([("p", sequence)])
        toks = toks.to(self.device)
        with self.torch.no_grad():
            out = self.model(toks, repr_layers=[self.repr_layer])
        rep = out["representations"][self.repr_layer][0, 1:len(sequence) + 1]
        assert rep.shape[0] == len(sequence), (rep.shape, len(sequence))
        return rep.cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------------- model input
def build_input(chain: Chain, esm_features: np.ndarray):
    """Chain + ESM features -> a single-graph PyG batch ready for the model."""
    import torch
    from torch_geometric.data import Data, Batch

    if esm_features.shape[0] != chain.n_residues:
        raise ValueError(f"ESM rows ({esm_features.shape[0]}) != residues ({chain.n_residues})")

    z = np.asarray([ONE_TO_IDX.get(c, -1) + 1 for c in chain.sequence], dtype=np.int64)
    d = Data()
    d.pos = torch.from_numpy(np.ascontiguousarray(chain.coords, dtype=np.float32))
    d.esm = torch.from_numpy(np.ascontiguousarray(esm_features, dtype=np.float32))
    d.z = torch.from_numpy(z)
    d.num_nodes = chain.n_residues
    return Batch.from_data_list([d])

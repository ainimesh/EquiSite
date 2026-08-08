#!/usr/bin/env python
"""Run EquiSite on bundled examples — no ESM-2 install, no downloads.

    python scripts/demo.py                 # all bundled examples
    python scripts/demo.py --name 1cx4A    # just one
    python scripts/demo.py --min-score 0.5 # only confident sites

The ESM-2 embeddings for these chains are precomputed and shipped in
``examples/precomputed/``, so this exercises the full model and decode without the
2.5 GB language model. To run on your own structure, use ``scripts/inference.py``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equisite.featurize import Chain                      # noqa: E402
from equisite.predict import DEFAULTS, EquiSitePredictor  # noqa: E402

# what the bundled chains actually contain, for context in the printout
NOTES = {
    "1fldA": "flavodoxin, 1 known site",
    "2flvA": "flavodoxin homolog, 1 known site",
    "1cx4A": "2 known sites, 28.9 A apart",
    "1ja9A": "2 known sites, 12.1 A apart",
    "1ex8A": "single-chain example",
}


def load(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    chain = Chain(chain_id=str(d["chain_id"]),
                  coords=d["coords"].astype(np.float32),
                  sequence=str(d["sequence"]),
                  res_ids=[str(x) for x in d["res_ids"]],
                  n_residues=int(d["coords"].shape[0]))
    return chain, d["esm"].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="EquiSite demo on bundled examples.")
    ap.add_argument("--name", default=None, help="run a single example, e.g. 1cx4A")
    ap.add_argument("--checkpoint", default="checkpoints/equisite_split0.9.ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--tau-q", type=float, default=None)
    ap.add_argument("--tau-m", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--tau-nms", type=float, default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(here, "examples", "precomputed", "*.npz")))
    if args.name:
        files = [f for f in files if os.path.basename(f).startswith(args.name)]
        if not files:
            print(f"error: no bundled example named {args.name!r}", file=sys.stderr)
            return 1
    if not files:
        print("error: no examples found in examples/precomputed/", file=sys.stderr)
        return 1

    kw = dict(device=args.device) if args.device else {}
    predictor = EquiSitePredictor(os.path.join(here, args.checkpoint), **kw)
    print(f"EquiSite  |  {predictor.device}  |  decode "
          f"{predictor.effective_decode(q_thresh=args.tau_q, lm_radius=args.rho, mask_thresh=args.tau_m, nms_iou=args.tau_nms)}")
    print(f"paper defaults: {DEFAULTS}\n" + "=" * 72)

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        chain, feats = load(f)
        ov = dict(q_thresh=args.tau_q, lm_radius=args.rho,
                  mask_thresh=args.tau_m, nms_iou=args.tau_nms)
        sites = predictor.predict(chain, feats, **ov)
        kept = [s for s in sites if s.score >= args.min_score]
        note = NOTES.get(name, "")
        print(f"\n{name}   chain {chain.chain_id}   {chain.n_residues} residues"
              + (f"   ({note})" if note else ""))
        print(f"  -> {len(kept)} predicted site(s)"
              + (f" at score >= {args.min_score}" if args.min_score else ""))
        for k, s in enumerate(kept, 1):
            cx, cy, cz = s.center
            head = " ".join(s.residue_ids[:10])
            more = f" ... (+{s.n_residues - 10} more)" if s.n_residues > 10 else ""
            print(f"     site {k}:  score {s.score:.3f}   {s.n_residues:3d} residues"
                  f"   center ({cx:7.2f},{cy:7.2f},{cz:7.2f})")
            print(f"               {head}{more}")
    print("\n" + "=" * 72)
    print("Tip: raise --min-score (e.g. 0.5) to keep only confident sites; the decode is")
    print("     deliberately recall-favouring, so low-scoring extras are expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

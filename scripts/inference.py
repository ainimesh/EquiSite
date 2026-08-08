#!/usr/bin/env python
"""Predict ligand-binding sites for one chain of a structure file.

    # 1. see what chains the file contains
    python scripts/inference.py --pdb examples/pdb/1cx4A.pdb --list-chains

    # 2. predict (EquiSite is a per-chain model)
    python scripts/inference.py --pdb examples/pdb/1cx4A.pdb --chain A

    # 3. change the decode thresholds; paper defaults are used when omitted
    python scripts/inference.py --pdb my.pdb --chain B --tau-q 0.4 --tau-m 0.5 --min-score 0.5

Requires ``fair-esm`` (weights download once, ~2.5 GB). To see the output format without
installing ESM-2, run ``python scripts/demo.py`` instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equisite.featurize import list_chains, parse_chain          # noqa: E402
from equisite.predict import DEFAULTS, EquiSitePredictor         # noqa: E402

CKPT = "checkpoints/equisite_split0.9.ckpt"


def main() -> int:
    ap = argparse.ArgumentParser(description="EquiSite: multi-site binding-site prediction.")
    ap.add_argument("--pdb", required=True, help="structure file (.pdb)")
    ap.add_argument("--chain", default=None,
                    help="chain to predict; omit to be prompted, or use --list-chains first")
    ap.add_argument("--list-chains", action="store_true", help="print the chains and exit")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--device", default=None, help="cuda:0 | cpu (default: cuda if available)")
    ap.add_argument("--out", default=None, help="write the sites to this JSON file")
    g = ap.add_argument_group("decode thresholds (paper defaults shown)")
    g.add_argument("--tau-q", type=float, default=None,
                   help=f"centerness threshold (default {DEFAULTS['q_thresh']})")
    g.add_argument("--rho", type=float, default=None,
                   help=f"local-maximum radius in Angstrom (default {DEFAULTS['lm_radius']})")
    g.add_argument("--tau-m", type=float, default=None,
                   help=f"mask threshold (default {DEFAULTS['mask_thresh']})")
    g.add_argument("--tau-nms", type=float, default=None,
                   help=f"mask-IoU suppression threshold (default {DEFAULTS['nms_iou']})")
    g.add_argument("--min-score", type=float, default=0.0,
                   help="only report sites at or above this confidence (default 0.0: report all)")
    args = ap.parse_args()

    if not os.path.exists(args.pdb):
        print(f"error: no such file: {args.pdb}", file=sys.stderr)
        return 1

    chains = list_chains(args.pdb)
    if not chains:
        print(f"error: no protein chains with C-alpha atoms found in {args.pdb}", file=sys.stderr)
        return 1

    if args.list_chains or args.chain is None:
        print(f"\n{os.path.basename(args.pdb)} contains {len(chains)} chain(s):")
        for c in chains:
            ch = parse_chain(args.pdb, c)
            print(f"   {c}   {ch.n_residues:5d} residues   {ch.sequence[:48]}"
                  f"{'...' if ch.n_residues > 48 else ''}")
        if args.list_chains:
            return 0
        if len(chains) == 1:
            args.chain = chains[0]
            print(f"\nonly one chain present; using {args.chain}")
        else:
            try:
                args.chain = input(f"\nselect a chain {chains}: ").strip()
            except EOFError:
                print("\nerror: no chain given; re-run with --chain", file=sys.stderr)
                return 1
            if args.chain not in chains:
                print(f"error: {args.chain!r} is not one of {chains}", file=sys.stderr)
                return 1

    kw = dict(device=args.device) if args.device else {}
    predictor = EquiSitePredictor(args.checkpoint, **kw)
    print(f"\nrunning EquiSite on chain {args.chain} ({predictor.device}) ...")
    ov = dict(q_thresh=args.tau_q, lm_radius=args.rho,
              mask_thresh=args.tau_m, nms_iou=args.tau_nms)
    eff = predictor.effective_decode(**ov)
    sites = predictor.predict_pdb(args.pdb, args.chain, **ov)
    kept = [s for s in sites if s.score >= args.min_score]

    print(f"\n{len(kept)} site(s)" + (f" at score >= {args.min_score}" if args.min_score else "")
          + f"   [decode: {eff}]\n")
    for k, s in enumerate(kept, 1):
        cx, cy, cz = s.center
        print(f"  Site {k}   score {s.score:.3f}   {s.n_residues} residues"
              f"   center ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        print(f"           residues: {' '.join(s.residue_ids)}\n")

    if args.out:
        payload = {"structure": os.path.basename(args.pdb), "chain": args.chain,
                   "decode": eff,
                   "sites": [{"rank": k, "score": s.score, "center": s.center.tolist(),
                              "n_residues": s.n_residues, "residues": s.residue_ids}
                             for k, s in enumerate(kept, 1)]}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Reproduce the reported EquiSite numbers from the released checkpoint.

    # headline row of Table 1 (UniSite-DS test, sequence-space protocol)
    python scripts/evaluate.py --data data/unisite_setpred_processed.pkl \
                               --splits data/unisite-ds-v1/_pkl_files

    # structure-space only (no split directory needed)
    python scripts/evaluate.py --data data/unisite_setpred_processed.pkl --space structure

See docs/DATA.md for how to obtain and build the processed dataset.

Sequence space scores masks over the L canonical UniProt positions (the protocol under which
all published baselines were computed); structure space scores them over the N residues resolved
in the representative structure. The two agree to within 0.003 AP.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equisite.method.center_head import decode_setpred_mask     # noqa: E402
from equisite.method.lightning import SetPredLightningModule    # noqa: E402
from equisite.utils.mean_ap import eval_map                     # noqa: E402

REPORTED = {"0.9": (0.6237, 0.4674), "0.7": (0.6221, 0.4534), "0.5": (0.5575, 0.3834)}
BASELINE = {"0.9": (0.5603, 0.3835), "0.7": (0.5579, 0.3734), "0.5": (0.4677, 0.2801)}


def load_split(splits_dir: str, split: str) -> set:
    path = os.path.join(splits_dir, f"_test_{split}.csv")
    with open(path) as fh:
        rows = csv.reader(fh)
        next(rows, None)
        return {r[0] for r in rows if r}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate EquiSite on UniSite-DS.")
    ap.add_argument("--data", required=True, help="processed .pkl (see docs/DATA.md)")
    ap.add_argument("--splits", default=None,
                    help="directory with _test_<split>.csv and per-protein .pkl files "
                         "(required for sequence-space)")
    ap.add_argument("--split", default="0.9", choices=["0.9", "0.7", "0.5"])
    ap.add_argument("--space", default="auto", choices=["auto", "sequence", "structure"])
    ap.add_argument("--checkpoint", default="checkpoints/equisite_split0.9.ckpt")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    space = args.space
    if space == "auto":
        space = "sequence" if args.splits else "structure"
    if space == "sequence" and not args.splits:
        print("error: --splits is required for the sequence-space protocol", file=sys.stderr)
        return 1

    print(f"loading model   : {args.checkpoint}")
    model = SetPredLightningModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device).eval()
    dec = model.hparams.eval_config["decode_mask"]

    print(f"loading data    : {args.data}")
    records = pickle.load(open(args.data, "rb"))
    by_name = {r["name"]: r for r in records}
    names = sorted(load_split(args.splits, args.split)) if args.splits else sorted(by_name)
    names = [n for n in names if n in by_name]
    print(f"split {args.split}      : {len(names)} test proteins   |   protocol: {space} space\n")

    from torch_geometric.data import Batch, Data
    dets, annos, skipped = [], [], 0
    for i, name in enumerate(names):
        r = by_name[name]
        d = Data()
        d.pos = torch.from_numpy(np.asarray(r["coordinates"], np.float32))
        d.esm = torch.from_numpy(np.asarray(r["res_features_esm"], np.float32))
        d.z = torch.from_numpy(np.asarray(r["z"], np.int64))
        d.num_nodes = int(r["num_residues"])
        batch = Batch.from_data_list([d]).to(args.device)
        with torch.no_grad():
            h, X = model.model(batch)
            out = model.head(h, X, batch.pos)
            det = decode_setpred_mask(model.mask_head, h, X, batch.pos, batch,
                                      out["q_logit"], **dec)[0]
        masks = np.asarray(det["pocket_masks"], np.float32)
        gt = np.asarray(r["site_masks"], np.float32)

        if space == "sequence":
            gpath = os.path.join(args.splits, f"{name}.pkl")
            if not os.path.exists(gpath):
                skipped += 1
                continue
            gt_seq = np.asarray(pickle.load(open(gpath, "rb"))["target"]["pocket_masks"], np.float32)
            up, L = np.asarray(r["uniprot_positions"], np.int64), gt_seq.shape[1]
            proj = np.zeros((masks.shape[0], L), np.float32)
            if masks.shape[0]:
                proj[:, up] = masks
            masks, gt = proj, gt_seq

        dets.append({"labels": np.zeros(masks.shape[0], np.int64),
                     "scores": np.asarray(det["scores"], np.float32),
                     "pocket_masks": masks})
        annos.append({"labels": np.zeros(gt.shape[0], np.int64), "pocket_masks": gt})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)

    ap50 = eval_map(dets, annos, num_classes=1, iou_thr=0.5, nproc=1)[0]
    ap30 = eval_map(dets, annos, num_classes=1, iou_thr=0.3, nproc=1)[0]

    print(f"\n{'=' * 60}\nEquiSite | UniSite-DS test, split {args.split} | {space} space")
    print(f"{'=' * 60}")
    print(f"  proteins scored : {len(dets)}" + (f"   ({skipped} skipped)" if skipped else ""))
    print(f"  AP@0.3          : {ap30:.4f}")
    print(f"  AP@0.5          : {ap50:.4f}")
    if space == "sequence" and args.split in REPORTED:
        p30, p50 = REPORTED[args.split]
        b30, b50 = BASELINE[args.split]
        ok = abs(ap50 - p50) < 5e-4 and abs(ap30 - p30) < 5e-4
        print(f"\n  paper           : AP@0.3 {p30:.4f}   AP@0.5 {p50:.4f}"
              f"   -> {'MATCH' if ok else 'MISMATCH'}")
        print(f"  UniSite-3D      : AP@0.3 {b30:.4f}   AP@0.5 {b50:.4f}"
              f"   (margin +{ap30 - b30:.4f} / +{ap50 - b50:.4f})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

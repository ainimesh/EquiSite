# EquiSite

**Lightweight SE(3)-equivariant, queryless set prediction for multi-site protein–ligand binding sites.**

<p align="center">
  <img src="assets/figures/equisite_architecture.pdf" alt="EquiSite architecture" width="100%">
</p>

A protein often binds ligands at more than one pocket, so finding binding sites is a *set prediction*
problem: return every site, each as a set of residues, without knowing in advance how many there are.
The dominant approach borrows DETR — a fixed pool of learned queries plus Hungarian matching. EquiSite
removes both. **Every residue nominates itself as a candidate site center**, confident local maxima
survive, and each survivor paints its own mask. The number of predicted sites is an output of the
model, not a budget fixed by the architecture.

| | UniSite-3D (previous SOTA) | **EquiSite** |
|---|---|---|
| AP@0.3 / AP@0.5 (UniSite-DS) | 0.5603 / 0.3835 | **0.6237 / 0.4674** |
| Trainable parameters | 32.8 M | **1.2 M** (27× fewer) |
| Training hardware | 8 GPUs | **1 GPU** |
| Inference latency | 50.1 ms/protein | **9.0 ms/protein** (5.6× faster) |
| Site count | capped at 32 queries | **unbounded** |
| Rotation guarantee | none | **exact SE(3)-equivariance** |

---

## Quick start — no data, no ESM-2

Five example chains ship with their ESM-2 embeddings precomputed, so you can run the full model
and decode immediately:

```bash
git clone https://github.com/<your-org>/EquiSite2.git && cd EquiSite2
pip install -r requirements.txt          # fair-esm is only needed for your own structures
python scripts/demo.py --min-score 0.5
```

```
1fldA   chain A   138 residues   (flavodoxin, 1 known site)
  -> 1 predicted site(s) at score >= 0.5
     site 1:  score 0.839    19 residues   center (  26.64,  18.44,  13.75)
               A:6 A:7 A:8 A:9 A:10 A:11 A:12 A:54 A:55 A:56 ... (+9 more)

1cx4A   chain A   275 residues   (2 known sites, 28.9 A apart)
  -> 2 predicted site(s) at score >= 0.5
     site 1:  score 0.766    16 residues   center (  93.77,  58.55,  -5.09)
     site 2:  score 0.764    19 residues   center ( 114.15,  61.98,  14.49)
```

---

## Predict on your own structure

EquiSite is a **per-chain** model. Inspect the file, pick a chain, predict:

```bash
# 1. what chains are in there?
python scripts/inference.py --pdb my_protein.pdb --list-chains

# 2. predict (omit --chain to be prompted)
python scripts/inference.py --pdb my_protein.pdb --chain A

# 3. tune the decode, write JSON
python scripts/inference.py --pdb my_protein.pdb --chain A \
       --tau-q 0.4 --tau-m 0.5 --min-score 0.5 --out sites.json
```

This needs `fair-esm` (weights download once, ~2.5 GB) because the model consumes frozen
ESM-2 650M embeddings. The featuriser mirrors how the transfer benchmarks were built, so a raw
PDB follows the same path as the reported COACH420 / HOLO4K-sc numbers.

### Decode thresholds

All four are **selected on validation and frozen** for every dataset, seed and baseline in the
paper. None affects training — no gradient depends on them, so changing one re-decodes a fixed
model without retraining.

| flag | symbol | default | what it does |
|---|---|---|---|
| `--tau-q` | $\tau_q$ | 0.30 | minimum centerness to be a candidate center |
| `--rho` | $\rho$ | 8.0 Å | radius for local-maximum suppression of nearby centers |
| `--tau-m` | $\tau_m$ | 0.40 | mask threshold: a residue joins a site above this probability |
| `--tau-nms` | $\tau_{\text{nms}}$ | 0.50 | drop a mask overlapping a higher-scoring one above this IoU |
| `--min-score` | — | 0.0 | reporting filter only; does not change the prediction |

The decode is deliberately **recall-favouring** — it emits ~4.5 sites per protein against 1.6 true
ones — but the scores rank well (AUROC 0.95). Use `--min-score 0.5` when you want precision.

### Python API

```python
from equisite.predict import EquiSitePredictor

p = EquiSitePredictor("checkpoints/equisite_split0.9.ckpt")
print(p.chains("my_protein.pdb"))                     # ['A', 'B']
for site in p.predict_pdb("my_protein.pdb", chain="A"):
    print(site.score, site.center, site.residue_ids)
```

---

## Reproduce the paper

Download the data (UniSite-DS from HuggingFace, COACH420/HOLO4K-sc from Zenodo — see
[docs/DATA.md](docs/DATA.md) for links, build commands and citations), then:

```bash
python scripts/evaluate.py \
    --data   data/unisite_setpred_processed.pkl \
    --splits data/unisite-ds-v1/_pkl_files \
    --split  0.9
```

```
EquiSite | UniSite-DS test, split 0.9 | sequence space
  proteins scored : 2293
  AP@0.3          : 0.6237
  AP@0.5          : 0.4674
  paper           : AP@0.3 0.6237   AP@0.5 0.4674   -> MATCH
  UniSite-3D      : AP@0.3 0.5603   AP@0.5 0.3835   (margin +0.0634 / +0.0839)
```

`--split 0.7` and `--split 0.5` reproduce the stricter sequence-identity rows;
`--space structure` scores over observed residues instead of UniProt positions (the two agree to
within 0.003 AP).

### Retrain

```bash
python scripts/train.py --config configs/equisite.yaml
```

One GPU, ≤150 epochs with early stopping on validation AP@0.5. `configs/equisite.yaml` is the exact
configuration behind the released checkpoint.

---

## How it works

```
protein  ──►  Cα graph (10 Å, 64 neighbours)  +  frozen ESM-2 features
                            │
                   equivariant encoder  (GDEGAN, 4 layers, 1.1 M params)
                            │
              h [N×128] invariant        X [N×8×128] equivariant
                            │
              ┌─────────────┴──────────── center nomination head ────────────┐
              │  Nominate   every residue scores itself: p, q, v             │
              │  Select     q > τ_q and local max within ρ                   │
              │  Delineate  each survivor paints its own mask, σ(u) > τ_m    │
              │  Suppress   drop duplicate masks at IoU > τ_nms              │
              └──────────────────────────────────────────────────────────────┘
                            │
                    {(score, center, residue mask)}   — count decided by the model
```

Nothing here solves an assignment problem. During training, each ground-truth site elects **one of
its own residues** (the one nearest its centroid) as its center, so the correspondence between
prediction and label is read off the data instead of being computed by Hungarian matching.

---

## Repository layout

```
equisite/
├── backbone/gda_hegnn.py     equivariant encoder -> (h, X)
├── method/center_head.py     nominate / select / delineate / suppress + losses
├── method/lightning.py       training and evaluation module
├── data/                     dataset conversion (UniSite-DS, COACH420, HOLO4K)
├── featurize.py              PDB -> chain -> Cα + ESM-2 -> model input
└── predict.py                high-level predictor API
scripts/
├── demo.py                   bundled examples, no ESM required
├── inference.py              predict on your own structure
├── evaluate.py               reproduce the reported numbers
└── train.py                  retrain from scratch
checkpoints/equisite_split0.9.ckpt   released model (14 MB) behind Table 1
examples/                     5 chains: PDB files + precomputed ESM embeddings
```

---

## Limitations

* Accuracy falls with site multiplicity (AP@0.5 0.665 at *K*=1 vs 0.121 at *K*≥4).
* The decode over-predicts count (mean 4.5 vs 1.6 true); scores rank well but are not
  probability-calibrated (ECE 0.26). Use a score or top-*K* cutoff when precision matters.
* Masks live on residues resolved in the input structure — a site whose residues are unresolved
  cannot be recovered.
* Inference embeds the chain's own observed sequence, whereas UniSite-DS training embedded the full
  canonical UniProt sequence. For chains that are large fragments of their parent entry this is a
  close match; for short fragments the features shift slightly.

## Citation

```bibtex
<FILL IN: bibtex once the paper is public>
```

## License

MIT — see [LICENSE](LICENSE). ESM-2 weights and the benchmark datasets carry their own licenses.

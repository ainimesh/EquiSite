# Datasets

No dataset files are shipped with this repository. This page lists where each benchmark comes
from and how to convert it into the format `scripts/evaluate.py` expects.

If you only want to see the model run, none of this is needed — `python scripts/demo.py` uses the
five example chains bundled in `examples/`.

---

## 1. UniSite-DS — training and headline evaluation

UniProt-centric and multi-site: annotations for a protein are pooled across every PDB structure
mapped to the same UniProt entry, so a protein's full set of sites is labelled rather than only the
ligands resolved in one crystal. 11,510 proteins, 31.9% with two or more sites (mean 1.59, max 38).

**Download:** <https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1/tree/main>

```bash
# e.g. with the HuggingFace CLI
huggingface-cli download quanlin-wu/unisite-ds_v1 --repo-type dataset --local-dir data/unisite-ds-v1
```

Expected layout:

```text
data/unisite-ds-v1/
└── _pkl_files/
    ├── _train_0.9.csv  _test_0.9.csv        # official sequence-identity splits
    ├── _train_0.7.csv  _test_0.7.csv
    ├── _train_0.5.csv  _test_0.5.csv
    └── <UNIPROT_ID>.pkl                     # sequence, pocket masks, map_dict, representative PDB
```

### Build the processed file

EquiSite consumes C-alpha coordinates plus frozen ESM-2 embeddings, so the release is converted
once. This is the slow step — it runs ESM-2 over every sequence.

```bash
python -m equisite.data.process_unisite \
    --dataset_dir data/unisite-ds-v1 \
    --out         data/unisite_setpred_processed.pkl \
    --device      cuda:0
```

`--no_esm` gives a fast geometry/label-only pass for sanity checking (the model cannot be evaluated
without ESM features). The converter prints a coverage report; expect ~99.55% of annotated binding
residues to survive projection onto the observed structure, with 58 pockets and one protein
(`P11140`) dropped because no annotated residue is resolved.

---

## 2. COACH420 and HOLO4K-sc — zero-shot transfer

PDB-centric single-chain benchmarks, evaluated **without retraining**. These are the *mlig* subsets
introduced with P2Rank; we take the prepared copies released by VN-EGNN, which is also how the
UniSite comparison was set up.

**Download:** <https://zenodo.org/records/17365855>

Expected layout after unpacking:

```text
data/coach420/raw/<pdbid><chain>/protein.pdb, ligand_*.pdb
data/holo4k/raw/<pdbid><chain>/protein.pdb,   ligand_*.pdb
```

### Build the processed files

```bash
python -m equisite.data.process_coach_holo \
    --raw_dir data/coach420/raw --out data/coach420_setpred.pkl \
    --single_chain_only --device cuda:0

python -m equisite.data.process_coach_holo \
    --raw_dir data/holo4k/raw   --out data/holo4k_sc_setpred.pkl \
    --single_chain_only --device cuda:0
```

`--single_chain_only` keeps entries whose directory name is `pdbid`+one chain, which is what makes
HOLO4K **-sc** (single chain) and matches the reported numbers.

---

## 3. AlphaFold structures — robustness experiment (optional)

Predicted structures for the UniSite-DS test entries, fetched from AlphaFold DB by UniProt
accession. Used only for the robustness analysis; not needed to reproduce Table 1.

**Download:** `https://alphafold.ebi.ac.uk/files/AF-<ACCESSION>-F1-model_v4.pdb`

---

## Citing the data

If you use these benchmarks, please cite the original sources as well as EquiSite.

```bibtex
@article{unisite2025,
  title  = {UniSite: The First Cross-Structure Dataset and Learning Framework
            for End-to-End Ligand Binding Site Detection},
  author = {Wu, Quanlin and others},          % TODO: complete author list from arXiv:2506.03237
  journal= {arXiv preprint arXiv:2506.03237},
  year   = {2025},
  url    = {https://arxiv.org/abs/2506.03237}
}

@inproceedings{vnegnn2024,
  title     = {{VN-EGNN}: {E}(3)-Equivariant Graph Neural Networks with Virtual Nodes
               Enhance Protein Binding Site Identification},
  author    = {Sestak, Florian and Schneckenreiter, Lisa and Brandstetter, Johannes
               and Hochreiter, Sepp and Mayr, Andreas and Klambauer, G{\"u}nter},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  url       = {https://zenodo.org/records/17365855}
}

@article{p2rank2018,
  title   = {{P2Rank}: machine learning based tool for rapid and accurate prediction of
             ligand binding sites from protein structure},
  author  = {Krivák, Radoslav and Hoksza, David},
  journal = {Journal of Cheminformatics},
  volume  = {10},
  number  = {1},
  pages   = {39},
  year    = {2018}
}
```

> Two entries carry a caveat worth checking before you submit: the UniSite **author list** is a
> placeholder (complete it from arXiv:2506.03237), and the VN-EGNN entry should be verified against
> the published version — COACH420 and HOLO4K-sc originate as P2Rank's *mlig* subsets, so **cite
> both** VN-EGNN (the copy you downloaded) and P2Rank (the original benchmark).

---

## Disk and time

| step | approximate cost |
|---|---|
| UniSite-DS raw download | a few GB |
| ESM-2 650M weights (first run) | 2.5 GB, cached once by `torch.hub` |
| Processing UniSite-DS with ESM | several hours on one GPU; output ~20 GB |
| COACH420 / HOLO4K-sc processing | minutes to ~1 hour |
| Running `scripts/demo.py` | nothing — embeddings are bundled |

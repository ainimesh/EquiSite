# EquiSite, end to end — a bit-by-bit walkthrough with a worked example

This explains the **whole pipeline** in one read: what goes in, what the encoder produces,
how a variable number of binding sites becomes training supervision (with the actual
calculations), and how a fresh protein is decoded at test time. A tiny toy protein carries
the numbers so every step is concrete.
---

## 0. The one-paragraph version

> A protein (3D structure + ESM language-model features) goes into one **equivariant encoder
> (GDEGAN)**, which gives every residue two feature vectors: an **invariant** one `h` and an
> **equivariant** one `X`. From these, **every residue predicts three things about itself**:
> "am I in a pocket?" (objectness), "am I at a pocket's center?" (centerness), and "where is my
> pocket's center?" (a 3D vote). A handful of residues that look most like centers are selected,
> and each selected center **paints its own site's residue mask**. No learned queries, no
> Hungarian matching — the *center residue itself* is the assignment.

---

## 1. The running example — a toy protein "P8"

8 residues, laid out on a line so the arithmetic is clean (real proteins are 3D pockets; the
logic is identical). Cα coordinates in Å:

| residue | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| pos (x) | −5 | 0 | 1 | 2 | 5 | 10 | 11 | 16 |

**Ground truth: 2 binding sites** (K = 2):
- **Site A** = residues {2, 3, 4}, centroid at x = 1 → nearest residue = **3** (its *center residue*).
- **Site B** = residues {6, 7}, centroid at x = 10.5 → nearest residue = **6**.
- Residues 1, 5, 8 are **background** (not in any site).

This "K = 2, these residues, these centers" is *all* the label information. Everything below is
derived from it.

---

## 2. Shapes cheat-sheet

For a protein with `N` residues (`N = 8` here):

| tensor | shape | what it is |
|---|---|---|
| `pos` | `[N, 3]` | Cα coordinates (input) |
| `esm` | `[N, 1280]` | frozen ESM2-650M per-residue features (input) |
| `h` | `[N, 128]` | **invariant** per-residue features (encoder output) |
| `X` | `[N, 8, 128]` | **equivariant** per-residue steerable features, ℓ=1 (3 comps) + ℓ=2 (5 comps) |
| `p` | `[N]` | objectness (per residue) |
| `q` | `[N]` | centerness (per residue) |
| `v` | `[N, 3]` | center vote (per residue) |
| `site_masks` | `[K, N]` | one binary residue mask **per ground-truth site** |
| `center_res_idx` | `[K]` | the center residue of each site |

Note the one that depends on the number of sites: **`site_masks` has `K` rows** — one per site.

---

## 3. Input → encoder → `h` and `X`

1. **Build a graph.** Connect residues whose Cα are within **10 Å** (a radius graph). Node
   features = the ESM vectors; edges carry the geometry (relative positions).
2. **Run GDEGAN** (4 layers of equivariant message passing). Each layer lets residues exchange
   information with neighbors using a **variance-adaptive attention** (attention is sharper where
   the local neighborhood is geometrically heterogeneous — i.e., at pocket walls).
3. **Out come two streams per residue:**
   - **`h` (invariant, `[N,128]`)** — a "what kind of residue am I, in what chemical/structural
     context" descriptor. Rotate the protein and `h` **does not change**.
   - **`X` (equivariant, `[N,8,128]`)** — a "geometric/directional" descriptor (ℓ=1 = directions,
     ℓ=2 = local shape). Rotate the protein and `X` **rotates with it**.

> Why two streams? Things that should be **rotation-invariant** (is this a pocket? which residues?)
> read from `h`. Things that are **geometric** (which *direction* is the center?) read from `X`.

---

## 4. What every residue predicts (the heads)

From `h` and `X`, **each residue i** produces three self-descriptions:

| output | formula | meaning |
|---|---|---|
| objectness `pᵢ` | `σ(MLP(hᵢ))` | probability residue i is inside *some* pocket |
| centerness `qᵢ` | `σ(MLP(hᵢ))` | is i the **center residue** of a site? (binary target: 1 at centers) |
| vote `vᵢ` | `W · Xᵢ⁽ℓ⁼¹⁾` → center = `posᵢ + vᵢ` | a 3D arrow from i to its pocket's center |

`p` and `q` come from the invariant `h` (they're scalars — must be rotation-invariant). The vote
`v` comes from the **equivariant** `X` (it's a 3D vector — must rotate with the protein). This is
the one place equivariance is *structurally required*.

---

## 5. Training — turning K sites into targets (the "depends on #sites" part)

From the label ("K=2, sites A and B, centers 3 and 6") we build a supervision target for every
head. **For P8:**

**(a) Objectness target `y`** — 1 if the residue is in any site, else 0:
```
residue:  1  2  3  4  5  6  7  8
y      :  0  1  1  1  0  1  1  0      (foreground = union of all sites)
```

**(b) Centerness target `q`** — a **binary** indicator: 1 at each site's chosen center residue, 0
everywhere else (there is no distance decay — it is one-hot per site):
```
residue:  1  2  3  4  5  6  7  8
q      :  0  0  1  0  0  1  0  0      (residues 3 and 6 are the two center residues)
```

**(c) Vote target** — for each foreground residue, the displacement `center − posᵢ`:
```
res 2: (1 − 0)  = +1        res 6: (10.5 − 10) = +0.5
res 3: (1 − 1)  =  0        res 7: (10.5 − 11) = −0.5
res 4: (1 − 2)  = −1        (background residues: no vote target)
```

**(d) Per-site masks — this is the part that scales with K.** One binary mask **per site**:
```
Mask_A = [0,1,1,1,0,0,0,0]     (site A = residues 2,3,4)
Mask_B = [0,0,0,0,0,1,1,0]     (site B = residues 6,7)
```
K = 2 → two masks. If the protein had 3 sites, there'd be three. There is **no fixed budget** —
the number of mask targets simply equals the number of ground-truth sites.

**(e) Center residues** `center_res_idx = [3, 6]` — one per site.

---

## 6. Training — the loss calculations

### 6a. Center-head losses (over *all* residues at once)
From [`center_head.py`](method/center_head.py):
```
L_obj    = focal(p_logit, y)                    # is-in-a-pocket, per residue
L_dice   = dice(p_logit, y)                     # overlap version of the same
L_center = focal(q_logit, q_target)             # centerness regression
L_vote   = smooth_L1(v[foreground], vote_target[foreground])   # only foreground residues vote
L_center_head = w_obj·L_obj + w_dice·L_dice + w_center·L_center + w_vote·L_vote
```
These need no per-site loop — they're dense per-residue maps.

### 6b. Per-site mask loss (loop over the K sites) — the decisive part
For **each** ground-truth site `k`, condition the mask head on that site's **center residue**, get
a predicted mask over all N residues, and compare to that site's GT mask:
```
for k in sites (k = A, B):
    c = center_res_idx[k]                                   # A→3, B→6
    logits_k = mask_head(h, X, pos, center=c)              # [N] logits for site k
    L_mask_k = dice(logits_k, site_masks[k]) + focal_BCE(logits_k, site_masks[k])
L_mask = mean over k of L_mask_k
```
The mask logit for center `c` and residue `j` is:
```
logit(c,j) = ⟨query(h_c), key(h_j)⟩/√D    (content: are c and j the same kind of residue?)
           + s · ⟨X_c, X_j⟩               (geometry: do their 3D features align? — invariant)
           + dist_prior(‖pos_c − pos_j‖)  (locality: near c is more likely in-pocket)
           + b
```

**Worked Dice for Site A** (condition on center residue 3). Say the head outputs these mask
probabilities (after sigmoid):
```
pred_A = [0.02, 0.88, 0.96, 0.79, 0.15, 0.03, 0.02, 0.01]   Σ = 2.86
GT_A   = [0,    1,    1,    1,    0,    0,    0,    0   ]    Σ = 3
Dice   = 2·(0.88+0.96+0.79) / (2.86 + 3) = 5.26 / 5.86 = 0.898  →  L_dice_A = 1 − 0.898 = 0.102
```
Same for Site B (center 6) gives ≈ 0.125, so `L_mask ≈ (0.102 + 0.125)/2 = 0.114` (plus the focal-BCE term).

### 6c. Total
```
L = L_center_head + L_mask
```

### The key point — **no Hungarian matching**
In DETR/UniSite, you have N anonymous queries and must *solve an assignment problem* (Hungarian
matching) to decide which query is responsible for which GT site before you can compute the loss.
Here, **each GT site already names its own center residue** (`center_res_idx`), so the assignment
is given for free. The per-site loop is the entire "set prediction" mechanism — and it costs one
line, not a matching algorithm, and has no cap on K.

---

## 7. Inference — decoding a *new* protein (no labels)

Same input → encoder → `h, X` → per-residue `p, q, v`. Then decode
([`decode_setpred_mask`](method/center_head.py), committed thresholds). **Note: only `q` and the mask
head are used here** — `p` (objectness) and `v` (the vote) are computed but never read at inference;
they exist only as training signals (§6a).

1. **Threshold centerness:** keep residues with `q > 0.3` as candidate centers.
2. **Local-max NMS (8 Å):** keep a candidate only if its `q` is ≥ every neighbor's within 8 Å.
   This collapses each pocket's cluster of high-centerness residues to a **single** center — it is
   a deterministic single pass, **not** a clustering algorithm with a bandwidth to tune.
3. **Paint a mask per surviving center:** run the mask head for each selected center → sigmoid →
   threshold at 0.4 → that center's residue mask.
4. **Mask-NMS (IoU 0.5):** drop near-duplicate masks.
5. **Score and center:** for center `c` with mask `m_c`,
   `score = q_c × mean(mask probability over m_c)` and `center = Cα centroid of m_c`.
6. **Output:** a **variable number** of sites, each = `{score, center, residue mask}`. The model
   decides how many — the true count is never given.

**Worked decode for P8.** Suppose the trained model outputs:
```
p (objectness): [0.05, 0.85, 0.92, 0.80, 0.20, 0.88, 0.83, 0.10]
q (centerness): [0.03, 0.50, 0.95, 0.50, 0.08, 0.72, 0.70, 0.04]
```
- Step 1: `q > 0.3` → candidates {2, 3, 4, 6, 7}.
- Step 2: local-max NMS → among {2,3,4}, residue **3** (q=0.95) wins; among {6,7}, residue **6**
  (0.72) wins → **selected centers = {3, 6}**.
- Step 3: mask head on center 3 → ≈ [0,1,1,1,0,0,0,0]; on center 6 → ≈ [0,0,0,0,0,1,1,0].
- Step 4: the two masks are disjoint → both kept.
- Step 5: scores. Site A: mask probs over {2,3,4} are (0.88, 0.96, 0.79), mean 0.877, so
  `score = q₃ × 0.877 = 0.95 × 0.877 = 0.83`. Site B: probs over {6,7} are (0.91, 0.86), mean 0.885,
  so `score = q₆ × 0.885 = 0.72 × 0.885 = 0.64`.
- **Output: 2 sites**, ranked A (0.83) then B (0.64). The model recovered K = 2 on its own.
  Note the objectness row above is shown only for completeness — it plays **no part** in this decode.

---

## 8. Why this is "queryless & matching-free" (vs UniSite), in one line each

| UniSite (DETR-style) | EquiSite |
|---|---|
| 32 learned **queries** (fixed budget) | **every residue** is a candidate (no budget) |
| **Hungarian matching** assigns queries↔sites | the **center residue** *is* the assignment |
| 32.8M-param transformer + GearNet | **1.2M**-param equivariant GDEGAN |
| rotation-robust via invariant features | **provably** SE(3)-equivariant by construction |

---

## 9. One-screen recap

```
INPUT      structure (Cα) + frozen ESM2  ──►  radius graph
ENCODER    GDEGAN (4 equivariant layers) ──►  h [N,128] invariant  +  X [N,8,128] equivariant
HEADS      every residue i predicts:  pᵢ (in a pocket?)   qᵢ (at a center?)   vᵢ (center is that way)
TRAIN      targets from the K GT sites:
             y (foreground), q (centerness), vote (→center)         ─ dense, per-residue
             K per-site masks, each tied to its GT center residue    ─ one loss per site, NO matching
           L = focal+dice(obj) + focal(center) + smoothL1(vote) + mean_k[ dice+bce(mask_k) ]
TEST       q>0.3 → local-max NMS (8Å) → selected centers → mask head paints each → threshold+mask-NMS
           OUTPUT: variable #sites, each {score, center, mask}   (model decides K itself)
```

**The whole novelty in one sentence:** replace *learned queries + Hungarian matching* with
*self-nominating center residues + per-site masks tied to those centers*, on a lightweight
equivariant encoder — which removes the site-count cap, deletes the matching step, and cuts the
parameters ~27×, while beating the heavy baseline on UniSite-DS and on independent transfer sets.

---

## 10. Block diagrams (for the paper figure)

### 10.0 · Level 0 — the whole model in three blocks

```
        protein structure
        (PDB or AlphaFold)
                │
                ▼
  ┌─────────────────────────────────────────────┐
  │  A ·  INPUT  →  GRAPH                       │
  │      Cα coordinates + frozen ESM2 features  │
  └──────────────────────┬──────────────────────┘
                         │  G = (N nodes, edges within 10 Å)
                         ▼
  ┌─────────────────────────────────────────────┐
  │  B ·  GDEGAN ENCODER          × 4 layers    │
  │      SE(3)-equivariant message passing      │
  └──────────────────────┬──────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      h  [N, 128]               X  [N, 8, 128]
      INVARIANT                 EQUIVARIANT
      "what kind of             "which direction /
       residue am I"             what local shape"
            │                         │
            └────────────┬────────────┘
                         ▼
  ┌─────────────────────────────────────────────┐
  │  C ·  CENTER NOMINATION HEAD                │
  │      nominate → select → delineate          │
  │      (no queries, no matching)              │
  └──────────────────────┬──────────────────────┘
                         ▼
              K binding sites, each
              { score , center , residue mask }
              ── K is decided by the model ──
```

---

### 10.1 · Expand **A** — input → graph

```
  pos  [N, 3]  ───┐
   (Cα coords)    │        radius graph
                  ├────►   ‖pos_i − pos_j‖ ≤ 10 Å   ───►  G
                  │        (max 64 neighbours*)
  esm  [N, 1280] ─┘
   (frozen ESM2-650M,
    never fine-tuned)          nodes = ESM features
                               edges = relative vectors r_ij

  * cap 64 > max degree 46 in the data ⇒ the cap never binds
    ⇒ the graph is rotation-stable ⇒ exact SE(3)-equivariance
```

---

### 10.2 · Expand **B** — the encoder stack, and one layer inside

```
   h⁰, X⁰ ──►┌───────┐──►┌───────┐──►┌───────┐──►┌───────┐──► h, X
             │layer 1│   │layer 2│   │layer 3│   │layer 4│
             └───────┘   └───────┘   └───────┘   └───────┘
```

one layer, in detail:

```
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. neighbourhood statistics   μ_i , σ²_i   (per channel)     │
  │                                                              │
  │ 2. variance-adaptive attention logit                         │
  │        e_ij = − ‖ (h_j − h_i) / σ_i ‖²  /  2ξ                │
  │        α_ij = softmax_j ( e_ij )                             │
  │        └─ sharper where the neighbourhood is heterogeneous   │
  │           (i.e. at pocket walls)                             │
  │                                                              │
  │ 3. invariant update      h_i ← h_i + Σ_j α_ij · W_h h_j      │
  │ 4. equivariant update    X_i ← X_i + Σ_j α_ij · (W_x X_j     │
  │                                          ⊗ geometry r_ij)    │
  └──────────────────────────────────────────────────────────────┘
      rotate the protein ⇒  h unchanged  ·  X rotates with it
```

---

### 10.3 · Expand **C** — the center nomination head (3 sub-blocks)

```
   h ──────────┐
               ▼
   ┌────────────────────────────────────────────────┐
   │ C1 · NOMINATE      (all residues, in parallel) │
   │                                                │
   │   p_i = σ(MLP(h_i))     am I in a pocket?      │
   │   q_i = σ(MLP(h_i))     am I at a centre?      │
   │   v_i = W · X_i^(ℓ=1)   where is my centre?    │◄── X
   │           centre_i = pos_i + v_i               │
   └───────────────────────┬────────────────────────┘
                           │  p, q  [N]   v [N,3]
                           ▼
   ┌────────────────────────────────────────────────┐
   │ C2 · SELECT               (decode, test time)  │
   │                                                │
   │   q_i > 0.3              keep candidates       │
   │        ▼                                       │
   │   local-max NMS (8 Å)    one winner per pocket │
   │        ▼                                       │
   │   centre residues  { c₁ , c₂ , … , c_K }       │
   │   └─ K is whatever survives — no fixed budget  │
   └───────────────────────┬────────────────────────┘
                           │  c₁ … c_K
                           ▼
   ┌────────────────────────────────────────────────┐
   │ C3 · DELINEATE     (mask head, per centre)     │
   │                                                │
   │   logit(c,j) = ⟨q(h_c), k(h_j)⟩/√D   content   │
   │              + s · ⟨X_c , X_j⟩       geometry  │◄── h, X
   │              + φ(‖pos_c − pos_j‖)    locality  │
   │              + b                               │
   │        ▼                                       │
   │   sigmoid → threshold 0.4 → mask-NMS 0.5       │
   └───────────────────────┬────────────────────────┘
                           ▼
              site k = { score , centre , mask }
```

---

### 10.4 · Train vs. test — the only difference is *where the centre comes from*

```
  TRAINING                                TEST / INFERENCE
  ────────                                ────────────────
  K ground-truth sites                    nothing given
        │                                       │
        ▼                                       ▼
  centres are GIVEN                       centres are PREDICTED
  (center_res_idx, one per site)          (C2: q > 0.3 → local-max NMS)
        │                                       │
        ▼                                       ▼
  for each GT centre c_k:                 for each predicted centre c_j:
     mask head → logits_k                    mask head → mask_j
     loss = Dice + focal-BCE                 threshold + mask-NMS
     vs. that site's GT mask                        │
        │                                          ▼
        ▼                                   variable # of sites
  L = L_obj + L_dice + L_centre                (model chooses K)
      + L_vote + mean_k L_mask_k
        │
        └─ NO Hungarian matching:
           each GT site already names its own centre residue,
           so the assignment is free
```

---

### 10.5 · Tensor shapes, end to end

```
  pos   [N,3] ┐
              ├─► GDEGAN ─┬─► h [N,128] ──┬─► p [N]      ─┐
  esm [N,1280]┘           │               ├─► q [N]      ─┤─► select ─► c₁…c_K
                          └─► X [N,8,128]─┴─► v [N,3]    ─┘        │
                                  │                                ▼
                                  └──────────────────► mask head ─► masks [K,N]
                                                                    scores [K]
                                                                    centres [K,3]
```


#### Extra for my understanding:

Graph → GDEGAN → h [8,128], X [8,8,128]. Nothing site-specific yet.

Step 2 — Heads emit dense per-residue maps, not a set

p = [0.05, 0.85, 0.92, 0.80, 0.20, 0.88, 0.83, 0.10]   objectness   [N]
q = [0.03, 0.50, 0.95, 0.50, 0.08, 0.72, 0.70, 0.04]   centerness   [N]
v = [ ... 8 vectors ... ]                              votes        [N,3]
Notice: these are length-N vectors, not a list of sites. There's no count here to compare against K.

Step 3 — The K labels were already converted into dense targets (offline, once)

y  = [0,0,1,0,0,1,0,0]... wait, y = [0,1,1,1,0,1,1,0]   foreground (union)   [N]
q* = [0,0,1,0,0,1,0,0]                                  1 at centers 3 and 6 [N]
t  = displacement to owner's centroid                                        [N,3]
Step 4 — Three losses, all elementwise. No sets, no matching.

L_obj    = focal(p, y)        ← compares two length-8 vectors
L_center = focal(q, q*)       ← compares two length-8 vectors
L_vote   = smoothL1(v, t)     ← on foreground residues only
This is ordinary dense supervision, exactly like semantic segmentation.

Step 5 — The mask loss: the only place K appears

for each ground-truth site k:
      c = center_res_idx[k]                 # KNOWN from the label: A→3, B→6
      logits_k = mask_head(h, X, pos, c)    # condition at that center → [N]
      L_mask_k = dice + focal_BCE(logits_k, m_k)
L_mask = mean over the K sites
This loop is not matching. We're not asking "which prediction corresponds to site A?" — we're saying "given site A's own center residue 3, reproduce site A's mask." The correspondence is fixed by the label itself. If the protein had 5 sites, the loop runs 5 times. If 1, it runs once.

Step 6 — Total loss, backprop
L = L_obj + L_dice + L_center + L_vote + L_mask. Done. The model never emitted a count.

Why this removes the matching problem entirely
Compare to UniSite/DETR: it has 32 anonymous queries that produce 32 masks. Before it can compute any loss it must answer "which of my 32 queries is responsible for site A?" — that's an assignment problem, solved by Hungarian matching. Set vs set.

EquiSite never faces that question, because the ground-truth site names its own center residue. Site A is "the site at residue 3." There's nothing to align.

Where a set-to-set comparison does happen
This is likely the other half of your confusion. There are two different things:

when	needs a predicted set?	matching?	differentiable?
Training loss	every step	❌ no	❌ no	✅ yes (backprop)
AP metric	validation / test only	✅ yes (decode runs)	✅ greedy one-to-one	❌ no
At validation, we do run the full decode (threshold q → local-max NMS → paint masks → mask-NMS), get a predicted set of size K̂, and greedily match it to the true set to compute IoU-AP. That's how we pick the best checkpoint (val_ap=0.5208). But that AP is a score we read off, never something we differentiate through.

The train/test asymmetry in one line
centers come from	K
Train	the label (center_res_idx)	known, used as a loop bound
Test	the model (threshold + local-max NMS on q)	discovered, K̂ = |C|
This is the standard teacher-forcing pattern: train the mask head conditioned on true centers, deploy it conditioned on predicted centers. It works because both training and inference ask the head the same question — "given this center, which residues form its site?" — only the source of the center changes.

So to answer you directly: we don't compare predicted-K to true-K during training at all. We supervise the dense fields (q, p, v) from which the count will later be decoded, and we supervise the mask head at known centers. The count-prediction ability emerges as a side effect of learning a good centerness field — and it's only exercised, and only measured, at inference.




---

# 11. The Center Nomination Head — why it exists, and how it works

> Notes for explaining the contribution. Part A is the *motivation* (the gap), Part B the
> *inspiration* (what we borrowed and what we discovered ourselves), Part C the *mechanism*
> step by step, Part D *training and backprop*.

---

## Part A · The gap we are closing

UniSite ports **DETR** to residues: `N_q = 32` learned "site queries" go through a transformer
decoder, each emits a mask, and a **Hungarian matching** in the loss decides which query is
responsible for which ground-truth site. That design carries four costs:

| # | Cost | Why it hurts here |
|---|---|---|
| 1 | **Fixed budget of 32 queries** | UniSite-DS contains proteins with **up to 38 sites** — more sites than queries. The architecture *cannot* represent them. |
| 2 | **Matching is a discrete optimisation inside the loss** | Non-differentiable; the assignment can flip between epochs, so the target a query chases is unstable early in training. |
| 3 | **Queries are anonymous** | A query is a learned vector with no location. The model must *learn* "which query owns which region" from scratch — pure overhead for a task where sites are inherently local. |
| 4 | **Heavy** | 32.8M parameters, 8 GPUs. |

The observation behind our head: **a binding site already has a natural name — the residue at its
centre.** If every ground-truth site is identified by one of its own residues, the assignment problem
disappears. There is nothing to match, because the label already says who is responsible.

---

## Part B · What inspired the design

### B1. Our own diagnostic (the decisive one)

Our first model (call it **exp-1**) did the obvious thing: predict one **merged foreground map**
(*"is this residue in any pocket?"*), then split that blob geometrically by assigning each foreground
residue to its nearest predicted centre. It scored **AP@0.5 ≈ 0.20**.

To find out *what* was broken we ran an **oracle decomposition** — replace one component at a time
with ground truth and see what the ceiling becomes:

| what we feed in | AP@0.5 |
|---|---|
| **GT centres + GT foreground** (oracle both) | **0.89** |
| GT centres + *learned* foreground | **0.27** |
| fully learned (exp-1) | ~0.20 |

Read that table carefully — it is the entire argument:

- With perfect masks the formulation reaches **0.89**, so *centre-based set prediction is not the
  problem*. The paradigm is sound.
- Swapping in learned masks collapses it to **0.27**. **The masks were the bottleneck**, not the
  centres, not the counting.

**Why** a merged foreground map can never fix this: it is one binary field for the whole protein. Two
sites that share residues (which happens — sites overlap) get merged into one blob, and nearest-centre
splitting cuts it with a straight geometric boundary that has nothing to do with either site's real
shape. The information "which residues form *site A specifically*" was never in the training signal.

**Conclusion:** each site needs **its own mask, supervised per site**. That is the design requirement
we derived ourselves.

### B2. Vision literature (how to get per-site masks without matching)

The diagnosis said *what* we needed; the anchor-free detection literature says *how*:

| Idea we use | Comes from | What we take |
|---|---|---|
| Objects as **centre points** + local-max NMS on a heatmap | **CenterNet** (Zhou et al., 2019) | represent each site by a single centre; deduplicate by local maximum, not clustering |
| Dense per-location prediction with a **"centerness"** branch | **FCOS** (Tian et al., ICCV'19) | the name and the role of $q_i$ |
| Each instance **conditions its own mask head** → per-instance masks, no matching | **CondInst** (Tian et al., ECCV'20), **SOLOv2** | the centre-conditioned mask head — the direct answer to B1 |
| Points **vote** a 3D displacement to their object's centre | **VoteNet** (Qi et al., ICCV'19) | the equivariant centre vote $\mathbf v_i$ (Hough voting in 3D) |

**One-line framing:** *DETR→CenterNet/CondInst was the shift in vision from query+matching detectors to
anchor-free, per-instance-conditioned ones. We are making the same shift for binding sites — and the
oracle experiment is what told us the mask, not the centre, is where that shift pays.*

---

## Part C · The mechanism, step by step

**We start with what the encoder gives us**, per residue $i$:

```
h_i ∈ R^128        invariant   — "what kind of residue am I, in what context"
X_i ∈ R^{8×128}    equivariant — "which direction / what local shape"  (ℓ=1: 3 comps, ℓ=2: 5)
```

### Step 1 · NOMINATE — every residue describes itself

Instead of 32 anonymous queries asking *"is there a site somewhere?"*, all $N$ residues each answer
three questions **about themselves**, in parallel:

| output | formula | the question |
|---|---|---|
| objectness $p_i$ | $\sigma(\mathrm{MLP}(h_i))$ | *Am I inside some pocket?* |
| centerness $q_i$ | $\sigma(\mathrm{MLP}(h_i))$ | *Am I the **centre** of a pocket?* |
| vote $\mathbf v_i$ | $X_i^{(1)}\mathbf w$, centre $=\mathbf p_i+\mathbf v_i$ | *Which way is my pocket's centre?* |

**Intuition:** the residues *are* the queries — but unlike DETR's, they are **not anonymous**. Each is
anchored at a physical location, so "which one owns this region" is answered by geometry, not learned
from scratch.

**Why the split matters:** $p_i,q_i$ are scalars, so they read the **invariant** stream $h$ — rotate the
protein and they must not change. $\mathbf v_i$ is a 3D vector, so it *must* read the **equivariant**
stream $X$ — rotate the protein and the arrow must rotate with it. This is the one place equivariance
is structurally forced rather than merely nice.

### Step 2 · SELECT — turn a smooth field into a discrete set

$q$ is a *smooth* field: residues *near* a centre also score high, so thresholding alone would emit a
blob of 5–10 near-duplicate "sites" per pocket. We keep only **local maxima**:

$$\mathcal C=\Big\{ i \;:\; \underbrace{q_i>\tau_q}_{\text{confident}} \;\wedge\; \underbrace{q_i \ge \max_{j:\,\|\mathbf p_i-\mathbf p_j\|\le\rho} q_j}_{\text{peak of its own blob}} \Big\}, \qquad \tau_q=0.3,\ \rho=8\,\text{Å}$$

**This is the step where $\widehat K$ is decided** — $\widehat K = |\mathcal C|$. Nothing caps it.

**Not clustering:** one deterministic pass, no iterations, no bandwidth to fit, no random init (unlike
the mean-shift/DBSCAN stage in predict-then-cluster methods).

**Why $\rho = 8$ Å is not arbitrary:** measured on the training split, a pocket's mean Cα-to-centroid
radius is **8.0 Å**, while 90% of distinct site pairs have centroids **> 8 Å** apart (10th pct 8.1 Å).
So 8 Å is exactly the scale that collapses one pocket to one centre without merging two pockets.

### Step 3 · DELINEATE — each centre paints its own site

For a selected centre $c$, ask of **every** residue $j$: *does $j$ belong to the site centred at $c$?*

$$\text{logit}(c,j) = \underbrace{\tfrac{1}{\sqrt{D}}\langle \mathrm{q}(h_c), \mathrm{k}(h_j)\rangle}_{\text{content}} + \underbrace{s\,\langle X_c, X_j\rangle}_{\text{geometry}} + \underbrace{\phi(\|\mathbf p_c-\mathbf p_j\|)}_{\text{locality}} + b$$

Three cues, each answering a different question:

- **content** — *are $c$ and $j$ the same kind of residue in the same kind of environment?*
- **geometry** — *do their local 3D features align?* (an inner product of steerable features ⇒
  rotation-**invariant**, so the mask stays invariant while using directional information)
- **locality** — *is $j$ physically near $c$?* The pocket prior.

Threshold at $\tau_m = 0.4$ → that centre's residue mask. **This is the component the oracle experiment
demanded**: the mask is conditioned on *which* site it belongs to, so two overlapping sites produce two
different masks from the same residues.

### Step 4 · DUPLICATE REMOVAL — mask NMS

Two centres more than $\rho$ apart both survive Step 2 but may paint nearly the same residues (e.g.
opposite rims of one large pocket). Step 2 compares **points**; this duplication is a property of
**sets**. So: sort by score, keep the best, drop anything with IoU ≥ 0.5 against it, repeat.

**Output:** $\{(s_c, \hat{\mathbf z}_c, \hat{\mathbf m}_c)\}$ — a variable-size set.

---

## Part D · Training and backprop

### D1. The crucial asymmetry

| | TRAINING | INFERENCE |
|---|---|---|
| centres come from | **the label** (`center_res_idx`) | **the model** (Steps 2) |
| decode run? | **no** — Steps 2 & 4 never execute | yes |
| $\widehat K$ | not produced at all | discovered |

**At training the model never emits a set**, so there is never a predicted set to align with a true
set — which is *why no matching is needed*. This is standard **teacher forcing**: train the mask head
on true centres, deploy it on predicted centres. Both ask the identical question ("given this centre,
which residues?"); only the source of the centre changes.

### D2. The five losses

Everything is a dense per-residue comparison except the mask loss, which loops over the $K$ labelled
sites:

```
L_obj    = focal(p, y)            α=0.25, γ=2      weight 0.5    dense [N]
L_dice   = dice(p, y)                               weight 0.5    dense [N]
L_center = focal(q, q*)           α=0.95, γ=2      weight 5.0    dense [N]
L_vote   = smoothL1(v, t)         foreground only  weight 1.0    dense [N,3]

L_mask   = mean over k=1..K of [ 2·Dice + 1·focalBCE ]( mask_head(h,X,pos, c_k), m_k )

L = 0.5·L_obj + 0.5·L_dice + 5.0·L_center + 1.0·L_vote + L_mask
```

Two weights are worth understanding:

- **`w_center = 5.0`** (the largest) — centerness is both the hardest signal and the one that decides
  $\widehat K$. Under-weight it and the model finds no centres.
- **`center_alpha = 0.95`** — centres are extremely rare: about **1.6 positives among ~300 residues**
  (~0.5%). Without heavy positive up-weighting the focal loss would trivially predict all-zeros.

### D3. How gradients flow

```
                     ┌──────────► L_obj , L_dice   (from p)
   ENCODER           ├──────────► L_center         (from q)
   h , X  ───────────┼──────────► L_vote           (from v, uses X^(ℓ=1))
                     └──────────► L_mask           (from the mask head, uses h AND X)
                                       │
        all five gradients sum back into the SAME encoder weights
```

Because every loss backprops into the shared $(h, X)$, the encoder is shaped by all objectives at
once: $h$ must simultaneously support "is this a pocket", "is this a centre", and "do these two
residues belong together"; $X$ must support both the 3D vote and the mask's geometry term.

Note the mask loss is the **only** one that touches the mask head's own parameters
($\mathrm{q},\mathrm{k},\phi,s,b$) — and it is also the one that reaches the encoder through the most
paths.

### D4. Worked example (toy protein P8, from §1)

$K=2$: site A = {2,3,4} centre **3**, site B = {6,7} centre **6**.

```
dense targets:   y  = [0,1,1,1,0,1,1,0]      (union foreground)
                 q* = [0,0,1,0,0,1,0,0]      (binary: 1 at centres 3 and 6)
                 t  = displacement to owner's centroid, 0 on background

mask loop (K = 2 iterations, no matching):
   k=A:  c=3 → logits_A = mask_head(h,X,pos, c=3) → compare to [0,1,1,1,0,0,0,0]
   k=B:  c=6 → logits_B = mask_head(h,X,pos, c=6) → compare to [0,0,0,0,0,1,1,0]
```

If the protein had 5 sites the loop runs 5 times. **The loop bound is the label, and the correspondence
is given, not solved.**

### D5. Did it work? (the evidence for B1)

Removing exactly this component — keeping everything else, reverting to merged-foreground supervision
— across 3 seeds:

| variant | AP@0.5 (3 seeds) |
|---|---|
| **full (per-site mask head)** | **0.464 ± 0.007** |
| − per-site masks (foreground split) | 0.201 ± 0.012 |

**Δ = +0.263 AP@0.5, ≈ 22× the seed noise.** It is by far the largest single factor in the model — and
it is precisely what the oracle decomposition in B1 predicted three months earlier.

---

## 11.6 · One-paragraph summary (for talks)

> UniSite inherits DETR's machinery: 32 anonymous queries and a Hungarian matching that decides which
> query answers for which site. We noticed two things. First, a binding site already has a natural
> name — the residue at its centre — so if the label identifies each site by one of its own residues,
> the assignment problem simply disappears. Second, and this we found by an oracle experiment: with
> perfect masks the centre-based formulation reaches 0.89 AP, but with learned merged-foreground masks
> it collapses to 0.27 — the *masks* were the bottleneck, not the centres. So we built a head where
> **every residue nominates itself** as a candidate centre (objectness, centerness, an equivariant vote),
> local-max suppression **selects** one winner per pocket, and each winner **delineates** its own site
> through a centre-conditioned mask. No queries, no matching, no cap on the number of sites — and the
> per-site mask supervision alone is worth +0.26 AP.

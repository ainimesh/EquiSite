"""
dcc_dca.py — DCC / DCA success-rate metrics + convex-hull center.

Vendored so EquiSite is fully self-contained (no import from the UniSite package). The math is
copied VERBATIM from UniSite for an exact head-to-head:
  * calc_dca_dcc  <- unisite/evaluation/dcc_dca.py :: calc_dca_dcc_metrics  (top-n / top-(n+2), 4A)
  * hull_center   <- unisite/model/utils.py        :: hull_center           (Delaunay volume centroid)
Aggregation (success_rate) mirrors unisite/evaluation/eval_dcc_dca.py: mean over all GT sites of (< 4A).
"""
import numpy as np


def hull_center(points):
    """Volume centroid of the convex hull of `points` [M,3] (Delaunay tetra decomposition,
    volume-weighted). Verbatim math from UniSite's model/utils.py::hull_center."""
    from scipy.spatial import ConvexHull, Delaunay
    hull = ConvexHull(points)
    com = np.zeros(3)
    tet = Delaunay(hull.points[hull.vertices])
    for s in tet.simplices:
        v = tet.points[s]
        a, b, c = v[0] - v[3], v[1] - v[3], v[2] - v[3]
        vol = abs(np.linalg.det([a, b, c])) / 6.0
        com += v.mean(0) * vol
    return com / hull.volume


def pocket_center(mask_row, node_atoms, ca, mode):
    """Collapse one predicted pocket (residue mask) to a 3D center.
    mode: 'ca' = Cα-centroid | 'geo' = all-heavy-atom mean | 'hull' = convex-hull volume centroid."""
    idx = np.where(mask_row > 0)[0]
    if mode == "ca" or len(idx) == 0:
        return ca[idx].mean(0) if len(idx) else np.zeros(3, np.float32)
    pts = np.concatenate([node_atoms[i] for i in idx], axis=0)     # all heavy atoms of masked residues
    if mode == "geo" or len(pts) < 4:
        return pts.mean(0)
    try:
        return hull_center(pts).astype(np.float32)
    except Exception:
        return pts.mean(0)


def calc_dca_dcc(pred_centers_list, pred_scores_list, ligs_list, top_n_plus=0):
    """Per-protein DCC/DCA of the top-(n + top_n_plus) predictions (n = #GT ligands).
    Verbatim math from UniSite's evaluation/dcc_dca.py::calc_dca_dcc_metrics.
    ligs_list[i] = list of ligand heavy-atom arrays [natoms,3]. Returns (dcc_results, dca_results),
    each a list of per-GT-site min-distances."""
    dcc_results, dca_results = [], []
    for i in range(len(pred_centers_list)):
        pc = np.asarray(pred_centers_list[i]); ps = np.asarray(pred_scores_list[i]); ligs = ligs_list[i]
        num_ligs = len(ligs)
        if num_ligs == 0:
            dcc_results.append([]); dca_results.append([]); continue
        if len(pc) == 0:
            dcc_results.append(np.full(num_ligs, np.inf)); dca_results.append(np.full(num_ligs, np.inf)); continue
        ids = np.argsort(ps)[::-1][: num_ligs + top_n_plus]
        pc = pc[ids]
        dccm = np.zeros((num_ligs, len(pc))); dcam = np.zeros((num_ligs, len(pc)))
        for j, lig in enumerate(ligs):
            lc = lig.mean(0)                                        # ligand centroid = GT site center (DCC)
            dccm[j] = np.linalg.norm(pc - lc, axis=1)
            dcam[j] = np.linalg.norm(pc[:, None, :] - lig[None], axis=-1).min(1)   # nearest ligand atom (DCA)
        dcc_results.append(dccm.min(1)); dca_results.append(dcam.min(1))
    return dcc_results, dca_results


def success_rate(results, thr=4.0):
    """Success Rate = fraction of GT sites with distance < thr, pooled over all proteins."""
    arrs = [np.asarray(a) for a in results if len(a)]
    return float(np.mean(np.concatenate(arrs) < thr)) if arrs else 0.0

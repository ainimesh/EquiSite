"""
gdegan.py — Option 2: HEGNN-core backbone + Geometric Distributional Attention.

WHAT THIS IS (and is not)
-------------------------
This is a drop-in replacement for `GaussianEnhancedGotenNet` that swaps the GotenNet
backbone for a HEGNN-style scalarization core, keeping your distributional attention as
the aggregation weight. It is intended as an INDEPENDENT SECOND BACKBONE: if the l>=1
steerable attention is inert here too (as it was on the GotenNet core), that hardens the
negative result into a task+ESM property rather than an architecture artifact. It is not a
likely path to a win on this task; treat it as a confirmation experiment.

MECHANISM (mirrors HEGNN.py, adapted to the multi-channel [N, dim, C] layout your pipeline
and directional loss require):
  - message from per-degree inner products  z_ij^(l,c) = <X_i^(l,c), X_j^(l,c)>   (HEGNN Eq. 6)
  - steerable update as a DIFFERENCE (X_i^(l) - X_j^(l)) scaled by an invariant scalar,
    attention-weighted                                                            (HEGNN Eq. 8)
  - aggregation is the Geometric Distributional Attention (per-degree standardized Gaussian,
    per-(degree,head) temperatures, direct softmax) instead of HEGNN's uniform mean.

REMOVED relative to your current model: hierarchical tensor refinement (HTR), the GotenNet
value/split (o^s/o^d/o^t), the spatial filter, and all position/velocity updates.

INTERFACE (matches GaussianEnhancedGotenNet, so the LightningModule is unchanged):
  forward(inputs) -> (h, X)
    h : [N, n_atom_basis]                      (consumed by output_head -> per-residue logit)
    X : [N, (lmax+1)^2 - 1, n_atom_basis]      (l=1 in X[:, :3, :] for the directional loss)

CONFIG: accepts all keys of your model config; GotenNet-only keys (edge_updates, sep_htr,
sep_dir, sep_tensor, evec_dim, emlp_dim, edge_ln, scale_edge, aggr) are accepted and ignored
via **kwargs. `gda_degrees` keeps the same semantics: "scalar" = l=0-only attention (HEGNN core
with scalar GDA); "all" = attention over l=0..lmax. `steerable_init` in {"spherical","zero"}.
"""

from functools import partial
from typing import Callable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter, softmax

# Featurization primitives reused unchanged from your repo so the backbone is the only variable.
from equisite.utils.utils import (
    NodeInit,
    EdgeInit,
    Distance,
    TensorInit,
    str2act,
    str2basis,
    get_weight_init_by_string,
)


# --------------------------------------------------------------------------------------
# small helpers (local copies so this file has no dependency on gdegnn_redefine2.py)
# --------------------------------------------------------------------------------------
def get_split_sizes_from_lmax(lmax: int, start: int = 1) -> List[int]:
    """Component sizes (2l+1) for l = start..lmax. Default excludes l=0 (scalars live in h)."""
    return [2 * l + 1 for l in range(start, lmax + 1)]


def split_to_components(tensor: Tensor, lmax: int, start: int = 1, dim: int = 1) -> List[Tensor]:
    """Split a packed steerable tensor [*, sum(2l+1), *] into a list of per-degree tensors."""
    return list(torch.split(tensor, get_split_sizes_from_lmax(lmax, start=start), dim=dim))


class BaseMLP(nn.Module):
    """2-layer MLP (copied from HEGNN.py, with optional residual)."""

    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.SiLU(),
                 residual=False, last_act=False):
        super().__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation if last_act else nn.Identity(),
        )

    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)


# --------------------------------------------------------------------------------------
# Geometric Distributional Attention  (Eqs. 5-11 of the method)
# --------------------------------------------------------------------------------------
class GeometricDistributionalAttention(nn.Module):
    """
    Returns per-edge, per-head attention weights alpha [E, H].

    l=0 term: variance-adaptive Gaussian on scalar features h (the original GDA).
    l>=1 terms: per-degree standardized steerable distances from X (only if gda_degrees != "scalar").
    Temperatures xi^(l,h): one per (degree, head), softplus() for positivity.
    Direct softmax over neighbors: alpha = softmax(-sum_l delta / 2 xi).
    """

    def __init__(self, hidden_dim: int, num_heads: int, lmax: int, gda_degrees: str,
                 learnable_temperature: bool = True, fixed_temperature_value: float = 2.0,
                 epsilon: float = 1e-8):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.H = num_heads
        self.C = hidden_dim
        self.Dh = hidden_dim // num_heads
        self.lmax = lmax
        self.gda_degrees = gda_degrees
        self.eps = epsilon

        _t = torch.ones(num_heads, lmax + 1) * fixed_temperature_value  # [H, lmax+1]
        if learnable_temperature:
            self.temperature = nn.Parameter(_t)
        else:
            self.register_buffer("temperature", _t)

        self._last_neighborhood_var = None  # l=0 variance, kept for the Appendix-F probe

    def forward(self, h: Tensor, X: Tensor, edge_index: Tensor, dim_size: int) -> Tensor:
        row, col = edge_index                       # row = target i, col = source j
        H, Dh = self.H, self.Dh
        xi = F.softplus(self.temperature)           # [H, lmax+1] strictly positive

        # ---- per-degree neighborhood variances: one invariant scalar per (node, channel, degree) ----
        # l = 0 (scalar)
        h_nb = h[col]                                                        # [E, C]
        mean0 = scatter(h_nb, row, dim=0, dim_size=dim_size, reduce="mean")
        var0 = scatter((h_nb - mean0[row]) ** 2, row, dim=0, dim_size=dim_size, reduce="mean")
        var_list = [var0]                                                    # [N, C]
        self._last_neighborhood_var = var0.detach()

        if self.gda_degrees != "scalar":
            for Xl in split_to_components(X, self.lmax, dim=1):              # each [N, 2l+1, C]
                Xl_nb = Xl[col]                                              # [E, 2l+1, C]
                mean_l = scatter(Xl_nb, row, dim=0, dim_size=dim_size, reduce="mean")
                centered = Xl_nb - mean_l[row]
                sqnorm = (centered ** 2).sum(dim=1)                          # [E, C] <- norm over 2l+1
                var_l = scatter(sqnorm, row, dim=0, dim_size=dim_size, reduce="mean")
                var_list.append(var_l)                                       # [N, C]

        # ---- attention logit ----
        # l = 0 : the original scalar GDA, as a per-head logit contribution
        hi = h[row].reshape(-1, H, Dh)
        hj = h[col].reshape(-1, H, Dh)
        v0 = var_list[0][row].reshape(-1, H, Dh)
        d0 = (hj - hi) / torch.sqrt(v0 + self.eps)
        logit = (d0 ** 2).sum(dim=-1) / (2.0 * xi[:, 0].unsqueeze(0))        # [E, H]

        # l >= 1 : per-degree distributional terms
        if self.gda_degrees != "scalar":
            Xi = split_to_components(X[row], self.lmax, dim=1)               # each [E, 2l+1, C]
            Xj = split_to_components(X[col], self.lmax, dim=1)
            for li in range(self.lmax):                                      # li=0 -> degree l=1
                v_l = var_list[li + 1][row]                                  # [E, C]  (invariant)
                D_l = (Xj[li] - Xi[li]) / torch.sqrt(v_l + self.eps).unsqueeze(1)
                d2 = (D_l ** 2).sum(dim=1)                                   # [E, C] <- sum over 2l+1
                logit = logit + d2.reshape(-1, H, Dh).sum(dim=-1) / (2.0 * xi[:, li + 1].unsqueeze(0))

        # direct softmax over neighbors (per head): alpha = softmax(-logit)
        alpha = softmax(-logit, row, num_nodes=dim_size)                     # [E, H]
        return alpha

class DotProductAttention(nn.Module):
    """Standard scaled dot-product attention on scalar features (transformer/GATA-style
    weighting). Returns per-edge, per-head weights alpha [E, H]. Ignores X by design:
    this isolates the *weighting rule*, holding the rest of the HEGNN pipeline fixed."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.H = num_heads
        self.Dh = hidden_dim // num_heads
        self.ln = nn.LayerNorm(hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self._last_neighborhood_var = None  # interface parity (unused)

    def forward(self, h, X, edge_index, dim_size):
        row, col = edge_index
        hn = self.ln(h)
        q = self.W_q(hn).reshape(-1, self.H, self.Dh)      # [N, H, Dh]
        k = self.W_k(hn).reshape(-1, self.H, self.Dh)
        logit = (q[row] * k[col]).sum(dim=-1) / (self.Dh ** 0.5)   # [E, H] scaled dot product
        return softmax(logit, row, num_nodes=dim_size)            # [E, H]
    

# --------------------------------------------------------------------------------------
# HEGNN-core interaction layer (attention-weighted)  (Eqs. 12-15 of the method)
# --------------------------------------------------------------------------------------
class GDEGANLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, num_heads: int, lmax: int,
                 gda_degrees: str, activation: Callable, dropout: float = 0.0,
                 learnable_temperature: bool = True, fixed_temperature_value: float = 2.0,
                 epsilon: float = 1e-8, attention_type='attention_type'):
        super().__init__()
        self.C = hidden_dim
        self.H = num_heads
        self.Dh = hidden_dim // num_heads
        self.lmax = lmax
        self.gda_degrees = gda_degrees
        self.dropout = dropout

        act = activation() if isinstance(activation, type) else activation
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=act)

        # message MLP (Eq. 12): [h_i, h_j, edge_attr, dist^2, (+)_l z_ij^(l)] -> m_ij
        msg_in = 2 * hidden_dim + edge_attr_dim + 1 + lmax * hidden_dim
        self.mlp_msg = MLP(input_dim=msg_in, output_dim=hidden_dim, last_act=True)

        # scalar update (Eq. 13): [h, aggregated message] -> h        (HEGNN cat+MLP form)
        self.mlp_h = MLP(input_dim=2 * hidden_dim, output_dim=hidden_dim)

        # steerable coefficients (Eq. 14): m_ij -> per-degree per-channel invariant scalar
        self.mlp_v = MLP(input_dim=hidden_dim, output_dim=lmax * hidden_dim)

        if attention_type == 'dot_product':
            self.attn = DotProductAttention(hidden_dim, num_heads, dropout)
        else: # our gaussian dynamic
            self.attn = GeometricDistributionalAttention(
                hidden_dim=hidden_dim, num_heads=num_heads, lmax=lmax, gda_degrees=gda_degrees,
                learnable_temperature=learnable_temperature,
                fixed_temperature_value=fixed_temperature_value, epsilon=epsilon,
        )

    def forward(self, h: Tensor, X: Tensor, edge_index: Tensor, t_ij: Tensor,
                dist2: Tensor) -> Tuple[Tensor, Tensor]:
        row, col = edge_index
        N = h.size(0)
        H, Dh = self.H, self.Dh

        # per-degree endpoint splits (reused by message and steerable update)
        Xi_deg = split_to_components(X[row], self.lmax, dim=1)               # each [E, 2l+1, C]
        Xj_deg = split_to_components(X[col], self.lmax, dim=1)

        # ---- Eq. 12: invariant message from per-degree per-channel inner products ----
        z = torch.cat([(Xi_deg[l] * Xj_deg[l]).sum(dim=1) for l in range(self.lmax)], dim=-1)  # [E, lmax*C]
        if getattr(self, 'neuter_z', False):           # Exp 2: cut the l>=1 -> scalar message pathway
            z = torch.zeros_like(z)
        m = self.mlp_msg(torch.cat([h[row], h[col], t_ij, dist2, z], dim=-1))                  # [E, C]

        # ---- attention (Eqs. 5-11) ----
        alpha = self.attn(h, X, edge_index, N)                              # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha_c = alpha.repeat_interleave(Dh, dim=1)                        # [E, C] (head weight -> its channels)

        # ---- Eq. 13: attention-weighted scalar update ----
        h_agg = scatter(alpha_c * m, row, dim=0, dim_size=N, reduce="sum")  # [N, C]
        
        # h = self.mlp_h(torch.cat([h, h_agg], dim=-1))                       # HEGNN cat+MLP (non-residual)
        h = h + self.mlp_h(torch.cat([h, h_agg], dim=-1))                   # residual scalar update

        # ---- Eq. 14: attention-weighted, difference-based steerable update ----
        coeff_deg = torch.split(self.mlp_v(m), self.C, dim=-1)              # lmax tensors [E, C]
        dX_parts = []
        for li in range(self.lmax):
            diff = Xi_deg[li] - Xj_deg[li]                                  # [E, 2l+1, C] = X_i - X_j
            w = (alpha_c * coeff_deg[li]).unsqueeze(1)                      # [E, 1, C] invariant scalar
            dX_parts.append(scatter(w * diff, row, dim=0, dim_size=N, reduce="sum"))  # [N, 2l+1, C]
        X = X + torch.cat(dX_parts, dim=1)                                  # residual, [N, equi_dim, C]

        return h, X


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
class GDEGAN(nn.Module):
    """HEGNN-core GDEGAN. Returns (h, X); see interface note at the top of the file."""

    def __init__(
        self,
        node_feat_dim: int = 1280,
        n_atom_basis: int = 128,
        n_interactions: int = 4,
        radial_basis: Union[Callable, str] = "expnorm",
        n_rbf: int = 32,
        max_num_neighbors: int = 32,
        cutoff_fn: Optional[Callable] = None,
        activation: Union[Callable, str] = F.silu,
        epsilon: float = 1e-8,
        weight_init: Union[Callable, str] = nn.init.xavier_uniform_,
        bias_init: Union[Callable, str] = nn.init.zeros_,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        learnable_temperature: bool = True,
        fixed_temperature_value: float = 2.0,
        lmax: int = 2,
        gda_degrees: str = "scalar",
        attention_type: str = "attention_type",
        steerable_init: str = "spherical",   # "spherical" (concavity init) or "zero"
        **kwargs,                            # swallows GotenNet-only keys; see header
    ):
        super().__init__()
        if isinstance(weight_init, str):
            weight_init = get_weight_init_by_string(weight_init)
        if isinstance(bias_init, str):
            bias_init = get_weight_init_by_string(bias_init)
        if isinstance(activation, str):
            activation = str2act(activation)

        self.hidden_dim = n_atom_basis
        self.n_interactions = n_interactions
        self.lmax = lmax
        self.gda_degrees = gda_degrees
        self.steerable_init = steerable_init
        self.equi_dim = (lmax + 1) ** 2 - 1

        self.cutoff = cutoff_fn.cutoff
        self.distance = Distance(self.cutoff, max_num_neighbors=max_num_neighbors, loop=True)

        # --- featurization (reused from your repo, unchanged) ---
        self.node_init = NodeInit(
            [self.hidden_dim, self.hidden_dim], n_rbf, self.cutoff,
            node_feat_dim=node_feat_dim, weight_init=weight_init, bias_init=bias_init,
            proj_ln="layer", activation=activation,
        )
        self.edge_init = EdgeInit(n_rbf, self.hidden_dim)
        self.radial_basis = str2basis(radial_basis)(cutoff=self.cutoff, n_rbf=n_rbf)
        self.sphere = TensorInit(l=lmax)  # edge spherical harmonics r_hat -> [E, equi_dim] (excludes l=0)

        # --- steerable initialization coefficients (concavity SH init, Eq. 4) ---
        act = activation() if isinstance(activation, type) else activation
        self.init_coeff = BaseMLP(
            input_dim=2 * self.hidden_dim + self.hidden_dim + 1,
            hidden_dim=self.hidden_dim, output_dim=lmax * self.hidden_dim,
            activation=act, last_act=True,
        )

        # --- HEGNN-core interaction layers (edge_attr = t_ij of width hidden_dim) ---
        self.layers = nn.ModuleList([
            GDEGANLayer(
                hidden_dim=self.hidden_dim, edge_attr_dim=self.hidden_dim, num_heads=num_heads,
                lmax=lmax, gda_degrees=gda_degrees, activation=activation, dropout=attn_dropout, attention_type=attention_type,
                learnable_temperature=learnable_temperature,
                fixed_temperature_value=fixed_temperature_value, epsilon=epsilon,
            )
            for _ in range(n_interactions)
        ])

        self.reset_parameters()

        # Exp 2 toggle: zero the l>=1 inner-product (z) block that feeds the scalar message.
        # Set on each layer post-construction so it needs no signature changes.
        self.neuter_z = kwargs.get('neuter_steerable_message', False)
        for _layer in self.layers:
            _layer.neuter_z = self.neuter_z

    def reset_parameters(self):
        self.node_init.reset_parameters()
        self.edge_init.reset_parameters()

    # ---- concavity-motivated steerable init (Eq. 4) ----
    def _init_steerable(self, h, t_ij, rl_ij, edge_index, dist2):
        if self.steerable_init == "zero":
            return torch.zeros(h.size(0), self.equi_dim, h.size(1), device=h.device)
        row, col = edge_index
        coeff = self.init_coeff(torch.cat([h[row], h[col], t_ij, dist2], dim=-1))   # [E, lmax*C]
        coeff_deg = torch.split(coeff, self.hidden_dim, dim=-1)                      # lmax x [E, C]
        rl_deg = split_to_components(rl_ij, self.lmax, dim=1)                        # lmax x [E, 2l+1]
        parts = [coeff_deg[l].unsqueeze(1) * rl_deg[l].unsqueeze(-1) for l in range(self.lmax)]
        X_edge = torch.cat(parts, dim=1)                                            # [E, equi_dim, C]
        return scatter(X_edge, row, dim=0, dim_size=h.size(0), reduce="mean")       # [N, equi_dim, C]

    def forward(self, inputs: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        h, pos, atomic_numbers, batch = inputs.esm, inputs.pos, inputs.z, inputs.batch

        edge_index, edge_diff, edge_vec = self.distance(pos, batch)
        phi = self.radial_basis(edge_diff)

        h = self.node_init(h, edge_index, edge_diff, phi)          # [N, C]
        t_ij = self.edge_init(edge_index, phi, h)                  # [E, C]

        # unit-normalize edge vectors (excluding self-loops) for the spherical harmonics
        mask = edge_index[0] != edge_index[1]
        edge_vec = edge_vec.clone()
        r0 = torch.norm(edge_vec[mask], dim=1, keepdim=True)
        edge_vec[mask] = edge_vec[mask] / r0
        rl_ij = self.sphere(edge_vec)                              # [E, equi_dim]

        dist2 = (edge_diff.reshape(-1, 1)) ** 2                    # [E, 1]

        X = self._init_steerable(h, t_ij, rl_ij, edge_index, dist2)  # [N, equi_dim, C]

        self._layer_variances = []
        for layer in self.layers:
            h, X = layer(h, X, edge_index, t_ij, dist2)
            if layer.attn._last_neighborhood_var is not None:
                self._layer_variances.append(layer.attn._last_neighborhood_var)

        return h, X                                               # h: [N, C] ; X: [N, equi_dim, C]
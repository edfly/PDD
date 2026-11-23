"""
Riemannian Graph Convolution (RGC).

Implements geodesic message passing on the product manifold H^{d_h} x R^{d_e}:

For each node v and neighbor u:
  1. logmap(z_v, z_u) — map neighbor to v's tangent space
  2. Aggregate in tangent space (GIN-style MLP)
  3. expmap(aggregate, z_v) — map back to manifold
  4. projx — ensure result stays on manifold

Hyperbolic component: uses exp/log from PoincareManifold.
Euclidean component: standard GCN/MLP (exp/log degenerate to identity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.utils import add_self_loops

from spikergnn.manifolds import ProductManifold, Point


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Create a 2-layer MLP for GIN convolution.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.

    Returns:
        nn.Sequential MLP.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class RiemannianGraphConv(nn.Module):
    """Riemannian Graph Convolution on the product manifold.

    Uses GIN-style message passing with geodesic exp/log mappings
    for the hyperbolic component and standard GIN for the Euclidean
    component.

    Attributes:
        manifold: ProductManifold instance.
        gin_hyp: GINConv for hyperbolic tangent space aggregation.
        gin_euc: GINConv for Euclidean space aggregation.
        ablation: Ablation variant. "euclidean" disables hyperbolic ops.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        hidden_dim: int = 128,
        num_layers: int = 1,
        ablation: str = "full",
    ):
        """Initialize RiemannianGraphConv.

        Args:
            manifold: ProductManifold instance.
            hidden_dim: MLP hidden dimension (default 2 * d_total for GIN).
            num_layers: Number of GIN layers (unused, kept for interface compat).
            ablation: Ablation variant name. "euclidean" replaces hyperbolic
                      operations with linear layers.
        """
        super().__init__()
        self.manifold = manifold
        self.d_h = manifold.d_h
        self.d_e = manifold.d_e
        self.ablation = ablation

        mlp_hidden = 2 * (self.d_h + self.d_e)

        # GIN for hyperbolic tangent space operations
        if ablation == "euclidean":
            # Ablation: replace hyperbolic ops with standard linear
            self.hyp_linear = nn.Linear(self.d_h, self.d_h)
        else:
            self.gin_hyp = GINConv(
                nn=make_mlp(self.d_h, mlp_hidden, self.d_h),
                train_eps=True,
            )

        # GIN for Euclidean space (standard)
        self.gin_euc = GINConv(
            nn=make_mlp(self.d_e, mlp_hidden, self.d_e),
            train_eps=True,
        )

        # Bug 7 fix: make residual step size learnable instead of hardcoded 0.1.
        # Using sigmoid constraint ensures alpha stays in (0, 1) for stability.
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

        # Degree-capped aggregation for unbounded-degree graphs.
        # GIN sum-aggregation amplifies hub-node messages by O(in-degree);
        # on social graphs (REDDIT-M5K: hub degree up to ~880) this explodes
        # the Euclidean state (observed ||z_euc|| ~ 5e5 -> logits ~ -1e8 ->
        # loss > 50 -> every training episode skipped). Nodes with in-degree
        # above DEG_CAP have incoming messages rescaled so the effective
        # in-degree is bounded by DEG_CAP. All Tier-1 + DD benchmarks have
        # max degree <= 25 < DEG_CAP, so the scale is exactly 1 and the
        # standard GINConv path is used (bit-wise identical behavior).
        self.deg_cap = 32
        # Diagnostic: track max node-level Euclidean norm (for paper evidence
        # that bounded-degree graphs operate far below any pathological range).
        # Disabled by default: the per-RGC .item() sync serialised the CUDA
        # pipeline (~16 syncs/forward). Set self.track_euc_norm = True when
        # collecting the metric; the value is then read lazily (one sync per
        # access) via the euc_max_norm property below.
        self.track_euc_norm = False
        self._euc_max_norm_t = None      # tensor max accumulator (no sync)
        self._euc_max_norm_val = 0.0     # last materialised value

    def _capped_gin(self, gin, x, edge_index):
        """GINConv with degree-capped aggregation (hub-safe).

        For nodes with in-degree > DEG_CAP, incoming messages are rescaled by
        DEG_CAP / in-degree (GCN-style mean normalization applied only to
        hubs), bounding the aggregation amplification at O(DEG_CAP). For
        bounded-degree graphs (in-degree <= DEG_CAP) the scale is exactly 1
        and the stock GINConv path is used, guaranteeing bit-wise identical
        behavior on MUTAG/PROTEINS/ENZYMES/DD (max degree <= 25).
        """
        from torch_geometric.utils import degree, scatter
        dst = edge_index[1]
        deg = degree(dst, num_nodes=x.size(0))
        # Branch on whether any node is a hub. The .any() host-sync is skipped
        # during CUDA-graph capture (fixed-shape input => hub presence is
        # invariant across replays); the decision is computed and cached on the
        # eager run that precedes capture.
        if getattr(self, "_suppress_host_sync", False):
            any_hub = getattr(self, "_deg_hub_flag", False)
        else:
            any_hub = bool((deg > self.deg_cap).any())
            self._deg_hub_flag = any_hub
        if not any_hub:
            return gin(x, edge_index)
        scale = (self.deg_cap / deg.clamp(min=1.0)).clamp(max=1.0)
        src = edge_index[0]
        agg = scatter(
            x[src] * scale[dst].unsqueeze(-1), dst,
            dim=0, dim_size=x.size(0), reduce='sum',
        )
        return gin.nn(agg + (1 + gin.eps) * x)

    def forward(
        self,
        z: Point,
        edge_index: torch.Tensor,
    ) -> Point:
        """Perform one round of Riemannian graph convolution.

        Args:
            z: (z_hyp, z_euc) product manifold node embeddings.
               z_hyp: (N, d_h), z_euc: (N, d_e).
            edge_index: (2, E) edge indices.

        Returns:
            z_new: (z_hyp_new, z_euc_new) updated embeddings on the product manifold.
        """
        z_hyp, z_euc = z

        # Add self-loops for stable message passing
        edge_index_with_sl, _ = add_self_loops(
            edge_index, num_nodes=z_hyp.size(0)
        )

        # ── Hyperbolic component ───────────────────────────────────
        if self.ablation == "euclidean":
            # Ablation: linear layer replaces geodesic message passing
            h_hyp = self.hyp_linear(z_hyp)
            z_hyp_new = self.manifold.hyp_projx(h_hyp)
        else:
            # Step 1: Logmap all neighbors to tangent space at origin
            # GIN aggregates in tangent space (Euclidean), then we project
            # the result back to the manifold via expmap0.
            log_hyp = self.manifold.hyp.logmap0(self.manifold.hyp_projx(z_hyp))

            # Step 2: GIN aggregation in tangent space (degree-capped)
            h_hyp = self._capped_gin(self.gin_hyp, log_hyp, edge_index_with_sl)

            # Step 3: Expmap back to manifold
            z_hyp_new = self.manifold.hyp.expmap0(h_hyp)
            z_hyp_new = self.manifold.hyp_projx(z_hyp_new)

            # Residual connection with geodesic interpolation
            # Bug 7 fix: use learnable alpha (sigmoid-constrained to (0,1))
            alpha = torch.sigmoid(self.residual_alpha)
            # Dedup: hyp_projx(z_hyp) was being computed twice (logmap arg and
            # expmap arg). Hoist both projections once; this also trims the
            # number of kernel launches in the hottest per-step op.
            z_hyp_proj = self.manifold.hyp_projx(z_hyp)
            z_hyp_new_proj = self.manifold.hyp_projx(z_hyp_new)
            v_hyp = self.manifold.hyp.logmap(z_hyp_new_proj, z_hyp_proj)
            z_hyp_new = self.manifold.hyp.expmap(alpha * v_hyp, z_hyp_proj)
            z_hyp_new = self.manifold.hyp_projx(z_hyp_new)

        # ── Euclidean component ────────────────────────────────────
        z_euc_new = self._capped_gin(self.gin_euc, z_euc, edge_index_with_sl)

        # Residual connection (Bug 7 fix: use same learnable alpha)
        alpha_euc = torch.sigmoid(self.residual_alpha)
        z_euc_new = z_euc + alpha_euc * (z_euc_new - z_euc)

        # Diagnostic: track max node-level Euclidean norm. Gated so the
        # per-RGC .item() sync disappears from the hot path during training.
        if self.track_euc_norm:
            nrm = z_euc_new.norm(dim=-1)
            mx_t = nrm.max()  # tensor, no host sync
            self._euc_max_norm_t = (
                mx_t if self._euc_max_norm_t is None
                else torch.maximum(self._euc_max_norm_t, mx_t)
            )

        return (z_hyp_new, z_euc_new)

    @property
    def euc_max_norm(self) -> float:
        """Lazily-materialised max Euclidean node norm (one sync on read).

        Returns the running max accumulated since the last read and resets the
        tensor accumulator, so each access reflects the most recent forward.
        When self.track_euc_norm is False the accumulator is never populated
        and this returns the cached value with NO host synchronisation.
        """
        if self._euc_max_norm_t is not None:
            self._euc_max_norm_val = float(self._euc_max_norm_t.item())
            self._euc_max_norm_t = None
        return self._euc_max_norm_val

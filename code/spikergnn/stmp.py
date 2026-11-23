"""
Spiking Temporal Message Passing (STMP).

T-step temporal loop: RGC -> ManifoldLIFNeuron -> accumulate.
Implements spike-driven sparse communication where only spiking nodes
transmit messages at each time step.

Time pooling:
  - Hyperbolic component: Fréchet mean over T steps.
  - Euclidean component: arithmetic mean over T steps.

Global pooling:
  - global_mean_pool for graph-level embeddings.

Output: (z_hyp_graph, z_euc_graph), spike_count_total
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from torch_geometric.nn import global_mean_pool

from spikergnn.manifolds import ProductManifold, Point
from spikergnn.rgc import RiemannianGraphConv
from spikergnn.neurons import ManifoldLIFNeuron
from spikergnn.dvm import apply_dvm


class SpikingTemporalMP(nn.Module):
    """Spiking Temporal Message Passing layer.

    Iterates T time steps, each performing:
      1. Riemannian Graph Convolution (geodesic message passing)
      2. Manifold LIF neuron update (spike generation)
      3. Spike-driven sparse communication mask

    After T steps, applies temporal pooling (Fréchet mean for hyperbolic,
    arithmetic mean for Euclidean) and global graph pooling.

    Attributes:
        manifold: ProductManifold instance.
        rgc: RiemannianGraphConv instance.
        lif: ManifoldLIFNeuron instance.
        T: Number of temporal steps.
        use_dvm: Whether to apply Differentiation-via-Manifold.
        dvm_window: Truncation window size for DvM.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        hidden_dim: int = 128,
        T: int = 8,
        num_layers: int = 3,
        beta_leak: float = 0.3,
        gamma_input: float = 0.5,
        alpha_decay: float = 0.9,
        u_threshold: float = 0.3,
        delta_reset: float = 1.0,
        surrogate_beta: float = 5.0,
        surrogate_type: str = "atan",
        use_dvm: bool = True,
        dvm_window: int = 8,
        ablation: str = "full",
        noise_std: float = 0.05,
        checkpoint_rgc: bool = False,
    ):
        """Initialize SpikingTemporalMP.

        Args:
            manifold: ProductManifold instance.
            hidden_dim: Total representation dimension.
            T: Number of temporal steps.
            num_layers: Number of RGC layers.
            beta_leak: LIF leak parameter.
            gamma_input: LIF input scaling parameter.
            alpha_decay: Membrane potential decay.
            u_threshold: Spike threshold.
            delta_reset: Refractory reset amount.
            surrogate_beta: Surrogate gradient steepness.
            surrogate_type: "atan" (recommended) or "sigmoid".
            use_dvm: Whether to use DvM for truncated BPTT.
            dvm_window: DvM truncation window size.
            ablation: Ablation variant name.
        """
        super().__init__()
        self.manifold = manifold
        self.T = T
        self.use_dvm = use_dvm
        self.dvm_window = dvm_window
        self.ablation = ablation
        # Path-2 memory fix: checkpoint the per-step RGC forward stack so its
        # internal saved tensors are dropped during the forward pass and
        # recomputed in backward (see config.checkpoint_rgc docstring for the
        # full rationale; only active under model.training).
        self.checkpoint_rgc = checkpoint_rgc

        # Riemannian Graph Convolution layers
        self.rgc_layers = nn.ModuleList([
            RiemannianGraphConv(
                manifold=manifold,
                hidden_dim=hidden_dim,
                ablation=ablation,
            )
            for _ in range(num_layers)
        ])

        # Manifold LIF neuron
        self.lif = ManifoldLIFNeuron(
            manifold=manifold,
            beta_leak=beta_leak,
            gamma_input=gamma_input,
            alpha_decay=alpha_decay,
            u_threshold=u_threshold,
            delta_reset=delta_reset,
            surrogate_beta=surrogate_beta,
            surrogate_type=surrogate_type,
            ablation=ablation,
            noise_std=noise_std,
        )

        self._total_spike_count = 0
        self._total_spike_count_acc = None  # GPU tensor accumulator (no per-step sync)

        # Measurement hooks for event-driven energy accounting (no gradient
        # path). debug_spike_gates[t] holds the node-level binary spike mask
        # (N,) for time step t; debug_sop holds the spike-driven synaptic
        # operation count Σ_t Σ_v s_t[v]·deg(v) over ALL T steps. (t=0 is NOT
        # excluded: the paper's bar{r} sums over all T steps, and the t=0
        # "open gate" is a message-passing initialization that is a separate
        # dense cost, independent of whether the LIF actually spikes.)
        self.debug_spike_gates = None   # list[T] of (N,) binary tensors
        self.debug_sop = 0              # int, spike-driven SOP (all T steps)
        self._sop_acc = None            # GPU tensor accumulator for debug_sop

    def _rgc_stack(self, z_hyp, z_euc, edge_index):
        """One full pass through all RGC layers (used by gradient checkpointing).

        Isolated into its own callable so torch.utils.checkpoint can discard its
        internal saved tensors during the forward pass and recompute them in
        backward. The LIF neuron and spike-driven masking are intentionally NOT
        inside this function: see config.checkpoint_rgc for the rationale (the
        LIF's cuRNG membrane noise would differ between the forward and the
        checkpoint recompute, corrupting the gradient, whereas the RGC is a
        deterministic pure function of its inputs so the recompute is exact).
        """
        z = (z_hyp, z_euc)
        for rgc_layer in self.rgc_layers:
            z = rgc_layer(z, edge_index)
        return z

    def forward(
        self,
        p_init: Point,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pool: str = "graph",
    ) -> Tuple[Point, int]:
        """Run T-step spiking temporal message passing.

        Args:
            p_init: (p_hyp_0, p_euc_0) initial manifold state.
            x_seq: (N, T, d_total) input spike sequence from MSPE.
            edge_index: (2, E) edge indices.
            batch: (N,) node-to-graph assignment vector.

        Returns:
            z_graph: (z_hyp_graph, z_euc_graph) graph-level embeddings.
                     z_hyp_graph: (num_graphs, d_h), z_euc_graph: (num_graphs, d_e).
            spike_count_total: Total number of spikes across all steps.
        """
        N = x_seq.size(0)
        p_hyp, p_euc = p_init
        p_hyp = p_hyp.clone()
        p_euc = p_euc.clone()

        # ── Ablation: "no_stmp" ────────────────────────────────────
        # Remove the spiking temporal message-passing loop and replace it with
        # a single, non-spiking RGC pass followed directly by global pooling.
        # This isolates the contribution of the STMP module (T-step temporal
        # integration, LIF gating, spike-driven sparse communication, temporal
        # pooling and DvM) while keeping the rest of the pipeline
        # (MSPE + RGC + FPC + GDC) intact.
        # NOTE: distinct from "no_spikes", which keeps the T-step loop and only
        # swaps the LIF for an identity pass.
        if self.ablation == "no_stmp":
            z_current = (p_hyp, p_euc)
            for rgc_layer in self.rgc_layers:
                z_current = rgc_layer(z_current, edge_index)
            z_hyp_s, z_euc_s = z_current
            z_hyp_s = self.manifold.hyp_projx(z_hyp_s)
            if pool == "node":
                # Per-node embeddings (after RGC, before global pooling).
                self._total_spike_count = 0
                self._total_spike_count_acc = None
                self.debug_spike_gates = []
                self.debug_sop = 0
                self._sop_acc = None
                return (z_hyp_s, z_euc_s), 0
            z_hyp_graph = self.manifold.hyp.projx(
                global_mean_pool(z_hyp_s, batch)
            )
            z_euc_graph = global_mean_pool(z_euc_s, batch)
            self._total_spike_count = 0
            self._total_spike_count_acc = None
            self.debug_spike_gates = []
            self.debug_sop = 0
            self._sop_acc = None
            return (z_hyp_graph, z_euc_graph), 0

        # Accumulate representations for temporal pooling
        hyp_accumulator: List[torch.Tensor] = []
        euc_accumulator: List[torch.Tensor] = []

        # Track states for DvM
        step_states: List[Point] = []

        # ── DvM memory windowing ────────────────────────────────
        # When DvM is active (training), only the last `dvm_window` steps need
        # live autograd graphs; earlier steps' values are still required for
        # temporal pooling but their gradients are truncated (standard truncated
        # BPTT), so we store them detached. This bounds the forward/backward
        # autograd graph to ~window/T of the full T-step graph, turning a
        # 169k-node full-graph forward (which needed >31 GB and OOM'd on a
        # 22 GB device) into a tractable ~window/T memory footprint.
        # NOTE: the default dv_m_window==T means no truncation (legacy behaviour);
        # set dv_m_window < T (e.g. via the ogbn-arxiv runner) to realise the
        # memory saving.
        dvm_active = self.use_dvm and self.training
        dvm_start = max(0, self.T - self.dvm_window) if dvm_active else 0

        self._total_spike_count = 0
        self._total_spike_count_acc = None
        # Reset episode-level spike stats so the reported rate reflects this
        # single encode pass (all T steps), not accumulated across episodes.
        self.lif.reset_episode_stats()

        # Measurement hooks for event-driven SOP accounting.
        self.debug_spike_gates = []
        self.debug_sop = 0
        self._sop_acc = None
        # Node degree (number of incident edges) for spike-driven SOP counting:
        # each spike at node v triggers one synaptic operation per incident edge.
        # degree is computed once here and reused across time steps.
        if edge_index is not None and edge_index.numel() > 0:
            degree = torch.zeros(N, dtype=torch.float32, device=edge_index.device)
            degree.scatter_add_(
                0,
                edge_index[0].long(),
                torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device),
            )
        else:
            degree = torch.zeros(N, dtype=torch.float32, device=x_seq.device)

        # Track previous step's LIF spike output for autoregressive masking
        prev_lif_spikes = None  # Will be set after first LIF call

        for t in range(self.T):
            # ── RGC: geodesic message passing ──────────────────────
            if self.checkpoint_rgc and self.training:
                # Gradient checkpoint (path-2 memory fix): discard the RGC
                # stack's internal saved tensors during the forward pass and
                # recompute them in backward. This is the ONLY mechanism that
                # lowers the FORWARD-peak memory of the T-loop — truncated-BPTT
                # (dv_m_window) only cuts the gradient chain, leaving all T
                # steps' saved tensors live simultaneously during the forward
                # (so it never reduces the forward high-water mark). RGC is a
                # deterministic pure function of (p_hyp, p_euc, edge_index), so
                # the recompute is exact and differs from the surrounding LIF
                # (which carries cuRNG noise and is therefore NOT checkpointed).
                z_current = torch.utils.checkpoint.checkpoint(
                    self._rgc_stack, p_hyp, p_euc, edge_index,
                    use_reentrant=False,
                )
            else:
                z_current = (p_hyp, p_euc)
                for rgc_layer in self.rgc_layers:
                    z_current = rgc_layer(z_current, edge_index)

            z_hyp_new, z_euc_new = z_current

            # ── Spike-driven masking ───────────────────────────────
            # Autoregressive mask: use STMP LIF's own spike output from the
            # previous time step as the communication gate, not MSPE's.
            # Rationale: MSPE's LIF may not fire (e.g., with u_th=1.0, input
            # magnitude after projection is too small), which would block ALL
            # communication. Using the STMP LIF's own spikes creates a more
            # biologically plausible self-regulating circuit where neurons that
            # fired recently gate their own message passing.
            # For t=0: all nodes communicate (initialization — no prior spikes).
            if t == 0 or prev_lif_spikes is None:
                # First step: all nodes communicate (open gate)
                spike_mask = torch.ones(N, 1, device=x_seq.device)  # (N, 1)
            else:
                # Subsequent steps: only spiking nodes transmit
                spike_mask = prev_lif_spikes  # (N, 1) — from previous LIF output

            # Expand mask to match manifold dimensions
            spike_mask_hyp = spike_mask.expand_as(p_hyp)  # (N, d_h)
            spike_mask_euc = spike_mask.expand_as(p_euc)  # (N, d_e)

            # Apply sparse communication: blend based on spike activity
            # Spiking nodes get updated representation, non-spiking keep old
            p_hyp = self.manifold.hyp_projx(
                spike_mask_hyp * z_hyp_new + (1 - spike_mask_hyp) * p_hyp
            )
            p_euc = spike_mask_euc * z_euc_new + (1 - spike_mask_euc) * p_euc

            # ── ManifoldLIF neuron update ──────────────────────────
            # Create input for LIF: use RGC-processed continuous features rather than
            # raw binary spikes. The RGC output is a continuous manifold representation
            # which is more appropriate as LIF input (matches the expected ||W*I|| ~ O(1)
            # after Bug 4 normalization). Binary spikes would give ||W*spike||/sqrt(d) ~ 0.06,
            # too small to reach threshold.
            rgc_features = self.manifold.merge(z_hyp_new, z_euc_new)  # (N, d_total)
            x_step = rgc_features.unsqueeze(1)  # (N, 1, d_total)
            spikes, (p_hyp, p_euc) = self.lif(x_step, (p_hyp, p_euc))

            # ── Truncated-BPTT memory boundary ──────────────────────
            # For steps before the DvM window, detach the recurrent manifold
            # state so the autograd graph does NOT chain across the whole
            # T-loop. Each step's live graph would otherwise be retained as the
            # input to the next step's RGC, forcing the full T-step graph to
            # stay alive during the forward. Detaching here is standard
            # truncated BPTT: gradient flows only through the last `dvm_window`
            # steps, and bounds peak memory to ~window/T of the full graph
            # (169k forward: 31 GB full graph -> ~window/T GB). The detached
            # values are still captured by the temporal-pooling accumulator,
            # so the forward output is unchanged; only the gradient path is
            # truncated.
            if dvm_active and t < dvm_start:
                p_hyp = p_hyp.detach()
                p_euc = p_euc.detach()

            # Store LIF spike output for next step's mask (detach to avoid
            # double-backprop through the mask path)
            # spikes shape: (N, 1, d_total), take mean across d as binary gate
            with torch.no_grad():
                s_out_binary = (spikes[:, 0, :].mean(dim=-1, keepdim=True) > 0.5).float()  # (N, 1)
                prev_lif_spikes = s_out_binary

            # Track spike count (accumulate as GPU tensor; materialise the
            # integer once at the end of forward to avoid per-step host syncs)
            sc_t = self.lif.spike_count_tensor
            if sc_t is not None:
                self._total_spike_count_acc = (
                    sc_t if self._total_spike_count_acc is None
                    else self._total_spike_count_acc + sc_t
                )

            # Measurement hook: record the node-level binary spike mask of this
            # step (detached) and accumulate spike-driven SOP. Per the paper's
            # definition (bar{r} = mean over ALL N neurons and T steps), every
            # spike event triggers one synaptic operation per incident edge, so
            # SOP = Σ_{t=0..T-1} Σ_v s_t[v]·deg(v) with NO step excluded. The
            # t=0 "open gate" is a message-passing initialization (all nodes
            # communicate) and is a separate dense cost, not part of spike SOP.
            s_t = self.lif._last_spike_mask  # (N,) binary, detached
            if s_t is not None:
                self.debug_spike_gates.append(s_t)
                # Perf: accumulate spike-driven SOP as a GPU tensor during the
                # loop and materialise to self.debug_sop once at the end of
                # forward (single host sync) instead of .item() every step.
                sop_t = (s_t * degree).sum()
                self._sop_acc = (
                    sop_t if self._sop_acc is None else self._sop_acc + sop_t
                )

            # Accumulate for temporal pooling
            # Outside the DvM window: detach (values needed for pooling, but
            # gradients are truncated -> no live graph held -> bounded memory).
            if dvm_active and t < dvm_start:
                hyp_accumulator.append(p_hyp.detach())
                euc_accumulator.append(p_euc.detach())
            else:
                hyp_accumulator.append(p_hyp)
                euc_accumulator.append(p_euc)

            # Save state for DvM
            step_states.append((p_hyp.detach().clone(), p_euc.detach().clone()))

        # Materialise spike-count / SOP integers ONCE here (single host sync
        # per forward) so the hot loop never calls .item().
        if self._total_spike_count_acc is not None:
            if getattr(self, "_suppress_host_sync", False):
                self._total_spike_count_acc_captured = self._total_spike_count_acc
            else:
                self._total_spike_count = int(self._total_spike_count_acc.item())
        if self._sop_acc is not None:
            if getattr(self, "_suppress_host_sync", False):
                self._sop_acc_captured = self._sop_acc
            else:
                self.debug_sop = int(self._sop_acc.item())

        # ── DvM: apply truncated BPTT ─────────────────────────────
        # DvM replaces the last accumulator entry with its output so that
        # gradients flow through DvM's custom backward (truncated BPTT with
        # conformal factor correction) instead of the full autograd graph.
        # Without this, the DvM output was assigned to p_hyp but never used
        # in temporal pooling — making full and no_dvm identical.
        if self.use_dvm and self.training:
            p_final = apply_dvm(
                manifold=self.manifold,
                p_init=p_init,
                step_fn_outputs=step_states,
                window=self.dvm_window,
            )
            p_hyp_dvm, p_euc_dvm = p_final
            # Replace the last accumulator entry so DvM's gradient path
            # is connected to the temporal pooling computation graph
            hyp_accumulator[-1] = p_hyp_dvm
            euc_accumulator[-1] = p_euc_dvm

        # ── Temporal pooling ───────────────────────────────────────
        # Hyperbolic: Fréchet mean over time steps (per node)
        hyp_stack = torch.stack(hyp_accumulator, dim=0)  # (T, N, d_h)
        euc_stack = torch.stack(euc_accumulator, dim=0)  # (T, N, d_e)

        # Vectorized tangent-space-at-origin approximation for temporal pooling.
        # Instead of a per-node Python loop (O(N) calls), we batch all nodes:
        #   1. logmap0 maps all (T, N, d_h) points to tangent space at origin
        #   2. Mean over T steps in tangent space
        #   3. expmap0 maps the means back to the manifold
        # This is mathematically identical but ~100x faster due to GPU batching.
        z_hyp_pooled = self.manifold.hyp.expmap0(
            self.manifold.hyp.logmap0(hyp_stack).mean(dim=0)  # (N, d_h)
        )
        z_hyp_pooled = self.manifold.hyp.projx(z_hyp_pooled)

        # Euclidean: arithmetic mean
        z_euc_pooled = euc_stack.mean(dim=0)  # (N, d_e)

        # ── Node-level early return (skip global graph pooling) ─────
        # Used by node classification: keep per-node temporal-pooled
        # embeddings on the product manifold; the caller maps the
        # hyperbolic part to tangent space (logmap0) and concatenates
        # with the Euclidean part before a linear classification head.
        if pool == "node":
            return (z_hyp_pooled, z_euc_pooled), self._total_spike_count

        # ── Global graph pooling ───────────────────────────────────
        z_hyp_graph = global_mean_pool(z_hyp_pooled, batch)  # (num_graphs, d_h)
        z_euc_graph = global_mean_pool(z_euc_pooled, batch)  # (num_graphs, d_e)

        # Ensure hyperbolic component stays on manifold
        z_hyp_graph = self.manifold.hyp.projx(z_hyp_graph)

        return (z_hyp_graph, z_euc_graph), self._total_spike_count

    @property
    def spike_count(self) -> int:
        """Return total spike count from the last forward pass."""
        return self._total_spike_count

"""
Manifold-valued Leaky Integrate-and-Fire (LIF) neuron.

Implements spike neuron dynamics on a product manifold H^{d_h} x R^{d_e}:

  State evolution (Eq. 3):
    p(t+1) = exp_{p(t)}(beta * logmap(p_bar, p(t)) + gamma * I(t))

  Membrane potential (Eq. 4):
    ||u(t+1)|| = alpha * ||u(t)|| + W * s_in(t) - s_out(t) * delta_reset

  Spike trigger (Eq. 5):
    s(t+1) = SigmoidSurrogate(||u(t+1)|| - u_threshold)

Includes SigmoidSurrogate for surrogate gradient (reimplemented from
msg_baseline.models.SigmoidSurrogate logic, not imported).

Bug 4 fix: w_input_mag is divided by sqrt(d_total) to normalize the input
magnitude to match the threshold. Without this, ||W*I|| ~ O(sqrt(d_total))
far exceeds u_threshold=1.0, causing all neurons to fire every step.
The membrane potential u_mag is persisted across forward calls (stored as a
buffer) so that the LIF can be called with T=1 from STMP without losing state.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from spikergnn.manifolds import ProductManifold, Point, TangentVec


class SigmoidSurrogate(nn.Module):
    """Sigmoid-based surrogate gradient for spike non-differentiability.

    During forward pass: returns binary spike (0 or 1).
    During backward pass: passes sigmoid gradient with controllable steepness.

    This is a standalone reimplementation matching msg_baseline.models.SigmoidSurrogate
    logic, to avoid circular imports.

    Attributes:
        beta: Steepness of the sigmoid surrogate gradient.
    """

    def __init__(self, beta: float = 5.0):
        """Initialize surrogate gradient function.

        Args:
            beta: Steepness parameter for the sigmoid. Higher = sharper.
        """
        super().__init__()
        self.beta = beta

    def forward(self, v_mem: torch.Tensor) -> torch.Tensor:
        """Apply surrogate gradient spike function.

        Args:
            v_mem: Membrane potential values (any shape).

        Returns:
            Binary spikes in forward, with sigmoid gradient in backward.
        """
        # Forward: hard threshold
        spike = (v_mem >= 0).float()

        # Backward: surrogate sigmoid gradient
        if self.training:
            sig = torch.sigmoid(self.beta * v_mem)
            surrogate = sig * (1 - sig) * self.beta
            return spike - surrogate.detach() + surrogate
        return spike


class ATanSurrogate(nn.Module):
    """Arc-tangent (ATan) surrogate gradient for spike non-differentiability.

    During forward pass: returns binary spike (0 or 1).
    During backward pass: passes atan gradient which does NOT vanish
    for large |v_mem|, unlike the sigmoid surrogate.

    The ATan surrogate gradient is:
        g'(x) = β / (π * (1 + (β*x/π)²))

    Key advantage over SigmoidSurrogate: the gradient magnitude decays
    as 1/x² (polynomial) rather than exponentially, providing non-trivial
    gradient signal even when u_mag >> u_th (saturated firing regime).
    This is critical for spike rate regularization (L_sr), which needs
    to push u_th upward when the network fires too frequently.

    Reference:
        Wu et al., "A Direct Training Algorithm for Spiking Neural Networks",
        AAAI 2021 (uses ATan surrogate for better gradient flow).

    Attributes:
        beta: Steepness of the ATan surrogate gradient.
    """

    def __init__(self, beta: float = 2.0):
        """Initialize ATan surrogate gradient function.

        Args:
            beta: Steepness parameter. Default 2.0 (empirically effective).
        """
        super().__init__()
        self.beta = beta

    def forward(self, v_mem: torch.Tensor) -> torch.Tensor:
        """Apply ATan surrogate gradient spike function.

        Args:
            v_mem: Membrane potential values (any shape).

        Returns:
            Binary spikes in forward, with ATan gradient in backward.
        """
        # Forward: hard threshold
        spike = (v_mem >= 0).float()

        # Backward: ATan surrogate gradient
        if self.training:
            # g(x) = atan(β*x) / π + 0.5  (shifted atan to range [0,1])
            # g'(x) = β / (π * (1 + (β*x/π)²))
            # Key: gradient decays as 1/x², not exponentially!
            pi = math.pi
            bp = self.beta / pi
            surrogate_grad = bp / (1 + (bp * v_mem) ** 2)
            # STE: forward = spike, backward = surrogate_grad
            return spike - surrogate_grad.detach() + surrogate_grad
        return spike


class ManifoldLIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron on a product manifold.

    The neuron state p(t) evolves ON the manifold (not in tangent space),
    using the Riemannian exponential map. Membrane potential magnitude
    determines spike firing via a surrogate gradient.

    Bug 4 fix: w_input_mag is normalized by sqrt(d_total) so that ||W*I||/sqrt(d)
    is O(1), matching the threshold u_th=1.0 and preventing all-fire behavior.
    The membrane potential u_mag is persisted across forward calls as a buffer,
    so calling with T=1 repeatedly (as STMP does) correctly accumulates potential.

    Attributes:
        manifold: ProductManifold instance.
        beta: Learnable leak decay parameter.
        gamma: Learnable input scaling parameter.
        alpha: Membrane potential decay constant.
        u_th: Spike firing threshold.
        delta_r: Refractory reset amount.
        surrogate: SigmoidSurrogate instance.
        W: Learnable synaptic weight matrix (d_total x d_total).
        p_bar_decay: Exponential moving average decay for p_bar.
    """

    def __init__(
        self,
        manifold: ProductManifold,
        beta_leak: float = 0.3,
        gamma_input: float = 0.5,
        alpha_decay: float = 0.9,
        u_threshold: float = 0.3,
        delta_reset: float = 1.0,
        surrogate_beta: float = 5.0,
        surrogate_type: str = "atan",
        p_bar_decay: float = 0.9,
        ablation: str = "full",
        noise_std: float = 0.05,
    ):
        """Initialize ManifoldLIFNeuron.

        Args:
            manifold: ProductManifold instance.
            beta_leak: Initial value for learnable leak parameter.
            gamma_input: Initial value for learnable input scaling parameter.
            alpha_decay: Membrane potential decay constant.
            u_threshold: Spike threshold.
            delta_reset: Refractory reset amount.
            surrogate_beta: Steepness of surrogate gradient.
            surrogate_type: "sigmoid" or "atan". ATan is recommended because
                its gradient decays polynomially (1/x²) rather than
                exponentially, enabling effective spike rate regularization.
            p_bar_decay: EMA decay for maintaining reference point p_bar.
            ablation: Ablation variant. "no_spikes" disables spiking.
        """
        super().__init__()
        self.manifold = manifold
        self.d_total = manifold.d_h + manifold.d_e
        self.alpha = alpha_decay
        self.u_th = u_threshold
        self.delta_r = delta_reset
        self.p_bar_decay = p_bar_decay
        self.ablation = ablation
        self.noise_std = noise_std  # Membrane noise to break deterministic oscillation

        # Learnable parameters
        self.beta = nn.Parameter(torch.tensor(beta_leak))
        self.gamma = nn.Parameter(torch.tensor(gamma_input))
        self.W = nn.Linear(self.d_total, self.d_total, bias=False)

        # Input normalization: ensures w_input_mag stays O(1) regardless of
        # RGC output scale changes during training. Without this, training
        # causes RGC weights to shrink, making w_input_mag ~0.01 and preventing
        # any spikes from firing (u_mag never reaches u_th).
        self.input_ln = nn.LayerNorm(self.d_total)

        # Bug 5 fix: learnable geodesic interpolation rate (replaces hardcoded alpha_mix)
        self.spike_alpha = nn.Parameter(torch.tensor(0.5))

        # Bug 4 fix: learnable threshold. After normalizing w_input_mag by 1/sqrt(d_total),
        # the effective input magnitude varies: ~0.2 for dense features, ~0.08 for sparse
        # Poincaré-projected features. However, with W multiplication, the effective
        # input magnitude is ~0.5-0.6 after normalization. Setting u_th_init too low
        # (e.g., 0.35) causes every time step to fire (input > threshold).
        # Use u_th_init = 1.0 (original paper value) which gives ~30-50% spike rate
        # for typical inputs: the membrane potential accumulates over 2-3 steps via
        # alpha decay before exceeding threshold.
        # NOTE (node-level fix): the init is now driven by the config/CLI u_threshold
        # (default 1.0, so graph-task behavior is unchanged). On small-magnitude
        # node-level features (ogbn-arxiv) the rate collapses to ~0 at u_th=1.0, so
        # the threshold must be lowered via this arg to actually engage spiking.
        _u_th_init_val = u_threshold
        self.u_th_learnable = nn.Parameter(torch.tensor(_u_th_init_val))
        # Fixed reference for the input-magnitude clamp. Decoupling the clamp
        # cap from the *learnable* threshold is essential: if cap = 0.6*u_th
        # (proportional), the firing period of a clamped neuron depends only on
        # (alpha, 0.6) and is INVARIANT to u_th, so the learnable threshold has
        # zero control authority over the firing rate and L_sr is structurally
        # void. Using the init value as a fixed cap restores u_th's control.
        self._u_th_init_ref = float(_u_th_init_val)

        # Surrogate gradient (ATan by default for better gradient flow with L_sr)
        if surrogate_type == "atan":
            self.surrogate = ATanSurrogate(beta=surrogate_beta)
        else:
            self.surrogate = SigmoidSurrogate(beta=surrogate_beta)

        # Spike count tracking
        self._spike_count = 0
        self._spike_count_acc = None   # GPU tensor accumulator (no per-step sync)
        self._total_neurons = 0   # N * T for spike rate computation
        self._spike_rate = 0.0    # Empirical spike rate from last forward pass
        self._last_spike_mask = None  # (N,) binary node spike mask of last step (measurement hook)

        # Episode-level accumulators (fix for the measurement artifact where
        # _spike_rate only reflected the LAST T=1 LIF call). STMP calls lif T
        # times with T=1; these aggregate across ALL calls within one encode
        # pass, so spike_rate_surrogate() returns the TRUE mean firing rate
        # over the full T-step episode, not a single step's phase.
        self._episode_surrogate_sum = torch.tensor(0.0)
        self._episode_total_neurons = 0
        self._episode_spike_count = 0
        self._episode_spike_count_acc = None  # GPU tensor accumulator

        # Bug 4 fix: persistent membrane potential as a buffer.
        # This ensures u_mag is preserved across forward calls when STMP
        # calls LIF with T=1 repeatedly. Without this, u_mag resets to 0
        # on every call, preventing potential accumulation for sparse inputs.
        self.register_buffer("_u_mag", None)

        # Track the number of nodes for shape validation
        self._last_N = None

        # Optional static per-neuron threshold heterogeneity (rate-control mechanism
        # diagnostic).  Default None => behaviour is BIT-IDENTICAL to the
        # published model.  When set to a (N,) tensor by the rate-control
        # diagnostic, a FIXED offset in added to the learnable threshold.  This
        # provides a noise-free way to break the identical-input period-2 lock
        # (no time-varying randomness), isolating whether L_sr's control
        # authority needs stochastic jitter or merely desynchronization.
        self.register_buffer("_het_th_offset", None)

    def _get_u_mag(self, N: int, device: torch.device) -> torch.Tensor:
        """Get or initialize the membrane potential buffer.

        Args:
            N: Number of nodes.
            device: Device for the tensor.

        Returns:
            (N,) membrane potential magnitude tensor.
        """
        if self._u_mag is None or self._u_mag.shape[0] != N or self._u_mag.device != device:
            self._u_mag = torch.zeros(N, device=device)
        return self._u_mag

    def forward(
        self,
        x_seq: torch.Tensor,
        p_init: Point,
    ) -> Tuple[torch.Tensor, Point]:
        """Run LIF neuron dynamics over T time steps on the product manifold.

        Args:
            x_seq: (N, T, d_total) input sequence in tangent space at each step.
                   N = number of nodes, T = time steps, d_total = d_h + d_e.
            p_init: (p_hyp, p_euc) initial manifold state.
                    p_hyp: (N, d_h), p_euc: (N, d_e).

        Returns:
            spikes: (N, T, d_total) spike sequence.
            p_final: (p_hyp_final, p_euc_final) final manifold state.
        """
        N, T, d_total = x_seq.shape
        device = x_seq.device

        # Initialize state
        p_hyp, p_euc = p_init
        p_hyp = p_hyp.clone()
        p_euc = p_euc.clone()

        # Bug 4 fix: use persistent membrane potential buffer.
        # This preserves u_mag across forward calls, allowing proper
        # accumulation when called with T=1 from STMP.
        u_mag = self._get_u_mag(N, device)

        # Initialize p_bar as EMA of manifold states
        p_bar_hyp = p_hyp.detach().clone()
        p_bar_euc = p_euc.detach().clone()

        # Ablation: no_spikes mode — skip LIF, use mean of inputs
        if self.ablation == "no_spikes":
            # Identity pass: average over time, no spiking
            x_mean = x_seq.mean(dim=1)  # (N, d_total)
            x_hyp_mean, x_euc_mean = self.manifold.split(x_mean)
            # Project back to manifold
            p_final_hyp = self.manifold.hyp.projx(p_hyp + x_hyp_mean * 0.1)
            p_final_euc = p_euc + x_euc_mean * 0.1
            # Return constant spike (all 1s) as passthrough
            spikes = torch.ones_like(x_seq)
            return spikes, (p_final_hyp, p_final_euc)

        spike_list = []
        self._spike_count = 0
        self._spike_count_acc = None   # reset GPU accumulator for this forward

        # Clamp learnable parameters
        beta_val = self.beta.clamp(min=0.01, max=1.0)
        gamma_val = self.gamma.clamp(min=0.01, max=2.0)

        for t in range(T):
            I_t = x_seq[:, t, :]  # (N, d_total)

            # ── State evolution (Eq. 3) ────────────────────────────
            # Compute tangent direction: beta * logmap(p_bar, p) + gamma * I(t)
            # logmap(p_bar, p) gives tangent vector at p pointing toward p_bar
            # (clamp both base points strictly inside the ball for stability)
            log_bar_hyp = self.manifold.hyp.logmap(
                self.manifold.hyp_projx(p_bar_hyp), self.manifold.hyp_projx(p_hyp)
            )  # (N, d_h)
            log_bar_euc = p_bar_euc - p_euc  # (N, d_e)

            # Split input into manifold components
            I_hyp, I_euc = self.manifold.split(I_t)

            # Tangent vector at p
            v_hyp = beta_val * log_bar_hyp + gamma_val * I_hyp
            v_euc = beta_val * log_bar_euc + gamma_val * I_euc

            # Exponential map: move p along tangent direction
            p_new_hyp = self.manifold.hyp.expmap(v_hyp, self.manifold.hyp_projx(p_hyp))
            p_new_euc = p_euc + v_euc

            # Project back to manifold (clamp strictly inside the ball)
            p_new_hyp = self.manifold.hyp_projx(p_new_hyp)

            # ── Membrane potential (Eq. 4) ─────────────────────────
            # W * s_in: synaptic weighted input (use I_t as s_in for first pass)
            # Apply LayerNorm to stabilize input magnitude across training.
            # Without this, RGC output shrinks during training, causing
            # w_input_mag to drop from ~0.5 to ~0.01 and eliminating all spikes.
            I_t_normed = self.input_ln(I_t)  # normalize to mean=0, std=1
            w_input = self.W(I_t_normed)  # (N, d_total)
            w_input_mag = w_input.norm(dim=-1)  # (N,)

            # Bug 4 fix: normalize by sqrt(d_total) so input magnitude matches threshold.
            # After LayerNorm, ||I_t|| ≈ sqrt(d_total), so ||W*I_t|| ≈ sigma_W * d_total.
            # Dividing by sqrt(d_total) gives w_input_mag ≈ sigma_W * sqrt(d_total) ≈ O(1).
            w_input_mag = w_input_mag / math.sqrt(self.d_total)

            # ── Input magnitude soft clamp (tanh) ──────────────────
            # SOFT clamp via tanh: w = cap * tanh(w / cap).
            # This compresses large inputs toward `cap` while PRESERVING relative
            # differences between neurons. A hard clamp (torch.clamp) forces ALL
            # supra-cap neurons to exactly the same value, causing them to enter
            # identical period-2 oscillation (rate=0.5 invariant to u_th in
            # [0.6, 1.14)). The tanh soft clamp maintains a distribution of input
            # magnitudes, so different neurons have different firing periods,
            # and the population-averaged rate becomes a SMOOTH, MONOTONIC
            # function of u_th — restoring L_sr's control authority.
            soft_cap = self._u_th_init_ref * 0.6
            w_input_mag = soft_cap * torch.tanh(w_input_mag / (soft_cap + 1e-8))

            u_mag = self.alpha * u_mag + w_input_mag

            # ── Membrane potential clamp ───────────────────────────
            # Safety clamp to prevent runaway excitation. With the soft input
            # clamp, the maximum per-step input is ~soft_cap, so steady-state
            # u_mag ≤ soft_cap/(1-α) = 0.6/0.1 = 6.0. The 1.5×init_ref clamp
            # catches any transient overshoot.
            u_mag = torch.clamp(u_mag, max=self._u_th_init_ref * 1.5)

            # ── Membrane potential noise ─────────────────────────
            # Add Gaussian noise to break the deterministic period-2 oscillation
            # that arises when all neurons receive identical clamped input.
            # Without noise, the population rate is quantized (0, 0.25, 0.33,
            # 0.5, 1.0) and invariant to u_th within each plateau, voiding L_sr.
            # With noise, neurons fire at slightly different phases, and the
            # population-averaged rate becomes a SMOOTH, MONOTONIC function of
            # u_th — restoring L_sr's control authority. Applied in BOTH train
            # and eval so the reported (inference) rate reflects the controllable
            # rate, not the deterministic plateau.
            if self.noise_std > 0 and not getattr(self, "_disable_internal_noise", False):
                u_mag = u_mag + torch.randn_like(u_mag) * self.noise_std

            # ── Spike trigger (Eq. 5) ──────────────────────────────
            # Bug 4 fix: use learnable threshold that adapts to input magnitude.
            effective_threshold = self.u_th_learnable.clamp(min=0.05)
            # M1 diagnostic hook: optional STATIC per-neuron threshold offset.
            # Default (None) leaves effective_threshold untouched.  This is a
            # measurement-only extension used to test whether deterministic
            # desynchronization (heterogeneous thresholds) restores L_sr control
            # at sigma=0, without any time-varying noise.
            if self._het_th_offset is not None:
                off = self._het_th_offset
                # Slice a static per-position offset to the current node count;
                # the offset is fixed in time and in position index, so it breaks
                # the identical-input period-2 lock without time-varying noise.
                if off.shape[0] < N:
                    off = off.repeat((N // off.shape[0]) + 1)[:N]
                else:
                    off = off[:N]
                effective_threshold = (effective_threshold + off.to(effective_threshold.device)).clamp(min=0.05)
            s_out = self.surrogate(u_mag - effective_threshold)  # (N,)
            # Defensive stabilizer: if a NaN ever reaches the membrane
            # (e.g. transient manifold saturation despite hyp_projx), neutralize
            # it so one bad step cannot crash the whole run. nan_to_num is a
            # pure CUDA op (no host sync), so it is applied UNCONDITIONALLY;
            # this keeps the CUDA-graph-captured forward bit-identical to the
            # eager forward. The earlier host-sync-gated version skipped it
            # under capture, which silently changed the result whenever NaNs
            # occurred in the membrane output.
            s_out = torch.nan_to_num(s_out, nan=0.0)
            u_mag = torch.nan_to_num(u_mag, nan=0.0)
            # Perf: accumulate spike sums as GPU tensors during the loop and
            # materialise the integer counts ONCE at the end of forward
            # (single host sync) instead of calling .item() every step, which
            # forced a device->host sync that serialised the whole CUDA pipeline.
            s_sum = s_out.sum()
            self._spike_count_acc = (
                s_sum if self._spike_count_acc is None
                else self._spike_count_acc + s_sum
            )
            self._episode_spike_count_acc = (
                s_sum if self._episode_spike_count_acc is None
                else self._episode_spike_count_acc + s_sum
            )
            # Expose the node-level binary spike mask of THIS step (detached,
            # off the autograd path) so downstream energy accounting can count
            # spike-driven synaptic operations per edge without re-deriving the
            # mask from the dimension-broadcast `spikes` tensor. This is a
            # measurement hook only; it does not alter the forward computation.
            self._last_spike_mask = s_out.detach()  # (N,) binary (forward)

            # Track surrogate spike rate for differentiable regularization
            # s_out is already the surrogate gradient output: binary in forward,
            # sigmoid in backward. So taking mean gives differentiable spike rate.
            if t == 0:
                self._surrogate_spike_sum = s_out.sum()
            else:
                self._surrogate_spike_sum = self._surrogate_spike_sum + s_out.sum()
            # Episode-level accumulation: aggregate across ALL T=1 LIF calls in
            # one encode pass so the reported rate is the true episode mean, not
            # the last step's phase (fix for the measurement artifact).
            if self._episode_total_neurons == 0:
                self._episode_surrogate_sum = s_out.sum()
            else:
                self._episode_surrogate_sum = self._episode_surrogate_sum + s_out.sum()
            self._episode_total_neurons += N

            # Membrane potential reset
            # Hard reset: set u_mag = 0 after spike (standard in SNN literature).
            # Previous soft reset (u_mag -= delta_r) failed because u_mag can
            # accumulate far above threshold (α≈0.9 + large input), making
            # the fixed reset amount insufficient. Hard reset prevents
            # immediate re-firing and enables sparse spike trains.
            u_mag = u_mag * (1 - s_out)  # 0 if spiked, unchanged if not

            # ── Apply spike mask to manifold state ─────────────────
            # Spike modulates the state update magnitude
            spike_mask = s_out.unsqueeze(-1)  # (N, 1)

            # Bug 5 fix: geodesic interpolation replaces Euclidean linear blend.
            # Linear interpolation p = (1-a)*p_old + a*p_new can produce points
            # outside the Poincaré ball, violating manifold structure.
            # Instead, when spike fires, use exp_{p_old}(alpha * log_{p_old}(p_new))
            # which stays on the manifold by construction.
            spike_alpha = torch.sigmoid(self.spike_alpha).clamp(min=0.01, max=0.99)

            if self.ablation == "euclidean":
                # Euclidean ablation: simple linear blend is fine for flat space
                p_hyp = (1 - spike_alpha * spike_mask) * self.manifold.hyp_projx(p_hyp) + spike_alpha * spike_mask * self.manifold.hyp_projx(p_new_hyp)
                p_hyp = self.manifold.hyp_projx(p_hyp)
            else:
                # Geodesic interpolation for hyperbolic component
                # For nodes that spiked: move along geodesic from p_hyp toward p_new_hyp
                # For nodes that didn't spike: keep current state (leak only)
                v_geo = self.manifold.hyp.logmap(
                    self.manifold.hyp_projx(p_new_hyp), self.manifold.hyp_projx(p_hyp)
                )  # log_{p_old}(p_new)
                p_hyp_spike = self.manifold.hyp.expmap(
                    spike_alpha * v_geo, self.manifold.hyp_projx(p_hyp)
                )
                p_hyp_spike = self.manifold.hyp_projx(p_hyp_spike)

                # Blend: spiking nodes use geodesic update, non-spiking keep p_hyp
                p_hyp = spike_mask * p_hyp_spike + (1 - spike_mask) * self.manifold.hyp_projx(p_hyp)
                p_hyp = self.manifold.hyp_projx(p_hyp)

            # Euclidean component: linear blend is correct (flat space)
            p_euc = (1 - spike_alpha * spike_mask) * p_euc + spike_alpha * spike_mask * p_new_euc

            # Update p_bar via exponential moving average
            with torch.no_grad():
                p_bar_hyp = self.p_bar_decay * p_bar_hyp + (1 - self.p_bar_decay) * p_hyp.detach()
                p_bar_hyp = self.manifold.hyp_projx(p_bar_hyp)
                p_bar_euc = self.p_bar_decay * p_bar_euc + (1 - self.p_bar_decay) * p_euc.detach()

            # Build spike output: broadcast spike across dimensions
            spike_full = spike_mask.expand_as(I_t)  # (N, d_total)
            spike_list.append(spike_full)

        # Persist membrane potential for next forward call
        self._u_mag = u_mag.detach().clone()

        # Materialise integer spike counts ONCE here (single host sync per
        # forward) instead of per step. This is the only .item() for spike
        # counting and does not serialise the CUDA pipeline.
        if self._spike_count_acc is not None:
            if getattr(self, "_suppress_host_sync", False):
                self._spike_count_acc_captured = self._spike_count_acc
            else:
                self._spike_count = int(self._spike_count_acc.item())
        if self._episode_spike_count_acc is not None:
            if getattr(self, "_suppress_host_sync", False):
                self._episode_spike_count_acc_captured = self._episode_spike_count_acc
            else:
                self._episode_spike_count = int(self._episode_spike_count_acc.item())

        # Compute and store spike rate
        self._total_neurons = N * T
        self._spike_rate = self._spike_count / max(self._total_neurons, 1)

        spikes = torch.stack(spike_list, dim=1)  # (N, T, d_total)
        return spikes, (p_hyp, p_euc)

    def reset_membrane(self):
        """Reset the persistent membrane potential to zero."""
        self._u_mag = None

    @property
    def spike_count_tensor(self) -> Optional[torch.Tensor]:
        """GPU tensor holding the spike sum for the latest LIF call (no sync).

        Used by STMP to accumulate total spikes across time steps without a
        per-step host synchronisation. Returns None before the first call.
        """
        return self._spike_count_acc

    @property
    def spike_count(self) -> int:
        """Return the total number of spikes from the last forward pass.

        Falls back to the lazily-materialised integer; if read mid-forward
        returns the current GPU-accumulated value (one sync on first read).
        """
        if self._spike_count_acc is not None:
            return int(self._spike_count_acc.item())
        return self._spike_count

    def reset_episode_stats(self):
        """Reset episode-level spike accumulators.

        Called by STMP at the start of each forward pass so that the
        episode rate reflects exactly one encode pass (T time steps).
        """
        self._episode_surrogate_sum = torch.zeros((), device=self.u_th_learnable.device)
        self._episode_total_neurons = 0
        self._episode_spike_count = 0
        self._episode_spike_count_acc = None  # GPU tensor accumulator

    def spike_rate(self) -> float:
        """True mean empirical firing rate over the last episode (all T steps)."""
        if self._episode_total_neurons == 0:
            return 0.0
        return self._episode_spike_count / self._episode_total_neurons

    def spike_rate_surrogate(self) -> torch.Tensor:
        """Differentiable surrogate spike rate over the LAST episode (all T steps).

        Aggregates the surrogate spike outputs across every LIF call within one
        encode pass, so this returns the true differentiable mean firing rate
        over the full T-step episode (not just the last step). L_sr should use
        this. Fix for the measurement artifact where the old per-call rate only
        reflected the last time step's phase.
        """
        if self._episode_total_neurons == 0:
            return torch.zeros((), device=self.u_th_learnable.device)
        return self._episode_surrogate_sum / self._episode_total_neurons

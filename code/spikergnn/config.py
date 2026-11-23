"""
SpikeRPGNN configuration: extends MSG baseline ExperimentConfig.

Adds hyperparameters for the product manifold, LIF neuron dynamics,
spiking thresholds, gate initialization, and ablation control.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

import sys
import os

# Ensure msg_baseline is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from msg_baseline.config import ExperimentConfig, DATASET_CONFIGS, ALL_DATASETS


@dataclass
class SpikeRPGNNConfig(ExperimentConfig):
    """Configuration for SpikeRPGNN experiments.

    Inherits all fields from msg_baseline.config.ExperimentConfig and adds
    SpikeRPGNN-specific hyperparameters for the product manifold, LIF neuron,
    gate fusion, and ablation variants.

    Attributes:
        hidden_dim: Total representation dimension (overridden to 128 per paper §6.1).
        d_h: Hyperbolic component dimension (Poincaré ball).
        d_e: Euclidean component dimension.
        learnable_c: Whether curvature c is learnable (curvature learning curve).
        c_init: Initial curvature c (if learnable_c=True).
        beta_leak: LIF leak decay factor (learnable initial value).
        gamma_input: LIF input scaling factor (learnable initial value).
        alpha_decay: Membrane potential decay constant.
        u_threshold: Spike firing threshold.
        delta_reset: Refractory reset amount.
        gate_init: Initial value for learnable gate fusion parameter lambda.
        ablation: Ablation variant name (controls model construction).
        use_dvm: Whether to use Differentiation-via-Manifold (truncated BPTT).
        backbone: GNN backbone type ("gin" or "gcn").
    """

    # ── Override baseline defaults ──────────────────────────────
    hidden_dim: int = 128           # Paper §6.1: total dim = 128
    backbone: str = "gin"           # Paper §6.1: GIN backbone

    # ── Product manifold dimensions ──────────────────────────────
    d_h: int = 32                   # Hyperbolic dimension (Poincaré ball)
    d_e: int = 96                   # Euclidean dimension (128 - 32)
    learnable_c: bool = False        # Whether curvature c is learnable 
    c_init: float = 1.0            # Initial curvature c (if learnable_c=True)

    # ── LIF neuron dynamics ──────────────────────────────────
    beta_leak: float = 0.3          # Leak decay (initial for nn.Parameter)
    gamma_input: float = 0.5        # Input scaling (initial for nn.Parameter)
    alpha_decay: float = 0.9        # Membrane potential decay
    u_threshold: float = 1.0        # Spike threshold
    delta_reset: float = 1.0        # Refractory reset amount

    # ── Gate fusion ───────────────────────────────────────
    gate_init: float = 0.5          # Initial lambda for d = lambda*d_H + (1-lambda)*d_E

    # ── Node preprocessing MLP (P5 encoder repair) ──────────
    node_pre_dim: int = 0           # If >0: lift raw in_dim -> node_pre_dim via 2-layer MLP
                                    # BEFORE MSPE projection. Targets ENZYMES' 3-dim feature
                                    # poverty. 0 = disabled (keeps original behaviour).

    # ── Spike rate regularization ──────────────────────────────
    lambda_sr: float = 0.5            # Spike rate regularization coefficient (Eq. spike_rate_reg)
    r_target: float = 0.4            # Target spike rate r* (Eq. spike_rate_reg)
    surrogate_type: str = "atan"      # Surrogate gradient: "atan" (recommended) or "sigmoid"

    # ── Membrane noise ──────────────────────────────────────────
    # Std of the Gaussian noise added to the membrane potential at every
    # integration step (training AND evaluation). The default 0.05 is the
    # reported configuration: it is what makes the population-averaged firing
    # rate a smooth, monotonic function of the threshold, and therefore what
    # gives L_sr its control authority. Setting it to 0 disables the noise and
    # is used by the sigma=0 control experiment that isolates the contribution
    # of the noise to controllability.
    noise_std: float = 0.05

    # ── Ablation control ───────────────────────────────────
    ablation: str = "full"          # One of: full, no_spikes, euclidean, no_frechet, no_dvm, no_meta

    # ── DvM control ───────────────────────────────────────
    use_dvm: bool = True            # Use Differentiation-via-Manifold

    # ── Gradient checkpointing (path-2 memory fix) ─────────
    # When True (and model.training), each STMP time-step's RGC forward stack
    # is wrapped in torch.utils.checkpoint so its internal saved tensors (GIN
    # MLP activations + manifold exp/log intermediates) are DISCARDED during
    # the forward pass and recomputed in backward. This is the only mechanism
    # that lowers the FORWARD-peak memory: truncated-BPTT (dv_m_window) only
    # cuts the gradient chain (saves backward compute) but cannot free the
    # per-step saved tensors that all sit live simultaneously during the T-loop
    # forward, so it never reduces the forward high-water mark. Checkpointing
    # the RGC stack drops ~16-24 GiB of retained tensors, bringing the 169k
    # full-batch step (T=8, d_e=96) from ~33 GiB down to fit a 22 GB device.
    # RGC-only (not the LIF) is deliberate: the LIF injects cuRNG membrane
    # noise that would differ between the forward and the checkpoint recompute,
    # corrupting the gradient; RGC is a deterministic pure function of its
    # inputs, so recompute is exact.
    checkpoint_rgc: bool = False    # Gradient-checkpoint the per-step RGC stack

    # ── Override training defaults ─────────────────────────────
    epochs: int = 50                # Paper Sec. 6.1: unified 50-epoch protocol

    def __post_init__(self):
        """Validate configuration consistency."""
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        assert self.d_h + self.d_e == self.hidden_dim, (
            f"d_h({self.d_h}) + d_e({self.d_e}) must equal hidden_dim({self.hidden_dim})"
        )
        assert self.ablation in {
            "full", "no_spikes", "euclidean", "no_frechet", "no_dvm", "no_meta",
            "no_stmp",
        }, f"Unknown ablation variant: {self.ablation}"


def get_spikergnn_config(quick_test: bool = False, **overrides) -> SpikeRPGNNConfig:
    """Return SpikeRPGNN config with optional quick-test overrides.

    Args:
        quick_test: If True, reduce epochs/runs for fast debugging.
        **overrides: Additional field overrides passed to the dataclass.

    Returns:
        Fully initialized SpikeRPGNNConfig.
    """
    cfg = SpikeRPGNNConfig(**overrides)
    if quick_test:
        cfg.quick_test = True
        cfg.epochs = 3
        cfg.n_runs = 1
        cfg.train_episodes = 50
        cfg.episodes_per_epoch = 10
        cfg.test_episodes = 20
    return cfg

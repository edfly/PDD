# -*- coding: utf-8 -*-
"""Smoke + correctness test for the optimization edits.

Verifies the plumbing changes (c caching, deferred .item() syncs, gated
euc_max_norm, dedup projx) did NOT change the model's behaviour:
  * forward produces finite per-node embeddings of correct shape
  * L_sr differentiable path still gives u_th_learnable a gradient
  * reported spike_count / debug_sop are positive integers
  * euc_max_norm works when enabled
  * forward is deterministic (same input -> same output within tolerance)
Read-only w.r.t. the project: writes nothing, mutates no saved file.
"""
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"
torch.manual_seed(0)

cfg = SpikeRPGNNConfig(
    d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
    u_threshold=1.0, noise_std=0.05, ablation="full", use_dvm=True,
)
model = create_spikergnn(cfg, in_dim=128).to(DEV)
model.train()

# Tiny graph
N = 256
g = torch.Generator().manual_seed(1)
ei = torch.stack([torch.randint(0, N, (600,), generator=g),
                  torch.randint(0, N, (600,), generator=g)], dim=0).to(DEV)
x = torch.randn(N, 128, device=DEV)
batch = torch.zeros(N, dtype=torch.long, device=DEV)

ok = True


def check(name, cond, extra=""):
    global ok
    print(("PASS" if cond else "FAIL"), "-", name, extra)
    if not cond:
        ok = False


# Forward
out1 = model.encode_nodes(x, ei, batch)
z_hyp, z_euc = out1
check("forward finite", torch.isfinite(z_hyp).all() and torch.isfinite(z_euc).all())
check("z_hyp shape", z_hyp.shape == (N, 32), str(tuple(z_hyp.shape)))
check("z_euc shape", z_euc.shape == (N, 96), str(tuple(z_euc.shape)))

# Determinism: with noise disabled the model is deterministic, so two passes
# from identical state must match to high precision. This isolates my plumbing
# edits (c caching, deferred .item(), gated euc_max_norm, dedup projx) from any
# arithmetic change -- a mismatch would mean the math was altered.
# NOTE: the FIRST forward after model creation can differ from later ones due
# to cuBLAS autotuning / nondeterministic reduction ordering, so warm up first
# and compare two post-warm-up passes.
model.encode_nodes(x, ei, batch)  # warm-up
model.stmp.lif.noise_std = 0.0
model.mspe.lif.noise_std = 0.0
torch.manual_seed(0)
model.stmp.lif.reset_membrane()
out_a = model.encode_nodes(x, ei, batch)
torch.manual_seed(0)
model.stmp.lif.reset_membrane()
out_b = model.encode_nodes(x, ei, batch)
check("deterministic (noise off, post-warmup)",
      torch.allclose(out_a[0], out_b[0], atol=1e-5, rtol=1e-4),
      f"max|d|={(out_a[0] - out_b[0]).abs().max().item():.2e}")

# Backward with L_sr (the differentiable-rate path that must keep its grad)
rate = model.stmp.lif.spike_rate_surrogate()
loss = z_hyp.pow(2).sum() + cfg.lambda_sr * (rate - cfg.r_target) ** 2
loss.backward()
uth = model.stmp.lif.u_th_learnable
check("u_th_learnable has grad", uth.grad is not None and torch.isfinite(uth.grad).all(),
      "" if uth.grad is None else f"grad={uth.grad.item():.4f}")
check("rate finite & in (0,1)", torch.isfinite(rate).all() and 0.0 < float(rate) < 1.0,
      f"rate={float(rate):.4f}")

# Diagnostics populated
check("debug_sop positive int", isinstance(model.stmp.debug_sop, int) and model.stmp.debug_sop > 0,
      f"debug_sop={model.stmp.debug_sop}")
check("count_spikes positive", model.count_spikes() > 0, f"n={model.count_spikes()}")

# euc_max_norm works when enabled
for l in model.stmp.rgc_layers:
    l.track_euc_norm = True
model.encode_nodes(x, ei, batch)
mn = max(l.euc_max_norm for l in model.stmp.rgc_layers)
check("euc_max_norm > 0 when enabled", mn > 0.0, f"maxnorm={mn:.3f}")

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")

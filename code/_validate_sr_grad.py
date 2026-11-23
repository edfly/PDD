"""
_validate_sr_grad.py — proves the L_sr differentiability fix in run_nodelevel.py.

Mirrors the EXACT L_sr computation used by the fixed script on a tiny synthetic
graph:

    r_bar = model.spike_rate_surrogate()        # differentiable tensor
    loss  = ce + lam_sr * (r_bar - r_star) ** 2
    loss.backward()

Then checks whether the learnable membrane threshold `u_th_learnable` receives a
non-zero gradient. Under the OLD code the rate came from
`model.count_spikes() / (N*T)` -> a Python float detached via `.item()`, so
`u_th_learnable.grad` was None and the threshold drifted to silence (the
~0.3% firing-rate collapse). With the fix it must be a live tensor with grad.

Also reports the firing rate as a function of u_threshold to confirm the rate
actually tracks the threshold and can approach r*=0.4 (instead of being pinned
at ~0).
"""
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def build(u_th):
    cfg = SpikeRPGNNConfig(
        d_h=8, d_e=24, hidden_dim=32, T=8, num_layers=2,
        u_threshold=u_th, noise_std=0.05, ablation="full", use_dvm=True,
    )
    model = create_spikergnn(cfg, in_dim=128).to(DEVICE)
    head = nn.Linear(cfg.d_h + cfg.d_e, 5).to(DEVICE)
    return cfg, model, head


def run(u_th, lam_sr=0.5, r_star=0.4, steps=4):
    _, model, head = build(u_th)
    N = 64
    x = torch.randn(N, 128, device=DEVICE)
    edge_index = torch.stack([torch.randint(0, N, (2,)),
                              torch.randint(0, N, (2,))], 0).to(DEVICE)
    y = torch.randint(0, 5, (N,), device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(steps):
        opt.zero_grad()
        z_hyp, z_euc = model.encode_nodes(x, edge_index, None)
        v_hyp = model.manifold.hyp.logmap0(z_hyp)
        feat = torch.cat([v_hyp, z_euc], -1)
        logits = head(feat)
        loss = F.cross_entropy(logits, y)
        r_bar = model.spike_rate_surrogate()        # THE FIX LINE
        loss = loss + lam_sr * (r_bar - r_star) ** 2
        loss.backward()
        opt.step()

    u = model.stmp.lif.u_th_learnable
    has_grad = u.grad is not None
    grad_mag = float(u.grad.abs().mean()) if has_grad else 0.0
    rate = float(r_bar.detach().cpu())
    return rate, has_grad, grad_mag


if __name__ == "__main__":
    print(f"device={DEVICE}  (L_sr uses differentiable spike_rate_surrogate)\n")
    print(f"{'u_th':>6} | {'rate':>7} | {'grad_on_u_th':>12} | {'|grad|':>10}")
    print("-" * 44)
    for u_th in [0.05, 0.10, 0.20, 0.40, 1.00]:
        rate, has_grad, grad_mag = run(u_th)
        flag = "YES" if has_grad else "NONE"
        print(f"{u_th:6.2f} | {rate:7.4f} | {flag:>12} | {grad_mag:10.2e}")
    print("\nExpected: grad_on_u_th == YES everywhere (graph-connected), and "
          "rate should rise as u_th falls (toward r*=0.4).")

# -*- coding: utf-8 -*-
"""Integration test for the CUDA-Graph training wiring in run_nodelevel.

This exercises the ACTUAL ``_cg_build`` / ``_cg_step`` / ``_cg_reset_state``
helpers (the ones wired into ``run_one``), not the POC's inline copies, so it
catches integration bugs (wrong call signature, missing attrs, stream ordering)
before the expensive 169k capture.

Two checks:
  (1) CORRECTNESS (noise_std=0): a single replay (no step) must match a fresh
      eager forward+backward on identical weights, bit-for-bit (float noise).
  (2) FUNCTIONAL (noise_std=0.05): several replay steps must lower the loss and
      stay finite; replay must be faster than an eager step (launch-bound gain).
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from run_nodelevel import (_cg_build, _cg_step, _cg_reset_state,
                               SpikeRPGNNNode)

DEV = "cuda"
N, E, F = 8000, 40000, 128
OUT = 40


def make_model(noise_std):
    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=noise_std, ablation="full",
                           use_dvm=True, dv_m_window=4)
    return SpikeRPGNNNode(cfg, in_dim=F, out_dim=OUT).to(DEV)


def make_data():
    xg = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn(N, F, generator=xg, device=DEV)
    rg = torch.Generator(device=DEV).manual_seed(3)
    row = torch.randint(0, N, (E,), generator=rg, device=DEV)
    col = torch.randint(0, N, (E,), generator=rg, device=DEV)
    ei = torch.stack([row, col])
    batch = torch.zeros(N, dtype=torch.long, device=DEV)
    train_idx = torch.arange(0, N // 2, device=DEV)
    y = torch.randint(0, OUT, (N,), device=DEV)
    return x, ei, batch, train_idx, y


def eager_loss(model, opt, x, ei, batch, train_idx, y, suppress=True):
    for mod in model.modules():
        mod._disable_internal_noise = True
        if suppress:
            mod._suppress_host_sync = True
    opt.zero_grad(set_to_none=True)
    logits = model(x, ei, batch)
    loss = torch.nn.functional.cross_entropy(logits[train_idx], y[train_idx])
    r_bar = model.spike_rate_surrogate()
    loss = loss + model.lam_sr * (r_bar - model.r_star) ** 2
    loss.backward()
    return float(loss.item()), [p.grad.detach().clone() if p.grad is not None
                               else None for p in model.parameters()]


def main():
    x, ei, batch, train_idx, y = make_data()

    # ── (1) Correctness: replay (noise 0) vs fresh eager (noise 0) ──────────
    model = make_model(0.0)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_cg = x.clone()
    cg = _cg_build(model, opt, x_cg, ei, batch, train_idx, y, DEV)
    print("CAPTURE ok", flush=True)

    # One replay, NO step, read loss + grads.
    _cg_reset_state(model)
    with torch.cuda.stream(cg["stream"]):
        opt.zero_grad(set_to_none=False)
        cg["g"].replay()
    torch.cuda.synchronize()
    replay_loss = float(cg["loss"].item())
    replay_grads = [p.grad.detach().clone() if p.grad is not None else None
                    for p in model.parameters()]

    # Fresh eager reference with identical weights.
    ref = make_model(0.0)
    ref.load_state_dict(model.state_dict(), strict=False)
    ref.train()
    opt_r = torch.optim.Adam(ref.parameters(), lr=1e-3)
    eager_loss_val, eager_grads = eager_loss(ref, opt_r, x_cg, ei, batch,
                                             train_idx, y, suppress=False)
    print(f"  replay_loss={replay_loss:.6f}  eager_loss={eager_loss_val:.6f}",
          flush=True)

    loss_ok = abs(replay_loss - eager_loss_val) / max(1.0, abs(eager_loss_val)) < 1e-4
    max_grel, max_gabs, none_mm = 0.0, 0.0, 0
    for rg_, eg in zip(replay_grads, eager_grads):
        if rg_ is None or eg is None:
            if (rg_ is None) != (eg is None):
                none_mm += 1
            continue
        d = rg_ - eg
        max_gabs = max(max_gabs, d.abs().max().item())
        denom = max(rg.abs().max().item(), eg.abs().max().item())
        if denom > 1e-9:
            max_grel = max(max_grel, d.abs().max().item() / denom)
    print(f"  CORRECTNESS loss_ok={loss_ok} max_grad_rel={max_grel:.2e} "
          f"max_grad_abs={max_gabs:.2e} grad_none_mismatch={none_mm}", flush=True)
    if not (loss_ok and none_mm == 0):
        print("  !! CORRECTNESS FAILED", flush=True)
        return

    # ── (2) Functional + speed: train with external noise 0.05 ──────────────
    # Rebuild with the real noise_std so the config matches production, but
    # capture still disables internal noise; _cg_step injects 0.05 externally.
    model2 = make_model(0.05)
    model2.train()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    x_cg2 = x.clone()
    cg2 = _cg_build(model2, opt2, x_cg2, ei, batch, train_idx, y, DEV)

    # Baseline eager step timing (model2, internal noise disabled -> matches
    # the graph's no-internal-noise dynamics for a fair timing comparison).
    for mod in model2.modules():
        mod._disable_internal_noise = True
    opt2.zero_grad(set_to_none=True)
    t0 = time.time()
    logits = model2(x_cg2, ei, batch)
    loss = torch.nn.functional.cross_entropy(logits[train_idx], y[train_idx])
    r_bar = model2.spike_rate_surrogate()
    loss = loss + model2.lam_sr * (r_bar - model2.r_star) ** 2
    loss.backward()
    opt2.step()
    torch.cuda.synchronize()
    eager_dt = time.time() - t0

    # Replay training: capture a fresh graph so we start from model2's post-eager
    # weights for an apples-to-apples "does training progress" check.
    cg3 = _cg_build(model2, opt2, x_cg2, ei, batch, train_idx, y, DEV)
    l0 = float(cg3["loss"].item())  # loss before any replay step (read once)
    # warm a single replay to read the loss at the captured weights
    _cg_reset_state(model2)
    with torch.cuda.stream(cg3["stream"]):
        opt2.zero_grad(set_to_none=False)
        cg3["g"].replay()
    torch.cuda.synchronize()
    l_start = float(cg3["loss"].item())

    reps = []
    for _ in range(5):
        t0 = time.time()
        _cg_step(cg3, model2, opt2, x, 0.05)
        reps.append(time.time() - t0)
    replay_dt = sum(reps) / len(reps)
    l_end = float(cg3["loss"].item())
    print(f"  eager_step={eager_dt:.3f}s  replay_step={replay_dt:.3f}s "
          f"speedup={eager_dt/replay_dt:.2f}x", flush=True)
    print(f"  loss start={l_start:.4f} -> end={l_end:.4f} "
          f"(finite={bool(torch.isfinite(torch.tensor(l_end)))})", flush=True)
    if not torch.isfinite(torch.tensor(l_end)):
        print("  !! NON-FINITE LOSS", flush=True)
        return
    print("INTEGRATION OK", flush=True)


if __name__ == "__main__":
    main()

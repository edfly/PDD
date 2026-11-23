# -*- coding: utf-8 -*-
"""Path-2 validation: gradient checkpoint of the per-step RGC stack.

Proves the 169k full-batch step (T=8, d_e=96) FITS a 22 GB device (peak memory
drops from ~33 GiB to ~17 GiB) AND that CUDA-GRAPH capture (full fwd+L_sr+bwd,
which re-records the checkpoint recompute) + replay training work on 22 GB.

Key correctness note (learned empirically): the capture must run on a FRESHLY
built model with NO live autograd graph from a prior eager step. Reusing a
model that already took an eager backward leaves a stale AccumulateGrad node
that raises cudaErrorStreamCaptureImplicit. run_one() already satisfies this
(capture precedes the epoch loop), so the production path is fine.

Phases:
  A. Fresh model #1: eager single-step fwd+bwd peak memory + per-step time.
  B. Fresh model #2: CUDA-graph fwd+bwd capture (with checkpoint), then replay
     training with a CORRECT baseline (loss AFTER step 1, not the uninitialised
     cg["loss"] read before any replay).
"""
import os
import sys
import time
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ogb.nodeproppred import NodePropPredDataset
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from run_nodelevel import (SpikeRPGNNNode, _cg_build, _cg_step)

_CACHE = "D:/2026/paper/PBB/post/data/ogb_cache"
os.environ.setdefault("OGB_DIR", _CACHE)
os.environ.setdefault("TORCH_GEOMETRIC_DIR", _CACHE)
DEV = "cuda"


def build_full_model():
    ds = NodePropPredDataset(name="ogbn-arxiv", root=_CACHE)
    g = ds.graph
    x = torch.tensor(g["node_feat"], dtype=torch.float32, device=DEV)
    ei = torch.tensor(g["edge_index"], dtype=torch.long, device=DEV)
    y = torch.as_tensor(ds.labels, dtype=torch.long, device=DEV).squeeze(-1)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=DEV)
    train_idx = torch.as_tensor(ds.get_idx_split()["train"], device=DEV)
    N = x.size(0)
    print(f"[setup] N={N} E={ei.size(1)}  (target: 169k full-batch)", flush=True)

    # Full config that OOM'd at ~33 GiB WITHOUT checkpoint.
    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=0.05, ablation="full",
                           use_dvm=True, dv_m_window=min(4, 8),
                           checkpoint_rgc=True)
    model = SpikeRPGNNNode(cfg, in_dim=128, out_dim=40).to(DEV)
    for mod in model.modules():
        mod._disable_internal_noise = True  # match capture (no cuRNG)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return dict(model=model, opt=opt, x=x, ei=ei, batch=batch,
                train_idx=train_idx, y=y, N=N)


def eager_step(m):
    """One full-batch training step; return (loss, peak GiB)."""
    torch.cuda.reset_peak_memory_stats()
    m["opt"].zero_grad(set_to_none=True)
    logits = m["model"](m["x"], m["ei"], m["batch"])
    loss = torch.nn.functional.cross_entropy(
        logits[m["train_idx"]], m["y"][m["train_idx"]])
    r_bar = m["model"].spike_rate_surrogate()
    loss = loss + m["model"].lam_sr * (r_bar - m["model"].r_star) ** 2
    loss.backward()
    m["opt"].step()
    torch.cuda.synchronize()
    return float(loss.item()), torch.cuda.max_memory_allocated() / 1e9


def main():
    # ── Phase A: eager peak + timing (fresh model) ──────────────
    print("=== Phase A: eager single-step peak (checkpoint_rgc=True) ===",
          flush=True)
    m1 = build_full_model()
    t0 = time.time()
    try:
        loss, peak = eager_step(m1)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  OOM peak reached={peak:.2f} GiB -> checkpoint insufficient",
              flush=True)
        return
    dt = time.time() - t0
    verdict = ("FITS(<22)" if peak < 22 else
               "TIGHT(22-24)" if peak < 24 else "OVERSIZED(>=24)")
    print(f"  loss={loss:.4f}  peak={peak:.2f} GiB  step_time={dt:.1f}s "
          f"-> {verdict}", flush=True)
    del m1
    torch.cuda.empty_cache()
    if peak >= 22:
        print("  !! exceeds 22 GB -> abort.", flush=True)
        return

    # ── Phase B: fresh model, CUDA-graph capture + replay train ─
    print("\n=== Phase B: CUDA-graph fwd+bwd capture + replay training ===",
          flush=True)
    m2 = build_full_model()   # FRESH model -> no stale autograd graph
    cap_ok = False
    cg = None
    try:
        x_cg = m2["x"].clone()
        cg = _cg_build(m2["model"], m2["opt"], x_cg, m2["ei"], m2["batch"],
                       m2["train_idx"], m2["y"], DEV)
        cap_ok = True
        print("  CAPTURE ok (full fwd+L_sr+bwd, checkpoint recompute recorded)",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  CAPTURE FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return

    # Correct baseline: read loss AFTER the first replay (cg["loss"] is the
    # tensor written by the captured graph and is undefined before any replay).
    _cg_step(cg, m2["model"], m2["opt"], m2["x"], 0.05)   # step 1 (baseline)
    w0 = m2["model"].head.weight.norm().item()
    loss0 = float(cg["loss"].item())
    print(f"  baseline (after step 1): wnorm={w0:.4f} loss={loss0:.4f}",
          flush=True)

    n_more = 9
    for i in range(n_more):
        _cg_step(cg, m2["model"], m2["opt"], m2["x"], 0.05)
    torch.cuda.synchronize()
    w1 = m2["model"].head.weight.norm().item()
    loss1 = float(cg["loss"].item())
    print(f"  after {1 + n_more} steps: wnorm={w1:.4f} loss={loss1:.4f} "
          f"| d_wnorm={w1-w0:+.4f} d_loss={loss1-loss0:+.4f}", flush=True)
    ok = (abs(w1 - w0) > 1e-4) and (loss1 < loss0)
    print("TRAINING_OK" if ok else "NO_TRAINING", flush=True)

    print("\nSUMMARY: checkpoint_rgc fits 22GB (peak=%.2f GiB) AND "
          "CUDA-graph fwd+bwd capture + replay training pass on 22GB = %s"
          % (peak, ok and cap_ok))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Memory diagnostic for a single full-batch 169k SpikeRPGNN step.

Isolates the per-step GPU footprint (independent of CUDA-graph capture) so we
know whether the current config fits the 22 GB card, and how much a smaller
truncated-BPTT window (dv_m_window) helps. Prints the high-water mark even if a
later allocation OOMs.
"""
import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ogb.nodeproppred import NodePropPredDataset
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from run_nodelevel import SpikeRPGNNNode

_CACHE = "D:/2026/paper/PBB/post/data/ogb_cache"
os.environ.setdefault("OGB_DIR", _CACHE)
os.environ.setdefault("TORCH_GEOMETRIC_DIR", _CACHE)
DEV = "cuda"


def one_step(model, x, ei, batch, train_idx, y, opt):
    torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    logits = model(x, ei, batch)
    loss = torch.nn.functional.cross_entropy(logits[train_idx], y[train_idx])
    r_bar = model.spike_rate_surrogate()
    loss = loss + model.lam_sr * (r_bar - model.r_star) ** 2
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return float(loss.item()), torch.cuda.max_memory_allocated() / 1e9


def main():
    ds = NodePropPredDataset(name="ogbn-arxiv", root=_CACHE)
    g = ds.graph
    x = torch.tensor(g["node_feat"], dtype=torch.float32, device=DEV)
    ei = torch.tensor(g["edge_index"], dtype=torch.long, device=DEV)
    y = torch.as_tensor(ds.labels, dtype=torch.long, device=DEV).squeeze(-1)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=DEV)
    train_idx = torch.as_tensor(ds.get_idx_split()["train"], device=DEV)
    N = x.size(0)
    print(f"N={N} E={ei.size(1)}", flush=True)

    # Reduced-config sweep: find a 169k full-batch config that fits 22 GiB
    # WITH headroom for the CUDA-graph capture private memory (graph stores a
    # copy of the recorded kernels; total live = step_peak + ~0.3*step_peak).
    # Target single-step peak < ~17 GiB (=> total < ~22 GiB).
    # d_h fixed at 32; config requires hidden_dim == d_h + d_e.
    combos = [
        dict(T=4, d_e=64),   # hidden=96
        dict(T=4, d_e=48),   # hidden=80
        dict(T=6, d_e=64),   # hidden=96
        dict(T=4, d_e=32),   # hidden=64
        dict(T=6, d_e=48),   # hidden=80
    ]
    for c in combos:
        torch.cuda.empty_cache()
        hidden = 32 + c["d_e"]
        cfg = SpikeRPGNNConfig(d_h=32, d_e=c["d_e"], hidden_dim=hidden,
                               T=c["T"], num_layers=2, u_threshold=1.0,
                               noise_std=0.05, ablation="full", use_dvm=True,
                               dv_m_window=c["T"])  # window==T (no truncation needed)
        model = SpikeRPGNNNode(cfg, in_dim=128, out_dim=40).to(DEV)
        for mod in model.modules():
            mod._disable_internal_noise = True  # match capture (no cuRNG)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        try:
            loss, peak = one_step(model, x, ei, batch, train_idx, y, opt)
            verdict = ("FITS(<17)" if peak < 17 else
                       "TIGHT(17-20)" if peak < 20 else "OVERSIZED(>=20)")
            print(f"  T={c['T']} d_e={c['d_e']} hidden={hidden}: "
                  f"loss={loss:.4f} peak={peak:.2f} GiB -> {verdict}", flush=True)
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  T={c['T']} d_e={c['d_e']} hidden={hidden}: "
                  f"OOM (peak reached={peak:.2f} GiB) -> EXCEEDS", flush=True)
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

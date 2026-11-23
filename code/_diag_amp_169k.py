# -*- coding: utf-8 -*-
"""AMP (fp16 autocast) memory diagnostic for a single 169k full-batch step.

If autocast halves activation memory, T=8/d_e=96 (33 GiB fp32) drops to ~16 GiB
and fits 22 GB with headroom for CUDA-graph capture private memory.
"""
import os
import sys

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


def one_step_amp(model, x, ei, batch, train_idx, y, opt):
    torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.float16):
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
    print(f"N={x.size(0)} E={ei.size(1)}", flush=True)

    for (T, d_e) in [(8, 96), (4, 64), (8, 64)]:
        hidden = 32 + d_e
        torch.cuda.empty_cache()
        cfg = SpikeRPGNNConfig(d_h=32, d_e=d_e, hidden_dim=hidden, T=T,
                               num_layers=2, u_threshold=1.0, noise_std=0.05,
                               ablation="full", use_dvm=True, dv_m_window=T)
        model = SpikeRPGNNNode(cfg, in_dim=128, out_dim=40).to(DEV)
        for mod in model.modules():
            mod._disable_internal_noise = True
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        try:
            loss, peak = one_step_amp(model, x, ei, batch, train_idx, y, opt)
            print(f"  AMP T={T} d_e={d_e} hidden={hidden}: loss={loss:.4f} "
                  f"peak={peak:.2f} GiB "
                  f"{'FITS(<18)' if peak < 18 else 'TIGHT(18-20)' if peak < 20 else 'OVERSIZED'}",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  AMP T={T} d_e={d_e} hidden={hidden}: OOM (peak={peak:.2f} GiB)",
                  flush=True)
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

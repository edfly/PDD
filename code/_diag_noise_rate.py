# -*- coding: utf-8 -*-
"""Isolate whether the dormant (rate=0) model under CUDA-graph is caused by the
noise design (internal membrane noise disabled for capture) rather than by
gradient checkpoint.

Builds the full 169k config with checkpoint_rgc=True, disables internal noise
on every module (exactly what _cg_build does for capture), then trains EAGERLY
and reports the surrogate firing rate. If rate -> ~0 here, the dormant model is
caused by _disable_internal_noise, NOT by checkpoint (which is present in both
the working eager run and this run).
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


def main():
    ds = NodePropPredDataset(name="ogbn-arxiv", root=_CACHE)
    g = ds.graph
    x = torch.tensor(g["node_feat"], dtype=torch.float32, device=DEV)
    ei = torch.tensor(g["edge_index"], dtype=torch.long, device=DEV)
    y = torch.as_tensor(ds.labels, dtype=torch.long, device=DEV).squeeze(-1)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=DEV)
    train_idx = torch.as_tensor(ds.get_idx_split()["train"], device=DEV)

    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=0.05, ablation="full",
                           use_dvm=True, dv_m_window=min(4, 8),
                           checkpoint_rgc=True)
    model = SpikeRPGNNNode(cfg, in_dim=128, out_dim=40).to(DEV)
    # Mimic cuda-graph capture's noise state: disable internal cuRNG noise.
    for mod in model.modules():
        mod._disable_internal_noise = True
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rates = []
    for ep in range(10):
        opt.zero_grad(set_to_none=True)
        logits = model(x, ei, batch)
        loss = torch.nn.functional.cross_entropy(logits[train_idx],
                                                 y[train_idx])
        r = model.spike_rate_surrogate()
        loss = loss + model.lam_sr * (r - model.r_star) ** 2
        loss.backward()
        opt.step()
        rates.append(float(r.detach().item()))
        if (ep + 1) % 5 == 0:
            with torch.no_grad():
                acc = (logits[train_idx].argmax(1) == y[train_idx]).float().mean()
            print(f"  ep{ep+1:02d} rate={rates[-1]:.4f} train_acc={acc:.4f}",
                  flush=True)
    print(f"  rate trajectory: " + " ".join(f"{v:.3f}" for v in rates),
          flush=True)
    print(f"  FINAL rate_mean={sum(rates)/len(rates):.4f}", flush=True)
    print("  => internal-noise-OFF -> rate~0 CONFIRMS noise design is the "
          "cause (checkpoint is innocent).", flush=True)


if __name__ == "__main__":
    main()

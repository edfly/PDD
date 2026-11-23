# -*- coding: utf-8 -*-
"""Does the 169k CUDA-graph replay actually train (move weights + lower loss)?

Rules out the "frozen accuracy" being a CUDA-graph replay bug vs. just a
dormant/low-capacity model.
"""
import os
import sys

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


def main():
    ds = NodePropPredDataset(name="ogbn-arxiv", root=_CACHE)
    g = ds.graph
    x = torch.tensor(g["node_feat"], dtype=torch.float32, device=DEV)
    ei = torch.tensor(g["edge_index"], dtype=torch.long, device=DEV)
    y = torch.as_tensor(ds.labels, dtype=torch.long, device=DEV).squeeze(-1)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=DEV)
    train_idx = torch.as_tensor(ds.get_idx_split()["train"], device=DEV)

    cfg = SpikeRPGNNConfig(d_h=32, d_e=32, hidden_dim=64, T=4, num_layers=2,
                          u_threshold=0.5, noise_std=0.05, ablation="full",
                          use_dvm=True, dv_m_window=4)
    model = SpikeRPGNNNode(cfg, in_dim=128, out_dim=40).to(DEV)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_cg = x.clone()
    cg = _cg_build(model, opt, x_cg, ei, batch, train_idx, y, DEV)
    print("CAPTURE ok", flush=True)

    w0 = model.head.weight.norm().item()
    loss0 = float(cg["loss"].item())
    # snapshot a few logits to see if predictions move
    with torch.no_grad():
        logits0 = model(x_cg, ei, batch)[:, 0].detach().clone()
    print(f"  before: wnorm={w0:.4f} loss={loss0:.4f}", flush=True)

    for i in range(10):
        _cg_step(cg, model, opt, x, 0.05)
    w1 = model.head.weight.norm().item()
    loss1 = float(cg["loss"].item())
    with torch.no_grad():
        logits1 = model(x_cg, ei, batch)[:, 0].detach().clone()
    print(f"  after 10 steps: wnorm={w1:.4f} loss={loss1:.4f} "
          f"| d_wnorm={w1-w0:+.4f} d_loss={loss1-loss0:+.4f} "
          f"| logits_maxdiff={ (logits1-logits0).abs().max().item():.4f}",
          flush=True)
    print("TRAINING_OK" if (abs(w1 - w0) > 1e-4 and loss1 < loss0) else "NO_TRAINING",
          flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Single-step 169k timing with window=2 to measure real per-step cost."""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"
N, E, F = 169343, 1_166_243, 128


def make():
    g = torch.Generator().manual_seed(2026)
    row = torch.randint(0, N, (E,), generator=g)
    col = torch.randint(0, N, (E,), generator=g)
    return Data(x=torch.randn(N, F, generator=g),
                edge_index=torch.stack([row, col])).to(DEV)


def main():
    data = make()
    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=0.0, ablation="full",
                           use_dvm=True, dv_m_window=2)
    m = create_spikergnn(cfg, in_dim=F).to(DEV)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    print("model built", flush=True)

    def step():
        opt.zero_grad(set_to_none=True)
        out = m.encode_nodes(data.x, data.edge_index, None)
        loss = out[0].pow(2).mean()
        loss.backward()
        opt.step()
        return float(loss.item())

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    l = step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"warmup: loss={l:.4f} time={dt:.1f}s peak_mem={peak:.2f} GB", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

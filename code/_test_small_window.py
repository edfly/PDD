# -*- coding: utf-8 -*-
"""Small-graph correctness + timing check for the truncated-BPTT state-detach fix.
Validates that windowing does not break forward/backward and that gradients flow,
using a tiny graph so it runs in seconds.
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

DEV = "cuda"
N, E, F = 8000, 40000, 128


def make():
    g = torch.Generator().manual_seed(7)
    row = torch.randint(0, N, (E,), generator=g)
    col = torch.randint(0, N, (E,), generator=g)
    return Data(x=torch.randn(N, F, generator=g),
                edge_index=torch.stack([row, col])).to(DEV)


def main():
    for window in (8, 4, 2):
        data = make()
        cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                               u_threshold=1.0, noise_std=0.0, ablation="full",
                               use_dvm=True, dv_m_window=window)
        m = create_spikergnn(cfg, in_dim=F).to(DEV)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        print(f"\n--- window={window} ---", flush=True)

        def step():
            opt.zero_grad(set_to_none=True)
            out = m.encode_nodes(data.x, data.edge_index, None)
            loss = out[0].pow(2).mean()
            loss.backward()
            opt.step()
            return float(loss.item())

        t0 = time.time()
        l0 = step()
        torch.cuda.synchronize()
        w = time.time() - t0
        # gradient sanity: u_th_learnable should have a grad
        u_th = [p for n, p in m.named_parameters() if "u_th_learnable" in n][0]
        print(f"  warmup loss={l0:.4f}  time={w:.2f}s  u_th.grad={'None' if u_th.grad is None else 'ok'}",
              flush=True)
        # one more timed step
        t0 = time.time()
        l1 = step()
        torch.cuda.synchronize()
        print(f"  step2  loss={l1:.4f}  time={time.time()-t0:.2f}s", flush=True)
        del m, data
        torch.cuda.empty_cache()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

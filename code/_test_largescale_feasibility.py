# -*- coding: utf-8 -*-
"""169k ogbn-arxiv-like full-batch feasibility test (forward + backward).

Validates the STMP accumulator-windowing memory fix: with use_dvm and
dv_m_window < T, only the last `dv_m_window` steps hold live autograd graphs,
so a 169k forward+backward should fit a 22 GB device instead of OOM'ing at 31 GB.

Sweeps dv_m_window in {2, 4, 8} and reports peak CUDA memory + per-step time.
"""
import os
import sys
import time
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"

N_NODES = 169343
N_EDGES = 1_166_243
FEAT = 128


def make_arxiv_like():
    g = torch.Generator().manual_seed(2026)
    row = torch.randint(0, N_NODES, (N_EDGES,), generator=g)
    col = torch.randint(0, N_NODES, (N_EDGES,), generator=g)
    edge_index = torch.stack([row, col], dim=0)
    x = torch.randn(N_NODES, FEAT, generator=g)
    return Data(x=x, edge_index=edge_index).to(DEV)


def run_window(window):
    data = make_arxiv_like()
    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=0.0, ablation="full",
                           use_dvm=True, dv_m_window=window)
    model = create_spikergnn(cfg, in_dim=FEAT).to(DEV)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"\n=== dv_m_window={window} (T=8) ===", flush=True)
    print(f"  model params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    def step():
        opt.zero_grad(set_to_none=True)
        out = model.encode_nodes(data.x, data.edge_index, None)
        loss = out[0].pow(2).mean()
        loss.backward()
        opt.step()
        return float(loss.item())

    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _ = step()  # warmup
        torch.cuda.synchronize()
        warm = time.time() - t0
        warm_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  warmup step: {warm:.1f}s  peak_mem={warm_peak:.2f} GB", flush=True)

        times = []
        for i in range(2):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            loss = step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            times.append(dt)
            print(f"  step {i+1}: loss={loss:.4f}  fwd+bwd+opt={dt:.1f}s  peak_mem={peak:.2f} GB", flush=True)
        print(f"  -> mean step={sum(times)/len(times):.1f}s", flush=True)
        print("  RESULT: FITS" if warm_peak < 21.0 else "  RESULT: TIGHT/OOM-RISK", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM: {str(e).splitlines()[0]}", flush=True)
        print("  RESULT: OOM (does NOT fit)", flush=True)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    for w in (4, 2):
        run_window(w)
    print("\nDONE", flush=True)

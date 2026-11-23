# -*- coding: utf-8 -*-
"""Test torch.compile as the real fix for the launch-bound ~1s encode.

torch.compile fuses the elementwise manifold/LIF ops and unrolls the T-loop,
cutting kernel launches and Python dispatch overhead (the actual bottleneck:
15.8k launches, 27ms GPU compute, ~1s wall). Zero arithmetic change.

Compares EAGER (current plumbing-fixed code) vs COMPILED at two batch sizes.
"""
import os
import sys
import time
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data, Batch
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"


def make_batch(G, N0, p_edge, feat_dim):
    graphs = []
    for i in range(G):
        n_edges = int(p_edge * N0 * N0 / 2)
        graphs.append(Data(
            x=torch.randn(N0, feat_dim, generator=torch.Generator().manual_seed(i)),
            edge_index=torch.stack([
                torch.randint(0, N0, (n_edges,), generator=torch.Generator().manual_seed(i)),
                torch.randint(0, N0, (n_edges,), generator=torch.Generator().manual_seed(i)),
            ]),
        ))
    return Batch.from_data_list(graphs)


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def build(use_dvm):
    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=0.05, ablation="full", use_dvm=use_dvm)
    return create_spikergnn(cfg, in_dim=128).to(DEV)


def run(label, model, Gs):
    model.train()
    print(f"\n=== {label} ===")
    for G in Gs:
        b = make_batch(G, 30, 0.35, 128).to(DEV)
        try:
            t = timeit(lambda: model.encode_nodes(b.x, b.edge_index, b.batch))
            print(f"  G={G:4d}  nodes={b.x.shape[0]:6d}  encode_nodes fwd = {t:7.1f} ms")
        except Exception as e:
            print(f"  G={G:4d}  FAILED: {e}")
            traceback.print_exc()
            break


print("torch version:", torch.__version__, "CUDA:", torch.version.cuda)
Gs = (16, 256, 512)

# Eager baseline
m_eager = build(use_dvm=True)
run("EAGER (plumbing-fixed)", m_eager, Gs)

# Compiled (use_dvm=True)
try:
    m_c = build(use_dvm=True)
    m_comp = torch.compile(m_c, mode="default", dynamic=False)
    run("COMPILED use_dvm=True", m_comp, Gs)
except Exception as e:
    print("COMPILE use_dvm=True FAILED:", e)
    traceback.print_exc()

# Compiled (use_dvm=False) as fallback if DvM graph-breaks
try:
    m_c2 = build(use_dvm=False)
    m_comp2 = torch.compile(m_c2, mode="default", dynamic=False)
    run("COMPILED use_dvm=False", m_comp2, Gs)
except Exception as e:
    print("COMPILE use_dvm=False FAILED:", e)
    traceback.print_exc()

print("\nDONE")

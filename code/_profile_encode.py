# -*- coding: utf-8 -*-
"""Diagnostic only: where does encode()'s batch-independent ~1 s go?

Hypothesis under test
--------------------
encode() costs ~1.0 s whether the batch holds 32 graphs or 1024. Work that
does not scale with batch is either (a) fixed per-call overhead such as
python loops and host-device syncs, or (b) something computed on a quantity
that does not depend on N. stmp.py accumulates `debug_sop` with
`int((s_t * degree).sum().item())` inside `for t in range(self.T)`, i.e. one
mandatory GPU->CPU sync per time step -- T=8 syncs per forward, each blocking
until the device drains. This script measures that cost against the total so
the fix is aimed at the real bottleneck instead of a plausible one.

It also times MSPE and STMP separately, because if the sync is NOT dominant
then the loop body itself is and a different fix is needed.

Read-only: patches nothing, writes nothing, mutates no source file.
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _HERE)
os.chdir(_REPO)

import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])

from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"
ds = PygGraphPropPredDataset(name="ogbg-molhiv", root="data")
idx = ds.get_idx_split()["train"]
cfg = SpikeRPGNNConfig(hidden_dim=128, num_layers=3)
model = create_spikergnn(cfg, in_dim=9).to(DEV)


def sync():
    torch.cuda.synchronize()


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    sync()
    t = time.time()
    for _ in range(n):
        fn()
    sync()
    return (time.time() - t) / n


print("=" * 74)
print("encode() breakdown on ogbg-molhiv (device=%s)" % DEV)
print("=" * 74)

for bs in (64, 512):
    loader = DataLoader(ds[idx[:bs]], batch_size=bs)
    b = next(iter(loader)).to(DEV)
    b.x = b.x.float()
    N = b.x.shape[0]

    # 1. total encode
    t_enc = timeit(lambda: model.encode(b.x, b.edge_index, b.batch))

    # 2. MSPE alone
    t_mspe = timeit(lambda: model.mspe(b.x, b.edge_index, b.batch))

    # 3. cost of T syncs: time T bare .item() calls on a real device tensor
    deg = torch.ones(N, device=DEV)
    s_t = torch.ones(N, device=DEV)
    T = cfg.T

    def t_syncs():
        acc = 0
        for _ in range(T):
            acc += int((s_t * deg).sum().item())
        return acc

    t_sync = timeit(t_syncs, n=10)

    # 4. cost of ONE sync, for reference
    def t_one():
        return int((s_t * deg).sum().item())

    t_one_sync = timeit(t_one, n=20)

    # 5. STMP residual (encode minus mspe) -- approximate, since encode also
    #    does pooling; reported only to size the loop.
    print("batch=%4d  N=%6d" % (bs, N))
    print("   encode total      : %6.1f ms" % (t_enc * 1000))
    print("   mspe only         : %6.1f ms" % (t_mspe * 1000))
    print("   stmp (residual)   : %6.1f ms" % ((t_enc - t_mspe) * 1000))
    print("   T=%d .item() syncs: %6.1f ms   (%.0f%% of encode)"
          % (T, t_sync * 1000, 100 * t_sync / t_enc))
    print("   one .item() sync  : %6.2f ms" % (t_one_sync * 1000))
    print()

print("=" * 74)
print("Interpretation: if 'T syncs' is a large share of 'encode total', the")
print("SOP measurement hook is the bottleneck and gating it is the fix. If it")
print("is a small share, the cost is in the loop body / manifold kernels and")
print("gating the hook will not help.")
print("=" * 74)

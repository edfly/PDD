# -*- coding: utf-8 -*-
"""Diagnostic 2: is the LIF neuron the fixed-cost sink?

Previous step ruled out the host-device sync (3 ms of ~1100 ms) and showed
encode() costs the same at N=994 and N=10057 nodes, i.e. the cost does not
scale with the data. The remaining candidate is the LIF neuron: MSPE calls it
once over the whole (N, T, d) sequence and STMP calls it once per time step,
so 16 LIF time-steps run per encode. If a single LIF step costs a fixed amount
regardless of N, the arithmetic accounts for the whole encode time.

This measures, at two batch sizes:
  * MSPE sub-steps (proj / expmap0 / merge / LIF-over-T)
  * one STMP step (RGC vs LIF vs manifold ops)
  * one bare LIF call at (N,1,d)

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


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


for bs in (64, 512):
    b = next(iter(DataLoader(ds[idx[:bs]], batch_size=bs))).to(DEV)
    b.x = b.x.float()
    N = b.x.shape[0]
    mspe = model.mspe
    lif = mspe.lif
    mani = mspe.manifold

    # ---- MSPE sub-steps -------------------------------------------------
    z = torch.relu(mspe.input_proj(b.x))
    z_h, z_e = mani.split(z)

    t_proj = timeit(lambda: torch.relu(mspe.input_proj(b.x)))
    t_exp = timeit(lambda: mani.hyp.projx(mani.hyp.expmap0(z_h)))
    z_h2 = mani.hyp.projx(mani.hyp.expmap0(z_h))
    t_merge = timeit(lambda: mani.merge(z_h2, z_e))
    x_seq = mani.merge(z_h2, z_e).unsqueeze(1).expand(-1, cfg.T, -1).clone()
    p_init = (z_h2, z_e)
    t_lif_T = timeit(lambda: (lif.reset_episode_stats(),
                              lif(x_seq, p_init))[1])

    # ---- one STMP step --------------------------------------------------
    stmp = model.stmp
    lif2 = stmp.lif
    x_step = x_seq[:, 0:1, :]
    t_lif_1 = timeit(lambda: (lif2.reset_episode_stats(),
                               lif2(x_step, (z_h2.clone(), z_e.clone())))[1])
    t_rgc = timeit(lambda: stmp.rgc_layers[0]((z_h2.clone(), z_e.clone()),
                                              b.edge_index))

    print("batch=%4d  N=%6d" % (bs, N))
    print("   MSPE: input_proj      %7.2f ms" % (t_proj * 1000))
    print("   MSPE: expmap0+projx   %7.2f ms" % (t_exp * 1000))
    print("   MSPE: merge           %7.2f ms" % (t_merge * 1000))
    print("   MSPE: LIF over T=%d     %7.2f ms" % (cfg.T, t_lif_T * 1000))
    print("   STMP: one LIF step    %7.2f ms" % (t_lif_1 * 1000))
    print("   STMP: one RGC layer   %7.2f ms" % (t_rgc * 1000))
    print("   -> LIF total/encode  %7.2f ms  (T in MSPE + T in STMP)"
          % ((t_lif_T + cfg.T * t_lif_1) * 1000))
    print()

print("If 'LIF total' accounts for most of the ~1100 ms encode, the neuron is")
print("the sink; open neurons.py and look for a per-step python loop or a")
print("surrogate-gradient formula built from many small ops.")

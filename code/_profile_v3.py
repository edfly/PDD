# -*- coding: utf-8 -*-
"""Profiler v3 (fast, streaming): confirm encode bottleneck before editing.

Regime 1: small-graph G-sweep (batch-independence) + torch.profiler hotspot
          table that surfaces sync events (aten::_local_scalar_dense / DtoH
          memcpy / cudaDeviceSynchronize).
Regime 2: forward-only + peak memory at ogbn-arxiv scale (no backward, which
          is the heavy/long part and the actual bottleneck to fix).
Output is flushed to _profile_v3.log so it can be read incrementally.
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

LOG = open(os.path.join(_HERE, "_profile_v3.log"), "w", buffering=1)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    print(s, file=LOG, flush=True)


from torch_geometric.data import Data, Batch
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"


def make_data(N, p_edge, feat_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    num_edges = max(N, int(p_edge * N * (N - 1) / 2))
    src = torch.randint(0, N, (num_edges,), generator=g)
    dst = torch.randint(0, N, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    x = torch.randn(N, feat_dim, generator=g)
    return Data(x=x, edge_index=edge_index)


def build_batch(G, N0, p_edge, feat_dim):
    return Batch.from_data_list([make_data(N0, p_edge, feat_dim, seed=i) for i in range(G)])


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


log("=" * 78)
log("BUILD MODEL")
log("=" * 78)
cfg = SpikeRPGNNConfig(
    d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
    u_threshold=1.0, noise_std=0.05, ablation="full", use_dvm=True,
)
model = create_spikergnn(cfg, in_dim=128).to(DEV)
model.train()
log("model ready; device=%s" % DEV)

log("\n" + "=" * 78)
log("REGIME 1: small graphs (N0=30 each), vary #graphs G  -> batch-independence")
log("=" * 78)
for G in (4, 16, 64, 256, 512):
    batch = build_batch(G, 30, 0.35, 128).to(DEV)
    t = timeit(lambda: model.encode_nodes(batch.x, batch.edge_index, batch.batch))
    log("  G=%4d  total_nodes=%6d  encode_nodes fwd = %7.1f ms" % (G, batch.x.shape[0], t))

log("\n" + "=" * 78)
log("REGIME 1b: torch.profiler on G=16 (hotspot + sync events)")
log("=" * 78)
batch = build_batch(16, 30, 0.35, 128).to(DEV)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
) as prof:
    model.encode_nodes(batch.x, batch.edge_index, batch.batch)
    torch.cuda.synchronize()
log(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
# surface sync-related events
log("\nSYNC-RELATED EVENTS (item / memcpy DtoH / deviceSynchronize):")
for e in prof.key_averages():
    name = e.key
    if ("scalar" in name or "synchronize" in name or "Memcpy" in name
            or "item" in name or "_local_scalar" in name):
        if e.count > 0:
            log("  %-40s count=%6d  cpu_total=%8.1f us  cuda_total=%8.1f us"
                % (name, e.count, e.cpu_time_total, e.device_time_total))

log("\n" + "=" * 78)
log("REGIME 2: forward-only + peak memory, single graph N=169000 (ogbn-arxiv scale)")
log("=" * 78)
N = 169000
feat_dim = 128
g = torch.Generator().manual_seed(1)
avg_deg = 15
num_edges = int(N * avg_deg / 2)
src = torch.randint(0, N, (num_edges,), generator=g)
dst = torch.randint(0, N, (num_edges,), generator=g)
edge_index = torch.stack([src, dst], dim=0).to(DEV)
x = torch.randn(N, feat_dim, device=DEV)
batch_vec = torch.zeros(N, dtype=torch.long, device=DEV)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
t_fwd = timeit(lambda: model.encode_nodes(x, edge_index, batch_vec), n=3, warm=1)
peak = torch.cuda.max_memory_allocated() / 1024 ** 3
log("  encode_nodes forward (N=%d): %7.1f ms   peak GPU mem = %.2f GB" % (N, t_fwd, peak))
log("\nDONE")
LOG.close()

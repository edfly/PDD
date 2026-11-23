# -*- coding: utf-8 -*-
"""Profiler v2: confirm the encode bottleneck before editing.

Two regimes, matching the two complaints:
  Regime 1 (small graphs, batched)  -> reproduce "~1.1 s encode independent of
              batch" and locate the cost with torch.profiler (kernel count,
              CPU/CUDA split, sync events).
  Regime 2 (large full-batch graph) -> measure forward time + peak memory of
              encode_nodes on a 169k-node graph (the ogbn-arxiv scale).

Read-only: patches nothing, writes nothing, mutates no source file.
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

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
    datas = [make_data(N0, p_edge, feat_dim, seed=i) for i in range(G)]
    batch = Batch.from_data_list(datas)
    return batch


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000  # ms


print("=" * 78)
print("BUILD MODEL")
print("=" * 78)
cfg = SpikeRPGNNConfig(
    d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
    u_threshold=1.0, noise_std=0.05, ablation="full", use_dvm=True,
)
model = create_spikergnn(cfg, in_dim=128).to(DEV)
model.train()
print("model ready; device=%s" % DEV)

# ── Regime 1: small graphs, vary #graphs (batch size) ───────────────────────
print("\n" + "=" * 78)
print("REGIME 1: small graphs (N0=30 nodes each), vary #graphs G")
print("encode_nodes() forward time should be ~constant in G if batch-independent")
print("=" * 78)
for G in (4, 16, 64, 256, 512):
    batch = build_batch(G, 30, 0.35, 128).to(DEV)
    t = timeit(lambda: model.encode_nodes(batch.x, batch.edge_index, batch.batch))
    print("  G=%4d  total_nodes=%6d  encode_nodes forward = %7.1f ms"
          % (G, batch.x.shape[0], t))

# ── Regime 1b: torch.profiler on one representative batch ───────────────────
print("\n" + "=" * 78)
print("REGIME 1b: torch.profiler on G=16 batch (locate hotspot + sync events)")
print("=" * 78)
batch = build_batch(16, 30, 0.35, 128).to(DEV)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    model.encode_nodes(batch.x, batch.edge_index, batch.batch)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Count CUDA events roughly: sum of self cuda time + number of ops
evt = prof.key_averages()
n_cuda_ops = sum(1 for e in evt if e.device_time_total > 0)
self_cpu_total = sum(e.cpu_time_total for e in evt)
print("\n  #CUDA ops (cpu_time_total>0 categories): %d" % n_cuda_ops)
print("  total CPU self time (us): %d" % self_cpu_total)

# ── Regime 2: large full-batch graph (ogbn-arxiv scale) ────────────────────
print("\n" + "=" * 78)
print("REGIME 2: large single graph, N = 169000 nodes, avg degree ~15")
print("measure forward time + peak memory of encode_nodes (forward only)")
print("=" * 78)
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
peak = torch.cuda.max_memory_allocated() / 1024**3
print("  encode_nodes forward (N=%d): %7.1f ms   peak GPU mem = %.2f GB"
      % (N, t_fwd, peak))

# forward + backward memory
torch.cuda.reset_peak_memory_stats()
model.zero_grad()
out = model.encode_nodes(x, edge_index, batch_vec)
loss = out[0].pow(2).sum() + out[1].pow(2).sum()
torch.cuda.synchronize()
t_bwd_start = time.time()
loss.backward()
torch.cuda.synchronize()
t_bwd = (time.time() - t_bwd_start) * 1000
peak2 = torch.cuda.max_memory_allocated() / 1024**3
print("  forward+backward: backward = %7.1f ms   peak GPU mem = %.2f GB" % (t_bwd, peak2))
print("\nDONE")

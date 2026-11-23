"""Diagnostic: does the 2-hop sampler explode on ogbn-arxiv (small-diameter
citation graph)? Measure (a) subgraph size for one batch, (b) single full-graph
forward timing + peak GPU memory. Informs whether to switch to full-batch."""
import os, time, torch
from torch_geometric.utils import subgraph

# weights_only patch + D: cache (mirror run_nodelevel.py)
_orig = torch.load
def _wl_off(*a, **k):
    k["weights_only"] = False; return _orig(*a, **k)
torch.load = _wl_off
_CACHE = "D:/2026/paper/PBB/post/data/ogb_cache"
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("OGB_DIR", _CACHE)
os.environ.setdefault("TORCH_GEOMETRIC_DIR", _CACHE)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ogb.nodeproppred import NodePropPredDataset

t0 = time.time()
ds = NodePropPredDataset(name="ogbn-arxiv", root=_CACHE)
g = ds.graph
x = torch.tensor(g["node_feat"], dtype=torch.float32)
edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
y = torch.as_tensor(ds.labels, dtype=torch.long).squeeze(-1)
split = ds.get_idx_split()
N = x.size(0); E = edge_index.size(1)
deg = (edge_index[0] >= 0).sum().item()  # dummy
d = torch.bincount(edge_index[0], minlength=N).float()
print(f"[load] {time.time()-t0:.1f}s | N={N} E={E} | deg mean={d.mean():.1f} "
      f"max={int(d.max())} | train={len(split['train'])} "
      f"val={len(split['valid'])} test={len(split['test'])}", flush=True)

# (a) 2-hop subgraph size for a 1024-seed batch, [15,10]
import random
rng = random.Random(0)
adj = [[] for _ in range(N)]
row = edge_index[0].tolist(); col = edge_index[1].tolist()
for c, r in zip(col, row):
    adj[c].append(r)
def sample_hop(nodes, k):
    out = []
    for n in nodes:
        nb = adj[n]
        if not nb: continue
        out.extend(rng.sample(nb, k) if len(nb) > k else nb)
    return out
seeds = split["train"][:1024].tolist()
h1 = sample_hop(seeds, 15); h2 = sample_hop(h1, 10)
sub = list(dict.fromkeys(seeds + h1 + h2))
print(f"[sampler] 1024 seeds -> hop1={len(h1)} hop2={len(h2)} "
      f"subgraph_unique={len(sub)} ({100*len(sub)/N:.1f}% of graph)", flush=True)

# (b) single full-graph forward of SpikeRPGNNNode (T=8) timing + peak mem
torch.cuda.empty_cache()
dev = torch.device("cuda")
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn
from run_nodelevel import SpikeRPGNNNode
cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                       u_threshold=1.0, noise_std=0.05, ablation="full", use_dvm=True)
model = SpikeRPGNNNode(cfg, in_dim=128, out_dim=40).to(dev)
xe = x.to(dev); eie = edge_index.to(dev); batch = torch.zeros(N, dtype=torch.long, device=dev)
torch.cuda.reset_peak_memory_stats(dev)
t1 = time.time()
with torch.no_grad():
    logits = model(xe, eie, batch)
    rate = model.spike_rate(N, cfg.T)
print(f"[full-fwd] {time.time()-t1:.1f}s | logits={tuple(logits.shape)} "
      f"rate={rate:.3f} | peak_mem={torch.cuda.max_memory_allocated(dev)/1e9:.1f}GB",
      flush=True)

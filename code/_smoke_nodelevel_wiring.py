"""Fast wiring smoke test for run_nodelevel.py's custom sampler path.

Uses a tiny synthetic graph (not the 169k real one) so it finishes in seconds
on CPU and isolates: build_adj -> sample_batch (subgraph + seed-first) ->
SpikeRPGNNNode.encode_nodes -> logmap0 concat -> head shape -> seed indexing.
"""
import sys, os, torch, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data
from spikergnn.config import SpikeRPGNNConfig
from run_nodelevel import SpikeRPGNNNode, EuclidBaseline, build_adj, sample_batch

torch.manual_seed(0)
N, F, C = 300, 16, 4
x = torch.randn(N, F)
edge_index = torch.randint(0, N, (2, 900))
y = torch.randint(0, C, (N,))
adj = build_adj(edge_index.cpu())
print("adj built, nodes:", len(adj), "total edges:", sum(len(a) for a in adj))

cfg = SpikeRPGNNConfig(d_h=8, d_e=8, hidden_dim=16, T=4, num_layers=2,
                       u_threshold=1.0, use_dvm=True)
model = SpikeRPGNNNode(cfg, in_dim=F, out_dim=C)
print("model params:", sum(p.numel() for p in model.parameters()))

rng = random.Random(0)
seeds = torch.arange(0, 64)  # first 64 nodes are seeds
x_sub, ei_sub, sub_t = sample_batch(adj, seeds, [10, 5], x, edge_index,
                                    "cpu", rng)
print("subgraph nodes:", x_sub.size(0), "edges:", ei_sub.size(1),
      "seed-first ok:", sub_t[:64].tolist() == list(range(64)))

logits = model(x_sub, ei_sub, torch.zeros(x_sub.size(0), dtype=torch.long))
print("logits shape:", tuple(logits.shape),
      "seed logits shape:", tuple(logits[:64].shape))
loss = torch.nn.functional.cross_entropy(logits[:64], y[:64])
loss.backward()
print("loss:", round(loss.item(), 4), "grad norm:",
      round(sum(p.grad.abs().sum().item() for p in model.parameters()
                if p.grad is not None), 3))

base = EuclidBaseline(F, 16, C, kind="gcn")
blogits = base(x_sub, ei_sub)
print("baseline logits shape:", tuple(blogits.shape))

print("WIRING OK")

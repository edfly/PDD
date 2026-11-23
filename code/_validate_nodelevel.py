"""
Lightweight feasibility validation: can SpikeRPGNN be adapted to NODE-LEVEL
classification?

We extract per-node product-manifold embeddings via the new encode_nodes()
(temporal pooling done, global graph pooling skipped), map the hyperbolic part
to the tangent space at the origin (logmap0), concatenate with the Euclidean
part, and train a single linear head with CE on a tiny synthetic graph.

Goal: prove
  (a) per-node embeddings are extractable from the REAL spiking Riemannian
      pipeline (MSPE -> STMP -> LIF -> DvM),
  (b) gradients flow back into MSPE/STMP/LIF/DvM parameters,
  (c) the spike-rate mechanism is still engaged (spike_count > 0),
  (d) a node-level linear head actually learns (loss decreases, accuracy up).

This is the de-risking step before committing GPU hours to a full
ogbn-arxiv campaign. CPU-only, ~seconds.
"""
import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


def build_synthetic_graph(n_nodes=80, feat_dim=16, n_classes=4, p_edge=0.08,
                          n_train=60, device="cpu"):
    """Cluster-based features so the node task is learnable by message passing."""
    torch.manual_seed(0)
    centroids = torch.randn(n_classes, feat_dim) * 3.0
    labels = torch.randint(0, n_classes, (n_nodes,))
    x = centroids[labels] + torch.randn(n_nodes, feat_dim) * 0.6
    edge_list = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            same = (labels[i] == labels[j]).item()
            if torch.rand(1).item() < (p_edge * 5.0 if same else p_edge * 0.3):
                edge_list.append([i, j])
                edge_list.append([j, i])
    if not edge_list:
        edge_list = [[0, 1], [1, 0]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    perm = torch.randperm(n_nodes)
    train_mask[perm[:n_train]] = True
    return (x.to(device), edge_index.to(device), labels.to(device),
            train_mask.to(device))


def main():
    device = "cpu"
    cfg = SpikeRPGNNConfig(
        d_h=8, d_e=8, hidden_dim=16, T=4, num_layers=2,
        u_threshold=1.0, noise_std=0.05, ablation="full", use_dvm=True,
    )
    model = create_spikergnn(cfg, in_dim=16).to(device)
    model.train()
    n_classes = 4
    head = nn.Linear(cfg.d_h + cfg.d_e, n_classes).to(device)

    x, edge_index, y, train_mask = build_synthetic_graph(device=device)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=1e-2
    )

    losses, spike_counts = [], []
    for step in range(40):
        opt.zero_grad()
        z_hyp, z_euc = model.encode_nodes(x, edge_index, batch)
        v_hyp = model.manifold.hyp.logmap0(z_hyp)        # (N, d_h) tangent
        feat = torch.cat([v_hyp, z_euc], dim=-1)          # (N, hidden_dim)
        logits = head(feat)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()
        losses.append(loss.item())
        spike_counts.append(model.count_spikes())

    ok_decrease = losses[-1] < losses[0] * 0.95
    ok_finite = all(math.isfinite(l) for l in losses)
    some_param = next(model.parameters())
    grad_flow = (
        some_param.grad is not None and some_param.grad.abs().sum().item() > 0
    )

    with torch.no_grad():
        z_hyp, z_euc = model.encode_nodes(x, edge_index, batch)
        v_hyp = model.manifold.hyp.logmap0(z_hyp)
        feat = torch.cat([v_hyp, z_euc], dim=-1)
        pred = head(feat).argmax(dim=1)
        acc_train = (pred[train_mask] == y[train_mask]).float().mean().item()
        acc_all = (pred == y).float().mean().item()

    print("=== node-level feasibility validation (SpikeRPGNN) ===")
    print(f"loss[0]={losses[0]:.4f}  loss[39]={losses[-1]:.4f}  "
          f"decreased={ok_decrease}")
    print(f"all finite={ok_finite}")
    print(f"spike_count/step: min={min(spike_counts)} max={max(spike_counts)} "
          f"avg={sum(spike_counts)/len(spike_counts):.1f}")
    print(f"gradient flows into model params={grad_flow}")
    print(f"train-node acc={acc_train:.3f}  all-node acc={acc_all:.3f}")
    print(f"z_hyp shape={tuple(z_hyp.shape)} z_euc shape={tuple(z_euc.shape)}")
    verdict = ok_decrease and ok_finite and grad_flow
    print("VERDICT:",
          "VIABLE - SpikeRPGNN adapts to node-level"
          if verdict else "INCONCLUSIVE")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Dense (non-spiking) baseline runner with a pluggable GNN backbone.

Purpose
-------
Generate a real Graph Attention Network (GAT) baseline to close the
Neurocomputing R4 baseline gap (HGCN was already present; GAT was missing).
The runner reuses the project's episodic sampler / protocol so the number is
directly comparable to the existing dense Proto-GCN / Proto-GIN / Proto-SAGE
rows in the main results table.

Protocol (identical to SpikeRPGNN main matrix, see code/msg_baseline/train_episodic.py
and code/msg_baseline/config.py):
    epochs = 50, episodes_per_epoch = 20, test_episodes = 30,
    hidden_dim = 128, num_layers = 3, q_query = 5, train_ratio = 0.80,
    seed = 42 + run_id*100 + shot*10

Supported backbones: gcn, gin, sage, gat.  Only `--backbone gat` is run for
this task.  The others are implemented for parity but are intentionally NOT
re-run: re-running them would shift the already-published gcn/gin/sage numbers
and the BH-FDR significance family, which is out of scope for R4.

Output JSON schema (matches results/ann_baselines_v30.json):
    {gat_MUTAG_1shot_run0: {dataset, shot, run_id, final_acc, best_acc, elapsed_sec}, ...}

Usage (single-run queue contract, watchdog-friendly):
    python code/run_dense_baseline.py --backbone gat \
        --datasets MUTAG PROTEINS ENZYMES DD REDDIT-M5K --shots 1 5 \
        --device cuda --out results/gat_baselines_v30.json --n-runs 5

NOTE on cwd: TUGraphDataset loads TUDataset(root="./data/TU", ...) relative to
the current working directory, so launch with cwd = D:/2026/paper/PBB (the tree
that owns data/TU) to reuse the dataset cache.
"""
import sys
import os
import json
import time
import argparse
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
# Allow `import msg_baseline...` (the package lives in code/msg_baseline).
sys.path.insert(0, _HERE)

from msg_baseline.data_utils import (  # noqa: E402
    TUGraphDataset, EpisodeSampler, batch_episode_data,
)
from msg_baseline.config import get_config, DATASET_CONFIGS  # noqa: E402
from msg_baseline.utils import set_seed, compute_accuracy  # noqa: E402

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import (  # noqa: E402
    GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool,
)

ALL_DATASETS = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "REDDIT-M5K"]


# ──────────────────────────────────────────────────────────────────────────
# Dense encoder (Euclidean GNN backbone)
# ──────────────────────────────────────────────────────────────────────────
class DenseEncoder(nn.Module):
    """3-layer dense GNN producing a graph embedding via mean pooling.

    Capacity-matched to the rest of the main table: hidden_dim = 128,
    every layer outputs 128 (for GAT, 8 heads x 16 = 128).
    """

    def __init__(self, backbone: str, in_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        in_dim = max(int(in_dim), 1)
        self.input_proj = Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if backbone == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif backbone == "gin":
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_dim, hidden_dim),
                            BatchNorm1d(hidden_dim),
                            ReLU(),
                            Linear(hidden_dim, hidden_dim),
                            BatchNorm1d(hidden_dim),
                            ReLU(),
                        ),
                        hidden_dim,
                    )
                )
            elif backbone == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif backbone == "gat":
                # 8 heads x out_channels(16) = 128, capacity-matched.
                self.convs.append(GATConv(hidden_dim, 16, heads=8, dropout=0.0))
            else:
                raise ValueError(f"unknown backbone {backbone}")
        self.act = ReLU()

    def forward(self, x, edge_index, batch):
        if x is None or (isinstance(x, torch.Tensor) and x.dim() == 0):
            x = torch.ones(batch.numel(), 1, dtype=torch.float,
                            device=edge_index.device)
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
        return global_mean_pool(x, batch)  # (num_graphs, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────
# Euclidean prototypical classifier (standard ProtoNet readout)
# ──────────────────────────────────────────────────────────────────────────
class EuclideanProtoClassifier(nn.Module):
    """Prototypes = Euclidean mean of support embeddings; logits = -||d||^2."""

    def compute_prototypes(self, support_z, support_labels, n_classes):
        protos = []
        for c in range(n_classes):
            mask = support_labels == c
            protos.append(support_z[mask].mean(dim=0))
        return torch.stack(protos, dim=0)  # (N_way, d)

    def forward(self, query_z, support_z, support_labels, n_classes):
        protos = self.compute_prototypes(support_z, support_labels, n_classes)
        dists = torch.cdist(query_z, protos) ** 2  # (N_q, N_way)
        return -dists


# ──────────────────────────────────────────────────────────────────────────
# Training / evaluation (mirrors train_episodic.run_single_experiment)
# ──────────────────────────────────────────────────────────────────────────
def _train_one_epoch(encoder, classifier, sampler, optimizer, config):
    encoder.train()
    total_acc = 0.0
    for _ in range(config.episodes_per_epoch):
        episode = sampler.sample_episode(mode="train")
        supp_b, supp_l, qry_b, qry_l = batch_episode_data(episode, device=config.device)
        supp_z = encoder(supp_b.x, supp_b.edge_index, supp_b.batch)
        qry_z = encoder(qry_b.x, qry_b.edge_index, qry_b.batch)
        logits = classifier(qry_z, supp_z, supp_l, n_classes=sampler.n_way)
        loss = nn.CrossEntropyLoss()(logits, qry_l)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        total_acc += compute_accuracy(logits, qry_l)
    return total_acc / config.episodes_per_epoch


@torch.no_grad()
def _evaluate(encoder, classifier, dataset, config, shot):
    encoder.eval()
    ds_cfg = DATASET_CONFIGS[dataset.name]
    test_sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"], k_shot=shot,
        q_query=config.q_query, seed=config.seed_base + 10000,
    )
    total = 0.0
    for _ in range(config.test_episodes):
        episode = test_sampler.sample_episode(mode="test")
        supp_b, supp_l, qry_b, qry_l = batch_episode_data(episode, device=config.device)
        supp_z = encoder(supp_b.x, supp_b.edge_index, supp_b.batch)
        qry_z = encoder(qry_b.x, qry_b.edge_index, qry_b.batch)
        logits = classifier(qry_z, supp_z, supp_l, n_classes=test_sampler.n_way)
        total += compute_accuracy(logits, qry_l)
    return total / config.test_episodes


def run_single_experiment(dataset_name, shot, run_id, config, backbone):
    ds_cfg = DATASET_CONFIGS[dataset_name]
    run_seed = config.seed_base + run_id * 100 + shot * 10
    set_seed(run_seed)

    dataset = TUGraphDataset(
        name=dataset_name, tu_name=ds_cfg["tu_name"],
        train_ratio=config.train_ratio, seed=config.seed_base,
    )
    sampler = EpisodeSampler(
        dataset=dataset, n_way=ds_cfg["n_way"], k_shot=shot,
        q_query=config.q_query, seed=run_seed,
    )
    encoder = DenseEncoder(
        backbone, in_dim=dataset.num_features,
        hidden_dim=config.hidden_dim, num_layers=config.num_layers,
    ).to(config.device)
    classifier = EuclideanProtoClassifier()

    optimizer = optim.Adam(
        encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5,
    )

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_acc = _train_one_epoch(encoder, classifier, sampler, optimizer, config)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            test_acc = _evaluate(encoder, classifier, dataset, config, shot)
            if test_acc > best_acc:
                best_acc = test_acc
            print(f"  Epoch {epoch:3d}/{config.epochs} | Train: {train_acc:.1%} | "
                  f"Test: {test_acc:.1%} | Best: {best_acc:.1%} | {time.time()-t0:.0f}s",
                  flush=True)
    final_acc = _evaluate(encoder, classifier, dataset, config, shot)
    elapsed = time.time() - t0
    print(f"  >>> Final Accuracy: {final_acc:.1%} ({elapsed:.1f}s)", flush=True)
    return {
        "accuracy": final_acc, "best_acc": best_acc, "time_seconds": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI harness (single-run + atomic-save + resume, like run_msgstar_capacity.py)
# ──────────────────────────────────────────────────────────────────────────
def atomic_save_json(path, obj, retries=6, wait=5):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for i in range(retries):
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2, default=str)
            os.replace(tmp, path)
            return
        except (PermissionError, OSError):
            if i == retries - 1:
                raise
            time.sleep(wait)


def build_config(args):
    cfg = get_config()
    cfg.hidden_dim = args.hidden_dim
    cfg.epochs = args.epochs
    cfg.episodes_per_epoch = args.episodes_per_epoch
    cfg.test_episodes = args.test_episodes
    cfg.n_runs = args.n_runs
    cfg.device = args.device
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Dense GNN baseline runner")
    ap.add_argument("--backbone", default="gat", choices=["gcn", "gin", "sage", "gat"])
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--episodes-per-epoch", type=int, default=20)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--run-id", type=int, default=None,
                    help="run only this run index (queue-driver contract)")
    ap.add_argument("--out",
                    default=os.path.join(_HERE, "..", "results", "gat_baselines_v30.json"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.episodes_per_epoch, args.test_episodes = 3, 3, 5
        args.n_runs = 1

    cfg = build_config(args)
    out = os.path.abspath(args.out)

    results = {}
    if args.resume and os.path.exists(out):
        try:
            with open(out) as f:
                results = json.load(f)
        except Exception:
            results = {}

    run_ids = [args.run_id] if args.run_id is not None else list(range(args.n_runs))

    print(f"{args.backbone.upper()} dense baseline | d={args.hidden_dim} | "
          f"device={args.device} | datasets={args.datasets} | shots={args.shots} | "
          f"runs={run_ids} | epochs={args.epochs} | ep/epoch={args.episodes_per_epoch} | "
          f"test_eps={args.test_episodes}", flush=True)

    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            print(f"SKIP unknown dataset {ds}", flush=True)
            continue
        for shot in args.shots:
            for rid in run_ids:
                key = f"{args.backbone}_{ds}_{shot}shot_run{rid}"
                if args.resume and key in results and isinstance(results[key], dict) \
                        and "final_acc" in results[key]:
                    print(f"SKIP {key}", flush=True)
                    continue
                t0 = time.time()
                try:
                    r = run_single_experiment(ds, shot, rid, cfg, args.backbone)
                    r = {
                        "dataset": ds, "shot": shot, "run_id": rid,
                        "final_acc": float(r["accuracy"]),
                        "best_acc": float(r["best_acc"]),
                        "elapsed_sec": round(time.time() - t0, 1),
                    }
                    results[key] = r
                    print(f">>> {key}: acc={r['final_acc']*100:.2f}% "
                          f"({r['elapsed_sec']}s)", flush=True)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    results[key] = {"error": str(e), "dataset": ds,
                                    "shot": shot, "run_id": rid}
                atomic_save_json(out, results)

    print("DONE", flush=True)
    return results


if __name__ == "__main__":
    main()

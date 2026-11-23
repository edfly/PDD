# -*- coding: utf-8 -*-
"""ogbg-molhiv supervised benchmark for SpikeRPGNN (audit extension).

Why this script exists
----------------------
The manuscript concedes that its largest benchmark (REDDIT-MULTI-5K,
4,999 graphs) is "modest by contemporary graph-learning standards", and the
Nature-grade audit scored Timeliness 3.0 partly for having no modern
large-scale reference. ogbg-molhiv is the standard OGB graph-property
benchmark: 41,127 molecular graphs, scaffold split, ROC-AUC.

Protocol note — READ BEFORE CITING ANY NUMBER FROM THIS FILE
-----------------------------------------------------------
This is a SUPERVISED full-data protocol, NOT the episodic few-shot protocol
of the main matrix. The two are not comparable and must never be pooled into
the same significance family. This script exists to answer one question the
main matrix cannot: does the rate controller still hit its commanded rate
at 8x more graphs and 2x more training signal, and where does the resulting
operating point sit relative to a capacity-matched dense backbone?

Consequences of the protocol change, stated explicitly:
  * Encoding: graph embedding is the Euclidean component z[1] of the product
    manifold, fed to a linear head. The prototypical classifier (FPC/GDC) is
    unused because there are no episodes.
  * Class imbalance: molhiv is ~1.4% positives, so ROC-AUC (not accuracy) is
    the metric, per OGB.
  * No few-shot claim of any kind comes out of this file.

GPU: GPU-first per project rule. Batch training on a 2080 Ti is ~100x faster
than the episodic runner, which is why this is the large-scale path.

Usage
-----
    python code/run_ogbg.py --smoke
    python code/run_ogbg.py --model spikergnn --n-runs 5 --epochs 50
    python code/run_ogbg.py --model gin --n-runs 5 --epochs 50

Output
------
    post/results/ogbg.json    (new file; never overwrites v30 artefacts)
"""
import sys
import os
import json
import time
import math
import argparse
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
os.chdir(_REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool

# PyTorch >= 2.6 flips torch.load to weights_only=True, which rejects the
# PyG storage classes OGB pickles into its cache. These are files we just
# generated ourselves, so allow-listing is safe.
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DATA_ROOT = os.path.join(_REPO, "data")
RESULTS_PATH = os.path.join(_REPO, "post", "results", "ogbg.json")

HIDDEN = 128          # capacity-matched with the main matrix (d=128)
LAYERS = 3
# Batch size is set large on purpose. Profiling showed encode() costs ~1.0 s
# regardless of batch size (32 -> 1024 graphs all land at 0.91-1.13 s), i.e.
# the cost is a fixed per-call overhead (the T-step / manifold python loops),
# not GPU work. Batches per epoch therefore dominate: 257 batches at 128
# graphs costs ~257 s/epoch forward, 32 batches at 1024 costs ~29 s -- an
# 8.9x speedup that is what makes this benchmark affordable at all.
# Two measurements reversed the earlier conclusion that this benchmark was
# unaffordable:
#   * encode() costs ~1.0 s REGARDLESS of batch (32 -> 1024 graphs), so the
#     only lever that matters is the NUMBER of batches per epoch. 1024 gives
#     32 batches, 512 gives 64, and the epoch cost scales with that count.
#   * a full train step (forward+backward+step) at bs=512 measures 2.47 s and
#     peaks at 3.9 GB, i.e. there is plenty of headroom on a 22 GB card.
# An earlier run appeared to take >1500 s/epoch; that was an artefact of
# num_workers=2, whose process spawn cost on Windows dominated the step.
# With num_workers=0 the real figure is ~1.5 min/epoch at bs=512 (measured
# live: e12->e13 in ~91 s), so the whole 50-epoch x 5-run study is ~6 h, not days.
# 1024 is fine for a forward pass but not for training: the forward already
# peaks at 13.4 GB there and the backward pushes past what fits, which shows
# up as a hang rather than a clean OOM. 512 is the measured sweet spot:
# 2.47 s per full train step, 3.9 GB peak, ~1.5 min per epoch (live-measured).
BATCH = 512
LR = 1e-3
WD = 5e-4
R_TARGET = 0.4
LAMBDA_SR = 0.5


# ──────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────
class DenseGNN(nn.Module):
    """Capacity-matched dense backbone (GIN or GCN) with a linear head.

    Deliberately mirrors the main matrix: hidden 128, 3 layers, mean pool.
    """

    def __init__(self, in_dim, hidden, layers, conv="gin"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            d_in = in_dim if i == 0 else hidden
            if conv == "gin":
                mlp = nn.Sequential(
                    nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINConv(mlp))
            else:
                self.convs.append(GCNConv(d_in, hidden))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
        g = global_mean_pool(x, batch)
        return self.head(g).squeeze(-1)


class SpikeRPGNNWrapper(nn.Module):
    """SpikeRPGNN encoder + linear head for supervised graph prediction."""

    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        cfg = SpikeRPGNNConfig(
            hidden_dim=hidden, num_layers=layers,
            lambda_sr=LAMBDA_SR, r_target=R_TARGET)
        self.model = create_spikergnn(cfg, in_dim=in_dim)
        # hidden=128 splits into a hyperbolic part d_h and a Euclidean part
        # d_e with d_h + d_e = hidden (asserted in SpikeRPGNNConfig). The
        # supervised head reads the Euclidean component only: the hyperbolic
        # component lives on the Poincare ball and cannot be fed to a linear
        # layer without a log-map. This differs from the main-matrix
        # prototypical head, which scores by geodesic distance and therefore
        # uses both components -- which is exactly why numbers from this file
        # are never pooled with the main matrix.
        self.head = nn.Linear(cfg.d_e, 1)
        self.d_e = cfg.d_e
        self.d_h = cfg.d_h

    def forward(self, x, edge_index, batch):
        z = self.model.encode(x, edge_index, batch)
        z_euc = z[1]                      # Euclidean component (N, d_e)
        return self.head(z_euc).squeeze(-1)

    def rate(self):
        return self.model.spike_rate_surrogate()


# ──────────────────────────────────────────────────────────────────
# Train / eval
# ──────────────────────────────────────────────────────────────────
def run_one(model_kind, run_id, epochs, device, batch_size=BATCH, hidden=HIDDEN):
    torch.manual_seed(42 + 100 * run_id)
    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=DATA_ROOT)
    split = ds.get_idx_split()
    train_ds = ds[split["train"]]
    val_ds = ds[split["valid"]]
    test_ds = ds[split["test"]]

    g0 = ds[0]
    in_dim = g0.x.shape[1]

    if model_kind == "spikergnn":
        model = SpikeRPGNNWrapper(in_dim, hidden, LAYERS).to(device)
    else:
        model = DenseGNN(in_dim, hidden, LAYERS, conv=model_kind).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    # num_workers=0 deliberately. Worker processes on Windows are spawned
    # (not forked), and the spawn cost per batch exceeded the GPU step itself
    # here: it made the run look ~10x slower than it is. With batches this
    # large the collate cost is not the bottleneck, so serial loading is both
    # simpler and faster. Do not "optimise" this back to workers>0 without
    # re-measuring end-to-end epoch time.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    r_target_t = torch.tensor(R_TARGET, device=device)
    evaluator = Evaluator(name="ogbg-molhiv")

    best_val, best_test = -1.0, -1.0
    rate_trace = []

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            # OGB molecular node features are integer-coded (int64); every
            # layer downstream expects floats.
            data.x = data.x.float()
            opt.zero_grad()
            logits = model(data.x, data.edge_index, data.batch)
            y = data.y.view(-1).float().to(device)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if model_kind == "spikergnn":
                r_bar = model.rate()
                loss = loss + LAMBDA_SR * (r_bar - r_target_t) ** 2
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # ── evaluate ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_auc = evaluator.eval(_collect(model, val_loader, device))["rocauc"]
            test_auc = evaluator.eval(_collect(model, test_loader, device))["rocauc"]
            if model_kind == "spikergnn":
                rate_trace.append(float(model.rate().item()))
        if val_auc > best_val:
            best_val, best_test = val_auc, test_auc
        print("  e%02d val=%.4f test=%.4f best_test=%.4f r=%.4f"
              % (epoch, val_auc, test_auc, best_test,
                 rate_trace[-1] if rate_trace else float("nan")), flush=True)

    return {
        "model": model_kind, "run_id": run_id, "epochs": epochs,
        "best_val_rocauc": best_val, "test_rocauc_at_best_val": best_test,
        "final_rate": (rate_trace[-1] if rate_trace else None),
        "rate_trace_last5": rate_trace[-5:],
        "r_target": R_TARGET, "lambda_sr": LAMBDA_SR,
        "hidden": hidden, "layers": LAYERS, "device": device,
        "d_e_head": getattr(model, "d_e", None),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@torch.no_grad()
def _collect(model, loader, device):
    """Return the dict shaped exactly as OGB's Evaluator expects.

    Both halves carry the same y_true on purpose: Evaluator.eval() takes one
    dict with 'y_true' and 'y_pred' keys, and a common mistake is to nest a
    second dict inside it, which raises
    "Arguments to Evaluator need to be either numpy ndarray or torch tensor".
    """
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        data.x = data.x.float()
        logits = model(data.x, data.edge_index, data.batch)
        ps.append(torch.sigmoid(logits).detach().cpu())
        ys.append(data.y.view(-1).float().detach().cpu())
    return {
        "y_true": torch.cat(ys).unsqueeze(-1).numpy(),
        "y_pred": torch.cat(ps).unsqueeze(-1).numpy(),
    }


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spikergnn",
                    choices=["spikergnn", "gin", "gcn"])
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_runs, args.epochs = 1, 2

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[abort] cuda requested but unavailable")
        sys.exit(1)
    print("model=%s device=%s | %s"
          % (args.model, args.device,
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    rows = load_results()
    t0 = time.time()
    for run_id in range(args.n_runs):
        # Keyed on (model, hidden, run_id): the dense-96 control must not be
        # treated as a repeat of the dense-128 run.
        if any(r.get("model") == args.model and r.get("run_id") == run_id
               and r.get("hidden") == args.hidden for r in rows):
            print("[skip done] %s run%d" % (args.model, run_id))
            continue
        try:
            out = run_one(args.model, run_id, args.epochs, args.device,
                          args.batch, args.hidden)
            rows.append(out)
            os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            print("[ok] %s run%d test_auc=%.4f"
                  % (args.model, run_id, out["test_rocauc_at_best_val"]), flush=True)
        except Exception:
            traceback.print_exc()
            print("[fail] %s run%d" % (args.model, run_id), flush=True)
    print("elapsed %.1f s" % (time.time() - t0))
    print("wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()

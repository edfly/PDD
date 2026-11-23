"""
run_nodelevel.py — SpikeRPGNN on ogbn-arxiv NODE-LEVEL classification.

Answers the paper's weakest axis (Novelty 3/10): "is the Riemannian product
manifold actually buying accuracy, or is geometry just a substrate?" By running
the SAME encoder head-to-head against a plain Euclidean GCN/SAGE on the exact
same node split, we get a clean controlled test of the hyperbolic component.

Design (de-risked by code/_validate_nodelevel.py):
  * SpikeRPGNN.encode_nodes() returns per-node (z_hyp, z_euc) product-manifold
    embeddings (temporal pooling, no graph pooling).
  * Node head = manifold.hyp.logmap0(z_hyp) concatenated with z_euc, then
    nn.Linear(hidden_dim, 40) + CE. Standard HGCN-style tangent-space head.
  * Spike-rate constraint L_sr = lambda_sr*(r_bar - r*)^2 kept (paper core).
  * 169k-node single graph -> custom 2-hop neighbor sampler (pure
    torch_geometric.utils.subgraph, NO pyg-lib dependency) for mini-batching.

Usage:
  python code/run_nodelevel.py --model spikergnn --n-runs 5 --epochs 50
  python code/run_nodelevel.py --model gcn  --n-runs 5 --epochs 200
  python code/run_nodelevel.py --model spikergnn --smoke --device cpu   # sanity

Outputs results/nodelevel.json (checkpointed per run so a kill loses at
most one run, not the whole campaign).
"""
import os
import sys
import json
import math
import argparse
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.nn import GCNConv, SAGEConv

# ── PyTorch >= 2.6 weights_only gotcha (same as ogbg) ──────────────────────
# OGB's pre-processed files reject weights_only=True. Trusted source -> off.
_orig_load = torch.load
def _weights_off_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _weights_off_load

# Redirect OGB/PyG cache to D: (C: is nearly full at ~9 GB)
_CACHE = "D:/2026/paper/PBB/post/data/ogb_cache"
os.makedirs(_CACHE, exist_ok=True)
os.environ.setdefault("OGB_DIR", _CACHE)
os.environ.setdefault("TORCH_GEOMETRIC_DIR", _CACHE)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ogb.nodeproppred import NodePropPredDataset
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


# ── Euclidean baselines (for the substrate comparison) ─────────────────────
class EuclidBaseline(nn.Module):
    """2-layer GCN or SAGE, standard node classification head."""
    def __init__(self, in_dim, hidden, out_dim, kind="gcn"):
        super().__init__()
        conv = GCNConv if kind == "gcn" else SAGEConv
        self.conv1 = conv(in_dim, hidden)
        self.conv2 = conv(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return self.head(h)


# ── SpikeRPGNN node classifier (reuses validated encode_nodes path) ────────
class SpikeRPGNNNode(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        self.model = create_spikergnn(cfg, in_dim=in_dim)
        self.head = nn.Linear(cfg.d_h + cfg.d_e, out_dim)
        self.lam_sr = cfg.lambda_sr
        self.r_star = cfg.r_target

    def forward(self, x, edge_index, batch):
        z_hyp, z_euc = self.model.encode_nodes(x, edge_index, batch)
        v_hyp = self.model.manifold.hyp.logmap0(z_hyp)   # (N, d_h) tangent
        feat = torch.cat([v_hyp, z_euc], dim=-1)          # (N, hidden_dim)
        return self.head(feat)

    def spike_rate_surrogate(self) -> torch.Tensor:
        """Differentiable mean firing rate over the last forward pass (paper Eq. r_bar).

        Uses the LIF surrogate spike rate (accumulated across all T steps of the
        latest encode_nodes pass), which is differentiable w.r.t. u_th_learnable.
        This is what L_sr MUST use: the old count_spikes()-based path detached the
        rate from the graph (.item()), so the learnable threshold received NO
        gradient from L_sr and drifted to silence under the CE gradient (the
        ~0.3% firing-rate collapse observed on ogbn-arxiv).
        """
        return self.model.spike_rate_surrogate()


# ── Dependency-free 2-hop neighbor sampler ──────────────────────────────────
def build_adj(edge_index_cpu):
    """Adjacency list (CPU) for sampling. edge_index_cpu: (2, E) long."""
    row = edge_index_cpu[0].tolist()
    col = edge_index_cpu[1].tolist()
    N = max(row + col) + 1
    adj = [[] for _ in range(N)]
    for c, r in zip(col, row):
        adj[c].append(r)
    return adj


def _sample_hop(adj, nodes, n_neigh, rng):
    out = []
    for n in nodes:
        nb = adj[n]
        if not nb:
            continue
        if len(nb) > n_neigh:
            nb = rng.sample(nb, n_neigh)
        out.extend(nb)
    return out


def sample_batch(adj, seeds, num_neighbors, x, edge_index, device, rng):
    """Return (x_sub, edge_index_sub, sub_nodes) with seeds first."""
    seeds = seeds.tolist()
    hop1 = _sample_hop(adj, seeds, num_neighbors[0], rng)
    hop2 = _sample_hop(adj, hop1, num_neighbors[1], rng)
    # seeds first so target positions are 0..len(seeds)-1
    sub = list(dict.fromkeys(seeds + hop1 + hop2))
    sub_t = torch.tensor(sub, dtype=torch.long, device=device)
    sub_edge, _ = subgraph(sub_t, edge_index.to(device), relabel_nodes=True)
    x_sub = x.to(device)[sub_t]
    return x_sub, sub_edge, sub_t


def batched(seeds, batch_size, shuffle, rng):
    idx = seeds.tolist()
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield torch.tensor(idx[i:i + batch_size], dtype=torch.long)


# ── Training / eval ─────────────────────────────────────────────────────────
def run_one(model_kind, run_id, epochs, device, hidden=128, in_dim=128,
            out_dim=40, num_neighbors=None, batch_size=1024, smoke=False,
            lr=1e-3, full_batch=True, u_threshold=1.0, cuda_graph=False,
            T=8, d_e=96, checkpoint_rgc=True, grad_clip=0.0, lambda_sr=None):
    device = torch.device(device)
    ds = NodePropPredDataset(name="ogbn-arxiv", root=_CACHE)
    g = ds.graph
    x = torch.tensor(g["node_feat"], dtype=torch.float32)
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    y = torch.as_tensor(ds.labels, dtype=torch.long).squeeze(-1)
    split = ds.get_idx_split()
    adj = build_adj(edge_index.cpu()) if not full_batch else None
    rng = random.Random(1234 + run_id)
    # NOTE: T now flows from the --T argument (default 1). It used to be
    # hardcoded `T = 1` here, which silently overrode --T so EVERY production
    # run (including the prior "169k full-batch" experiments) was actually T=1
    # regardless of the flag. Removing the override lets T=8 -- the manifold
    # temporal dimension the gradient-checkpoint memory-fix targets -- actually
    # be trained on 169k full-batch (checkpoint drops the T=8 peak from ~33 GiB
    # to ~17.6 GiB, fitting the 22 GB card).
    noise_std = 0.0  # external-noise level for CUDA-graph replay (SpikeRPGNN only)

    if model_kind == "spikergnn":
        cfg = SpikeRPGNNConfig(
            # d_h fixed at 32; hidden_dim must equal d_h + d_e (config invariant).
            d_h=32, d_e=d_e, hidden_dim=32 + d_e, T=T, num_layers=2,
            u_threshold=u_threshold, noise_std=0.05, ablation="full", use_dvm=True,
            lambda_sr=(lambda_sr if lambda_sr is not None else 0.5),
            # Truncated-BPTT window. The default dv_m_window==T means ZERO
            # truncation. We cap it at min(4, T): gating the STMP recurrent
            # state at t < T-window cuts BACKWARD COMPUTE (gradient path), but
            # because the RGC layers are shared across all T steps every step's
            # forward saved-tensors must still reside simultaneously, so it does
            # NOT reduce peak memory. It is kept only as a compute-saving knob;
            # fitting 169k full-batch in 22 GB requires a smaller config or
            # checkpointing, not windowing. (See _diag_mem_169k.py.)
            dv_m_window=min(4, T),
            # Path-2 memory fix: gradient-checkpoint the per-step RGC stack so
            # the 169k full-batch step fits the 22 GB card (drops ~16-24 GiB of
            # retained RGC saved tensors from the forward peak). No-op in eval.
            checkpoint_rgc=checkpoint_rgc,
        )
        model = SpikeRPGNNNode(cfg, in_dim=in_dim, out_dim=out_dim).to(device)
        num_neighbors = num_neighbors or [15, 10]
        T = cfg.T
        noise_std = cfg.noise_std
    else:
        kind = "gcn" if model_kind == "gcn" else "sage"
        model = EuclidBaseline(in_dim, hidden, out_dim, kind=kind).to(device)
        num_neighbors = num_neighbors or [25, 20]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Preload tensors to device for full-batch (standard for single-graph
    # ogbn-arxiv; forward is ~3 s / 3.6 GB on 169k nodes).
    x_dev = x.to(device)
    ei_dev = edge_index.to(device)
    y_dev = y.to(device)
    batch_all = torch.zeros(x.size(0), dtype=torch.long, device=device)
    train_idx = torch.as_tensor(split["train"]).to(device)
    val_idx = torch.as_tensor(split["valid"]).to(device)
    test_idx = torch.as_tensor(split["test"]).to(device)

    # ── CUDA Graph capture (SpikeRPGNN, full-batch, CUDA only) ─────────────
    # Records the full fwd + L_sr + bwd step ONCE. Replay then removes the
    # per-kernel launch overhead that dominated the 169k backward (>10 min/step
    # on this 22 GB / 4-core box). Verified bit-identical to eager on a 30k
    # POC (_test_cudagraph_small.py, 3.72x) for NON-checkpointed models.
    # Variable-topology mini-batching cannot be captured, so it stays eager.
    # PATH-2 CRITICAL CAVEAT: capturing the gradient-checkpointed graph is
    # SILENTLY BROKEN -- the replay produces a DORMANT model (surrogate firing
    # rate collapses to 0 on replay, so spiking is disabled and the logits are
    # degenerate: test acc ~0.01). The capture itself succeeds and loss/weights
    # even move, which masks the failure unless accuracy/rate are checked. The
    # checkpoint recompute inside the captured backward corrupts the replayed
    # forward. So when checkpoint_rgc is on we train EAGERLY (memory already
    # fits via checkpoint); the CUDA graph stays for the non-checkpointed small
    # configs only.
    cg = None
    _want_graph = cuda_graph and model_kind == "spikergnn" and full_batch \
        and device.type == "cuda"
    if _want_graph and getattr(cfg, "checkpoint_rgc", False):
        print("  [cuda_graph] DISABLED: capturing the gradient-checkpointed "
              "graph yields a DORMANT model (surrogate firing rate -> 0 -> "
              "spiking disabled -> degenerate logits) on replay, though the "
              "capture succeeds. Training 169k full-batch EAGERLY with "
              "checkpoint (memory fits via checkpoint).", flush=True)
        _want_graph = False
    if _want_graph:
        x_cg = x_dev.clone()  # dedicated input the graph reads (re-noised in place)
        cg = _cg_build(model, opt, x_cg, ei_dev, batch_all,
                       train_idx, y_dev, device)
        print(f"  [cuda_graph] captured full training step; replaying in loop",
              flush=True)

    best_val, best_test, rates = 0.0, 0.0, []
    _cuda = device.type == "cuda"
    if _cuda:
        # Reset so the per-report peak reflects the largest step since here,
        # giving hard evidence that the T=8 full-batch step fits the 22 GB card.
        torch.cuda.reset_peak_memory_stats(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        epoch_rate_sum, epoch_rate_n = 0.0, 0  # reset per-epoch firing-rate accumulator
        if full_batch:
            if cg is not None:
                # CUDA-graph replay: fwd + L_sr + bwd + step are all inside the
                # captured graph. External noise + membrane reset happen on the
                # capture stream immediately before replay (see _cg_step).
                _cg_step(cg, model, opt, x_dev, noise_std, grad_clip)
                # Read the surrogate rate the replayed forward accumulated
                # (pure tensor division; host-sync allowed outside capture).
                r_bar = model.spike_rate_surrogate()
                epoch_rate_sum += r_bar.detach().item()
                epoch_rate_n += 1
            else:
                logits = model(x_dev, ei_dev, batch_all)
                loss = F.cross_entropy(logits[train_idx], y_dev[train_idx])
                if model_kind == "spikergnn":
                    # FIX: differentiable surrogate rate (was count_spikes()/.item(),
                    # which detached L_sr from the graph -> threshold drifted silent).
                    r_bar = model.spike_rate_surrogate()
                    loss = loss + model.lam_sr * (r_bar - model.r_star) ** 2
                    epoch_rate_sum += r_bar.detach().item()
                    epoch_rate_n += 1
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        else:
            for seeds in batched(split["train"], batch_size, True, rng):
                x_sub, ei_sub, sub_t = sample_batch(adj, seeds, num_neighbors,
                                                    x, edge_index, device, rng)
                opt.zero_grad()
                logits = model(x_sub, ei_sub,
                               torch.zeros(x_sub.size(0), dtype=torch.long,
                                           device=device))
                n_seed = batch_size if len(sub_t) >= batch_size else len(sub_t)
                target = y[sub_t[:n_seed].cpu()].to(device)
                loss = F.cross_entropy(logits[:n_seed], target)
                if model_kind == "spikergnn":
                    # FIX: differentiable surrogate rate (see full-batch note).
                    r_bar = model.spike_rate_surrogate()
                    loss = loss + model.lam_sr * (r_bar - model.r_star) ** 2
                    epoch_rate_sum += r_bar.detach().item()
                    epoch_rate_n += 1
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
        val_acc, test_acc = evaluate(model, model_kind, adj, x, edge_index,
                                     y, split, num_neighbors, batch_size,
                                     device, T, full_batch,
                                     x_dev, ei_dev, y_dev, val_idx, test_idx)
        # Report the epoch-level mean surrogate firing rate (correctly
        # normalized over each forward's own batch); the old code divided the
        # last mini-batch's spike count by the full-graph N=169343 -> ~0.08x
        # under-report.
        rates.append(epoch_rate_sum / epoch_rate_n
                     if (model_kind == "spikergnn" and epoch_rate_n) else 0.0)
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc
        if not smoke and (ep + 1) % 5 == 0:
            _mem = (f" mem={torch.cuda.max_memory_allocated(device)/1e9:.1f}GiB"
                    if _cuda else "")
            print(f"  [{model_kind} r{run_id}] ep{ep+1:03d} "
                  f"val={val_acc:.4f} test={test_acc:.4f}{_mem}", flush=True)
    return {"model": model_kind, "run_id": run_id, "best_val": best_val,
            "best_test": best_test,
            "rate_mean": (sum(rates) / len(rates) if rates else 0.0),
            "peak_mem_gib": (torch.cuda.max_memory_allocated(device) / 1e9
                             if _cuda else 0.0)}


@torch.no_grad()
def evaluate(model, model_kind, adj, x, edge_index, y, split, num_neighbors,
             batch_size, device, T, full_batch=False,
             x_dev=None, ei_dev=None, y_dev=None, val_idx=None, test_idx=None):
    model.eval()
    if full_batch:
        logits = model(x_dev, ei_dev,
                       torch.zeros(x_dev.size(0), dtype=torch.long, device=device))
        val_acc = (logits[val_idx].argmax(1) == y_dev[val_idx]).float().mean().item()
        test_acc = (logits[test_idx].argmax(1) == y_dev[test_idx]).float().mean().item()
        return val_acc, test_acc
    # sampler path (memory-constrained fallback)
    rng = random.Random(999)
    out = torch.zeros(x.size(0), int(y.max().item()) + 1, device=device)
    for split_name in ("valid", "test"):
        for seeds in batched(split[split_name], batch_size, False, rng):
            x_sub, ei_sub, sub_t = sample_batch(adj, seeds, num_neighbors, x,
                                                edge_index, device, rng)
            logits = model(x_sub, ei_sub,
                           torch.zeros(x_sub.size(0), dtype=torch.long, device=device))
            out[sub_t[:len(seeds)]] = logits[:len(seeds)].detach()
    val_acc = (out[val_idx].argmax(1) == y_dev[val_idx]).float().mean().item()
    test_acc = (out[test_idx].argmax(1) == y_dev[test_idx]).float().mean().item()
    return val_acc, test_acc


# ── CUDA-graph training (full-batch SpikeRPGNN, fixed topology) ─────────────
def _cg_reset_state(model):
    """Zero persistent LIF membrane buffers in place (CUDA-graph-safe).

    lif._u_mag is a persistent registered buffer that accumulates across
    encode calls. A captured graph records its exact contents at capture time,
    so every replay must start from the SAME contents or the replayed dynamics
    diverge from the eager forward. Zeroing in place keeps the tensor address
    stable (CUDA graphs forbid address changes) and matches the eager forward,
    which begins from a zero-initialised buffer.
    """
    for mod in model.modules():
        um = getattr(mod, "_u_mag", None)
        if um is not None:
            um.zero_()


def _cg_build(model, opt, x_cg, edge_index, batch_all, train_idx, y_dev, device):
    """Warmup + capture the full training step (forward + L_sr + backward).

    Returns a handle with the captured graph and its stream. The captured step
    is: logits = model(x_cg, ...); loss = CE + lam_sr*(r_bar - r_star)^2;
    loss.backward(). Mirrors _test_cudagraph_small.py, which is verified to
    replay bit-identically to an eager step.
    """
    for mod in model.modules():
        mod._disable_internal_noise = True  # cuRNG cannot be captured
    capture_stream = torch.cuda.Stream()
    # Warmup (host-sync ALLOWED) on the capture stream: populates lazy caches
    # (_deg_hub_flag, manifolds _c_cached, episode buffers, step_states clones)
    # and the autograd graph on this stream with no legacy-default-stream
    # AccumulateGrad nodes (which would break capture with
    # cudaErrorStreamCaptureImplicit).
    with torch.cuda.stream(capture_stream):
        logits = model(x_cg, edge_index, batch_all)
        loss = F.cross_entropy(logits[train_idx], y_dev[train_idx])
        r_bar = model.spike_rate_surrogate()
        loss = loss + model.lam_sr * (r_bar - model.r_star) ** 2
        loss.backward()
    # Switch to capture mode (suppress host syncs) for the recorded pass.
    for mod in model.modules():
        mod._suppress_host_sync = True
    _cg_reset_state(model)  # deterministic zero start for the recorded graph
    opt.zero_grad(set_to_none=False)  # non-None grads so capture can accumulate
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=capture_stream):
        torch.cuda.set_stream(capture_stream)
        logits = model(x_cg, edge_index, batch_all)
        loss = F.cross_entropy(logits[train_idx], y_dev[train_idx])
        r_bar = model.spike_rate_surrogate()
        loss = loss + model.lam_sr * (r_bar - model.r_star) ** 2
        loss.backward()
    torch.cuda.synchronize()
    # Keep a handle to the captured loss/output tensors so a caller can read the
    # replayed value (e.g. for a correctness check) without re-running a forward.
    return {"g": g, "stream": capture_stream, "x_cg": x_cg, "loss": loss}


def _cg_step(cg, model, opt, x_dev, noise_std, grad_clip=0.0):
    """One training step via CUDA-graph replay with external noise injection.

    Everything the graph reads (x_cg, the membrane buffer) is (re)written on
    the SAME capture stream, in submission order, before ``replay()`` is
    launched — so there is no cross-stream race regardless of default-stream
    semantics. The cuRNG noise and the in-place zeroing are regular kernels,
    not part of the captured graph.
    """
    x_cg = cg["x_cg"]
    with torch.cuda.stream(cg["stream"]):
        if noise_std and noise_std > 0:
            # External noise: re-noise x_cg IN PLACE (same address the graph
            # reads). In-model cuRNG noise is disabled during capture, so the
            # stochasticity is injected here, on the capture stream, before
            # each replay.
            x_cg.copy_(x_dev).add_(torch.randn_like(x_cg) * noise_std)
        _cg_reset_state(model)  # deterministic membrane start for every replay
        opt.zero_grad(set_to_none=False)
        cg["g"].replay()
        if grad_clip and grad_clip > 0:
            # Clip AFTER replay (outside the captured graph): the replayed
            # backward populates .grad, so clipping reads valid gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["spikergnn", "gcn", "sage"], default="spikergnn")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full-batch", action="store_true", default=False,
                    help="full-graph forward (baselines only; SpikeRPGNN "
                         "needs >31GB for T=8 autograd graph, so sampler is "
                         "the default for it)")
    ap.add_argument("--u-threshold", type=float, default=1.0,
                    help="SpikeRPGNN membrane threshold init. ogbn-arxiv "
                         "features are small, so 1.0 -> rate~0 (dormant); "
                         "lower it to engage spiking and approach r*=0.4.")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture the full fwd+L_sr+bwd step into a CUDA graph "
                         "and replay it (removes per-kernel launch overhead that "
                         "dominated the 169k backward). SpikeRPGNN full-batch "
                         "only; ignored on CPU / mini-batch.")
    ap.add_argument("--T", type=int, default=1,
                    help="SpikeRPGNN spike-time steps (temporal dimension). "
                         "Default 1 (fast node-level protocol, matches the "
                         "published ~53%% mini-batch baseline). For 169k "
                         "full-batch the UN-checkpointed T=8 peak is ~33 GiB "
                         "(exceeds 22 GB); gradient-checkpoint (ON by default) "
                         "drops it to ~17.6 GiB so T=8 fits. Pass --T 8 "
                         "--full-batch to exercise that path.")
    ap.add_argument("--d-e", type=int, default=96,
                    help="SpikeRPGNN Euclidean-embedding dim (hidden_dim = "
                         "32 + d_e). Smaller => less memory.")
    ap.add_argument("--no-checkpoint-rgc", action="store_true",
                    help="disable gradient-checkpointing of the per-step RGC "
                         "stack (path-2 memory fix, ON by default for "
                         "SpikeRPGNN full-batch). Use only if you want to "
                         "compare against the un-checkpointed baseline.")
    ap.add_argument("--lambda-sr", type=float, default=None,
                    help="override SpikeRPGNNConfig.lambda_sr (rate-regularization "
                         "strength; default 0.5). Raise it to strengthen the pull "
                         "of L_sr toward r*=0.4 at node level.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Adam learning rate (default 1e-3). Full-batch "
                         "training gets only 1 true gradient per epoch and is "
                         "prone to threshold/firing-dynamics instability at "
                         "high LR; lower it (e.g. 3e-4) for the 169k run.")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="grad-norm clipping (default 0.0 = OFF). Suppresses "
                         "the instability-driven late-epoch collapse seen in "
                         "169k full-batch runs. Apply with e.g. --grad-clip 1.0.")
    ap.add_argument("--out", default="results/nodelevel.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_runs, args.epochs = 1, 2
        args.device = "cpu"
        args.out = "results/nodelevel_smoke.json"

    # CUDA-graph capture needs the fixed-topology full-graph path; force it.
    if args.cuda_graph:
        args.full_batch = True

    # Path-2 memory fix: gradient-checkpoint the per-step RGC stack by default
    # for SpikeRPGNN (it is a no-op with no effect on correctness or eval); the
    # --no-checkpoint-rgc flag disables it for A/B comparison.
    ckpt_rgc = (not args.no_checkpoint_rgc)

    device = torch.device(args.device)
    print(f"model={args.model} device={device} u_th={args.u_threshold} | "
          f"ogbn-arxiv node-level | checkpoint_rgc={ckpt_rgc} "
          f"lr={args.lr} grad_clip={args.grad_clip}", flush=True)

    results = []
    for r in range(args.n_runs):
        t0 = time.time()
        res = run_one(args.model, r, args.epochs, device, smoke=args.smoke,
                      full_batch=args.full_batch, u_threshold=args.u_threshold,
                      cuda_graph=args.cuda_graph, T=args.T, d_e=args.d_e,
                      checkpoint_rgc=ckpt_rgc, lr=args.lr,
                      grad_clip=args.grad_clip, lambda_sr=args.lambda_sr)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "runs": results}, f, indent=2)
        print(f"  run{r} done: {res}", flush=True)

    tests = [r["best_test"] for r in results]
    print(f"=== {args.model} node-level: test acc "
          f"mean={sum(tests)/len(tests):.4f} "
          f"min={min(tests):.4f} max={max(tests):.4f} ===", flush=True)


if __name__ == "__main__":
    main()

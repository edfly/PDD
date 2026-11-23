# -*- coding: utf-8 -*-
"""Rate-control mechanism diagnostic: surrogate-vs-empirical rate gap and a
NOISE-FREE desynchronization control.

Context
-------
A methodological review flagged that firing-rate controllability
is *trained* through the differentiable surrogate rate (tilde r) but *reported*
through the empirical hard-spike rate (r_bar), and that on MUTAG at sigma=0 the
L_sr term has no control authority because noiseless dynamics quantize to a
period-2 plateau.  This script supplies the two missing measurements:

  (1) RATE-GAP: at the reported working point (lambda_sr=0.5, noise=0.05,
      r*=0.4), measure BOTH r_bar (empirical) and tilde r (surrogate) on the
      SAME final-epoch episodes, and report their quantitative gap per run.

  (2) DESYNC CONTROL: replace per-timestep Gaussian membrane noise with a
      FIXED, per-neuron heterogeneous threshold offset (deterministic given the
      seed; NO time-varying noise).  If L_sr regains control authority at
      sigma=0 under this static heterogeneity, then controllability does not
      depend on stochastic jitter per se, but on breaking the identical-input
      period-2 lock -- which is exactly the claim we want to test.

Design
------
  * Dataset: MUTAG (the dataset where the sigma=0 failure is worst).
  * 1-shot, 2-way; 5 runs; 20 epochs x 10 episodes (same reduced protocol as
    the sigma=0 control so numbers are comparable).
  * Three arms:  A = noisy baseline (noise=0.05, uniform threshold)
                 B = sigma=0 (noise=0, uniform threshold)  [reproduce plateau]
                 C = sigma=0 + static per-neuron threshold heterogeneity
  * Every arm records, at final epoch, both empirical and surrogate rate.

The script NEVER modifies the model source.  Heterogeneity is injected by
monkey-patching a single LIF instance's effective threshold computation via a
registered buffer, applied only to arm C.

Usage
-----
  PYTHONPATH= python verify_rate_control_mechanism.py --device cuda --runs 5

Output
------
  results/rate_control_mechanism.json  (machine-readable)
  console table (human-readable)
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)          # .../post/code
_ROOT = os.path.dirname(_CODE)          # .../post
sys.path.insert(0, _CODE)

from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed, compute_accuracy
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


def _make_cfg(noise: float, lam: float, seed: int, epochs: int, episodes: int) -> SpikeRPGNNConfig:
    cfg = SpikeRPGNNConfig(
        d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
        u_threshold=1.0, noise_std=noise, ablation="full", use_dvm=True,
        lambda_sr=lam, r_target=0.4, seed_base=seed,
    )
    cfg.epochs = epochs
    cfg.episodes_per_epoch = episodes
    cfg.test_episodes = 5
    return cfg


def _inject_static_heterogeneity(model, N: int, spread: float, seed: int):
    """Give each neuron a FIXED threshold offset in [-spread, spread].

    Uses the measurement-only ``_het_th_offset`` buffer added to
    ManifoldLIFNeuron.  Default (None) leaves the published model untouched;
    here we set a static (time-invariant) per-neuron offset, so there is NO
    time-varying randomness -- only deterministic heterogeneity.
    """
    lif = model.stmp.lif
    gen = torch.Generator(device="cpu").manual_seed(seed)
    offset = (torch.rand(N, generator=gen) * 2 - 1) * spread
    lif._het_th_offset = offset.to(lif.u_th_learnable.device)


def _measure_final_rates(model, sampler, cfg, device) -> Dict[str, float]:
    """Measure both empirical (r_bar) and surrogate (tilde r) rates at eval."""
    model.eval()
    emp, sur = [], []
    with torch.no_grad():
        for _ in range(cfg.test_episodes):
            ep = sampler.sample_episode(mode="test")
            sb, sl, qb, ql = batch_episode_data(ep, device=device)
            model.stmp.lif.reset_membrane()
            model.encode(sb.x, sb.edge_index, sb.batch)
            model.encode(qb.x, qb.edge_index, qb.batch)
            emp.append(float(model.stmp.lif.spike_rate()))
            sur.append(float(model.stmp.lif.spike_rate_surrogate().item()))
    return {
        "r_empirical": float(np.mean(emp)),
        "r_surrogate": float(np.mean(sur)),
        "r_gap": float(abs(np.mean(emp) - np.mean(sur))),
    }


def run_arm(arm: str, noise: float, lam: float, het_spread: float,
            runs: int, device: str, epochs: int, episodes: int) -> Dict[str, Any]:
    ds_cfg = DATASET_CONFIGS["MUTAG"]
    rows: List[Dict[str, float]] = []
    for rid in range(runs):
        seed = 1000 + rid * 100
        set_seed(seed)
        cfg = _make_cfg(noise=noise, lam=lam, seed=seed, epochs=epochs, episodes=episodes)
        dataset = TUGraphDataset(
            name="MUTAG", tu_name=ds_cfg["tu_name"],
            train_ratio=cfg.train_ratio, seed=seed,
        )
        sampler = EpisodeSampler(
            dataset=dataset, n_way=ds_cfg["n_way"], k_shot=1,
            q_query=cfg.q_query, seed=seed,
        )
        in_dim = max(dataset.num_features, 1)
        model = create_spikergnn(cfg, in_dim=in_dim).to(device)

        if het_spread > 0:
            # 256 static per-position offsets, sliced/repeated to any episode
            # size inside the neuron (fixed in time and position => no jitter).
            _inject_static_heterogeneity(model, N=256, spread=het_spread, seed=seed)
            print(f"  [{arm}] injected static het_th_offset (256 positions, "
                  f"spread={het_spread})", flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for epoch in range(cfg.epochs):
            model.train()
            for _ in range(cfg.episodes_per_epoch):
                ep = sampler.sample_episode(mode="train")
                sb, sl, qb, ql = batch_episode_data(ep, device=device)
                opt.zero_grad(set_to_none=True)
                sz = model.encode(sb.x, sb.edge_index, sb.batch)
                qz = model.encode(qb.x, qb.edge_index, qb.batch)
                logits = model.classify(qz, sz, sl, sampler.n_way)
                loss = nn.functional.cross_entropy(logits, ql)
                if cfg.lambda_sr > 0:
                    r = model.spike_rate_surrogate()
                    loss = loss + cfg.lambda_sr * (r - cfg.r_target) ** 2
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                opt.step()

        rates = _measure_final_rates(model, sampler, cfg, device)
        rows.append(rates)
        print(f"  [{arm}] run{rid}: r_bar={rates['r_empirical']:.4f} "
              f"tilde_r={rates['r_surrogate']:.4f} gap={rates['r_gap']:.4f}", flush=True)

    return {
        "arm": arm, "noise_std": noise, "lambda_sr": lam,
        "het_spread": het_spread, "runs": runs, "rows": rows,
        "r_empirical_mean": float(np.mean([r["r_empirical"] for r in rows])),
        "r_surrogate_mean": float(np.mean([r["r_surrogate"] for r in rows])),
        "r_gap_mean": float(np.mean([r["r_gap"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--het-spread", type=float, default=0.3)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; refusing silent CPU fallback (GPU-first rule).")
    # NOTE: do NOT set torch.set_default_device('cuda').  The episode sampler
    # and PyG data pipeline create many small CPU tensors; forcing them all onto
    # the GPU makes sampling pathologically slow.  The production runner keeps
    # data on CPU and moves only the batches to device (batch_episode_data does
    # this), so we follow the same pattern.

    print(f"device={args.device} torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    arms = [
        ("A_noisy", 0.05, 0.5, 0.0),
        ("B_sigma0", 0.0, 0.5, 0.0),
        ("C_sigma0_het", 0.0, 0.5, args.het_spread),
    ]
    result: Dict[str, Any] = {
        "_protocol": {
            "dataset": "MUTAG", "shot": 1, "n_way": 2,
            "epochs": args.epochs, "episodes_per_epoch": args.episodes,
            "test_episodes": 5, "runs": args.runs, "device": args.device,
            "note": "Rate-control mechanism diagnostic (reduced protocol "
                    "for cost): empirical vs surrogate rate gap and static "
                    "(per-neuron) heterogeneous-threshold desync at sigma=0. "
                    "Directional mechanism evidence, NOT a reproduction of the "
                    "reported 50-epoch headline numbers.",
        },
        "arms": {},
    }
    t0 = time.time()
    for arm, noise, lam, het in arms:
        print(f"\n=== ARM {arm} noise={noise} lam={lam} het_spread={het} ===", flush=True)
        result["arms"][arm] = run_arm(arm, noise, lam, het, args.runs, args.device,
                                      args.epochs, args.episodes)

    out = os.path.join(_ROOT, "results", "rate_control_mechanism.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== SUMMARY (final epoch, 5 runs) ===")
    print(f"{'arm':16} {'r_bar':>8} {'tilde_r':>8} {'gap':>8}")
    for arm in result["arms"]:
        a = result["arms"][arm]
        print(f"{arm:16} {a['r_empirical_mean']:8.4f} {a['r_surrogate_mean']:8.4f} {a['r_gap_mean']:8.4f}")
    print(f"\nwrote {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

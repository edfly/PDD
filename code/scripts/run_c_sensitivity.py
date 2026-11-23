"""Curvature sensitivity scan for SpikeRPGNN (MUTAG, 1-shot).

Ported from archive_2026-07-08/run_c_sensitivity.py so that it runs against the
CURRENT implementation in code/spikergnn (the archived copy drifted in 8 files
and must not be used to produce numbers reported in the paper).

Protocol matches the original 2-run scan so that the two are directly
comparable: fixed c (learnable_c=False), 30 epochs of 20 episodes, 20 test
episodes, lambda_sr = 0.5, r_target = 0.4. Only the number of runs changes.

Usage:
  python run_c_sensitivity.py --runs 5 --c_values 0.5 1.0 2.0
  python run_c_sensitivity.py --runs 1 --epochs 1 --c_values 1.0     # smoke test

The output JSON keeps the same schema as results/c_sensitivity.json:
  { "<c>": {"c": float, "accuracies": [...], "mean": float, "std": float}, ... }
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)          # .../post/code
sys.path.insert(0, _CODE_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from msg_baseline.utils import set_seed
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn


def run_with_fixed_c(c_value, ds_name="MUTAG", shot=1, run_id=0, epochs=30,
                     ep_per_epoch=20, test_eps=20, lambda_sr=0.5, r_target=0.4,
                     lr=1e-3, device=None, verbose=False):
    """One run with curvature c held fixed. Returns the best test accuracy."""
    if device is None:
        device = torch.device("cpu")

    ds_cfg = DATASET_CONFIGS[ds_name]
    n_way = ds_cfg["n_way"]
    seed = 42 + run_id * 100 + shot * 10
    set_seed(seed)

    cfg = SpikeRPGNNConfig(
        epochs=1,
        episodes_per_epoch=ep_per_epoch,
        test_episodes=test_eps,
        lambda_sr=lambda_sr,
        r_target=r_target,
        learnable_c=False,
        c_init=c_value,
    )
    dataset = TUGraphDataset(name=ds_name, tu_name=ds_cfg["tu_name"],
                             train_ratio=0.8, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=n_way, k_shot=shot,
                             q_query=5, seed=seed)
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(cfg, in_dim=in_dim).to(device)

    u_th_params = [p for n, p in model.named_parameters() if 'u_th_learnable' in n]
    other_params = [p for n, p in model.named_parameters() if 'u_th_learnable' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': lr, 'weight_decay': 5e-4},
        {'params': u_th_params, 'lr': 1e-4, 'weight_decay': 0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    r_target_t = torch.tensor(r_target, device=device)
    best_acc = 0.0
    final_acc = 0.0

    for epoch in range(epochs):
        if epoch < 3:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * (epoch + 1) / 3.0

        model.train()
        total_acc, n_ok, n_skip = 0, 0, 0
        for _ in range(ep_per_epoch):
            try:
                episode = sampler.sample_episode(mode='train')
                s_b, s_l, q_b, q_l = batch_episode_data(episode)
                s_b, q_b = s_b.to(device), q_b.to(device)
                s_l, q_l = s_l.to(device), q_l.to(device)

                optimizer.zero_grad()
                s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                if torch.isnan(s_z[0]).any() or torch.isnan(q_z[0]).any():
                    n_skip += 1
                    continue

                logits = model.classify(q_z, s_z, s_l, n_way)
                if torch.isnan(logits).any():
                    n_skip += 1
                    continue

                loss_ce = nn.functional.cross_entropy(logits, q_l)
                if torch.isnan(loss_ce) or torch.isinf(loss_ce) or loss_ce.item() > 50:
                    n_skip += 1
                    continue

                r_bar = model.spike_rate_surrogate()
                loss = loss_ce + lambda_sr * (r_bar - r_target_t) ** 2

                loss.backward()
                if any(p.grad is not None and
                       (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                       for p in model.parameters()):
                    optimizer.zero_grad()
                    n_skip += 1
                    continue

                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                with torch.no_grad():
                    model.stmp.lif.u_th_learnable.clamp_(0.1, 5.0)
                    model.mspe.lif.u_th_learnable.clamp_(0.1, 5.0)

                total_acc += (logits.argmax(-1) == q_l).float().mean().item()
                n_ok += 1
            except Exception:
                n_skip += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            eval_accs = []
            with torch.no_grad():
                for _ in range(min(test_eps, 20)):
                    try:
                        ep = sampler.sample_episode(mode='test')
                        s_b, s_l, q_b, q_l = batch_episode_data(ep)
                        s_b, q_b = s_b.to(device), q_b.to(device)
                        s_l, q_l = s_l.to(device), q_l.to(device)
                        s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                        q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                        if torch.isnan(s_z[0]).any():
                            continue
                        logits = model.classify(q_z, s_z, s_l, n_way)
                        if torch.isnan(logits).any():
                            continue
                        eval_accs.append((logits.argmax(-1) == q_l).float().mean().item())
                    except Exception:
                        continue
            ea = sum(eval_accs) / max(len(eval_accs), 1) if eval_accs else 0.0
            best_acc = max(best_acc, ea)
            final_acc = ea          # accuracy at the last evaluated epoch
            if verbose:
                print(f"  c={model.manifold.c.item():.3f} E{epoch + 1:3d}: "
                      f"train={total_acc / max(n_ok, 1) * 100:.1f}% "
                      f"test={ea * 100:.1f}% best={best_acc * 100:.1f}% "
                      f"skip={n_skip}/{ep_per_epoch}", flush=True)

    # Both metrics are returned because they answer different questions and must
    # not be conflated:
    #   final_acc - accuracy at the final epoch, the definition used by the main
    #               protocol (Section "Experimental Setup") and by the lambda_sr
    #               sweep; this is the value the paper reports.
    #   best_acc  - maximum test accuracy over epochs. Selecting the epoch on the
    #               test split is an optimistic bias, so it is retained only for
    #               provenance and is never the headline number.
    return final_acc, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_values", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ep_per_epoch", type=int, default=20)
    ap.add_argument("--test_eps", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="output JSON (default: results/c_sensitivity.json)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the runs already in --out and continue from there. "
                         "Run indices (and hence seeds) are preserved, so a "
                         "resumed scan is identical to an uninterrupted one.")
    args = ap.parse_args()

    out = args.out or os.path.join(_CODE_DIR, os.pardir, "results", "c_sensitivity.json")
    out = os.path.normpath(out)

    # `accuracies` (and therefore `mean`/`std`) is the FINAL-EPOCH metric, i.e.
    # the definition the manuscript states for the main protocol and uses in the
    # lambda_sr sweep. `best_accuracies` is kept only for provenance.
    results = {"_protocol": {
        "epochs": args.epochs,
        "episodes_per_epoch": args.ep_per_epoch,
        "test_episodes": args.test_eps,
        "runs": args.runs,
        "dataset": "MUTAG",
        "shot": 1,
        "learnable_c": False,
        "lambda_sr": 0.5,
        "r_target": 0.4,
        "u_th_lr": 1e-4,
        "seed": "42 + run_id*100 + shot*10",
        "reported_metric": "final_acc (accuracy at the final epoch over "
                           "test_episodes test episodes)",
        "note": "Dedicated scan protocol: fewer epochs/episodes than the main "
                "protocol, so absolute levels are not comparable to the main "
                "results table; comparisons are valid only within this scan.",
    }}

    def _flush():
        """Persist whatever has been computed so far (survives interruption)."""
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if args.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            prior = json.load(f)
        for k, v in prior.items():
            if k.startswith("_") or not any(abs(float(k) - c) < 1e-9
                                            for c in args.c_values):
                continue
            results[k] = {"c": v["c"], "accuracies": list(v["accuracies"]),
                          "best_accuracies": list(v.get("best_accuracies", [])),
                          "mean": v["mean"], "std": v["std"]}
            print(f"[resume] c={float(k):.2f}: {len(v['accuracies'])} runs "
                  f"already done", flush=True)

    t0 = time.time()
    for c in args.c_values:
        key = f"{c:.3f}"
        accs = list(results.get(key, {}).get("accuracies", []))
        baccs = list(results.get(key, {}).get("best_accuracies", []))
        trun = time.time()
        for r in range(len(accs), args.runs):
            fin, bst = run_with_fixed_c(c, run_id=r, epochs=args.epochs,
                                        ep_per_epoch=args.ep_per_epoch,
                                        test_eps=args.test_eps, verbose=True)
            accs.append(float(fin) * 100.0)
            baccs.append(float(bst) * 100.0)
            results[key] = {"c": float(c), "accuracies": list(accs),
                            "best_accuracies": list(baccs),
                            "mean": float(np.mean(accs)),
                            "std": float(np.std(accs))}
            _flush()   # write after every run so a kill never loses everything
            print(f"c={c:.2f} run{r}: final={fin * 100:.2f}% best={bst * 100:.2f}%  "
                  f"({time.time() - trun:.0f}s for this run, "
                  f"{time.time() - t0:.0f}s total)", flush=True)
        print(f"[done] c={c:.2f} mean={np.mean(accs):.2f} std={np.std(accs):.2f} "
              f"n={len(accs)}", flush=True)

    _flush()
    print(f"wrote {out}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Capacity-matched MSG* baseline runner (hidden_dim configurable).

Why this exists
---------------
The archived MSG* baseline (`results/msgstar_results.json`, d=32) follows the
original MSG configuration width, whereas every method in the main table is
capacity-matched at hidden_dim=128. Using a width-32 baseline to support the
headline gaps (+22.2% on Rdt-M5k 5-shot, +11.4% on DD 5-shot) confounds
capacity with mechanism. This runner re-runs MSG* at hidden_dim=128 under the
unified main protocol so the comparison is capacity-matched, and (optionally)
re-runs width 32 under the same protocol so the capacity effect itself is
isolated rather than inferred across protocols.

Protocol (identical to the SpikeRPGNN main matrix, Section 6.1):
    epochs = 50, episodes_per_epoch = 20, test_episodes = 30
    seed = 42 + 100*run_id + 10*shot   (set inside run_single_experiment)

Single-run contract (watchdog-friendly):
    --run-id N runs exactly one (dataset, shot, run_id) and writes it to --out.
    A queue driver spawns many of these concurrently; results are merged later.

NOTE: `TUGraphDataset` loads `TUDataset(root="./data/TU", ...)` relative to the
current working directory, so this runner must be launched with cwd set to the
tree that owns `data/TU` (i.e. D:/2026/paper/PBB) to reuse the dataset cache.
"""
import sys
import os
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msg_baseline.config import get_config, DATASET_CONFIGS          # noqa: E402
from msg_baseline.train_episodic import run_single_experiment         # noqa: E402

ALL_DATASETS = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "REDDIT-M5K"]


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
    return cfg


def run_one(dataset, shot, run_id, cfg):
    res = run_single_experiment(
        dataset_name=dataset, shot=shot, run_id=run_id, config=cfg
    )
    return {
        "dataset": dataset,
        "shot": shot,
        "run_id": run_id,
        "hidden_dim": cfg.hidden_dim,
        "final_acc": float(res["accuracy"]),
        "best_acc": float(res.get("best_acc", res["accuracy"])),
        "epoch_accuracies": [float(x) for x in res.get("epoch_accuracies", [])],
        "time_seconds": float(res.get("time_seconds", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Capacity-matched MSG* runner")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--episodes-per-epoch", type=int, default=20)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--run-id", type=int, default=None,
                    help="run only this run index (queue-driver contract)")
    ap.add_argument("--out", default=None,
                    help="JSON output path (per-task file, or merged file)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.episodes_per_epoch, args.test_episodes = 3, 3, 5
        args.n_runs = 1

    cfg = build_config(args)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "spikergnn_results",
        f"msgstar_d{args.hidden_dim}_results.json",
    )

    results = {}
    if args.resume and os.path.exists(out):
        try:
            with open(out) as f:
                results = json.load(f)
        except Exception:
            results = {}

    run_ids = [args.run_id] if args.run_id is not None else list(range(args.n_runs))

    print(
        f"MSG* capacity runner | d={args.hidden_dim} | datasets={args.datasets} "
        f"| shots={args.shots} | runs={run_ids} | epochs={args.epochs} "
        f"| ep/epoch={args.episodes_per_epoch} | test_eps={args.test_episodes}",
        flush=True,
    )

    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            print(f"SKIP unknown dataset {ds}", flush=True)
            continue
        for shot in args.shots:
            for rid in run_ids:
                key = f"d{args.hidden_dim}_{ds}_{shot}shot_run{rid}"
                if args.resume and key in results and isinstance(results[key], dict) \
                        and "final_acc" in results[key]:
                    print(f"SKIP {key}", flush=True)
                    continue
                t0 = time.time()
                try:
                    r = run_one(ds, shot, rid, cfg)
                    r["elapsed_sec"] = round(time.time() - t0, 1)
                    results[key] = r
                    print(f">>> {key}: acc={r['final_acc']*100:.2f}% "
                          f"({r['elapsed_sec']}s)", flush=True)
                except Exception as e:                       # noqa: BLE001
                    traceback.print_exc()
                    results[key] = {"error": str(e), "dataset": ds,
                                    "shot": shot, "run_id": rid,
                                    "hidden_dim": args.hidden_dim}
                atomic_save_json(out, results)

    print("DONE", flush=True)
    return results


if __name__ == "__main__":
    main()

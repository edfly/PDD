# -*- coding: utf-8 -*-
"""50ep / 5-run SIX-VARIANT ABLATION backfill (task ② of the separation fix).

Why this exists:
  The original ablation study (run_enzymes_ablation.py / mutag / proteins) ran the
  6 SpikeRPGNN variants at 100ep / 3runs. That protocol disagrees with the main
  matrix (spikergnn & euc at 50ep / 5runs), producing the "100ep vs 50ep split"
  that made the geometry gain look contradictory (Euclidean-better in ablation,
  Riemannian-better in matrix). This runner re-runs the SAME 6 variants at the
  matrix protocol (50ep / 5runs) so the ablation table is internally 50ep-consistent
  and directly comparable to spikergnn/euc.

Protocol is IDENTICAL to run_spike_gnn_euc.py / the main matrix:
  epochs=50, ep/epoch=20, test_eps=30, 5 runs, seed = seed_base + run*100 + shot*10
  (run_single_experiment sets the seed; full & euclidean share seed_base -> paired).

Cells: 6 variants x {MUTAG, PROTEINS, ENZYMES} x 1-shot x 5 runs.
  (MUTAG/PROTEINS are 2-way; ENZYMES is 5-way. 1-shot only, matching the original
  ablation. 5-shot geometry comparison comes from the main matrix euc/spikergnn.)

Single-run contract (watchdog-friendly): --run-id runs exactly one
(variant, dataset, shot, run_id). --resume skips completed runs. Atomic JSON save.
"""
import sys
import os
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.ablation import get_all_ablation_configs
from spikergnn.train import run_single_experiment
from msg_baseline.config import DATASET_CONFIGS

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "spikergnn_results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "ablation_50ep_v30.json")

VARIANTS = ["full", "no_spikes", "euclidean", "no_frechet", "no_dvm", "no_meta",
            "no_stmp"]
DATASETS = ["MUTAG", "PROTEINS", "ENZYMES"]
SHOTS = [1]
N_RUNS = 5


def atomic_save_json(path, obj, retries=6, wait=10):
    tmp = path + ".tmp"
    for i in range(retries):
        try:
            with open(tmp, "w") as f:
                json.dump(obj, f, indent=2, default=str)
            os.replace(tmp, path)
            return
        except (PermissionError, OSError) as e:
            if i == retries - 1:
                raise
            time.sleep(wait)


def run_one(variant, ds_name, shot, run_id, epochs=50):
    base = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation=variant, epochs=epochs,
    )
    configs = get_all_ablation_configs(base)
    config = configs[variant]
    res = run_single_experiment(
        dataset_name=ds_name, shot=shot, run_id=run_id,
        config=config, device="cpu",
    )
    return {
        "dataset": ds_name, "shot": shot, "variant": variant,
        "run_id": run_id,
        "final_acc": float(res["final_acc"]),
        "total_spikes": int(res.get("total_spikes", 0)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--shots", nargs="+", type=int, default=SHOTS)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--run-id", type=int, default=None,
                        help="run only this run index (orchestrator contract)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-file", dest="results_file", default=None)
    args = parser.parse_args()

    if args.results_file:
        RESULTS_FILE = args.results_file

    epochs = 3 if args.quick else args.epochs
    n_runs = 1 if args.quick else args.n_runs

    results = {}
    if args.resume and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    print(f"50ep ablation: variants={args.variants} x {args.datasets} x "
          f"{args.shots} x {n_runs} runs x {epochs} epochs", flush=True)

    run_ids = ([args.run_id] if args.run_id is not None
               else list(range(n_runs)))

    for variant in args.variants:
        if variant not in VARIANTS:
            print(f"SKIP unknown variant {variant}", flush=True)
            continue
        for ds in args.datasets:
            if ds not in DATASET_CONFIGS:
                continue
            for shot in args.shots:
                for run_id in run_ids:
                    key = f"{variant}_{ds}_{shot}shot_run{run_id}"
                    if key in results and isinstance(results[key], dict) \
                            and "final_acc" in results[key]:
                        print(f"SKIP {key}", flush=True)
                        continue
                    t0 = time.time()
                    try:
                        r = run_one(variant, ds, shot, run_id, epochs=epochs)
                        r["elapsed_sec"] = round(time.time() - t0, 1)
                        results[key] = r
                        print(f">>> {key}: acc={r['final_acc']*100:.2f}% "
                              f"spikes={r['total_spikes']} ({r['elapsed_sec']}s)",
                              flush=True)
                    except Exception as e:
                        traceback.print_exc()
                        results[key] = {"error": str(e)}
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    atomic_save_json(RESULTS_FILE, results)

    print("\nDONE", flush=True)

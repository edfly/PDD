# -*- coding: utf-8 -*-
"""One-shot generator: build post/code/run_largescale.py.

Why a generator instead of hand-writing the file: the training loop is
extracted verbatim from run_full_experiments_v24.py so that the large-scale
runs cannot silently drift from the protocol that produced the published
main-matrix numbers. Only the device plumbing is patched in. Every
substitution is counted and asserted, so a rename upstream fails loudly
instead of producing a script that quietly runs on CPU.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FN = os.path.join(HERE, "scripts", "_tmp_fn.txt")

fn = io.open(FN, encoding="utf-8").read()

subs = [
    # signature: add device
    ("learnable_c=False):", 'learnable_c=False, device="cuda"):', 1),
    # model onto device
    ("model = create_spikergnn(cfg, in_dim=in_dim)",
     "model = create_spikergnn(cfg, in_dim=in_dim).to(device)", 1),
    # loss target tensor
    ("r_target_t = torch.tensor(r_target)",
     "r_target_t = torch.tensor(r_target, device=device)", 1),
    ("loss_sr = torch.tensor(0.0)",
     "loss_sr = torch.tensor(0.0, device=device)", 1),
    # episode batches onto device (two call spellings)
    ("batch_episode_data(episode)", "batch_episode_data(episode, device)", 1),
    ("batch_episode_data(ep)", "batch_episode_data(ep, device)", 2),
]

for old, new, expect in subs:
    got = fn.count(old)
    if got != expect:
        print("[abort] %r occurs %d times, expected %d" % (old, got, expect))
        sys.exit(1)
    fn = fn.replace(old, new)
print("all %d substitutions applied" % len(subs))

HEADER = '''# -*- coding: utf-8 -*-
"""Large-scale benchmark runner for SpikeRPGNN (audit extension).

Purpose
-------
The manuscript concedes that its largest benchmark (REDDIT-MULTI-5K,
4,999 graphs) is "modest by contemporary graph-learning standards". This
runner adds the two largest TU datasets that are already staged on disk
(NCI1 4,110 graphs; NCI109 4,127 graphs) under the *identical* episodic
protocol, so the new cells are comparable with the published main matrix
rather than being a second, incompatible protocol.

Design rule
-----------
The training loop below is extracted verbatim from
scripts/run_full_experiments_v24.py and only the device plumbing is added,
so a large-scale number cannot diverge from a main-matrix number through
a silently rewritten protocol. Do not edit the loop here; if the protocol
changes, regenerate.

GPU: device defaults to "cuda". The project rule is GPU-first; the original
runner was CPU-only because it never moved the model or the batches.

Usage
-----
    python code/run_largescale.py --smoke
    python code/run_largescale.py --datasets NCI1 NCI109 --shots 1 5 --n-runs 5

Output
------
    results/largescale.json   (new file; never overwrites v30 artefacts)
"""
import sys, os, json, time, argparse, traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Data lives at PBB/data (repo root), the TU loader resolves "./data/TU".
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
os.chdir(_REPO)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np

from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

# ── Large-scale TU datasets, injected at runtime ──────────────────
# Deliberately NOT written into msg_baseline/config.py: that file is read
# by the verification gates, and adding datasets there would change
# ALL_DATASETS and therefore every gate that enumerates the main matrix.
# Runtime injection keeps the published matrix untouched.
# Graph counts and class counts were measured from the staged data
# (TUDataset(root=./data/TU)), not copied from a paper.
DATASET_CONFIGS["NCI1"] = {
    "n_way": 2, "shots": [1, 5], "n_graphs": 4110,
    "n_classes": 2, "tu_name": "NCI1",
}
DATASET_CONFIGS["NCI109"] = {
    "n_way": 2, "shots": [1, 5], "n_graphs": 4127,
    "n_classes": 2, "tu_name": "NCI109",
}

RESULTS_PATH = os.path.join(_REPO, "post", "results", "largescale.json")


'''

MAIN = '''

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(rows):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def already_done(rows, ds, shot, run_id, lambda_sr):
    for r in rows:
        if (r.get("dataset") == ds and r.get("shot") == shot
                and r.get("run_id") == run_id
                and r.get("lambda_sr") == lambda_sr):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NCI1", "NCI109"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ep-per-epoch", type=int, default=20)
    ap.add_argument("--test-eps", type=int, default=30)
    ap.add_argument("--lambda-sr", type=float, default=0.5)
    ap.add_argument("--r-target", type=float, default=0.4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.datasets = [args.datasets[0]]
        args.shots = [args.shots[0]]
        args.n_runs = 1
        args.epochs = 2
        args.ep_per_epoch = 2
        args.test_eps = 3

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[abort] cuda requested but unavailable")
        sys.exit(1)
    print("device =", args.device, "|", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "cpu")

    rows = load_results()
    t0 = time.time()
    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            print("[skip] unknown dataset", ds)
            continue
        for shot in args.shots:
            for run_id in range(args.n_runs):
                if already_done(rows, ds, shot, run_id, args.lambda_sr):
                    print("[skip done] %s %d-shot run%d" % (ds, shot, run_id))
                    continue
                try:
                    out = run_experiment(
                        ds, shot, run_id, epochs=args.epochs,
                        ep_per_epoch=args.ep_per_epoch, test_eps=args.test_eps,
                        lambda_sr=args.lambda_sr, r_target=args.r_target,
                        device=args.device)
                    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    rows.append(out)
                    save_results(rows)
                    print("[ok] %s %d-shot run%d acc=%.4f r_bar=%.4f"
                          % (ds, shot, run_id, out["final_acc"],
                             out["final_spike_rate"]), flush=True)
                except Exception:
                    traceback.print_exc()
                    print("[fail] %s %d-shot run%d" % (ds, shot, run_id), flush=True)
    print("elapsed %.1f s" % (time.time() - t0))
    print("wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
'''

out = HEADER + fn.rstrip() + "\n" + MAIN
dst = os.path.join(HERE, "run_largescale.py")
io.open(dst, "w", encoding="utf-8").write(out)
print("wrote", dst, len(out), "bytes")

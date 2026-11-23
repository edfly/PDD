#!/bin/bash
# Threshold sweep (post-fix): the learnable threshold was previously HARDCODED
# to 1.0 in neurons.py, so the earlier sweep was a no-op (rate stuck at 0.0025).
# Now u_threshold is config-driven; sweep LOW values to find rate ~= 0.4 on
# ogbn-arxiv (small-magnitude node features). Each candidate is a separate
# process/json so a reclaim loses at most the in-flight candidate.
set -u
# Select the interpreter that carries torch/torch_geometric (see README).
PY="${PY:-python3}"
cd "$(dirname "$0")/.."
echo "[sweep2] start $(date)" | tee code/results/_sweep2.log
for U in 0.05 0.10 0.15 0.25; do
  echo "=== SpikeRPGNN sweep u_th=$U 1x8 ===" | tee -a code/results/_sweep2.log
  $PY code/run_nodelevel.py --model spikergnn --n-runs 1 --epochs 8 \
      --device cuda --u-threshold $U \
      --out code/results/nodelevel_sweep_u${U}.json 2>&1 | tee -a code/results/_sweep2.log
done
echo "[sweep2] DONE $(date)" | tee -a code/results/_sweep2.log

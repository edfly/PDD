#!/bin/bash
# Sequential node-level campaign on ogbn-arxiv (single GPU, serialized to
# avoid contention). Each phase writes its own json; run_one checkpoints
# per-run so a reclaim loses at most the in-flight run, not the whole phase.
set -u
# Select the interpreter that carries torch/torch_geometric (see README).
PY="${PY:-python3}"
cd "$(dirname "$0")/.."
echo "[campaign] start $(date)" | tee code/results/_campaign.log
echo "=== SpikeRPGNN 3x30 ===" | tee -a code/results/_campaign.log
$PY code/run_nodelevel.py --model spikergnn --n-runs 3 --epochs 15 --device cuda --out code/results/nodelevel.json 2>&1 | tee -a code/results/nodelevel.log
echo "=== GCN 3x50 ===" | tee -a code/results/_campaign.log
$PY code/run_nodelevel.py --model gcn --n-runs 3 --epochs 50 --device cuda --out code/results/nodelevel_gcn.json 2>&1 | tee -a code/results/nodelevel_gcn.log
echo "=== SAGE 3x50 ===" | tee -a code/results/_campaign.log
$PY code/run_nodelevel.py --model sage --n-runs 3 --epochs 50 --device cuda --out code/results/nodelevel_sage.json 2>&1 | tee -a code/results/nodelevel_sage.log
echo "[campaign] DONE $(date)" | tee -a code/results/_campaign.log

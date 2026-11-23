#!/usr/bin/env bash
# Calibrated node-level campaign AFTER the L_sr differentiability fix.
#   - SpikeRPGNN 1x15 @ u_th=1.0 (CALIBRATION: confirm firing rate is now alive)
#   - GCN  3x50, SAGE 3x50 (substrate baselines, same sampler path)
# SpikeRPGNN uses the sampler (full-batch T=8 needs >31GB). Baselines too,
# for an apples-to-apples substrate comparison.
# Select the interpreter that carries torch/torch_geometric (see README).
PY="${PY:-python3}"
cd "$(dirname "$0")"
mkdir -p results

echo "=== [$(date)] SpikeRPGNN node-level CALIBRATED (fix applied) u_th=1.0 1x15 ===" | tee -a results/calibrated.log
$PY run_nodelevel.py --model spikergnn --n-runs 1 --epochs 15 --u-threshold 1.0 \
    --out results/nodelevel_calib_spikergnn.json 2>&1 | tee -a results/calib_spikergnn.log

echo "=== [$(date)] GCN baseline 3x50 ===" | tee -a results/calibrated.log
$PY run_nodelevel.py --model gcn --n-runs 3 --epochs 50 \
    --out results/nodelevel_calib_gcn.json 2>&1 | tee -a results/calib_gcn.log

echo "=== [$(date)] SAGE baseline 3x50 ===" | tee -a results/calibrated.log
$PY run_nodelevel.py --model sage --n-runs 3 --epochs 50 \
    --out results/nodelevel_calib_sage.json 2>&1 | tee -a results/calib_sage.log

echo "=== [$(date)] CALIBRATED CAMPAIGN DONE ===" | tee -a results/calibrated.log

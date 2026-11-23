# ogbn-arxiv Node-Level Results (SpikeRPGNN adaptation)

Campaign task `9Gjr4G` (2026-10-23 16:06 → 19:04, ~3 h, survived reclaim).
Driver: `code/run_nodelevel_campaign.sh`. Raw data: `nodelevel.json`,
`nodelevel_gcn.json`, `nodelevel_sage.json` (all in `code/results/`).

## Head-to-head on the SAME node split (ogbn-arxiv, 40 classes)

| Model | test acc (3 runs) | val acc | realized firing rate |
|---|---|---|---|
| **SpikeRPGNN** (3×15, T=8, u_th=1.0) | **0.540 ± 0.003** | 0.605 | **0.0025 (0.25%)** ❗ |
| GCN (3×50) | 0.541 ± 0.003 | 0.595 | 0 (Euclidean baseline) |
| SAGE (3×50) | 0.552 ± 0.003 | 0.610 | 0 (Euclidean baseline) |

Per-run test acc:
- SpikeRPGNN: 0.5367 / 0.5411 / 0.5429  (mean 0.5402)
- GCN:          0.5414 / 0.5382 / 0.5434  (mean 0.5410)
- SAGE:         0.5534 / 0.5483 / 0.5534  (mean 0.5517)

## Two findings

1. **No geometric accuracy advantage.** SpikeRPGNN (54.0%) ≈ GCN (54.1%) and
   trails SAGE (55.2%). Consistent with the paper's thesis that the Riemannian
   product manifold is a *substrate*, not an accuracy device.

2. **⚠ Firing-rate collapse (critical).** Realized rate = 0.25%, far below the
   calibrated target r* = 0.4. The spiking / rate-control mechanism was
   **dormant** at u_threshold = 1.0 (threshold too high for ogbn-arxiv feature
   scale — the same root cause the paper debugged on graph tasks, replayed on
   node-level). Therefore this run is NOT valid evidence for the
   "rate-calibrated spiking" claim; it is only a *dormant-regime* accuracy
   comparison (which still shows no geometric edge).

## Recommended next step
Re-run SpikeRPGNN node-level with a **calibrated threshold** (sweep
u_threshold ∈ {0.3, 0.5, 0.7}) so the realized rate ≈ 0.4, to obtain a
genuinely rate-controlled geometric comparison. Only then does the experiment
support the paper's rate-calibrated-spiking thesis and answer the Nature
review's "is geometry just a substrate?" question cleanly.

## Config notes (de-risk)
- 2-hop custom sampler (no pyg-lib); subgraph ≈ 8% of graph per 1024-seed batch.
- Full-batch training infeasible for SpikeRPGNN: T=8 autograd graph on 169k
  nodes needs >31 GB (no_grad forward is only 3.6 GB).
- SpikeRPGNN ≈ 300 s/epoch (5 min); GCN/SAGE ≈ 8 s/epoch on same sampler.
- `run_one` checkpoints per run → reclaim loses at most the in-flight run.

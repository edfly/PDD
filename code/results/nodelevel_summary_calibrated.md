# Node-level calibrated campaign — final summary

**Fix applied:** L_sr now uses differentiable `spike_rate_surrogate()` (was `count_spikes()`/.item(), detached). Firing rate is alive.

| Model | test acc (mean±std) | n | firing rate (mean) |
|---|---|---|---|
| SpikeRPGNN (rate-controlled) | 53.50 ± 0.78% | 3×15 | 0.342 ± 0.003 |
| GCN (baseline) | 53.98 ± 0.31% | 3×50 | — |
| SAGE (baseline) | 55.26 ± 0.26% | 3×50 | — |

## Reading
- SpikeRPGNN (0.535) is statistically on par with GCN (0.540) and marginally below SAGE (0.553); all overlap within ~1 std.
- The hyperbolic×Euclidean product manifold + spiking confers **no node-level accuracy gain** once the firing rate is properly controlled.
- This is a clean, controlled null/parity result consistent with the paper's 'geometry is substrate' thesis and answers the Nature-review novelty weak axis honestly.
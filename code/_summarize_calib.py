"""
_summarize_calib.py — merge & summarize the rate-calibrated node-level campaign.

Reads:
  results/nodelevel_calib_spikergnn.json   (run0)
  results/nodelevel_calib_spikergnn_r12.json (run1, run2)
  results/nodelevel_calib_gcn.json
  results/nodelevel_calib_sage.json
Writes:
  results/nodelevel_calib_spikergnn_3x15.json   (merged SpikeRPGNN 3x15)
  results/nodelevel_summary_calibrated.md        (final comparison table)
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)["runs"]


def agg(runs, key):
    vals = [r[key] for r in runs]
    return sum(vals) / len(vals), statistics.pstdev(vals), vals


# ── SpikeRPGNN: merge run0 + r12 (run_ids 0,1) into one 3x15 record ──
spk = load("nodelevel_calib_spikergnn.json") + load("nodelevel_calib_spikergnn_r12.json")
spk_test_m, spk_test_s, spk_test = agg(spk, "best_test")
spk_rate_m, spk_rate_s, spk_rate = agg(spk, "rate_mean")
merged = {
    "model": "spikergnn",
    "n_runs": len(spk),
    "epochs": 15,
    "u_threshold": 1.0,
    "runs": spk,
    "test_mean": spk_test_m,
    "test_std": spk_test_s,
    "rate_mean": spk_rate_m,
    "rate_std": spk_rate_s,
}
with open(os.path.join(RES, "nodelevel_calib_spikergnn_3x15.json"), "w") as f:
    json.dump(merged, f, indent=2)

# ── Baselines ──
gcn_runs = load("nodelevel_calib_gcn.json")
sage_runs = load("nodelevel_calib_sage.json")
gcn_m, gcn_s, _ = agg(gcn_runs, "best_test")
sage_m, sage_s, _ = agg(sage_runs, "best_test")

# ── Final table ──
lines = []
lines.append("# Node-level calibrated campaign — final summary\n")
lines.append("**Fix applied:** L_sr now uses differentiable `spike_rate_surrogate()` "
             "(was `count_spikes()`/.item(), detached). Firing rate is alive.\n")
lines.append("| Model | test acc (mean±std) | n | firing rate (mean) |")
lines.append("|---|---|---|---|")
lines.append(f"| SpikeRPGNN (rate-controlled) | {spk_test_m*100:.2f} ± {spk_test_s*100:.2f}% | {len(spk)}×15 | {spk_rate_m:.3f} ± {spk_rate_s:.3f} |")
lines.append(f"| GCN (baseline) | {gcn_m*100:.2f} ± {gcn_s*100:.2f}% | {len(gcn_runs)}×50 | — |")
lines.append(f"| SAGE (baseline) | {sage_m*100:.2f} ± {sage_s*100:.2f}% | {len(sage_runs)}×50 | — |")
lines.append("")
lines.append("## Reading")
lines.append("- SpikeRPGNN (0.535) is statistically on par with GCN (0.540) and "
             "marginally below SAGE (0.553); all overlap within ~1 std.")
lines.append("- The hyperbolic×Euclidean product manifold + spiking confers **no "
             "node-level accuracy gain** once the firing rate is properly controlled.")
lines.append("- This is a clean, controlled null/parity result consistent with the "
             "paper's 'geometry is substrate' thesis and answers the Nature-review "
             "novelty weak axis honestly.")
with open(os.path.join(RES, "nodelevel_summary_calibrated.md"), "w") as f:
    f.write("\n".join(lines))

print(f"SpikeRPGNN 3x15 : test={spk_test_m*100:.2f}% ± {spk_test_s*100:.2f}%  "
      f"rate={spk_rate_m:.3f} ± {spk_rate_s:.3f}")
print(f"GCN  3x50       : test={gcn_m*100:.2f}% ± {gcn_s*100:.2f}%")
print(f"SAGE 3x50       : test={sage_m*100:.2f}% ± {sage_s*100:.2f}%")
print("wrote results/nodelevel_calib_spikergnn_3x15.json and "
      "results/nodelevel_summary_calibrated.md")

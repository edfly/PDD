"""
merge_nodelevel_8run.py — Merge the original 3-run node-level campaign
(code/results/nodelevel_calib_spikergnn_3x15.json) with the new 5-run
reproduction (results/nodelevel_calib_spikergnn_3x15_extra5.json) into an
n=8 aggregate, recompute mean/std, and emit a merged JSON.

Both source files share the SAME protocol: model=spikergnn, epochs=15,
u_threshold=1.0, T=1, sampler path, default lambda_sr/r_target, device=cuda.
Only the run count grows (3 -> 8), tightening std and letting the 53.50%
point estimate be cross-checked against an independent 5-run block.

Usage:
  python code/merge_nodelevel_8run.py
"""
import json
import os
import sys

_POST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(_POST, "results", "nodelevel_calib_spikergnn_3x15.json")
EXTRA = os.path.join(_POST, "results", "nodelevel_calib_spikergnn_3x15_extra5.json")
OUT = os.path.join(_POST, "results", "nodelevel_calib_spikergnn_8run.json")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    var = _mean([(x - m) ** 2 for x in xs])
    return var ** 0.5


def main():
    orig = _load(ORIG)
    extra = _load(EXTRA)
    runs = list(orig.get("runs", [])) + list(extra.get("runs", []))
    if not runs:
        print("ERROR: no runs found", file=sys.stderr)
        return 1
    tests = [r["best_test"] for r in runs]
    rates = [r["rate_mean"] for r in runs if "rate_mean" in r]
    m = {
        "model": "spikergnn",
        "n_runs": len(runs),
        "epochs": orig.get("epochs", 15),
        "u_threshold": orig.get("u_threshold", 1.0),
        "runs": runs,
        "test_mean": _mean(tests),
        "test_std": _std(tests),
        "rate_mean": _mean(rates) if rates else 0.0,
        "rate_std": _std(rates) if rates else 0.0,
    }
    with open(OUT, "w") as f:
        json.dump(m, f, indent=2)
    print(f"n_runs={m['n_runs']}  test_mean={m['test_mean']*100:.2f}%  "
          f"test_std={m['test_std']*100:.2f}%  "
          f"rate_mean={m['rate_mean']:.3f}  rate_std={m['rate_std']:.3f}")
    print(f"individual best_test (%): " +
          ", ".join(f"{t*100:.2f}" for t in tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())

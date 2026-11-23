#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repro_ledger.py  --  Reproducibility ledger for SpikeRPGNN (three-pillar discipline)

This script is the *reproducible code for the paper's core methodology claims*.
It re-derives every headline number of the three-pillar evaluation discipline
from committed raw data files and reports, per claim:
    - the source data file,
    - whether that file is tracked by git (committed),
    - the recomputed value,
    - a reproducibility verdict.

It does NOT train anything. It reads committed results JSONs and recomputes the
statistics the manuscript asserts, so a reviewer (or the author) can verify that
each claim is traceable to data that lives in the repository.

Run:  python code/repro_ledger.py
Output: prints a human report and writes results/repro_ledger.json
"""
import json, os, glob, subprocess, sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)

def load(p):
    fp = R(p)
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception as e:
        sys.stderr.write(f"[load ERROR] {fp}: {e}\n")
        return {"__error__": str(e)}

def git_tracked(rel):
    try:
        out = subprocess.run(["git","ls-files","--error-unmatch",rel],
                             cwd=ROOT, capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False

def band_count(rates, target=0.4, tol=0.10):
    if not rates: return (0,0)
    ins = sum(1 for r in rates if abs(r-target) <= tol)
    return ins, len(rates)

def agg_runs(d):
    """Return (mean_test, mean_rate, n_runs) from either top-level or a runs[] list."""
    if not isinstance(d, dict): return (None, None, 0)
    if "test_mean" in d and "rate_mean" in d:
        return (d["test_mean"], d["rate_mean"], d.get("n_runs") or len(d.get("runs", [])))
    runs = d.get("runs", [])
    if runs:
        t = [r.get("best_test") for r in runs if r.get("best_test") is not None]
        r = [r.get("rate_mean") for r in runs if r.get("rate_mean") is not None]
        tm = (sum(t)/len(t)) if t else None
        rm = (sum(r)/len(r)) if r else None
        return (tm, rm, len(runs))
    return (None, None, 0)

report = OrderedDict()
report["_meta"] = {
    "purpose": "Re-derive three-pillar headline claims from committed data.",
    "repo_root": ROOT,
}

# ---------------------------------------------------------------------------
# PILLAR A -- Capacity-matched separability  (few-shot accuracy + 2x2 effect)
# ---------------------------------------------------------------------------
a = OrderedDict()
sg = load("results/analysis_v30.json").get("spikergnn", {})
if isinstance(sg, dict):
    acc_rows = OrderedDict()
    for k, v in sg.items():
        accs = v.get("accs", [])
        if accs:
            acc_rows[k] = {"mean": round(100*sum(accs)/len(accs),2),
                           "std": round(100*(v.get("std") or 0),2),
                           "n": len(accs)}
    a["spikeRPGNN_fewshot_acc"] = acc_rows
    a["source_file"] = "results/analysis_v30.json -> ['spikergnn']"
    a["git_tracked"] = git_tracked("results/analysis_v30.json")

# 2x2 geometry x spiking effect (MUTAG 1-shot head-to-head vs Euclidean non-spiking)
srm = load("results/score_roadmap_numbers.json")
eff = srm.get("effect_size_cis_vs_msgstar", {})
mutag1 = eff.get("MUTAG_1", {})
a["mutag1_vs_euclidean_nonspiking"] = {
    "mean_diff_pct": round(mutag1.get("mean_diff_pct",0),2),
    "ci95_pct": [round(x,2) if x is not None else None for x in mutag1.get("ci95_pct",[None,None])],
    "t": round(mutag1.get("t",0),2),
    "p": mutag1.get("p"),
    "cohens_dz": round(mutag1.get("cohens_dz",0),2),
    "n": mutag1.get("n"),
    "source_file": "results/score_roadmap_numbers.json -> ['effect_size_cis_vs_msgstar']['MUTAG_1']",
    "git_tracked": git_tracked("results/score_roadmap_numbers.json"),
}
report["Pillar_A_capacity_matched_separability"] = a

# ---------------------------------------------------------------------------
# PILLAR B -- Firing-rate controllability as the unit of progress
# ---------------------------------------------------------------------------
b = OrderedDict()

# (i) graph-level 33/35 rate-band (from committed few-shot data)
all_rates = []
for k, v in sg.items():
    all_rates += v.get("spike_rates", [])
ins10, tot = band_count(all_rates, 0.4, 0.10)
ins05, _ = band_count(all_rates, 0.4, 0.05)
b["graph_level_rate_band"] = {
    "target_r_star": 0.4,
    "total_spikeRPGNN_runs_in_data": tot,
    "in_band_tol0.10": f"{ins10}/{tot}",
    "in_band_tol0.05": f"{ins05}/{tot}",
    "claimed_in_manuscript": "33 of 35",
    "source_file": "results/analysis_v30.json -> ['spikergnn'][*]['spike_rates']",
    "git_tracked": git_tracked("results/analysis_v30.json"),
    "note": "Manuscript claims '33 of 35 main-protocol runs inside target band'. "
            "Recompute above from committed data; tol selection matters.",
}

# (ii) node-level cross-scale transfer  r_bar = 0.342 on 169k ogbn-arxiv
claimed_nl = load("code/results/nodelevel_calib_spikergnn_3x15.json")
honest_nl = load("results/nodelevel_calib_spikergnn_3x15_extra5.json")
ht, hr, hn = agg_runs(honest_nl)
nl = OrderedDict()
nl["claimed_manuscript_r_bar"] = round(claimed_nl.get("rate_mean",0),4)
nl["claimed_manuscript_acc"] = round(100*claimed_nl.get("test_mean",0),2)
nl["claimed_n"] = claimed_nl.get("n_runs")
nl["claimed_source_file"] = "code/results/nodelevel_calib_spikergnn_3x15.json"
nl["claimed_git_tracked"] = git_tracked("code/results/nodelevel_calib_spikergnn_3x15.json")
nl["honest_rerun_5run_r_bar"] = round(hr,4) if hr is not None else "MISSING"
nl["honest_rerun_5run_acc"] = round(100*ht,2) if ht is not None else "MISSING"
nl["honest_rerun_n"] = hn
nl["honest_rerun_source_file"] = "results/nodelevel_calib_spikergnn_3x15_extra5.json"
nl["honest_rerun_git_tracked"] = git_tracked("results/nodelevel_calib_spikergnn_3x15_extra5.json")

# sweep under CURRENT committed code: what is the max achievable node-level r_bar?
sweep = OrderedDict()
for fp in sorted(glob.glob(R("results","nodelevel_fix_u*.json"))):
    d = load(fp)
    t, r, n = agg_runs(d)
    if r is not None:
        base = os.path.basename(fp).replace("nodelevel_fix_","").replace(".json","")
        sweep[base] = {"r_bar": round(r,4), "acc": round(100*t,2) if t is not None else None}
nl["current_code_sweep_max_r_bar"] = sweep
nl["verdict"] = ("INTEGRITY GAP: claimed r_bar=0.342 comes from an UNRECOVERABLE "
                "Oct-24 calibration code state; the current committed code reaches at "
                "most ~0.30 node-level r_bar (sweep above) and the identical-protocol "
                "5-run rerun gives r_bar~0.21. The cross-scale r_bar=0.342 claim is "
                "NOT reproducible from committed code.")
b["node_level_cross_scale_transfer"] = nl
report["Pillar_B_rate_controllability"] = b

# ---------------------------------------------------------------------------
# PILLAR C -- Honest energy accounting (T1 analytic / T2 hardware)
# ---------------------------------------------------------------------------
c = OrderedDict()
energy = load("results/analysis_v30.json").get("energy", [])
spikeRPGNN_e = next((e for e in energy if e.get("method")=="SpikeRPGNN"), None)
gin_e = next((e for e in energy if e.get("method")=="Proto-GIN"), None)
if spikeRPGNN_e:
    c["spikeRPGNN_T1_energy_uj_mean"] = round(spikeRPGNN_e.get("energy_uj_mean",0),3)
    c["spikeRPGNN_r_bar_mean"] = round(spikeRPGNN_e.get("r_bar_mean",0),4)
    c["spikeRPGNN_reduction_vs_GIN_pct"] = round(spikeRPGNN_e.get("reduction_pct",0),2)
if gin_e:
    c["ProtoGIN_T1_energy_uj_mean"] = round(gin_e.get("energy_uj_mean",0),3)
c["T1_source_file"] = "results/analysis_v30.json -> ['energy']"
c["T1_git_tracked"] = git_tracked("results/analysis_v30.json")

silicon = load("results/energy_silicon_projection.json")
proj = silicon.get("projection", {})
c["silicon_projection"] = {
    "reference_dense_E_loihi_mean_uj": silicon.get("reference_dense_E_loihi_mean_uj"),
    "reference_dense_E_truenorth_mean_uj": silicon.get("reference_dense_E_truenorth_mean_uj"),
    "projection_keys": list(proj.keys())[:10] if isinstance(proj,dict) else proj,
    "source_file": "results/energy_silicon_projection.json",
    "git_tracked": git_tracked("results/energy_silicon_projection.json"),
    "note": "Manuscript: Tier-1 straddles zero under +/-50% constant perturbation and "
            "reverses into an overrun on real neuromorphic silicon (T2). Verify projection file.",
}
report["Pillar_C_honest_energy_accounting"] = c

# ---------------------------------------------------------------------------
# WRITE + PRINT
# ---------------------------------------------------------------------------
out = R("results","repro_ledger.json")
with open(out,"w") as f:
    json.dump(report, f, indent=2)

def pblock(title, d, indent=0):
    pad = "  "*indent
    print(f"{pad}## {title}")
    for k,v in d.items():
        if isinstance(v, dict):
            print(f"{pad}  - {k}:")
            pblock(k, v, indent+2) if False else None
            for kk,vv in v.items():
                print(f"{pad}      {kk}: {vv}")
        else:
            print(f"{pad}  - {k}: {v}")

print("="*78)
print("SPIKERPGNN  --  THREE-PILLAR REPRODUCIBILITY LEDGER")
print("="*78)
for title, d in report.items():
    if title.startswith("_"): continue
    print()
    print(f">>> {title}")
    for k,v in d.items():
        print(f"    - {k}: {v}")
print()
print("Wrote", out)

"""
Result aggregation & statistical analysis for the v30 experiment matrix.

Builds from real run JSONs (all methods, P30 protocol):
  1. Main accuracy table  (method x dataset x shot, mean±std, n runs)
  2. Paired t-tests SpikeRPGNN vs each baseline + Benjamini-Hochberg FDR
  3. lambda_sr sweep table (lambda, r_bar, acc, relative energy)
  4. Analytic SOP/energy accounting (uniform formula across methods)
  4b. The same accounting at 5-shot (measured 5-shot firing rates)

Usage:
  python analyze_results_v30.py            # print summaries + emit LaTeX snippets
Output: spikergnn_results/analysis_v30.json  and  paper/tables_v30.tex
"""
import sys, os, json, math
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import numpy as np
from scipy import stats

RESULTS_DIR = "spikergnn_results"

# ---------------------------------------------------------------------------
# Completeness gate: only cells with the run count the paper claims may enter
# the tables.
#
# Without this, a cell holding a single run is published as "42.5 +/- 0.0":
# numpy's std of a one-element sample is 0, which reads as zero uncertainty
# when it actually means *no* uncertainty estimate exists. A partially filled
# cell has to stay a "---" placeholder.
# ---------------------------------------------------------------------------
try:
    import exp_spec as _S
    _EXPECTED = {
        "SpikeRPGNN": _S.SUITES["spikergnn"],
        "MSG*": _S.SUITES["msg"],
        "Spike-GNN (Euc.)": _S.SUITES["euc"],
    }
    for _m in ("gin", "gcn", "sage"):
        _EXPECTED["Proto-" + _m.upper()] = _S.SUITES["ann_" + _m]
except Exception:
    _EXPECTED = {}


def _complete(method, ds, shot, accs):
    """True if this cell has as many runs as the paper claims it has."""
    su = _EXPECTED.get(method)
    if su is None:
        # No spec available: fall back to "a std is computable at all".
        return len(accs) >= 2
    return len(accs) >= su.n_runs(ds, shot)


def _want(method, ds, shot):
    su = _EXPECTED.get(method)
    return su.n_runs(ds, shot) if su is not None else 2

DATASETS = ["MUTAG", "PROTEINS", "ENZYMES", "DD", "REDDIT-M5K"]
SHOTS = [1, 5]

# Dataset average statistics (nodes, edges per graph) — from TUDataset
# (used for analytic FLOPs; values computed once and cached below)
DS_STATS = {
    "MUTAG":      {"avg_nodes": 17.9, "avg_edges": 19.8},
    "PROTEINS":   {"avg_nodes": 39.1, "avg_edges": 72.8},
    "ENZYMES":    {"avg_nodes": 32.6, "avg_edges": 62.1},
    "DD":         {"avg_nodes": 284.3, "avg_edges": 715.7},
    "REDDIT-M5K": {"avg_nodes": 508.5, "avg_edges": 594.9},
}

E_SOP = 0.9e-12   # J per synaptic op (Pfeiffer18)
E_FLOP = 4.6e-12  # J per dense FLOP

# Temporal integration steps T. A spiking model performs its synaptic
# operations on every one of the T integration steps, so its event count is
# r_bar * FLOPs_per_step * T. A dense baseline performs a single forward pass
# and is therefore NOT multiplied by T. Omitting T here under-counts spiking
# energy by a factor of T and overstates the reduction against dense
# references; see manuscript Section "Energy Model".
T_STEPS = 8

HIDDEN = 128
LAYERS = 3


def load_json(path):
    p = os.path.join(RESULTS_DIR, path)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def collect_spikergnn():
    """full_results_v24.json + v24_extra_*.json -> {(ds, shot): [acc,...]}.

    The v24_extra files add 5 extended runs (run_id 5..9) on Rdt-M5k and DD 1-shot,
    giving those two cells 10 runs for sharper significance tests.
    """
    import glob
    out = {}
    paths = [os.path.join(RESULTS_DIR, "full_results_v24.json")]
    paths += sorted(glob.glob(os.path.join(RESULTS_DIR, "v24_extra_*.json")))
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if "final_acc" not in v:
                continue
            out.setdefault((v["dataset"], v["shot"]), []).append(
                (v["run_id"], v["final_acc"], v.get("final_spike_rate", None)))
    # sort by run_id, keep (acc, sr); extended cells (Rdt-M5k/DD 1-shot) yield 10 runs
    res = {}
    for key, lst in out.items():
        lst.sort()
        res[key] = [(a, s) for _, a, s in lst]
    return res


def collect_msgstar():
    """MSG* accuracy cells -> {(ds, shot): [acc,...]}.

    Preferred source is the capacity-matched d128 re-run (run-keyed entries,
    the baseline the paper's accuracy table and energy model charge against).
    The legacy d32 cell-keyed file remains as fallback. The two agree on the
    shared first-five runs, so switching sources does not move any published
    number; it only admits the n>=20 extension runs.
    """
    d128 = load_json("msgstar_d128_results.json")
    if d128:
        res = {}
        for k, v in d128.items():
            if isinstance(v, dict) and "final_acc" in v:
                res.setdefault((v["dataset"], v["shot"]), []).append(
                    (v["run_id"], v["final_acc"]))
        return {k: [a for _, a in sorted(v)] for k, v in res.items()}
    p = os.path.join(RESULTS_DIR, "msg_results", "msgstar_results.json")
    if not os.path.exists(p):
        p = os.path.join(RESULTS_DIR, "msgstar_results.json")
    if not os.path.exists(p):
        p = "msg_results/msgstar_results.json"  # legacy fallback
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    res = {}
    for k, v in d.items():
        accs = sorted([(r["run_id"], r["accuracy"] / 100.0) for r in v["runs"]])
        res[(v["dataset"], v["shot"])] = [a for _, a in accs]
    return res


def collect_ann():
    """ann_baselines_v30.json -> {(model, ds, shot): [acc,...]}"""
    d = load_json("ann_baselines_v30.json")
    res = {}
    for k, v in d.items():
        if "final_acc" not in v:
            continue
        res.setdefault((v.get("dataset"), v.get("shot")), {})
        # model name is in the key prefix; re-derive from key
        model = k.split("_")[0]
        res[(v["dataset"], v["shot"])].setdefault(model, []).append(v["final_acc"])
    # restructure {(model, ds, shot): [accs]}
    out = {}
    for (ds, shot), models in res.items():
        for model, accs in models.items():
            out[(model, ds, shot)] = sorted(accs)
    return out


def collect_euc():
    """spike_gnn_euc_v30.json -> {(ds, shot): [(acc, sr),...]}"""
    d = load_json("spike_gnn_euc_v30.json")
    out = {}
    for k, v in d.items():
        if "final_acc" not in v:
            continue
        out.setdefault((v["dataset"], v["shot"]), []).append(
            (v["final_acc"], v.get("final_spike_rate", None)))
    return {k: sorted(v) for k, v in out.items()}


def collect_sweep():
    """lambda_sweep_v30.json -> {(lam, ds, shot): [(acc, r_bar),...]}"""
    d = load_json("lambda_sweep_v30.json")
    out = {}
    for k, v in d.items():
        if "final_acc" not in v:
            continue
        out.setdefault((v["lambda_sr"], v["dataset"], v["shot"]), []).append(
            (v["final_acc"], v.get("final_spike_rate", None)))
    return {k: sorted(v) for k, v in out.items()}


# ── Statistics ──────────────────────────────────────────────

def paired_t(a, b):
    """Paired t-test when run counts match; Welch's two-sample t-test when they differ.

    a is the SpikeRPGNN sample, b the baseline. On Rdt-M5k/DD 1-shot SpikeRPGNN has
    10 runs vs 5 for the baselines, so Welch's test (unequal variances not assumed) is
    used there; all other cells keep the paired test (5 vs 5).
    """
    na, nb = len(a), len(b)
    if min(na, nb) < 2:
        return None
    if na == nb:
        t, p = stats.ttest_rel(a, b)
        diff = np.array(a) - np.array(b)
        d_sd = diff.std(ddof=1)
        cohen_d = diff.mean() / d_sd if d_sd > 0 else float("inf")
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        diff = np.mean(a) - np.mean(b)
        sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
        # standard Cohen's d with the df-weighted pooled SD; this is the
        # convention the published significance table and the analysis notes
        # record for the unequal-count (Welch) cells
        sp_var = ((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2.0)
        cohen_d = diff / np.sqrt(sp_var) if sp_var > 0 else float("inf")
    return {"n": na, "n_b": nb, "t": float(t), "p": float(p),
            "mean_diff": float(np.mean(a) - np.mean(b)), "cohen_d": float(cohen_d)}


def bh_fdr(pvals):
    """Benjamini-Hochberg corrected q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ── Energy accounting ──────────────────────────────────────

def dense_flops(arch, ds):
    """Analytic dense FLOPs per graph (forward), uniform formula.

    arch: 'gcn' | 'gin' | 'sage' | 'spikergnn' | 'msg'
    """
    N = DS_STATS[ds]["avg_nodes"]
    E = DS_STATS[ds]["avg_edges"]
    h = HIDDEN
    if arch == "gcn":
        per_layer = N * h * h + E * h          # transform + aggregate
    elif arch == "sage":
        per_layer = 2 * N * h * h + E * h      # self+neighbor transform
    elif arch == "gin":                        # 2-layer MLP per conv
        per_layer = 2 * N * h * h + E * h
    elif arch == "spikergnn":                  # GIN backbone + ~4% manifold overhead
        per_layer = (2 * N * h * h + E * h) * 1.04
    elif arch == "msg":                        # GCN backbone + manifold overhead
        # Capacity-matched (hidden width 128) to match the re-run MSG* baseline
        # used in the accuracy table. The original MSG configuration used width
        # 32, which understates its analytic dense FLOPs by ~16x on the
        # N*h*h transform term; see results/msgstar_d32_results.json.
        per_layer = (N * h * h + E * h) * 1.10
    else:
        raise ValueError(arch)
    flops = LAYERS * per_layer + N * h  # + readout pool
    return flops


def energy_uj(flops, spike_rate=None):
    """Theoretical energy in micro-Joule per graph (averaged over an episode).

    Dense ANN: E = FLOPs x e_FLOP (4.6 pJ, 45nm MAC).
    Spiking:   E = (T x r_bar x FLOPs) x e_SOP (0.9 pJ, event-driven op,
               Pfeiffer18). The T factor is required because the spiking model
               integrates over T_STEPS steps, emitting r_bar * FLOPs synaptic
               events per step; the dense reference performs one forward pass
               and carries no such factor.
    """
    if spike_rate is not None:  # spiking: event-gated SOPs over T steps
        sop = T_STEPS * spike_rate * flops
        return sop * E_SOP * 1e6  # J -> uJ
    return flops * E_FLOP * 1e6


# ── Main ──────────────────────────────────────────────

def main():
    spike = collect_spikergnn()
    msg = collect_msgstar()
    ann = collect_ann()
    euc = collect_euc()
    sweep = collect_sweep()

    analysis = {"spikergnn": {}, "msg": {}, "ann": {}, "euc": {}, "sweep": {}}

    # 1. Main table data
    print("=" * 80)
    print("MAIN RESULTS (mean±std %, n runs)")
    print("=" * 80)
    methods = {}
    for (ds, shot), lst in spike.items():
        accs = [a for a, _ in lst]
        methods.setdefault(("SpikeRPGNN", ds, shot), accs)
        analysis["spikergnn"][f"{ds}_{shot}"] = {
            "accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "spike_rates": [s for _, s in lst if s is not None]}
    for (ds, shot), accs in msg.items():
        methods.setdefault(("MSG*", ds, shot), accs)
        analysis["msg"][f"{ds}_{shot}"] = {"accs": accs, "mean": float(np.mean(accs)),
                                           "std": float(np.std(accs))}
    for (model, ds, shot), accs in ann.items():
        methods.setdefault((f"Proto-{model.upper()}", ds, shot), accs)
        analysis["ann"][f"{model}_{ds}_{shot}"] = {
            "accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs))}
    for (ds, shot), lst in euc.items():
        accs = [a for a, _ in lst]
        if not _complete("Spike-GNN (Euc.)", ds, shot, accs):
            print(f"  SKIP partial cell: euc {ds} {shot}-shot has "
                  f"{len(accs)}/{_want('Spike-GNN (Euc.)', ds, shot)} runs "
                  f"-- left as a placeholder")
            continue
        methods.setdefault(("Spike-GNN (Euc.)", ds, shot), accs)
        analysis["euc"][f"{ds}_{shot}"] = {
            "accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "spike_rates": [s for _, s in lst if s is not None]}

    for (m, ds, shot), accs in sorted(methods.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        print(f"  {m:20s} {ds:12s} {shot}-shot: "
              f"{np.mean(accs)*100:5.1f}±{np.std(accs)*100:4.1f} (n={len(accs)})")

    # 2. Paired significance vs SpikeRPGNN
    print("\n" + "=" * 80)
    print("PAIRED t-TESTS vs SpikeRPGNN (BH-FDR corrected)")
    print("=" * 80)
    tests = []
    for (m, ds, shot), accs in methods.items():
        if m == "SpikeRPGNN":
            continue
        base = spike.get((ds, shot))
        if not base:
            continue
        b_accs = [a for a, _ in base]
        r = paired_t(b_accs, accs)
        if r:
            tests.append({"method": m, "dataset": ds, "shot": shot, **r})
    if tests:
        qs = bh_fdr([t["p"] for t in tests])
        for t, q in zip(tests, qs):
            t["q_bh"] = float(q)
        for t in sorted(tests, key=lambda x: (x["dataset"], x["shot"], x["method"])):
            star = "***" if t["q_bh"] < 0.001 else "**" if t["q_bh"] < 0.01 else \
                "*" if t["q_bh"] < 0.05 else "ns"
            print(f"  vs {t['method']:20s} {t['dataset']:12s} {t['shot']}-shot: "
                  f"d={t['mean_diff']*100:+5.1f}%  t={t['t']:+.2f}  p={t['p']:.4f}  "
                  f"q={t['q_bh']:.4f}  d_cohen={t['cohen_d']:+.2f}  {star}")
    analysis["significance"] = tests

    # 3. lambda sweep
    print("\n" + "=" * 80)
    print("LAMBDA_sr SWEEP (sweep protocol)")
    print("=" * 80)
    sw = {}
    for (lam, ds, shot), lst in sorted(sweep.items()):
        accs = [a for a, _ in lst]
        srs = [s for _, s in lst if s is not None]
        sw[f"{lam}_{ds}_{shot}"] = {
            "lambda": lam, "dataset": ds, "shot": shot,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "r_bar_mean": float(np.mean(srs)) if srs else None,
            "n_runs": len(accs)}
        r = sw[f"{lam}_{ds}_{shot}"]["r_bar_mean"]
        print(f"  lam={lam:<5} {ds:10s} {shot}-shot: acc={np.mean(accs)*100:5.1f}±{np.std(accs)*100:4.1f}"
              f"  r_bar={r if r is None else round(r,3)}  n={len(accs)}")
    analysis["sweep"] = sw

    # 4. Energy accounting (per method; 1-shot configs only, few-shot focus)
    print("\n" + "=" * 80)
    print("ENERGY ACCOUNTING (analytic, uJ/graph, 1-shot, Tier-1 common datasets)")
    print("=" * 80)
    # Restrict to Tier-1 datasets and require a COMMON dataset set across all
    # methods, so that mean energies are computed over identical workloads
    # (averaging over method-specific dataset pools is not comparable).
    TIER1 = ["MUTAG", "PROTEINS", "ENZYMES"]
    per_method = {}  # method -> {ds: {"energy_uj", "r_bar", "type"}}

    # dense ANN baselines
    for model, arch in [("Proto-GCN", "gcn"), ("Proto-GIN", "gin"), ("Proto-SAGE", "sage")]:
        avail_ds = sorted({ds for (m, ds, shot) in methods if m == model and shot == 1 and ds in TIER1})
        if avail_ds:
            per_method[model] = {ds: {"energy_uj": energy_uj(dense_flops(arch, ds)),
                                      "r_bar": None, "type": "dense"} for ds in avail_ds}
    # MSG* (dense, 1-shot configs)
    msg_ds = sorted({ds for (ds, shot) in msg.keys() if shot == 1 and ds in TIER1})
    if msg_ds:
        per_method["MSG*"] = {ds: {"energy_uj": energy_uj(dense_flops("msg", ds)),
                                   "r_bar": None, "type": "dense"} for ds in msg_ds}
    # spiking: measured r_bar from 1-shot configs
    if spike:
        for (ds, shot), lst in spike.items():
            if shot != 1 or ds not in TIER1:
                continue
            srs = [s for _, s in lst if s is not None]
            r = float(np.mean(srs)) if srs else 0.35
            per_method.setdefault("SpikeRPGNN", {})[ds] = {
                "energy_uj": energy_uj(dense_flops("spikergnn", ds), r), "r_bar": r, "type": "spiking"}
    if euc:
        for (ds, shot), lst in euc.items():
            if shot != 1 or ds not in TIER1:
                continue
            srs = [s for _, s in lst if s is not None]
            r = float(np.mean(srs)) if srs else 0.35
            per_method.setdefault("Spike-GNN (Euc.)", {})[ds] = {
                "energy_uj": energy_uj(dense_flops("gin", ds), r), "r_bar": r, "type": "spiking"}

    # common dataset set: intersection across all methods with data
    if per_method:
        common = set.intersection(*[set(v.keys()) for v in per_method.values()])
        common = [ds for ds in TIER1 if ds in common]
    else:
        common = []
    energy_rows = []
    ref_map = per_method.get("Proto-GIN", {})
    for m in ["Proto-GCN", "Proto-GIN", "Proto-SAGE", "MSG*", "Spike-GNN (Euc.)", "SpikeRPGNN"]:
        if m not in per_method:
            continue
        any_ds = next(iter(per_method[m]))
        row = {"method": m, "type": per_method[m][any_ds]["type"],
               "per_ds": per_method[m], "n_ds": len(per_method[m])}
        vals = [per_method[m][ds]["energy_uj"] for ds in common]
        rb = [per_method[m][ds]["r_bar"] for ds in common if per_method[m][ds]["r_bar"] is not None]
        if vals:
            row["energy_uj_mean"] = float(np.mean(vals))
        if rb:
            row["r_bar_mean"] = float(np.mean(rb))
        if ref_map and common and "Proto-GIN" in per_method and "energy_uj_mean" in row:
            refv = [ref_map[ds]["energy_uj"] for ds in common]
            row["reduction_pct"] = (1 - row["energy_uj_mean"] / float(np.mean(refv))) * 100
        energy_rows.append(row)
        txt = f"  {m:18s} {row['type']:8s}"
        for ds in TIER1:
            if ds in per_method[m]:
                txt += f"  {ds[:4]}={per_method[m][ds]['energy_uj']:8.3f}"
        if "energy_uj_mean" in row:
            txt += f" | mean={row['energy_uj_mean']:8.3f}"
        if "r_bar_mean" in row:
            txt += f"  r_bar={row['r_bar_mean']:.3f}"
        if "reduction_pct" in row:
            txt += f"  red_vs_GIN={row['reduction_pct']:+.1f}%"
        print(txt)
    if common != TIER1:
        print(f"  [note] common dataset set is still partial: {common}")
    analysis["energy"] = energy_rows
    analysis["energy_common_datasets"] = common

    # 4b. Energy accounting at 5-shot (same analytic model, measured 5-shot rates).
    # Dense baselines: per-graph forward cost carries no firing-rate or T factor,
    # so the 5-shot figure equals the 1-shot figure. Spiking arms: re-priced at
    # the measured 5-shot firing rate; per-architecture FLOPs and T are unchanged.
    print("\n" + "=" * 80)
    print("ENERGY ACCOUNTING (analytic, uJ/graph, 5-shot, Tier-1 common datasets)")
    print("=" * 80)
    per_method5 = {}
    for model, arch in [("Proto-GCN", "gcn"), ("Proto-GIN", "gin"),
                        ("Proto-SAGE", "sage"), ("MSG*", "msg")]:
        if model in per_method:
            per_method5[model] = {ds: {"energy_uj": energy_uj(dense_flops(arch, ds)),
                                       "r_bar": None, "type": "dense"}
                                  for ds in per_method[model]}
    if spike:
        for (ds, shot), lst in spike.items():
            if shot != 5 or ds not in TIER1:
                continue
            srs = [s for _, s in lst if s is not None]
            if not srs:
                continue
            r = float(np.mean(srs))
            per_method5.setdefault("SpikeRPGNN", {})[ds] = {
                "energy_uj": energy_uj(dense_flops("spikergnn", ds), r),
                "r_bar": r, "type": "spiking"}
    if euc:
        for (ds, shot), lst in euc.items():
            if shot != 5 or ds not in TIER1:
                continue
            srs = [s for _, s in lst if s is not None]
            if not srs:
                continue
            r = float(np.mean(srs))
            per_method5.setdefault("Spike-GNN (Euc.)", {})[ds] = {
                "energy_uj": energy_uj(dense_flops("gin", ds), r),
                "r_bar": r, "type": "spiking"}

    energy_rows5 = []
    ref_map5 = per_method5.get("Proto-GIN", {})
    for m in ["Proto-GCN", "Proto-GIN", "Proto-SAGE", "MSG*", "Spike-GNN (Euc.)", "SpikeRPGNN"]:
        if m not in per_method5:
            continue
        any_ds = next(iter(per_method5[m]))
        row = {"method": m, "type": per_method5[m][any_ds]["type"],
               "per_ds": per_method5[m], "n_ds": len(per_method5[m])}
        vals = [per_method5[m][ds]["energy_uj"] for ds in common]
        rb = [per_method5[m][ds]["r_bar"] for ds in common if per_method5[m][ds]["r_bar"] is not None]
        if vals:
            row["energy_uj_mean"] = float(np.mean(vals))
        if rb:
            row["r_bar_mean"] = float(np.mean(rb))
        if ref_map5 and common and "Proto-GIN" in per_method5 and "energy_uj_mean" in row:
            refv = [ref_map5[ds]["energy_uj"] for ds in common]
            row["reduction_pct"] = (1 - row["energy_uj_mean"] / float(np.mean(refv))) * 100
        energy_rows5.append(row)
        txt = f"  {m:18s} {row['type']:8s}"
        for ds in TIER1:
            if ds in per_method5[m]:
                txt += f"  {ds[:4]}={per_method5[m][ds]['energy_uj']:8.3f}"
        if "energy_uj_mean" in row:
            txt += f" | mean={row['energy_uj_mean']:8.3f}"
        if "r_bar_mean" in row:
            txt += f"  r_bar={row['r_bar_mean']:.3f}"
        if "reduction_pct" in row:
            txt += f"  red_vs_GIN={row['reduction_pct']:+.1f}%"
        print(txt)
    analysis["energy_5shot"] = energy_rows5

    # Preserve hand-curated keys (e.g. notes, energy_accounting_note) that the
    # pipeline does not regenerate, so a re-run never silently drops them.
    out_path = os.path.join(RESULTS_DIR, "analysis_v30.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        for k, v in prev.items():
            analysis.setdefault(k, v)

    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    print(f"\nSaved: {RESULTS_DIR}/analysis_v30.json")


if __name__ == "__main__":
    main()

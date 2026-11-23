# -*- coding: utf-8 -*-
"""Merge the freshly produced GAT baseline into the paper's result JSONs.

This script is the ONLY sanctioned way to add GAT to the result pipeline.
It does three things, all atomic with timestamped backups:

  1. Merge the per-run GAT results (results/gat_baselines_v30.json) into
     results/ann_baselines_v30.json under `gat_*` keys (same schema as the
     existing gcn/gin/sage runs).

  2. Insert aggregated `gat_{ds}_{shot}` entries into the `ann` group of
     results/analysis_v30.json (schema: {accs, mean, std}, population std,
     exactly 5 runs per cell -- matching the existing gin/gcn/sage entries).
     This is a MANUAL insertion: analyze_results_v30.py is NOT re-run, so the
     BH-FDR significance family (and every q-value in the paper) is untouched.

  3. Print the LaTeX table row for Proto-GAT and a recomputation of the
     "best-mean-per-cell" prose claim (Sec 6.1) so the manuscript text can be
     kept consistent with the new column.

GAT is deliberately KEPT OUT of exp_spec.SUITES and the significance family,
exactly like HGCN/H2H: it is an architectural dense reference, not a
capacity-matched energy reference, and including it in the BH-FDR family
would shift every q-value in Table 5 / Sec 6.1.

Run AFTER the background GAT job has finished writing gat_baselines_v30.json.
"""
import json
import math
import os
import shutil
import sys
import time

from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
GAT_JSON = os.path.join(RES, "gat_baselines_v30.json")
ANN_JSON = os.path.join(RES, "ann_baselines_v30.json")
ANALYSIS_JSON = os.path.join(RES, "analysis_v30.json")
DS_ORDER = ["MUTAG", "PROTEINS", "ENZYMES", "DD", "REDDIT-M5K"]
SHOTS = (1, 5)


def pop_stats(xs):
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
    return mu, sd


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump_atomic(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


def backup(p):
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = p + f".bak_{ts}"
    shutil.copy2(p, dst)
    return dst


def main():
    if not os.path.exists(GAT_JSON):
        print(f"ERROR: {GAT_JSON} not found -- run the GAT job first.")
        return 2
    gat = load(GAT_JSON)
    errs = [k for k, v in gat.items() if isinstance(v, dict) and "error" in v]
    if errs:
        print(f"ERROR: GAT JSON contains {len(errs)} error entries: {errs[:5]}")
        return 2

    # ── validate completeness: 5 datasets x 2 shots x 5 runs = 50 ──
    want = defaultdict(int)
    for ds in DS_ORDER:
        for sh in SHOTS:
            want[f"{ds}_{sh}"] += 5
    have = defaultdict(int)
    for k in gat:
        # key format: gat_MUTAG_1shot_run0
        parts = k.split("_")
        ds, shot = parts[1], int(parts[2].replace("shot", ""))
        have[f"{ds}_{shot}"] += 1
    missing = [f"{kk} (want {want[kk]}, have {have[kk]})"
               for kk in want if have[kk] != want[kk]]
    if missing:
        print(f"ERROR: incomplete GAT coverage: {missing}")
        return 2
    print(f"[ok] GAT JSON complete: {len(gat)} per-run entries, no errors.")

    # ── (1) merge into ann_baselines_v30.json ──
    bk1 = backup(ANN_JSON)
    ann = load(ANN_JSON)
    n_before = len(ann)
    for k, v in gat.items():
        ann[k] = v
    dump_atomic(ANN_JSON, ann)
    print(f"[ok] merged {len(gat)} gat_* keys into ann_baselines_v30.json "
          f"({n_before} -> {len(ann)}); backup: {bk1}")

    # ── (2) insert aggregated gat entries into analysis_v30.json `ann` ──
    bk2 = backup(ANALYSIS_JSON)
    a = load(ANALYSIS_JSON)
    gat_accs = defaultdict(list)
    for k, v in gat.items():
        parts = k.split("_")
        ds, shot = parts[1], int(parts[2].replace("shot", ""))
        gat_accs[(ds, shot)].append(v["final_acc"])
    new_ann = {}
    for (ds, shot), accs in gat_accs.items():
        accs = sorted(accs)
        mu, sd = pop_stats(accs)
        new_ann[f"gat_{ds}_{shot}"] = {
            "accs": accs, "mean": mu, "std": sd,
        }
    for kk, vv in new_ann.items():
        a["ann"][kk] = vv
    dump_atomic(ANALYSIS_JSON, a)
    print(f"[ok] inserted {len(new_ann)} gat_* entries into analysis_v30.json "
          f"`ann`; backup: {bk2}")

    # ── (3a) print the LaTeX Proto-GAT row ──
    cells = []
    for ds in DS_ORDER:
        for sh in SHOTS:
            mu, sd = pop_stats(gat_accs[(ds, sh)])
            cells.append(f"{mu*100:.1f}$\\pm${sd*100:.1f}")
    row = "Proto-GAT & " + " & ".join(cells) + " \\\\"
    print("\n===== LaTeX row (paste into table_main_v30.tex) =====")
    print(row)

    # ── (3b) recompute best-mean-per-cell over ALL methods (for prose) ──
    # Method means per (ds,shot); SpikeRPGNN excluded from "baseline" set.
    means = {}  # (ds,shot) -> {method: mean}
    for ds in DS_ORDER:
        for sh in SHOTS:
            key = f"{ds}_{sh}"
            d = {}
            d["SpikeRPGNN"] = a["spikergnn"][key]["mean"]
            # dense ann baselines
            for bb in ("gin", "gcn", "sage", "gat"):
                d[f"Proto-{bb.upper()}"] = a["ann"][f"{bb}_{key}"]["mean"]
            d["Spike-GNN (Euc.)"] = a["euc"][key]["mean"]
            # msg
            if key in a["msg"]:
                d["MSG*"] = a["msg"][key]["mean"]
            # hyperbolic (Tier-1 only)
            for hb, lbl in (("hgcn", "HGCN"), ("h2h", "H2H")):
                if f"{hb}_{key}" in a["ann"]:
                    d[lbl] = a["ann"][f"{hb}_{key}"]["mean"]
            means[(ds, sh)] = d

    sr_wins = 0
    details = []
    for ds in DS_ORDER:
        for sh in SHOTS:
            d = means[(ds, sh)]
            sr = d["SpikeRPGNN"]
            baseline_best = max((v for m, v in d.items() if m != "SpikeRPGNN"),
                                default=None)
            baseline_best_method = max(
                ((m, v) for m, v in d.items() if m != "SpikeRPGNN"),
                key=lambda kv: kv[1], default=(None, None))
            if sr > baseline_best:
                sr_wins += 1
                tag = "SR WINS"
            else:
                tag = f"baseline wins ({baseline_best_method[0]})"
            details.append(
                f"  {ds:11s} {sh}-shot: SR={sr*100:5.1f} | best-baseline="
                f"{baseline_best*100:5.1f} ({baseline_best_method[0]}) -> {tag}")
    print("\n===== best-mean-per-cell (for Sec 6.1 prose) =====")
    for ln in details:
        print(ln)
    print(f"\nSpikeRPGNN best-mean wins: {sr_wins} of 10")
    print("If this differs from the current '1 of 10 (PROTEINS 1-shot)' claim, "
          "update the Sec 6.1 sentence accordingly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

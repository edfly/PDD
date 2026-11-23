# -*- coding: utf-8 -*-
"""Fix the sign-reversed Proto-GAT significance entries in analysis_v30.json.

Root cause (audited 2026-10-21): the 8 Proto-GAT significance entries were
inserted manually into `analysis_v30.json["significance"]` by the GAT merge
step, which deliberately does NOT re-run analyze_results_v30.py (so the
BH-FDR family and every q-value stay untouched). The manual insertion used
the convention mean(baseline) - mean(SpikeRPGNN), while the sanctioned
pipeline (analyze_results_v30.py::paired_t) uses
mean(SpikeRPGNN) - mean(baseline). All 8 entries are therefore sign-flipped.

For a (paired or Welch) two-sided t-test the sign of mean_diff carries the
sign of t and cohen_d, while the two-sided p is invariant; BH ranking is
over the p-values, so q_bh is invariant too. This script nevertheless
RECOMPUTES everything from the raw per-run accs (using the very same
paired_t() logic imported from analyze_results_v30.py) and asserts that the
recomputed |t| and p match the stored values before writing, so any other
discrepancy would abort the run instead of silently propagating.

What is written: method/dataset/shot/n/p/q_bh are preserved verbatim;
mean_diff / t / cohen_d are replaced by the recomputed values.
"""
import json
import math
import os
import shutil
import sys
import time

import numpy as np
from scipy import stats

# this file lives in code/scripts/, so the project root is three levels up
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS_JSON = os.path.join(ROOT, "results", "analysis_v30.json")

DS_ORDER = ["MUTAG", "PROTEINS", "ENZYMES", "DD", "REDDIT-M5K"]
SHOTS = (1, 5)


def paired_t(a, b):
    """Verbatim copy of analyze_results_v30.py::paired_t (sanctioned convention)."""
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
        cohen_d = diff / np.sqrt((sa2 + sb2) / 2.0) if (sa2 + sb2) > 0 else float("inf")
    return {"n": na, "t": float(t), "p": float(p),
            "mean_diff": float(np.mean(a) - np.mean(b)), "cohen_d": float(cohen_d)}


def main():
    with open(ANALYSIS_JSON, encoding="utf-8") as f:
        a = json.load(f)

    sig = a["significance"]
    gat_idx = [i for i, s in enumerate(sig) if s.get("method") == "Proto-GAT"]
    if len(gat_idx) != 8:
        print(f"ABORT: expected 8 Proto-GAT significance entries, found {len(gat_idx)}")
        return 2

    print(f"{'cell':22s} {'stored md':>10s} {'recomp md':>10s} {'stored t':>9s} "
          f"{'recomp t':>9s} {'p ok':>5s} {'verdict':>8s}")
    changed = 0
    for i in gat_idx:
        s = sig[i]
        ds, shot = s["dataset"], s["shot"]
        sr = a["spikergnn"][f"{ds}_{shot}"]["accs"]
        gat = a["ann"][f"gat_{ds}_{shot}"]["accs"]
        r = paired_t(sr, gat)
        if r is None:
            print(f"ABORT: cannot recompute {ds} {shot}-shot")
            return 2
        # guard: |t| and p must already agree; only the sign may be wrong
        t_ok = abs(abs(r["t"]) - abs(s["t"])) < 1e-6
        p_ok = abs(r["p"] - s["p"]) < 1e-9
        n_ok = r["n"] == s["n"]
        verdict = "FLIP" if (t_ok and p_ok and n_ok) else "ABORT"
        print(f"{ds + ' ' + str(shot) + '-shot':22s} {s['mean_diff']:+10.4f} "
              f"{r['mean_diff']:+10.4f} {s['t']:+9.4f} {r['t']:+9.4f} "
              f"{str(p_ok and n_ok):>5s} {verdict:>8s}")
        if not (t_ok and p_ok and n_ok):
            print("ABORT: recomputation does not match stored |t|/p/n -- "
                  "investigate before writing.")
            return 2
        if (abs(r["mean_diff"] - s["mean_diff"]) > 1e-12
                or abs(r["t"] - s["t"]) > 1e-12
                or abs(r["cohen_d"] - s["cohen_d"]) > 1e-12):
            changed += 1
        sig[i] = {"method": s["method"], "dataset": ds, "shot": shot,
                  "n": r["n"], "t": r["t"], "p": r["p"],
                  "mean_diff": r["mean_diff"], "cohen_d": r["cohen_d"],
                  "q_bh": s["q_bh"]}

    if changed == 0:
        print("No changes needed -- all 8 entries already follow the "
              "mean(SpikeRPGNN) - mean(baseline) convention.")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = ANALYSIS_JSON + f".bak_gatsign_{ts}"
    shutil.copy2(ANALYSIS_JSON, dst)
    tmp = ANALYSIS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2)
    os.replace(tmp, ANALYSIS_JSON)
    print(f"\n[ok] rewrote {changed} Proto-GAT entries in {ANALYSIS_JSON}")
    print(f"[ok] backup: {dst}")
    print("[ok] q_bh preserved verbatim (BH ranking over two-sided p is "
          "sign-invariant); p preserved; n preserved.")

    # quick post-check: every GAT entry now satisfies
    # mean_diff == mean(SR accs) - mean(GAT accs)
    with open(ANALYSIS_JSON, encoding="utf-8") as f:
        a2 = json.load(f)
    bad = 0
    for s in a2["significance"]:
        if s.get("method") != "Proto-GAT":
            continue
        ds, shot = s["dataset"], s["shot"]
        md = (np.mean(a2["spikergnn"][f"{ds}_{shot}"]["accs"])
              - np.mean(a2["ann"][f"gat_{ds}_{shot}"]["accs"]))
        if abs(md - s["mean_diff"]) > 1e-9:
            bad += 1
    print(f"[{'ok' if bad == 0 else 'FAIL'}] post-check: {bad} GAT entries "
          f"still violate the convention (expect 0)")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

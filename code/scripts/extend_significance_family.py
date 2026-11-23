#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extend_significance_family.py
=============================
Enlarges the capacity-matched significance family from 58 to 60 cells and puts
the effect-size column on a single stated convention.

TWO DEFECTS FIXED
-----------------
1. Exclusion rule was self-contradictory.
   Ten of the 58 cells are Welch tests, run precisely because SpikeRPGNN has
   10 runs against a baseline's 5 there (DD / Rdt-M5k 1-shot, five baselines).
   Proto-GAT's two such cells were nevertheless excluded on the grounds that
   "no paired test is defined" -- but the same is true of the other ten, which
   were retained under Welch. Dropping them made the correction family two
   cells smaller and therefore every surviving q-value slightly more
   favourable, which is the opposite of what the paper says it does. Both
   cells are now computed under the identical Welch convention and included.

2. The "d" column mixed two effect-size definitions.
   Paired cells store Cohen's d_z (mean difference over its own SD). The
   Welch cells stored mean difference over sqrt((s_a^2 + s_b^2)/2), which is
   not the standard Cohen's d (that uses the degree-of-freedom-weighted
   pooled SD). The Welch cells now carry the standard pooled value; the
   paired cells keep d_z, and the column header says which is which.

Everything else -- p-values, t statistics, mean differences, paired d_z -- is
left exactly as computed by analyze_results_v30.py. The BH step-up is
recomputed over the enlarged family and cross-checked against scipy.

Exit 0 on success. The script is idempotent: re-running it detects that the
family is already 60 and does nothing.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control as _fdc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P_JSON = os.path.join(ROOT, "results", "analysis_v30.json")

# The two cells that were missing: baseline key in `ann`, dataset key in
# `spikergnn`, and the method label used in the significance table.
MISSING = [
    # (ann_key, dataset_label, method_label)
    ("gat_DD_1", "DD", "Proto-GAT"),
    ("gat_REDDIT-M5K_1", "REDDIT-M5K", "Proto-GAT"),
]
UNEQUAL_DATASETS = {"DD", "REDDIT-M5K"}


def bh_stepup(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    a = p[order] * m / np.arange(1, m + 1)
    a = np.minimum.accumulate(a[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(a, 0.0, 1.0)
    return out


def pooled_cohens_d(a, b):
    """Standard Cohen's d for two independent samples (pooled SD, df-weighted)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp2 = ((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2)
    d = (np.mean(a) - np.mean(b)) / np.sqrt(sp2) if sp2 > 0 else float("inf")
    return float(d)


def main() -> int:
    with open(P_JSON, encoding="utf-8") as fh:
        A = json.load(fh)

    sig = A["significance"]
    sp = A["spikergnn"]
    ann = A["ann"]

    if len(sig) == 60:
        print("family is already 60 cells; nothing to do")
        return 0
    if len(sig) != 58:
        print(f"FAILURE: expected 58 cells, found {len(sig)}")
        return 1

    # ── 1. recompute the effect size of the ten Welch cells ────────────
    n_fixed = 0
    for t in sig:
        if t["n"] != 10:
            continue                      # paired cell -> keep d_z
        if t["dataset"] not in UNEQUAL_DATASETS:
            continue
        a = sp[f"{t['dataset']}_{t['shot']}"]["accs"]
        if t["method"] == "Spike-GNN (Euc.)":
            # the Euclidean spiking control lives in its own block
            b = A["euc"][f"{t['dataset']}_{t['shot']}"]["accs"]
        elif t["method"] == "MSG*":
            b = A["msg"][f"{t['dataset']}_{t['shot']}"]["accs"]
        else:
            base_key = {"Proto-GCN": "gcn", "Proto-GIN": "gin",
                        "Proto-SAGE": "sage", "Proto-GAT": "gat"}.get(t["method"])
            if base_key is None:
                continue
            b = ann[f"{base_key}_{t['dataset']}_{t['shot']}"]["accs"]
        new_d = pooled_cohens_d(a, b)
        if abs(new_d - t["cohen_d"]) > 1e-9:
            print(f"  d {t['method']:<16} {t['dataset']:<11} {t['shot']}-shot : "
                  f"{t['cohen_d']:+.3f} -> {new_d:+.3f} (pooled)")
            t["cohen_d"] = new_d
            n_fixed += 1

    # ── 2. add the two missing Proto-GAT cells ─────────────────────────
    added = []
    for ann_key, ds, method in MISSING:
        a = sp[f"{ds}_1"]["accs"]
        b = ann[ann_key]["accs"]
        assert len(a) == 10 and len(b) == 5, (ann_key, len(a), len(b))
        res = stats.ttest_ind(a, b, equal_var=False)
        added.append({
            "method": method,
            "dataset": ds,
            "shot": 1,
            "n": len(a),
            "t": float(res.statistic),
            "p": float(res.pvalue),
            "mean_diff": float(np.mean(a) - np.mean(b)),
            "cohen_d": pooled_cohens_d(a, b),
            "q_bh": None,                  # filled in below
        })
        print(f"  + {method} {ds} 1-shot : n=10/5  diff={added[-1]['mean_diff']*100:+.2f}pp  "
              f"t={added[-1]['t']:.3f}  p={added[-1]['p']:.4f}  d={added[-1]['cohen_d']:+.3f}")

    sig.extend(added)

    # ── 3. recompute BH over the enlarged family ───────────────────────
    p_all = np.array([t["p"] for t in sig], dtype=float)
    q = bh_stepup(p_all)
    q_ref = np.asarray(_fdc(p_all, method="bh"), dtype=float)
    if not np.allclose(q, q_ref, atol=1e-12):
        print("FAILURE: local BH disagrees with scipy; refusing to write")
        return 1
    for t, qq in zip(sig, q):
        t["q_bh"] = float(qq)

    A["significance"] = sig
    if "notes" not in A or not isinstance(A.get("notes"), dict):
        A["notes"] = {}
    A["notes"]["significance_family"] = (
        "Family enlarged from 58 to 60 cells so that Proto-GAT's two "
        "10-versus-5-run cells (DD and Rdt-M5k, 1-shot) are treated exactly "
        "like the other baselines' unequal-count cells, i.e. by Welch's test "
        "rather than by exclusion. BH-FDR recomputed over all 60. Effect "
        "sizes: paired cells report Cohen's d_z; the 12 Welch cells report "
        "standard Cohen's d with the df-weighted pooled SD."
    )

    with open(P_JSON, "w", encoding="utf-8") as fh:
        json.dump(A, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    n_sig_new = int((q < 0.05).sum())
    print(f"\nwrote {P_JSON}")
    print(f"  family size        : 60")
    print(f"  Welch d recomputed : {n_fixed}")
    print(f"  scipy cross-check  : OK")
    print(f"  significant at .05 : {n_sig_new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

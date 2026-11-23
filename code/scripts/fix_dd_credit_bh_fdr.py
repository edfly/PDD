#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_dd_credit_bh_fdr.py
=======================
Recomputes the Benjamini--Hochberg q-values of the DD credit-assignment study
and regenerates its LaTeX table.

WHY THIS EXISTS
---------------
The audit found that ``results/dd_credit_assignment_signif.json`` does not
carry standard BH q-values: the step-up stage is missing, so a group of tied
p-values is scaled by m/rank of the *first* member only. The effect is not
cosmetic -- it inflated the smallest q from 0.0195 to 0.0586 and turned four
genuinely significant comparisons into "ns", which is what let L1166 claim
"no scheme attains a significant advantage after correction".

WHAT IT CHANGES
---------------
  * q_bh_fdr  : recomputed with the standard BH step-up procedure
  * verdict   : recomputed from the new q at the 0.05 threshold
Raw p-values, means, standard deviations, Hedges' g and the run count are all
left EXACTLY as stored -- nothing about the experiments is touched.

The recomputation is cross-checked against ``scipy.stats
.false_discovery_control``; the script refuses to write if the two disagree.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    from scipy.stats import false_discovery_control as _fdc
except Exception:                                            # pragma: no cover
    _fdc = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(ROOT, "results", "dd_credit_assignment_signif.json")
TEX_PATH = os.path.join(ROOT, "manuscript", "tables", "table_dd_signif_v30.tex")

ALPHA = 0.05
BOLD_PAIR = ("Full BPTT", "DvM M=8 truncated")   # the paper's chosen estimator


def bh_stepup(p, m=None):
    """Standard Benjamini-Hochberg adjusted p-values (step-up + monotonicity)."""
    p = np.asarray(p, dtype=float)
    m = len(p) if m is None else m
    order = np.argsort(p)
    ps = p[order]
    adj = ps * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]      # enforce monotonicity
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    with open(JSON_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    rows = data["rows"]
    p_raw = np.array([r["p_wilcoxon"] for r in rows], dtype=float)
    q_old = np.array([r["q_bh_fdr"] for r in rows], dtype=float)

    m = len(p_raw)
    q_new = bh_stepup(p_raw, m)

    # --- independent cross-check ------------------------------------------
    if _fdc is not None:
        q_ref = np.asarray(_fdc(p_raw, method="bh"), dtype=float)
        if not np.allclose(q_new, q_ref, atol=1e-12):
            print("FAILURE: local BH disagrees with scipy; refusing to write.")
            print("  local:", q_new)
            print("  scipy:", q_ref)
            return 1
        print(f"cross-check: matches scipy.stats.false_discovery_control "
              f"(m={m})")
    else:
        print("WARNING: scipy unavailable -- no independent cross-check")

    # --- write back --------------------------------------------------------
    flips = 0
    for r, q in zip(rows, q_new):
        new_verdict = "SIGNIFICANT" if q < ALPHA else "NOT_SIGNIFICANT"
        if new_verdict != r["verdict"]:
            flips += 1
        r["q_bh_fdr"] = float(q)
        r["verdict"] = new_verdict

    data["bh_fdr_note"] = (
        "q_bh_fdr recomputed with the standard Benjamini-Hochberg step-up "
        "procedure (cross-checked against scipy.stats.false_discovery_control). "
        "The previously stored q-values omitted the step-up stage; raw "
        "p-values, means, SDs and Hedges' g are unchanged."
    )
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    n_sig = int((q_new < ALPHA).sum())
    print(f"\nrewrote {JSON_PATH}")
    print(f"  family size          : {m}")
    print(f"  min q  {q_old.min():.4f} -> {q_new.min():.4f}")
    print(f"  significant at 0.05  : 0 -> {n_sig}")
    print(f"  verdict flips        : {flips}")
    print("\n  %-44s %-8s %-8s %-8s %s" %
          ("pair", "p", "q_old", "q_new", "verdict"))
    for r, qo, qn in zip(rows, q_old, q_new):
        print("  %-44s %-8s %-8s %-8s %s%s" %
              (r["A"] + " vs " + r["B"], fmt(r["p_wilcoxon"]), fmt(qo),
               fmt(qn), r["verdict"],
               "   <-- CHANGED" if (qo < ALPHA) != (qn < ALPHA) else ""))

    write_table(data, q_new)
    return 0


# ---------------------------------------------------------------- table ----
def write_table(data: dict, q) -> None:
    """Regenerate table_dd_signif_v30.tex from the corrected q-values."""
    rows = data["rows"]
    n_sig = int((q < ALPHA).sum())
    m = len(rows)

    cap = (
        "Paired exact Wilcoxon tests for the credit-assignment schemes on DD "
        f"(1,178 graphs, 9 runs each), Benjamini--Hochberg FDR corrected "
        f"across the family of {m} comparisons ($q<0.05$). Hedges' $g$ is the "
        "paired standardized effect size. "
        f"{n_sig} of the {m} comparisons survive correction, and none of them "
        "involves the estimator this paper adopts: truncated BPTT $+$ DvM at "
        "$M{=}8$ remains statistically indistinguishable from exact Full BPTT "
        "($p{=}0.074$, $q{=}0.124$). What the surviving comparisons do show is "
        "that window length matters (Full BPTT over DvM $M{=}4$, $q{=}0.0195$) "
        "and that the uncorrected $M{=}8$ truncation scores above the "
        "corrected $M{=}8$ variants ($q{=}0.0195$, $0.0293$)."
    )

    lines = []
    lines.append("% Generated from results/dd_credit_assignment_signif.json "
                 "(recomputed: paired exact Wilcoxon, BH-FDR step-up)")
    lines.append("\\begin{table*}[!htbp]")
    lines.append("\\caption{%s}" % cap)
    lines.append("\\label{tab:dd_signif}")
    lines.append("\\centering \\scriptsize")
    lines.append("\\begin{tabular*}{\\tblwidth}{@{} l l c c c l @{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Scheme A} & \\textbf{Scheme B} & $p$ & $q$ (BH) "
                 "& Hedges' $g$ & Verdict \\\\")
    lines.append("\\midrule")
    for r, qq in zip(rows, q):
        a, b = r["A"], r["B"]
        g = r["hedges_g"]
        sig = qq < ALPHA
        verdict = "sig." if sig else "ns"
        cell = "%s & %s & %s & %s & $%+.2f$ & %s \\\\" % (
            a, b, fmt(r["p_wilcoxon"]), fmt(qq), g, verdict)
        if (a, b) == BOLD_PAIR:
            # keep the paper's own comparison visually marked
            cell = "\\textbf{%s} & \\textbf{%s} & \\textbf{%s} & " \
                   "\\textbf{%s} & $%+.2f$ & \\textbf{%s} \\\\" % (
                       a, b, fmt(r["p_wilcoxon"]), fmt(qq), g, verdict)
        lines.append(cell)
    lines.append("\\bottomrule")
    lines.append(
        "\\multicolumn{6}{p{\\dimexpr\\tblwidth-2\\tabcolsep\\relax}}"
        "{\\scriptsize ns: not significant at $q<0.05$ after BH-FDR; sig.: "
        "significant at that threshold. Every surviving comparison involves a "
        "scheme other than the adopted estimator: window length matters "
        "($M{=}4$ is genuinely worse than Full BPTT) and, on DD at this "
        "50-epoch, 9-run protocol, the uncorrected $M{=}8$ truncation scores "
        "above the corrected $M{=}8$ variants --- although it is itself "
        "indistinguishable from Full BPTT ($p{=}0.570$). Retaining DvM is "
        "therefore justified by the manifold gradient correctness of "
        "Table~\\ref{tab:metric_fidelity}, not by accuracy.}"
    )
    lines.append("\\end{tabular*}")
    lines.append("\\end{table*}")
    lines.append("")

    with open(TEX_PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines))
    print(f"\nregenerated {TEX_PATH}  ({n_sig}/{m} significant)")


if __name__ == "__main__":
    sys.exit(main())

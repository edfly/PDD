# -*- coding: utf-8 -*-
"""Recompute the 2x2 factorial (geometry x spiking) statistics from raw per-run
accuracies, under the CORRECT paired design.

Design fact (verified 2026-10-21): all four corners were run with the shared
seed convention seed = 42 + run_id*100 + shot*10, dataset train_ratio=0.8,
sampler seeded identically (run_isolated.py / run_ann_baselines.py /
run_euc_missing_seq.py / run_hgcn_seq.py). Run i of RS, ES, RNS, ENS therefore
sees the SAME train/test episodes, so the interaction

    interaction_i = (RS_i - ES_i) - (RNS_i - ENS_i)

is a PAIRED contrast over the 5 shared seeds (df=4), not an independent-groups
two-way ANOVA (df_err=16). The unpaired ANOVA understates significance when the
within-pair correlation is positive; this script reports both.

Outputs (printed):
  * per-cell means of RS/ES/RNS/ENS (+ H2H as alternative RNS corner)
  * paired interaction t-test per cell, Welch cross-check
  * unpaired two-way ANOVA interaction F/p (legacy table values)
  * BH-FDR q-values across the 6-cell family for BOTH tests
  * H2H-specific numbers used in the manuscript text
"""
import json
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
A = json.load(open(os.path.join(RES, "analysis_v30.json"), encoding="utf-8"))
H = json.load(open(os.path.join(RES, "hgcn_baselines_v30.json"), encoding="utf-8"))
G = json.load(open(os.path.join(RES, "gat_baselines_v30.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "gat_baselines_v30.json")) else None

CELLS = [("MUTAG", 1), ("MUTAG", 5), ("PROTEINS", 1), ("PROTEINS", 5),
         ("ENZYMES", 1), ("ENZYMES", 5)]
N_RUNS = 5


def corner_accs(sub, key):
    v = sub.get(key)
    if not v or "accs" not in v:
        return None
    return np.asarray(v["accs"], dtype=float)


def hgcn_like(model, ds, shot):
    return np.asarray(
        [float(H[f"{model}_{ds}_{shot}shot_run{i}"]["final_acc"]) for i in range(N_RUNS)]
    )


def bh_q(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = q
    return out


rows = []
for ds, shot in CELLS:
    key = f"{ds}_{shot}"
    RS = corner_accs(A["spikergnn"], key)
    ES = corner_accs(A["euc"], key)
    RNS = hgcn_like("hgcn", ds, shot)
    H2H = hgcn_like("h2h", ds, shot)
    ENS = corner_accs(A["ann"], f"gin_{ds}_{shot}")
    assert all(len(x) == N_RUNS for x in (RS, ES, RNS, H2H, ENS)), (ds, shot)

    # paired interaction contrast over the 5 shared seeds
    d = (RS - ES) - (RNS - ENS)
    t_p, p_p = stats.ttest_rel(RS - ES, RNS - ENS)  # same as 1-sample t on d
    t_1s, p_1s = stats.ttest_1samp(d, 0.0)
    assert abs(t_p - t_1s) < 1e-9

    # legacy unpaired balanced two-way ANOVA interaction (df_err = 16)
    groups = [RS, ES, RNS, ENS]
    allv = np.concatenate(groups)
    gm = allv.mean()
    ss_tot = ((allv - gm) ** 2).sum()
    cell_means = [g.mean() for g in groups]
    ss_cells = sum(len(groups[0]) * (cm - gm) ** 2 for cm in cell_means)
    ss_err = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_err = len(allv) - 4
    ms_err = ss_err / df_err
    # interaction contrast for a 2x2 balanced design = 4 * cell_mean combination
    ic = (cell_means[0] - cell_means[1]) - (cell_means[2] - cell_means[3])
    ss_int = (len(groups[0]) * ic ** 2) / 4
    F_int = (ss_int / 1) / ms_err
    p_int = stats.f.sf(F_int, 1, df_err)

    rows.append(dict(ds=ds, shot=shot, RS=RS, ES=ES, RNS=RNS, ENS=ENS, H2H=H2H,
                     dG_spk=float((RS - ES).mean()), dG_nospk=float((RNS - ENS).mean()),
                     interaction=float(ic), d=d, t=t_p, p=p_p,
                     F=F_int, p_anova=p_int))

print(f"{'cell':<16}{'RS':>7}{'ES':>7}{'RNS':>7}{'ENS':>7}{'H2H':>7}"
      f"{'dGspk':>8}{'dGnon':>8}{'inter':>8}{'t(4)':>7}{'p_pair':>9}{'F(1,16)':>9}{'p_ANOVA':>9}")
for r in rows:
    print(f"{r['ds']+' '+str(r['shot'])+'-shot':<16}"
          f"{r['RS'].mean():>7.2f}{r['ES'].mean():>7.2f}{r['RNS'].mean():>7.2f}"
          f"{r['ENS'].mean():>7.2f}{r['H2H'].mean():>7.2f}"
          f"{r['dG_spk']:>+8.2f}{r['dG_nospk']:>+8.2f}{r['interaction']:>+8.2f}"
          f"{r['t']:>7.2f}{r['p']:>9.4f}{r['F']:>9.2f}{r['p_anova']:>9.4f}")

p_pair = [r["p"] for r in rows]
p_anov = [r["p_anova"] for r in rows]
q_pair = bh_q(p_pair)
q_anov = bh_q(p_anov)
print("\nBH-FDR across the 6-cell interaction family")
for r, qp, qa in zip(rows, q_pair, q_anov):
    print(f"  {r['ds']:<9}{r['shot']}-shot  p_paired={r['p']:.4f} q={qp:.4f}   "
          f"p_ANOVA={r['p_anova']:.4f} q={qa:.4f}")
print(f"\npaired: significant cells = {sum(1 for x in p_pair if x < 0.05)}/6; "
      f"BH-surviving = {sum(1 for x in q_pair if x < 0.05)}/6")
print(f"ANOVA : significant cells = {sum(1 for x in p_anov if x < 0.05)}/6; "
      f"BH-surviving = {sum(1 for x in q_anov if x < 0.05)}/6")

# H2H as the alternative non-spiking Riemannian corner
print("\nH2H corner: paired interaction t-tests + H2H vs HGCN deviations")
q_h2h = []
for r in rows:
    d2 = (r["RS"] - r["ES"]) - (r["H2H"] - r["ENS"])
    t2, p2 = stats.ttest_rel(r["RS"] - r["ES"], r["H2H"] - r["ENS"])
    dev = r["H2H"].mean() - r["RNS"].mean()
    q_h2h.append(p2)
    print(f"  {r['ds']:<9}{r['shot']}-shot  interaction={float(d2.mean()):+.2f}  "
          f"t(4)={t2:+.2f}  p={p2:.4f}   H2H-HGCN={dev:+.2f}")
qh = bh_q(q_h2h)
print("  H2H family BH q:", [f"{x:.3f}" for x in qh])

# MUTAG 1-shot: H2H vs ENS (text claims H2H falls BELOW ENS)
m1 = rows[0]
print(f"\nMUTAG 1-shot: H2H={m1['H2H'].mean()*100:.2f}  ENS={m1['ENS'].mean()*100:.2f}  "
      f"HGCN={m1['RNS'].mean()*100:.2f}  RS={m1['RS'].mean()*100:.2f}")
tw, pw = stats.ttest_rel(m1["H2H"], m1["ENS"])
print(f"  paired H2H vs ENS: t={tw:.3f} p={pw:.4f}")
th, ph = stats.ttest_rel(m1["H2H"], m1["RNS"])
print(f"  paired H2H vs HGCN: t={th:.3f} p={ph:.4f}")

# ---- M7 evidence: ENZYMES 1-shot Proto-GCN vs SpikeRPGNN (Pareto claim) ----
print("\nM7 evidence (Pareto):")
sig = A["significance"]
e1g = corner_accs(A["ann"], "gcn_ENZYMES_1")
print(f"  ENZYMES 1-shot Proto-GCN accs={np.round(e1g*100,2)} mean={e1g.mean()*100:.2f}")
rs1 = corner_accs(A["spikergnn"], "ENZYMES_1")
print(f"  ENZYMES 1-shot SpikeRPGNN accs={np.round(rs1*100,2)} mean={rs1.mean()*100:.2f}")
for k, v in sig.items():
    if "ENZYMES_1" in k and ("gcn" in k.lower()):
        print(f"  significance[{k}] = {json.dumps(v)[:220]}")
# energy coordinates from table_energy_v30.tex (T-aware analytic SOP model)
print("  energy: Proto-GCN 7.50 uJ vs SpikeRPGNN 9.55 uJ (ENZYMES, table_energy)")

"""
Verdict-migration figure: how the 60 paired significance verdicts change when
each cell is re-tested at n=20 instead of n=5.

Story: at n=5 the picture flatters the proposed method (21 significant leads,
zero deficits). After the n>=20 extension the honest picture is 23 leads and
4 deficits: six ns cells turn into confirmed leads, four n=5 leads dissolve
back to ns, and four previously hidden deficits become significant. Two-sided
paired t-tests over the first-k seeds of each arm, BH-corrected within the
same 60-test family at both sample sizes.

Style: Times New Roman, IEEE palette, 300 DPI, vector PDF.
Usage: python make_verdict_migration.py
Outputs: figures/fig_verdict_migration.{pdf,png} and
         results/verdict_migration.json
"""
import os, sys, json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["font.size"] = 11
rcParams["axes.linewidth"] = 1.0

_HERE = _SCRIPT_DIR
_RES = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, "results"))
_FIG = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, "figures"))

C_LEAD = "#D0021B"   # significant lead (SpikeRPGNN better)
C_NS   = "#7F8C9B"   # not significant
C_DEF  = "#1E8449"   # significant deficit (baseline better, honest loss)

def main():
    a = json.load(open(os.path.join(_RES, "analysis_v30.json"), encoding="utf-8"))
    sig = {(t["method"], t["dataset"], t["shot"]): t for t in a["significance"]}
    from scipy import stats

    def get_accs(m, ds, sh, n):
        if m in ("Proto-GIN", "Proto-GCN", "Proto-SAGE", "Proto-GAT"):
            return a["ann"][f"{m.split('-')[1].lower()}_{ds}_{sh}"]["accs"][:n]
        if m == "MSG*":
            return a["msg"][f"{ds}_{sh}"]["accs"][:n]
        return a["euc"][f"{ds}_{sh}"]["accs"][:n]

    cells = [(t["method"], t["dataset"], t["shot"]) for t in a["significance"]]

    def verdicts(n):
        out = {}
        for (m, ds, sh) in cells:
            s = a["spikergnn"][f"{ds}_{sh}"]["accs"][:n]
            b = get_accs(m, ds, sh, n)
            t, p = stats.ttest_rel(s, b)
            out[(m, ds, sh)] = (float(np.mean([x - y for x, y in zip(s, b)])),
                                float(p))
        order = sorted((v[1], c) for c, v in out.items())
        q = {}
        prev = 1.0
        for i in range(len(order) - 1, -1, -1):
            p, c = order[i]
            prev = min(prev, p * len(order) / (i + 1))
            q[c] = prev
        verd = {}
        for c in cells:
            md, qv = out[c][0], q[c]
            verd[c] = ("LEAD" if md > 0 else "DEFICIT") if qv < 0.05 else "ns"
        return verd, q, out

    v5, q5, st5 = verdicts(5)
    v20, q20, st20 = {c: ("LEAD" if t["mean_diff"] > 0 else "DEFICIT")
                      if t["q_bh"] < 0.05 else "ns"
                      for c, t in sig.items()}, None, None

    # flows n5 -> n20
    flows = {}
    for c in cells:
        flows[(v5[c], v20[c])] = flows.get((v5[c], v20[c]), 0) + 1
    n5_counts = {k: sum(1 for c in cells if v5[c] == k) for k in ("LEAD", "ns", "DEFICIT")}
    n20_counts = {k: sum(1 for c in cells if v20[c] == k) for k in ("LEAD", "ns", "DEFICIT")}

    # ---- draw ----
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cats = ["LEAD", "ns", "DEFICIT"]
    labels = {"LEAD": "significant lead\n(SpikeRPGNN wins)",
              "ns": "not significant",
              "DEFICIT": "significant deficit\n(baseline wins)"}
    colr = {"LEAD": C_LEAD, "ns": C_NS, "DEFICIT": C_DEF}
    x0, x1, bw = 0.0, 1.0, 0.34
    tops = {}
    for x, counts in ((x0, n5_counts), (x1, n20_counts)):
        y = 0.0
        for k in cats:
            h = counts[k]
            tops[(x, k)] = (y, y + h)
            ax.add_patch(plt.Rectangle((x - bw / 2, y), bw, h,
                                       facecolor=colr[k], edgecolor="white",
                                       linewidth=0.8, zorder=3))
            if h:
                ax.text(x, y + h / 2, str(h), ha="center", va="center",
                        color="white", fontsize=13, fontweight="bold", zorder=4)
            y += h
    total = sum(n5_counts.values())
    # flow bands
    def band(k5, k20, alpha=0.45, color=None):
        cnt = flows.get((k5, k20), 0)
        if not cnt:
            return
        ya, yb = tops[(x0, k5)]
        yc, yd = tops[(x1, k20)]
        h5, h20 = cnt, cnt
        # offset within the source/target category proportional to flow order
        band.setdefault
        ax.add_patch(matplotlib.patches.Polygon(
            [[x0 + bw / 2, ya], [x0 + bw / 2, ya + h5],
             [x1 - bw / 2, yd + h20], [x1 - bw / 2, yd]],
            closed=True, facecolor=color or colr[k20], alpha=alpha,
            edgecolor="none", zorder=1))
        return (ya, ya + h5, yd, yd + h20)

    used = {k5: 0.0 for k5 in cats}
    used20 = {k20: 0.0 for k20 in cats}
    def band2(k5, k20, alpha=0.45):
        cnt = flows.get((k5, k20), 0)
        if not cnt:
            return
        ya, _ = tops[(x0, k5)]
        y5 = ya + used[k5]; used[k5] += cnt
        yc, _ = tops[(x1, k20)]
        y20 = yc + used20[k20]; used20[k20] += cnt
        ax.add_patch(matplotlib.patches.Polygon(
            [[x0 + bw / 2, y5], [x0 + bw / 2, y5 + cnt],
             [x1 - bw / 2, y20 + cnt], [x1 - bw / 2, y20]],
            closed=True, facecolor=colr[k20], alpha=alpha,
            edgecolor="none", zorder=1))

    band2("LEAD", "LEAD", 0.35)
    band2("LEAD", "ns", 0.50)      # dissolved leads
    band2("ns", "ns", 0.25)
    band2("ns", "LEAD", 0.50)      # confirmed
    band2("ns", "DEFICIT", 0.60)   # revealed deficits

    for x, lab in ((x0, "$n{=}5$ runs/arm\n(initial protocol)"),
                   (x1, "$n{=}20$ runs/arm\n(extended protocol)")):
        ax.text(x, -3.2, lab, ha="center", va="top", fontsize=11)

    # totals above each bar
    ax.text(x0, 62.4, "21 / 39 / 0", ha="center", fontsize=10)
    ax.text(x1, 62.4, "23 / 33 / 4", ha="center", fontsize=10, fontweight="bold")

    # side annotations, placed in the clear left/right margins
    ax.annotate("4 leads dissolve:\nGCN / GIN / SAGE on\nPROTEINS 5-shot,\nEuc. on MUTAG 5-shot",
                xy=(x0 - bw / 2, 22), xytext=(-0.50, 34),
                fontsize=8.5, ha="left", va="center", color=C_NS,
                arrowprops=dict(arrowstyle="-", color=C_NS, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("6 confirmed leads:\nMSG* $\\times$4,\nProto-GAT $\\times$2",
                xy=(x1 + bw / 2, 42), xytext=(1.42, 20),
                fontsize=8.5, ha="right", va="center", color=C_LEAD,
                arrowprops=dict(arrowstyle="-", color=C_LEAD, lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("4 hidden deficits revealed:\nEuc. PROTEINS 5-shot,\nGAT PROTEINS/DD 1-shot,\nGIN Rdt-M5k 1-shot",
                xy=(x1 + bw / 2, 56), xytext=(1.42, 52),
                fontsize=8.5, ha="right", va="center", color=C_DEF,
                arrowprops=dict(arrowstyle="-", color=C_DEF, lw=0.8,
                                connectionstyle="arc3,rad=0.2"))

    # color legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_LEAD, label="significant lead"),
                       Patch(facecolor=C_NS, label="not significant"),
                       Patch(facecolor=C_DEF, label="significant deficit")],
              loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=9)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 66)
    ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(_FIG, exist_ok=True)
    fig.savefig(os.path.join(_FIG, "fig_verdict_migration.pdf"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(_FIG, "fig_verdict_migration.png"), dpi=300,
                bbox_inches="tight")

    # persist transition record for verification
    dissolved = [f"{c[0]} {c[1]}-{c[2]}" for c in cells
                 if v5[c] == "LEAD" and v20[c] == "ns"]
    confirmed = [f"{c[0]} {c[1]}-{c[2]}" for c in cells
                 if v5[c] == "ns" and v20[c] == "LEAD"]
    revealed = [f"{c[0]} {c[1]}-{c[2]}" for c in cells
                if v5[c] == "ns" and v20[c] == "DEFICIT"]
    rec = {
        "family": "60 paired tests (6 baselines x 5 datasets x 2 shots), BH within family",
        "n5_counts": n5_counts, "n20_counts": n20_counts,
        "flows": {f"{k[0]}->{k[1]}": v for k, v in sorted(flows.items())},
        "dissolved_leads": dissolved,
        "confirmed_leads": confirmed,
        "revealed_deficits": revealed,
    }
    with open(os.path.join(_RES, "verdict_migration.json"), "w",
              encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))

if __name__ == "__main__":
    main()

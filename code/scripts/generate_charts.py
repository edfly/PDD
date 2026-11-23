"""
Generate v31 paper charts from REAL experimental data.

Updates from v30:
  - Pareto chart now covers all five benchmarks (2x3 layout).
  - Energy chart is restricted to MUTAG/PROTEINS/ENZYMES and grouped by dataset
    to match Table 3 (table_energy_v30.tex).
  - Curvature sensitivity figure regenerated with IEEE/Times style.
  - Ablation chart keeps real ENZYMES data with cleaner styling.
  - Graphical abstract: pipeline schematic plus a REAL-data energy-accuracy
    inset (it previously used hard-coded illustrative numbers).

Style: Times New Roman, IEEE palette, 300 DPI, vector PDF.
Usage: python generate_charts_v31.py
"""
import sys, os, re, json, itertools, math
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["font.size"] = 11
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 12
rcParams["axes.linewidth"] = 1.0
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["legend.fontsize"] = 9

# IEEE-ish colorblind-safe palette
C = {
    "Proto-GCN": "#4A90D9",
    "Proto-GIN": "#5B8DBE",
    "Proto-SAGE": "#7FB3D5",
    "MSG*": "#F5A623",
    "Spike-GNN (Euc.)": "#BD10E0",
    "SpikeRPGNN": "#D0021B",
}
HIGHLIGHT = "#D0021B"

# --- Path resolution -------------------------------------------------------
# The script must run unchanged both from this submission package
# (code/scripts/ -> ../../results, ../../figures) and from a standalone
# repro bundle. Candidate directories are probed in order; the environment
# variables SPIKERPGNN_RESULTS / SPIKERPGNN_FIGURES override everything.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RES_CANDIDATES = [
    os.environ.get("SPIKERPGNN_RESULTS"),
    os.path.join(_HERE, os.pardir, os.pardir, "results"),
    os.path.join(_HERE, "spikergnn_results"),
    os.path.join(_HERE, os.pardir, "spikergnn_results"),
    os.path.join(_HERE, os.pardir, os.pardir, "spikergnn_results"),
]
_FIG_CANDIDATES = [
    os.environ.get("SPIKERPGNN_FIGURES"),
    os.path.join(_HERE, os.pardir, os.pardir, "figures"),
    os.path.join(_HERE, "figures"),
    os.path.join(_HERE, os.pardir, "figures"),
]
RESULTS_DIR = next((p for p in _RES_CANDIDATES if p and os.path.isdir(p)),
                   os.path.join(_HERE, os.pardir, os.pardir, "results"))
FIG_DIR = next((p for p in _FIG_CANDIDATES if p and os.path.isdir(p)),
               os.path.join(_HERE, os.pardir, os.pardir, "figures"))
os.makedirs(FIG_DIR, exist_ok=True)
print("RESULTS_DIR = %s" % os.path.normpath(RESULTS_DIR))
print("FIG_DIR     = %s" % os.path.normpath(FIG_DIR))


def load(path):
    p = os.path.join(RESULTS_DIR, path)
    return json.load(open(p)) if os.path.exists(p) else {}


def gather():
    """Collect all real result data into a unified dict."""
    data = {"methods": {}}   # (method, ds, shot) -> list of accs
    data["sr"] = {}          # (ds, shot) -> list of (acc, r_bar)
    data["euc"] = {}
    data["sweep"] = {}       # (lam, ds, shot) -> list of (acc, r_bar)

    for k, v in load("full_results_v24.json").items():
        if "final_acc" in v:
            data["sr"].setdefault((v["dataset"], v["shot"]), []).append(
                (v["final_acc"], v.get("final_spike_rate")))
            data["methods"].setdefault(("SpikeRPGNN", v["dataset"], v["shot"]), []).append(
                v["final_acc"])

    # NOTE: the MSG* results live at results/msgstar_results.json. This used to
    # be opened via a bare relative path ("msg_results/msgstar_results.json")
    # resolved against the CWD, so it silently found nothing and every chart
    # that legends MSG* was drawn without its points. Route it through load().
    for k, v in load("msgstar_results.json").items():
        if "runs" in v:
            accs = sorted(r["accuracy"] / 100.0 for r in v["runs"])
            data["methods"].setdefault(("MSG*", v["dataset"], v["shot"]), []).extend(accs)

    for k, v in load("ann_baselines_v30.json").items():
        if "final_acc" in v:
            model = k.split("_")[0].upper()
            data["methods"].setdefault((f"Proto-{model}", v["dataset"], v["shot"]), []).append(
                v["final_acc"])

    for k, v in load("spike_gnn_euc_v30.json").items():
        if "final_acc" in v:
            data["euc"].setdefault((v["dataset"], v["shot"]), []).append(
                (v["final_acc"], v.get("final_spike_rate")))
            data["methods"].setdefault(("Spike-GNN (Euc.)", v["dataset"], v["shot"]), []).append(
                v["final_acc"])

    for k, v in load("lambda_sweep_v30.json").items():
        if "final_acc" in v:
            data["sweep"].setdefault((v["lambda_sr"], v["dataset"], v["shot"]), []).append(
                (v["final_acc"], v.get("final_spike_rate")))
    return data


# ── analytic energy model (same as analyze_results_v30.py / table) ──
DS_STATS = {
    "MUTAG":      {"avg_nodes": 17.9, "avg_edges": 19.8},
    "PROTEINS":   {"avg_nodes": 39.1, "avg_edges": 72.8},
    "ENZYMES":    {"avg_nodes": 32.6, "avg_edges": 62.1},
    "DD":         {"avg_nodes": 284.3, "avg_edges": 715.7},
    "REDDIT-M5K": {"avg_nodes": 508.5, "avg_edges": 594.9},
}
E_SOP, E_FLOP = 0.9e-12, 4.6e-12
H, L = 128, 3


def dense_flops(arch, ds):
    N, E = DS_STATS[ds]["avg_nodes"], DS_STATS[ds]["avg_edges"]
    if arch == "gcn":
        pl = N * H * H + E * H
    elif arch == "sage":
        pl = 2 * N * H * H + E * H
    elif arch == "gin":
        pl = 2 * N * H * H + E * H
    elif arch == "spikergnn":
        pl = (2 * N * H * H + E * H) * 1.04
    elif arch == "msg":
        pl = (N * 32 * 32 + E * 32) * 1.10
    return L * pl + N * H


def energy_uj(arch, ds, r_bar=None):
    f = dense_flops(arch, ds)
    if r_bar is not None:
        return r_bar * f * E_SOP * 1e6
    return f * E_FLOP * 1e6


METHOD_ARCH = {"Proto-GCN": "gcn", "Proto-GIN": "gin", "Proto-SAGE": "sage",
               "MSG*": "msg", "SpikeRPGNN": "spikergnn", "Spike-GNN (Euc.)": "gin"}


def _short(m):
    return m.replace("Proto-", "P-").replace("Spike-GNN (Euc.)", "Euc.")


def _repel_labels(ax, points, fontsize=7.5):
    """Simple greedy 1-D vertical repulsion for point labels."""
    # points: list of (x, y, text, color)
    # initial offsets
    labels = []
    y_step = 1.8
    x_step = 0.03
    for x, y, text, color in points:
        labels.append({
            "x": x, "y": y, "text": text, "color": color,
            "dx": 8, "dy": 3,
        })
    # nudge to reduce vertical overlap
    for _ in range(40):
        moved = False
        for i, li in enumerate(labels):
            for lj in labels[i+1:]:
                yi = li["y"] + li["dy"]
                yj = lj["y"] + lj["dy"]
                if abs(yi - yj) < 2.0 and abs(li["x"] - lj["x"]) < 0.25:
                    if yi >= yj:
                        li["dy"] += y_step * 0.35
                        lj["dy"] -= y_step * 0.35
                    else:
                        li["dy"] -= y_step * 0.35
                        lj["dy"] += y_step * 0.35
                    moved = True
        if not moved:
            break
    # draw annotations
    for li in labels:
        use_arrow = abs(li["dy"]) > 3.5
        ax.annotate(li["text"], (li["x"], li["y"]),
                    textcoords="offset points",
                    xytext=(li["dx"], li["dy"]),
                    fontsize=fontsize, color=li["color"],
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5,
                                    relpos=(0, 0.5)) if use_arrow else None)


# ============================================================
# Chart 1: Pareto frontier (all five datasets, 2x3 layout)
# ============================================================
def plot_pareto(data):
    datasets = ["MUTAG", "PROTEINS", "ENZYMES", "DD", "REDDIT-M5K"]
    datasets = [ds for ds in datasets
                if any(("SpikeRPGNN", ds, 1) == k[:3] for k in data["methods"])]
    if not datasets:
        print("pareto: no data yet")
        return

    fig = plt.figure(figsize=(13.2, 8.6))
    axes = fig.subplots(2, 3, sharex=False, sharey=False)
    axes = axes.flatten()
    # hide the sixth (legend) axis
    leg_ax = axes[-1]
    for spine in leg_ax.spines.values():
        spine.set_visible(False)
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])
    leg_ax.set_facecolor("none")

    # universal legend handles
    from matplotlib.lines import Line2D
    handles = []
    for m in ["SpikeRPGNN", "MSG*", "Spike-GNN (Euc.)", "Proto-GIN", "Proto-SAGE", "Proto-GCN"]:
        if m == "SpikeRPGNN":
            h = Line2D([0], [0], marker="*", color="w", markerfacecolor=C[m],
                       markeredgecolor="black", markeredgewidth=1.0, markersize=16,
                       label=m)
        else:
            h = Line2D([0], [0], marker="o", color="w", markerfacecolor=C.get(m, "#888"),
                       markeredgecolor="black", markeredgewidth=1.0, markersize=10,
                       label=m)
        handles.append(h)
    sweep_line = Line2D([0], [0], linestyle="--", color=HIGHLIGHT, alpha=0.75,
                        linewidth=1.8, label=r"$\lambda_{sr}$ sweep (ours)")
    handles.append(sweep_line)
    leg_ax.legend(handles=handles, loc="center", framealpha=0.95,
                  title="Methods", fontsize=10, title_fontsize=11)

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        # method points
        plotted = []
        for (m, d, shot), accs in data["methods"].items():
            if d != ds or shot != 1 or not accs:
                continue
            acc = np.mean(accs) * 100
            std = np.std(accs) * 100  # population std, matches table convention
            if m in ("SpikeRPGNN", "Spike-GNN (Euc.)"):
                src = data["sr"] if m == "SpikeRPGNN" else data["euc"]
                srs = [s for _, s in src.get((ds, 1), []) if s is not None]
                r = np.mean(srs) if srs else 0.35
                e = energy_uj(METHOD_ARCH[m], ds, r)
            else:
                e = energy_uj(METHOD_ARCH[m], ds)
            # vertical error bar: +/- 1 population std of accuracy across runs
            ax.errorbar(e, acc, yerr=std, fmt="none", ecolor=C.get(m, "#888"),
                        elinewidth=1.1, capsize=3, capthick=1.1, alpha=0.85, zorder=3)
            marker = "*" if m == "SpikeRPGNN" else "o"
            size = 260 if m == "SpikeRPGNN" else 110
            ax.scatter(e, acc, s=size, color=C.get(m, "#888"), edgecolor="black",
                       linewidth=1.0, zorder=4, marker=marker)
            plotted.append((e, acc, _short(m), C.get(m, "#888")))

        # lambda sweep curve (SpikeRPGNN operating points) for datasets that have it
        pts = []
        for (lam, d, shot), lst in sorted(data["sweep"].items()):
            if d != ds or shot != 1 or lam == 0:
                continue
            accs = [a for a, _ in lst]
            srs = [s for _, s in lst if s is not None]
            if not accs or not srs:
                continue
            r = np.mean(srs)
            pts.append((energy_uj("spikergnn", ds, r), np.mean(accs) * 100, lam, r))
        if len(pts) >= 3:
            pts.sort()
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, "--", color=HIGHLIGHT, alpha=0.75, lw=1.8, zorder=3)
            # label sparse lambda points
            for k, (x, y, lam, r) in enumerate(pts):
                if k % 2 == 0:
                    ax.annotate(f"{lam:.2g}", (x, y), textcoords="offset points",
                                xytext=(4, -10), fontsize=7, color=HIGHLIGHT,
                                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                          edgecolor="none", alpha=0.8))

        # label methods; prioritize SpikeRPGNN and dense Proto-*
        priority = {"SpikeRPGNN": 0}
        for pm in ["Proto-GIN", "Proto-SAGE", "Proto-GCN"]:
            priority[pm] = 1
        plotted.sort(key=lambda t: priority.get(t[2], 2))
        _repel_labels(ax, plotted)

        ax.set_xlabel(r"Energy ($\mu$J/graph)", fontsize=11)
        ax.set_ylabel(r"1-shot accuracy (%)", fontsize=11)
        ax.set_title(ds, fontsize=12, fontweight="bold")

    # remove unused axes if fewer than 5 datasets
    for ax in axes[len(datasets):-1]:
        ax.axis("off")

    fig.suptitle("Energy--Accuracy Pareto Frontier across all five benchmarks (1-shot, analytic SOP model)",
                 y=0.985, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(os.path.join(FIG_DIR, "chart_pareto.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "chart_pareto.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figures/chart_pareto.pdf")


# ============================================================
# Chart 2: Energy comparison bars (MUTAG/PROTEINS/ENZYMES only)
# ============================================================
def plot_energy(data):
    datasets = ["MUTAG", "PROTEINS", "ENZYMES"]
    methods = ["Proto-GCN", "Proto-GIN", "Proto-SAGE", "MSG*", "SpikeRPGNN", "Spike-GNN (Euc.)"]
    # collect energy per (method, dataset)
    energies = {}
    for m in methods:
        arch = METHOD_ARCH[m]
        for ds in datasets:
            accs = data["methods"].get((m, ds, 1), [])
            if not accs:
                energies[(m, ds)] = np.nan
                continue
            if m in ("SpikeRPGNN", "Spike-GNN (Euc.)"):
                src = data["sr"] if m == "SpikeRPGNN" else data["euc"]
                srs = [s for _, s in src.get((ds, 1), []) if s is not None]
                r = np.mean(srs) if srs else 0.35
            else:
                r = None
            energies[(m, ds)] = energy_uj(arch, ds, r)

    x = np.arange(len(datasets))
    width = 0.13
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, m in enumerate(methods):
        vals = [energies[(m, ds)] for ds in datasets]
        offset = width * (i - len(methods)/2 + 0.5)
        bars = ax.bar(x + offset, vals, width, label=m, color=C.get(m, "#888"),
                      edgecolor="black", linewidth=0.8, zorder=3)
        # label dense methods on top; spiking methods get reduction % above
        for b, v, ds in zip(bars, vals, datasets):
            if math.isnan(v):
                continue
            ref = energies[("Proto-GIN", ds)]
            red = (1 - v / ref) * 100 if not math.isnan(ref) else 0
            if m in ("Proto-GCN", "Proto-GIN", "Proto-SAGE"):
                ax.text(b.get_x() + b.get_width() / 2, v + max(energies.values()) * 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            elif red > 30:
                ax.text(b.get_x() + b.get_width() / 2, v + max(energies.values()) * 0.015,
                        f"{red:.0f}%", ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

    ax.set_ylabel(r"Energy ($\mu$J/graph)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_title(r"Theoretical energy per graph (1-shot, analytic SOP model)", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.set_ylim(0, max(energies.values()) * 1.22)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart_energy.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "chart_energy.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: figures/chart_energy.pdf")


# ============================================================
# Chart 3: Ablation (ENZYMES, real data + paired t significance)
# ============================================================
def plot_ablation():
    """ENZYMES 5-way 1-shot ablation (both panels).

    Data source: ablation_50ep_v30.json -- the 50-epoch main-protocol
    ablation that Table 8 reports. Keys have the form
        {variant}_{DATASET}_{shot}shot_run{N}
    The run count shown in the panel title is derived from the data, so the
    figure can never disagree with the underlying result file.
    Energy for the spiking variants other than Full uses the Full-model rate
    as an approximation; those bars are flagged with a "~" marker and the
    approximation is disclosed in the caption.
    """
    p = os.path.join(RESULTS_DIR, "ablation_50ep_v30.json")
    if not os.path.exists(p):
        print("ablation: no data (%s)" % p)
        return
    from scipy import stats as st
    d = json.load(open(p))
    KEY_RE = re.compile(
        r"^(full|no_spikes|euclidean|no_frechet|no_dvm|no_meta)"
        r"_(MUTAG|PROTEINS|ENZYMES)_(\d+)shot_run(\d+)$")
    order = ["full", "no_spikes", "euclidean", "no_frechet", "no_dvm", "no_meta"]
    labels = {"full": "Full", "no_spikes": "w/o Spikes", "euclidean": "w/o Riem.",
              "no_frechet": "w/o Fr\u00e9chet", "no_dvm": "w/o DvM", "no_meta": "w/o Meta"}

    accs_by = {}
    for k, v in d.items():
        m = KEY_RE.match(k)
        if not m:
            continue
        var, dsv, shot = m.group(1), m.group(2), int(m.group(3))
        if dsv != "ENZYMES" or shot != 1:
            continue
        accs_by.setdefault(var, []).append(v["final_acc"])
    if "full" not in accs_by:
        print("ablation: no ENZYMES 1-shot runs found in %s" % p)
        return
    n_runs = len(accs_by["full"])

    # measured full-model firing rate on ENZYMES (1-shot) from main-protocol runs
    full_data = gather()["sr"].get(("ENZYMES", 1), [])
    full_rate = np.mean([s for _, s in full_data if s is not None]) if full_data else 0.40

    names, accs, pvals, energies, errs = [], [], [], [], []
    base = np.array(accs_by["full"])
    for v in order:
        if v not in accs_by:
            continue
        a = np.array(accs_by[v]) * 100
        names.append(labels[v])
        accs.append(a.mean())
        errs.append(a.std())
        if v == "full":
            pvals.append(None)
            energies.append(energy_uj("spikergnn", "ENZYMES", full_rate))
        elif v == "no_spikes":
            pvals.append(None)
            energies.append(energy_uj("gin", "ENZYMES"))
        else:
            # spiking variants: reliable firing rates were not isolated from the
            # support-set spike accumulation, so energy uses the Full-model rate
            # as an approximation (the dominant qualitative point is the gap to
            # the dense w/o Spikes bar).
            pvals.append(None if np.allclose(a / 100.0, base)
                         else st.ttest_rel(base, a / 100.0).pvalue)
            energies.append(energy_uj("spikergnn", "ENZYMES", full_rate))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = [HIGHLIGHT if n == "Full" else "#4A90D9" for n in names]
    bars = ax1.bar(names, accs, color=colors, edgecolor="black", linewidth=0.9, zorder=3,
                   yerr=errs, capsize=3)
    for b, p_ in zip(bars, pvals):
        if p_ is not None:
            sig = "***" if p_ < 0.001 else "**" if p_ < 0.01 else "*" if p_ < 0.05 else "ns"
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, sig,
                     ha="center", fontsize=9, color="black")
    ax1.set_ylabel(r"Accuracy (%)", fontsize=12)
    ax1.set_title(r"(a) ENZYMES 5-way 1-shot ablation (%d runs)" % n_runs, fontsize=12)
    ax1.tick_params(axis="x", rotation=20)
    ax1.set_ylim(bottom=min(accs) - 4)

    bars2 = ax2.bar(names, energies, color=colors, edgecolor="black", linewidth=0.9, zorder=3)
    for b, v, n in zip(bars2, energies, names):
        # only Full and w/o Spikes have reliable energy assignments
        if n in ("Full", "w/o Spikes"):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.2f}",
                     ha="center", fontsize=8)
        else:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, "~",
                     ha="center", fontsize=8, color="gray")
    ax2.set_ylabel(r"Energy ($\mu$J/graph)", fontsize=12)
    ax2.set_title(r"(b) Theoretical energy", fontsize=12)
    ax2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chart_ablation.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "chart_ablation.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: %s" % os.path.join(FIG_DIR, "chart_ablation.pdf"))


# ============================================================
# Chart 4: Curvature sensitivity (MUTAG)
# ============================================================
def plot_curvature():
    p = os.path.join(RESULTS_DIR, "c_sensitivity.json")
    if not os.path.exists(p):
        print("curvature: c_sensitivity.json not found")
        return
    d = json.load(open(p))
    # Drop the "_protocol" metadata block (not a curvature point) before sorting.
    d = {k: v for k, v in d.items() if not str(k).startswith("_")}
    cs = sorted(d.keys(), key=lambda k: float(k))
    means = [d[c]["mean"] for c in cs]
    stds = [d[c]["std"] for c in cs]
    cs_float = [float(c) for c in cs]
    # Run count is read from the data, never hard-coded: the scan is re-runnable
    # with any n, and a frozen "(2-run scan)" title silently lies once it is.
    n_runs = max(len(d[c].get("accuracies", [])) for c in cs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(cs_float, means, yerr=stds, fmt="-o", color="#4A90D9",
                ecolor="#333333", capsize=5, linewidth=2, markersize=10,
                markeredgecolor="black", markeredgewidth=1.0, zorder=3)
    # Mark the DEFAULT c=1.0 rather than the argmax. The paper retains the
    # a-priori value instead of selecting from the scan (the scanned points are
    # statistically indistinguishable), and the caption says so explicitly, so
    # labelling the argmax "Best" would misstate the experiment.
    def_idx = next((i for i, c in enumerate(cs_float) if abs(c - 1.0) < 1e-9), None)
    if def_idx is not None:
        ax.scatter([cs_float[def_idx]], [means[def_idx]], s=240, color=HIGHLIGHT,
                   zorder=5, marker="o", edgecolor="black", linewidth=1.2,
                   label="Default $c=1.0$")
    ax.set_xlabel("Poincaré curvature $c$", fontsize=12)
    ax.set_ylabel("1-shot accuracy (%)", fontsize=12)
    ax.set_title(f"Curvature sensitivity on MUTAG ({n_runs}-run scan)", fontsize=12)
    ax.set_xticks(cs_float)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_ylim(bottom=max(0, min(np.array(means) - np.array(stds)) - 3))
    plt.tight_layout()
    out_pdf = os.path.join(FIG_DIR, "fig_c_sensitivity.pdf")
    out_png = os.path.join(FIG_DIR, "fig_c_sensitivity.png")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


# ============================================================
# Chart 5: Graphical abstract (schematic + Pareto inset)
# ============================================================
def plot_graphical_abstract(data):
    """Schematic pipeline on the left, REAL energy-accuracy data on the right.

    The right-hand inset used to be drawn from hard-coded illustrative numbers
    (a made-up `r_bar`/`acc` sweep plus invented Proto-GIN and MSG* coordinates).
    It is now built from the same real results as Figure 4: the lambda_sr sweep
    of results/lambda_sweep_v30.json under the analytic SOP energy model, the
    lambda_sr=0.5 operating point actually used in the paper, and the real
    Proto-GIN / MSG* baselines on the same dataset.
    """
    fig = plt.figure(figsize=(12, 4.2))

    # Left: architecture pipeline
    ax = fig.add_axes([0.015, 0.05, 0.56, 0.90])
    ax.axis('off')
    stages = [
        ('Input', 'Graph $G$', '#E8E8E8'),
        ('MSPE', 'product\nmanifold', '#DCEEFB'),
        ('STMP', 'spiking\nmessage passing', '#FDEBD0'),
        ('FPC', 'Fr\'echet\nprototypes', '#E8DAEF'),
        ('GDC', 'geodesic\nclassify', '#D5F5E3'),
    ]
    n = len(stages)
    x0, dx, bw, bh, y = 0.02, 0.19, 0.17, 0.30, 0.28
    for i, (title, subtitle, color) in enumerate(stages):
        x = x0 + i * dx
        box = FancyBboxPatch((x, y), bw, bh,
                             boxstyle='round,pad=0.012', facecolor=color,
                             edgecolor='black', linewidth=1.1, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x + bw / 2, y + bh * 0.66, title, ha='center', va='center',
                fontsize=11.5, weight='bold', transform=ax.transAxes)
        ax.text(x + bw / 2, y + bh * 0.30, subtitle, ha='center', va='center',
                fontsize=7.6, linespacing=1.25, transform=ax.transAxes)
        if i < n - 1:
            ax.annotate('', xy=(x + dx, y + bh / 2),
                        xytext=(x + bw, y + bh / 2),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', lw=1.6, color='black'))

    # L_sr feedback arc from GDC back toward MSPE/STMP, label clear of the arc
    arc = FancyArrowPatch((x0 + (n - 1) * dx + bw / 2, y + bh + 0.05),
                          (x0 + dx + bw / 2, y + bh + 0.05),
                          connectionstyle='arc3,rad=0.28',
                          arrowstyle='-|>', mutation_scale=16,
                          linewidth=1.8, color=HIGHLIGHT,
                          transform=ax.transAxes)
    ax.add_patch(arc)
    ax.text(0.50, 0.875,
            '$\\mathcal{L}_{sr}=\\lambda_{sr}(\\bar{r}-r^{*})^{2}$ (Pareto scalarization)',
            ha='center', va='center', fontsize=10.5, color=HIGHLIGHT,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor='none', alpha=0.95),
            transform=ax.transAxes)
    ax.text(0.50, 0.10,
            'SpikeRPGNN: spiking message passing on Riemannian manifolds',
            ha='center', fontsize=10.5, style='italic', transform=ax.transAxes)

    # ---- Right: energy-accuracy inset, built from REAL data (MUTAG, 1-shot) ----
    axp = fig.add_axes([0.645, 0.175, 0.335, 0.715])
    ds = 'MUTAG'
    sweep_pts = []
    for (lam, d, shot), lst in sorted(data['sweep'].items()):
        if d != ds or shot != 1 or lam == 0:
            continue
        accs = [a for a, _ in lst]
        srs = [s for _, s in lst if s is not None]
        if not accs or not srs:
            continue
        sweep_pts.append((energy_uj('spikergnn', ds, np.mean(srs)),
                          np.mean(accs) * 100, lam))
    sweep_pts.sort()

    if len(sweep_pts) >= 2:
        sx = [p[0] for p in sweep_pts]
        sy = [p[1] for p in sweep_pts]
        axp.plot(sx, sy, '--', color=HIGHLIGHT, alpha=0.8, lw=1.8, zorder=3)
        axp.scatter(sx, sy, s=44, color=HIGHLIGHT, edgecolor='black',
                    linewidth=0.7, zorder=4)
        # The sweep points sit within a very small energy/accuracy window, so
        # per-point lambda labels collide. Summarise the range with one arrow.
        mid_x, mid_y = float(np.mean(sx)), float(np.mean(sy))
        axp.annotate(r'$\lambda_{sr}\in\{0.05,\ldots,2.0\}$',
                     (mid_x, mid_y), textcoords='offset points',
                     xytext=(64, -30), fontsize=8, ha='left', zorder=7,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='none', alpha=0.92),
                     arrowprops=dict(arrowstyle='->', lw=0.9, color=HIGHLIGHT))
        tgt = [p for p in sweep_pts if abs(p[2] - 0.5) < 1e-9]
        if tgt:
            tx, ty, _ = tgt[0]
            axp.scatter([tx], [ty], marker='*', s=330, color=HIGHLIGHT,
                        edgecolor='black', linewidths=0.9, zorder=6)
            axp.annotate(r'$\lambda_{sr}{=}0.5$ ($r^{*}$ op. point)',
                         (tx, ty), textcoords='offset points',
                         xytext=(14, 16), fontsize=8, ha='left', zorder=7,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor='none', alpha=0.92),
                         arrowprops=dict(arrowstyle='->', lw=0.9, color='black'))

    for m, mk, dy in [('Proto-GIN', 's', -14), ('MSG*', '^', 8)]:
        accs = data['methods'].get((m, ds, 1))
        if not accs:
            continue
        e = energy_uj(METHOD_ARCH[m], ds)
        ay = np.mean(accs) * 100
        axp.scatter([e], [ay], marker=mk, s=90, color=C.get(m, '#888'),
                    edgecolor='black', linewidth=0.9, zorder=5)
        axp.annotate(_short(m), (e, ay), textcoords='offset points',
                     xytext=(6, dy), fontsize=7.5, color=C.get(m, '#888'),
                     zorder=6)

    axp.set_xlabel(r'Energy ($\mu$J/graph)', fontsize=9.5)
    axp.set_ylabel('1-shot accuracy (%)', fontsize=9.5)
    axp.set_title('Energy--Accuracy Pareto (%s)' % ds, fontsize=10.5, pad=7)
    axp.grid(alpha=0.25, linewidth=0.6)
    axp.margins(x=0.16, y=0.18)
    axp.tick_params(labelsize=8)
    axp.set_axisbelow(True)

    out_pdf = os.path.join(FIG_DIR, 'graphical_abstract.pdf')
    out_png = os.path.join(FIG_DIR, 'graphical_abstract.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_pdf}')


if __name__ == "__main__":
    data = gather()
    plot_pareto(data)
    plot_energy(data)
    plot_ablation()
    plot_curvature()
    plot_graphical_abstract(data)
    print("\nAll charts generated (real data only).")

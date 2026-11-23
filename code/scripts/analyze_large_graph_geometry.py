# -*- coding: utf-8 -*-
"""Task 3: geometry ablation on the LARGE graphs from EXISTING data.

The archived spike_gnn_euc_v30.json holds the Euclidean-spiking control (ES)
at the SAME capacity (hidden_dim=128), SAME protocol, SAME shared seeds as
SpikeRPGNN (RS) -- for all five datasets, 1- and 5-shot, 5 runs. That is a
geometry-only contrast (RS - ES) on DD and Rdt-M5K already, so no new run is
needed; we only have to read it out honestly.

Outputs a compact table + paired Welch t-tests on the shared seed pairs.
"""
import os
import json
import statistics as st

try:
    from scipy import stats as _stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

POST = r"D:/2026/paper/PBB/post"
RES = os.path.join(POST, "results")

FILES_RS = ["full_results_v24.json", "v24_5shot.json", "v24_dd.json",
            "v24_rm5k_1shot.json", "v24_rm5k_5shot.json",
            "v24_extra_DD_1shot.json", "v24_extra_REDDIT-M5K_1shot.json"]
FILE_ES = "spike_gnn_euc_v30.json"

CELLS = [(d, s) for d in ["MUTAG", "PROTEINS", "ENZYMES", "DD", "REDDIT-M5K"]
         for s in (1, 5)]


def load_rs():
    acc = {}
    for fn in FILES_RS:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        for k, v in d.items():
            if not isinstance(v, dict) or "final_acc" not in v:
                continue
            acc[(v["dataset"], v["shot"], v["run_id"])] = v["final_acc"] * 100
    return acc


def load_es():
    with open(os.path.join(RES, FILE_ES)) as f:
        d = json.load(f)
    acc = {}
    for k, v in d.items():
        if not isinstance(v, dict) or "final_acc" not in v:
            continue
        r = v.get("run_id")
        if r is None:
            r = int(k.rsplit("run", 1)[-1])
        acc[(v["dataset"], v["shot"], int(r))] = v["final_acc"] * 100
    return acc


def main():
    rs, es = load_rs(), load_es()
    print(f"scipy available: {HAVE_SCIPY}\n")
    hdr = f"{'cell':<16} {'RS(128)':>16} {'ES(128)':>16} {'RS-ES':>8} {'t':>7} {'p':>8} {'n':>3}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for ds, shot in CELLS:
        rr = {r: rs[(ds, shot, r)] for r in range(5) if (ds, shot, r) in rs}
        ee = {r: es[(ds, shot, r)] for r in range(5) if (ds, shot, r) in es}
        common = sorted(set(rr) & set(ee))
        if not common:
            continue
        a = [rr[r] for r in common]
        b = [ee[r] for r in common]
        diff = [x - y for x, y in zip(a, b)]
        md = st.mean(diff)
        if HAVE_SCIPY and len(common) > 1:
            t, p = _stats.ttest_rel(a, b)
            # Welch for reference
            tw, pw = _stats.ttest_ind(a, b, equal_var=False)
        else:
            t = p = tw = pw = float("nan")
        print(f"{ds+' '+str(shot)+'shot':<16} "
              f"{st.mean(a):>7.2f}+-{st.pstdev(a):<4.2f}   "
              f"{st.mean(b):>7.2f}+-{st.pstdev(b):<4.2f}   "
              f"{md:>+7.2f} {t:>7.2f} {p:>8.4f} {len(common):>3}")
        rows.append(dict(dataset=ds, shot=shot, rs_mean=st.mean(a),
                         rs_std=st.pstdev(a), es_mean=st.mean(b),
                         es_std=st.pstdev(b), gain=md, t=t, p=p,
                         n=len(common), n_rs_total=len(rr), n_es_total=len(ee)))

    print("\nLARGE-GRAPH SUBSET (DD + Rdt-M5K), pooled over the 4 cells:")
    lg = [r for r in rows if r["dataset"] in ("DD", "REDDIT-M5K")]
    if lg:
        g = [r["gain"] for r in lg]
        print(f"  gains: {[f'{x:+.2f}' for x in g]}")
        print(f"  mean gain = {st.mean(g):+.2f} pp ; "
              f"max |gain| = {max(abs(x) for x in g):.2f} pp")
        sig = [r for r in lg if HAVE_SCIPY and r["p"] < 0.05]
        sdesc = [(r["dataset"], "%+.2f" % r["gain"]) for r in sig]
        print(f"  cells with paired p<0.05: {len(sig)}/{len(lg)} -> {sdesc}")

    out = os.path.join(RES, "large_graph_geometry_ablation.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

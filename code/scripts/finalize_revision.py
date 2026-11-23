# -*- coding: utf-8 -*-
"""Finalize the revision once the experiment queue has finished.

Steps
-----
A. Convert the per-task MSG* d=128 JSONs into the canonical
   `results/msgstar_results.json` shape that the analysis + gate scripts read,
   archiving the original width-32 baseline as `results/msgstar_d32_results.json`.
B. Merge the per-task `w/o STMP` JSONs into `results/ablation_50ep_v30.json`.
C. Re-run the analysis pipeline in a staging directory (the scripts resolve
   `spikergnn_results/` and `msg_results/` relative to their own location),
   refresh `results/analysis_v30.json`, and regenerate every table fragment
   into `manuscript/tables/`.
D. Print every number the prose has to be updated with.

Safe to re-run: all writes are atomic and step C rebuilds from scratch.
"""
import os
import re
import sys
import json
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(POST, "results")
TABLES = os.path.join(POST, "manuscript", "tables")
STAGE_MAIN = os.path.join(HERE, "spikergnn_results")
STAGE_MSG = os.path.join(HERE, "msg_results")
PY = r"C:/ProgramData/miniconda3/envs/d2l/python.exe"
PYLIBS = os.path.join(POST, "paper", "review_v37", "pylibs")

DATASETS = ["MUTAG", "PROTEINS", "ENZYMES", "DD", "REDDIT-M5K"]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(p, o):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, default=float)
    os.replace(tmp, p)


# ── A. canonical MSG* (d=128) ─────────────────────────────────────
def step_a():
    files = sorted(glob.glob(os.path.join(RESULTS, "tmp_msgstar", "*.json")))
    if not files:
        print("[A] no tmp_msgstar results yet -- skipped")
        return None
    canon, n = {}, 0
    for p in files:
        d = jload(p)
        for k, v in d.items():
            if not isinstance(v, dict) or "final_acc" not in v:
                continue
            key = f"{v['dataset']}_{v['shot']}shot"
            canon.setdefault(key, {"dataset": v["dataset"], "shot": v["shot"],
                                   "runs": []})
            canon[key]["runs"].append({
                "run_id": v["run_id"],
                "accuracy": v["final_acc"] * 100.0,
                "epoch_accuracies": v.get("epoch_accuracies", []),
            })
            n += 1
    for key in canon:
        canon[key]["runs"].sort(key=lambda r: r["run_id"])
    print(f"[A] collected {n} MSG* d=128 run(s) over {len(canon)} cells")
    if n < 50:
        print(f"[A] INCOMPLETE ({n}/50) -- canonical file NOT overwritten")
        return canon
    cur = os.path.join(RESULTS, "msgstar_results.json")
    arc = os.path.join(RESULTS, "msgstar_d32_results.json")
    if os.path.exists(cur) and not os.path.exists(arc):
        shutil.copy2(cur, arc)
        print(f"[A] archived width-32 baseline -> {arc}")
    jdump(cur, canon)
    print(f"[A] wrote capacity-matched MSG* -> {cur}")
    return canon


# ── B. merge w/o STMP into the ablation file ──────────────────────
def step_b():
    files = sorted(glob.glob(os.path.join(RESULTS, "tmp_abl_stmp", "*.json")))
    if not files:
        print("[B] no tmp_abl_stmp results yet -- skipped")
        return None
    tgt = os.path.join(RESULTS, "ablation_50ep_v30.json")
    d = jload(tgt) if os.path.exists(tgt) else {}
    added = 0
    for p in files:
        for k, v in jload(p).items():
            if isinstance(v, dict) and "final_acc" in v:
                d[k] = v
                added += 1
    jdump(tgt, d)
    stmp = {k: v for k, v in d.items() if k.startswith("no_stmp")}
    print(f"[B] merged {added} run(s); no_stmp keys now {len(stmp)} -> {tgt}")
    return stmp


# ── C. rebuild analysis + tables ──────────────────────────────────
def step_c():
    for p in (STAGE_MAIN, STAGE_MSG):
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p)
    for src in glob.glob(os.path.join(RESULTS, "*.json")):
        shutil.copy2(src, os.path.join(STAGE_MAIN, os.path.basename(src)))
    shutil.copy2(os.path.join(RESULTS, "msgstar_results.json"),
                 os.path.join(STAGE_MSG, "msgstar_results.json"))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([HERE, PYLIBS, env.get("PYTHONPATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"

    print("[C] running analyze_results_v30.py ...")
    r = subprocess.run([PY, os.path.join(HERE, "analyze_results_v30.py")],
                       cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit("analyze_results_v30.py failed")
    shutil.copy2(os.path.join(STAGE_MAIN, "analysis_v30.json"),
                 os.path.join(RESULTS, "analysis_v30.json"))
    print("[C] refreshed results/analysis_v30.json")

    print("[C] running paper_tables_to_tex.py ...")
    r = subprocess.run(
        [PY, os.path.join(HERE, "paper_tables_to_tex.py"), "--out-dir", TABLES],
        cwd=HERE, env=env, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("paper_tables_to_tex.py failed")

    for p in (STAGE_MAIN, STAGE_MSG):
        shutil.rmtree(p, ignore_errors=True)
    print("[C] staging dirs removed")


# ── D. summary ───────────────────────────────────────────────────
def step_d():
    a = jload(os.path.join(RESULTS, "analysis_v30.json"))
    rs, msg = a["spikergnn"], a["msg"]
    print("\n=== MSG* (d=128) vs SpikeRPGNN ===")
    print(f"{'cell':<16}{'SpikeRPGNN':>14}{'MSG* d128':>14}{'gap':>9}{'q_bh':>9}")
    sig = {(t['dataset'], t['shot']): t for t in a.get("significance", [])
           if t.get("method") == "MSG*"}
    for ds in DATASETS:
        for shot in (1, 5):
            k = f"{ds}_{shot}"
            if k not in msg:
                continue
            g = (rs[k]["mean"] - msg[k]["mean"]) * 100
            t = sig.get((ds, shot))
            q = f"{t['q_bh']:.4f}" if t else "-"
            print(f"{ds+' '+str(shot)+'shot':<16}"
                  f"{rs[k]['mean']*100:>8.1f}+-{rs[k]['std']*100:<4.1f}"
                  f"{msg[k]['mean']*100:>8.1f}+-{msg[k]['std']*100:<4.1f}"
                  f"{g:>+9.1f}{q:>9}")

    abl = jload(os.path.join(RESULTS, "ablation_50ep_v30.json"))
    print("\n=== w/o STMP vs Full (1-shot, paired) ===")
    try:
        from scipy import stats
    except Exception:
        stats = None
    for ds in ["MUTAG", "PROTEINS", "ENZYMES"]:
        def accs(var):
            return [v["final_acc"] for k, v in abl.items()
                    if isinstance(v, dict) and v.get("variant") == var
                    and v.get("dataset") == ds and v.get("shot") == 1]
        base, st_ = accs("full"), accs("no_stmp")
        if not base or not st_:
            continue
        import statistics as stt
        d = stt.mean(st_) - stt.mean(base)
        p = stats.ttest_rel(base, st_).pvalue if stats and len(base) == len(st_) else float("nan")
        print(f"  {ds:<10} Full {stt.mean(base)*100:.1f}+-{stt.pstdev(base)*100:.1f} | "
              f"w/o STMP {stt.mean(st_)*100:.1f}+-{stt.pstdev(st_)*100:.1f} | "
              f"dAcc {d*100:+.1f} | p={p:.3f} | n={len(st_)}")

    print("\n=== energy rows (Tier-1, 1-shot) ===")
    for e in a.get("energy", []):
        print("  ", e)


if __name__ == "__main__":
    step_a()
    step_b()
    step_c()
    step_d()

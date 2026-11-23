# -*- coding: utf-8 -*-
"""Queue driver for the two long experiment batches of the revision.

Batch A -- capacity-matched MSG* (hidden_dim=128)
    5 datasets x {1,5}-shot x 5 runs = 50 single-run tasks.
    Protocol identical to the SpikeRPGNN main matrix (50 epochs,
    20 episodes/epoch, 30 test episodes, seed = 42 + 100*run + 10*shot).

Batch B -- w/o STMP ablation
    {MUTAG, PROTEINS, ENZYMES} x 1-shot x 5 runs = 15 single-run tasks,
    using the SAME runner/protocol as the archived six-variant ablation
    (run_ablation_50ep.py), so the new variant lands in the same table.

Scheduling
    Longest-processing-time-first across P parallel workers, each task a
    separate subprocess with OMP/MKL threads pinned to 1 so that P workers
    map cleanly onto P cores. Per-task JSONs are written independently (no
    shared-file races); the driver merges them into the two master files at
    the end and is itself re-runnable (--merge-only) if the session ends.

Usage
    python run_capacity_queue.py --workers 4
    python run_capacity_queue.py --merge-only        # just rebuild masters
"""
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(os.path.dirname(HERE))          # .../PBB/post
PBB = os.path.dirname(POST)                            # .../PBB
RESULTS = os.path.join(POST, "results")
TMP_MSG = os.path.join(RESULTS, "tmp_msgstar")
TMP_ABL = os.path.join(RESULTS, "tmp_abl_stmp")

PYLIBS = os.path.join(POST, "paper", "review_v37", "pylibs")
PY = r"C:/ProgramData/miniconda3/envs/d2l/python.exe"

MASTER_MSG = os.path.join(RESULTS, "msgstar_d128_results.json")
MASTER_ABL = os.path.join(RESULTS, "ablation_stmp_v30.json")

MSG_DATASETS = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "REDDIT-M5K"]
ABL_DATASETS = ["MUTAG", "PROTEINS", "ENZYMES"]

# Rough per-run cost (seconds) at hidden_dim=128, used only for LPT ordering.
COST_MSG = {"MUTAG": 240, "PROTEINS": 500, "ENZYMES": 600,
            "DD": 1500, "REDDIT-M5K": 5000}
COST_ABL = {"MUTAG": 450, "PROTEINS": 500, "ENZYMES": 650}


def build_tasks():
    tasks = []
    for ds in MSG_DATASETS:
        for shot in (1, 5):
            for r in range(5):
                key = f"d128_{ds}_{shot}shot_run{r}"
                tasks.append({
                    "kind": "msgstar",
                    "name": key,
                    "cost": COST_MSG[ds] * (1.0 if shot == 5 else 0.85),
                    "cmd": [PY, os.path.join(HERE, "run_msgstar_capacity.py"),
                            "--datasets", ds, "--shots", str(shot),
                            "--run-id", str(r), "--hidden-dim", "128",
                            "--out", os.path.join(TMP_MSG, key + ".json")],
                })
    for ds in ABL_DATASETS:
        for r in range(5):
            key = f"no_stmp_{ds}_1shot_run{r}"
            tasks.append({
                "kind": "abl_stmp",
                "name": key,
                "cost": COST_ABL[ds],
                "cmd": [PY, os.path.join(HERE, "run_ablation_50ep.py"),
                        "--variants", "no_stmp", "--datasets", ds,
                        "--shots", "1", "--run-id", str(r),
                        "--results-file", os.path.join(TMP_ABL, key + ".json")],
            })
    # Longest processing time first.
    tasks.sort(key=lambda t: -t["cost"])
    return tasks


def worker_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.join(POST, "code"), PYLIBS,
        env.get("PYTHONPATH", ""),
    ]).strip(os.pathsep)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def task_done(task):
    """A task counts as done if its per-task JSON already holds a result.

    This makes the driver cheaply resumable: relaunching after an interrupted
    session re-runs only the tasks that never produced a result file.
    """
    if task["kind"] == "msgstar":
        p = os.path.join(TMP_MSG, task["name"] + ".json")
    else:
        p = os.path.join(TMP_ABL, task["name"] + ".json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return False
    return any(isinstance(v, dict) and "final_acc" in v for v in d.values())


def run_task(task, env, log_dir, timeout=7200):
    """Run one task. `timeout` (s) kills a hung worker so a single stuck
    subprocess can never block the whole driver (the 2026-09-29 10:21 stall
    was caused by exactly this — no per-task timeout)."""
    os.makedirs(log_dir, exist_ok=True)
    logf = os.path.join(log_dir, task["name"] + ".log")
    t0 = time.time()
    try:
        with open(logf, "w") as fh:
            rc = subprocess.call(task["cmd"], cwd=PBB, env=env,
                                 stdout=fh, stderr=subprocess.STDOUT,
                                 timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = -9   # mark as timed out; driver logs it and moves on
    dt = time.time() - t0
    return task["name"], rc, dt


def merge(kind, tmp_dir, master, extra=None):
    merged = {}
    if os.path.exists(master):
        try:
            with open(master) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    if not os.path.isdir(tmp_dir):
        return merged
    for fn in sorted(os.listdir(tmp_dir)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(tmp_dir, fn)
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "final_acc" in v:
                    merged[k] = v
    if extra:
        merged.update(extra)
    tmp = master + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    os.replace(tmp, master)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run tasks whose per-task JSON already exists")
    ap.add_argument("--log-dir", default=os.path.join(RESULTS, "_queue_logs"))
    args = ap.parse_args()

    os.makedirs(TMP_MSG, exist_ok=True)
    os.makedirs(TMP_ABL, exist_ok=True)

    if not args.merge_only:
        tasks = build_tasks()
        if not args.force:
            before = len(tasks)
            tasks = [t for t in tasks if not task_done(t)]
            if before != len(tasks):
                print(f"[queue] skipped {before - len(tasks)} already-done task(s)",
                      flush=True)
        if not tasks:
            print("[queue] nothing to do", flush=True)
        env = worker_env()
        print(f"[queue] {len(tasks)} tasks, {args.workers} workers", flush=True)
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_task, t, env, args.log_dir): t for t in tasks}
            for fut in futs:
                pass
            from concurrent.futures import as_completed
            for fut in as_completed(futs):
                name, rc, dt = fut.result()
                done += 1
                print(f"[queue] ({done}/{len(tasks)}) {name} rc={rc} "
                      f"{dt:.0f}s  elapsed={time.time()-t0:.0f}s", flush=True)
                if done % 5 == 0:
                    merge("msgstar", TMP_MSG, MASTER_MSG)
                    merge("abl_stmp", TMP_ABL, MASTER_ABL)
        print(f"[queue] all tasks done in {time.time()-t0:.0f}s", flush=True)

    m1 = merge("msgstar", TMP_MSG, MASTER_MSG)
    m2 = merge("abl_stmp", TMP_ABL, MASTER_ABL)
    print(f"[merge] MSG* d=128 entries: {len(m1)} -> {MASTER_MSG}", flush=True)
    print(f"[merge] w/o STMP entries:  {len(m2)} -> {MASTER_ABL}", flush=True)


if __name__ == "__main__":
    main()

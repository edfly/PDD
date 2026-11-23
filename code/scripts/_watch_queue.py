# -*- coding: utf-8 -*-
"""Lightweight progress/ETA monitor for the capacity queue.

Polls results/_queue_logs/*.log every POLL seconds. Whenever a run prints a
new `Epoch N/50 ... <elapsed>s` line it derives the steady-state per-epoch
rate since the previous print and projects the remaining wall time for that
run. Prints one compact line per update, then exits when either
  * all per-task result JSONs are present, or
  * MAX_WAIT seconds have elapsed.
"""
import os
import re
import sys
import time
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(os.path.dirname(HERE))
LOGS = os.path.join(POST, "results", "_queue_logs")
TMP_MSG = os.path.join(POST, "results", "tmp_msgstar")
TMP_ABL = os.path.join(POST, "results", "tmp_abl_stmp")

POLL = 300
MAX_WAIT = 20 * 3600
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\s*(\d+)\s*\|.*\|\s*(\d+)s\s*$")


def snapshot():
    out = {}
    for p in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        name = os.path.basename(p)[:-4]
        try:
            with open(p, "r", errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        ms = EPOCH_RE.findall(txt)
        out[name] = [(int(a), int(b), int(c)) for a, b, c in ms]
    return out


def n_done():
    n = 0
    for d in (TMP_MSG, TMP_ABL):
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(p) as f:
                    j = json.load(f)
            except Exception:
                continue
            if any(isinstance(v, dict) and "final_acc" in v for v in j.values()):
                n += 1
    return n


def main():
    t_start = time.time()
    seen = {}
    prev = snapshot()
    last_done = n_done()
    print(f"[watch] start: {last_done}/65 per-task JSONs present, "
          f"{len(prev)} log(s) tracked", flush=True)
    while time.time() - t_start < MAX_WAIT:
        time.sleep(POLL)
        cur = snapshot()
        updated = 0
        for name, eps in cur.items():
            if not eps:
                continue
            e, tot, el = eps[-1]
            key = (name, e)
            if key in seen:
                continue
            seen[key] = True
            updated += 1
            pname = prev.get(name) or []
            rate = None
            if pname:
                pe, _, pel = pname[-1]
                if e > pe and el > pel:
                    rate = (el - pel) / (e - pe)
            if rate:
                remain = rate * (tot - e)
                print(f"[watch] {name}  epoch {e}/{tot} @{el}s  "
                      f"rate={rate:.0f}s/ep  eta_this_run={remain/60:.0f}min",
                      flush=True)
            else:
                print(f"[watch] {name}  epoch {e}/{tot} @{el}s", flush=True)
        prev = cur
        d = n_done()
        if d != last_done:
            print(f"[watch] results {last_done} -> {d}/65", flush=True)
            last_done = d
        if d >= 65:
            print(f"[watch] ALL 65 DONE in {(time.time()-t_start)/60:.0f}min",
                  flush=True)
            return
        if not updated:
            tick = int((time.time() - t_start) // POLL)
            if tick % 4 == 0:
                print(f"[watch] idle tick  results={last_done}/65  "
                      f"elapsed={(time.time()-t_start)/60:.0f}min", flush=True)
    print("[watch] MAX_WAIT reached", flush=True)


if __name__ == "__main__":
    main()

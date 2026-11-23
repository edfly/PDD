# -*- coding: utf-8 -*-
"""Backfill n>=20 extension runs into the canonical result files.

Splits tmp_n20 per-task JSONs into per-arm files matching each arm's
canonical consumption format, then neutralises the glob-landmine merged
file (its mixed-arm payload would be mis-ingested by collect_spikergnn's
v24_extra_* glob). Idempotent: merging into canonical files dedupes by key.
"""
import json, os, shutil, glob

R = r"D:/2026/paper/PBB/post/results"
TMP = os.path.join(R, "tmp_n20")

def load_all(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(TMP, pattern))):
        with open(p) as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, dict) and "final_acc" in v:
                out[k] = v
    return out

def merge_into(path, new_entries, expect_min):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    before = len(data)
    data.update(new_entries)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"{os.path.basename(path)}: {before} -> {len(data)} entries (+{len(new_entries)})")
    if len(data) < expect_min:
        raise SystemExit(f"FATAL: {path} has {len(data)} < expected {expect_min}")

# 1. spike -> fresh glob file (spike-only keys, raw format)
spike = load_all("spike_*.json")
spike = {k: v for k, v in spike.items() if v.get("run_id", 0) >= 5}
with open(os.path.join(R, "v24_extra_n20_spike.json"), "w") as f:
    json.dump(spike, f, indent=2, default=float)
print(f"v24_extra_n20_spike.json: {len(spike)} spike runs (expect 65)")

# 2. euc -> canonical euc file (keys already euc_* via tmp payloads? tmp keys
#    are raw DS_shot_runR; canonical wants euc_ prefix)
euc = {}
for k, v in load_all("euc_*.json").items():
    if v.get("run_id", 0) >= 5:
        euc[f"euc_{k}"] = v
merge_into(os.path.join(R, "spike_gnn_euc_v30.json"), euc, 5 * 10 + 75)

# 3. dense -> ann_baselines_v30.json (keys gcn_/gin_/sage_ already)
dense = {k: v for k, v in load_all("dense_*.json").items() if v.get("run_id", 0) >= 5}
merge_into(os.path.join(R, "ann_baselines_v30.json"), dense, 5 * 10 + 225)

# 4. msg -> msgstar_d128_results.json (keys d128_* already)
msg = {k: v for k, v in load_all("msg_*.json").items() if v.get("run_id", 0) >= 5}
merge_into(os.path.join(R, "msgstar_d128_results.json"), msg, 50 + 75)

# 5. neutralise the merged landmine (rename out of the v24_extra_* glob)
landmine = os.path.join(R, "v24_extra_n20_merged.json")
if os.path.exists(landmine):
    shutil.move(landmine, os.path.join(TMP, "v24_extra_n20_merged.BAK"))
    print("landmine merged file moved to tmp_n20/*.BAK")

print("counts: spike %d euc %d dense %d msg %d" % (len(spike), len(euc), len(dense), len(msg)))

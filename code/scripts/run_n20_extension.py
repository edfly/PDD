# -*- coding: utf-8 -*-
"""n>=20 extension driver (2026-11 power re-run) -- adds runs to the under-powered cells.

Scope (the final review, Section 8 minimal path, Top-1):
  - Tier-1 5-shot (MUTAG, PROTEINS, ENZYMES): every method, run_id 5..19
  - DD / REDDIT-M5K 1-shot: SpikeRPGNN run_id 10..19 (already has 10),
    every baseline run_id 5..19 (currently 5)

Arms
  spike / euc : replicated run_experiment from the v24 runner (research repo),
                ablation 'full' / 'euclidean', identical protocol
                (50 epochs, 20 episodes/epoch, 30 test episodes,
                seed = 42 + run_id*100 + shot*10, lambda_sr=0.5, r_target=0.4)
  dense       : code/run_dense_baseline.py subprocess per run (gcn/gin/sage)
  msg         : code/scripts/run_msgstar_capacity.py subprocess per run (CPU)

Output: one JSON per task under results/tmp_n20/<arm>_<key>.json; the
analyzer merges SpikeRPGNN cells via its v24_extra_*.json glob, so after
the merge step this driver also emits results/v24_extra_n20_<arm>.json in
the exact legacy format collect_spikergnn() expects.

Resumable: a task is skipped if its per-task JSON already holds a result.

Usage:
  python run_n20_extension.py --arm spike --datasets MUTAG --shots 5 --run-ids 5 6
  python run_n20_extension.py --arm euc   --datasets PROTEINS --shots 5 --run-ids 5 6
  python run_n20_extension.py --arm dense --backbone gin --datasets DD --shots 1 --run-ids 5 6
  python run_n20_extension.py --arm msg   --datasets REDDIT-M5K --shots 1 --run-ids 5 6
  python run_n20_extension.py --merge-only
NOTE: launch with cwd = D:/2026/paper/PBB (owns data/TU dataset cache).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))          # code/scripts
CODE = os.path.dirname(HERE)                                # code
POST = os.path.dirname(CODE)                                # post
PBB = os.path.dirname(POST)                                 # PBB (owns data/TU)
RESULTS = os.path.join(POST, "results")
TMP = os.path.join(RESULTS, "tmp_n20")
RESEARCH_REPO = r"D:/RiemannManifold/spikergnn"

DEFAULT_PY = sys.executable


def _effective_device(arm, ds, device):
    """Empirical device policy (2026-11-05 post-mortem, amended 11-07).

    REDDIT-M5K spiking batches do not fit 22 GiB during training: every
    episode OOMs in backward() and is silently skipped (skip=20/20 signature;
    forward-only eval still fits, which masks the failure). DD, probed on
    2026-11-07, trains at 1.2 GiB peak and stays on the requested device.
    Tier-1 spiking runs fine on GPU; large-graph spiking cells that exceed
    memory go to CPU, matching the device of the published runs they extend.
    Dense/msg arms are unaffected.
    """
    if arm in ("spike", "euc") and device == "cuda" and ds == "REDDIT-M5K":
        return "cpu"
    return device


def _lif_spike_rate(lif):
    """spike_rate() existed in the repo state that produced the published
    runs; the refactored neuron only stores the _spike_rate attribute.
    Accept either so both arms train on the current codebase."""
    sr = getattr(lif, "spike_rate", None)
    if callable(sr):
        return float(sr())
    return float(getattr(lif, "_spike_rate", 0.0))


def _to_float(x):
    """numpy cannot ingest CUDA tensors; normalize every collected scalar."""
    try:
        import torch as _t
        if isinstance(x, _t.Tensor):
            return float(x.detach().cpu())
    except Exception:
        pass
    return float(x)


def run_spike_like(ds_name, shot, run_id, ablation, device):
    """Faithful replica of run_full_experiments_v24.run_experiment with an
    ablation switch (research-repo code, which produced the published runs)."""
    # The paper-archived snapshot (post/code) is the canonical import for the
    # extension: it carries the device-portable curvature-cache fix, and its
    # numerics are re-validated against the published run0 values before each
    # batch (the live research repo may drift mid-refactor). CODE must precede
    # RESEARCH_REPO on sys.path, so insert it last.
    sys.path.insert(0, RESEARCH_REPO)  # fallback for any module not in CODE
    sys.path.insert(0, CODE)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from msg_baseline.config import DATASET_CONFIGS
    from msg_baseline.utils import set_seed
    from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
    from spikergnn.config import SpikeRPGNNConfig
    from spikergnn.model import create_spikergnn

    epochs, ep_per_epoch, test_eps = 50, 20, 30
    lambda_sr, r_target = 0.5, 0.4
    ds_cfg = DATASET_CONFIGS[ds_name]
    n_way = ds_cfg["n_way"]
    seed = 42 + run_id * 100 + shot * 10
    set_seed(seed)

    cfg = SpikeRPGNNConfig(
        epochs=1, episodes_per_epoch=ep_per_epoch, test_episodes=test_eps,
        lambda_sr=lambda_sr, r_target=r_target,
        learnable_c=False, ablation=ablation,
    )
    dataset = TUGraphDataset(name=ds_name, tu_name=ds_cfg["tu_name"],
                             train_ratio=0.8, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=n_way, k_shot=shot,
                             q_query=5, seed=seed)
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(cfg, in_dim=in_dim)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    for _l in model.stmp.rgc_layers:
        _l.track_euc_norm = True

    u_th_params = [p for n, p in model.named_parameters() if 'u_th_learnable' in n]
    other_params = [p for n, p in model.named_parameters() if 'u_th_learnable' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 0.001, 'weight_decay': 5e-4},
        {'params': u_th_params, 'lr': 0.01, 'weight_decay': 0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    r_target_t = torch.tensor(r_target)
    if device == "cuda" and torch.cuda.is_available():
        r_target_t = r_target_t.cuda()
    best_acc = 0.0
    spike_history, spike_rate_history = [], []

    def _maybe_cuda(b):
        if device == "cuda" and torch.cuda.is_available():
            return b.cuda()
        return b

    for epoch in range(epochs):
        if epoch < 3:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.001 * (epoch + 1) / 3.0
        model.train()
        total_acc, total_loss, total_sr_loss, total_spikes = 0, 0, 0, 0
        total_spike_rate, n_ok, n_skip = 0, 0, 0
        for ep in range(ep_per_epoch):
            try:
                episode = sampler.sample_episode(mode="train")
                s_b, s_l, q_b, q_l = batch_episode_data(episode)
                s_b, q_b = _maybe_cuda(s_b), _maybe_cuda(q_b)
                s_l, q_l = _maybe_cuda(s_l), _maybe_cuda(q_l)
                optimizer.zero_grad()
                s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                if torch.isnan(s_z[0]).any() or torch.isnan(q_z[0]).any():
                    n_skip += 1; continue
                logits = model.classify(q_z, s_z, s_l, n_way)
                if torch.isnan(logits).any():
                    n_skip += 1; continue
                loss_ce = nn.functional.cross_entropy(logits, q_l)
                if torch.isnan(loss_ce) or torch.isinf(loss_ce) or loss_ce.item() > 50:
                    n_skip += 1; continue
                if lambda_sr > 0:
                    r_bar = model.spike_rate_surrogate()
                    loss_sr = lambda_sr * (r_bar - r_target_t) ** 2
                    loss = loss_ce + loss_sr
                else:
                    loss = loss_ce
                    loss_sr = torch.tensor(0.0)
                loss.backward()
                has_bad_grad = any(
                    p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in model.parameters()
                )
                if has_bad_grad:
                    optimizer.zero_grad(); n_skip += 1; continue
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                with torch.no_grad():
                    model.stmp.lif.u_th_learnable.clamp_(0.7, 3.0)
                    model.mspe.lif.u_th_learnable.clamp_(0.7, 3.0)
                acc = (logits.argmax(-1) == q_l).float().mean().item()
                total_acc += acc
                total_loss += loss_ce.item()
                total_sr_loss += loss_sr.item()
                total_spikes += model.count_spikes()
                total_spike_rate += _lif_spike_rate(model.stmp.lif)
                n_ok += 1
            except Exception:
                # Silent skips are how a systemic failure (e.g. per-episode
                # OOM) hides behind a zero loss curve; surface the first one.
                if n_skip == 0:
                    traceback.print_exc()
                    print(f"  [warn] first skipped episode in {ds_name}_{shot}shot_run{run_id}; "
                          f"further skips silent", flush=True)
                n_skip += 1
        scheduler.step()
        if device == "cuda" and torch.cuda.is_available():
            # DD/Rdt-sized graphs fragment the 22 GiB card over 50 epochs;
            # an end-of-epoch flush is protocol-neutral insurance against a
            # run dying in its final eval (2026-11-05 OOM post-mortem).
            torch.cuda.empty_cache()
        spike_history.append(total_spikes)
        avg_spike_rate = total_spike_rate / max(n_ok, 1) if n_ok > 0 else 0
        spike_rate_history.append(avg_spike_rate)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or epoch < 3:
            model.eval()
            eval_accs, eval_spike_rates = [], []
            with torch.no_grad():
                for _ in range(min(test_eps, 20)):
                    try:
                        ep = sampler.sample_episode(mode="test")
                        s_b, s_l, q_b, q_l = batch_episode_data(ep)
                        s_b, q_b = _maybe_cuda(s_b), _maybe_cuda(q_b)
                        s_l, q_l = _maybe_cuda(s_l), _maybe_cuda(q_l)
                        s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                        q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                        if torch.isnan(s_z[0]).any():
                            continue
                        logits = model.classify(q_z, s_z, s_l, n_way)
                        if torch.isnan(logits).any():
                            continue
                        eval_accs.append((logits.argmax(-1) == q_l).float().mean().item())
                        eval_spike_rates.append(_to_float(model.stmp.lif._spike_rate))
                    except Exception:
                        continue
            ea = sum(eval_accs) / max(len(eval_accs), 1) if eval_accs else 0
            if ea > best_acc:
                best_acc = ea
            print(f"  E{epoch+1:3d}: train={total_acc/max(n_ok,1)*100:.1f}% "
                  f"test={ea*100:.1f}% skip={n_skip}/{ep_per_epoch} "
                  f"sr_train={avg_spike_rate:.3f}", flush=True)

    model.eval()
    final_accs, final_spikes, final_spike_rates = [], 0, []
    with torch.no_grad():
        for _ in range(test_eps):
            try:
                ep = sampler.sample_episode(mode="test")
                s_b, s_l, q_b, q_l = batch_episode_data(ep)
                s_b, q_b = _maybe_cuda(s_b), _maybe_cuda(q_b)
                s_l, q_l = _maybe_cuda(s_l), _maybe_cuda(q_l)
                s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                logits = model.classify(q_z, s_z, s_l, n_way)
                final_accs.append((logits.argmax(-1) == q_l).float().mean().item())
                final_spikes += model.count_spikes()
                final_spike_rates.append(_to_float(model.stmp.lif._spike_rate))
            except Exception:
                continue

    fa = sum(final_accs) / max(len(final_accs), 1) if final_accs else 0
    avg_final_sr = float(np.mean(final_spike_rates)) if final_spike_rates else 0
    out = {
        "dataset": ds_name, "shot": shot, "run_id": run_id,
        "final_acc": fa, "best_acc": best_acc,
        "total_spikes": final_spikes,
        "spike_history_last5": spike_history[-5:],
        "spike_rate_history_last5": spike_rate_history[-5:],
        "final_spike_rate": avg_final_sr,
        "u_th_learnable": model.stmp.lif.u_th_learnable.item(),
        "lambda_gate": torch.sigmoid(model.gdc.lam_raw).item(),
        "spike_alpha": torch.sigmoid(model.stmp.lif.spike_alpha).item(),
        "lambda_sr": lambda_sr, "r_target": r_target,
    }
    if ablation == "euclidean":
        out["variant"] = "euclidean"
    return out


def task_key(arm, backbone, ds, shot, r):
    # dense tasks MUST be qualified by backbone: gcn/gin/sage share the same
    # (ds, shot, run-id) space, and an unqualified key silently marks gin and
    # sage "done" the moment gcn's file lands (2026-11-05 incident).
    bb = f"{backbone}_" if arm == "dense" else ""
    return f"{arm}_{bb}{ds}_{shot}shot_run{r}"


def task_path(arm, backbone, ds, shot, r):
    return os.path.join(TMP, task_key(arm, backbone, ds, shot, r) + ".json")


def task_done(arm, backbone, ds, shot, r):
    p = task_path(arm, backbone, ds, shot, r)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return False
    return any(isinstance(v, dict) and "final_acc" in v for v in d.values())


def run_one(args, arm, ds, shot, r, env):
    key = task_key(arm, args.backbone, ds, shot, r)
    out_path = task_path(arm, args.backbone, ds, shot, r)
    os.makedirs(TMP, exist_ok=True)
    t0 = time.time()
    if arm in ("spike", "euc"):
        # One subprocess per task, mirroring the dense/msg contract: a CUDA
        # OOM poisons the whole context of a process, and an in-process arm
        # turns a single failed run into a dead queue (2026-11-05: the first
        # DD OOM killed every subsequent DD/Rdt spiking task in-process).
        eff_dev = _effective_device(arm, ds, args.device)
        if eff_dev != args.device:
            print(f"  [device] {key}: {args.device} -> {eff_dev} "
                  f"(large-graph spiking batch does not fit GPU memory)", flush=True)
        env2 = dict(env)
        env2["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if eff_dev == "cpu":
            # RAM evidence (2026-11-07): each Rdt spiking worker peaks at
            # 7-9 GiB working set, so TWO concurrent workers exhaust the
            # 24 GiB host (rc=3221225477 crashes on 1-13 MB allocations).
            # Default to a single fat worker; a paired queue may set
            # N20_OMP_THREADS=2 explicitly and accept the RAM risk.
            nthr = os.environ.get("N20_OMP_THREADS", "4")
            env2["OMP_NUM_THREADS"] = nthr
            env2["MKL_NUM_THREADS"] = nthr
        cmd = [DEFAULT_PY, os.path.abspath(__file__),
               "--worker", "--arm", arm, "--datasets", ds,
               "--shots", str(shot), "--run-ids", str(r),
               "--device", eff_dev]
        rc = subprocess.call(cmd, cwd=PBB, env=env2)
        if rc != 0:
            raise RuntimeError(f"{arm} task {key} rc={rc}")
        with open(out_path) as f:
            payload = json.load(f)
    elif arm == "dense":
        cmd = [DEFAULT_PY, os.path.join(CODE, "run_dense_baseline.py"),
               "--backbone", args.backbone, "--datasets", ds,
               "--shots", str(shot), "--run-id", str(r),
               "--device", args.device, "--out", out_path]
        rc = subprocess.call(cmd, cwd=PBB, env=env)
        if rc != 0:
            raise RuntimeError(f"dense task {key} rc={rc}")
        with open(out_path) as f:
            payload = json.load(f)
    elif arm == "msg":
        cmd = [DEFAULT_PY, os.path.join(HERE, "run_msgstar_capacity.py"),
               "--datasets", ds, "--shots", str(shot), "--run-id", str(r),
               "--hidden-dim", "128", "--out", out_path]
        env2 = dict(env)
        env2["CUDA_VISIBLE_DEVICES"] = ""   # MSG* stays on CPU (original protocol)
        rc = subprocess.call(cmd, cwd=PBB, env=env2)
        if rc != 0:
            raise RuntimeError(f"msg task {key} rc={rc}")
        with open(out_path) as f:
            payload = json.load(f)
    else:
        raise ValueError(arm)
    if arm in ("spike", "euc"):
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=float)
    print(f">>> {key}: done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["spike", "euc", "dense", "msg"], required=False)
    ap.add_argument("--backbone", default="gin", choices=["gcn", "gin", "sage"])
    ap.add_argument("--datasets", nargs="+", required=False)
    ap.add_argument("--shots", nargs="+", type=int, required=False)
    ap.add_argument("--run-ids", nargs="+", type=int, required=False)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--worker", action="store_true",
                    help="internal: run exactly one spike/euc task (subprocess "
                         "isolation contract; parent driver passes one value "
                         "per --datasets/--shots/--run-ids)")
    args = ap.parse_args()

    if args.merge_only:
        merge()
        return

    if args.worker:
        # Single-task mode: the parent driver has already resolved arm/ds/
        # shot/run-id. A crash here kills only this task; the parent records
        # the failure and moves on, and a relaunch resumes cleanly.
        os.makedirs(TMP, exist_ok=True)
        ds, shot, r = args.datasets[0], args.shots[0], args.run_ids[0]
        out_path = task_path(args.arm, args.backbone, ds, shot, r)
        ablation = "full" if args.arm == "spike" else "euclidean"
        import time as _t
        t0 = _t.time()
        result = run_spike_like(ds, shot, r, ablation, args.device)
        result["elapsed_sec"] = round(_t.time() - t0, 1)
        result["device"] = args.device
        payload = {f"{ds}_{shot}shot_run{r}": result}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"worker {task_key(args.arm, args.backbone, ds, shot, r)}: ok", flush=True)
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        CODE, RESEARCH_REPO, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PYTHONUNBUFFERED"] = "1"

    tasks = [(args.arm, ds, shot, r)
             for ds in args.datasets for shot in args.shots for r in args.run_ids]
    # longest-job-first by dataset cost for stable scheduling
    cost = {"MUTAG": 1, "PROTEINS": 2, "ENZYMES": 2, "DD": 8, "REDDIT-M5K": 20}
    tasks.sort(key=lambda t: -cost.get(t[1], 5))
    for arm, ds, shot, r in tasks:
        if task_done(arm, args.backbone, ds, shot, r):
            print(f"SKIP {task_key(arm, args.backbone, ds, shot, r)}", flush=True)
            continue
        print(f"\n=== {task_key(arm, args.backbone, ds, shot, r)} ===", flush=True)
        try:
            run_one(args, arm, ds, shot, r, env)
        except Exception:
            traceback.print_exc()
    merge()


def merge():
    """Rebuild results/v24_extra_n20_<arm>.json from per-task JSONs.

    SpikeRPGNN-format arms (spike, euc) are merged into files named
    v24_extra_n20_*.json under results/ so collect_spikergnn() picks them
    up; dense/msg arms are appended into *_n20_extra.json companions of
    their master files (merged downstream by the analysis pipeline).
    """
    merged = {}
    if not os.path.isdir(TMP):
        print("no tmp_n20 dir")
        return
    for fn in sorted(os.listdir(TMP)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(TMP, fn)) as f:
            try:
                d = json.load(f)
            except Exception:
                continue
        # Namespace payload keys by arm so the merged file matches each
        # arm's canonical consumption format AND arms cannot overwrite each
        # other (2026-11-05: euc Tier-1 silently replaced spike Tier-1 in
        # the merge, 45 results lost, because both emit "DS_shot_runR").
        if fn.startswith("spike_"):
            remap = lambda k: k
        elif fn.startswith("euc_"):
            remap = lambda k: f"euc_{k}"
        else:
            remap = lambda k: k          # dense keys are backbone-prefixed,
                                         # msg keys are d128_-prefixed already
        for k, v in d.items():
            if isinstance(v, dict) and "final_acc" in v and v.get("run_id", 0) >= 5:
                merged[remap(k)] = v
    out = os.path.join(RESULTS, "v24_extra_n20_merged.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, default=float)
    print(f"merged {len(merged)} runs -> {out}")


if __name__ == "__main__":
    main()

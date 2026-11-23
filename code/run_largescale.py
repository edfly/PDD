# -*- coding: utf-8 -*-
"""Large-scale benchmark runner for SpikeRPGNN (audit extension).

Purpose
-------
The manuscript concedes that its largest benchmark (REDDIT-MULTI-5K,
4,999 graphs) is "modest by contemporary graph-learning standards". This
runner adds the two largest TU datasets that are already staged on disk
(NCI1 4,110 graphs; NCI109 4,127 graphs) under the *identical* episodic
protocol, so the new cells are comparable with the published main matrix
rather than being a second, incompatible protocol.

Design rule
-----------
The training loop below is extracted verbatim from
scripts/run_full_experiments_v24.py and only the device plumbing is added,
so a large-scale number cannot diverge from a main-matrix number through
a silently rewritten protocol. Do not edit the loop here; if the protocol
changes, regenerate.

GPU: device defaults to "cuda". The project rule is GPU-first; the original
runner was CPU-only because it never moved the model or the batches.

Usage
-----
    python code/run_largescale.py --smoke
    python code/run_largescale.py --datasets NCI1 NCI109 --shots 1 5 --n-runs 5

Output
------
    results/largescale.json   (new file; never overwrites v30 artefacts)
"""
import sys, os, json, time, argparse, traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Data lives at PBB/data (repo root), the TU loader resolves "./data/TU".
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
os.chdir(_REPO)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np

from msg_baseline.config import DATASET_CONFIGS
from msg_baseline.utils import set_seed
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

# ── Large-scale TU datasets, injected at runtime ──────────────────
# Deliberately NOT written into msg_baseline/config.py: that file is read
# by the verification gates, and adding datasets there would change
# ALL_DATASETS and therefore every gate that enumerates the main matrix.
# Runtime injection keeps the published matrix untouched.
# Graph counts and class counts were measured from the staged data
# (TUDataset(root=./data/TU)), not copied from a paper.
DATASET_CONFIGS["NCI1"] = {
    "n_way": 2, "shots": [1, 5], "n_graphs": 4110,
    "n_classes": 2, "tu_name": "NCI1",
}
DATASET_CONFIGS["NCI109"] = {
    "n_way": 2, "shots": [1, 5], "n_graphs": 4127,
    "n_classes": 2, "tu_name": "NCI109",
}

RESULTS_PATH = os.path.join(_REPO, "post", "results", "largescale.json")


def run_experiment(ds_name, shot, run_id, epochs=50, ep_per_epoch=20,
                  test_eps=30, lambda_sr=0.5, r_target=0.4,
                  learnable_c=False, device="cuda"):
    """Run one experiment with spike rate regularization.

    Args:
        ds_name: Dataset name (e.g. MUTAG, PROTEINS).
        shot: K-shot value (1 or 5).
        run_id: Run index (0-4).
        epochs: Number of training epochs.
        ep_per_epoch: Episodes per epoch.
        test_eps: Number of test episodes for evaluation.
        lambda_sr: Spike rate regularization coefficient (Eq.spike_rate_reg).
        r_target: Target spike rate r* (default 0.4).
        learnable_c: If True, treat curvature c as learnable.
    """
    ds_cfg = DATASET_CONFIGS[ds_name]
    n_way = ds_cfg["n_way"]
    seed = 42 + run_id * 100 + shot * 10
    set_seed(seed)

    cfg = SpikeRPGNNConfig(
        epochs=1, episodes_per_epoch=ep_per_epoch, test_episodes=test_eps,
        lambda_sr=lambda_sr, r_target=r_target,
        learnable_c=learnable_c,
    )
    dataset = TUGraphDataset(name=ds_name, tu_name=ds_cfg["tu_name"], train_ratio=0.8, seed=seed)
    sampler = EpisodeSampler(dataset=dataset, n_way=n_way, k_shot=shot, q_query=5, seed=seed)
    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(cfg, in_dim=in_dim).to(device)

    # Re-enable the euc_max_norm diagnostic (gated off by default to avoid a
    # per-RGC host sync in the hot path); this runner reads it for paper evidence.
    for _l in model.stmp.rgc_layers:
        _l.track_euc_norm = True

    # Separate optimizer groups: u_th_learnable gets 10x smaller LR
    # to prevent rapid threshold changes that destabilize spike rate.
    u_th_params = [p for n, p in model.named_parameters() if 'u_th_learnable' in n]
    other_params = [p for n, p in model.named_parameters() if 'u_th_learnable' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 0.001, 'weight_decay': 5e-4},
        {'params': u_th_params, 'lr': 0.01, 'weight_decay': 0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    r_target_t = torch.tensor(r_target, device=device)  # for loss computation
    best_acc = 0.0
    spike_history = []
    spike_rate_history = []

    for epoch in range(epochs):
        # LR warmup
        if epoch < 3:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.001 * (epoch + 1) / 3.0

        model.train()
        total_acc, total_loss, total_sr_loss, total_spikes = 0, 0, 0, 0
        total_spike_rate, n_ok, n_skip = 0, 0, 0

        for ep in range(ep_per_epoch):
            try:
                episode = sampler.sample_episode(mode="train")
                s_b, s_l, q_b, q_l = batch_episode_data(episode, device)
                optimizer.zero_grad()

                s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                if torch.isnan(s_z[0]).any() or torch.isnan(q_z[0]).any():
                    n_skip += 1; continue

                logits = model.classify(q_z, s_z, s_l, n_way)
                if torch.isnan(logits).any():
                    n_skip += 1; continue

                # ── Classification loss ──────────────────────────
                loss_ce = nn.functional.cross_entropy(logits, q_l)
                if torch.isnan(loss_ce) or torch.isinf(loss_ce) or loss_ce.item() > 50:
                    n_skip += 1; continue

                # ── Spike rate regularization (Eq.spike_rate_reg) ─
                if lambda_sr > 0:
                    r_bar = model.spike_rate_surrogate()  # differentiable spike rate
                    loss_sr = lambda_sr * (r_bar - r_target_t) ** 2
                    loss = loss_ce + loss_sr
                else:
                    loss = loss_ce
                    loss_sr = torch.tensor(0.0, device=device)

                loss.backward()

                # NaN gradient check
                has_bad_grad = any(
                    p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in model.parameters()
                )
                if has_bad_grad:
                    optimizer.zero_grad()
                    n_skip += 1; continue

                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                # Clamp u_th to prevent runaway spikes
                with torch.no_grad():
                    model.stmp.lif.u_th_learnable.clamp_(0.7, 3.0)
                    model.mspe.lif.u_th_learnable.clamp_(0.7, 3.0)

                acc = (logits.argmax(-1) == q_l).float().mean().item()
                total_acc += acc
                total_loss += loss_ce.item()
                total_sr_loss += loss_sr.item()
                total_spikes += model.count_spikes()
                total_spike_rate += model.stmp.lif.spike_rate()
                n_ok += 1

            except Exception as e:
                n_skip += 1

        scheduler.step()
        spike_history.append(total_spikes)
        avg_spike_rate = total_spike_rate / max(n_ok, 1) if n_ok > 0 else 0
        spike_rate_history.append(avg_spike_rate)

        # Periodic eval
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or epoch < 3:
            model.eval()
            eval_accs = []
            eval_spike_rates = []
            with torch.no_grad():
                for _ in range(min(test_eps, 20)):
                    try:
                        ep = sampler.sample_episode(mode="test")
                        s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
                        s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                        q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                        if torch.isnan(s_z[0]).any(): continue
                        logits = model.classify(q_z, s_z, s_l, n_way)
                        if torch.isnan(logits).any(): continue
                        eval_accs.append((logits.argmax(-1) == q_l).float().mean().item())
                        eval_spike_rates.append(model.stmp.lif.spike_rate())
                    except: continue

            ea = sum(eval_accs) / max(len(eval_accs), 1) if eval_accs else 0
            if ea > best_acc: best_acc = ea
            avg_eval_sr = np.mean(eval_spike_rates) if eval_spike_rates else 0
            max_norm = max(l.euc_max_norm for l in model.stmp.rgc_layers)
            print(f"  E{epoch+1:3d}: train={total_acc/max(n_ok,1)*100:.1f}% test={ea*100:.1f}% "
                  f"skip={n_skip}/{ep_per_epoch} u_th={model.stmp.lif.u_th_learnable.item():.3f} "
                  f"spikes={total_spikes} sr_train={avg_spike_rate:.3f} sr_eval={avg_eval_sr:.3f} "
                  f"L_sr={total_sr_loss/max(n_ok,1):.4f} maxnorm={max_norm:.1f}", flush=True)

    # Final eval
    model.eval()
    final_accs = []
    final_spikes = 0
    final_spike_rates = []
    with torch.no_grad():
        for _ in range(test_eps):
            try:
                ep = sampler.sample_episode(mode="test")
                s_b, s_l, q_b, q_l = batch_episode_data(ep, device)
                s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                logits = model.classify(q_z, s_z, s_l, n_way)
                final_accs.append((logits.argmax(-1) == q_l).float().mean().item())
                final_spikes += model.count_spikes()
                final_spike_rates.append(model.stmp.lif.spike_rate())
            except: continue

    fa = sum(final_accs) / max(len(final_accs), 1) if final_accs else 0
    avg_final_sr = np.mean(final_spike_rates) if final_spike_rates else 0

    return {
        "dataset": ds_name, "shot": shot, "run_id": run_id,
        "final_acc": fa, "best_acc": best_acc,
        "total_spikes": final_spikes,
        "spike_history_last5": spike_history[-5:],
        "spike_rate_history_last5": spike_rate_history[-5:],
        "final_spike_rate": avg_final_sr,
        "u_th_learnable": model.stmp.lif.u_th_learnable.item(),
        "lambda_gate": torch.sigmoid(model.gdc.lam_raw).item(),
        "spike_alpha": torch.sigmoid(model.stmp.lif.spike_alpha).item(),
        "lambda_sr": lambda_sr,
        "r_target": r_target,
    }


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(rows):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def already_done(rows, ds, shot, run_id, lambda_sr):
    for r in rows:
        if (r.get("dataset") == ds and r.get("shot") == shot
                and r.get("run_id") == run_id
                and r.get("lambda_sr") == lambda_sr):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NCI1", "NCI109"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ep-per-epoch", type=int, default=20)
    ap.add_argument("--test-eps", type=int, default=30)
    ap.add_argument("--lambda-sr", type=float, default=0.5)
    ap.add_argument("--r-target", type=float, default=0.4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.datasets = [args.datasets[0]]
        args.shots = [args.shots[0]]
        args.n_runs = 1
        args.epochs = 2
        args.ep_per_epoch = 2
        args.test_eps = 3

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[abort] cuda requested but unavailable")
        sys.exit(1)
    print("device =", args.device, "|", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "cpu")

    rows = load_results()
    t0 = time.time()
    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            print("[skip] unknown dataset", ds)
            continue
        for shot in args.shots:
            for run_id in range(args.n_runs):
                if already_done(rows, ds, shot, run_id, args.lambda_sr):
                    print("[skip done] %s %d-shot run%d" % (ds, shot, run_id))
                    continue
                try:
                    out = run_experiment(
                        ds, shot, run_id, epochs=args.epochs,
                        ep_per_epoch=args.ep_per_epoch, test_eps=args.test_eps,
                        lambda_sr=args.lambda_sr, r_target=args.r_target,
                        device=args.device)
                    out["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    rows.append(out)
                    save_results(rows)
                    print("[ok] %s %d-shot run%d acc=%.4f r_bar=%.4f"
                          % (ds, shot, run_id, out["final_acc"],
                             out["final_spike_rate"]), flush=True)
                except Exception:
                    traceback.print_exc()
                    print("[fail] %s %d-shot run%d" % (ds, shot, run_id), flush=True)
    print("elapsed %.1f s" % (time.time() - t0))
    print("wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()

"""
SpikeRPGNN Full Experiment Runner — v24 with spike rate regularization.

Key changes from v23:
- Added spike rate regularization loss: L_sr = lambda_sr * (r_bar - r_target)^2
- Tracks spike_rate and L_sr contribution per episode
- Output: spikergnn_results/full_results_v24.json

Usage:
  python run_full_experiments_v24.py                    # Full: 5 datasets x 2 shots x 5 runs
  python run_full_experiments_v24.py --quick             # Quick test: 1 dataset x 1 run x 5 epochs
  python run_full_experiments_v24.py --dataset MUTAG     # Single dataset
  python run_full_experiments_v24.py --resume            # Resume from saved results
  python run_full_experiments_v24.py --lambda_sr 0.0     # Disable spike rate reg (ablation)
"""
import sys, os, json, time, traceback

# ── Environment setup ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from msg_baseline.config import DATASET_CONFIGS, ALL_DATASETS
from msg_baseline.utils import set_seed
from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

RESULTS_DIR = "spikergnn_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "full_results_v24.json")


def run_experiment(ds_name, shot, run_id, epochs=50, ep_per_epoch=20,
                  test_eps=30, lambda_sr=0.5, r_target=0.4,
                  learnable_c=False):
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
        learnable_c: If True, treat curvature c as learnable .
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
    model = create_spikergnn(cfg, in_dim=in_dim)

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

    r_target_t = torch.tensor(r_target)  # for loss computation
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
                s_b, s_l, q_b, q_l = batch_episode_data(episode)
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
                    loss_sr = torch.tensor(0.0)

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
                        s_b, s_l, q_b, q_l = batch_episode_data(ep)
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
                s_b, s_l, q_b, q_l = batch_episode_data(ep)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--shot", type=int, default=None, choices=[1, 5])
    parser.add_argument("--lambda_sr", type=float, default=0.5,
                        help="Spike rate regularization coefficient (0.0 to disable)")
    parser.add_argument("--r_target", type=float, default=0.4,
                        help="Target spike rate r* (default 0.4)")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Override results JSON path (for parallel tasks writing separate files)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="run only this run index (orchestrator contract)")
    args = parser.parse_args()

    results_file = args.results_file if args.results_file else RESULTS_FILE

    epochs = 5 if args.quick else 50
    n_runs = 1 if args.quick else 5
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    shots = [args.shot] if args.shot else [1, 5]

    # Load existing
    results = {}
    if args.resume and os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

    print(f"\nSpikeRPGNN v24 Experiment: {datasets} x {shots} x {n_runs} runs x {epochs} epochs", flush=True)
    print(f"  lambda_sr={args.lambda_sr}, r_target={args.r_target}\n", flush=True)

    # Uniform runner contract (see exp_spec.py): --run-id restricts execution
    # to a single run so a driver can bound the cost of a hang. Default None
    # preserves the legacy "run everything" behaviour.
    run_ids = ([args.run_id] if args.run_id is not None
               else list(range(n_runs)))

    for ds_name in datasets:
        for shot in shots:
            for run_id in run_ids:
                key = f"{ds_name}_{shot}shot_run{run_id}"
                if key in results and "final_acc" in results.get(key, {}):
                    print(f"SKIP {key}", flush=True)
                    continue

                print(f"\n=== {key} ===", flush=True)
                t0 = time.time()
                try:
                    result = run_experiment(
                        ds_name, shot, run_id, epochs=epochs,
                        lambda_sr=args.lambda_sr, r_target=args.r_target,
                    )
                    result["elapsed_sec"] = round(time.time() - t0, 1)
                    results[key] = result
                    print(f">>> {key}: acc={result['final_acc']*100:.2f}% best={result['best_acc']*100:.2f}% "
                          f"spikes={result['total_spikes']} sr={result['final_spike_rate']:.3f} "
                          f"u_th={result['u_th_learnable']:.4f} "
                          f"time={result['elapsed_sec']:.0f}s", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    results[key] = {"error": str(e), "dataset": ds_name, "shot": shot}

                os.makedirs(RESULTS_DIR, exist_ok=True)
                # Atomic + retried save: tolerate transient file locks (AV/sandbox)
                # and never lose the whole JSON on a single failed write.
                tmp_file = results_file + ".tmp"
                last_err = None
                for attempt_save in range(6):
                    try:
                        with open(tmp_file, "w") as f:
                            json.dump(results, f, indent=2, default=float)
                        os.replace(tmp_file, results_file)
                        last_err = None
                        break
                    except PermissionError as pe:
                        last_err = pe
                        print(f"[save] PermissionError (attempt {attempt_save + 1}), retrying in 10s...", flush=True)
                        time.sleep(10)
                if last_err is not None:
                    # last resort: dump to a sidecar so data is never lost
                    sidecar = results_file + f".{key}.bak"
                    with open(sidecar, "w") as f:
                        json.dump(results, f, indent=2, default=float)
                    print(f"[save] PRIMARY SAVE FAILED; wrote sidecar {sidecar}", flush=True)
                    traceback.print_exc()

    # Summary
    print(f"\n{'='*60}\n  RESULTS SUMMARY (v24)\n{'='*60}")
    for ds_name in datasets:
        for shot in shots:
            accs = [results[k]["final_acc"] for k in results
                    if k.startswith(f"{ds_name}_{shot}shot") and "final_acc" in results[k]]
            srs = [results[k]["final_spike_rate"] for k in results
                   if k.startswith(f"{ds_name}_{shot}shot") and "final_spike_rate" in results[k]]
            if accs:
                sr_str = f" sr={np.mean(srs):.3f}" if srs else ""
                print(f"  {ds_name:<15} {shot}-shot: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%{sr_str}")

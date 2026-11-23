# -*- coding: utf-8 -*-
"""Train a small MUTAG model (full variant, ~30 epochs) for SOP event-count
cross-validation (Step 3 of the energy simulation-level measurement plan).

Purpose:
  Step 1/2 already proved the A/B SOP counting code is unbiased (discrepancy=0)
  using a randomly-initialized model. That is a CODE-correctness proof, not a
  MEANINGFUL-NUMBER proof: a random model fires in full synchrony (period-2),
  so its SOP is uninformative.

  This script trains a real MUTAG model briefly (30 epochs, a few minutes),
  saves a checkpoint, and reports the trained model's firing rate + test acc,
  so the SOP cross-validation can run against a model whose spike statistics
  resemble the paper's (r_target = 0.4).

  It does NOT change any reported paper number. It only produces a checkpoint
  usable by verify_sop_counting.py to run the A/B reconciliation on a trained
  model.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.train import run_single_experiment

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(ROOT, "spikergnn_results", "mutag_sop_checkpoint.pt")

EPOCHS = 30


def main():
    config = SpikeRPGNNConfig(
        hidden_dim=128, num_layers=3, tau=20.0, use_dvm=True,
        ablation="full", epochs=EPOCHS,
    )

    # Run one experiment. run_single_experiment builds + trains the model but
    # does not return it; we re-create the identical model and manually run a
    # short loop so we can retain the state_dict for the SOP reconciliation.
    from msg_baseline.config import DATASET_CONFIGS
    from msg_baseline.data_utils import TUGraphDataset, EpisodeSampler, batch_episode_data
    from msg_baseline.utils import set_seed, compute_accuracy
    from spikergnn.model import create_spikergnn
    import torch.nn as nn
    import torch.optim as optim

    ds_cfg = DATASET_CONFIGS["MUTAG"]
    n_way = ds_cfg["n_way"]
    shot = 1
    run_id = 0
    seed = config.seed_base + run_id * 100 + shot * 10
    set_seed(seed)

    dataset = TUGraphDataset(
        name="MUTAG", tu_name=ds_cfg["tu_name"],
        train_ratio=config.train_ratio, seed=seed,
    )
    sampler = EpisodeSampler(
        dataset=dataset, n_way=n_way, k_shot=shot,
        q_query=config.q_query, seed=seed,
    )

    in_dim = max(dataset.num_features, 1)
    model = create_spikergnn(config, in_dim=in_dim)

    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5)

    best_acc = 0.0
    print(f"Training MUTAG full-variant {EPOCHS} epochs "
          f"(ep/epoch={config.episodes_per_epoch}, T={config.T}, "
          f"lambda_sr={config.lambda_sr}, r_target={config.r_target})", flush=True)

    for epoch in range(config.epochs):
        model.train()
        total_acc = 0.0
        total_spikes = 0
        n_ep = config.episodes_per_epoch
        for ep in range(n_ep):
            episode = sampler.sample_episode(mode="train")
            s_b, s_l, q_b, q_l = batch_episode_data(episode, device="cpu")
            optimizer.zero_grad()
            s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
            q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
            logits = model.classify(q_z, s_z, s_l, sampler.n_way)
            loss = nn.functional.cross_entropy(logits, q_l)
            if config.lambda_sr > 0:
                r_bar = model.spike_rate_surrogate()
                loss = loss + config.lambda_sr * (r_bar - config.r_target) ** 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            total_acc += compute_accuracy(logits.detach(), q_l)
            total_spikes += model.count_spikes()
        scheduler.step()

        # Evaluate
        model.eval()
        total_test_acc = 0.0
        total_test_spikes = 0
        n_test = config.test_episodes
        with torch.no_grad():
            for ep in range(n_test):
                episode = sampler.sample_episode(mode="test")
                s_b, s_l, q_b, q_l = batch_episode_data(episode, device="cpu")
                model.stmp.lif.reset_membrane()
                s_z = model.encode(s_b.x, s_b.edge_index, s_b.batch)
                q_z = model.encode(q_b.x, q_b.edge_index, q_b.batch)
                logits = model.classify(q_z, s_z, s_l, sampler.n_way)
                total_test_acc += compute_accuracy(logits, q_l)
                total_test_spikes += model.count_spikes()
        test_acc = total_test_acc / n_test
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            r_bar = model.spike_rate_surrogate()
            print(f"  epoch {epoch+1:2d}: train_acc={total_acc/n_ep*100:.1f}% "
                  f"test_acc={test_acc*100:.1f}% train_spikes={total_spikes} "
                  f"r_bar={float(r_bar):.3f}", flush=True)

    # Save checkpoint (state_dict only; model is CPU)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {
        "hidden_dim": 128, "num_layers": 3, "tau": 20.0, "use_dvm": True,
        "ablation": "full", "epochs": EPOCHS, "in_dim": in_dim,
        "T": config.T, "d_h": config.d_h, "d_e": config.d_e,
    }}, CKPT_PATH)

    print(f"\nSaved checkpoint -> {CKPT_PATH}")
    print(f"best_test_acc={best_acc*100:.2f}%  "
          f"final_r_bar={float(model.spike_rate_surrogate()):.3f}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Smoke test for layer2_shd.py: validate FlatManifold.split fix + SHD forward/backward.
Uses SHD *train* split for both tr and te (test split not needed for smoke)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_shd import (FlatSpikeSequenceNet, SpikeSequenceNet, load_shd_full,
                         build_net, train_one, count_params, _mean_ci)
import numpy as np, argparse, types

# minimal args
args = types.SimpleNamespace(
    root=r"D:\MSVN\data\SHD", epochs=3, lr=1e-3, hidden_dim=128, d_h=32, T=25,
    lambda_sr=0.1, r_target=0.4, batch=32, n_seeds=1, seed0=0)

# import device helper
from spikergnn.experiments.common import get_device, set_seed
device = get_device()
print(f"[smoke] device={device}")

# param parity
p_man = count_params(build_net("manifold", 700, 20, args).cpu())
p_flat = count_params(build_net("flat", 700, 20, args).cpu())
print(f"[smoke] param parity: manifold={p_man} flat={p_flat} "
      f"{'OK' if p_man==p_flat else 'MISMATCH!'}")

# Use the (already-valid) TEST split for both tr and te so the smoke runs
# without waiting for the slow train download. Validates the full real-data
# pipeline: load -> manifold/flat forward -> train -> acc.
tr = load_shd_full(args.root, T=args.T, split="test")
print(f"[smoke] loaded test samples={len(tr)}  sample0: x.shape={tr[0][0].shape}, label={tr[0][1]}, spk={tr[0][2]}")
# small subset so smoke is fast
tr = tr[:512]
te = tr  # test-as-test for smoke
r_man = train_one("manifold", tr, te, 700, 20, args, device, 0)
r_flat = train_one("flat", tr, te, 700, 20, args, device, 0)
print(f"[smoke] manifold test_acc={r_man['test_acc']:.4f} shift_acc={r_man['shift_acc']:.4f} params={r_man['params']}")
print(f"[smoke] flat     test_acc={r_flat['test_acc']:.4f} shift_acc={r_flat['shift_acc']:.4f} params={r_flat['params']}")
print("[smoke] DONE - shape fix works, both nets train on SHD")

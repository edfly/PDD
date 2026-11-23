"""End-to-end loop smoke test for run_nodelevel.run_one on a TINY graph.

Patches NodePropPredDataset with a 2000-node synthetic graph so the full
run_one pipeline (load -> sample_batch train loop -> evaluate accumulation)
executes in seconds on CPU. Validates spikergnn AND gcn baselines without the
1.5h cost of the real 169k graph.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_nodelevel as M


class FakeDS:
    def __init__(self, *args, **kwargs):
        N, F, C, E = 2000, 16, 5, 8000
        self.labels = torch.randint(0, C, (N, 1))
        ei = torch.randint(0, N, (2, E))
        self.graph = {
            "node_feat": torch.randn(N, F).numpy(),
            "edge_index": ei.numpy(),
        }
        self._N = N

    def get_idx_split(self):
        N = self._N
        return {
            "train": torch.arange(0, int(N * 0.7)),
            "valid": torch.arange(int(N * 0.7), int(N * 0.85)),
            "test": torch.arange(int(N * 0.85), N),
        }


M.NodePropPredDataset = FakeDS

for kind in ["spikergnn", "gcn"]:
    res = M.run_one(kind, 0, epochs=2, device="cpu", in_dim=16, out_dim=5,
                    batch_size=256)
    print(f"{kind}: {res}")
    assert 0.0 <= res["best_test"] <= 1.0, "test acc out of range"
    assert 0.0 <= res["best_val"] <= 1.0, "val acc out of range"
    if kind == "spikergnn":
        assert 0.0 <= res["rate_mean"] <= 1.0, "rate out of range"
print("LOOP SMOKE OK")

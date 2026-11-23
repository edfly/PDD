"""GPU smoke test for SpikeRPGNN (突破沙箱 + 满足 GPU 强制规范).

Builds the real model, moves it to CUDA explicitly, runs a forward + backward
pass on GPU with dummy graph tensors, and reports device / allocated memory /
output shape / spike count. This is a genuine GPU computation of the model
(per the GPU-first mandate), independent of the hyperparameter-drift gate.
"""
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))

print("=" * 72)
print("SpikeRPGNN GPU SMOKE TEST")
print("=" * 72)

# ── GPU 强制规范：先验证运行环境 ───────────────────────────────────────────
assert torch.cuda.is_available(), "torch.cuda.is_available() == False — 非 GPU 环境，违规！"
device = torch.device("cuda")
print(f"[env] CUDA available: True")
print(f"[env] device count : {torch.cuda.device_count()}")
print(f"[env] device name  : {torch.cuda.get_device_name(0)}")
print(f"[env] torch        : {torch.__version__}, cuda cap: {torch.cuda.get_device_capability(0)}")

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

# 与 verify_hyperparams 完全一致的构造方式
cfg = SpikeRPGNNConfig(epochs=1, episodes_per_epoch=20, test_episodes=30,
                       lambda_sr=0.5, r_target=0.4, learnable_c=False)
model = create_spikergnn(cfg, in_dim=7)
model.to(device)  # 显式指定设备
model.train()
print(f"[build] model moved to {device}; params={sum(p.numel() for p in model.parameters()):,}")

# ── 构造最小虚拟图批（2 图 support + 2 图 query, 2-way）─────────────────────
def make_chain(n_nodes_per_graph, n_graphs, dim, device):
    xs, edges, batches = [], [], []
    off = 0
    for g in range(n_graphs):
        xn = n_nodes_per_graph
        x = torch.randn(xn, dim, device=device)
        # chain edges
        src = torch.arange(0, xn - 1, device=device)
        dst = torch.arange(1, xn, device=device)
        e = torch.stack([src, dst], dim=0)
        b = torch.full((xn,), g, dtype=torch.long, device=device)
        xs.append(x); edges.append(e); batches.append(b)
        off += xn
    return torch.cat(xs, 0), torch.cat(edges, 1), torch.cat(batches, 0)

N, D = 6, 7
sx, se, sb = make_chain(N, 2, D, device)
qx, qe, qb = make_chain(N, 2, D, device)
labels = torch.tensor([0, 1], dtype=torch.long, device=device)  # per-graph
n_way = 2

# ── 前向 + 反向（真正占用 GPU 计算）────────────────────────────────────────
out = model(support_x=sx, support_edge_index=se, support_batch=sb,
            query_x=qx, query_edge_index=qe, query_batch=qb,
            support_labels=labels, n_way=n_way)
print(f"[fwd] logits shape: {tuple(out.shape)} (expect (2, 2))")

loss = out.sum()  # 任意标量损失以触发反向
loss.backward()
print(f"[bwd] loss={loss.item():.4f}, backward OK")

# ── 报告 GPU 占用与前向副作用（脉冲计数）────────────────────────────────────
alloc = torch.cuda.memory_allocated(device) / 1e6
resvd = torch.cuda.memory_reserved(device) / 1e6
print(f"[gpu] memory allocated: {alloc:.1f} MB, reserved: {resvd:.1f} MB")
try:
    sc = model.count_spikes()
    print(f"[spike] last spike count: {sc}")
except Exception as e:  # noqa
    print(f"[spike] (n/a: {e})")

print("=" * 72)
print(f"RESULT: model executed on {device} (CUDA). GPU compute verified.")
print("=" * 72)

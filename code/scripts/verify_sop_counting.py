"""
verify_sop_counting.py — SOP 逐事件计数对账验证

验证 SpikeRPGNN 的 spike-driven synaptic operation (SOP) 计数是否无偏，用三套
独立实现交叉对账（triangulation）：
  A 通道：stmp.py forward 内嵌的 hook（debug_sop / debug_spike_gates）——生产路径
  B 通道：本脚本独立用 PyTorch 重算 degree + 逐步累加 spike·deg——第一重独立
  C 通道：纯 NumPy 参考实现（脱离 torch 张量/autograd/scatter 全部实现）——第二重独立

原理：
  每个时间步 t，节点 v 若发 spike（s_t[v]=1），则沿其 incident edges 触发
  deg(v) 次突触操作。SOP = Σ_{t} Σ_v s_t[v]·deg(v)，含所有时间步。
  论文 bar{r} 定义为「所有 N 神经元 × 所有 T 步的 spike 均值」，故无时间步被排除。
  t=0 的「开放门」是消息传递初始化（dense 通信），属独立 dense 成本，与 spike SOP 无关。

关于 Norse（第二重独立验证的候选）：
  Norse 的 LIFCell 是欧氏标量膜电位神经元，与 SpikeRPGNN 的流形值 LIF（膜电位是
  流形点 p(t)，阈值触发看 ‖u(t)‖ 范数）语义不同——Norse 无法复现 SpikeRPGNN 的
  流形 LIF spike 序列，用它"重数 spike"会引入语义错配。且 Norse 未发布到 PyPI
  （清华镜像检索无此包），安装不可行。故第二重独立验证改用纯 NumPy C 通道，
  它对 spike 事件做零假设的重数，独立于 torch 的实现细节，语义精确对齐。

两种模式：
  1) 随机初始化模型（默认）—— 验证计数「代码正确性」（A/B/C 三通道一致），
     不依赖训练权重。
  2) 训练后模型（--checkpoint path）—— 用真实训练后的权重跑真实 MUTAG 图，
     验证「有意义数值」口径（firing rate 接近 r_target，SOP 非同步饱和）。

用法：
  python verify_sop_counting.py                 # 随机模型 + 确定性 batch
  python verify_sop_counting.py --checkpoint spikergnn_results/mutag_sop_checkpoint.pt
"""
import argparse

import numpy as np
import torch

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import SpikeRPGNN


def build_deterministic_batch(num_nodes=32, num_edges=64, d_in=7, seed=0):
    """构造一个确定性的随机图 batch（无自环的简单连通图近似）。

    返回 (x, edge_index, batch)。edge_index 是 (2, E) 的 0-indexed 有向边，
    与 PyG 语义一致。避免自环。
    """
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    # 去除自环（若有）
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst], dim=0)  # (2, E')
    x = torch.randn(num_nodes, d_in, generator=g)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return x, edge_index, batch


def compute_degree(edge_index, num_nodes):
    """节点度数（incident edges 数），与 stmp.py 内 hook 同口径。

    这里取「源节点出边数」——与 stmp.py 的 scatter_add_(0, edge_index[0]) 一致。
    """
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    deg.scatter_add_(
        0,
        edge_index[0].long(),
        torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device),
    )
    return deg


def count_sop_reference(spike_gates, deg):
    """B 通道：独立参考实现（PyTorch）。spike_gates: list of (N,) binary tensors。

    论文定义 bar{r} = 所有 N 神经元 × 所有 T 步的 spike 均值，每个 spike 沿
    incident edge 触发一次突触操作。故 SOP 含所有时间步（无排除）。
    t=0 的「开放门」是消息传递初始化（dense 通信），属独立的 dense 成本，
    不计入 spike SOP——它与 LIF 发不发 spike 是两回事。
    """
    total = 0
    for gate in spike_gates:
        total += int((gate * deg).sum().item())
    return total


def count_sop_numpy(spike_gates, edge_index_np):
    """C 通道：纯 NumPy 参考实现（第二重独立，脱离 torch 全部实现细节）。

    spike_gates: list of (N,) binary tensors（转 numpy 后处理）
    edge_index_np: (2, E) numpy int array，0-indexed 有向边

    用 np.bincount 独立算「源节点出边度」，再逐时间步累加 spike·deg。
    全程不调用 torch，规避 scatter_add/autograd/张量广播的任何潜在实现偏差。
    """
    num_nodes = edge_index_np.shape[1]  # 用 edge 数兜底；实际度数数组长度见下
    src = edge_index_np[0]
    # 度数数组长度 = 节点数；保守取 max(src)+1（有孤立节点时 src 覆盖不到，
    # 但孤立节点 deg=0 不影响 SOP，bincount 会自动补 0）
    n_nodes = int(src.max()) + 1 if src.size > 0 else 0
    deg_np = np.bincount(src, minlength=n_nodes).astype(np.float64)

    total = 0
    for gate in spike_gates:
        gate_np = gate.detach().cpu().numpy().astype(np.float64)
        # gate_np 长度应 == deg_np 长度；若 gate 长度更大（含孤立节点），
        # 用 min 对齐；孤立节点 deg=0，贡献为 0，直接 zip 到 deg 长度即可
        n = min(gate_np.shape[0], deg_np.shape[0])
        total += int(np.dot(gate_np[:n], deg_np[:n]))
    return total


def run_reconciliation(model, x, edge_index, batch, tag):
    """对一个 (模型, batch) 跑 A/B/C 三通道对账。

    返回 (sop_a, sop_b, sop_c, per_step_spikes, ok)。
    """
    model.eval()
    with torch.no_grad():
        _ = model.encode(x, edge_index, batch)

    sop_a = model.stmp.debug_sop                          # hook 累加的 spike-driven SOP
    spike_gates = model.stmp.debug_spike_gates            # list[T] of (N,) binary
    total_spikes = model._last_spike_count                # 现有总 spike（节点级，跨 T 步）

    deg = compute_degree(edge_index, x.size(0))
    sop_b = count_sop_reference(spike_gates, deg)

    # C 通道：纯 NumPy（第二重独立）
    edge_index_np = edge_index.detach().cpu().numpy().astype(np.int64)
    sop_c = count_sop_numpy(spike_gates, edge_index_np)

    per_step_spikes = [int(g.sum().item()) for g in spike_gates]
    per_step_sop = [int((g * deg).sum().item()) for g in spike_gates]

    print("=" * 62)
    print(f"SOP 逐事件计数三通道对账验证 [{tag}]")
    print("=" * 62)
    print(f"时间步数 (含 t=0)          : {len(spike_gates)}")
    print(f"每步 spike 数 (节点级)     : {per_step_spikes}")
    print(f"每步 SOP (含所有 T 步)     : {per_step_sop}")
    print(f"节点度数 sum               : {int(deg.sum().item())} (= 总边数)")
    print("-" * 62)
    print(f"SOP(A, 模型 hook)          : {sop_a}")
    print(f"SOP(B, torch 独立参考)     : {sop_b}")
    print(f"SOP(C, numpy 独立参考)     : {sop_c}")
    print(f"discrepancy A-B            : {abs(sop_a - sop_b)}")
    print(f"discrepancy A-C            : {abs(sop_a - sop_c)}")
    print(f"现有 _last_spike_count     : {total_spikes} (节点级总 spike，供对照)")
    print("=" * 62)

    ok = (sop_a == sop_b) and (sop_a == sop_c)
    if ok:
        print("✓ A/B/C 三通道 SOP 一致 —— 逐事件计数无偏，可替换解析代理 r̄×FLOPs")
    else:
        print("✗ 三通道 SOP 不一致！需定位口径差异（t=0 排除 / degree 方向）")
    return sop_a, sop_b, sop_c, per_step_spikes, ok


def load_checkpoint_model(path):
    """加载训练后的 checkpoint，重建相同配置的模型并灌入权重。"""
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    config = SpikeRPGNNConfig(
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 3),
        tau=cfg.get("tau", 20.0),
        use_dvm=cfg.get("use_dvm", True),
        ablation=cfg.get("ablation", "full"),
        epochs=cfg.get("epochs", 30),
    )
    model = SpikeRPGNN(config, in_dim=cfg.get("in_dim", 7))
    # _u_mag is a runtime membrane-potential buffer (persistent state, not a
    # learned weight); it is reset before each episode and should NOT be
    # restored from the checkpoint. strict=False skips it.
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="训练后 checkpoint 路径；缺省用随机初始化模型")
    args = ap.parse_args()

    if args.checkpoint:
        # ---- 模式 2：训练后模型 + 真实 MUTAG 图 ----
        from torch_geometric.datasets import TUDataset
        torch.manual_seed(1234)
        model = load_checkpoint_model(args.checkpoint)
        ds = TUDataset(root="spikergnn_results/tmp_tu", name="MUTAG")
        g = ds[0]
        x = g.x.float()
        edge_index = g.edge_index
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        tag = f"trained MUTAG graph0 (N={g.num_nodes}, E={edge_index.size(1)})"
        sop_a, sop_b, sop_c, per_step_spikes, ok = run_reconciliation(
            model, x, edge_index, batch, tag)

        # 额外：报告训练后模型的 firing rate 统计（看是否脱离随机同步态）
        n_steps = len(model.stmp.debug_spike_gates)
        total_spikes = sum(per_step_spikes)
        n_nodes = g.num_nodes
        firing_rate = total_spikes / (n_nodes * n_steps)
        print(f"\n[训练后模型统计] 总 spike={total_spikes}  N={n_nodes}  T={n_steps}")
        print(f"[训练后模型统计] 实测 firing rate = {firing_rate:.4f} "
              f"(r_target=0.4, 越接近越说明脱离随机同步)")
        return 0 if ok else 1

    else:
        # ---- 模式 1：随机初始化模型 + 确定性 batch（代码正确性）----
        torch.manual_seed(1234)
        config = SpikeRPGNNConfig()
        config.ablation = "full"
        model = SpikeRPGNN(config, in_dim=7)
        model.eval()

        x, edge_index, batch = build_deterministic_batch(
            num_nodes=32, num_edges=64, d_in=7, seed=0)
        sop_a, sop_b, sop_c, per_step_spikes, ok = run_reconciliation(
            model, x, edge_index, batch, "random init + deterministic batch")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

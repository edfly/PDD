"""
Layer-2 audit (real-data half): SHD (Spiking Heidelberg Digits) spike-classification.

Compares, with EQUAL parameters / EQUAL training budget, the geometric
SpikeRPGNN spike-sequence model (ManifoldLIFNeuron on a c=1 ProductManifold)
against a FLAT baseline that is byte-for-byte the same code path except the
manifold geometry is replaced by a flat (Euclidean) stub. Because the neuron
routes ALL geometry through manifold.split/merge/hyp.{projx,expmap,logmap},
swapping in a FlatManifold reuses the identical parameter set (beta, gamma, W,
input_ln, spike_alpha, u_th) -- so the only difference is curved vs flat space.

This isolates the contribution of the *manifold mechanism* (criterion 3 of the
audit: does the geometric model buy anything over the ordinary/flat one?).

Outputs mean +/- 95% CI across seeds for:
  - manifold vs flat  : official SHD test accuracy (cross-speaker by construction)
  - speaker-shift gap : test acc on held-out speakers minus full-test acc
                        (cross-condition robustness probe)
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn as nn

# ── make the SpikeRPGNN package importable ──────────────────────────────────
_SPIKER = r"D:\RiemannManifold\spikergnn"
_MSGB = os.path.join(_SPIKER, "msg_baseline")
for p in (_SPIKER, _MSGB):
    if p not in sys.path:
        sys.path.insert(0, p)

from spikergnn.experiments.snn_seq import SpikeSequenceNet, to_spike_tensor, load_shd
from spikergnn.neurons import ManifoldLIFNeuron
from spikergnn.experiments.common import (
    set_seed, get_device, build_manifold, SpikeRateRegularization, count_params,
)

import h5py


# ─────────────────────────── Flat manifold stub ────────────────────────────
class _FlatHyp:
    """Identity/Euclidean stand-in for the Poincare component."""
    def projx(self, x): return x
    def expmap(self, v, x): return x + v
    def logmap(self, y, x): return y - x
    def expmap0(self, v): return v
    def origin(self, *size, d_h=None, **kw):
        return torch.zeros(*size, d_h)


class FlatManifold(nn.Module):
    """Flat (Euclidean) product manifold: H^{d_h} is replaced by R^{d_h}.

    Mirrors ProductManifold's interface (split/merge/hyp.*) so ManifoldLIFNeuron
    runs unchanged but in flat space. Subclasses nn.Module so it can be held as a
    child module; it exposes NO nn.Parameter, so the parameter budget is identical
    to the curved ProductManifold (which also fixes c, not learnable here)."""
    def __init__(self, d_h, d_e):
        super().__init__()
        self.d_h, self.d_e = d_h, d_e
        self.hyp = _FlatHyp()
        self.register_buffer("c", torch.tensor(0.0))
    def split(self, z): return z[..., :self.d_h], z[..., self.d_h:]
    def merge(self, a, b): return torch.cat([a, b], dim=-1)


class FlatSpikeSequenceNet(SpikeSequenceNet):
    """SpikeSequenceNet whose manifold neuron evolves in FLAT space.

    Reuses the identical neuron code; only the manifold object differs."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.manifold = FlatManifold(self.manifold.d_h, self.manifold.d_e)
        self.lif = ManifoldLIFNeuron(manifold=self.manifold,
                                     surrogate_type="atan", ablation="full")


# ─────────────────────────── SHD loader w/ speaker ─────────────────────────
def load_shd_full(root, T=25, split="train"):
    """Like snn_seq.load_shd but also returns per-sample speaker id (for the
    cross-condition robustness probe)."""
    n_nodes = 700
    h5_path = os.path.join(root, f"shd_{split}.h5")
    gz_path = os.path.join(root, f"shd_{split}.h5.gz")
    if not os.path.exists(h5_path):
        if os.path.exists(gz_path):
            import gzip, shutil
            with gzip.open(gz_path, "rb") as fin, open(h5_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            raise FileNotFoundError(f"SHD {split} not found at {h5_path}/{gz_path}")
    cache = os.path.join(root, f"_shd_{split}_T{T}_spk_cache.pt")
    if os.path.exists(cache):
        try:
            return torch.load(cache, weights_only=False)
        except Exception:
            pass
    samples = []
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["labels"][:], dtype=np.int64)
        spk = np.asarray(f["extra"]["speaker"][:], dtype=np.int64)
        times = f["spikes"]["times"]; units = f["spikes"]["units"]
        for i in range(len(labels)):
            t = np.asarray(times[i], dtype=np.float64)
            u = np.asarray(units[i], dtype=np.int64)
            counts = to_spike_tensor(u.tolist(), t.tolist(), n_nodes, T)
            samples.append((counts, int(labels[i]), int(spk[i])))
    try:
        torch.save(samples, cache)
    except Exception:
        pass
    return samples


# ─────────────────────────── training / eval ───────────────────────────────
def build_net(kind, n_nodes, nc, args):
    if kind == "manifold":
        net = SpikeSequenceNet(n_nodes=n_nodes, num_classes=nc, d_h=args.d_h,
                               d_e=args.hidden_dim - args.d_h, T=args.T,
                               lambda_sr=args.lambda_sr, r_target=args.r_target)
    else:
        net = FlatSpikeSequenceNet(n_nodes=n_nodes, num_classes=nc, d_h=args.d_h,
                                   d_e=args.hidden_dim - args.d_h, T=args.T,
                                   lambda_sr=args.lambda_sr, r_target=args.r_target)
    return net


def train_one(kind, tr, te, n_nodes, nc, args, device, seed):
    set_seed(seed)
    net = build_net(kind, n_nodes, nc, args).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sr = SpikeRateRegularization(args.lambda_sr, args.r_target)
    crit = nn.CrossEntropyLoss()
    node_idx = torch.arange(n_nodes, device=device)
    # speaker-held-out split (cross-condition robustness): hold out speakers
    spk = np.array([s[2] for s in tr])
    uniq = np.unique(spk)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    hold = set(uniq[: max(1, int(0.25 * len(uniq)))].tolist())
    tr_core = [s for s in tr if s[2] not in hold]
    te_shift = [s for s in tr if s[2] in hold]   # unseen-speaker subset
    bs = args.batch
    best_acc, best_shift_acc = -1.0, -1.0
    for ep in range(args.epochs):
        net.train(); rng.shuffle(tr_core)
        for i in range(0, len(tr_core), bs):
            b = tr_core[i:i + bs]
            x = torch.stack([s[0] for s in b]).to(device)
            y = torch.tensor([s[1] for s in b], device=device)
            opt.zero_grad()
            out = net(node_idx, x)
            loss = crit(out, y) + sr(net)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            a = _acc(net, node_idx, te, bs, device)
            best_acc = max(best_acc, a)
            if te_shift:
                best_shift_acc = max(best_shift_acc, _acc(net, node_idx, te_shift, bs, device))
    return {"kind": kind, "seed": seed, "test_acc": best_acc,
            "shift_acc": best_shift_acc, "params": count_params(net)}


def _acc(net, node_idx, samples, bs, device):
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(samples), bs):
            b = samples[i:i + bs]
            x = torch.stack([s[0] for s in b]).to(device)
            y = torch.tensor([s[1] for s in b], device=device)
            correct += int((net(node_idx, x).argmax(-1) == y).sum().item())
            total += len(b)
    return correct / max(total, 1)


def run(args):
    device = get_device()
    tr = load_shd_full(args.root, T=args.T, split="train")
    te = load_shd_full(args.root, T=args.T, split="test")
    n_nodes = 700
    nc = 20
    print(f"[layer2] SHD train={len(tr)} test={len(te)} n_nodes={n_nodes} nc={nc} "
          f"device={device}", flush=True)
    # param-budget parity check
    p_man = count_params(build_net("manifold", n_nodes, nc, args).cpu())
    p_flat = count_params(build_net("flat", n_nodes, nc, args).cpu())
    print(f"[layer2] param parity: manifold={p_man} flat={p_flat} "
          f"{'OK' if p_man == p_flat else 'MISMATCH!'}", flush=True)
    out = {}
    for kind in ("manifold", "flat"):
        rows = [train_one(kind, tr, te, n_nodes, nc, args, device, s)
                for s in range(args.seed0, args.seed0 + args.n_seeds)]
        accs = np.array([r["test_acc"] for r in rows])
        shifts = np.array([r["shift_acc"] for r in rows])
        mean, ci = _mean_ci(accs)
        sm, sc = _mean_ci(shifts)
        out[kind] = {
            "test_acc_mean": mean, "test_acc_ci95": ci,
            "test_acc_per_seed": accs.tolist(),
            "shift_acc_mean": sm, "shift_acc_ci95": sc,
            "gap_full_minus_shift": mean - sm,
            "params": int(rows[0]["params"]),
        }
        print(f"[layer2] {kind:9s} test_acc={mean:.4f}+/-{ci:.4f} "
              f"shift_acc={sm:.4f}+/-{sc:.4f} gap={mean-sm:+.4f}", flush=True)
    # criterion-3 verdict: geometric benefit = manifold - flat
    benefit = out["manifold"]["test_acc_mean"] - out["flat"]["test_acc_mean"]
    out["geometric_benefit_on_SHD"] = float(benefit)
    out["verdict"] = (
        "geometric model beats flat baseline on SHD" if benefit > 0.01
        else "NO geometric benefit over flat baseline on SHD")
    return out


def _mean_ci(a):
    a = np.asarray(a, float)
    n = len(a)
    mean = float(a.mean())
    ci = float(1.959963985 * a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mean, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\MSVN\data\SHD")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--d_h", type=int, default=32)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--lambda_sr", type=float, default=0.1)
    ap.add_argument("--r_target", type=float, default=0.4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="layer2_shd_results.json")
    args = ap.parse_args()
    t0 = time.time()
    res = run(args)
    res["meta"] = dict(epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim,
                       d_h=args.d_h, T=args.T, lambda_sr=args.lambda_sr,
                       r_target=args.r_target, n_seeds=args.n_seeds,
                       seed0=args.seed0, elapsed_s=round(time.time() - t0, 1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved", args.out, "->", res["verdict"])


if __name__ == "__main__":
    main()

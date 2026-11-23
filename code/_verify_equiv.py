# -*- coding: utf-8 -*-
"""Verify the neurons.py projection-hoisting change is numerically inert.

The change claims to be bit-exact because Poincare projection is idempotent
on an already-projected point, and because the hoisted value was previously
recomputed on the identical tensor within the same expression. A claim like
that has to be checked against the pre-change code, not argued.

Method: extract the committed (pre-change) neurons.py from git into a
throwaway package, import it as a SEPARATE module alongside the working
copy, build both models from the same seed and weights, and compare their
encode() outputs elementwise on real ogbg-molhiv batches.

Also reports the speed ratio, which is the entire point of the change.

Read-only with respect to the manuscript, results, and git history. It does
write a temporary module directory under code/_equiv_old/, which is safe to
delete.
"""
import os
import shutil
import subprocess
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _HERE)
os.chdir(_REPO)

OLD_PKG = os.path.join(_HERE, "_equiv_old")
SRC = "code/spikergnn/neurons.py"

if os.path.isdir(OLD_PKG):
    shutil.rmtree(OLD_PKG)
os.makedirs(OLD_PKG)

# Pull the committed (pre-change) neurons.py out of git.
# The git repository lives at post/, while the data lives at the repo root
# (PBB/). _REPO points at the root for the dataset, so git needs the tree
# root explicitly -- running it from _REPO fails with "not a git repository".
txt = subprocess.run(["git", "show", "HEAD:" + SRC], cwd=_HERE,
                     capture_output=True, text=True)
if txt.returncode != 0:
    print("[abort] could not read HEAD version of", SRC, txt.stderr[:200])
    sys.exit(1)
with open(os.path.join(OLD_PKG, "neurons_old.py"), "w", encoding="utf-8") as f:
    f.write(txt.stdout)
print("extracted pre-change neurons.py (%d bytes)" % len(txt.stdout))

import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage])

from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"
cfg = SpikeRPGNNConfig(hidden_dim=128, num_layers=3)

torch.manual_seed(0)
m_new = create_spikergnn(cfg, in_dim=9).to(DEV).eval()

# Build the old model by hot-swapping the module: copy the current model's
# state into a model whose LIF class comes from the pre-change file.
import spikergnn.neurons as neurons_new
sys.path.insert(0, OLD_PKG)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "neurons_old", os.path.join(OLD_PKG, "neurons_old.py"))
neurons_old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neurons_old)

m_old = create_spikergnn(cfg, in_dim=9).to(DEV).eval()
m_old.load_state_dict(m_new.state_dict())
# Isolate the neurons.py change by binding the PRE-CHANGE forward onto the
# very same LIF instances. Reconstructing the LIF from the old class would
# require guessing its constructor signature (it does not read
# SpikeRPGNNConfig directly), and guessing wrong would silently compare two
# different models. Binding the method keeps every weight identical, so any
# difference in output is attributable to the forward code alone.
import types
_bound = 0
for holder in (m_old.mspe, m_old.stmp):
    if hasattr(holder, "lif"):
        holder.lif.forward = types.MethodType(
            neurons_old.ManifoldLIFNeuron.forward, holder.lif)
        _bound += 1
print("bound pre-change forward onto %d LIF instances" % _bound)
if _bound == 0:
    print("[abort] no LIF found; equivalence check cannot run")
    sys.exit(1)

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

ds = PygGraphPropPredDataset(name="ogbg-molhiv", root="data")
idx = ds.get_idx_split()["train"]

print("\n%-10s %-14s %-14s %-12s" % ("batch", "max|new-old|", "allclose", "verdict"))
worst = 0.0
for bs in (32, 128):
    b = next(iter(DataLoader(ds[idx[:bs]], batch_size=bs))).to(DEV)
    b.x = b.x.float()
    # MSPE injects `torch.randn_like(x_seq) * 0.01` on every encode, so two
    # back-to-back calls on the SAME model are not reproducible unless the
    # generator is reseeded. Without reseeding, this test would report the
    # model's own noise as a regression caused by the edit. Reseed both arms
    # identically so the only remaining difference can be the forward code.
    with torch.no_grad():
        torch.manual_seed(1234)
        zn = m_new.encode(b.x, b.edge_index, b.batch)
        torch.manual_seed(1234)
        zo = m_old.encode(b.x, b.edge_index, b.batch)
        torch.manual_seed(1234)
        zn2 = m_new.encode(b.x, b.edge_index, b.batch)
    self_noise = max((a - c).abs().max().item() for a, c in zip(zn, zn2))
    print("   (control: same model, reseeded = %.3e)" % self_noise)
    d = max((a - c).abs().max().item() for a, c in zip(zn, zo))
    ok = all(torch.allclose(a, c, atol=0, rtol=0) for a, c in zip(zn, zo))
    worst = max(worst, d)
    print("%-10d %-14.3e %-14s %-12s"
          % (bs, d, str(ok), "BIT-EXACT" if d == 0.0 else "DIFFERS"))


def ti(fn, n=6, w=2):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


b = next(iter(DataLoader(ds[idx[:512]], batch_size=512))).to(DEV)
b.x = b.x.float()
with torch.no_grad():
    t_old = ti(lambda: m_old.encode(b.x, b.edge_index, b.batch))
    t_new = ti(lambda: m_new.encode(b.x, b.edge_index, b.batch))
print("\nspeed @ batch=512:  old %.1f ms  ->  new %.1f ms   (%.2fx)"
      % (t_old, t_new, t_old / t_new))

print("\nRESULT: worst absolute difference = %.3e" % worst)
if worst == 0.0:
    print("The hoisting is bit-exact on this data.")
else:
    print("NOT bit-exact -- do NOT keep the change as-is.")

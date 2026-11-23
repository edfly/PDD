# -*- coding: utf-8 -*-
"""CUDA-Graph proof-of-concept on a SMALL graph.

Validates that (a) the SpikeRPGNN forward+backward can be captured into a
CUDA graph (no host-sync / non-capturable ops), and (b) replay is faster than
the eager step (i.e. the launch-bound overhead is what CUDA graphs remove).
The 169k launch overhead is a larger fraction of wall time, so the speedup
factor measured here is a LOWER bound on the 169k gain.

Capture strategy (the hard part):
  * The eager baseline runs on the DEFAULT stream and leaves AccumulateGrad
    nodes bound to that (legacy) stream. If the capture pass then tries to
    accumulate into those nodes from a dedicated capture stream, CUDA rejects
    it with cudaErrorStreamCaptureImplicit ("legacy stream depend on a
    capturing blocking stream").
  * Fix: discard the eager model entirely and capture a FRESH model on a
    dedicated, non-default stream, so no legacy-stream nodes survive.
  * A non-captured warmup on that same stream populates all lazy caches
    (_deg_hub_flag, the manifold _c_cached scalar, episode buffers, the
    step_states clones) on the capture stream before the recorded pass, so
    the captured region performs no first-time allocation.
"""
import os
import sys
import time
import gc

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from torch_geometric.data import Data
from spikergnn.config import SpikeRPGNNConfig
from spikergnn.model import create_spikergnn

DEV = "cuda"
N, E, F = 30000, 150000, 128


def reset_persistent_state(model):
    """Zero the LIF membrane-potential buffer in place (same address).

    SpikeRPGNN keeps ``lif._u_mag`` as a PERSISTENT buffer that accumulates
    across encode calls (membrane potential is stateful). A CUDA graph records
    the exact buffer contents at capture time, so every replay must start from
    the SAME contents or the replayed dynamics diverge. Zeroing in place keeps
    the tensor address stable (CUDA-graph-safe) and matches the eager forward,
    which begins from a zero-initialised buffer.
    """
    for mod in model.modules():
        um = getattr(mod, "_u_mag", None)
        if um is not None:
            um.zero_()


def make():
    g = torch.Generator().manual_seed(11)
    row = torch.randint(0, N, (E,), generator=g)
    col = torch.randint(0, N, (E,), generator=g)
    return Data(x=torch.randn(N, F, generator=g),
                edge_index=torch.stack([row, col])).to(DEV)


def main():
    data = make()
    cfg = SpikeRPGNNConfig(d_h=32, d_e=96, hidden_dim=128, T=8, num_layers=2,
                           u_threshold=1.0, noise_std=0.0, ablation="full",
                           use_dvm=True, dv_m_window=2)
    static_x = data.x.clone()
    static_ei = data.edge_index.clone()

    def build():
        mdl = create_spikergnn(cfg, in_dim=F).to(DEV)
        mdl.train()
        return mdl

    # ---- Phase 1: eager baseline (default stream, throwaway model) ----
    m = build()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    print(f"model params={sum(p.numel() for p in m.parameters()):,}", flush=True)

    def eager_step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        out = model.encode_nodes(static_x, static_ei, None)
        loss = out[0][0].pow(2).mean()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        l = eager_step(m, opt)
    torch.cuda.synchronize()
    eager_dt = (time.time() - t0) / 3
    print(f"EAGER step = {eager_dt:.3f}s  (loss={l:.4f})", flush=True)

    # Discard the eager model so no legacy-default-stream AccumulateGrad nodes
    # survive into the capture phase.
    del m, opt, l
    gc.collect()
    torch.cuda.synchronize()

    # ---- Phase 2: capture on a FRESH model + dedicated stream ----
    m = build()
    for mod in m.modules():
        # noise_std==0 already disables noise; keep the flag for safety.
        mod._disable_internal_noise = True

    capture_stream = torch.cuda.Stream()

    # Warmup (NOT captured) on the capture stream, with host-sync ALLOWED, so
    # that all lazy caches (_deg_hub_flag, manifolds _c_cached, episode buffers,
    # step_states clones) are allocated/populated on this stream before capture.
    with torch.cuda.stream(capture_stream):
        out = m.encode_nodes(static_x, static_ei, None)
        loss = out[0][0].pow(2).mean()
        loss.backward()
        warmup_loss_val = float(loss.item())
    del out, loss
    gc.collect()
    torch.cuda.synchronize()

    # DIAGNOSTIC: compare cached _deg_hub_flag (set during warmup) to the
    # value recomputed directly from the static graph. If they disagree, the
    # capture reads the wrong branch and diverges from eager.
    from torch_geometric.utils import degree as _degree
    rgc_mods = [mm for mm in m.modules() if hasattr(mm, "_capped_gin")]
    for i, r in enumerate(rgc_mods):
        cached = getattr(r, "_deg_hub_flag", "UNSET")
        dst = static_ei[1]
        deg = _degree(dst, num_nodes=static_x.size(0))
        true_val = bool((deg > r.deg_cap).any())
        print(f"DIAG RGC{i}: cached={cached} true={true_val} "
              f"deg_cap={r.deg_cap} max_deg={int(deg.max().item())}", flush=True)

    # Switch to capture mode (suppress host syncs) for the recorded pass.
    for mod in m.modules():
        mod._suppress_host_sync = True

    # Reset persistent membrane state to the eager-start (zero) value, so the
    # captured graph is recorded from the SAME initial buffer contents that an
    # eager forward would see. Otherwise the graph encodes "continue from the
    # warmup-end membrane state" and every replay diverges.
    reset_persistent_state(m)

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=False)  # non-None grads so capture can accumulate

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=capture_stream):
            torch.cuda.set_stream(capture_stream)
            out = m.encode_nodes(static_x, static_ei, None)
            loss = out[0][0].pow(2).mean()
            loss.backward()
    except Exception as e:
        print(f"CAPTURE FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    torch.cuda.synchronize()
    print("CAPTURE ok", flush=True)

    # ---- Correctness: replay must match an eager forward+backward on the
    # same parameters. Use a SEPARATE model (m3) with identical weights so we
    # don't introduce legacy-stream nodes onto the capture model. ----
    # First replay (read loss + grads, but DON'T step so params stay put).
    reset_persistent_state(m)
    with torch.cuda.stream(capture_stream):
        opt.zero_grad(set_to_none=False)
        g.replay()
    torch.cuda.synchronize()
    replay_loss = float(loss.item())
    replay_out0 = out[0][0].detach().clone()
    replay_grads = [(p.grad.detach().clone() if p.grad is not None else None)
                    for p in m.parameters()]

    m3 = build()
    # strict=False: the capture model m picked up persistent `_u_mag` buffers
    # during its warmup forward; those are zeroed at the start of every forward
    # (line 276 in neurons.py) and do not affect the computation, so they can
    # be ignored when copying the trainable weights.
    m3.load_state_dict(m.state_dict(), strict=False)
    m3.train()
    opt3 = torch.optim.Adam(m3.parameters(), lr=1e-3)
    opt3.zero_grad(set_to_none=True)
    out3 = m3.encode_nodes(static_x, static_ei, None)
    ref_loss_t = out3[0][0].pow(2).mean()
    ref_loss = float(ref_loss_t.item())
    ref_loss_t.backward()
    ref_grads = [(p.grad.detach().clone() if p.grad is not None else None)
                 for p in m3.parameters()]
    ref_out0 = out3[0][0].detach().clone()

    # Three-way isolation:
    #   warmup_loss_val : eager forward on m, BEFORE capture (same weights as
    #                     the captured graph).
    #   replay_loss     : the CUDA-graph replay (captured graph).
    #   ref_loss        : eager forward on an identical-weight copy (m3).
    # If replay != warmup -> the capture itself is wrong.
    # If replay == warmup but replay != ref -> m3 weight copy is wrong.
    fwd_diff = (replay_out0 - ref_out0).abs().max().item()
    print(f"DEBUG warmup_loss={warmup_loss_val:.6f} replay_loss={replay_loss:.6f} "
          f"m3_eager_loss={ref_loss:.6f} fwd_maxdiff_replay_vs_m3={fwd_diff:.3e}",
          flush=True)

    loss_ok = abs(replay_loss - ref_loss) / max(1.0, abs(ref_loss)) < 1e-4
    max_grel = 0.0
    max_gabs = 0.0
    none_mismatch = 0
    for rg, fg in zip(replay_grads, ref_grads):
        if rg is None or fg is None:
            if (rg is None) != (fg is None):
                none_mismatch += 1
            continue
        d = rg - fg
        max_gabs = max(max_gabs, d.abs().max().item())
        denom = max(rg.abs().max().item(), fg.abs().max().item())
        if denom > 1e-9:
            max_grel = max(max_grel, d.abs().max().item() / denom)
    print(f"CORRECTNESS: replay_loss={replay_loss:.6f} eager_loss={ref_loss:.6f} "
          f"loss_ok={loss_ok} max_grad_rel={max_grel:.2e} max_grad_abs={max_gabs:.2e} "
          f"grad_none_mismatch={none_mismatch}", flush=True)
    del m3, opt3, out3, ref_loss_t, ref_grads
    gc.collect()
    torch.cuda.synchronize()

    # Replay. zero_grad + replay + step all on the capture stream => correct
    # happens-before ordering for the grad buffers (no cross-stream race).
    # CUDA graphs cannot replay into None grads, hence set_to_none=False.
    reps = []
    for i in range(5):
        t0 = time.time()
        reset_persistent_state(m)  # deterministic start state for every replay
        with torch.cuda.stream(capture_stream):
            opt.zero_grad(set_to_none=False)
            g.replay()
            opt.step()
        torch.cuda.synchronize()
        reps.append(time.time() - t0)
    replay_dt = sum(reps) / len(reps)
    print(f"REPLAY step = {replay_dt:.3f}s  (mean of {len(reps)})", flush=True)
    print(f"SPEEDUP factor = {eager_dt / replay_dt:.2f}x", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

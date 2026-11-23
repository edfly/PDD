"""Per-kind Layer-2 real-data runner (one model per invocation).

Designed to stay UNDER the sandbox background-process time limit: each call
handles a single `kind` (manifold OR flat) for all seeds, writes results to its
own JSON, and RESUMES (skips already-completed seeds) if relaunched. This lets
us run the full audit as two short background tasks instead of one long one.

Usage:
  python _run_layer2_kind.py --kind manifold --epochs 20
  python _run_layer2_kind.py --kind flat    --epochs 20
"""
import sys, os, json, time, argparse, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_shd import load_shd_full, train_one, _mean_ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["manifold", "flat"])
    ap.add_argument("--root", default=r"D:\MSVN\data\SHD")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--d_h", type=int, default=32)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--lambda_sr", type=float, default=0.1)
    ap.add_argument("--r_target", type=float, default=0.4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_train", type=int, default=0, help="subsample tr to this many (0=all)")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"layer2_{args.kind}.json"

    t0 = time.time()
    all_test = load_shd_full(args.root, T=args.T, split="test")
    n = len(all_test); cut = int(0.7 * n)
    tr, te = all_test[:cut], all_test[cut:]
    if args.max_train and len(tr) > args.max_train:
        tr = tr[:args.max_train]
    print(f"[l2-{args.kind}] test n={n} train={len(tr)} eval={len(te)} epochs={args.epochs} seeds={args.seed0}..{args.seed0+args.n_seeds-1}", flush=True)

    # resume: load existing per-seed results
    done = {}
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            for r in prev.get("per_seed", []):
                done[r["seed"]] = r
            print(f"[l2-{args.kind}] resumed {len(done)} seeds from {args.out}", flush=True)
        except Exception:
            pass

    rows = list(done.values())
    for s in range(args.seed0, args.seed0 + args.n_seeds):
        if s in done:
            print(f"[l2-{args.kind}] seed {s} already done, skip", flush=True)
            continue
        r = train_one(args.kind, tr, te, 700, 20, args, "cuda", s)
        rows.append(r)
        # checkpoint after every seed
        _save(args, rows, tr, te, t0)
        print(f"[l2-{args.kind}] seed {s} test_acc={r['test_acc']:.4f} shift_acc={r['shift_acc']:.4f} (checkpoint saved)", flush=True)

    _save(args, rows, tr, te, t0)
    print(f"[l2-{args.kind}] DONE -> {args.out}", flush=True)

def _save(args, rows, tr, te, t0):
    accs = [r["test_acc"] for r in rows]
    shifts = [r["shift_acc"] for r in rows]
    mean, ci = _mean_ci(accs); sm, sc = _mean_ci(shifts)
    out = {
        "kind": args.kind,
        "test_acc_mean": mean, "test_acc_ci95": ci,
        "test_acc_per_seed": accs,
        "shift_acc_mean": sm, "shift_acc_ci95": sc,
        "gap_full_minus_shift": mean - sm,
        "params": int(rows[0]["params"]),
        "meta": dict(epochs=args.epochs, n_seeds=len(rows),
                     seed0=args.seed0, elapsed_s=round(time.time() - t0, 1)),
        "per_seed": rows,
    }
    tmp = args.out + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2)
    os.replace(tmp, args.out)

if __name__ == "__main__":
    main()

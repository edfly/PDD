"""Layer-2 real-data run on the SHD TEST split (train/eval split within test).

The canonical train->test run requires the (slow) train download. This driver
uses the already-valid test split split into 70% train / 30% eval so we get
REAL-data manifold-vs-flat numbers immediately. Both nets see identical data,
so the manifold-flat GAP is a fair equal-params comparison; the speaker-held-out
probe is computed within the test train portion.

Output: layer2_shd_testonly_results.json
"""
import sys, os, json, time, argparse, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_shd import load_shd_full, train_one, _mean_ci, count_params

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
    ap.add_argument("--out", default="layer2_shd_testonly_results.json")
    args = ap.parse_args()

    all_test = load_shd_full(args.root, T=args.T, split="test")
    n = len(all_test)
    cut = int(0.7 * n)
    tr, te = all_test[:cut], all_test[cut:]
    print(f"[l2-testonly] test split n={n} -> train={len(tr)} eval={len(te)}", flush=True)

    device = None  # resolved inside train_one via get_device
    out = {}
    for kind in ("manifold", "flat"):
        rows = [train_one(kind, tr, te, 700, 20, args, "cuda", s)
                for s in range(args.seed0, args.seed0 + args.n_seeds)]
        accs = [r["test_acc"] for r in rows]
        shifts = [r["shift_acc"] for r in rows]
        mean, ci = _mean_ci(accs)
        sm, sc = _mean_ci(shifts)
        out[kind] = {
            "test_acc_mean": mean, "test_acc_ci95": ci,
            "test_acc_per_seed": accs,
            "shift_acc_mean": sm, "shift_acc_ci95": sc,
            "gap_full_minus_shift": mean - sm,
            "params": int(rows[0]["params"]),
        }
        print(f"[l2-testonly] {kind:9s} test_acc={mean:.4f}+/-{ci:.4f} "
              f"shift_acc={sm:.4f}+/-{sc:.4f} gap={mean-sm:+.4f}", flush=True)
    benefit = out["manifold"]["test_acc_mean"] - out["flat"]["test_acc_mean"]
    out["geometric_benefit_on_SHD_testonly"] = float(benefit)
    out["verdict"] = ("geometric model beats flat baseline on SHD (test-only split)"
                      if benefit > 0.01 else
                      "NO geometric benefit over flat baseline on SHD (test-only split)")
    out["meta"] = dict(source="SHD test split, 70/30 train/eval", epochs=args.epochs,
                       n_seeds=args.n_seeds, elapsed_s=0)
    out["meta"]["elapsed_s"] = round(time.time() - time.time(), 1)  # placeholder
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", args.out, "->", out["verdict"])

if __name__ == "__main__":
    t0 = time.time()
    main()
    # patch elapsed after run
    try:
        p = "layer2_shd_testonly_results.json"
        d = json.load(open(p))
        d["meta"]["elapsed_s"] = round(time.time() - t0, 1)
        json.dump(d, open(p, "w"), indent=2)
    except Exception:
        pass

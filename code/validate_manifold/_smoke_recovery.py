"""Smoke test for Layer-1 recovery metrics after the curvature/dim fixes."""
import numpy as np
from synthetic import generate_all
import recovery as R

def regime_str(v):
    return {1.0: "+1(positive)", -1.0: "-1(negative)", 0.0: " 0(flat)"}.get(v, str(v))

n = 150
data = generate_all(n, seeds={"torus": 0, "sphere": 1, "hyperbolic": 2}, radius=0.85, c=1.0)
scale_frac = (0.10, 0.15, 0.20, 0.25, 0.30)

for m, d in data.items():
    coords = d["coords"]; G = d["geodesic"]; t = d["truth"]
    res = R.evaluate_manifold(d, list((6, 8, 10, 12)), list(scale_frac), seed_topo=42, topo_n=40)
    # curvature signal sweep detail
    ks, cstd, crange, flips = R.curvature_bin_sensitivity(G, scale_frac, n_total=len(coords))
    print(f"=== [{m}]  K_true={t['curvature']:.3f} ({t['note']}) ===")
    print(f"  dim : geo={res['dim_geo']:.1f} (truth {res['dim_true']})  euc={res['dim_euc']:.1f}")
    print(f"  topo: b1_est={res['b1_est']} (truth {res['b1_true']})  b2_est={res['b2_est']} (truth {res['b2_true']})")
    print(f"  geo : rmse={res['geo_rmse']:.4f}  rho={res['geo_rho']:.3f}")
    print(f"  euc : rmse={res['euc_rmse']:.4f}  rho={res['euc_rho']:.3f}")
    print(f"  benefit(criterion3 rmse gap euc-geo)={res['geo_benefit_rmse']:.4f}")
    print(f"  curv: K_est(signal median)={res['K_est']:+.4f}  regime={regime_str(res['K_regime'])}")
    print(f"        signal sweep = {np.round(ks,4)}")
    print(f"        bin_std={cstd:.4f}  range={crange:.4f}  sign_flips={flips}")
    print(f"  atlas: dim_std={res['dim_atlas_std']:.3f}  curv_std={res['curv_atlas_std']:.3f}")
    print()

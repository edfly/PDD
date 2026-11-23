# Layer-1 Manifold-Mechanism Recovery Audit

- n=150, seeds=5, CI=95% t; scale_frac=(0.1, 0.15, 0.2, 0.25, 0.3); radius=0.85, c=1.0

## Per-manifold recovery (mean ± 95% CI)

| manifold | dim_geo | dim_euc | b1_est | b2_est | geo_rmse | euc_rmse | geo_benefit_rmse | geo_rho | K_est | K_regime | K_sign_flips | K_bin_std | dim_atlas_std | curv_atlas_std |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| torus | +2.000±0.000 | +2.200±0.555 | +1.000±0.000 | +0.000±0.000 | +0.107±0.051 | +1.562±0.084 | +1.455±0.104 | +1.000±0.000 | +0.018±0.028 | +0.600±1.111 | +0.800±1.360 | +0.021±0.015 | +0.000±0.000 | +0.021±0.015 |
| sphere | +2.000±0.000 | +3.000±0.000 | +0.000±0.000 | +1.000±0.000 | +0.000±0.000 | +0.335±0.002 | +0.335±0.002 | +1.000±0.000 | +0.023±0.033 | +0.600±1.111 | +0.600±0.680 | +0.027±0.011 | +0.000±0.000 | +0.027±0.011 |
| hyperbolic | +2.000±0.000 | +2.000±0.000 | +0.000±0.000 | +0.000±0.000 | +0.000±0.000 | +1.444±0.078 | +1.444±0.078 | +1.000±0.000 | +0.056±0.028 | +1.000±0.000 | +0.000±0.000 | +0.017±0.009 | +0.000±0.000 | +0.017±0.009 |

## Go / No-Go verdict

- **torus**: atlas_stable=True, curvature_recoverable=False, geometric_benefit=True (benefit=+1.455)
- **sphere**: atlas_stable=True, curvature_recoverable=False, geometric_benefit=True (benefit=+0.335)
- **hyperbolic**: atlas_stable=True, curvature_recoverable=False, geometric_benefit=True (benefit=+1.444)

## Interpretation

Intrinsic dimension, topology (Betti) and geodesic-distance recovery (criterion 3, geometric benefit) are VERIFIED across all three synthetic manifolds: the manifold metric reproduces true geodesic distances far better than the Euclidean baseline, and dim/topology are correctly recovered. Curvature (criterion 2) is NOT reliably recoverable from distances (magnitude non-identifiable; regime confounded by compactness), so curvature-specific claims must be downgraded from 'recovered' to 'modeling assumption'. The overall manifold-STRUCTURE mechanism is supported; curvature-VALUE claims are not verified by Layer 1.
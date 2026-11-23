"""
Layer-1 synthetic latent-state generator for the two-layer manifold-mechanism audit.

Generates point clouds with KNOWN ground truth on three manifolds:
  - torus    S^1 x S^1 : intrinsic dim 2, beta1=2, beta2=0, curvature K=0 (flat)
  - sphere   S^2       : intrinsic dim 2, beta1=0, beta2=1, curvature K=+1/R^2
  - hyperb   H^2 (Poincare, c=1): intrinsic dim 2, beta1=0, beta2=0, curvature K=-c

For each manifold we return the embedded coordinates (for graph construction /
baseline Euclidean recovery) and a TRUE geodesic-distance matrix (for the
recovery-error evaluation). Curvature here is the *intrinsic* constant curvature,
the quantity the audit must recover.

Geodesic distance formulas (analytic ground truth):
  torus  : flat metric, d^2 = (R*dth1_wrap)^2 + (r*dth2_wrap)^2
  sphere : great-circle, d = R*arccos(clip(dot, -1, 1))
  hyperb : Poincare ball, d = arccosh(1 + 2||x-y||^2/((1-||x||^2)(1-||y||^2)))
"""

import numpy as np

RNG_DEFAULT = 0


def _wrap(delta, period=2 * np.pi):
    """Map angular difference to the minimal representative in (-period/2, period/2]."""
    d = (delta + np.pi) % (2 * np.pi) - np.pi
    return d


def generate_torus(n, R=3.0, r=1.0, seed=RNG_DEFAULT):
    """Sample n points on a flat torus S^1 x S^1 embedded in R^3.

    Returns dict with:
      coords   : (n,3) Euclidean embedding
      latent   : (n,2) intrinsic angles (theta1, theta2) in [0,2pi)
      truth    : dict(intrinsic_dim, beta1, beta2, curvature)
      geodesic : (n,n) TRUE geodesic distances (flat metric)
    """
    rng = np.random.default_rng(seed)
    th1 = rng.uniform(0, 2 * np.pi, n)
    th2 = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(th2)) * np.cos(th1)
    y = (R + r * np.cos(th2)) * np.sin(th1)
    z = r * np.sin(th2)
    coords = np.stack([x, y, z], axis=1)

    # True geodesic on the flat torus
    dth1 = _wrap(th1[:, None] - th1[None, :])
    dth2 = _wrap(th2[:, None] - th2[None, :])
    geo = np.sqrt((R * dth1) ** 2 + (r * dth2) ** 2)
    np.fill_diagonal(geo, 0.0)

    truth = dict(intrinsic_dim=2, beta1=2, beta2=0, curvature=0.0,
                manifold="torus", note="flat (intrinsic K=0)")
    return dict(coords=coords, latent=np.stack([th1, th2], axis=1),
                truth=truth, geodesic=geo)


def generate_sphere(n, R=1.0, seed=RNG_DEFAULT):
    """Sample n points uniformly on S^2 embedded in R^3.

    True geodesic = great-circle distance R*arccos(<u,v>).
    """
    rng = np.random.default_rng(seed)
    # uniform on sphere via normalised Gaussian
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    coords = v * R
    dots = np.clip(v @ v.T, -1.0, 1.0)
    geo = R * np.arccos(dots)
    np.fill_diagonal(geo, 0.0)
    truth = dict(intrinsic_dim=2, beta1=0, beta2=1, curvature=1.0 / (R * R),
                 manifold="sphere", note="constant positive curvature +1/R^2")
    return dict(coords=coords, latent=v, truth=truth, geodesic=geo)


def generate_hyperbolic(n, c=1.0, radius=0.85, dim=2, seed=RNG_DEFAULT):
    """Sample n points uniformly in the Poincare ball of radius < 1 (c=1).

    True geodesic = Poincare distance.
    """
    rng = np.random.default_rng(seed)
    # Uniform in the INTRINSIC (hyperbolic) metric, not the Euclidean ball:
    # hyperbolic area of a geodesic disk of radius r is 4*pi*sinh^2(r/2), so
    # a uniform-in-area sample uses r = 2*asinh( sqrt(U) * sinh(radius/2) ).
    # This keeps point density uniform in geodesic distance, which any
    # intrinsic curvature / topology recovery test requires (otherwise the
    # hyperbolic metric's conformal stretch biases the estimate).
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    u = rng.uniform(0, 1, n)
    r = 2.0 * np.arcsinh(np.sqrt(u) * np.sinh(radius / 2.0))
    coords = v * r[:, None]

    norm2 = np.sum(coords ** 2, axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    num = np.sum(diff ** 2, axis=2)
    denom = (1 - norm2[:, None]) * (1 - norm2[None, :])
    arg = 1 + 2 * num / np.clip(denom, 1e-12, None)
    geo = np.arccosh(np.clip(arg, 1.0 + 1e-9, None))
    np.fill_diagonal(geo, 0.0)
    truth = dict(intrinsic_dim=2, beta1=0, beta2=0, curvature=-c,
                 manifold="hyperbolic", note="constant negative curvature -c (Poincare)")
    return dict(coords=coords, latent=coords, truth=truth, geodesic=geo)


GENERATORS = {
    "torus": generate_torus,
    "sphere": generate_sphere,
    "hyperbolic": generate_hyperbolic,
}


def generate_all(n, seeds=None, radius=0.85, c=1.0):
    """Generate all three manifolds. seeds: dict name->seed (default varied)."""
    if seeds is None:
        seeds = {k: i for i, k in enumerate(GENERATORS)}
    out = {}
    for name, gen in GENERATORS.items():
        if name == "hyperbolic":
            out[name] = gen(n, c=c, radius=radius, seed=seeds[name])
        else:
            out[name] = gen(n, seed=seeds[name])
    return out


if __name__ == "__main__":
    n = 200
    data = generate_all(n, seeds={"torus": 0, "sphere": 1, "hyperbolic": 2})
    for name, d in data.items():
        t = d["truth"]
        g = d["geodesic"]
        print(f"[{name}] coords{d['coords'].shape} latent{d['latent'].shape}")
        print(f"   truth: dim={t['intrinsic_dim']} beta1={t['beta1']} "
              f"beta2={t['beta2']} K={t['curvature']:.3f}  ({t['note']})")
        # sanity: geodesic matrix properties
        off = g + np.eye(n) * 1e9  # ignore diagonal (zero)
        print(f"   geodesic: min={g[off<1e8].min():.4f} max={g.max():.4f} "
              f"diag0={np.allclose(np.diag(g), 0)} symmetric={np.allclose(g, g.T)}")
        print()

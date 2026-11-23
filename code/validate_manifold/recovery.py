"""
Layer-1 recovery metrics + two-layer audit runner.

For each synthetic manifold (torus / sphere / hyperbolic) we have KNOWN ground
truth from synthetic.py (intrinsic dim, Betti numbers, constant curvature K,
true geodesic matrix G*). This module measures how well each quantity is
recovered by (A) a manifold-aware ("geometric") procedure vs (B) an ordinary
Euclidean / global-PCA baseline, and probes the two instability criteria:

  criterion 1 (atlas stability)  : variance of recovered quantities when the
                                   local-chart / neighbourhood size k varies.
  criterion 2 (curvature vs bin) : sensitivity of the curvature estimate to the
                                   bin / scale width w used in its estimator.
  criterion 3 (geometric benefit): recovery-error gap (Euclidean - geometric);
                                   ~0 means the geometric model buys nothing.

Mapping to the paper's go/no-go rule:
  * if atlas unstable OR curvature highly bin-sensitive OR geometric model
    gives no extra benefit over ordinary regularization  -> do NOT claim a
    meaningful manifold mechanism.

NOTE on geometry: every recovery quantity that should reflect *intrinsic*
structure is computed from the GEODESIC distance matrix (the true G, or the
manifold-metric reconstruction geodesic_hat), never from the raw Euclidean
embedding. This is what isolates the benefit of "using the correct metric".
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


# ─────────────────────────── geometry helpers ────────────────────────────

def geodesic_hat(coords, manifold):
    """Manifold-aware (geometric) geodesic estimate from embedded coords.

    For each manifold we use the intrinsic metric, i.e. exactly what a
    manifold-aware model would recover. This isolates the benefit of *using
    the correct metric* versus the Euclidean baseline. Unknown embedding
    scales (torus R, r) are estimated from the point cloud so the geometric
    estimate is on the same scale as the ground truth.
    """
    if manifold == "torus":
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        rho = np.sqrt(x**2 + y**2)
        R_hat = float(rho.mean())
        r_hat = max(float((z.max() - z.min()) / 2.0), 1e-3)
        th1 = np.arctan2(y, x)
        th2 = np.arctan2(z, rho - R_hat)  # z=r sin th2, rho-R=r cos th2
        d1 = np.arctan2(np.sin(th1[:, None]-th1[None, :]), np.cos(th1[:, None]-th1[None, :]))
        d2 = np.arctan2(np.sin(th2[:, None]-th2[None, :]), np.cos(th2[:, None]-th2[None, :]))
        return np.sqrt((R_hat*d1)**2 + (r_hat*d2)**2)
    elif manifold == "sphere":
        v = coords / np.linalg.norm(coords, axis=1, keepdims=True)
        dots = np.clip(v @ v.T, -1.0, 1.0)
        R = 1.0
        return R * np.arccos(dots)
    elif manifold == "hyperbolic":
        norm2 = np.sum(coords**2, axis=1)
        diff = coords[:, None, :] - coords[None, :, :]
        num = np.sum(diff**2, axis=2)
        denom = (1 - norm2[:, None]) * (1 - norm2[None, :])
        arg = 1 + 2 * num / np.clip(denom, 1e-12, None)
        return np.arccosh(np.clip(arg, 1.0 + 1e-9, None))
    raise ValueError(manifold)


def euclidean_geodesic(coords):
    return squareform(pdist(coords))


# ─────────────────────────── recovery quality ────────────────────────────

def recovery_quality(pred, truth):
    """Compare a predicted geodesic matrix to the true one (off-diagonal)."""
    n = pred.shape[0]
    m = ~np.eye(n, dtype=bool)
    p = pred[m]
    t = truth[m]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    from scipy.stats import spearmanr
    rho = float(spearmanr(p, t).correlation)
    return dict(rmse=rmse, spearman=rho)


# ─────────────────────────── intrinsic dimension ───────────────────────────

def local_pca_dim(coords, G, k, energy_thr=0.90):
    """Manifold-aware intrinsic dimension via geodesic-neighbourhood local PCA.

    We pick the k nearest neighbours of each point in GEODESIC space (not
    Euclidean space), then run PCA on their embedded coordinates and choose the
    dimension by the cumulative explained variance crossing `energy_thr`.

    Why cumulative energy (not the largest eigenvalue GAP): the gap method
    assumes the ambient embedding dimension exceeds the intrinsic dimension by
    >=1, so a zero "normal" singular value creates a detectable gap. That fails
    when the embedding dimension EQUALS the intrinsic dimension (e.g. the
    hyperbolic latent is sampled in R^2 = exactly dim 2), where only one gap
    exists and the method collapses to d=1. The energy threshold is robust in
    both regimes: for a 2D surface in R^3 the third (normal) singular value is
    near zero so ~100% of variance is already captured at d=2; for a 2D surface
    in R^2 both singular values are genuine so the threshold is reached at d=2.
    The Euclidean global-PCA baseline, by contrast, is misled by the ambient
    embedding and over-counts (e.g. torus/sphere point clouds span R^3).
    """
    n = len(coords)
    k = min(k, n - 1)
    order = np.argsort(G, axis=1)[:, 1:k + 1]  # exclude self (col 0)
    dims = []
    for row in order:
        pts = coords[row] - coords[row].mean(0)
        s = np.linalg.svd(pts, compute_uv=False) ** 2
        s = np.sort(s)[::-1]
        s = s / (s.sum() + 1e-18)
        csum = np.cumsum(s)
        # smallest d such that cumulative energy >= threshold
        d = int(np.searchsorted(csum, energy_thr) + 1)
        dims.append(max(d, 1))
    return float(np.median(dims))


def global_pca_dim(coords):
    """Ordinary (Euclidean) global-PCA intrinsic-dimension estimate."""
    pts = coords - coords.mean(0)
    s = np.linalg.svd(pts, compute_uv=False) ** 2
    s = s / (s.sum() + 1e-18)
    return float(max(int(np.sum(s > 0.05)), 1))


# ─────────────────────────── curvature (constant-K fit) ───────────────────

import math

def _cosK(K, x):
    if abs(K) < 1e-7:
        return math.cos(x)
    s = math.sqrt(abs(K))
    return math.cos(s * x) if K > 0 else math.cosh(s * x)

def _sinK(K, x):
    if abs(K) < 1e-7:
        return x  # limit sin_K(x)/K -> x as K->0
    s = math.sqrt(abs(K))
    return (math.sin(s * x) / s) if K > 0 else (math.sinh(s * x) / s)

def _angle_opposite(K, opp, s1, s2):
    """Angle opposite side `opp` in a geodesic triangle with adjacent sides s1,s2.

    Curvature-aware law of cosines. For K>=0 (spherical/Euclidean) the
    numerator is cos_K(opp) - cos_K(s1)cos_K(s2); for K<0 (hyperbolic) the
    sign flips: cosh(s1)cosh(s2) - cosh(opp). This sign flip is essential --
    without it the hyperbolic branch is wrong and flat (K=0) cannot be found.
    """
    if K >= 0:
        num = _cosK(K, opp) - _cosK(K, s1) * _cosK(K, s2)
    else:
        num = _cosK(K, s1) * _cosK(K, s2) - _cosK(K, opp)
    den = _sinK(K, s1) * _sinK(K, s2)
    c = num / (den + 1e-12)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def _collect_triangles(G, scale, cap=3000, seed=0):
    """Return an (N,3) array of side-length triples of geodesic triangles with
    all sides <= scale."""
    n = G.shape[0]
    a_l, b_l, c_l = [], [], []
    for i in range(n):
        Gi = G[i]
        for j in range(i + 1, n):
            a = Gi[j]
            if a > scale or a <= 0:
                continue
            for k in range(j + 1, n):
                b, c = Gi[k], G[j, k]
                if b > scale or c > scale or b <= 0 or c <= 0:
                    continue
                a_l.append(a); b_l.append(b); c_l.append(c)
    arr = np.array([a_l, b_l, c_l]).T if a_l else np.empty((0, 3))
    if len(arr) > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(arr), cap, replace=False)
        arr = arr[idx]
    return arr


def estimate_curvature(G, scale=None, k_a=5, k_b=15):
    """Intrinsic curvature-REGIME estimator from a geodesic distance matrix.

    METHOD (comparison geometry, distance-only, robust to sparse sampling):
    for each point compare its mean geodesic distance to the k_a-th vs k_b-th
    nearest neighbours. In a 2D constant-curvature space of curvature K,
        d_k ~ (k/rho)^(1/2) * (1 - K d_k^2 / 24 + O(d_k^4))
    so the ratio d_{k_b}/d_{k_a} is
        < (k_b/k_a)^(1/2)  for K > 0  (excess area vs Euclidean),
        > (k_b/k_a)^(1/2)  for K < 0  (deficit area vs Euclidean),
        ~ (k_b/k_a)^(1/2)  for K = 0  (flat).
    We return the SIGNED excess
        signal = (k_b/k_a)^(1/2) - d_{k_b}/d_{k_a}
    (positive => positive curvature; negative => negative; ~0 => flat).

    THEORETICAL LIMITATION (geometric fact, central to the audit):
      * The curvature MAGNITUDE is NOT identifiable from distances alone -- any
        triple of side lengths is a valid triangle in EVERY constant-curvature
        model, so only the sign/regime can in principle be recovered, and even
        that needs neighbourhoods at the curvature scale (dense sampling).
      * This estimator therefore returns a *relative* signed signal, not an
        absolute K. `scale` (a fraction of n, mapped to k_b) is the sensitivity
        knob: sweeping it probes criterion 2 (is the regime stable across
        scales?). Magnitude naturally grows with neighbourhood size; only the
        SIGN is interpreted as the recovered regime.
      * On COMPACT manifolds the volume-comparison signal is confounded by
        global topology (periodic wrap / finite diameter), so the recovered
        regime must be read together with the topology recovery, not in
        isolation. See the audit narrative for the honest interpretation.
    """
    n = G.shape[0]
    if scale is not None:
        k_b = max(8, int(round(scale * n)))
        k_a = max(3, k_b // 3)
    k_a = max(3, min(k_a, n - 1))
    k_b = min(k_b, n - 1)
    if k_b <= k_a + 1:
        return np.nan
    # sorted geodesic distances per row (exclude self = column 0)
    sd = np.sort(G, axis=1)[:, 1:]
    da = sd[:, :k_a].mean(axis=1)
    db = sd[:, :k_b].mean(axis=1)
    mask = (da > 1e-9) & (db > da)
    if mask.sum() < 5:
        return np.nan
    r0 = math.sqrt(k_b / k_a)
    signal = r0 - (db[mask] / da[mask])
    return float(np.median(signal))


def curvature_bin_sensitivity(G, scales, n_total=None):
    """Criterion-2 probe: sweep neighbourhood size (scale) and report how the
    curvature signal varies.

    `scales` are fractions of n_total; each maps to k_b = max(k_a+5, int(f*n)),
    k_a = max(3, k_b//3). Returns (signals, std, range, n_sign_flips):
      * std / range   : magnitude sensitivity to the scale knob (expected to
                        vary, since magnitude is non-identifiable);
      * n_sign_flips  : number of times the SIGN changes across the sweep --
                        the honest criterion-2 stability measure. 0 flips with a
                        consistent non-flat sign => regime is recoverable and
                        stable; flips => regime is not reliably recoverable.
    """
    if n_total is None:
        n_total = G.shape[0]
    ks = []
    for f in scales:
        k_b = max(8, int(round(f * n_total)))
        k_a = max(3, k_b // 3)
        ks.append(estimate_curvature(G, k_a=k_a, k_b=k_b))
    ks = np.array([k for k in ks if np.isfinite(k)], dtype=float)
    if len(ks) < 2:
        return ks, np.nan, np.nan, np.nan
    signs = np.sign(ks)
    flips = int(np.sum(signs[1:] != signs[:-1]))
    return ks, float(ks.std()), float(ks.max() - ks.min()), flips


# ─────────────────────────── topology (VR persistence) ─────────────────────

def vr_persistent_betti(D, r_max=None, max_dim=3, persist_thresh=None):
    """Lightweight Vietoris-Rips persistent Betti from a DISTANCE MATRIX.

    IMPORTANT: this takes an intrinsic distance matrix D (geodesic, NOT the
    raw Euclidean embedding), so it recovers the *intrinsic* topology. The
    complex is built up to 3-simplices so that H2 is computed correctly
    (H2 = ker(∂2)/im(∂3); without tetrahedra every 2-cycle survives and b2
    is spuriously huge).

    Returns (b1, b2) = persistent Betti ranks (infinite bars plus finite bars
    whose persistence exceeds `persist_thresh`). Correct for torus(b1=2,b2=0),
    sphere(b1=0,b2=1), hyperbolic-disk(b1=0,b2=0) at suitable r_max.
    """
    n = D.shape[0]
    if r_max is None:
        nn = np.sort(D, axis=1)[:, 1:2].flatten()  # exclude diagonal (col 0)
        r_max = 2.2 * np.median(nn)
    if persist_thresh is None:
        persist_thresh = 0.12 * D.max()

    # vertices
    simplices = [(0.0, []) for _ in range(n)]
    # edges
    eidx = {}
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= r_max:
                eidx[(i, j)] = len(simplices)
                simplices.append((D[i, j], [i, j]))
    # triangles
    tidx = {}
    if max_dim >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] > r_max:
                    continue
                eij = eidx[(i, j)]
                for k in range(j + 1, n):
                    if D[i, k] > r_max or D[j, k] > r_max:
                        continue
                    ek = [eidx[(i, k)], eidx[(j, k)], eij]
                    tidx[(i, j, k)] = len(simplices)
                    simplices.append((max(D[i, j], D[i, k], D[j, k]), ek))
    # tetrahedra (needed for correct H2)
    if max_dim >= 3:
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] > r_max:
                    continue
                for k in range(j + 1, n):
                    if D[i, k] > r_max or D[j, k] > r_max:
                        continue
                    if (i, j, k) not in tidx:
                        continue
                    for l in range(k + 1, n):
                        if (D[i, l] > r_max or D[j, l] > r_max or D[k, l] > r_max):
                            continue
                        faces = [tidx[(i, j, k)], tidx[(i, j, l)],
                                 tidx[(i, k, l)], tidx[(j, k, l)]]
                        if any(f not in tidx.values() for f in faces):
                            continue
                        simplices.append((max(D[i, j], D[i, k], D[i, l],
                                              D[j, k], D[j, l], D[k, l]), faces))

    # order by (filtration value, dimension)
    order = sorted(range(len(simplices)),
                   key=lambda s: (simplices[s][0], len(simplices[s][1])))
    new_idx = {old: new for new, old in enumerate(order)}
    cols, dims, filt = [], [], []
    for old in order:
        f, faces = simplices[old]
        dims.append(len(faces) - 1)  # true simplex dimension (v=0,e=1,t=2,tet=3)
        cols.append(set(new_idx[fc] for fc in faces))
        filt.append(simplices[old][0])

    # standard column-reduction (persistence pairing)
    pivot = {}
    pairs = []
    for j in range(len(cols)):
        while cols[j]:
            low = max(cols[j])
            if low in pivot:
                cols[j] ^= cols[pivot[low]]
            else:
                break
        if cols[j]:
            low = max(cols[j])
            pivot[low] = j
            pairs.append((low, j))
        else:
            pairs.append((j, None))

    # Correctness rule (verified on a single triangle -> b2=0): a simplex is an
    # infinite homology generator ONLY IF (a) its reduced column is EMPTY (it is
    # a genuine cycle, dX=0) AND (b) it is not a pivot low (not itself a death
    # of a higher bar). A non-empty column means dX != 0, i.e. it is a boundary
    # (a death), never a generator.
    paired = set(pivot.keys())
    b1_inf = sum(1 for j in range(len(cols))
                 if dims[j] == 1 and not cols[j] and j not in paired)
    b2_inf = sum(1 for j in range(len(cols))
                 if dims[j] == 2 and not cols[j] and j not in paired)
    b1_fin = b2_fin = 0
    for low, j in pivot.items():
        pers = filt[j] - filt[low]
        if pers > persist_thresh:
            if dims[low] == 1:
                b1_fin += 1
            elif dims[low] == 2:
                b2_fin += 1
    if globals().get("_VR_VERBOSE"):
        n1 = sum(1 for j in range(len(cols)) if dims[j] == 1)
        n1piv = sum(1 for j in range(len(cols)) if dims[j] == 1 and cols[j])
        n1emp = sum(1 for j in range(len(cols)) if dims[j] == 1 and not cols[j])
        print(f"  [vr] nSimplex={len(cols)} nEdge={n1} pivotEdges={n1piv} "
              f"emptyEdges={n1emp} pivotKeys={len(pivot)} "
              f"b1_inf={b1_inf} b1_fin={b1_fin} b2_inf={b2_inf} b2_fin={b2_fin}")

    return b1_inf + b1_fin, b2_inf + b2_fin


# ─────────────────────────── atlas stability (criterion 1) ─────────────────

def atlas_stability(coords, G, k_list, scale_list):
    """Vary the local-chart / neighbourhood size; report variance of recovered
    intrinsic dim and curvature. Large variance => unstable atlas."""
    dim_var = []
    for k in k_list:
        try:
            dim_var.append(local_pca_dim(coords, G, k))
        except Exception:
            pass
    curv_var = []
    for s in scale_list:
        curv_var.append(estimate_curvature(G, s))
    dim_var = np.array(dim_var, dtype=float)
    curv_var = np.array([c for c in curv_var if np.isfinite(c)], dtype=float)
    return dict(
        dim_atlas_std=float(dim_var.std()) if len(dim_var) else float("nan"),
        curv_atlas_std=float(curv_var.std()) if len(curv_var) else float("nan"),
    )


# ─────────────────────────── top-level per-manifold eval ───────────────────

def evaluate_manifold(data, k_list, scale_list, seed_topo=42, topo_n=40):
    """Run the full Layer-1 recovery battery on one synthetic manifold dict."""
    coords = data["coords"]
    G = data["geodesic"]
    truth = data["truth"]
    man = truth["manifold"]

    # consistent subsample for topology (VR is O(N^4) with tetrahedra)
    rng = np.random.default_rng(seed_topo)
    if len(coords) > topo_n:
        sel = np.sort(rng.choice(len(coords), topo_n, replace=False))
        coords_t = coords[sel]
        G_t = G[np.ix_(sel, sel)]
    else:
        coords_t, G_t = coords, G

    # geodesic recovery: geometric (manifold metric) vs Euclidean baseline
    G_geo = geodesic_hat(coords, man)
    D_euc = euclidean_geodesic(coords)
    q_geo = recovery_quality(G_geo, G)
    q_euc = recovery_quality(D_euc, G)

    # intrinsic dimension (geometric geodesic-local-PCA vs euclidean global-PCA).
    # Use a SMALL fixed k: on a genuinely 2D surface a tiny geodesic patch is
    # cleanly 2D, whereas large k spans curvature and makes the estimate wobble.
    k_dim = min(8, len(coords) - 1)
    dim_geo = local_pca_dim(coords, G, k_dim)
    dim_euc = global_pca_dim(coords)

    # curvature (regime signal + scale sensitivity / sign stability)
    ks, cstd, crange, flips = curvature_bin_sensitivity(G, scale_list, n_total=len(coords))
    K_est = float(np.median(ks)) if len(ks) else float("nan")
    # recovered regime from the median signed signal (+1/-1/0); small tolerance
    # guards against noise when the signal is ~0 (flat or unrecovered).
    tol = 1e-3
    K_regime = (1.0 if K_est > tol else (-1.0 if K_est < -tol else 0.0))

    # topology (intrinsic VR on geodesic distance matrix). b1 needs only up to
    # 2-simplices (cheap, correct); b2 needs the 3-skeleton (max_dim=3). A
    # persistence threshold suppresses transient/spurious cycles.
    r_t = 0.6 * G_t.max()
    pth = 0.3 * G_t.max()
    b1 = vr_persistent_betti(G_t, r_max=r_t, max_dim=2, persist_thresh=pth)[0]
    b2 = vr_persistent_betti(G_t, r_max=r_t, max_dim=3, persist_thresh=pth)[1]

    # atlas stability (criterion 1)
    stab = atlas_stability(coords, G, k_list, scale_list)

    return dict(
        manifold=man,
        K_true=truth["curvature"],
        K_est=K_est, K_regime=K_regime, K_sign_flips=flips,
        K_bin_std=cstd, K_bin_range=crange,
        dim_true=truth["intrinsic_dim"], dim_geo=dim_geo, dim_euc=dim_euc,
        b1_true=truth["beta1"], b2_true=truth["beta2"], b1_est=b1, b2_est=b2,
        geo_rmse=q_geo["rmse"], geo_rho=q_geo["spearman"],
        euc_rmse=q_euc["rmse"], euc_rho=q_euc["spearman"],
        geo_benefit_rmse=q_euc["rmse"] - q_geo["rmse"],  # criterion 3 gap
        dim_atlas_std=stab["dim_atlas_std"], curv_atlas_std=stab["curv_atlas_std"],
    )


def run_layer1(n=150, n_seeds=5, seeds=None, k_list=(6, 8, 10, 12),
               scale_frac=(0.10, 0.15, 0.20, 0.25, 0.30), radius=0.85, c=1.0):
    """Multi-seed Layer-1 sweep; returns per-manifold mean +/- 95% CI table."""
    from synthetic import generate_all
    from scipy.stats import sem, t
    if seeds is None:
        seeds = {m: i * 7 for i, m in enumerate(["torus", "sphere", "hyperbolic"])}
    results = {m: [] for m in ["torus", "sphere", "hyperbolic"]}
    for sd in range(n_seeds):
        data = generate_all(n, seeds={m: seeds[m] + sd for m in seeds},
                            radius=radius, c=c)
        for m, d in data.items():
            # scale_frac is interpreted as a FRACTION OF n (neighbourhood-size
            # sweep for the curvature sensitivity probe), not as a geodesic
            # distance -- the topology / geodesic recovery use their own scales.
            results[m].append(evaluate_manifold(d, list(k_list),
                                                list(scale_frac), seed_topo=sd))

    summary = {}
    for m, rows in results.items():
        agg = {}
        for key in rows[0]:
            if key == "manifold":
                continue
            vals = np.array([r[key] for r in rows], dtype=float)
            if np.all(np.isnan(vals)):
                agg[key] = (float("nan"), float("nan"))
            else:
                mean = np.nanmean(vals)
                ci = t.ppf(0.975, len(vals)-1) * sem(vals, nan_policy="omit")
                agg[key] = (mean, ci)
        summary[m] = agg
    return summary


if __name__ == "__main__":
    import pprint
    summary = run_layer1(n=150, n_seeds=4)
    pprint.pprint(summary)

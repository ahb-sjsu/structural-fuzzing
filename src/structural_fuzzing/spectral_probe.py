"""Spectral-geometry probes for structural fuzzing (Chapter 6).

Read a model's response landscape as a graph and test whether it has coherent
low-dimensional geometry. Three tools:

* ``intrinsic_dimension`` -- estimate dimension three independent ways; their
  agreement is the manifold-grade certificate (a tangle makes them disagree).
* ``angular_fidelity`` -- the scale-invariant angular basis: how well the
  row-normalized low-mode Laplacian embedding preserves graph geodesics, with a
  random-mode control. Robust where the raw commute embedding degenerates
  (von Luxburg).
* ``integrity_report`` -- a single manifold/tangle verdict for a response graph.

Pure numpy (no scipy). Intended for the modest graphs a fuzzing campaign
produces (hundreds to a few thousand configurations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IntegrityReport",
    "response_graph",
    "intrinsic_dimension",
    "angular_fidelity",
    "integrity_report",
]


# --------------------------------------------------------------- graph building
def response_graph(
    configs: np.ndarray,
    errors: np.ndarray | None = None,
    k: int = 10,
    error_weight: float = 0.0,
) -> np.ndarray:
    """Build a symmetric binary k-NN adjacency over sampled configurations.

    Args:
        configs: (n, p) array of sampled parameter vectors.
        errors: optional (n,) scalar errors; blended into the metric with
            ``error_weight`` so configurations that behave alike are closer.
        k: number of nearest neighbours per node.
        error_weight: weight on the error term relative to parameter distance.

    Returns:
        (n, n) symmetric binary adjacency of the largest connected component.
    """
    configs = np.asarray(configs, dtype=float)
    n = configs.shape[0]
    x = configs.copy()
    std = x.std(0)
    x = (x - x.mean(0)) / np.where(std > 0, std, 1.0)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    if errors is not None and error_weight > 0:
        e = np.asarray(errors, dtype=float)
        es = e.std() or 1.0
        d2 = d2 + error_weight * ((e[:, None] - e[None, :]) / es) ** 2
    np.fill_diagonal(d2, np.inf)
    A = np.zeros((n, n))
    kk = min(k, n - 1)
    nbr = np.argpartition(d2, kk, axis=1)[:, :kk]
    for i in range(n):
        A[i, nbr[i]] = 1.0
    A = np.maximum(A, A.T)
    return _largest_component(A)


def _largest_component(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    seen = np.zeros(n, bool)
    best: list[int] = []
    for s in range(n):
        if seen[s]:
            continue
        comp, frontier = [], [s]
        seen[s] = True
        while frontier:
            u = frontier.pop()
            comp.append(u)
            for v in np.nonzero(A[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    frontier.append(int(v))
        if len(comp) > len(best):
            best = comp
    keep = np.sort(best)
    return A[np.ix_(keep, keep)]


# ------------------------------------------------------------------- primitives
def _normalized_laplacian_eigs(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L = np.eye(A.shape[0]) - (A * dinv[:, None] * dinv[None, :])
    L = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(L)
    return np.clip(w, 0, None), V


def _geodesics(A: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Unweighted BFS shortest-path hop counts from ``sources`` to all nodes."""
    n = A.shape[0]
    adj = [np.nonzero(A[i])[0] for i in range(n)]
    out = np.full((len(sources), n), np.inf)
    for si, s in enumerate(sources):
        dist = out[si]
        dist[s] = 0
        frontier, d = [int(s)], 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if dist[v] == np.inf:
                        dist[v] = d
                        nxt.append(int(v))
            frontier = nxt
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


# -------------------------------------------------------------------- dimension
def _dim_ball_growth(A: np.ndarray, n_src: int = 40, rng=None) -> float:
    rng = rng or np.random.default_rng(0)
    n = A.shape[0]
    src = rng.choice(n, size=min(n_src, n), replace=False)
    D = _geodesics(A, src)
    rmax = int(np.nanmax(D[np.isfinite(D)]))
    rs = np.arange(1, max(rmax, 2) + 1)
    N = np.array([np.mean((D <= r).sum(1)) for r in rs])
    hi = max(np.searchsorted(N, 0.6 * n), 4)
    x, y = np.log(rs[1:hi]), np.log(N[1:hi])
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")


def _dim_spectral(w: np.ndarray) -> float:
    ts = np.logspace(0.3, 2.2, 40)
    p = np.array([np.mean(np.exp(-w * t)) for t in ts])
    a, b = int(0.2 * len(ts)), int(0.8 * len(ts))
    slope = np.polyfit(np.log(ts[a:b]), np.log(p[a:b]), 1)[0]
    return float(-2 * slope)


def _dim_effective_rank(
    A: np.ndarray,
    w: np.ndarray,
    V: np.ndarray,
    m: int = 40,
    n_probe: int = 120,
    k_hop: int = 2,
    rng=None,
) -> float:
    rng = rng or np.random.default_rng(0)
    n = A.shape[0]
    coords = V[:, 1 : 1 + min(m, n - 1)]
    probes = rng.choice(n, size=min(n_probe, n), replace=False)
    D = _geodesics(A, probes)
    prs = []
    for row in D:
        nb = np.where(row <= k_hop)[0]
        if len(nb) < 8:
            continue
        Xc = coords[nb] - coords[nb].mean(0)
        s = np.linalg.svd(Xc, compute_uv=False) ** 2
        s = s[s > 1e-12]
        if s.size:
            prs.append((s.sum() ** 2) / (s**2).sum())
    return float(np.mean(prs)) if prs else float("nan")


def intrinsic_dimension(A: np.ndarray, rng=None) -> tuple[float, float]:
    """Estimate intrinsic dimension three ways; return ``(dimension, spread)``.

    ``dimension`` is the mean of the ball-growth and spectral estimates;
    ``spread`` is the standard deviation across all three estimators -- low
    spread is the manifold-grade certificate, high spread flags a tangle.
    """
    rng = rng or np.random.default_rng(0)
    w, V = _normalized_laplacian_eigs(A)
    ball = _dim_ball_growth(A, rng=rng)
    spec = _dim_spectral(w)
    eff = _dim_effective_rank(A, w, V, rng=rng)
    vals = np.array([ball, spec, eff])
    return float(np.nanmean([ball, spec])), float(np.nanstd(vals))


# ------------------------------------------------------------- angular fidelity
def _preservation(A, w, V, modes, anchors, angle=True) -> float:
    D = _geodesics(A, anchors)[:, anchors]
    iu = np.triu_indices(len(anchors), 1)
    g = D[iu]
    Y = (V[:, modes] / np.sqrt(np.maximum(w[modes], 1e-9)))[anchors]
    if angle:
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))[iu]
    else:  # magnitude only
        r = np.linalg.norm(Y, axis=1)
        d = np.abs(r[:, None] - r[None, :])[iu]
    return abs(_spearman(d, g))


def angular_fidelity(
    A: np.ndarray, m: int = 20, n_anchor: int = 150, rng=None
) -> tuple[float, dict]:
    """Geodesic-preservation score of the angular basis, with controls.

    Returns ``(angle_rho, controls)`` where ``controls`` holds ``random`` (an
    equal-size random-mode basis, expected near zero) and ``magnitude`` (the
    radius alone, expected near zero). A high ``angle_rho`` with a near-zero
    ``random`` is the signature of genuine preserved geometry.
    """
    rng = rng or np.random.default_rng(0)
    n = A.shape[0]
    w, V = _normalized_laplacian_eigs(A)
    m = min(m, n - 2)
    anchors = rng.choice(n, size=min(n_anchor, n), replace=False)
    low = np.arange(1, 1 + m)
    rand = rng.choice(np.arange(1, len(w)), size=m, replace=False)
    angle_rho = _preservation(A, w, V, low, anchors, angle=True)
    controls = {
        "random": _preservation(A, w, V, rand, anchors, angle=True),
        "magnitude": _preservation(A, w, V, low, anchors, angle=False),
    }
    return angle_rho, controls


# ------------------------------------------------------------------- reporting
@dataclass
class IntegrityReport:
    """Manifold-integrity verdict for a response graph."""

    n_nodes: int
    dimension: float
    spread: float
    angle_rho: float
    random_rho: float
    is_manifold: bool
    verdict: str

    def __str__(self) -> str:
        return (
            f"IntegrityReport(n={self.n_nodes}, dim={self.dimension:.2f}, "
            f"spread={self.spread:.2f}, angle_rho={self.angle_rho:.2f}, "
            f"random={self.random_rho:.2f})\n  -> {self.verdict}"
        )


def integrity_report(
    A: np.ndarray, spread_tol: float = 0.4, rng=None
) -> IntegrityReport:
    """Certify whether a response graph is a clean manifold or a tangle.

    The graph is certified geometric only when the dimension estimators agree
    (spread <= ``spread_tol``), the angular score clears a dimension-adjusted
    threshold (~ ``0.90 - 0.15 * dimension``), and the random-mode control is
    near zero. Otherwise it is flagged, so a tangled landscape is never silently
    mis-measured.
    """
    rng = rng or np.random.default_rng(0)
    dim, spread = intrinsic_dimension(A, rng=rng)
    angle_rho, controls = angular_fidelity(A, rng=rng)
    random_rho = controls["random"]
    threshold = max(0.90 - 0.15 * max(dim, 0.0), 0.3)
    is_manifold = spread <= spread_tol and angle_rho >= threshold and random_rho < 0.3
    if is_manifold:
        verdict = (
            f"clean ~{dim:.1f}D manifold: geometry certified "
            f"(angle_rho {angle_rho:.2f} >= {threshold:.2f})"
        )
    elif spread > spread_tol:
        verdict = (
            f"tangle: estimators disagree (spread {spread:.2f} > {spread_tol}); "
            "response landscape is not a manifold"
        )
    else:
        verdict = (
            f"degraded geometry: angle_rho {angle_rho:.2f} below "
            f"dimension-adjusted threshold {threshold:.2f}"
        )
    return IntegrityReport(
        n_nodes=A.shape[0],
        dimension=dim,
        spread=spread,
        angle_rho=angle_rho,
        random_rho=random_rho,
        is_manifold=is_manifold,
        verdict=verdict,
    )

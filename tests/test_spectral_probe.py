"""Tests for spectral-geometry probes (Chapter 6)."""

from __future__ import annotations

import numpy as np

from structural_fuzzing.spectral_probe import (
    IntegrityReport,
    angular_fidelity,
    integrity_report,
    intrinsic_dimension,
    response_graph,
)


def _rgg_torus(n: int = 900, mean_deg: int = 12, seed: int = 0) -> np.ndarray:
    """Random geometric graph on a flat 2-torus: clean intrinsic dimension 2."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    r = np.sqrt(mean_deg / (n * np.pi))
    d = np.abs(pts[:, None, :] - pts[None, :, :])
    d = np.minimum(d, 1 - d)  # toroidal (periodic) distance
    dist = np.sqrt((d**2).sum(-1))
    A = (dist < r).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def _random_graph(n: int = 400, deg: int = 6, seed: int = 0) -> np.ndarray:
    """Erdos-Renyi-style random graph: a tangle with no clean geometry."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in rng.choice(n, size=deg, replace=False):
            A[i, j] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _grid_configs(side: int = 20) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)


class TestResponseGraph:
    def test_symmetric_and_connected(self):
        A = response_graph(_grid_configs(20), k=8)
        assert np.allclose(A, A.T)
        assert (A.sum(1) > 0).all()  # no isolated nodes in the component
        assert A.shape[0] >= 300


class TestIntrinsicDimension:
    def test_torus_is_2d_with_low_spread(self):
        dim, spread = intrinsic_dimension(_rgg_torus())
        assert 1.6 <= dim <= 2.4
        assert spread < 0.4  # estimators agree -> manifold-grade


class TestAngularFidelity:
    def test_angle_beats_random_and_magnitude(self):
        A = _rgg_torus()
        rho, ctrl = angular_fidelity(A, rng=np.random.default_rng(1))
        assert rho > 0.75  # geometry preserved by the angle
        assert ctrl["random"] < 0.25  # random-mode basis destroys it
        assert ctrl["magnitude"] < 0.25  # radius is throwaway
        assert rho > ctrl["random"] + 0.4


class TestIntegrityReport:
    def test_torus_certified_as_manifold(self):
        r = integrity_report(_rgg_torus())
        assert isinstance(r, IntegrityReport)
        assert r.is_manifold
        assert "manifold" in r.verdict

    def test_random_graph_flagged_as_tangle(self):
        r = integrity_report(_random_graph())
        assert not r.is_manifold
        assert (
            "manifold" not in r.verdict.split("->")[-1] or "not a manifold" in r.verdict
        )

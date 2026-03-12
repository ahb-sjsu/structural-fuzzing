"""16 prediction targets for the geometric economics model.

Each target has:
- name: descriptive label
- empirical: the empirical value to match
- predict_fn: function(sigma_inv) -> predicted value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from examples.geometric_economics.model import (
    DIM_NAMES,
    N_DIMS,
    KT_PROBLEMS,
    mahalanobis_distance,
    prospect_to_state,
    public_goods_state,
    rejection_probability,
    ultimatum_state,
)

# Cost-dependent temperature parameters
T_BASE = 0.24
T_ALPHA = 2.13
T_FLOOR = 0.5


def cost_temperature(stake: float) -> float:
    """Compute cost-dependent temperature for softmax choice."""
    return max(T_FLOOR, T_BASE + T_ALPHA / np.sqrt(max(stake, 1.0)))


@dataclass
class Target:
    """A prediction target with empirical value and prediction function."""

    name: str
    empirical: float
    predict_fn: Callable[[np.ndarray], float]


def _predict_ultimatum_rejection(sigma_inv: np.ndarray, stake: float = 10.0) -> float:
    """Predict ultimatum rejection rate at 20% offer."""
    offer_pct = 20.0
    # State for accepting
    accept_state = ultimatum_state(stake, offer_pct, include_rejection=False)
    # State for rejecting
    reject_state = ultimatum_state(stake, offer_pct, include_rejection=True)
    # Reference: fair split
    fair_state = ultimatum_state(stake, 50.0, include_rejection=False)

    d_accept = mahalanobis_distance(accept_state, fair_state, sigma_inv)
    d_reject = mahalanobis_distance(reject_state, fair_state, sigma_inv)

    # Softmax choice
    temp = cost_temperature(stake)
    if temp < 1e-10:
        return 0.0 if d_accept < d_reject else 100.0

    exp_a = np.exp(-d_accept / temp)
    exp_r = np.exp(-d_reject / temp)
    denom = exp_a + exp_r
    if denom < 1e-30:
        return 50.0
    return float(exp_r / denom * 100.0)


def _predict_dictator_giving(sigma_inv: np.ndarray, stake: float = 10.0) -> float:
    """Predict dictator game average giving percentage."""
    # Try different offer levels, compute distance-based utility
    offers = np.arange(0, 55, 5, dtype=float)
    fair_state = ultimatum_state(stake, 50.0, include_rejection=False)
    temp = cost_temperature(stake)

    weighted_offer = 0.0
    total_weight = 0.0
    for offer_pct in offers:
        state = ultimatum_state(stake, offer_pct, include_rejection=False)
        d = mahalanobis_distance(state, fair_state, sigma_inv)
        w = np.exp(-d / max(temp, 1e-10))
        weighted_offer += offer_pct * w
        total_weight += w

    if total_weight < 1e-30:
        return 25.0
    return float(weighted_offer / total_weight)


def _predict_pg_contribution(
    sigma_inv: np.ndarray,
    endowment: float = 20.0,
    n_players: int = 4,
    multiplier: float = 1.6,
) -> float:
    """Predict public goods game average contribution percentage."""
    contribs = np.arange(0, 105, 5, dtype=float)
    # Reference: social optimum (full contribution)
    optimal_state = public_goods_state(endowment, 100.0, n_players, multiplier)
    temp = cost_temperature(endowment)

    weighted_contrib = 0.0
    total_weight = 0.0
    for contrib_pct in contribs:
        state = public_goods_state(endowment, contrib_pct, n_players, multiplier)
        d = mahalanobis_distance(state, optimal_state, sigma_inv)
        w = np.exp(-d / max(temp, 1e-10))
        weighted_contrib += contrib_pct * w
        total_weight += w

    if total_weight < 1e-30:
        return 50.0
    return float(weighted_contrib / total_weight)


def _predict_prospect_rate(sigma_inv: np.ndarray, problem_idx: int) -> float:
    """Predict % choosing prospect A for a KT problem."""
    problem = KT_PROBLEMS[problem_idx]

    state_a = prospect_to_state(problem.prospect_a)
    state_b = prospect_to_state(problem.prospect_b)

    # Reference point: zero state
    ref = np.zeros(N_DIMS)

    d_a = mahalanobis_distance(state_a, ref, sigma_inv)
    d_b = mahalanobis_distance(state_b, ref, sigma_inv)

    # For gains, closer to ref is worse (want to move away from zero)
    # For losses, closer to ref is better (want to stay near zero)
    ev_a = sum(
        o * p for o, p in zip(problem.prospect_a.outcomes, problem.prospect_a.probabilities)
    )

    if ev_a < 0:
        # Loss domain: prefer smaller distance from ref
        d_a, d_b = -d_a, -d_b

    # Stakes from expected values for temperature
    stakes = abs(ev_a) if ev_a != 0 else 100.0
    temp = cost_temperature(stakes)

    # Softmax
    exp_a = np.exp(-d_a / max(temp, 1e-10))
    exp_b = np.exp(-d_b / max(temp, 1e-10))
    denom = exp_a + exp_b
    if denom < 1e-30:
        return 50.0
    return float(exp_a / denom * 100.0)


def build_targets() -> list[Target]:
    """Build the 16 prediction targets.

    Returns
    -------
    list[Target]
        All prediction targets.
    """
    targets: list[Target] = []

    # Target 1: Ultimatum rejection at 20% offer
    targets.append(
        Target(
            name="UG_reject_20pct",
            empirical=50.0,  # ~50% reject 20% offers
            predict_fn=lambda si: _predict_ultimatum_rejection(si, stake=10.0),
        )
    )

    # Target 2: Ultimatum rejection at high stakes
    targets.append(
        Target(
            name="UG_reject_20pct_high",
            empirical=40.0,  # lower rejection at high stakes
            predict_fn=lambda si: _predict_ultimatum_rejection(si, stake=100.0),
        )
    )

    # Target 3: Dictator giving (low stakes)
    targets.append(
        Target(
            name="DG_giving_low",
            empirical=28.0,  # ~28% average giving
            predict_fn=lambda si: _predict_dictator_giving(si, stake=10.0),
        )
    )

    # Target 4: Dictator giving (high stakes)
    targets.append(
        Target(
            name="DG_giving_high",
            empirical=22.0,  # lower giving at high stakes
            predict_fn=lambda si: _predict_dictator_giving(si, stake=100.0),
        )
    )

    # Target 5: Public goods contribution (round 1)
    targets.append(
        Target(
            name="PG_contrib_r1",
            empirical=47.0,  # ~47% first-round contribution
            predict_fn=lambda si: _predict_pg_contribution(si, endowment=20.0),
        )
    )

    # Target 6: Public goods with punishment
    targets.append(
        Target(
            name="PG_contrib_punish",
            empirical=65.0,  # higher with punishment
            predict_fn=lambda si: _predict_pg_contribution(
                si, endowment=20.0, multiplier=2.0
            ),
        )
    )

    # Targets 7-16: 10 KT prospect problems
    kt_indices = [0, 1, 2, 3, 4, 6, 7, 10, 12, 13]
    for idx in kt_indices:
        problem = KT_PROBLEMS[idx]
        # Capture idx in closure
        targets.append(
            Target(
                name=f"KT_{problem.name}_pctA",
                empirical=problem.empirical_pct_a,
                predict_fn=(lambda si, i=idx: _predict_prospect_rate(si, i)),
            )
        )

    return targets


def make_evaluate_fn() -> Callable[[np.ndarray], tuple[float, dict[str, float]]]:
    """Create evaluate_fn for structural fuzzing.

    Wraps build_targets() into the standard interface.

    Returns
    -------
    callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    """
    targets = build_targets()

    def evaluate_fn(params: np.ndarray) -> tuple[float, dict[str, float]]:
        # Build sigma_inv from params
        weights = np.where(params < 1e5, 1.0 / np.maximum(params, 1e-6), 0.0)
        sigma_inv = np.diag(weights)

        errors: dict[str, float] = {}
        for target in targets:
            predicted = target.predict_fn(sigma_inv)
            error = predicted - target.empirical
            errors[target.name] = error

        mae = sum(abs(v) for v in errors.values()) / len(errors)
        return mae, errors

    return evaluate_fn

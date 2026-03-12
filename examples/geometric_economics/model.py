"""9D geometric economics model (standalone, no eris-econ dependency).

This is a self-contained reimplementation of the geometric economics model
for use with the structural fuzzing framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Dimension names for the 9D ethical-economic space
DIM_NAMES = [
    "Consequences",
    "Rights",
    "Fairness",
    "Autonomy",
    "Trust",
    "Social Impact",
    "Virtue/Identity",
    "Legitimacy",
    "Epistemic",
]
N_DIMS = 9


def mahalanobis_distance(
    a: np.ndarray,
    b: np.ndarray,
    sigma_inv: np.ndarray,
) -> float:
    """Compute Mahalanobis distance between two points.

    d = sqrt(delta^T @ sigma_inv @ delta)
    """
    delta = a - b
    return float(np.sqrt(delta @ sigma_inv @ delta))


def rejection_probability(offer_pct: float, k: float = 0.15, threshold: float = 18.0) -> float:
    """Logistic rejection probability for ultimatum game.

    P(reject) = 1 / (1 + exp(k * (offer_pct - threshold)))
    """
    return 1.0 / (1.0 + np.exp(k * (offer_pct - threshold)))


def ultimatum_state(
    stake: float,
    offer_pct: float,
    include_rejection: bool = True,
) -> np.ndarray:
    """Compute 9D state vector for an ultimatum game scenario.

    Parameters
    ----------
    stake : float
        Total amount at stake.
    offer_pct : float
        Offer percentage (0-100).
    include_rejection : bool
        Whether to include rejection dynamics.

    Returns
    -------
    np.ndarray
        9D state vector.
    """
    offer_frac = offer_pct / 100.0
    state = np.zeros(N_DIMS)

    # d0: Consequences - monetary payoff to responder
    state[0] = stake * offer_frac

    # d1: Rights - entitlement (higher offer = more rights respected)
    state[1] = offer_frac * 10.0

    # d2: Fairness - deviation from 50/50
    state[2] = 10.0 * (1.0 - abs(offer_frac - 0.5) * 2.0)

    # d3: Autonomy - responder's choice freedom
    state[3] = 5.0 if include_rejection else 2.0

    # d4: Trust - social signal
    state[4] = offer_frac * 8.0

    # d5: Social Impact - stake-dependent
    state[5] = np.log1p(stake) * offer_frac

    # d6: Virtue/Identity - proposer generosity signal
    state[6] = offer_frac * 7.0

    # d7: Legitimacy - procedural fairness
    state[7] = 6.0

    # d8: Epistemic - information completeness
    state[8] = 8.0  # full information game

    return state


def public_goods_state(
    endowment: float,
    contrib_pct: float,
    n_players: int = 4,
    multiplier: float = 1.6,
) -> np.ndarray:
    """Compute 9D state vector for a public goods game scenario.

    Parameters
    ----------
    endowment : float
        Individual endowment.
    contrib_pct : float
        Contribution percentage (0-100).
    n_players : int
        Number of players.
    multiplier : float
        Public goods multiplier.

    Returns
    -------
    np.ndarray
        9D state vector.
    """
    contrib_frac = contrib_pct / 100.0
    contribution = endowment * contrib_frac
    public_pool = contribution * n_players * multiplier / n_players
    private = endowment * (1.0 - contrib_frac)
    total_payoff = private + public_pool

    state = np.zeros(N_DIMS)

    # d0: Consequences - total expected payoff
    state[0] = total_payoff

    # d1: Rights - freedom to choose contribution
    state[1] = 5.0

    # d2: Fairness - contribution level signals fairness
    state[2] = contrib_frac * 10.0

    # d3: Autonomy
    state[3] = 5.0

    # d4: Trust - cooperation signal
    state[4] = contrib_frac * 10.0

    # d5: Social Impact - group benefit
    state[5] = contribution * multiplier

    # d6: Virtue/Identity - prosocial behavior
    state[6] = contrib_frac * 8.0

    # d7: Legitimacy - institutional design
    state[7] = 5.0

    # d8: Epistemic - game structure knowledge
    state[8] = 7.0

    return state


@dataclass
class Prospect:
    """A prospect (lottery) with outcomes and probabilities."""

    outcomes: list[float]
    probabilities: list[float]


@dataclass
class KTProblem:
    """A Kahneman-Tversky choice problem between two prospects."""

    name: str
    prospect_a: Prospect
    prospect_b: Prospect
    empirical_pct_a: float  # % choosing prospect A


def prospect_to_state(
    prospect: Prospect,
    endowment: float = 0.0,
    scale: float = 1000.0,
) -> np.ndarray:
    """Convert a prospect to a 9D state vector.

    Parameters
    ----------
    prospect : Prospect
        The lottery to encode.
    endowment : float
        Reference point / endowment.
    scale : float
        Scaling factor for monetary values.

    Returns
    -------
    np.ndarray
        9D state vector.
    """
    ev = sum(o * p for o, p in zip(prospect.outcomes, prospect.probabilities))
    var = sum(
        p * (o - ev) ** 2 for o, p in zip(prospect.outcomes, prospect.probabilities)
    )
    std = np.sqrt(var)

    # Check if in loss domain
    is_loss = ev < endowment

    state = np.zeros(N_DIMS)

    # d0: Consequences - expected value
    state[0] = ev / scale * 10.0

    # d1: Rights - property rights framing
    state[1] = 5.0

    # d2: Fairness - equality of outcomes
    state[2] = max(0, 10.0 - std / scale * 5.0)

    # d3: Autonomy - choice freedom
    state[3] = 5.0

    # d4: Trust - certainty equivalent (inversely related to variance)
    state[4] = max(0, 8.0 - std / scale * 3.0)

    # d5: Social Impact
    # In loss domain, flip sign to capture loss aversion asymmetry
    if is_loss:
        state[5] = -abs(ev) / scale * 8.0
    else:
        state[5] = ev / scale * 5.0

    # d6: Virtue/Identity - risk attitude signal
    state[6] = 5.0

    # d7: Legitimacy - formal structure
    state[7] = 7.0

    # d8: Epistemic - probability information
    # In loss domain, flip to capture probability weighting asymmetry
    max_p = max(prospect.probabilities)
    if is_loss:
        state[8] = (1.0 - max_p) * 10.0
    else:
        state[8] = max_p * 10.0

    return state


# 17 problems from Kahneman & Tversky (1979)
KT_PROBLEMS = [
    KTProblem(
        "KT1",
        Prospect([2500, 2400, 0], [0.33, 0.66, 0.01]),
        Prospect([2400], [1.0]),
        empirical_pct_a=18.0,
    ),
    KTProblem(
        "KT2",
        Prospect([2500, 0], [0.33, 0.67]),
        Prospect([2400, 0], [0.34, 0.66]),
        empirical_pct_a=83.0,
    ),
    KTProblem(
        "KT3",
        Prospect([4000, 0], [0.80, 0.20]),
        Prospect([3000], [1.0]),
        empirical_pct_a=20.0,
    ),
    KTProblem(
        "KT4",
        Prospect([4000, 0], [0.20, 0.80]),
        Prospect([3000, 0], [0.25, 0.75]),
        empirical_pct_a=65.0,
    ),
    KTProblem(
        "KT5",
        Prospect([6000, 0], [0.45, 0.55]),
        Prospect([3000, 0], [0.90, 0.10]),
        empirical_pct_a=14.0,
    ),
    KTProblem(
        "KT6",
        Prospect([6000, 0], [0.001, 0.999]),
        Prospect([3000, 0], [0.002, 0.998]),
        empirical_pct_a=73.0,
    ),
    KTProblem(
        "KT7",
        Prospect([-4000, 0], [0.80, 0.20]),
        Prospect([-3000], [1.0]),
        empirical_pct_a=92.0,
    ),
    KTProblem(
        "KT8",
        Prospect([-4000, 0], [0.20, 0.80]),
        Prospect([-3000, 0], [0.25, 0.75]),
        empirical_pct_a=42.0,
    ),
    KTProblem(
        "KT9",
        Prospect([-6000, 0], [0.45, 0.55]),
        Prospect([-3000, 0], [0.90, 0.10]),
        empirical_pct_a=92.0,
    ),
    KTProblem(
        "KT10",
        Prospect([-6000, 0], [0.001, 0.999]),
        Prospect([-3000, 0], [0.002, 0.998]),
        empirical_pct_a=30.0,
    ),
    KTProblem(
        "KT11",
        Prospect([1000, 0], [0.50, 0.50]),
        Prospect([500], [1.0]),
        empirical_pct_a=16.0,
    ),
    KTProblem(
        "KT12",
        Prospect([-1000, 0], [0.50, 0.50]),
        Prospect([-500], [1.0]),
        empirical_pct_a=69.0,
    ),
    KTProblem(
        "KT13",
        Prospect([5000, 0], [0.001, 0.999]),
        Prospect([5], [1.0]),
        empirical_pct_a=72.0,
    ),
    KTProblem(
        "KT14",
        Prospect([-5000, 0], [0.001, 0.999]),
        Prospect([-5], [1.0]),
        empirical_pct_a=17.0,
    ),
    KTProblem(
        "KT3'",
        Prospect([-4000, 0], [0.80, 0.20]),
        Prospect([-3000], [1.0]),
        empirical_pct_a=92.0,
    ),
    KTProblem(
        "KT4'",
        Prospect([-4000, 0], [0.20, 0.80]),
        Prospect([-3000, 0], [0.25, 0.75]),
        empirical_pct_a=42.0,
    ),
    KTProblem(
        "KT_Insurance",
        Prospect([-5000, 0], [0.001, 0.999]),
        Prospect([-5], [1.0]),
        empirical_pct_a=17.0,
    ),
]


def make_evaluate_fn(
    sigma_inv: np.ndarray | None = None,
) -> Callable[[np.ndarray], tuple[float, dict[str, float]]]:
    """Create an evaluate_fn compatible with the structural fuzzing framework.

    The params array contains one "variance" value per dimension. These act
    as inverse weights in the Mahalanobis metric: larger variance = less weight.

    Parameters
    ----------
    sigma_inv : np.ndarray or None
        Base inverse covariance matrix. If None, uses identity.

    Returns
    -------
    callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    """
    # Import targets here to avoid circular imports
    from examples.geometric_economics.targets import build_targets

    targets = build_targets()

    def evaluate_fn(params: np.ndarray) -> tuple[float, dict[str, float]]:
        # Build sigma_inv from params: diag(1/params[i])
        # Large params[i] -> small weight -> dimension less important
        weights = np.where(params < 1e5, 1.0 / np.maximum(params, 1e-6), 0.0)
        sigma_inv_local = np.diag(weights)

        errors: dict[str, float] = {}
        for target in targets:
            predicted = target.predict_fn(sigma_inv_local)
            error = predicted - target.empirical
            errors[target.name] = error

        mae = sum(abs(v) for v in errors.values()) / len(errors)
        return mae, errors

    return evaluate_fn

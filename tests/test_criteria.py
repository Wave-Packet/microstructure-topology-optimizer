from __future__ import annotations

import numpy as np
import pytest

from criteria import CRITERIA


@pytest.fixture
def stiffness() -> np.ndarray:
    return np.array(
        [
            [3.2, 0.7, 0.1],
            [0.7, 2.8, -0.05],
            [0.1, -0.05, 1.4],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("criterion_name", tuple(CRITERIA))
def test_analytic_derivative_matches_finite_difference(
    criterion_name: str, stiffness: np.ndarray
) -> None:
    rng = np.random.default_rng(42)
    perturbation = rng.normal(size=(3, 3))
    perturbation = 0.5 * (perturbation + perturbation.T)
    criterion = CRITERIA[criterion_name]

    value, derivative = criterion.evaluate(stiffness, perturbation[None, ...])
    epsilon = 1.0e-7
    value_plus, _ = criterion.evaluate(
        stiffness + epsilon * perturbation, np.zeros((1, 3, 3))
    )
    value_minus, _ = criterion.evaluate(
        stiffness - epsilon * perturbation, np.zeros((1, 3, 3))
    )
    finite_difference = (value_plus - value_minus) / (2 * epsilon)

    assert np.isfinite(value)
    assert derivative[0] == pytest.approx(finite_difference, rel=2.0e-6, abs=2.0e-8)


def test_all_article_criteria_are_available() -> None:
    assert set(CRITERIA) == {
        "horizontal_rigidity_max",
        "bulk_modulus_max",
        "shear_modulus_max",
        "modified_poisson_ratio_min",
        "modified_poisson_ratio_max",
        "poisson_ratio_min",
        "poisson_ratio_max",
    }

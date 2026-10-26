"""Objective functions used by the topology optimizer.

The criteria are written in terms of the homogenized compliance matrix
``S = inv(C_hom)``. Each function returns the scalar objective and its
cell-wise derivative from the cell-wise derivative of ``C_hom``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class Criterion:
    """Description and implementation of a minimization criterion."""

    name: str
    label: str
    equation: int
    default_volume_penalty: float
    _value: Callable[[Array], float]
    _gradient: Callable[[Array, Array], Array]

    def evaluate(self, c_hom: Array, dc_hom: Array) -> tuple[float, Array]:
        """Return objective value and derivative for all mesh cells."""
        c_hom = np.asarray(c_hom, dtype=float)
        dc_hom = np.asarray(dc_hom, dtype=float)
        if c_hom.shape != (3, 3):
            raise ValueError(f"C_hom must have shape (3, 3), got {c_hom.shape}")
        if dc_hom.ndim != 3 or dc_hom.shape[1:] != (3, 3):
            raise ValueError(
                "DC_hom must have shape (number_of_cells, 3, 3), "
                f"got {dc_hom.shape}"
            )

        compliance = np.linalg.inv(c_hom)
        # d(C^-1) = -C^-1 (dC) C^-1
        d_compliance = -np.einsum(
            "ij,njk,kl->nil", compliance, dc_hom, compliance, optimize=True
        )
        return self._value(compliance), self._gradient(compliance, d_compliance)


def _entry(i: int, j: int, factor: float = 1.0):
    return (
        lambda s: float(factor * s[i, j]),
        lambda _s, ds: factor * ds[:, i, j],
    )


def _linear(terms: tuple[tuple[float, int, int], ...]):
    def value(s: Array) -> float:
        return float(sum(factor * s[i, j] for factor, i, j in terms))

    def gradient(_s: Array, ds: Array) -> Array:
        return sum(factor * ds[:, i, j] for factor, i, j in terms)

    return value, gradient


def _poisson(sign: float):
    """Criterion sign * (S01/S00 + S01/S11)."""

    def value(s: Array) -> float:
        return float(sign * (s[0, 1] / s[0, 0] + s[0, 1] / s[1, 1]))

    def gradient(s: Array, ds: Array) -> Array:
        if np.isclose(s[0, 0], 0.0) or np.isclose(s[1, 1], 0.0):
            raise ZeroDivisionError("Poisson-ratio criterion has a zero denominator")
        first = (ds[:, 0, 1] * s[0, 0] - s[0, 1] * ds[:, 0, 0]) / s[0, 0] ** 2
        second = (ds[:, 0, 1] * s[1, 1] - s[0, 1] * ds[:, 1, 1]) / s[1, 1] ** 2
        return sign * (first + second)

    return value, gradient


_horizontal = _entry(0, 0)
_bulk = _linear(((1.0, 0, 0), (2.0, 0, 1), (1.0, 1, 1)))
_shear = _entry(2, 2, 4.0)
_modified_min = _entry(0, 1, -1.0)
_modified_max = _entry(0, 1, 1.0)
_poisson_min = _poisson(-1.0)
_poisson_max = _poisson(1.0)


CRITERIA: dict[str, Criterion] = {
    "horizontal_rigidity_max": Criterion(
        "horizontal_rigidity_max", "Maximisation de la rigidite horizontale", 49, 10.0, *_horizontal
    ),
    "bulk_modulus_max": Criterion(
        "bulk_modulus_max", "Maximisation du module de compression", 50, 20.0, *_bulk
    ),
    "shear_modulus_max": Criterion(
        "shear_modulus_max", "Maximisation du module de cisaillement", 51, 50.0, *_shear
    ),
    "modified_poisson_ratio_min": Criterion(
        "modified_poisson_ratio_min", "Minimisation du coefficient de Poisson modifie", 52, 0.0, *_modified_min
    ),
    "modified_poisson_ratio_max": Criterion(
        "modified_poisson_ratio_max", "Maximisation du coefficient de Poisson modifie", 54, 0.0, *_modified_max
    ),
    "poisson_ratio_min": Criterion(
        "poisson_ratio_min", "Minimisation du coefficient de Poisson", 56, 0.0, *_poisson_min
    ),
    "poisson_ratio_max": Criterion(
        "poisson_ratio_max", "Maximisation du coefficient de Poisson", 58, 0.0, *_poisson_max
    ),
}


DEFAULT_CRITERION = "modified_poisson_ratio_max"


def get_criterion(name: str) -> Criterion:
    """Return a registered criterion by its command-line name."""
    try:
        return CRITERIA[name]
    except KeyError as exc:
        available = ", ".join(CRITERIA)
        raise ValueError(f"Unknown criterion {name!r}. Available: {available}") from exc


def criteria_table() -> str:
    """Return a human-readable list suitable for the CLI."""
    lines = []
    for criterion in CRITERIA.values():
        lines.append(
            f"{criterion.name:30s} equation ({criterion.equation}), "
            f"penalite volume par defaut = {criterion.default_volume_penalty:g}"
        )
    return "\n".join(lines)

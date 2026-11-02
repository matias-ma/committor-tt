"""Closed-form / ground-truth committors used only to validate the solver.

These are not part of the solving pipeline itself -- they exist so tests and
diagnostics can compare a trained :class:`~committor.ansatz.CommittorAnsatz`
against a known-correct answer on small, exactly solvable systems.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ScalarFn = Callable[[np.ndarray], np.ndarray]


def exact_committor_1d(
    V: ScalarFn,
    beta: float,
    left: float = -1.0,
    right: float = 1.0,
    ngrid: int = 50_001,
) -> ScalarFn:
    """Exact 1D committor on ``[left, right]`` by ODE quadrature.

    For 1D overdamped Langevin dynamics, the committor's derivative satisfies
    ``q'(x) ~ exp(beta * V(x))``, so

        q(x) = int_left^x exp(beta V(s)) ds / int_left^right exp(beta V(s)) ds.
    """
    xs = np.linspace(left, right, ngrid)
    f = np.exp(beta * V(xs))
    dx = xs[1] - xs[0]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[:-1] + f[1:]) * dx)])
    total = cum[-1]
    if total < 1e-300:
        raise ValueError(
            "exact_committor_1d: integral of exp(beta*V) is numerically zero; "
            "try reducing beta or widening [left, right]."
        )
    q_vals = cum / total

    def q_true(x: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), xs, q_vals)

    return q_true


def double_well_1d_committor(beta: float) -> ScalarFn:
    """Exact 1D committor for ``V(x) = (x^2 - 1)^2`` between the two wells."""
    return exact_committor_1d(lambda x: (x ** 2 - 1.0) ** 2, beta=beta, left=-1.0, right=1.0)

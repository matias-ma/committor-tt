"""The soft variational committor loss.

The committor solves the backward Kolmogorov equation with hard boundary
conditions q|_A = 0, q|_B = 1, which are incompatible with a low-rank
function-space parametrization (boundary measures aren't absolutely
continuous w.r.t. Lebesgue measure). Following Chen, Hoskins, Khoo & Lindsey
(arXiv:2106.12515), we instead minimize the soft variational functional

    L(q) = E_p[|grad q|^2] + rho * E_{p_A}[q^2] + rho * E_{p_B}[(q - 1)^2]

whose minimizer converges to the true committor as rho -> infinity. Each
expectation is estimated by plain Monte Carlo over samples from p / p_A / p_B
(no TT/Tucker operator algebra needed, since chebdGE provides none).
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from committor.ansatz import CommittorAnsatz

Array = jnp.ndarray


def soft_committor_loss(
    ansatz: CommittorAnsatz,
    x_p: Array,
    x_a: Array,
    x_b: Array,
    rho: float,
) -> Array:
    """Monte Carlo estimate of the soft committor variational loss.

    Parameters
    ----------
    ansatz : the current committor approximation.
    x_p    : (n_p, d) samples from the equilibrium density p.
    x_a    : (n_a, d) samples from the surrogate measure p_A.
    x_b    : (n_b, d) samples from the surrogate measure p_B.
    rho    : boundary-penalty weight.
    """
    _, grad_p = ansatz.value_and_grad(x_p)
    dirichlet = jnp.mean(jnp.sum(grad_p ** 2, axis=-1))

    q_a = ansatz(x_a)
    q_b = ansatz(x_b)
    penalty_a = jnp.mean(q_a ** 2)
    penalty_b = jnp.mean((q_b - 1.0) ** 2)

    return dirichlet + rho * (penalty_a + penalty_b)


def rho_schedule(final_rho: float, n_stages: int = 4) -> Sequence[float]:
    """A geometric rho-continuation schedule ending at ``final_rho``.

    Starting the boundary penalty small and increasing it avoids initially
    over-penalizing a still-poor committor guess, mirroring the paper's own
    continuation strategy.
    """
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    factors = [10.0 ** (-(n_stages - 1 - i)) for i in range(n_stages)]
    return [final_rho * f for f in factors]

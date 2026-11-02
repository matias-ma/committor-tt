"""Sampling helpers for the soft boundary-condition surrogate measures.

Equilibrium (``p``) sampling is delegated entirely to each
:class:`~committor.potentials.Potential`'s own ``sample`` method (e.g.
``ftde.sample_ginzburg_landau``, or the double well's exact per-dimension
sampler). The surrogate measures ``p_A``/``p_B`` for the two metastable
basins are built the same way for every system: the Laplace (Gaussian)
approximation of the equilibrium density around each basin minimizer,

    p_A(x) ~= N(x; basin_a, [-Hessian(log_density)(basin_a)]^-1),

i.e. the harmonic approximation of the well local to that minimum. This
needs no MCMC at all (it's an exact eigendecomposition of a d x d Hessian)
and, since it comes from the potential's own local curvature, automatically
gets every direction's natural spread right -- both the double well's
weakly-confined "spectator" coordinates and Ginzburg-Landau's coupled
fluctuations around a homogeneous state.

Two earlier designs were tried and rejected. A synthetic isotropic Gaussian
of a hand-picked width pins every coordinate near the basin point equally
hard, which is wrong whenever some coordinates are only weakly coupled to
the transition (confirmed empirically: training accuracy on the double well
*degraded* with more training at d=6, because the loss had no signal
correcting the ansatz's spurious dependence on those over-constrained
spectator directions). Conditioning true equilibrium samples on proximity to
the basin along the reaction coordinate fixes that, but is only tractable
when the basin is easy to reach under ordinary equilibrium sampling: for
Ginzburg-Landau, landing simultaneously near the homogeneous +-1 state in
every one of d coordinates is itself a rare event (that rarity is exactly
why this is a hard committor problem), so filtering equilibrium draws for it
is prohibitively slow already at d=20 (confirmed empirically: minutes per
basin, and it can still fail to collect enough samples in a fixed sampling
budget). The Laplace approximation sidesteps this: it needs no sampling of
the target density at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from committor.potentials import Potential

Array = jnp.ndarray


def laplace_basin_sample(potential: "Potential", seed: int, n: int, center: Array) -> Array:
    """``n`` samples from the Laplace approximation of ``potential`` at ``center``.

    Recomputes the Hessian on every call; use :func:`laplace_basin_sampler` to
    draw repeatedly from the same basin.
    """
    return laplace_basin_sampler(potential, center)(seed, n)


def laplace_basin_sampler(potential: "Potential", center: Array):
    """``sample(seed, n)`` drawing from the Laplace approximation of ``potential`` at ``center``.

    ``center`` should be a local maximum of ``potential.log_density`` (e.g. a
    basin minimizer of the energy). Symmetrizes the Hessian-derived
    covariance before the Cholesky factorization as a safeguard against
    floating-point asymmetry, and clips samples to ``potential.domain``: a
    weakly-confined direction's Laplace std can be a sizeable fraction of the
    domain half-width (e.g. the double well's spectator coordinates), so a
    Gaussian tail regularly reaches past the domain edge, and evaluating a
    high-degree Chebyshev TT/Tucker outside its domain blows up catastrophically
    (confirmed empirically: a handful of such outliers among thousands of
    samples was enough to send the training loss into the hundreds of
    millions and collapse the fit to q~=0 everywhere).
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    hessian = jax.hessian(potential.log_density)(center)
    neg_hessian = -0.5 * (hessian + hessian.T)
    # ``center`` comes from a numerical search (``find_local_maxima``) on a
    # possibly *estimated* (fitted-TT) log-density, so it is not guaranteed
    # to be a strict local maximum -- it can land on a box constraint, or on
    # a shallow/spurious bump in a noisy density fit. Either way
    # ``neg_hessian`` need not be positive definite there, and a Cholesky
    # factorization of an indefinite matrix silently returns NaN (confirmed:
    # this, not a domain/evaluation issue, was the source of the "final loss
    # = nan" committor training for the deconvolved-noisy density -- its
    # basins were noisy/off-mode). Flooring the eigenvalues keeps the
    # covariance always well-defined instead of propagating that NaN.
    eigvals, eigvecs = jnp.linalg.eigh(neg_hessian)
    floor = 1e-6 * jnp.maximum(jnp.max(eigvals), 1.0)
    eigvals = jnp.clip(eigvals, floor, None)
    covariance = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
    covariance = 0.5 * (covariance + covariance.T)
    scale_tril = jnp.linalg.cholesky(covariance)

    domain = jnp.asarray(potential.domain, dtype=jnp.float64)

    def sample(seed: int, n: int) -> Array:
        key = jax.random.PRNGKey(seed)
        z = jax.random.normal(key, (n, center.shape[0]), dtype=jnp.float64)
        samples = center[None, :] + z @ scale_tril.T
        return jnp.clip(samples, domain[:, 0], domain[:, 1])

    return sample

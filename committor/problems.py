"""Double-well problem setup and error metrics.

Provides convenience functions for the d-dimensional double-well benchmark
(paper Section 4.1, eq. 4.1-4.4) and general error-measurement utilities.

Public API
----------
double_well_potential           — V(x) = (x^2 - 1)^2
double_well_nd_measure_fns      — per-dimension measure functions for double-well
build_double_well_nd_problem    — assemble basis + per-dim matrices (Legendre)
build_double_well_nd_problem_weighted — same, density-weighted orthogonal basis
exact_committor_1d              — exact 1D committor by ODE quadrature
lift_to_d_dimensions            — wrap a 1D function so it accepts (n, d) input
relative_error_1d               — L^2(p) relative error (1D case)
relative_error_nd_mc            — L^2(p) relative error by MC sampling (nd case)
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from committor._types import Array, ScalarFn
from committor.basis import (
    TensorProductBasis,
    tensor_product_legendre_basis,
    double_well_density_weighted_basis,
)
from committor.assembly import PerDimMatrices, quadrature_matrices_nd


# ---------------------------------------------------------------------------
# The double-well potential
# ---------------------------------------------------------------------------

def double_well_potential(x: Array) -> Array:
    """1D double well  V(x) = (x^2 - 1)^2.

    The d-dimensional version used in the paper (eq. 4.1) is
        V(x) = (x_1^2 - 1)^2 + 0.3 * sum_{i>=2} x_i^2,
    whose committor equals the 1D committor by symmetry.  Only this 1D
    marginal is needed here; the coupling enters via the per-dimension
    weight functions in :func:`double_well_nd_measure_fns`.
    """
    x = np.asarray(x, dtype=float)
    return (x ** 2 - 1.0) ** 2


# ---------------------------------------------------------------------------
# Per-dimension measure functions
# ---------------------------------------------------------------------------

def double_well_nd_measure_fns(
    d:     int,
    beta:  float,
    sigma: float = 0.05,
    alpha: float = 0.3,
) -> Tuple[List[ScalarFn], List[ScalarFn], List[ScalarFn]]:
    """Per-dimension measure functions for the d-dimensional double-well.

    Factorises the measures of the paper's potential (eq. 4.1)
        V(x) = (x_1^2 - 1)^2 + alpha * sum_{k>=2} x_k^2
    into per-dimension callables suitable for :func:`~committor.assembly.quadrature_matrices_nd`.

    Parameters
    ----------
    d     : number of dimensions.
    beta  : inverse temperature 1/T.
    sigma : width of the soft-boundary Gaussians (paper: 0.05).
    alpha : coupling constant for the harmonic dimensions (paper: 0.3).

    Returns
    -------
    weight_fns, wA_fns, wB_fns : each a list of d callables.
        weight_fns[k](x) = p_k(x), the k-th marginal of the Boltzmann weight.
        wA_fns[k](x)     = pA_k(x), the k-th marginal of the soft-A measure.
        wB_fns[k](x)     = pB_k(x), the k-th marginal of the soft-B measure.
    """
    weight_fns: List[ScalarFn] = [
        lambda x, b=beta: np.exp(-b * double_well_potential(x)),
    ] + [
        lambda x, b=beta, a=alpha: np.exp(-b * a * np.asarray(x, dtype=float) ** 2)
        for _ in range(d - 1)
    ]

    pA_0:    ScalarFn = lambda x, s=sigma: np.exp(-0.5 * ((np.asarray(x) + 1.0) / s) ** 2)
    pB_0:    ScalarFn = lambda x, s=sigma: np.exp(-0.5 * ((np.asarray(x) - 1.0) / s) ** 2)
    ones_fn: ScalarFn = lambda x: np.ones_like(np.asarray(x, dtype=float))

    wA_fns: List[ScalarFn] = [pA_0] + [ones_fn] * (d - 1)
    wB_fns: List[ScalarFn] = [pB_0] + [ones_fn] * (d - 1)

    return weight_fns, wA_fns, wB_fns


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------

def build_double_well_nd_problem(
    d:        int,
    beta:     float,
    nbasis:   int,
    sigma:    float = 0.05,
    nquad:    int   = 600,
    interval: Tuple[float, float] = (-2.0, 2.0),
) -> Tuple[TensorProductBasis, List[PerDimMatrices]]:
    """Assemble all ingredients for the d-dim double-well committor problem.

    Convenience wrapper around :func:`tensor_product_legendre_basis`,
    :func:`double_well_nd_measure_fns`, and
    :func:`~committor.assembly.quadrature_matrices_nd` so callers do not need
    to repeat the boilerplate.

    Uses standard Legendre polynomials (orthonormal w.r.t. Lebesgue measure).
    For low temperatures use :func:`build_double_well_nd_problem_weighted`
    instead, which builds a density-weighted basis for improved accuracy.

    Parameters
    ----------
    d        : number of dimensions (paper uses 20).
    beta     : inverse temperature 1/T.
    nbasis   : number of basis functions per dimension.
    sigma    : Gaussian soft-boundary width (paper: 0.05).
    nquad    : Gauss-Legendre quadrature points per dimension.
    interval : (a, b) domain for every dimension (default: (-2, 2)).

    Returns
    -------
    basis   : TensorProductBasis — d Legendre bases, each of size ``nbasis``.
    per_dim : list of d PerDimMatrices — ready for the ALS / dense solvers.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}.")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")
    if nbasis < 1:
        raise ValueError(f"nbasis must be >= 1, got {nbasis}.")

    a, b = interval
    basis = tensor_product_legendre_basis([nbasis] * d, [(a, b)] * d)
    weight_fns, wA_fns, wB_fns = double_well_nd_measure_fns(
        d, beta=beta, sigma=sigma
    )
    per_dim = quadrature_matrices_nd(basis, weight_fns, wA_fns, wB_fns, nquad=nquad)
    return basis, per_dim


def build_double_well_nd_problem_weighted(
    d:      int,
    beta:   float,
    nbasis: int,
    sigma:  float = 0.05,
    nquad:  int   = 600,
    alpha:  float = 0.3,
    a:      float = -2.0,
    b:      float =  2.0,
) -> Tuple[TensorProductBasis, List[PerDimMatrices]]:
    """Double-well problem with a density-weighted orthogonal basis.

    Like :func:`build_double_well_nd_problem` but builds, per dimension,
    a basis orthonormal under the marginal equilibrium density p_k rather
    than the flat Lebesgue measure.  This typically reduces the number of
    basis functions needed for a given accuracy, especially at low temperature
    (paper Section 4.1 recommendation).

    The basis is constructed via
    :func:`~committor.basis.double_well_density_weighted_basis` which uses
    Gram-Schmidt / Cholesky orthogonalisation of the standard Legendre basis.

    Parameters
    ----------
    d      : number of dimensions.
    beta   : inverse temperature 1/T.
    nbasis : number of density-weighted basis functions per dimension.
    sigma  : Gaussian soft-boundary width (paper: 0.05).
    nquad  : quadrature points for both the weighted basis and the matrices.
    alpha  : harmonic coupling constant for dimensions k >= 1 (paper: 0.3).
    a, b   : domain interval for every dimension (default: (-2, 2)).

    Returns
    -------
    basis   : TensorProductBasis — density-weighted orthonormal bases.
    per_dim : list of d PerDimMatrices.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}.")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")

    # Build per-dimension density-weighted bases
    uvbases = tuple(
        double_well_density_weighted_basis(
            n=nbasis, k=k, beta=beta, a=a, b=b, alpha=alpha, nquad=nquad
        )
        for k in range(d)
    )
    basis = TensorProductBasis(bases=uvbases)

    weight_fns, wA_fns, wB_fns = double_well_nd_measure_fns(
        d, beta=beta, sigma=sigma, alpha=alpha
    )
    per_dim = quadrature_matrices_nd(basis, weight_fns, wA_fns, wB_fns, nquad=nquad)
    return basis, per_dim


# ---------------------------------------------------------------------------
# Exact 1D reference committor
# ---------------------------------------------------------------------------

def exact_committor_1d(
    V:     ScalarFn,
    beta:  float,
    left:  float = -1.0,
    right: float =  1.0,
    ngrid: int   = 50_001,
) -> ScalarFn:
    """Exact 1D committor for the interval [left, right].

    For overdamped Langevin in 1D, the gradient of q satisfies
        q'(x) ∝ exp(beta * V(x)),
    so the exact committor is
        q(x) = ∫_{left}^{x} exp(beta V(s)) ds
                / ∫_{left}^{right} exp(beta V(s)) ds.

    This ODE-based formula produces the ground-truth used in the paper
    (eq. 4.3) to evaluate the relative error E (eq. 4.4).

    Parameters
    ----------
    V     : potential function V(x) → array.
    beta  : inverse temperature 1/T.
    left  : left boundary of the interval (A boundary, q → 0).
    right : right boundary (B boundary, q → 1).
    ngrid : number of uniform grid points for quadrature.

    Returns
    -------
    callable q_true(x) → ndarray.
        Values outside [left, right] are clamped via ``np.interp`` to
        0 (left) and 1 (right), which is the correct asymptotic behaviour.
    """
    xs  = np.linspace(left, right, ngrid)
    f   = np.exp(beta * V(xs))
    dx  = xs[1] - xs[0]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[:-1] + f[1:]) * dx)])
    total = cum[-1]
    if total < 1e-300:
        raise ValueError(
            "exact_committor_1d: integral of exp(beta*V) is numerically zero. "
            "Try reducing beta or extending the interval."
        )
    q_vals = cum / total

    def q_true(x: Array) -> Array:
        return np.interp(np.asarray(x, dtype=float), xs, q_vals)

    return q_true


# ---------------------------------------------------------------------------
# Helper: lift a 1D committor to d dimensions
# ---------------------------------------------------------------------------

def lift_to_d_dimensions(q_1d: Callable) -> Callable:
    """Wrap a 1D committor so it accepts n-D sample arrays.

    Given a callable ``q_1d(x1)`` that evaluates the committor on 1D points,
    returns a new callable ``q_nd(X)`` where ``X`` has shape ``(n_samples, d)``
    and only the first column ``X[:, 0]`` is passed to ``q_1d``.

    This is valid for the double-well problem because by symmetry
        q_true(x) = q_true_1d(x_1).

    Parameters
    ----------
    q_1d : callable mapping array of shape (n,) → array of shape (n,).

    Returns
    -------
    callable mapping array of shape (n_samples, d) → array of shape (n_samples,).
    """
    def q_nd(X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return q_1d(X)
        return q_1d(X[:, 0])

    return q_nd


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def relative_error_1d(
    q_approx:  Callable,
    q_true:    Callable,
    V:         ScalarFn,
    beta:      float,
    left:      float = -1.0,
    right:     float =  1.0,
    nquad:     int   = 10_000,
) -> float:
    """Relative L^2(p) error for a 1D committor approximation.

    Estimates
        E = ||q - q_true||_{L2([left,right], p)}
              / ||q_true||_{L2([left,right], p)}

    by Gauss-Legendre quadrature on [left, right].

    Parameters
    ----------
    q_approx : callable (n,) → (n,) — numerical committor.
    q_true   : callable (n,) → (n,) — reference committor.
    V        : potential function.
    beta     : inverse temperature.
    left, right : integration interval.
    nquad    : number of quadrature points.

    Returns
    -------
    float — relative error E >= 0.
    """
    from numpy.polynomial.legendre import leggauss
    xs_std, ws_std = leggauss(nquad)
    xs = 0.5 * (right - left) * xs_std + 0.5 * (right + left)
    ws = 0.5 * (right - left) * ws_std
    p  = np.exp(-beta * V(xs))
    wp = ws * p

    q_n = q_approx(xs)
    q_r = q_true(xs)

    num = float(np.sum(wp * (q_n - q_r) ** 2))
    den = float(np.sum(wp * q_r ** 2))
    if den < 1e-30:
        warnings.warn(
            "relative_error_1d: denominator near zero; q_true is nearly 0 "
            "on the integration interval.",
            stacklevel=2,
        )
        return float("nan")
    return float(np.sqrt(num / den))


def relative_error_nd_mc(
    result:    object,
    q_true_1d: ScalarFn,
    beta:      float,
    alpha:     float = 0.3,
    n_samples: int   = 50_000,
    seed:      int   = 42,
    x1_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Approximate the relative L^2(p) error (paper eq. 4.4) by Monte Carlo.

    Estimates
        E = ||q_TT - q_true||_{L2(Omega \\ (A∪B), p)}
              / ||q_true||_{L2(Omega \\ (A∪B), p)}

    using self-normalised Monte Carlo.  Samples X ~ p are drawn from the
    separable double-well equilibrium density:

    * x_1 : rejection sampling on [a1, b1] with uniform proposal.
             The unnormalised density exp(-beta*(x_1^2-1)^2) has maximum 1
             (achieved at x_1 = ±1), so acceptance is straightforward.
    * x_k, k >= 1 : independent Gaussians N(0, 1/(2*beta*alpha)), clipped
                    to the basis interval.

    Parameters
    ----------
    result    : CommittorTTResult — TT approximation to benchmark.
    q_true_1d : callable — exact 1D committor; q_true_1d(x1) -> array.
    beta      : inverse temperature 1/T.
    alpha     : harmonic coupling for dims k >= 1 (paper: 0.3).
    n_samples : MC sample count.  Use >= 10_000 for reliable estimates.
    seed      : RNG seed for reproducibility.
    x1_bounds : (a, b) — integration range in x_1.

    Returns
    -------
    float — relative error E (dimensionless, >= 0).
    """
    d   = result.basis.d
    rng = np.random.default_rng(seed)
    a1, b1 = x1_bounds

    # Sample x_1 from p_1 ∝ exp(-beta*(x_1^2-1)^2) via rejection sampling
    x1_list: list = []
    batch = max(n_samples * 5, 20_000)
    while len(x1_list) < n_samples:
        cands = rng.uniform(a1, b1, size=batch)
        probs = np.exp(-beta * double_well_potential(cands))
        x1_list.extend(cands[rng.uniform(size=batch) < probs].tolist())
    x1 = np.array(x1_list[:n_samples])

    # Sample x_k, k >= 1, from N(0, σ_k^2) clipped to the basis interval
    if d > 1:
        std_k  = 1.0 / np.sqrt(2.0 * beta * alpha)
        a_k    = float(result.basis.bases[1].a)
        b_k    = float(result.basis.bases[1].b)
        X_rest = rng.normal(0.0, std_k, size=(n_samples, d - 1))
        X_rest = np.clip(X_rest, a_k, b_k)
        X = np.column_stack([x1, X_rest])
    else:
        X = x1.reshape(-1, 1)

    q_tt  = result.q(X)
    q_ref = q_true_1d(x1)

    num = float(np.mean((q_tt - q_ref) ** 2))
    den = float(np.mean(q_ref ** 2))
    if den < 1e-30:
        warnings.warn(
            "relative_error_nd_mc: denominator near zero; q_true is nearly 0 "
            "on the sampled points.  Check x1_bounds.",
            stacklevel=2,
        )
        return float("nan")
    return float(np.sqrt(num / den))

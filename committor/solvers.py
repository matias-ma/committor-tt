"""Solver entry points and result types for the committor problem.

Implements the Galerkin solvers described in paper Sections 3.1-3.5:

* 1D dense solver  (solve_committor_1d)
* n-D dense Kronecker solver  (solve_committor_nd_dense) — small d only
* n-D TT-ALS solver  (solve_committor_nd_tt) — scales to large d

Public API
----------
Committor1DResult        — result of solve_committor_1d
CommittorNDDenseResult   — result of solve_committor_nd_dense
CommittorTTResult        — result of solve_committor_nd_tt
solve_committor_1d       — 1D Galerkin solver (closed-form linear system)
solve_committor_nd_dense — dense n-D Galerkin solver
solve_committor_nd_tt    — TT-ALS n-D solver (scalable to high dimensions)
"""

from __future__ import annotations

import string
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from committor._types import Array, ScalarFn
from committor.basis import (
    UnivariateBasis, TensorProductBasis,
    shifted_orthonormal_legendre_basis,
)
from committor.tensor_train import TTTrain, MPOTrain, tt_evaluate
from committor.assembly import (
    PerDimMatrices,
    quadrature_matrices_1d,
    quadrature_matrices_nd,
    assemble_dense_nd,
    assemble_mpo_rank1,
    assemble_hb_tt,
)
from committor.als import als_solve


# ---------------------------------------------------------------------------
# Committor1DResult
# ---------------------------------------------------------------------------

@dataclass
class Committor1DResult:
    """Result of solve_committor_1d: Galerkin committor on an interval [a, b].

    Attributes
    ----------
    coeffs : ndarray, shape (nbasis,).  Galerkin coefficients.
    basis  : UnivariateBasis.
    a, b   : float.  Solver domain endpoints.
    beta   : float.  Inverse temperature.
    rho    : float.  Soft-boundary penalty weight.
    """
    coeffs: Array
    basis:  UnivariateBasis
    a:      float
    b:      float
    beta:   float
    rho:    float

    def __post_init__(self) -> None:
        if len(self.coeffs) != self.basis.n:
            raise ValueError(
                f"coeffs length {len(self.coeffs)} != basis.n {self.basis.n}"
            )

    def q(self, x: Array) -> Array:
        """Evaluate the committor approximation at x in [a, b]."""
        x = np.asarray(x, dtype=float)
        if np.any(x < self.a) or np.any(x > self.b):
            warnings.warn(
                f"Some x values lie outside the basis interval "
                f"[{self.a}, {self.b}].  The Legendre basis is unreliable "
                "outside this range.",
                stacklevel=2,
            )
        out = np.zeros_like(x)
        for c, phi in zip(self.coeffs, self.basis.fns):
            out = out + c * phi(x)
        return out

    def dq(self, x: Array) -> Array:
        """Evaluate the derivative q'(x) (needed for the reactive flow, eq. 4.7)."""
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for c, dphi in zip(self.coeffs, self.basis.dfns):
            out = out + c * dphi(x)
        return out


# ---------------------------------------------------------------------------
# CommittorNDDenseResult
# ---------------------------------------------------------------------------

def _eval_nd_dense(coeffs: Array, basis: TensorProductBasis, x: Array) -> Array:
    """Evaluate a dense nd committor at sample points via einsum contraction."""
    d = basis.d
    ns = basis.ns
    C = coeffs.reshape(ns)
    Phis = [basis.eval_marginal(k, x[:, k]) for k in range(d)]
    dim_letters = string.ascii_lowercase[:d]
    s_letter = 'z'
    phi_idx = [f'{letter}{s_letter}' for letter in dim_letters]
    einsum_str = f"{dim_letters},{','.join(phi_idx)}->{s_letter}"
    return np.einsum(einsum_str, C, *Phis)


@dataclass
class CommittorNDDenseResult:
    """Result of solve_committor_nd_dense: dense Kronecker-assembled nd committor.

    Attributes
    ----------
    coeffs : ndarray, shape (N,), N = prod(basis.ns).
    basis  : TensorProductBasis.
    rho    : float.  Soft-constraint penalty weight.
    """
    coeffs: Array
    basis:  TensorProductBasis
    rho:    float

    def __post_init__(self) -> None:
        expected = int(np.prod(self.basis.ns))
        if self.coeffs.shape != (expected,):
            raise ValueError(
                f"CommittorNDDenseResult: coeffs has shape {self.coeffs.shape}, "
                f"expected ({expected},) for basis.ns={self.basis.ns}."
            )

    def q(self, x: Array) -> Array:
        """Evaluate the committor at x, shape (n_samples, d)."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.basis.d:
            raise ValueError(
                f"CommittorNDDenseResult.q expects x of shape "
                f"(n_samples, d={self.basis.d}), got {x.shape}."
            )
        return _eval_nd_dense(self.coeffs, self.basis, x)


# ---------------------------------------------------------------------------
# CommittorTTResult
# ---------------------------------------------------------------------------

@dataclass
class CommittorTTResult:
    """Committor function represented in TT/MPS format (paper eq. 3.13 + 3.1).

    Satisfies the CommittorResult protocol via its ``.q`` method.

    Attributes
    ----------
    tt    : TTTrain — the d TT cores G_1,...,G_d for the coefficient tensor Q.
    basis : TensorProductBasis — must match tt.d and tt.ns.
    rho   : float — soft-boundary penalty weight used by the solver.
    """
    tt:    TTTrain
    basis: TensorProductBasis
    rho:   float

    def __post_init__(self) -> None:
        if self.tt.d != self.basis.d:
            raise ValueError(
                f"CommittorTTResult: tt.d={self.tt.d} != basis.d={self.basis.d}."
            )
        if self.tt.ns != self.basis.ns:
            raise ValueError(
                f"CommittorTTResult: tt.ns={self.tt.ns} != basis.ns={self.basis.ns}."
            )

    def q(self, x: Array) -> Array:
        """Evaluate the committor at sample points x, shape (n_samples, d)."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.basis.d:
            raise ValueError(
                f"CommittorTTResult.q expects x of shape "
                f"(n_samples, d={self.basis.d}), got {x.shape}."
            )
        return tt_evaluate(self.tt, self.basis, x)


# ---------------------------------------------------------------------------
# solve_committor_1d
# ---------------------------------------------------------------------------

def solve_committor_1d(
    V:       ScalarFn,
    beta:    float,
    a:       float,
    b:       float,
    nbasis:  int = 20,
    rho:     float = 400.0,
    pA:      Optional[ScalarFn] = None,
    pB:      Optional[ScalarFn] = None,
    sigma:   float = 0.05,
    nquad:   int = 500,
) -> Committor1DResult:
    """Solve the soft-committor variational problem in 1D (paper Section 2.2-3.1).

    Minimises (paper eq. 2.7):
        integral |q'|^2 p dx  +  rho * integral q^2 pA dx
                               +  rho * integral (q-1)^2 pB dx

    The Gateaux derivative gives (S + rho*MA + rho*MB) c = rho * bvec.

    Parameters
    ----------
    V      : potential energy function; p(x) ∝ exp(-beta * V(x)).
    beta   : inverse temperature.
    a, b   : solver domain endpoints.
    nbasis : number of Legendre basis functions.
    rho    : soft-boundary penalty weight.
    pA, pB : soft-boundary measures (defaults: Gaussians at x=-1 and x=+1).
    sigma  : width of the default Gaussian soft-boundary measures.
    nquad  : Gauss-Legendre quadrature points.

    Returns
    -------
    Committor1DResult with .q(x) and .dq(x) methods.
    """
    if a >= b:
        raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}.")
    if nbasis < 1:
        raise ValueError(f"nbasis must be >= 1, got {nbasis}.")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")
    if pA is None:
        pA = lambda x: np.exp(-0.5 * ((np.asarray(x) + 1.0) / sigma) ** 2)
    if pB is None:
        pB = lambda x: np.exp(-0.5 * ((np.asarray(x) - 1.0) / sigma) ** 2)

    weight = lambda x: np.exp(-beta * V(x))
    basis  = shifted_orthonormal_legendre_basis(nbasis, a, b)

    S, MA, MB, bvec = quadrature_matrices_1d(
        basis=basis, weight=weight, wA=pA, wB=pB, nquad=nquad,
    )

    lhs    = S + rho * (MA + MB)
    rhs    = rho * bvec
    coeffs = np.linalg.solve(lhs, rhs)

    return Committor1DResult(coeffs=coeffs, basis=basis, a=a, b=b, beta=beta, rho=rho)


# ---------------------------------------------------------------------------
# solve_committor_nd_dense
# ---------------------------------------------------------------------------

def solve_committor_nd_dense(
    basis:       TensorProductBasis,
    weight_fns:  List[ScalarFn],
    wA_fns:      List[ScalarFn],
    wB_fns:      List[ScalarFn],
    rho:         float = 400.0,
    nquad:       int = 500,
) -> CommittorNDDenseResult:
    """Solve the d-dimensional soft-committor problem via dense Kronecker assembly.

    Only feasible for small d (d <= 4 with ns_k ~ 10).  For large d,
    use solve_committor_nd_tt.

    Parameters
    ----------
    basis       : TensorProductBasis describing the trial space.
    weight_fns  : length-d list.  weight_fns[k](x) = p_k(x_k).
    wA_fns      : length-d list.  wA_fns[k](x) = pA_k(x_k).
    wB_fns      : length-d list.  wB_fns[k](x) = pB_k(x_k).
    rho         : soft-constraint penalty weight.
    nquad       : Gauss-Legendre quadrature points per dimension.

    Returns
    -------
    CommittorNDDenseResult
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")
    N = int(np.prod(basis.ns))
    if N > 50_000:
        raise ValueError(
            f"Dense assembly: total basis size N={N} (= prod({basis.ns})) is too "
            "large. Use solve_committor_nd_tt for high-dimensional problems."
        )
    per_dim = quadrature_matrices_nd(basis, weight_fns, wA_fns, wB_fns, nquad)
    lhs, rhs = assemble_dense_nd(per_dim, rho)
    coeffs = np.linalg.solve(lhs, rhs)
    return CommittorNDDenseResult(coeffs=coeffs, basis=basis, rho=rho)


# ---------------------------------------------------------------------------
# solve_committor_nd_tt
# ---------------------------------------------------------------------------

def solve_committor_nd_tt(
    per_dim:      List[PerDimMatrices],
    basis:        TensorProductBasis,
    rho:          float,
    tt_rank:      int               = 4,
    n_sweeps:     int               = 20,
    tol:          float             = 1e-8,
    seed:         int               = 0,
    verbose:      bool              = False,
    rho_schedule: Optional[List[float]] = None,
) -> CommittorTTResult:
    """Solve the d-dimensional soft-committor problem via TT-ALS.

    Assumes a rank-1 product-measure for p, pA, pB (encoded in per_dim).
    For the Ginzburg-Landau problem with a non-product density, use
    ginzburg_landau.solve_gl_committor instead.

    Parameters
    ----------
    per_dim  : list of d PerDimMatrices, output of quadrature_matrices_nd.
    basis    : TensorProductBasis — must match len(per_dim) and ns.
    rho      : soft-boundary penalty weight.
    tt_rank  : fixed internal bond dimension for the solution TT.
    n_sweeps : max ALS sweeps per rho stage.
    tol      : relative convergence tolerance.
    seed     : RNG seed.
    verbose  : if True, print sweep-level objective values.
    rho_schedule : optional list of rho values for a continuation schedule.

    Returns
    -------
    CommittorTTResult
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")
    if tt_rank < 1:
        raise ValueError(f"tt_rank must be >= 1, got {tt_rank}.")
    d = len(per_dim)
    if d != basis.d:
        raise ValueError(f"len(per_dim)={d} != basis.d={basis.d}.")

    # rho continuation schedule
    if rho_schedule is None:
        if rho >= 10.0:
            rho_schedule = [rho * f for f in (1e-3, 1e-2, 1e-1, 1.0)]
        else:
            rho_schedule = [rho]
    else:
        if abs(rho_schedule[-1] - rho) > 1e-10 * abs(rho):
            warnings.warn(
                f"rho_schedule[-1]={rho_schedule[-1]:.4g} != rho={rho:.4g}; "
                "the final ALS solve will use rho_schedule[-1], not rho.",
                stacklevel=2,
            )

    warnings.warn(
        "solve_committor_nd_tt: using Legendre basis (orthogonal w.r.t. "
        "Lebesgue measure). The paper recommends density-weighted orthogonal "
        "polynomials w.r.t. p_k for better accuracy, especially at low T. "
        "Increase nbasis if the error is larger than expected.",
        UserWarning,
        stacklevel=2,
    )

    hb_tt = assemble_hb_tt(per_dim)

    rng = np.random.default_rng(seed)
    ns  = basis.ns

    if d == 1:
        init_cores = [rng.standard_normal((1, ns[0], 1)) * 0.1]
    else:
        init_cores = []
        for k in range(d):
            r_left  = 1 if k == 0     else tt_rank
            r_right = 1 if k == d - 1 else tt_rank
            init_cores.append(rng.standard_normal((r_left, ns[k], r_right)) * 0.1)

    tt_current = TTTrain(cores=init_cores)

    for stage_idx, rho_i in enumerate(rho_schedule):
        mpo_i = assemble_mpo_rank1(per_dim, rho_i)
        if verbose:
            print(f"  [rho stage {stage_idx+1}/{len(rho_schedule)}]"
                  f"  rho={rho_i:.4g}  ({n_sweeps} sweeps max)")
        tt_current, history = als_solve(
            tt_current, mpo_i, hb_tt, rho_i,
            n_sweeps=n_sweeps, tol=tol, verbose=verbose,
        )
        if verbose:
            print(f"    -> converged in {len(history)} sweep(s), "
                  f"J_final={history[-1]:.6g}")

    return CommittorTTResult(tt=tt_current, basis=basis, rho=rho)

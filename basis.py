"""Univariate and tensor-product basis functions for the committor solver.

Implements the basis infrastructure described in paper Section 3.1 and used
throughout the Galerkin discretisation of the variational problem (eq. 3.1).

Public API
----------
UnivariateBasis                      — dataclass holding n basis functions
TensorProductBasis                   — d-dimensional tensor product of UnivariateBases
shifted_orthonormal_legendre_basis   — Legendre basis on [a, b]
fourier_basis                        — Fourier basis on [-gamma, gamma]
tensor_product_legendre_basis        — convenience factory for TensorProductBasis
density_weighted_orthogonal_basis    — basis orthonormal under a weighted L^2 inner product
double_well_density_weighted_basis   — density-weighted basis for the double-well problem
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss

from committor._types import Array, ScalarFn


# ---------------------------------------------------------------------------
# UnivariateBasis
# ---------------------------------------------------------------------------

@dataclass
class UnivariateBasis:
    """A finite set of 1D basis functions and their derivatives on [a, b].

    This is the building block for a tensor-product multi-dimensional basis
    (paper Section 3.1).  In d dimensions, one UnivariateBasis per dimension k
    gives basis functions {phi_j^(k)}, and the full d-dimensional basis is the
    tensor product phi^(1) x ... x phi^(d).

    Attributes
    ----------
    fns  : list of callables, length n.  fns[k](x) evaluates the k-th basis fn.
    dfns : list of callables, length n.  dfns[k](x) evaluates its derivative.
    a, b : float.  Interval of orthogonality.
    n    : int.  Number of basis functions.  Equals len(fns).
    """
    fns:  list
    dfns: list
    a:    float
    b:    float
    n:    int

    def __post_init__(self) -> None:
        if len(self.fns) != self.n or len(self.dfns) != self.n:
            raise ValueError(
                f"UnivariateBasis: n={self.n} but len(fns)={len(self.fns)}, "
                f"len(dfns)={len(self.dfns)}"
            )


def shifted_orthonormal_legendre_basis(
    n: int, a: float, b: float
) -> UnivariateBasis:
    """Return a UnivariateBasis of orthonormal Legendre functions on [a, b].

    phi_k(x) = sqrt((2k+1)/(b-a)) * P_k(s),  s = (2x - (a+b)) / (b-a)

    Orthonormality: integral_a^b phi_i(x) phi_j(x) dx = delta_{ij}.
    """
    if a >= b:
        raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    fns: list = []
    dfns: list = []

    for k in range(n):
        P  = Legendre.basis(k)
        dP = P.deriv()
        norm = np.sqrt((2 * k + 1) / (b - a))

        def phi(x, P=P, norm=norm):
            return norm * P(scale * np.asarray(x) + shift)

        def dphi(x, dP=dP, norm=norm):
            return norm * dP(scale * np.asarray(x) + shift) * scale

        fns.append(phi)
        dfns.append(dphi)

    return UnivariateBasis(fns=fns, dfns=dfns, a=a, b=b, n=n)


def fourier_basis(n_fns: int, gamma: float) -> UnivariateBasis:
    """Return a UnivariateBasis of Fourier functions on [-gamma, gamma].

    For n_fns = 2K + 1 (must be a positive odd integer), the basis is::

        phi_0(x)      = 1
        phi_{2k-1}(x) = cos(k * pi * x / gamma)    k = 1, ..., K
        phi_{2k}(x)   = sin(k * pi * x / gamma)    k = 1, ..., K

    These are the Fourier basis functions used in paper Section 4.2 for the
    Ginzburg-Landau experiment (5 functions per dimension, K=2, gamma=2.6).

    Parameters
    ----------
    n_fns : int
        Number of basis functions.  Must be a positive odd integer.
    gamma : float
        Half-width of the domain; the basis is defined on [-gamma, gamma].
    """
    if n_fns < 1 or n_fns % 2 == 0:
        raise ValueError(
            f"n_fns must be a positive odd integer; got n_fns={n_fns}."
        )
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive; got gamma={gamma}.")

    K = (n_fns - 1) // 2   # number of cos/sin pairs

    fns: list = []
    dfns: list = []

    # phi_0 = 1,  dphi_0 = 0
    fns.append(lambda x: np.ones_like(np.asarray(x, dtype=float)))
    dfns.append(lambda x: np.zeros_like(np.asarray(x, dtype=float)))

    for k in range(1, K + 1):
        freq = k * np.pi / gamma

        def phi_cos(x, freq=freq):
            return np.cos(freq * np.asarray(x, dtype=float))

        def dphi_cos(x, freq=freq):
            return -freq * np.sin(freq * np.asarray(x, dtype=float))

        def phi_sin(x, freq=freq):
            return np.sin(freq * np.asarray(x, dtype=float))

        def dphi_sin(x, freq=freq):
            return freq * np.cos(freq * np.asarray(x, dtype=float))

        fns.append(phi_cos)
        dfns.append(dphi_cos)
        fns.append(phi_sin)
        dfns.append(dphi_sin)

    return UnivariateBasis(fns=fns, dfns=dfns, a=-gamma, b=gamma, n=n_fns)


# ---------------------------------------------------------------------------
# TensorProductBasis
# ---------------------------------------------------------------------------

@dataclass
class TensorProductBasis:
    """A d-dimensional basis built from a tensor product of UnivariateBases.

    Represents the family of functions

        Phi_{i_1,...,i_d}(x) = phi^(1)_{i_1}(x_1) * ... * phi^(d)_{i_d}(x_d)

    following paper Section 3.1.

    Attributes
    ----------
    bases : tuple of UnivariateBasis, one per dimension.
    """
    bases: tuple  # tuple[UnivariateBasis, ...]

    def __post_init__(self) -> None:
        if len(self.bases) == 0:
            raise ValueError("TensorProductBasis requires at least one UnivariateBasis.")
        for i, b in enumerate(self.bases):
            if not isinstance(b, UnivariateBasis):
                raise TypeError(
                    f"bases[{i}] must be a UnivariateBasis, got {type(b).__name__}."
                )

    @property
    def d(self) -> int:
        """Number of dimensions."""
        return len(self.bases)

    @property
    def ns(self) -> Tuple[int, ...]:
        """Number of basis functions per dimension."""
        return tuple(b.n for b in self.bases)

    @property
    def lower(self) -> Tuple[float, ...]:
        """Left endpoints of the per-dimension intervals."""
        return tuple(b.a for b in self.bases)

    @property
    def upper(self) -> Tuple[float, ...]:
        """Right endpoints of the per-dimension intervals."""
        return tuple(b.b for b in self.bases)

    def eval_marginal(self, k: int, x: Array) -> Array:
        """Evaluate all n_k basis functions for dimension k at 1D points x.

        Parameters
        ----------
        k : int
            Dimension index, 0-based.
        x : array-like, shape (npts,)

        Returns
        -------
        Phi_k : ndarray, shape (ns[k], npts)
        """
        if not (0 <= k < self.d):
            raise ValueError(f"k={k} out of range [0, {self.d}).")
        b = self.bases[k]
        return np.vstack([f(x) for f in b.fns])

    def deval_marginal(self, k: int, x: Array) -> Array:
        """Evaluate derivatives of all n_k basis functions for dimension k.

        Parameters
        ----------
        k : int
            Dimension index, 0-based.
        x : array-like, shape (npts,)

        Returns
        -------
        DPhi_k : ndarray, shape (ns[k], npts)
        """
        if not (0 <= k < self.d):
            raise ValueError(f"k={k} out of range [0, {self.d}).")
        b = self.bases[k]
        return np.vstack([f(x) for f in b.dfns])


def tensor_product_legendre_basis(
    ns: Sequence[int],
    intervals: Sequence[Tuple[float, float]],
) -> TensorProductBasis:
    """Create a d-dimensional tensor-product orthonormal Legendre basis.

    Parameters
    ----------
    ns        : sequence of int, length d.  Number of basis functions per dim.
    intervals : sequence of (a, b) float pairs, length d.
    """
    if len(ns) != len(intervals):
        raise ValueError(
            f"ns (length {len(ns)}) and intervals (length {len(intervals)}) "
            "must have the same length."
        )
    bases = tuple(
        shifted_orthonormal_legendre_basis(int(n), float(a), float(b))
        for n, (a, b) in zip(ns, intervals)
    )
    return TensorProductBasis(bases=bases)


# ---------------------------------------------------------------------------
# Density-weighted orthonormal basis
# ---------------------------------------------------------------------------

def density_weighted_orthogonal_basis(
    n: int,
    weight_fn: ScalarFn,
    a: float,
    b: float,
    nquad: int = 1000,
) -> UnivariateBasis:
    """Return a UnivariateBasis of n functions orthonormal under a weighted L2 inner product.

    Constructs n functions {psi_k} satisfying

        <psi_i, psi_j>_w  :=  integral_a^b psi_i(x) psi_j(x) weight_fn(x) dx  =  delta_{ij}

    via Gram-Schmidt / Cholesky orthogonalisation of a plain Legendre basis:

    1. Build a Legendre basis {phi_k}_0^{n-1} orthonormal w.r.t. Lebesgue.
    2. Compute the weighted Gram matrix G[i,j] = <phi_i, phi_j>_w.
    3. Cholesky-factor G = L L^T.
    4. The new basis is psi = L^{-T} phi (i.e. L^{-1} acts on the coefficient
       vector), so that <psi_i, psi_j>_w = delta_{ij}.

    Parameters
    ----------
    n         : number of basis functions.
    weight_fn : callable, weight_fn(x) = w(x) >= 0.
    a, b      : interval endpoints.
    nquad     : Gauss-Legendre quadrature points for the weighted Gram matrix.
    """
    if a >= b:
        raise ValueError(f"Interval must satisfy a < b, got a={a}, b={b}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if nquad < n:
        raise ValueError(f"nquad={nquad} must be >= n={n}.")

    # Step 1: plain Legendre basis on [a, b]
    leg_basis = shifted_orthonormal_legendre_basis(n, a, b)

    # Step 2: weighted Gram matrix
    xs_std, ws_std = leggauss(nquad)
    x = 0.5 * (b - a) * xs_std + 0.5 * (b + a)
    w = 0.5 * (b - a) * ws_std
    wq = w * weight_fn(x)                                     # (nquad,)

    Phi = np.vstack([f(x) for f in leg_basis.fns])            # (n, nquad)
    G   = (Phi * wq[np.newaxis, :]) @ Phi.T                   # (n, n)

    # Step 3: Cholesky factorisation  G = L @ L.T
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "density_weighted_orthogonal_basis: Cholesky factorisation of the "
            f"weighted Gram matrix failed.  The weight function may be "
            "non-positive or numerically near-zero on [a, b].  "
            f"Original error: {exc}"
        ) from exc

    # Step 4: L^{-1}
    L_inv = np.linalg.inv(L)                                   # (n, n)

    # Build new basis functions as closures
    fns: list = []
    dfns: list = []

    for k in range(n):
        coeffs_k = L_inv[k, :]   # row k of L_inv

        def psi(x, c=coeffs_k, leg=leg_basis):
            xarr = np.asarray(x, dtype=float)
            out  = np.zeros_like(xarr)
            for j, phi_j in enumerate(leg.fns):
                out = out + c[j] * phi_j(xarr)
            return out

        def dpsi(x, c=coeffs_k, leg=leg_basis):
            xarr = np.asarray(x, dtype=float)
            out  = np.zeros_like(xarr)
            for j, dphi_j in enumerate(leg.dfns):
                out = out + c[j] * dphi_j(xarr)
            return out

        fns.append(psi)
        dfns.append(dpsi)

    return UnivariateBasis(fns=fns, dfns=dfns, a=a, b=b, n=n)


def double_well_density_weighted_basis(
    n:     int,
    k:     int,
    beta:  float,
    a:     float = -2.0,
    b:     float =  2.0,
    alpha: float = 0.3,
    nquad: int   = 1000,
) -> UnivariateBasis:
    """Density-weighted orthonormal basis for dimension k of the double-well problem.

    For the d-dimensional double-well potential (paper eq. 4.1)

        V(x) = (x_1^2 - 1)^2 + alpha * sum_{i>=2} x_i^2

    the equilibrium density factors as:

        p_0(x) = exp(-beta * (x^2 - 1)^2)          (dimension k=0)
        p_k(x) = exp(-beta * alpha * x^2)           (dimension k >= 1)

    Parameters
    ----------
    n     : number of basis functions.
    k     : dimension index (0-based).
    beta  : inverse temperature.
    a, b  : interval (default -2.0 to 2.0).
    alpha : quadratic coupling for dimensions k >= 1 (paper: 0.3).
    nquad : quadrature points.
    """
    if k == 0:
        weight_fn: ScalarFn = lambda x, b_=beta: np.exp(
            -b_ * (np.asarray(x, dtype=float) ** 2 - 1.0) ** 2
        )
    else:
        weight_fn = lambda x, b_=beta, a_=alpha: np.exp(
            -b_ * a_ * np.asarray(x, dtype=float) ** 2
        )
    return density_weighted_orthogonal_basis(n=n, weight_fn=weight_fn, a=a, b=b, nquad=nquad)
"""
Notebook-friendly front end for committor_tt.

This module provides a small convenience API for interactive use from a
Jupyter notebook.

What it does well
------------------
- 1D problems with an arbitrary potential V(x)
- nD problems whose equilibrium density and soft-boundary sets factor as
  products of 1D factors (the current TT / dense implementations)
- a single `find_committor(...)` entry point that chooses a solver

What it does not do
-------------------
The current TT code in this repository assumes a rank-1 / product structure
for p, pA, and pB in the scalable nD solver. A genuinely non-product density
or a fully general metastable set needs a new assembly path, not just a wrapper.

Typical notebook usage
----------------------
    from committor.notebook_api import find_committor, make_gaussian_product_states

    # 1D
    result = find_committor(
        V=lambda x: (x**2 - 1.0)**2,
        beta=5.0,
        a=-2.0,
        b=2.0,
        nbasis=30,
        rho=400.0,
    )

    # nD product-measure example
    weight_fns = [lambda x: np.exp(-5.0 * (x**2 - 1.0)**2)] * 4
    wA_fns, wB_fns = make_gaussian_product_states(
        a_center=[-1.0, 0.0, 0.0, 0.0],
        b_center=[+1.0, 0.0, 0.0, 0.0],
        sigma=0.05,
    )
    result = find_committor(
        basis_kind="legendre",
        ns=[20, 20, 20, 20],
        intervals=[(-2.0, 2.0)] * 4,
        weight_fns=weight_fns,
        wA_fns=wA_fns,
        wB_fns=wB_fns,
        rho=400.0,
        method="tt",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from committor._types import ScalarFn
from committor.basis import (
    TensorProductBasis,
    fourier_basis,
    tensor_product_legendre_basis,
)
from committor.assembly import assemble_dense_nd, quadrature_matrices_nd
from committor.solvers import (
    Committor1DResult,
    CommittorNDDenseResult,
    CommittorTTResult,
    solve_committor_1d,
    solve_committor_nd_tt,
)

SolverMethod = Literal["1d", "dense", "tt"]
BasisKind = Literal["legendre", "fourier"]


@dataclass(frozen=True)
class ProductStateSpec:
    """
    Separable soft-boundary state specification.

    Each dimension gets a 1D Gaussian factor centered at the corresponding
    coordinate of `center`.
    """

    center: Sequence[float]
    sigma: float = 0.05
    amplitude: float = 1.0

    def make_fns(self) -> List[ScalarFn]:
        c = np.asarray(self.center, dtype=float)
        if c.ndim != 1:
            raise ValueError("center must be a 1D sequence.")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}.")
        amp = float(self.amplitude)
        sig = float(self.sigma)

        out: List[ScalarFn] = []
        for ck in c:
            def fn(x, ck=ck, amp=amp, sig=sig):
                x = np.asarray(x, dtype=float)
                return amp * np.exp(-0.5 * ((x - ck) / sig) ** 2)
            out.append(fn)
        return out


def make_gaussian_product_states(
    a_center: Sequence[float],
    b_center: Sequence[float],
    sigma: float = 0.05,
    amplitude: float = 1.0,
) -> Tuple[List[ScalarFn], List[ScalarFn]]:
    """
    Return separable 1D Gaussian soft-boundary factors for A and B.
    """
    return (
        ProductStateSpec(a_center, sigma=sigma, amplitude=amplitude).make_fns(),
        ProductStateSpec(b_center, sigma=sigma, amplitude=amplitude).make_fns(),
    )


def make_basis(
    *,
    basis_kind: BasisKind,
    ns: Sequence[int],
    intervals: Sequence[Tuple[float, float]],
) -> TensorProductBasis:
    """
    Build a tensor-product basis for notebook use.
    """
    if len(ns) != len(intervals):
        raise ValueError("ns and intervals must have the same length.")

    if basis_kind == "legendre":
        return tensor_product_legendre_basis(ns, intervals)

    if basis_kind == "fourier":
        bases = []
        for n, (a, b) in zip(ns, intervals):
            if abs(a + b) > 1e-12:
                raise ValueError(
                    "Fourier basis is only implemented on symmetric intervals [-gamma, gamma]."
                )
            gamma = max(abs(a), abs(b))
            bases.append(fourier_basis(int(n), gamma))
        return TensorProductBasis(tuple(bases))

    raise ValueError(f"Unknown basis_kind={basis_kind!r}.")


def find_committor(
    *,
    V: Optional[ScalarFn] = None,
    beta: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    nbasis: int = 20,
    rho: float = 400.0,
    pA: Optional[ScalarFn] = None,
    pB: Optional[ScalarFn] = None,
    sigma: float = 0.05,
    nquad: int = 500,
    basis_kind: BasisKind = "legendre",
    ns: Optional[Sequence[int]] = None,
    intervals: Optional[Sequence[Tuple[float, float]]] = None,
    weight_fns: Optional[Sequence[ScalarFn]] = None,
    wA_fns: Optional[Sequence[ScalarFn]] = None,
    wB_fns: Optional[Sequence[ScalarFn]] = None,
    method: SolverMethod = "tt",
    tt_rank: int = 4,
    n_sweeps: int = 20,
    tol: float = 1e-8,
    seed: int = 0,
    verbose: bool = False,
    rho_schedule: Optional[List[float]] = None,
) -> Union[Committor1DResult, CommittorNDDenseResult, CommittorTTResult]:
    """
    Solve a committor problem from a notebook-friendly interface.

    Dispatch rules
    --------------
    - If V, beta, a, b are provided, the 1D solver is used.
    - Otherwise an nD product-measure problem is assembled from the supplied
      basis and per-dimension measure functions.
    """

    # 1D front door.
    if V is not None or beta is not None or a is not None or b is not None:
        if V is None or beta is None or a is None or b is None:
            raise ValueError("For the 1D solver, provide all of V, beta, a, and b.")
        return solve_committor_1d(
            V=V,
            beta=float(beta),
            a=float(a),
            b=float(b),
            nbasis=int(nbasis),
            rho=float(rho),
            pA=pA,
            pB=pB,
            sigma=float(sigma),
            nquad=int(nquad),
        )

    # nD front door.
    if ns is None or intervals is None or weight_fns is None or wA_fns is None or wB_fns is None:
        raise ValueError(
            "For nD problems, provide ns, intervals, weight_fns, wA_fns, and wB_fns."
        )

    basis = make_basis(basis_kind=basis_kind, ns=ns, intervals=intervals)
    per_dim = quadrature_matrices_nd(
        basis=basis,
        weight_fns=list(weight_fns),
        wA_fns=list(wA_fns),
        wB_fns=list(wB_fns),
        nquad=int(nquad),
    )

    if method == "dense":
        lhs, rhs = assemble_dense_nd(per_dim, float(rho))
        coeffs = np.linalg.solve(lhs, rhs)
        return CommittorNDDenseResult(coeffs=coeffs, basis=basis, rho=float(rho))

    if method == "tt":
        return solve_committor_nd_tt(
            per_dim=per_dim,
            basis=basis,
            rho=float(rho),
            tt_rank=int(tt_rank),
            n_sweeps=int(n_sweeps),
            tol=float(tol),
            seed=int(seed),
            verbose=bool(verbose),
            rho_schedule=rho_schedule,
        )

    raise ValueError(f"Unknown method={method!r}.")


__all__ = [
    "ProductStateSpec",
    "make_gaussian_product_states",
    "make_basis",
    "find_committor",
]

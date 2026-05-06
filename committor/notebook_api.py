"""
Notebook-friendly front end for committor_tt.

This module provides a small convenience API for interactive use from a
Jupyter notebook.


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
from committor.assembly_tt_density import (
    TTDensitySpec,
    assemble_mpo_tt_density,
    assemble_hb_tt_density,
    product_density_to_tt_spec,
)
from committor.solvers import (
    Committor1DResult,
    CommittorNDDenseResult,
    CommittorTTResult,
    solve_committor_1d,
    solve_committor_nd_tt,
)

from committor.tensor_train import TTTrain

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


def create_mixture_tt_density(
    product_weight_fns_list: Sequence[Sequence[ScalarFn]],
    committor_basis: TensorProductBasis,
    component_weights: Optional[Sequence[float]] = None,
    nquad: int = 500,
) -> TTDensitySpec:
    """Create a rank-J TT density from J product densities (optionally weighted).
    
    The resulting TT represents a mixture:
        ρ(x) = Σⱼ wⱼ * ρⱼ(x)
    
    where each ρⱼ(x) = Πₖ ρⱼ,ₖ(xₖ) is a product density.
    
    This is useful for creating non-separable densities: while each component
    is separable, their sum is rank-J in TT format and NOT a product.
    
    Parameters
    ----------
    product_weight_fns_list : sequence of J sequences
        Each element is a length-d sequence of 1D callables representing one
        product density component. For example:
            [
                [lambda x: np.exp(-5*(x-1)**2)] * 4,      # component 1
                [lambda x: np.exp(-5*(x+1)**2)] * 4,      # component 2
            ]
    committor_basis : TensorProductBasis
        Shared basis for all components and the committor.
    component_weights : length-J sequence, optional
        Weights for each component (will be normalized to sum to 1).
        Default: equal weight 1/J for each component.
    nquad : int
        Gauss-Legendre quadrature points per dimension.
    
    Returns
    -------
    TTDensitySpec with TT rank J.
    
    Example
    -------
    Create a rank-2 mixture: 70% Gaussian at x=-1, 30% Gaussian at x=+1:
    
        basis = tensor_product_legendre_basis([20]*4, [(-2, 2)]*4)
        
        component_1 = [lambda x: np.exp(-5*(x-1)**2)] * 4
        component_2 = [lambda x: np.exp(-5*(x+1)**2)] * 4
        
        density_spec = create_mixture_tt_density(
            [component_1, component_2],
            basis,
            component_weights=[0.7, 0.3],
        )
    """
    J = len(product_weight_fns_list)
    d = committor_basis.d
    
    if J == 0:
        raise ValueError("Must provide at least one product density component.")
    
    if component_weights is None:
        component_weights = [1.0 / J] * J
    else:
        component_weights = np.asarray(component_weights, dtype=float)
        if len(component_weights) != J:
            raise ValueError(
                f"component_weights has length {len(component_weights)} "
                f"but {J} components provided."
            )
        # Normalize
        component_weights = component_weights / np.sum(component_weights)
    
    # For each site, compute projection vectors for each component
    # proj_vectors[k][j] = projection of j-th component's factor at site k
    proj_vectors = []
    
    for k in range(d):
        from numpy.polynomial.legendre import leggauss
        
        uvb = committor_basis.bases[k]
        a_k, b_k = uvb.a, uvb.b
        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_k - a_k) * xs_std + 0.5 * (b_k + a_k)
        ws = 0.5 * (b_k - a_k) * ws_std
        
        # Evaluate basis functions at quadrature nodes: (n_k, nquad)
        Phi = np.vstack([f(xs) for f in uvb.fns])
        
        proj_at_k = []
        for j in range(J):
            # j-th component's k-th factor evaluated at quadrature nodes
            weight_factor = product_weight_fns_list[j][k](xs)
            # Project onto basis: weighted inner product
            coeff = Phi @ (ws * component_weights[j] * weight_factor)
            proj_at_k.append(coeff)
        proj_vectors.append(proj_at_k)
    
    # Build TT cores in the standard sum representation
    # For rank J: diagonal pattern with right boundary reduction
    cores = []
    for k in range(d):
        n_k = committor_basis.bases[k].n
        
        if k == 0:
            # Left boundary: (1, n_k, J)
            # Each component enters on a separate bond index
            core = np.zeros((1, n_k, J))
            for j in range(J):
                core[0, :, j] = proj_vectors[k][j]
        elif k == d - 1:
            # Right boundary: (J, n_k, 1)
            # Each component is summed at the end
            core = np.zeros((J, n_k, 1))
            for j in range(J):
                core[j, :, 0] = proj_vectors[k][j]
        else:
            # Interior: diagonal pattern (J, n_k, J)
            # Component j flows through bond index j
            core = np.zeros((J, n_k, J))
            for j in range(J):
                core[j, :, j] = proj_vectors[k][j]
        
        cores.append(core)
    
    density_tt = TTTrain(cores=cores)
    return TTDensitySpec(
        density_tt=density_tt,
        density_bases=list(committor_basis.bases),
    )


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
    density_spec: Optional[TTDensitySpec] = None,
) -> Union[Committor1DResult, CommittorNDDenseResult, CommittorTTResult]:
    """
    Solve a committor problem from a notebook-friendly interface.

    Dispatch rules
    --------------
    - If V, beta, a, b are provided, the 1D solver is used.
    - If density_spec is provided, solves an nD problem with a general TT-format
      (non-separable) density using the TT-density path. Requires wA_fns, wB_fns,
      and method='tt'.
    - Otherwise solves an nD product-measure problem assembled from the supplied
      basis and per-dimension measure functions (original behavior).
    
    Parameters
    ----------
    V, beta, a, b : 1D problem specification
    nbasis : basis size for 1D solver
    rho : soft-boundary penalty weight
    pA, pB : 1D soft-boundary measures (optional, defaults to unit Gaussians)
    sigma : width for default soft-boundary Gaussians
    nquad : Gauss-Legendre quadrature points
    basis_kind : "legendre" or "fourier" (for nD)
    ns : basis sizes per dimension (for nD)
    intervals : domain intervals per dimension (for nD)
    weight_fns : per-dimension product density factors (for nD product mode)
    wA_fns, wB_fns : per-dimension soft-boundary factors (required for nD)
    method : solver method ("1d", "dense", or "tt")
    tt_rank : internal TT bond dimension
    n_sweeps : max ALS sweeps per rho stage
    tol : ALS convergence tolerance
    seed : RNG seed
    verbose : enable detailed solver output
    rho_schedule : custom rho continuation schedule
    density_spec : Optional[TTDensitySpec]
        If provided, activates the general TT-density mode. This allows
        non-separable densities represented in low-rank TT format.
        Requires wA_fns, wB_fns, and method='tt'.
    
    Returns
    -------
    Committor1DResult, CommittorNDDenseResult, or CommittorTTResult
    
    Examples
    --------
    1D problem (original API)::
    
        result = find_committor(
            V=lambda x: (x**2 - 1)**2,
            beta=5.0,
            a=-2.0,
            b=2.0,
            nbasis=30,
            rho=400.0,
        )
    
    nD product density (original API)::
    
        result = find_committor(
            basis_kind="legendre",
            ns=[20]*4,
            intervals=[(-2.0, 2.0)]*4,
            weight_fns=[lambda x: np.exp(-5*(x**2-1)**2)] * 4,
            wA_fns=wA_fns,
            wB_fns=wB_fns,
            rho=400.0,
            method="tt",
        )
    
    nD TT-format (non-separable) density::
    
        basis = make_basis(
            basis_kind="legendre",
            ns=[20]*4,
            intervals=[(-2.0, 2.0)]*4,
        )
        
        # Create a rank-2 mixture: 70% centered at -1, 30% at +1
        density_spec = create_mixture_tt_density(
            [
                [lambda x: np.exp(-5*(x-1)**2)] * 4,      # component 1
                [lambda x: np.exp(-5*(x+1)**2)] * 4,      # component 2
            ],
            basis,
            component_weights=[0.7, 0.3],
        )
        
        result = find_committor(
            ns=[20]*4,
            intervals=[(-2.0, 2.0)]*4,
            wA_fns=wA_fns,
            wB_fns=wB_fns,
            rho=400.0,
            method="tt",
            density_spec=density_spec,
        )
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

    # nD TT-density front door (new)
    if density_spec is not None:
        if method != "tt":
            raise ValueError("TT-density mode requires method='tt'.")
        if wA_fns is None or wB_fns is None:
            raise ValueError(
                "For TT-density mode, wA_fns and wB_fns must be provided."
            )
        
        # Reconstruct the full TensorProductBasis from density_bases
        committor_basis = TensorProductBasis(tuple(density_spec.density_bases))
        
        return solve_committor_nd_tt(
            per_dim=None,
            basis=committor_basis,
            rho=float(rho),
            density_spec=density_spec,
            wA_fns=list(wA_fns),
            wB_fns=list(wB_fns),
            nquad=int(nquad),
            tt_rank=int(tt_rank),
            n_sweeps=int(n_sweeps),
            tol=float(tol),
            seed=int(seed),
            verbose=bool(verbose),
            rho_schedule=rho_schedule,
        )

    # nD product-density front door (original)
    if ns is None or intervals is None or weight_fns is None or wA_fns is None or wB_fns is None:
        raise ValueError(
            "For nD problems, provide ns, intervals, weight_fns, wA_fns, and wB_fns. "
            "Alternatively, provide a density_spec for general TT densities."
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
    "create_mixture_tt_density",
    "find_committor",
]
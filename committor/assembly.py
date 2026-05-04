"""Problem assembly: per-dimension quadrature matrices and MPO construction.

Implements the Galerkin discretisation described in paper Sections 3.1-3.4,
including the per-dimension integral matrices and their assembly into MPO and
h^B TT formats for both rank-1 product measures (double-well) and rank-J
density TTs (Ginzburg-Landau).

Public API
----------
PerDimMatrices           — per-dimension quadrature matrices (S, M, MA, MB, bvec)
quadrature_matrices_1d   — 1D quadrature (used by solve_committor_1d)
quadrature_matrices_nd   — multi-D quadrature for rank-1 product measures
assemble_dense_nd        — assemble full Kronecker LHS/RHS for small d
assemble_mpo_rank1       — MPO for H + rho*H_A + rho*H_B (rank-1 density)
assemble_hb_tt           — h^B as a rank-1 TTTrain (rank-1 density)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from committor._types import Array, ScalarFn
from committor.basis import UnivariateBasis, TensorProductBasis
from committor.tensor_train import TTTrain, MPOTrain


# ---------------------------------------------------------------------------
# PerDimMatrices
# ---------------------------------------------------------------------------

@dataclass
class PerDimMatrices:
    """Per-dimension quadrature matrices for one factor of a rank-1 product measure.

    For p(x) = prod_k p_k(x_k) and likewise for pA, pB, each of the full
    d-dimensional matrices decomposes into Kronecker products of these small
    per-dimension matrices (see assemble_dense_nd).

    These correspond to the paper's marginal matrices in Fig 3.4:
        S  ↔  Ĩ_k  (stiffness, dphi-weighted)
        M  ↔  I_k   (mass, phi-weighted, for equilibrium density)
        MA ↔  I_k^A (soft-A mass)
        MB ↔  I_k^B (soft-B mass)

    Attributes
    ----------
    S    : ndarray, shape (n_k, n_k).  integral dphi'_i dphi'_j p_k dx_k
    M    : ndarray, shape (n_k, n_k).  integral phi_i phi_j p_k dx_k
    MA   : ndarray, shape (n_k, n_k).  integral phi_i phi_j pA_k dx_k
    MB   : ndarray, shape (n_k, n_k).  integral phi_i phi_j pB_k dx_k
    bvec : ndarray, shape (n_k,).      integral phi_i pB_k dx_k
    """
    S:    Array
    M:    Array
    MA:   Array
    MB:   Array
    bvec: Array


# ---------------------------------------------------------------------------
# 1D quadrature helper (used by solve_committor_1d)
# ---------------------------------------------------------------------------

def quadrature_matrices_1d(
    basis:  UnivariateBasis,
    weight: ScalarFn,
    wA:     ScalarFn,
    wB:     ScalarFn,
    nquad:  int,
) -> Tuple[Array, Array, Array, Array]:
    """Compute 1D variational matrices by Gauss-Legendre quadrature.

    Returns (S, MA, MB, bvec):

        S    = integral phi'_i phi'_j p dx     stiffness
        MA   = integral phi_i  phi_j  pA dx    soft-A mass
        MB   = integral phi_i  phi_j  pB dx    soft-B mass
        bvec = integral phi_i  pB dx           h^B vector  (paper eq. 3.6)
    """
    a, b = basis.a, basis.b
    xs, ws = leggauss(nquad)
    x = 0.5 * (b - a) * xs + 0.5 * (b + a)
    w = 0.5 * (b - a) * ws

    Phi  = np.vstack([f(x) for f in basis.fns])    # (nbasis, nquad)
    DPhi = np.vstack([f(x) for f in basis.dfns])

    diag_p = w * weight(x)
    diag_A = w * wA(x)
    diag_B = w * wB(x)

    S    = DPhi @ (diag_p[:, None] * DPhi.T)
    MA   = Phi  @ (diag_A[:, None] * Phi.T)
    MB   = Phi  @ (diag_B[:, None] * Phi.T)
    bvec = Phi  @ diag_B

    return S, MA, MB, bvec


# ---------------------------------------------------------------------------
# Multi-dimensional quadrature for rank-1 product measures
# ---------------------------------------------------------------------------

def quadrature_matrices_nd(
    basis:       TensorProductBasis,
    weight_fns:  Sequence[ScalarFn],
    wA_fns:      Sequence[ScalarFn],
    wB_fns:      Sequence[ScalarFn],
    nquad:       int,
) -> List[PerDimMatrices]:
    """Compute per-dimension quadrature matrices for a rank-1 product measure.

    For a product measure p(x) = prod_k p_k(x_k), the d-dimensional Galerkin
    matrices decompose into independent per-dimension 1D integrals (paper
    Section 3.2).

    Parameters
    ----------
    basis       : TensorProductBasis with d UnivariateBases.
    weight_fns  : length-d sequence.  weight_fns[k](x) = p_k(x).
    wA_fns      : length-d sequence.  wA_fns[k](x) = pA_k(x).
    wB_fns      : length-d sequence.  wB_fns[k](x) = pB_k(x).
    nquad       : Gauss-Legendre quadrature points per dimension.

    Returns
    -------
    list of d PerDimMatrices objects, one per dimension.
    """
    d = basis.d
    if len(weight_fns) != d:
        raise ValueError(f"weight_fns must have length d={d}, got {len(weight_fns)}.")
    if len(wA_fns) != d:
        raise ValueError(f"wA_fns must have length d={d}, got {len(wA_fns)}.")
    if len(wB_fns) != d:
        raise ValueError(f"wB_fns must have length d={d}, got {len(wB_fns)}.")

    result: List[PerDimMatrices] = []
    for k in range(d):
        uvb = basis.bases[k]
        a_k, b_k = uvb.a, uvb.b

        xs, ws = leggauss(nquad)
        x = 0.5 * (b_k - a_k) * xs + 0.5 * (b_k + a_k)
        w = 0.5 * (b_k - a_k) * ws

        Phi  = basis.eval_marginal(k, x)   # (ns[k], nquad)
        DPhi = basis.deval_marginal(k, x)  # (ns[k], nquad)

        diag_p = w * weight_fns[k](x)
        diag_A = w * wA_fns[k](x)
        diag_B = w * wB_fns[k](x)

        S_k    = DPhi @ (diag_p[:, None] * DPhi.T)
        M_k    = Phi  @ (diag_p[:, None] * Phi.T)
        MA_k   = Phi  @ (diag_A[:, None] * Phi.T)
        MB_k   = Phi  @ (diag_B[:, None] * Phi.T)
        bvec_k = Phi  @ diag_B

        result.append(PerDimMatrices(S=S_k, M=M_k, MA=MA_k, MB=MB_k, bvec=bvec_k))

    return result


# ---------------------------------------------------------------------------
# Dense Kronecker assembly (only feasible for small d)
# ---------------------------------------------------------------------------

def assemble_dense_nd(
    per_dim: List[PerDimMatrices],
    rho:     float,
) -> Tuple[Array, Array]:
    """Assemble the dense Galerkin LHS and RHS from per-dimension matrices.

    For a rank-1 product measure the full d-dimensional matrices have
    Kronecker structure:

        H   = sum_k M_0 ⊗ ... ⊗ S_k ⊗ ... ⊗ M_{d-1}    [stiffness]
        H_A = MA_0 ⊗ MA_1 ⊗ ... ⊗ MA_{d-1}              [soft-A mass]
        H_B = MB_0 ⊗ MB_1 ⊗ ... ⊗ MB_{d-1}              [soft-B mass]
        h_B = bvec_0 ⊗ bvec_1 ⊗ ... ⊗ bvec_{d-1}        [RHS]

    Only suitable for small d (d <= 4 with ns_k ~ 10).  For large d,
    use TT-ALS (solvers.solve_committor_nd_tt).

    Returns
    -------
    lhs : ndarray, shape (N, N),  N = prod(ns_k)
    rhs : ndarray, shape (N,)
    """
    d = len(per_dim)

    def _kron(arrays):
        out = arrays[0]
        for arr in arrays[1:]:
            out = np.kron(out, arr)
        return out

    M_list    = [pd.M    for pd in per_dim]
    S_list    = [pd.S    for pd in per_dim]
    MA_list   = [pd.MA   for pd in per_dim]
    MB_list   = [pd.MB   for pd in per_dim]
    bvec_list = [pd.bvec for pd in per_dim]

    H = sum(
        _kron([S_list[k] if l == k else M_list[l] for l in range(d)])
        for k in range(d)
    )

    H_A = _kron(MA_list)
    H_B = _kron(MB_list)
    h_B = _kron(bvec_list)

    lhs = H + rho * (H_A + H_B)
    rhs = rho * h_B
    return lhs, rhs


# ---------------------------------------------------------------------------
# MPO assembly — rank-1 product-measure case (paper Section 3.2)
# ---------------------------------------------------------------------------

def assemble_mpo_rank1(
    per_dim: List[PerDimMatrices],
    rho:     float,
) -> MPOTrain:
    """Build the variational MPO  W = H + rho*H_A + rho*H_B for a rank-1 measure.

    Assumes that p, pA, pB each factor as a tensor product (rank-1 TT).
    The combined operator W is encoded as a single MPO with bond dimension 4
    via a 4-state finite-state machine (FSM):

        state 0 — "S not yet applied" (active path for H)
        state 1 — "S already applied" (done path for H)
        state 2 — H_A path
        state 3 — H_B path

    Special case d=1: the single core collapses to (S + rho*MA + rho*MB).

    Parameters
    ----------
    per_dim : list of d PerDimMatrices.
    rho     : soft-boundary penalty weight.

    Returns
    -------
    MPOTrain with d cores, bond dim 4 on interior bonds.
    """
    d = len(per_dim)

    if d == 1:
        pd = per_dim[0]
        W = (pd.S + rho * pd.MA + rho * pd.MB).reshape(1, pd.S.shape[0], pd.S.shape[0], 1)
        return MPOTrain(cores=[W])

    cores: List[Array] = []

    for k, pd in enumerate(per_dim):
        n = pd.S.shape[0]

        if k == 0:
            W = np.zeros((1, n, n, 4))
            W[0, :, :, 0] = pd.M
            W[0, :, :, 1] = pd.S
            W[0, :, :, 2] = rho * pd.MA
            W[0, :, :, 3] = rho * pd.MB

        elif k < d - 1:
            W = np.zeros((4, n, n, 4))
            W[0, :, :, 0] = pd.M
            W[0, :, :, 1] = pd.S
            W[1, :, :, 1] = pd.M
            W[2, :, :, 2] = pd.MA
            W[3, :, :, 3] = pd.MB

        else:
            W = np.zeros((4, n, n, 1))
            W[0, :, :, 0] = pd.S
            W[1, :, :, 0] = pd.M
            W[2, :, :, 0] = pd.MA
            W[3, :, :, 0] = pd.MB

        cores.append(W)

    return MPOTrain(cores=cores)


# ---------------------------------------------------------------------------
# h^B TT assembly — rank-1 case
# ---------------------------------------------------------------------------

def assemble_hb_tt(per_dim: List[PerDimMatrices]) -> TTTrain:
    """Build the h^B TT from per-dimension bvec arrays (paper eq. 3.6).

    For a rank-1 product measure:
        h^B(i_1,...,i_d) = bvec_0[i_0] * bvec_1[i_1] * ... * bvec_{d-1}[i_{d-1}]

    This is a rank-1 TT: core_k = bvec_k.reshape(1, n_k, 1).

    Parameters
    ----------
    per_dim : list of d PerDimMatrices.

    Returns
    -------
    TTTrain with d rank-1 cores encoding h^B.
    """
    cores = [pd.bvec.reshape(1, pd.bvec.size, 1) for pd in per_dim]
    return TTTrain(cores=cores)

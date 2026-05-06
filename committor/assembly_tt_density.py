"""Generalized MPO and h^B assembly for non-product (TT-format) equilibrium densities.

This module lifts the GL-specific logic in ginzburg_landau._assemble_mpo_gl into a
reusable abstraction.  The key difference from assembly.assemble_mpo_rank1 is:

    rank-1 density  →  per-site blocks are plain (n×n) matrices
    rank-J TT density →  per-site blocks are (J_left × n × n × J_right) tensors,
                          and the FSM bond dimension grows from 4 to 2J+2.

Public API
----------
TTDensitySpec         — bundles a density TTTrain with its per-site basis functions
assemble_mpo_tt_density — MPO for H + rho*H^A + rho*H^B (TT-format density)
assemble_hb_tt_density  — h^B TT (rank-1 if pB is a product measure)

Relation to existing code
--------------------------
* assembly.assemble_mpo_rank1  ←→  assemble_mpo_tt_density  with J=1
  (rank-1 is a special case; no code duplication needed)
* ginzburg_landau._assemble_mpo_gl  ←→  assemble_mpo_tt_density
  (the GL function is now a thin wrapper that builds a TTDensitySpec and calls this)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss

from committor._types import Array, ScalarFn
from committor.basis import TensorProductBasis, UnivariateBasis
from committor.tensor_train import TTTrain, MPOTrain


# ---------------------------------------------------------------------------
# TTDensitySpec — bundles density TT + per-site basis
# ---------------------------------------------------------------------------

@dataclass
class TTDensitySpec:
    """A TT-format density ρ(x) expanded in per-dimension univariate bases.

    The density is represented as the matrix-product sum

        ρ(x_1,...,x_d) = G_1(x_1) G_2(x_2) ... G_d(x_d)

    where each factor G_k(x_k) is recovered from the TT core and a basis:

        G_k(x_k)[α_left, α_right] = sum_n density_tt.cores[k][α_left, n, α_right]
                                     * density_bases[k].fns[n](x_k)

    Attributes
    ----------
    density_tt    : TTTrain — coefficient tensor for ρ in the per-site bases.
                    Core k has shape (J_{k-1}, m_k, J_k) where m_k is the number of
                    density basis functions at site k and J_k is the TT rank.
    density_bases : list of d UnivariateBases — the bases used to expand ρ at each site.
                    density_bases[k] must have exactly density_tt.cores[k].shape[1] functions.
    """
    density_tt:    TTTrain
    density_bases: List[UnivariateBasis]

    def __post_init__(self) -> None:
        d = self.density_tt.d
        if len(self.density_bases) != d:
            raise ValueError(
                f"TTDensitySpec: density_tt.d={d} but len(density_bases)={len(self.density_bases)}."
            )
        for k, (core, uvb) in enumerate(zip(self.density_tt.cores, self.density_bases)):
            if core.shape[1] != uvb.n:
                raise ValueError(
                    f"TTDensitySpec: site {k} — density core physical dim {core.shape[1]} "
                    f"!= density_bases[k].n={uvb.n}."
                )

    @property
    def d(self) -> int:
        return self.density_tt.d

    @property
    def J(self) -> int:
        """Interior TT bond dimension (rank of density)."""
        return self.density_tt.ranks[1]  # first interior rank; works for uniform rank


# ---------------------------------------------------------------------------
# Per-site 3-index quadrature tensor
# ---------------------------------------------------------------------------

def _compute_raw_integrals(
    committor_uvb: UnivariateBasis,
    density_uvb:   UnivariateBasis,
    nquad:         int,
) -> tuple:
    """Compute the 3-index quadrature tensors at one site.

    Returns
    -------
    raw_I      : ndarray (n_commit, n_commit, m_density)
        raw_I[i, j, n] = integral phi_i(x) phi_j(x) psi_n(x) dx
    raw_Itilde : ndarray (n_commit, n_commit, m_density)
        raw_Itilde[i, j, n] = integral dphi_i(x) dphi_j(x) psi_n(x) dx

    where {phi_i} are the committor basis functions and {psi_n} are the
    density basis functions at this site.
    """
    a, b = committor_uvb.a, committor_uvb.b
    xs_std, ws_std = leggauss(nquad)
    xs = 0.5 * (b - a) * xs_std + 0.5 * (b + a)
    ws = 0.5 * (b - a) * ws_std

    # Committor basis evaluated at quadrature nodes
    Phi  = np.vstack([f(xs)  for f in committor_uvb.fns])   # (n_commit, nquad)
    DPhi = np.vstack([f(xs)  for f in committor_uvb.dfns])  # (n_commit, nquad)

    # Density basis evaluated at quadrature nodes
    Psi  = np.vstack([f(xs)  for f in density_uvb.fns])     # (m_density, nquad)

    raw_I      = np.einsum('iq,jq,nq,q->ijn', Phi,  Phi,  Psi, ws)
    raw_Itilde = np.einsum('iq,jq,nq,q->ijn', DPhi, DPhi, Psi, ws)
    return raw_I, raw_Itilde


# ---------------------------------------------------------------------------
# MPO assembly — general TT-density case
# ---------------------------------------------------------------------------

def assemble_mpo_tt_density(
    density_spec:    TTDensitySpec,
    committor_basis: TensorProductBasis,
    wA_fns:          Sequence[ScalarFn],
    wB_fns:          Sequence[ScalarFn],
    rho:             float,
    nquad:           int = 500,
) -> MPOTrain:
    """Build the variational MPO  W = H + rho*H^A + rho*H^B  for a TT-format density.

    Generalises assembly.assemble_mpo_rank1 to the case where the equilibrium
    density ρ is represented as a rank-J tensor train rather than a rank-1 product.

    The MPO bond dimension is 2J + 2:
        states 0..J-1     — "H stiffness not yet applied" (tracking J density-TT states)
        states J..2J-1    — "H stiffness already applied" (tracking J density-TT states)
        state  2J         — H^A path (rank-1 product boundary)
        state  2J+1       — H^B path (rank-1 product boundary)

    This is a direct generalisation of the 4-state FSM in assemble_mpo_rank1
    (which corresponds to J=1: states {0,1,2,3} become {pending, done, A, B}).

    Parameters
    ----------
    density_spec    : TTDensitySpec — TT density ρ with its per-site basis.
    committor_basis : TensorProductBasis — trial basis for the committor q.
    wA_fns          : length-d sequence.  wA_fns[k](x) = pA_k(x).
    wB_fns          : length-d sequence.  wB_fns[k](x) = pB_k(x).
    rho             : soft-boundary penalty weight.
    nquad           : Gauss-Legendre quadrature points per site.

    Returns
    -------
    MPOTrain with bond dimension 2*J + 2.
    """
    d   = committor_basis.d
    J   = density_spec.J  # interior TT bond dimension

    if d != density_spec.d:
        raise ValueError(
            f"assemble_mpo_tt_density: committor basis has d={d} "
            f"but density has d={density_spec.d}."
        )
    if d == 0:
        raise ValueError("d must be >= 1.")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")

    mpo_cores: List[Array] = []

    for l in range(d):
        commit_uvb   = committor_basis.bases[l]
        density_uvb  = density_spec.density_bases[l]
        n_k          = commit_uvb.n                    # committor basis size at site l
        p_core       = density_spec.density_tt.cores[l]  # (J_left, m, J_right)
        J_left, m, J_right = p_core.shape

        # Quadrature nodes on the committor basis interval
        a_l, b_l = commit_uvb.a, commit_uvb.b
        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_l - a_l) * xs_std + 0.5 * (b_l + a_l)
        ws = 0.5 * (b_l - a_l) * ws_std

        # 3-index quadrature tensors
        raw_I, raw_Itilde = _compute_raw_integrals(commit_uvb, density_uvb, nquad)
        # raw_I[i, j, n], raw_Itilde[i, j, n]

        # Contract with density core:  I_l[α, i, j, β] = sum_n core[α,n,β] * raw_I[i,j,n]
        I_l      = np.einsum('anb,ijn->aijb', p_core, raw_I)        # (J_left, n, n, J_right)
        Itilde_l = np.einsum('anb,ijn->aijb', p_core, raw_Itilde)   # (J_left, n, n, J_right)

        # Soft-boundary mass matrices (product form — only 1D integrals needed)
        Phi  = np.vstack([f(xs) for f in commit_uvb.fns])   # (n_k, nquad)
        MA_l = np.einsum('iq,jq,q->ij', Phi, Phi, ws * wA_fns[l](xs))  # (n_k, n_k)
        MB_l = np.einsum('iq,jq,q->ij', Phi, Phi, ws * wB_fns[l](xs))  # (n_k, n_k)

        # --- Assemble MPO core with FSM bond layout [pending(J), done(J), A(1), B(1)] ---
        # Bond sizes: left_bond × n_k × n_k × right_bond
        if d == 1:
            # Collapse everything into a single (1, n, n, 1) core
            W = np.zeros((1, n_k, n_k, 1))
            # H = stiffness contracted with full density (boundary ranks are 1)
            W[0, :, :, 0] = Itilde_l[0, :, :, 0] + rho * MA_l + rho * MB_l
            mpo_cores.append(W)
            continue

        if l == 0:
            # Left boundary: w_left = 1 → use first row of p_core (J_left=1)
            # Output has 2J+2 states
            bond_out = 2 * J_right + 2
            W = np.zeros((1, n_k, n_k, bond_out))
            # H pending — output states 0..J_right-1
            W[0, :, :, :J_right]           = I_l[0, :, :, :]       # pending starts here
            # H done after first site — output states J_right..2*J_right-1
            W[0, :, :, J_right:2*J_right]  = Itilde_l[0, :, :, :]  # stiffness at site 0
            # H^A path — state 2J_right
            W[0, :, :, 2*J_right]          = rho * MA_l
            # H^B path — state 2J_right+1
            W[0, :, :, 2*J_right + 1]      = rho * MB_l

        elif l < d - 1:
            # Interior site; both bonds have size 2J+2
            J_in  = J_left   # density TT left rank at this site
            J_out = J_right  # density TT right rank at this site
            bond_in  = 2 * J_in  + 2
            bond_out = 2 * J_out + 2
            W = np.zeros((bond_in, n_k, n_k, bond_out))

            # Pending → pending:  mass matrix (stiffness not yet applied)
            W[:J_in,        :, :, :J_out]        = I_l              # H pending propagates
            # Pending → done:     stiffness matrix (applying stiffness at this site)
            W[:J_in,        :, :, J_out:2*J_out] = Itilde_l         # transition
            # Done → done:        mass matrix (stiffness already applied earlier)
            W[J_in:2*J_in,  :, :, J_out:2*J_out] = I_l             # H done propagates
            # H^A path propagates
            W[2*J_in,       :, :, 2*J_out]       = MA_l
            # H^B path propagates
            W[2*J_in + 1,   :, :, 2*J_out + 1]  = MB_l

        else:
            # Right boundary: w_right = 1 → use last column of p_core (J_right=1)
            bond_in = 2 * J_left + 2
            W = np.zeros((bond_in, n_k, n_k, 1))

            # Pending → done (last site must apply stiffness here if not yet done)
            # But note: after d sites only "done" contributes to <q|H|q>.
            # At the last site: pending path must pick up stiffness, done picks up mass.
            W[:J_left,          :, :, 0] = Itilde_l[:, :, :, 0]     # pending → stiffness at last site
            W[J_left:2*J_left,  :, :, 0] = I_l[:, :, :, 0]          # done → mass at last site
            W[2*J_left,         :, :, 0] = MA_l
            W[2*J_left + 1,     :, :, 0] = MB_l

        mpo_cores.append(W)

    return MPOTrain(cores=mpo_cores)


# ---------------------------------------------------------------------------
# h^B TT assembly — product-form pB (most common case)
# ---------------------------------------------------------------------------

def assemble_hb_tt_density(
    committor_basis: TensorProductBasis,
    wB_fns:          Sequence[ScalarFn],
    nquad:           int = 500,
) -> TTTrain:
    """Build the h^B right-hand-side TT when pB is a product measure.

    h^B(i) = integral phi_i(x) pB(x) dx
           = prod_k integral phi_k_{i_k}(x_k) pB_k(x_k) dx_k

    This is unchanged from assembly.assemble_hb_tt but accepts per-dim
    callables rather than PerDimMatrices, making it composable with the
    TT-density path.

    Returns
    -------
    TTTrain with d rank-1 cores.
    """
    d = committor_basis.d
    if len(wB_fns) != d:
        raise ValueError(f"wB_fns must have length d={d}.")

    cores: List[Array] = []
    for k in range(d):
        uvb = committor_basis.bases[k]
        a_k, b_k = uvb.a, uvb.b
        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_k - a_k) * xs_std + 0.5 * (b_k + a_k)
        ws = 0.5 * (b_k - a_k) * ws_std

        Phi  = np.vstack([f(xs) for f in uvb.fns])   # (n_k, nquad)
        bvec = Phi @ (ws * wB_fns[k](xs))             # (n_k,)
        cores.append(bvec.reshape(1, uvb.n, 1))

    return TTTrain(cores=cores)


# ---------------------------------------------------------------------------
# Convenience: wrap a rank-1 product density as a TTDensitySpec
# ---------------------------------------------------------------------------

def product_density_to_tt_spec(
    weight_fns:    Sequence[ScalarFn],
    committor_basis: TensorProductBasis,
    nquad:         int = 500,
) -> TTDensitySpec:
    """Convert a product-form density into a rank-1 TTDensitySpec.

    This is provided so callers can pass a product density through the
    general TT-density solver path without changing the assembly logic.
    The resulting TTDensitySpec has J=1 at every bond and the density basis
    is the same as the committor basis.

    For each dimension k the density core is shape (1, n_k, 1) with entries

        core[0, i, 0] = integral phi_i(x) weight_fns[k](x) dx

    so that the density is approximated in the committor basis.

    Parameters
    ----------
    weight_fns       : length-d list of per-dimension density callables.
    committor_basis  : TensorProductBasis.
    nquad            : quadrature points.

    Returns
    -------
    TTDensitySpec with J=1 (rank-1 density TT).
    """
    d = committor_basis.d
    if len(weight_fns) != d:
        raise ValueError(f"weight_fns must have length d={d}.")

    cores: List[Array] = []
    for k in range(d):
        uvb  = committor_basis.bases[k]
        a_k, b_k = uvb.a, uvb.b
        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_k - a_k) * xs_std + 0.5 * (b_k + a_k)
        ws = 0.5 * (b_k - a_k) * ws_std

        Phi = np.vstack([f(xs) for f in uvb.fns])   # (n_k, nquad)
        # Project p_k onto the basis: coeff[i] = integral phi_i(x) p_k(x) dx
        # (orthonormal Legendre → this recovers the Legendre coefficients of p_k)
        coeff = Phi @ (ws * weight_fns[k](xs))       # (n_k,)
        cores.append(coeff.reshape(1, uvb.n, 1))

    density_tt = TTTrain(cores=cores)
    return TTDensitySpec(
        density_tt=density_tt,
        density_bases=list(committor_basis.bases),
    )
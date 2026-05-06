"""Alternating Least Squares (ALS) solver for the TT committor problem.

Implements the ALS algorithm described in paper Section 3.5 for minimising
the variational objective J(Q) = <Q|W|Q> - 2*rho*<Q|h^B> over the
tensor-train parametrisation of the coefficient tensor Q.

Public API
----------
als_local_matrix         — assemble effective dense matrix at one site
als_local_rhs            — assemble effective RHS vector at one site
als_core_to_vec          — flatten a TT core to a vector
als_vec_to_core          — reshape a vector to a TT core
als_single_site_update   — solve the local linear system at one site
als_left_to_right_sweep  — one L->R sweep over all d sites
als_right_to_left_sweep  — one R->L sweep over all d sites
als_solve                — multi-sweep driver with convergence check
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from committor._types import Array
from committor.tensor_train import (
    TTTrain, MPOTrain,
    tt_left_envs, tt_right_envs,
    mpo_left_envs, mpo_right_envs,
    mpo_inner, tt_inner,
    tt_update_left_env, tt_update_right_env,
    mpo_update_left_env, mpo_update_right_env,
)


# ---------------------------------------------------------------------------
# Local problem assemblers (per-site)
# ---------------------------------------------------------------------------

def als_local_matrix(L: Array, mpo_core: Array, R: Array) -> Array:
    """Assemble the dense effective matrix for the ALS update at one site.

    M[a,i,c, b,j,d] = sum_{w,x}  L[a, w, b]  *  W[w, i, j, x]  *  R[c, x, d]

    Parameters
    ----------
    L        : ndarray, shape (rL, wL, rL)
    mpo_core : ndarray, shape (wL, n, n, wR)
    R        : ndarray, shape (rR, wR, rR)

    Returns
    -------
    M : ndarray, shape (rL*n*rR, rL*n*rR)
    """
    rL, wL, rL2 = L.shape
    wL2, n_bra, n_ket, wR = mpo_core.shape
    rR, wR2, rR2 = R.shape

    if rL != rL2:
        raise ValueError(f"als_local_matrix: L must be square in bond dims; got L.shape={L.shape}.")
    if rR != rR2:
        raise ValueError(f"als_local_matrix: R must be square in bond dims; got R.shape={R.shape}.")
    if n_bra != n_ket:
        raise ValueError(f"als_local_matrix: MPO core must have square physical legs; got {mpo_core.shape}.")
    if wL != wL2:
        raise ValueError(f"als_local_matrix: L right bond {wL} does not match MPO left bond {wL2}.")
    if wR != wR2:
        raise ValueError(f"als_local_matrix: R middle bond {wR2} does not match MPO right bond {wR}.")

    n = n_bra
    M_full = np.einsum('awb,wijx,cxd->aicbjd', L, mpo_core, R)
    return M_full.reshape(rL * n * rR, rL * n * rR)


def als_core_to_vec(core: Array) -> Array:
    """Flatten a TT core of shape (rL, n, rR) to a 1-D vector (C order)."""
    return core.ravel()


def als_vec_to_core(vec: Array, rL: int, n: int, rR: int) -> Array:
    """Reshape a flat vector to a TT core of shape (rL, n, rR) (C order)."""
    return vec.reshape(rL, n, rR)


def als_local_rhs(L: Array, rhs_core: Array, R: Array) -> Array:
    """Assemble the dense local RHS vector for the ALS update at one site.

    f[a, i, c] = sum_{b, d}  L[a, b]  *  rhs_core[b, i, d]  *  R[c, d]

    Parameters
    ----------
    L        : ndarray, shape (rL_Q, rL_hB)
    rhs_core : ndarray, shape (rL_hB, n, rR_hB)  — k-th core of h^B TT.
    R        : ndarray, shape (rR_Q, rR_hB)

    Returns
    -------
    f : ndarray, shape (rL_Q * n * rR_Q,)
    """
    rL_Q, rL_hB  = L.shape
    sL, n, sR    = rhs_core.shape
    rR_Q, rR_hB  = R.shape

    if rL_hB != sL:
        raise ValueError(
            f"als_local_rhs: L.shape[1]={rL_hB} does not match rhs_core.shape[0]={sL}."
        )
    if rR_hB != sR:
        raise ValueError(
            f"als_local_rhs: R.shape[1]={rR_hB} does not match rhs_core.shape[2]={sR}."
        )

    f_full = np.einsum('ab,bid,cd->aic', L, rhs_core, R)
    return f_full.ravel()


def als_single_site_update(
    k:     int,
    tt:    TTTrain,
    mpo:   MPOTrain,
    hb_tt: TTTrain,
    rho:   float,
    L_mpo: Array,
    R_mpo: Array,
    L_hb:  Array,
    R_hb:  Array,
) -> Array:
    """Perform one ALS site update: solve the local linear system for core k.

    Assembles M_k and f_k, then solves M_k @ vec = rho * f_k.

    Parameters
    ----------
    k     : site index in 0..d-1.
    tt    : current TTTrain; core k is read only for its shape.
    mpo   : combined operator MPO for H + rho*H_A + rho*H_B.
    hb_tt : TTTrain for h^B (paper eq. 3.6).
    rho   : soft-boundary penalty weight.
    L_mpo : left MPO environment at site k, shape (rL, wL, rL).
    R_mpo : right MPO environment at site k, shape (rR, wR, rR).
    L_hb  : left h^B environment at site k, shape (rL, rL_hB).
    R_hb  : right h^B environment at site k, shape (rR, rR_hB).

    Returns
    -------
    new_core : ndarray, shape (rL, n, rR).
    """
    rL, n, rR = tt.cores[k].shape

    M = als_local_matrix(L_mpo, mpo.cores[k], R_mpo)
    f = rho * als_local_rhs(L_hb, hb_tt.cores[k], R_hb)

    vec, *_ = np.linalg.lstsq(M, f, rcond=None)

    return als_vec_to_core(vec, rL, n, rR)


# ---------------------------------------------------------------------------
# Left-to-right ALS sweep
# ---------------------------------------------------------------------------

def als_left_to_right_sweep(
    tt:             TTTrain,
    mpo:            MPOTrain,
    hb_tt:          TTTrain,
    rho:            float,
    right_envs_mpo: Optional[List[Array]] = None,
    right_envs_hb:  Optional[List[Array]] = None,
) -> Tuple[TTTrain, float]:
    """Perform one left-to-right ALS sweep, updating all d cores in sequence.

    Parameters
    ----------
    tt             : TTTrain — initial / current iterate.
    mpo            : MPOTrain — combined operator H + rho*H^A + rho*H^B.
    hb_tt          : TTTrain — h^B right-hand side TT.
    rho            : soft-boundary penalty weight.
    right_envs_mpo : optional precomputed right MPO environments.
    right_envs_hb  : optional precomputed right h^B environments.

    Returns
    -------
    updated_tt : TTTrain.
    objective  : float — variational objective after the sweep.
    """
    d = tt.d

    if right_envs_mpo is None:
        right_envs_mpo = mpo_right_envs(tt, mpo, tt)
    if right_envs_hb is None:
        right_envs_hb = tt_right_envs(tt, hb_tt)

    L_mpo: Array = np.ones((1, 1, 1))
    L_hb:  Array = np.ones((1, 1))

    working_cores: List[Array] = list(tt.cores)

    for k in range(d):
        R_mpo = right_envs_mpo[d - k - 1]
        R_hb  = right_envs_hb[d - k - 1]

        working_tt = TTTrain(cores=working_cores)
        new_core = als_single_site_update(
            k, working_tt, mpo, hb_tt, rho,
            L_mpo=L_mpo, R_mpo=R_mpo, L_hb=L_hb, R_hb=R_hb,
        )
        working_cores[k] = new_core

        L_mpo = mpo_update_left_env(L_mpo, new_core, mpo.cores[k], new_core)
        L_hb  = tt_update_left_env(L_hb, new_core, hb_tt.cores[k])

    updated_tt = TTTrain(cores=working_cores)
    obj = (mpo_inner(updated_tt, mpo, updated_tt)
           - 2.0 * rho * tt_inner(updated_tt, hb_tt))

    return updated_tt, float(obj)


# ---------------------------------------------------------------------------
# Right-to-left ALS sweep
# ---------------------------------------------------------------------------

def als_right_to_left_sweep(
    tt:            TTTrain,
    mpo:           MPOTrain,
    hb_tt:         TTTrain,
    rho:           float,
    left_envs_mpo: Optional[List[Array]] = None,
    left_envs_hb:  Optional[List[Array]] = None,
) -> Tuple[TTTrain, float]:
    """Perform one right-to-left ALS sweep, updating all d cores in sequence.

    Mirror of als_left_to_right_sweep; sites visited in order d-1, d-2, ..., 0.

    Parameters
    ----------
    tt            : TTTrain — initial / current iterate.
    mpo           : MPOTrain — combined operator.
    hb_tt         : TTTrain — h^B right-hand side TT.
    rho           : soft-boundary penalty weight.
    left_envs_mpo : optional precomputed left MPO environments.
    left_envs_hb  : optional precomputed left h^B environments.

    Returns
    -------
    updated_tt : TTTrain.
    objective  : float — variational objective after the sweep.
    """
    d = tt.d

    if left_envs_mpo is None:
        left_envs_mpo = mpo_left_envs(tt, mpo, tt)
    if left_envs_hb is None:
        left_envs_hb = tt_left_envs(tt, hb_tt)

    R_mpo: Array = np.ones((1, 1, 1))
    R_hb:  Array = np.ones((1, 1))

    working_cores: List[Array] = list(tt.cores)

    for k in range(d - 1, -1, -1):
        L_mpo = left_envs_mpo[k]
        L_hb  = left_envs_hb[k]

        working_tt = TTTrain(cores=working_cores)
        new_core = als_single_site_update(
            k, working_tt, mpo, hb_tt, rho,
            L_mpo=L_mpo, R_mpo=R_mpo, L_hb=L_hb, R_hb=R_hb,
        )
        working_cores[k] = new_core

        R_mpo = mpo_update_right_env(R_mpo, new_core, mpo.cores[k], new_core)
        R_hb  = tt_update_right_env(R_hb, new_core, hb_tt.cores[k])

    updated_tt = TTTrain(cores=working_cores)
    obj = (mpo_inner(updated_tt, mpo, updated_tt)
           - 2.0 * rho * tt_inner(updated_tt, hb_tt))

    return updated_tt, float(obj)


# ---------------------------------------------------------------------------
# Multi-sweep driver
# ---------------------------------------------------------------------------

def als_solve(
    tt:            TTTrain,
    mpo:           MPOTrain,
    hb_tt:         TTTrain,
    rho:           float,
    n_sweeps:      int   = 20,
    tol:           float = 1e-8,
    verbose:       bool  = False,
    bidirectional: bool  = True,
) -> Tuple[TTTrain, List[float]]:
    """Minimise the variational objective by repeated ALS sweeps.

    Minimises J(Q) = <Q|W|Q> - 2*rho*<Q|h^B> over the TT parametrisation.

    When ``bidirectional=True`` (default) each sweep = L->R + R->L, sharing
    environments for efficiency.  When ``bidirectional=False`` each sweep is
    a single L->R pass.

    Convergence criterion: |J_s - J_{s-1}| / (1 + |J_{s-1}|) < tol.

    Parameters
    ----------
    tt            : TTTrain — initial iterate; not mutated.
    mpo           : MPOTrain — combined operator H + rho*H^A + rho*H^B.
    hb_tt         : TTTrain — h^B right-hand side TT.
    rho           : float — soft-boundary penalty weight (constant).
    n_sweeps      : int — maximum number of sweeps.
    tol           : float — relative-change stopping threshold.
    verbose       : bool — if True, print objective after each sweep.
    bidirectional : bool — if True (default), sweep = L->R + R->L.

    Returns
    -------
    tt_out  : TTTrain — final iterate.
    history : list of float — objective value after each sweep.
    """
    if n_sweeps < 1:
        raise ValueError(f"als_solve: n_sweeps must be >= 1, got {n_sweeps}.")
    if tol < 0:
        raise ValueError(f"als_solve: tol must be >= 0, got {tol}.")

    current_tt = tt
    history: List[float] = []

    for s in range(n_sweeps):
        if bidirectional:
            lr_tt, _obj_lr = als_left_to_right_sweep(current_tt, mpo, hb_tt, rho)

            left_envs_mpo = mpo_left_envs(lr_tt, mpo, lr_tt)
            left_envs_hb  = tt_left_envs(lr_tt, hb_tt)

            current_tt, obj = als_right_to_left_sweep(
                lr_tt, mpo, hb_tt, rho,
                left_envs_mpo=left_envs_mpo,
                left_envs_hb=left_envs_hb,
            )
        else:
            current_tt, obj = als_left_to_right_sweep(current_tt, mpo, hb_tt, rho)

        history.append(obj)

        if verbose:
            direction = "L->R+R->L" if bidirectional else "L->R"
            print(f"  als_solve sweep {s + 1:3d}/{n_sweeps} ({direction}):  J = {obj:.10g}")

        if s > 0:
            rel_change = abs(history[-1] - history[-2]) / (1.0 + abs(history[-2]))
            if rel_change < tol:
                if verbose:
                    print(f"  Converged after {s + 1} sweeps "
                          f"(rel_change={rel_change:.2e} < tol={tol:.2e}).")
                break

    return current_tt, history
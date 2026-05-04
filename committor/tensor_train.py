"""Tensor-train (MPS) and matrix product operator (MPO) data structures and operations.

Implements the TT/MPS infrastructure described in paper Sections 2.3, 3.2-3.4.
These are the core data structures manipulated by the ALS solver (als.py) and
the problem assembly routines (assembly.py).

Public API
----------
TTTrain               — d-core tensor train (paper eq. 3.13)
MPOTrain              — d-core matrix product operator (paper eq. 2.14)
tt_from_dense         — convert a dense tensor to TT via sequential SVD
tt_evaluate           — evaluate a TT-parametrised function at sample points
tt_inner              — coefficient inner product <u|v>
mpo_inner             — operator inner product <u|O|v>
tt_left_envs          — left partial overlap environments
tt_right_envs         — right partial overlap environments
mpo_left_envs         — left partial MPO environments
mpo_right_envs        — right partial MPO environments
tt_update_left_env    — incremental left-env update (one site)
tt_update_right_env   — incremental right-env update (one site)
mpo_update_left_env   — incremental MPO left-env update (one site)
mpo_update_right_env  — incremental MPO right-env update (one site)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from committor._types import Array
from committor.basis import TensorProductBasis


# ---------------------------------------------------------------------------
# TTTrain — minimal tensor-train container  (paper eq. 3.13)
# ---------------------------------------------------------------------------

@dataclass
class TTTrain:
    """Tensor-train (MPS) representation of the coefficient tensor Q.

    Stores d cores G_1, ..., G_d following paper eq. (3.13):

        Q(i_1, ..., i_d) = G_1[0, i_1, :] G_2[:, i_2, :] ... G_d[:, i_d, 0]

    Boundary ranks satisfy r_0 = r_d = 1 by convention.

    Attributes
    ----------
    cores : list of d 3-tensors, cores[k] shape (r_{k-1}, n_k, r_k).
    """
    cores: List[Array]

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("TTTrain requires at least one core.")
        for k, G in enumerate(self.cores):
            if G.ndim != 3:
                raise ValueError(
                    f"Core {k} must be a 3-tensor (r_left, n, r_right); "
                    f"got ndim={G.ndim}."
                )
        if self.cores[0].shape[0] != 1:
            raise ValueError(
                "Left boundary rank must be 1 (r_0 = 1 by convention); "
                f"core[0].shape[0] = {self.cores[0].shape[0]}."
            )
        if self.cores[-1].shape[2] != 1:
            raise ValueError(
                "Right boundary rank must be 1 (r_d = 1 by convention); "
                f"core[-1].shape[2] = {self.cores[-1].shape[2]}."
            )
        for k in range(len(self.cores) - 1):
            r_right = self.cores[k].shape[2]
            r_left  = self.cores[k + 1].shape[0]
            if r_right != r_left:
                raise ValueError(
                    f"Bond dimension mismatch: core[{k}].shape[2]={r_right} "
                    f"!= core[{k+1}].shape[0]={r_left}."
                )

    @property
    def d(self) -> int:
        """Number of dimensions / cores."""
        return len(self.cores)

    @property
    def ns(self) -> Tuple[int, ...]:
        """Physical (basis) dimension of each site: (n_1, ..., n_d)."""
        return tuple(G.shape[1] for G in self.cores)

    @property
    def ranks(self) -> Tuple[int, ...]:
        """Bond dimensions (r_0, r_1, ..., r_d); r_0 = r_d = 1."""
        rs = [G.shape[0] for G in self.cores]
        rs.append(self.cores[-1].shape[2])
        return tuple(rs)


# ---------------------------------------------------------------------------
# MPOTrain — matrix product operator container  (paper eq. 2.13-2.14)
# ---------------------------------------------------------------------------

@dataclass
class MPOTrain:
    """Matrix product operator (MPO) with 4-index cores (paper eq. 2.14, Fig 2.3b).

    Each core has shape ``(w_left, n, n, w_right)`` where n is the shared
    physical (basis) dimension.  The operator is assembled as:

        O(i_1,...,i_d ; j_1,...,j_d)
            = W_1[0, i_1, j_1, :] W_2[:, i_2, j_2, :] ... W_d[:, i_d, j_d, 0]

    Boundary bonds satisfy w_0 = w_d = 1 by convention.

    Attributes
    ----------
    cores : list of d 4-tensors, cores[k].shape == (w_{k-1}, n_k, n_k, w_k).
    """
    cores: List[Array]

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("MPOTrain requires at least one core.")
        for k, W in enumerate(self.cores):
            if W.ndim != 4:
                raise ValueError(
                    f"MPOTrain core {k} must be a 4-tensor "
                    f"(w_left, n, n, w_right); got ndim={W.ndim}."
                )
            if W.shape[1] != W.shape[2]:
                raise ValueError(
                    f"MPOTrain core {k}: physical dimensions must be equal "
                    f"(square operator); got shape {W.shape}."
                )
        if self.cores[0].shape[0] != 1:
            raise ValueError(
                "Left boundary bond must be 1 (w_0 = 1 by convention); "
                f"core[0].shape[0] = {self.cores[0].shape[0]}."
            )
        if self.cores[-1].shape[3] != 1:
            raise ValueError(
                "Right boundary bond must be 1 (w_d = 1 by convention); "
                f"core[-1].shape[3] = {self.cores[-1].shape[3]}."
            )
        for k in range(len(self.cores) - 1):
            w_right = self.cores[k].shape[3]
            w_left  = self.cores[k + 1].shape[0]
            if w_right != w_left:
                raise ValueError(
                    f"MPOTrain bond mismatch at site {k}: "
                    f"core[{k}].shape[3]={w_right} "
                    f"!= core[{k+1}].shape[0]={w_left}."
                )

    @property
    def d(self) -> int:
        return len(self.cores)

    @property
    def ns(self) -> Tuple[int, ...]:
        return tuple(W.shape[1] for W in self.cores)

    @property
    def bonds(self) -> Tuple[int, ...]:
        ws = [W.shape[0] for W in self.cores]
        ws.append(self.cores[-1].shape[3])
        return tuple(ws)


# ---------------------------------------------------------------------------
# TT utility functions
# ---------------------------------------------------------------------------

def tt_from_dense(tensor: Array, max_rank: Optional[int] = None) -> TTTrain:
    """Convert a dense d-tensor to TT format via sequential left-to-right SVD.

    Parameters
    ----------
    tensor   : ndarray, shape (n_1, ..., n_d).
    max_rank : optional int.  Truncate each SVD to at most this many singular values.

    Returns
    -------
    TTTrain with d = tensor.ndim cores.
    """
    tensor = np.asarray(tensor, dtype=float)
    ns = tensor.shape
    d  = len(ns)
    if d == 0:
        raise ValueError("tensor must have at least one dimension.")

    cores: List[Array] = []
    T = tensor.copy()
    r = 1

    for k in range(d - 1):
        T = T.reshape(r * ns[k], -1)
        U, s, Vt = np.linalg.svd(T, full_matrices=False)
        r_new = len(s) if max_rank is None else min(max_rank, len(s))
        U, s, Vt = U[:, :r_new], s[:r_new], Vt[:r_new, :]
        cores.append(U.reshape(r, ns[k], r_new))
        T = np.diag(s) @ Vt
        r = r_new

    cores.append(T.reshape(r, ns[-1], 1))
    return TTTrain(cores=cores)


def tt_evaluate(tt: TTTrain, basis: TensorProductBasis, x: Array) -> Array:
    """Evaluate a TT-parametrised committor at sample points.

    Implements paper eq. (3.1) + (3.13) by contracting left-to-right.

    Parameters
    ----------
    tt    : TTTrain, d cores.
    basis : TensorProductBasis — must satisfy basis.d == tt.d and basis.ns == tt.ns.
    x     : ndarray, shape (n_samples, d).

    Returns
    -------
    ndarray, shape (n_samples,).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != tt.d:
        raise ValueError(
            f"tt_evaluate expects x of shape (n_samples, d={tt.d}), got {x.shape}."
        )
    if basis.d != tt.d:
        raise ValueError(f"basis.d={basis.d} does not match tt.d={tt.d}.")
    if basis.ns != tt.ns:
        raise ValueError(f"basis.ns={basis.ns} does not match tt.ns={tt.ns}.")

    n_samples = x.shape[0]
    v = np.ones((n_samples, 1))

    for k in range(tt.d):
        G     = tt.cores[k]                          # (r_{k-1}, n_k, r_k)
        Phi_k = basis.eval_marginal(k, x[:, k])     # (n_k, n_samples)
        G_eff = np.einsum('aib,is->abs', G, Phi_k)  # (r_left, r_right, n_samples)
        v = np.einsum('sa,abs->sb', v, G_eff)        # (n_samples, r_right)

    return v[:, 0]


# ---------------------------------------------------------------------------
# Overlap environments
# ---------------------------------------------------------------------------

def tt_left_envs(u: TTTrain, v: TTTrain) -> List[Array]:
    """Left partial overlap environments for the coefficient inner product <u|v>.

    Returns list of d+1 arrays; envs[k] has shape (r_k^u, r_k^v).
    """
    if u.d != v.d:
        raise ValueError(f"u.d={u.d} != v.d={v.d}.")
    if u.ns != v.ns:
        raise ValueError(f"u.ns={u.ns} != v.ns={v.ns}.")

    envs: List[Array] = [np.ones((1, 1))]
    for k in range(u.d):
        L  = envs[-1]
        Gu = u.cores[k]
        Gv = v.cores[k]
        envs.append(np.einsum('ab,aic,bid->cd', L, Gu, Gv))
    return envs


def tt_right_envs(u: TTTrain, v: TTTrain) -> List[Array]:
    """Right partial overlap environments (stored in reverse order, R_d first).

    Returns list of d+1 arrays; envs[j] = R_{d-j}, shape (r_{d-j}^u, r_{d-j}^v).
    """
    if u.d != v.d:
        raise ValueError(f"u.d={u.d} != v.d={v.d}.")
    if u.ns != v.ns:
        raise ValueError(f"u.ns={u.ns} != v.ns={v.ns}.")

    envs: List[Array] = [np.ones((1, 1))]
    for k in range(u.d - 1, -1, -1):
        R  = envs[-1]
        Gu = u.cores[k]
        Gv = v.cores[k]
        envs.append(np.einsum('aic,bid,cd->ab', Gu, Gv, R))
    return envs


def tt_inner(u: TTTrain, v: TTTrain) -> float:
    """Coefficient inner product <u|v> = sum_i u(i) * v(i)."""
    return float(tt_left_envs(u, v)[-1][0, 0])


def mpo_left_envs(u: TTTrain, mpo: MPOTrain, v: TTTrain) -> List[Array]:
    """Left partial environments for the operator inner product <u|O|v>.

    Returns list of d+1 arrays; envs[k] has shape (r_k^u, w_k, r_k^v).
    """
    if u.d != v.d or u.d != mpo.d:
        raise ValueError(f"d mismatch: u.d={u.d}, mpo.d={mpo.d}, v.d={v.d}.")
    if u.ns != mpo.ns:
        raise ValueError(f"Physical dims mismatch: u.ns={u.ns}, mpo.ns={mpo.ns}.")
    if v.ns != mpo.ns:
        raise ValueError(f"Physical dims mismatch: v.ns={v.ns}, mpo.ns={mpo.ns}.")

    envs: List[Array] = [np.ones((1, 1, 1))]
    for k in range(u.d):
        L  = envs[-1]
        Gu = u.cores[k]
        W  = mpo.cores[k]
        Gv = v.cores[k]
        envs.append(np.einsum('awb,aic,wijx,bjd->cxd', L, Gu, W, Gv))
    return envs


def mpo_right_envs(u: TTTrain, mpo: MPOTrain, v: TTTrain) -> List[Array]:
    """Right partial MPO environments (stored in reverse order, R_d first).

    Returns list of d+1 arrays; envs[j] = R_{d-j}, shape (r_{d-j}^u, w_{d-j}, r_{d-j}^v).
    """
    if u.d != v.d or u.d != mpo.d:
        raise ValueError(f"d mismatch: u.d={u.d}, mpo.d={mpo.d}, v.d={v.d}.")
    if u.ns != mpo.ns:
        raise ValueError(f"Physical dims mismatch: u.ns={u.ns}, mpo.ns={mpo.ns}.")
    if v.ns != mpo.ns:
        raise ValueError(f"Physical dims mismatch: v.ns={v.ns}, mpo.ns={mpo.ns}.")

    envs: List[Array] = [np.ones((1, 1, 1))]
    for k in range(u.d - 1, -1, -1):
        R  = envs[-1]
        Gu = u.cores[k]
        W  = mpo.cores[k]
        Gv = v.cores[k]
        envs.append(np.einsum('aic,wijx,bjd,cxd->awb', Gu, W, Gv, R))
    return envs


def mpo_inner(u: TTTrain, mpo: MPOTrain, v: TTTrain) -> float:
    """Full operator inner product <u|O|v> = sum_{i,j} u(i) O(i,j) v(j)."""
    return float(mpo_left_envs(u, mpo, v)[-1][0, 0, 0])


# ---------------------------------------------------------------------------
# Incremental environment updaters (used by ALS sweeps)
# ---------------------------------------------------------------------------

def tt_update_left_env(L: Array, core_u: Array, core_v: Array) -> Array:
    """Extend a TT-overlap left environment by one site.

    L_new[c, d] = sum_{a, b, i}  L[a, b] * G^u[a, i, c] * G^v[b, i, d]
    """
    return np.einsum('ab,aic,bid->cd', L, core_u, core_v)


def mpo_update_left_env(
    L: Array, core_u: Array, mpo_core: Array, core_v: Array,
) -> Array:
    """Extend an MPO left environment by one site.

    L_new[c, x, d] = sum_{a,w,b,i,j}
                       L[a,w,b] * G^u[a,i,c] * W[w,i,j,x] * G^v[b,j,d]
    """
    return np.einsum('awb,aic,wijx,bjd->cxd', L, core_u, mpo_core, core_v)


def tt_update_right_env(R: Array, core_u: Array, core_v: Array) -> Array:
    """Extend a TT-overlap right environment by one site (rightward contraction).

    R_new[a, b] = sum_{c, d, i}  G^u[a,i,c] * G^v[b,i,d] * R[c,d]
    """
    return np.einsum('aic,bid,cd->ab', core_u, core_v, R)


def mpo_update_right_env(
    R: Array, core_u: Array, mpo_core: Array, core_v: Array,
) -> Array:
    """Extend an MPO right environment by one site (rightward contraction).

    R_new[a, w, b] = sum_{c,x,d,i,j}
                       G^u[a,i,c] * W[w,i,j,x] * G^v[b,j,d] * R[c,x,d]
    """
    return np.einsum('aic,wijx,bjd,cxd->awb', core_u, mpo_core, core_v, R)

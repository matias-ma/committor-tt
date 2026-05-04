"""Ginzburg-Landau committor: density TT, MPO assembly, and solver.

Implements the paper Section 4.2 pipeline for the discretised Ginzburg-Landau
model.  The equilibrium density is NOT a product measure, so the rank-1
machinery in assembly.py does not apply; instead, a transfer-matrix
eigendecomposition (paper Appendix B) yields a rank-J TT for the density.

Public API
----------
ginzburg_landau_kernel          — nearest-neighbour transfer kernel K(x,y)
compute_gl_kernel_eigenfunctions — top-J eigenfunctions of the GL kernel
compute_gl_tt_cores             — build density TT from eigenfunctions (Appendix B)
compute_gl_minimizers           — numerically find the two global minima U_±
solve_gl_committor              — full Section 4.2 pipeline → CommittorTTResult

Internal helpers (prefixed with _) are also importable for testing.

Architecture note
-----------------
The key difference from the double-well (Section 4.1) is that the equilibrium
density p here has bond dimension J (≈ 6) rather than 1.  This propagates
through MPO assembly: the resulting MPO bond dimension is O(J) rather than 4.
The _assemble_mpo_gl and _assemble_hb_gl functions handle this generalisation.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

from committor._types import Array
from committor.basis import TensorProductBasis, fourier_basis
from committor.tensor_train import TTTrain, MPOTrain
from committor.als import als_solve
from committor.solvers import CommittorTTResult


# ---------------------------------------------------------------------------
# Transfer kernel
# ---------------------------------------------------------------------------

def ginzburg_landau_kernel(
    x: Array, y: Array, lam: float, h: float, beta: float = 1.0
) -> Array:
    """Nearest-neighbour transfer kernel K_beta(x, y) for the GL model.

    K_beta(x, y) = exp(-beta/(8*lam) * (1-x^2)^2)
                 * exp(-beta*lam/(2*h^2) * (x-y)^2)
                 * exp(-beta/(8*lam) * (1-y^2)^2)

    At beta=1 this is the kernel from paper Appendix B.
    x and y may be arrays; the result is broadcast.

    Parameters
    ----------
    x, y  : array-like — evaluation points.
    lam   : float      — GL coupling constant (paper: 0.03).
    h     : float      — lattice spacing h = 1/(d+1).
    beta  : float      — inverse temperature 1/T (default 1.0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    single_x = -(beta / (8.0 * lam)) * (1.0 - x ** 2) ** 2
    coupling  = -(beta * lam / (2.0 * h ** 2)) * (x - y) ** 2
    single_y  = -(beta / (8.0 * lam)) * (1.0 - y ** 2) ** 2
    return np.exp(single_x + coupling + single_y)


# ---------------------------------------------------------------------------
# Kernel eigenfunction computation (Appendix B, Step 1)
# ---------------------------------------------------------------------------

def compute_gl_kernel_eigenfunctions(
    lam:   float,
    h:     float,
    R:     float,
    J:     int,
    nquad: int   = 200,
    beta:  float = 1.0,
) -> Tuple[Array, Array, Array]:
    """Compute the top-J eigenfunctions of the GL transfer kernel.

    Numerically solves the integral eigenvalue problem

        H0[phi](x) = integral_{-R}^{R} K_beta(x, y) phi(y) dy = lambda * phi(x)

    via Gauss-Legendre discretisation and a symmetrised eigendecomposition.

    Returns v_j = sqrt(lambda_j) * u_j (scaled eigenfunctions), where
    {u_j} are the orthonormal eigenfunctions of H0 and {lambda_j} the
    corresponding eigenvalues in descending order.

    These are used in the Mercer expansion
        K(x,y) ≈ sum_j v_j(x) * v_j(y)
    and appear in the TT representation of p (paper eq. B.1-B.2).

    Parameters
    ----------
    lam   : GL coupling constant.
    h     : lattice spacing 1/(d+1).
    R     : domain half-width (paper: 2.6).
    J     : number of eigenpairs to retain (paper: 6).
    nquad : Gauss-Legendre points for discretisation (200 suffices).
    beta  : inverse temperature (default 1.0).

    Returns
    -------
    xs      : ndarray, shape (nquad,)  — quadrature nodes on [-R, R].
    ws      : ndarray, shape (nquad,)  — positive quadrature weights.
    eig_fns : ndarray, shape (J, nquad) — v_j(xs) in descending order.
    """
    if lam <= 0 or h <= 0 or R <= 0:
        raise ValueError("lam, h, R must all be positive.")
    if J < 1:
        raise ValueError(f"J must be >= 1, got {J}.")
    if nquad < J:
        raise ValueError(f"nquad={nquad} must be >= J={J}.")

    # Gauss-Legendre quadrature on [-R, R]
    raw_pts, raw_ws = leggauss(nquad)
    xs = R * raw_pts
    ws = R * raw_ws

    # Kernel matrix K[i, j] = K_beta(xs[i], xs[j])
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    K_mat  = ginzburg_landau_kernel(XX, YY, lam, h, beta=beta)

    # Symmetrised Gram matrix G = diag(sqrt(w)) K diag(sqrt(w))
    # converts H0 u = lambda u to a standard symmetric eigenproblem G g = lambda g
    sqrt_ws = np.sqrt(ws)
    G       = K_mat * np.outer(sqrt_ws, sqrt_ws)

    evals_all, evecs_all = np.linalg.eigh(G)  # ascending order

    # Sort descending (dominant modes first)
    idx       = np.argsort(evals_all)[::-1]
    evals_all = evals_all[idx]
    evecs_all = evecs_all[:, idx]

    evals = np.maximum(evals_all[:J], 0.0)
    evecs = evecs_all[:, :J]

    if evals_all[J - 1] < -1e-10:
        warnings.warn(
            f"Eigenvalue {J-1} = {evals_all[J-1]:.2e} is significantly negative.",
            stacklevel=2,
        )

    # Recover function-space eigenfunctions; scale to v_j = sqrt(lambda_j) * u_j
    u_fns   = evecs / sqrt_ws[:, None]        # (nquad, J)
    eig_fns = (np.sqrt(evals) * u_fns).T      # (J, nquad)

    return xs, ws, eig_fns


# ---------------------------------------------------------------------------
# Density TT construction (Appendix B, Step 2)
# ---------------------------------------------------------------------------

def compute_gl_tt_cores(
    eig_fns: Array,
    xs:      Array,
    ws:      Array,
    cheb_N:  int,
    R:       float,
    c_lam:   float,
    v0:      Array,
    d:       int = 50,
) -> TTTrain:
    """Build the GL equilibrium density p as a rank-J TT (Appendix B).

    The density factorises as (paper eq. B.1):
        p(U_1,...,U_d) ~ c_lam * K(0,U_1) * K(U_1,U_2) * ... * K(U_d,0)

    After Mercer expansion K(x,y) ≈ sum_j v_j(x)*v_j(y) and Chebyshev
    expansion of each univariate factor, the density becomes a rank-J TT
    with Chebyshev-coefficient physical indices (eq. B.2).

    Parameters
    ----------
    eig_fns : ndarray, shape (J, nquad) — scaled eigenfunctions v_j(xs).
    xs, ws  : ndarray, shape (nquad,)   — GL quadrature nodes / weights.
    cheb_N  : int   — Chebyshev truncation order.
    R       : float — domain half-width.
    c_lam   : float — normalisation constant exp(-beta/(4*lam)).
    v0      : ndarray, shape (J,) — v_j(0) (boundary values).
    d       : int   — number of GL lattice sites.

    Returns
    -------
    TTTrain with ranks (1, J, J, ..., J, 1) and physical dims cheb_N+1.
    """
    J, nquad = eig_fns.shape
    N = cheb_N

    # Step 1: Chebyshev polynomials on quadrature grid
    t = xs / R
    cheb_vals = np.empty((N + 1, nquad), dtype=float)
    cheb_vals[0] = 1.0
    if N >= 1:
        cheb_vals[1] = t
    for n in range(2, N + 1):
        cheb_vals[n] = 2.0 * t * cheb_vals[n - 1] - cheb_vals[n - 2]

    # Step 2: Chebyshev moment tensor A[j,l,n] = integral v_j(x)*v_l(x)*T_n(x/R) dx
    A = np.einsum('jq,lq,nq,q->jln', eig_fns, eig_fns, cheb_vals, ws)  # (J, J, N+1)

    # Step 3: Interior core — (J, N+1, J) ordering: (alpha_left, physical, alpha_right)
    interior = A.transpose(0, 2, 1).copy()

    # Step 4: Boundary cores absorbing v0 and c_lam
    first = (c_lam * np.einsum('a,abn->nb', v0, A)).reshape(1, N + 1, J)
    last  = np.einsum('abn,b->an', A, v0).reshape(J, N + 1, 1)

    # Step 5: Assemble TT
    if d == 1:
        single = (c_lam * np.einsum('a,abn,b->n', v0, A, v0)).reshape(1, N + 1, 1)
        return TTTrain(cores=[single])
    if d == 2:
        return TTTrain(cores=[first, last])

    cores = [first] + [interior.copy() for _ in range(d - 2)] + [last]
    return TTTrain(cores=cores)


# ---------------------------------------------------------------------------
# Gaussian Chebyshev cores for soft-boundary measures
# ---------------------------------------------------------------------------

def _gaussian_cheb_cores(
    mu:     Array,
    sigma:  float,
    basis:  TensorProductBasis,
    cheb_N: int,
    nquad:  int,
) -> List[Array]:
    """Expand per-dimension Gaussians N(mu_k, sigma^2) in the Chebyshev basis.

    The soft-boundary measures pA, pB (paper eq. 3.8) are product Gaussians
    N(U; mu, sigma^2 I_d), which factor dimension-by-dimension.  This function
    computes, for each dimension k, the Chebyshev expansion of the k-th
    marginal Gaussian pA_k(x) = exp(-(x - mu_k)^2 / (2*sigma^2)).

    Parameters
    ----------
    mu     : ndarray, shape (d,) — mean of the Gaussian (U_- or U_+).
    sigma  : float — isotropic standard deviation.
    basis  : TensorProductBasis with d Fourier bases on [-gamma, gamma].
    cheb_N : int — Chebyshev truncation order.
    nquad  : int — quadrature points per dimension.

    Returns
    -------
    list of d ndarrays, each shape (1, cheb_N+1, 1) — rank-1 TT cores
    for the Chebyshev expansion of pA or pB.
    """
    d  = basis.d
    N  = cheb_N

    cores: List[Array] = []
    for k in range(d):
        uvb = basis.bases[k]
        a_k, b_k = uvb.a, uvb.b
        R_k = (b_k - a_k) / 2.0

        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_k - a_k) * xs_std + 0.5 * (b_k + a_k)
        ws = 0.5 * (b_k - a_k) * ws_std

        # Chebyshev basis T_n((x - mid) / R_k) on [a_k, b_k]
        mid = 0.5 * (a_k + b_k)
        t   = (xs - mid) / R_k
        cheb_at_xs = np.empty((N + 1, nquad))
        cheb_at_xs[0] = 1.0
        if N >= 1:
            cheb_at_xs[1] = t
        for n in range(2, N + 1):
            cheb_at_xs[n] = 2.0 * t * cheb_at_xs[n - 1] - cheb_at_xs[n - 2]

        # Gaussian values at quadrature nodes
        pA_vals = np.exp(-0.5 * ((xs - mu[k]) / sigma) ** 2)

        # Chebyshev expansion coefficients: c_n = integral pA(x) T_n((x-mid)/R_k) dx
        c = cheb_at_xs @ (ws * pA_vals)  # shape (N+1,)
        cores.append(c.reshape(1, N + 1, 1))

    return cores


# ---------------------------------------------------------------------------
# h^B TT for GL (generalised from rank-1 case)
# ---------------------------------------------------------------------------

def _assemble_hb_gl(
    tt_p:     TTTrain,
    basis:    TensorProductBasis,
    pB_cores: List[Array],
    nquad:    int,
) -> TTTrain:
    """Assemble the h^B right-hand side TT for the GL committor problem.

    h^B(i) = integral phi_{i_1,...,i_d}(x) pB(x) dx
           = prod_k integral phi^(k)_{i_k}(x_k) pB_k(x_k) dx_k

    where pB = pB_1 ⊗ ... ⊗ pB_d is a product Gaussian (rank-1).
    The committor basis {phi^(k)} are the Fourier functions.

    This returns a rank-1 TTTrain (one core per dimension) where core_k
    holds the vector of inner products < phi^(k)_j, pB_k >.

    Parameters
    ----------
    tt_p     : TTTrain — density TT (not directly used; present for API consistency).
    basis    : TensorProductBasis — Fourier bases on [-gamma, gamma].
    pB_cores : list of d ndarray each (1, cheb_N+1, 1) — Chebyshev expansion of pB_k.
    nquad    : int — quadrature points per dimension.

    Returns
    -------
    TTTrain with d rank-1 cores, physical dim = n_k (Fourier basis size).
    """
    d  = basis.d
    N  = pB_cores[0].shape[1] - 1   # cheb_N

    hb_cores: List[Array] = []
    for k in range(d):
        uvb = basis.bases[k]
        a_k, b_k = uvb.a, uvb.b
        R_k = (b_k - a_k) / 2.0
        mid = 0.5 * (a_k + b_k)

        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_k - a_k) * xs_std + 0.5 * (b_k + a_k)
        ws = 0.5 * (b_k - a_k) * ws_std

        # Reconstruct pB_k from Chebyshev coefficients
        t = (xs - mid) / R_k
        cheb_at_xs = np.empty((N + 1, nquad))
        cheb_at_xs[0] = 1.0
        if N >= 1:
            cheb_at_xs[1] = t
        for n in range(2, N + 1):
            cheb_at_xs[n] = 2.0 * t * cheb_at_xs[n - 1] - cheb_at_xs[n - 2]

        cheb_coeffs = pB_cores[k][0, :, 0]       # (N+1,)
        pB_vals     = cheb_coeffs @ cheb_at_xs    # (nquad,)

        # Fourier basis at quadrature nodes
        Phi = basis.eval_marginal(k, xs)           # (n_k, nquad)
        bvec = Phi @ (ws * pB_vals)                # (n_k,)

        hb_cores.append(bvec.reshape(1, uvb.n, 1))

    return TTTrain(cores=hb_cores)


# ---------------------------------------------------------------------------
# MPO assembly for GL (generalised from rank-1)
# ---------------------------------------------------------------------------

def _assemble_mpo_gl(
    tt_p:     TTTrain,
    basis:    TensorProductBasis,
    nquad:    int,
    rho:      float,
    pA_cores: List[Array],
    pB_cores: List[Array],
) -> MPOTrain:
    """Assemble the variational MPO W = H + rho*H^A + rho*H^B for the GL problem.

    This is the generalised version of assemble_mpo_rank1 for a non-product
    density (rank-J TT).  The density TT has bond dimension J; the resulting
    MPO bond dimension is 2*J + 2 (H pending/done tracks J states; H^A, H^B
    are rank-1 so they add 2 more states).

    Parameters
    ----------
    tt_p     : TTTrain — equilibrium density in Chebyshev TT format, rank J.
    basis    : TensorProductBasis — Fourier bases on [-gamma, gamma].
    nquad    : int   — quadrature points per dimension.
    rho      : float — soft-boundary penalty weight.
    pA_cores : list of d ndarray each (1, N+1, 1) — Chebyshev cores for pA.
    pB_cores : list of d ndarray each (1, N+1, 1) — Chebyshev cores for pB.

    Returns
    -------
    MPOTrain with 2*J+2 bond dimension.
    """
    d     = basis.d
    J     = tt_p.ranks[1]   # interior bond dimension of density TT
    N     = tt_p.ns[0] - 1  # cheb_N (physical dim of density TT)
    R     = (basis.bases[0].b - basis.bases[0].a) / 2.0
    mid0  = 0.5 * (basis.bases[0].a + basis.bases[0].b)

    mpo_cores: List[Array] = []

    for l in range(d):
        uvb        = basis.bases[l]
        a_l, b_l   = uvb.a, uvb.b
        n_k        = uvb.n
        R_l        = (b_l - a_l) / 2.0
        mid_l      = 0.5 * (a_l + b_l)
        p_core     = tt_p.cores[l]         # (J_left, N+1, J_right)
        J_left, _, J_right = p_core.shape

        # Gauss-Legendre quadrature on [a_l, b_l]
        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b_l - a_l) * xs_std + 0.5 * (b_l + a_l)
        ws = 0.5 * (b_l - a_l) * ws_std

        # Committor Fourier basis on quadrature grid
        Phi  = basis.eval_marginal(l, xs)     # (n_k, nquad)
        DPhi = basis.deval_marginal(l, xs)    # (n_k, nquad)

        # Chebyshev polynomials T_n((x - mid_l) / R_l)
        t = (xs - mid_l) / R_l
        cheb_at_xs = np.empty((N + 1, nquad))
        cheb_at_xs[0] = 1.0
        if N >= 1:
            cheb_at_xs[1] = t
        for n in range(2, N + 1):
            cheb_at_xs[n] = 2.0 * t * cheb_at_xs[n - 1] - cheb_at_xs[n - 2]

        # Integral matrices contracted with density core
        # raw_I[i,j,n] = sum_q ws[q] T_n(q) phi_i(q) phi_j(q)
        raw_I      = np.einsum('nq,iq,jq,q->ijn', cheb_at_xs, Phi,  Phi,  ws)
        raw_Itilde = np.einsum('nq,iq,jq,q->ijn', cheb_at_xs, DPhi, DPhi, ws)

        # I_l[alpha, i, j, beta] = sum_n p_core[alpha,n,beta] raw_I[i,j,n]
        I_l      = np.einsum('anb,ijn->aijb', p_core, raw_I)
        Itilde_l = np.einsum('anb,ijn->aijb', p_core, raw_Itilde)

        # pA, pB mass matrices (rank-1)
        pA_cheb = pA_cores[l][0, :, 0]
        pB_cheb = pB_cores[l][0, :, 0]
        pA_vals = pA_cheb @ cheb_at_xs    # (nquad,)
        pB_vals = pB_cheb @ cheb_at_xs
        MA_l    = np.einsum('iq,jq,q->ij', Phi, Phi, ws * pA_vals)
        MB_l    = np.einsum('iq,jq,q->ij', Phi, Phi, ws * pB_vals)

        # Assemble MPO core using 4-state FSM extended to J states
        # Bond layout: [H_pending(J), H_done(J), H_A(1), H_B(1)] = 2J+2 total
        if d == 1:
            W = np.zeros((1, n_k, n_k, 1))
            W[0, :, :, 0] = Itilde_l[0, :, :, 0] + rho * MA_l + rho * MB_l
        elif l == 0:
            W = np.zeros((1, n_k, n_k, 2 * J + 2))
            W[0, :, :, :J]        = I_l[0, :, :, :]
            W[0, :, :, J:2 * J]   = Itilde_l[0, :, :, :]
            W[0, :, :, 2 * J]     = rho * MA_l
            W[0, :, :, 2 * J + 1] = rho * MB_l
        elif l < d - 1:
            W = np.zeros((2 * J + 2, n_k, n_k, 2 * J + 2))
            W[:J,        :, :, :J]        = I_l
            W[:J,        :, :, J:2 * J]   = Itilde_l
            W[J:2 * J,   :, :, J:2 * J]   = I_l
            W[2 * J,     :, :, 2 * J]     = MA_l
            W[2 * J + 1, :, :, 2 * J + 1] = MB_l
        else:
            W = np.zeros((2 * J + 2, n_k, n_k, 1))
            W[:J,        :, :, 0] = Itilde_l[:, :, :, 0]
            W[J:2 * J,   :, :, 0] = I_l[:, :, :, 0]
            W[2 * J,     :, :, 0] = MA_l
            W[2 * J + 1, :, :, 0] = MB_l

        mpo_cores.append(W)

    return MPOTrain(cores=mpo_cores)


# ---------------------------------------------------------------------------
# GL energy minimisers
# ---------------------------------------------------------------------------

def compute_gl_minimizers(d: int, lam: float) -> Tuple[Array, Array]:
    """Numerically find the two global minimisers U_± of the GL energy.

    Minimises the discretised Ginzburg-Landau energy (paper eq. 4.6)
        V(U) = sum_{i=1}^{d+1} [(lam/2h^2)(U_i - U_{i-1})^2 + (1/4lam)(1-U_i^2)^2]
    with Dirichlet BCs U_0 = U_{d+1} = 0.

    Uses L-BFGS-B with two tanh initialisations (positive / negative bump).

    Parameters
    ----------
    d   : number of interior lattice sites (paper: 50).
    lam : GL coupling constant (paper: 0.03).

    Returns
    -------
    (U_minus, U_plus) : tuple of ndarray, each shape (d,).
    """
    from scipy.optimize import minimize

    h = 1.0 / (d + 1)

    def gl_energy(U: Array) -> float:
        U_ext   = np.concatenate([[0.0], U, [0.0]])
        kinetic = 0.5 * lam / h ** 2 * float(np.sum(np.diff(U_ext) ** 2))
        pot     = 0.25 / lam * float(np.sum((1.0 - U ** 2) ** 2))
        return kinetic + pot

    def gl_grad(U: Array) -> Array:
        U_ext    = np.concatenate([[0.0], U, [0.0]])
        kin_grad = (lam / h ** 2) * (2.0 * U - U_ext[:-2] - U_ext[2:])
        pot_grad = (1.0 / lam) * U * (U ** 2 - 1.0)
        return kin_grad + pot_grad

    i_arr     = np.arange(1, d + 1, dtype=float)
    x_arr     = i_arr / (d + 1)
    sharpness = 10.0 / lam
    tanh_half = np.tanh(sharpness * (x_arr - 0.1)) - np.tanh(sharpness * (x_arr - 0.9))
    tanh_half = tanh_half / (tanh_half.max() + 1e-12)

    res_plus  = minimize(gl_energy,  tanh_half, jac=gl_grad, method='L-BFGS-B',
                         options={'maxiter': 2000, 'ftol': 1e-14})
    res_minus = minimize(gl_energy, -tanh_half, jac=gl_grad, method='L-BFGS-B',
                         options={'maxiter': 2000, 'ftol': 1e-14})

    return res_minus.x, res_plus.x


# ---------------------------------------------------------------------------
# Full Section 4.2 solver
# ---------------------------------------------------------------------------

def solve_gl_committor(
    d:        int   = 50,
    lam:      float = 0.03,
    T:        float = 8.0,
    gamma:    float = 2.6,
    n_basis:  int   = 5,
    cheb_N:   int   = 50,
    tt_rank:  int   = 6,
    nquad:    int   = 200,
    rho:      float = 100.0,
    n_sweeps: int   = 20,
    J:        int   = 6,
    R_ball:   float = 2.5,
    seed:     int   = 0,
    verbose:  bool  = False,
) -> CommittorTTResult:
    """Run the full Section 4.2 Ginzburg-Landau committor pipeline.

    Wires together all GL building blocks to produce a TT-ALS solution of
    the soft-committor variational problem for the Ginzburg-Landau potential.

    Pipeline
    --------
    1. Build Fourier TensorProductBasis (n_basis functions per dim).
    2. Compute kernel eigenfunctions and build density TT (Appendix B).
    3. Find GL minimisers U_± and build Gaussian soft-boundary TTs (eq. 3.8).
    4. Assemble h^B right-hand-side TT.
    5. Initialise random solution TT with bond dimension tt_rank.
    6. Run ρ-continuation ALS (4 stages: rho * [1e-3, 1e-2, 1e-1, 1.0]).

    Parameters
    ----------
    d        : number of GL lattice sites (paper: 50).
    lam      : GL coupling constant (paper: 0.03).
    T        : temperature; beta = 1/T (paper: 8 or 16).
    gamma    : domain half-width Ω = [-γ, γ]^d (paper: 2.6).
    n_basis  : Fourier basis functions per dimension (paper: 5; must be odd).
    cheb_N   : Chebyshev truncation order (use >= 50 for paper parameters).
    tt_rank  : fixed bond dimension of the solution TT (paper: 6).
    nquad    : Gauss-Legendre quadrature points per dimension.
    rho      : final soft-boundary penalty weight (paper: 100).
    n_sweeps : max ALS sweeps per ρ stage.
    J        : number of kernel eigenfunctions for the density TT (paper: 6).
    R_ball   : ball radius for metastable sets A and B (paper: 2.5).
    seed     : RNG seed for reproducible initialisation.
    verbose  : if True, print per-sweep diagnostics.

    Returns
    -------
    CommittorTTResult — evaluate with ``result.q(x)`` where x shape (n_samples, d).
    """
    beta = 1.0 / T
    h    = 1.0 / (d + 1)

    if verbose:
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  GL committor  d={d}, λ={lam}, T={T}, γ={gamma}")
        print(f"  n_basis={n_basis}, cheb_N={cheb_N}, tt_rank={tt_rank}")
        print(f"  rho={rho}, n_sweeps={n_sweeps}, J={J}, R_ball={R_ball}")
        print(sep)

    # 1. Fourier TensorProductBasis
    if verbose:
        print("  [1/6] Building Fourier TensorProductBasis …", flush=True)
    fourier_ub = fourier_basis(n_basis, gamma)
    basis = TensorProductBasis(bases=tuple(fourier_ub for _ in range(d)))

    # 2. Equilibrium density TT (Appendix B)
    if verbose:
        print("  [2/6] Computing kernel eigenfunctions and density TT …", flush=True)
    xs, ws, eig_fns = compute_gl_kernel_eigenfunctions(
        lam=lam, h=h, R=gamma, J=J, nquad=nquad, beta=beta
    )
    c_lam = np.exp(-beta / (4.0 * lam))
    v0    = np.array([np.interp(0.0, xs, eig_fns[j]) for j in range(J)])
    tt_p  = compute_gl_tt_cores(
        eig_fns=eig_fns, xs=xs, ws=ws,
        cheb_N=cheb_N, R=gamma, c_lam=c_lam, v0=v0, d=d,
    )

    # 3. Gaussian soft-boundary cores
    if verbose:
        print("  [3/6] Computing GL minimisers and Gaussian Chebyshev cores …",
              flush=True)
    U_minus, U_plus = compute_gl_minimizers(d=d, lam=lam)
    sigma = R_ball / np.sqrt(float(d))
    pA_cores = _gaussian_cheb_cores(
        mu=U_minus, sigma=sigma, basis=basis, cheb_N=cheb_N, nquad=nquad
    )
    pB_cores = _gaussian_cheb_cores(
        mu=U_plus, sigma=sigma, basis=basis, cheb_N=cheb_N, nquad=nquad
    )

    # 4. h^B TT
    if verbose:
        print("  [4/6] Assembling h^B TT …", flush=True)
    hb_tt = _assemble_hb_gl(
        tt_p=tt_p, basis=basis, pB_cores=pB_cores, nquad=nquad
    )

    # 5. Random initial TT
    rng = np.random.default_rng(seed)
    ns  = basis.ns
    init_cores: List[Array] = []
    for k in range(d):
        r_left  = 1 if k == 0     else tt_rank
        r_right = 1 if k == d - 1 else tt_rank
        init_cores.append(rng.standard_normal((r_left, ns[k], r_right)) * 0.1)
    tt_current = TTTrain(cores=init_cores)

    # 6. ρ-continuation ALS
    if verbose:
        print("  [5/6] Running ρ-continuation ALS …", flush=True)
    rho_schedule: List[float] = [rho * f for f in (1e-3, 1e-2, 1e-1, 1.0)]
    for stage_idx, rho_i in enumerate(rho_schedule):
        mpo_i = _assemble_mpo_gl(
            tt_p=tt_p, basis=basis, nquad=nquad, rho=rho_i,
            pA_cores=pA_cores, pB_cores=pB_cores,
        )
        if verbose:
            print(f"    [stage {stage_idx+1}/{len(rho_schedule)}] ρ={rho_i:.4g}",
                  flush=True)
        tt_current, history = als_solve(
            tt_current, mpo_i, hb_tt, rho_i,
            n_sweeps=n_sweeps, tol=1e-8, verbose=verbose,
        )
        if verbose:
            print(f"    → {len(history)} sweep(s), J_final={history[-1]:.6g}")

    if verbose:
        print("  [6/6] Done.")

    return CommittorTTResult(tt=tt_current, basis=basis, rho=rho)

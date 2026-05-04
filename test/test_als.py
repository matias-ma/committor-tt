"""Tests for committor.als — alternating least-squares solver."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from committor.als import (
    als_local_matrix,
    als_local_rhs,
    als_core_to_vec,
    als_vec_to_core,
    als_single_site_update,
    als_left_to_right_sweep,
    als_right_to_left_sweep,
    als_solve,
)
from committor.tensor_train import TTTrain, MPOTrain, mpo_inner, tt_inner
from committor.assembly import assemble_mpo_rank1, assemble_hb_tt
from committor.problems import build_double_well_nd_problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_problem(d: int = 1, n: int = 6, beta: float = 2.0, rho: float = 10.0):
    """Return (mpo, hb_tt, init_tt) for a small double-well problem."""
    basis, per_dim = build_double_well_nd_problem(d=d, beta=beta, nbasis=n, nquad=200)
    mpo = assemble_mpo_rank1(per_dim, rho)
    hb_tt = assemble_hb_tt(per_dim)
    # Random small-norm initial TT
    rng = np.random.default_rng(0)
    if d == 1:
        init_cores = [rng.standard_normal((1, n, 1)) * 0.1]
    else:
        r = 2
        init_cores = []
        for k in range(d):
            rl = 1 if k == 0 else r
            rr = 1 if k == d - 1 else r
            init_cores.append(rng.standard_normal((rl, n, rr)) * 0.1)
    tt = TTTrain(cores=init_cores)
    return mpo, hb_tt, tt, rho


# ---------------------------------------------------------------------------
# Unit tests for local assemblers
# ---------------------------------------------------------------------------


class TestLocalAssemblers:
    def test_als_local_matrix_shape(self):
        rL, wL, n, wR, rR = 2, 3, 4, 3, 2
        L = np.ones((rL, wL, rL))
        mpo_core = np.ones((wL, n, n, wR))
        R = np.ones((rR, wR, rR))
        M = als_local_matrix(L, mpo_core, R)
        assert M.shape == (rL * n * rR, rL * n * rR)

    def test_als_local_matrix_symmetric_for_symmetric_mpo(self):
        rL, wL, n, wR, rR = 1, 2, 5, 2, 1
        L = np.eye(wL).reshape(rL, wL, rL)
        R = np.eye(wR).reshape(rR, wR, rR)
        # Symmetric MPO core: W[w,i,j,x] = W[w,j,i,x]
        W = np.random.default_rng(0).standard_normal((wL, n, n, wR))
        W = 0.5 * (W + W.transpose(0, 2, 1, 3))  # symmetrize physical indices
        L_eye = np.eye(rL).reshape(rL, wL, rL)
        M = als_local_matrix(L_eye, W, np.eye(rR).reshape(rR, wR, rR))
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_als_local_rhs_shape(self):
        rL_Q, rL_hB = 2, 3
        n = 4
        rR_Q, rR_hB = 2, 3
        L = np.ones((rL_Q, rL_hB))
        rhs_core = np.ones((rL_hB, n, rR_hB))
        R = np.ones((rR_Q, rR_hB))
        f = als_local_rhs(L, rhs_core, R)
        assert f.shape == (rL_Q * n * rR_Q,)

    def test_core_roundtrip(self):
        core = np.random.default_rng(0).standard_normal((3, 5, 2))
        vec = als_core_to_vec(core)
        assert vec.shape == (3 * 5 * 2,)
        restored = als_vec_to_core(vec, 3, 5, 2)
        np.testing.assert_array_equal(restored, core)


# ---------------------------------------------------------------------------
# ALS sweep tests (1D is simplest to check against dense solve)
# ---------------------------------------------------------------------------


class TestALSSweeps:
    def test_left_right_sweep_returns_tt(self):
        mpo, hb_tt, tt, rho = _make_simple_problem(d=1, n=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_tt, obj = als_left_to_right_sweep(tt, mpo, hb_tt, rho)
        assert isinstance(result_tt, TTTrain)
        assert isinstance(obj, float)

    def test_right_left_sweep_returns_tt(self):
        mpo, hb_tt, tt, rho = _make_simple_problem(d=1, n=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_tt, obj = als_right_to_left_sweep(tt, mpo, hb_tt, rho)
        assert isinstance(result_tt, TTTrain)
        assert isinstance(obj, float)


# ---------------------------------------------------------------------------
# als_solve: convergence and monotone decrease
# ---------------------------------------------------------------------------


class TestALSSolve:
    def test_objective_decreases_bidirectional_1d(self):
        """Bidirectional ALS must monotonically decrease the objective in 1D."""
        mpo, hb_tt, tt_init, rho = _make_simple_problem(d=1, n=8, rho=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tt, history = als_solve(
                tt_init, mpo, hb_tt, rho,
                n_sweeps=10, tol=0.0, verbose=False, bidirectional=True,
            )
        slack = 1e-10 * (1.0 + abs(history[0]))
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + slack, (
                f"Objective increased at sweep {i}: "
                f"J[{i-1}]={history[i-1]:.10g}, J[{i}]={history[i]:.10g}"
            )

    def test_objective_decreases_unidirectional_1d(self):
        """Unidirectional ALS must also decrease the objective in 1D."""
        mpo, hb_tt, tt_init, rho = _make_simple_problem(d=1, n=8, rho=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tt, history = als_solve(
                tt_init, mpo, hb_tt, rho,
                n_sweeps=20, tol=0.0, verbose=False, bidirectional=False,
            )
        slack = 1e-10 * (1.0 + abs(history[0]))
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + slack

    def test_bidirectional_and_unidirectional_converge_similarly(self):
        """Both directions should converge to roughly the same minimum value."""
        mpo, hb_tt, tt_init, rho = _make_simple_problem(d=1, n=8, rho=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tt_bi,  hist_bi  = als_solve(tt_init, mpo, hb_tt, rho,
                                          n_sweeps=15, tol=0.0, bidirectional=True)
            _tt_uni, hist_uni = als_solve(tt_init, mpo, hb_tt, rho,
                                          n_sweeps=30, tol=0.0, bidirectional=False)
        err = abs(hist_bi[-1] - hist_uni[-1])
        assert err < 1e-6 * (1.0 + abs(hist_bi[-1])), (
            f"Bi and uni finals disagree: {hist_bi[-1]:.8g} vs {hist_uni[-1]:.8g}"
        )

    def test_1d_als_matches_dense_solve(self):
        """In 1D, TT-ALS (rank-1) should match the direct dense linear solve."""
        from committor.solvers import solve_committor_1d
        from committor.problems import double_well_potential

        beta, rho = 2.0, 50.0
        result_dense = solve_committor_1d(
            double_well_potential, beta=beta, a=-2.0, b=2.0,
            nbasis=8, rho=rho, nquad=300
        )
        mpo, hb_tt, tt_init, _ = _make_simple_problem(d=1, n=8, beta=beta, rho=rho)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tt_final, _history = als_solve(
                tt_init, mpo, hb_tt, rho, n_sweeps=30, tol=1e-12
            )
        # Evaluate both at a grid of points
        x_grid = np.linspace(-1.0, 1.0, 15)
        q_dense = result_dense.q(x_grid)
        # TT result needs basis evaluation
        from committor.problems import build_double_well_nd_problem
        from committor.tensor_train import tt_evaluate
        basis, _ = build_double_well_nd_problem(d=1, beta=beta, nbasis=8, nquad=300)
        q_tt = tt_evaluate(tt_final, basis, x_grid.reshape(-1, 1))
        np.testing.assert_allclose(q_tt, q_dense, atol=1e-5)

    def test_convergence_tol_stops_early(self):
        """Setting tol > 0 should terminate before max sweeps when converged."""
        mpo, hb_tt, tt_init, rho = _make_simple_problem(d=1, n=8, rho=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tt, history = als_solve(
                tt_init, mpo, hb_tt, rho, n_sweeps=200, tol=1e-6
            )
        # Should converge well before 200 sweeps
        assert len(history) < 200

    def test_invalid_arguments(self):
        mpo, hb_tt, tt, rho = _make_simple_problem(d=1, n=4)
        with pytest.raises(ValueError):
            als_solve(tt, mpo, hb_tt, rho, n_sweeps=0)
        with pytest.raises(ValueError):
            als_solve(tt, mpo, hb_tt, rho, tol=-1.0)

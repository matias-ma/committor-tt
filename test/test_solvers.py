"""Tests for committor.solvers — high-level solver entry points."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from committor.solvers import (
    Committor1DResult,
    CommittorNDDenseResult,
    CommittorTTResult,
    solve_committor_1d,
    solve_committor_nd_dense,
    solve_committor_nd_tt,
)
from committor.problems import (
    double_well_potential,
    exact_committor_1d,
    build_double_well_nd_problem,
    build_double_well_nd_problem_weighted,
    relative_error_1d,
    relative_error_nd_mc,
    lift_to_d_dimensions,
)
from committor._types import CommittorResult


# ---------------------------------------------------------------------------
# solve_committor_1d
# ---------------------------------------------------------------------------


class TestSolveCommittor1D:
    def _default_result(self, nbasis=20):
        return solve_committor_1d(
            V=double_well_potential, beta=5.0,
            a=-2.0, b=2.0, nbasis=nbasis, rho=400.0, nquad=300,
        )

    def test_returns_correct_type(self):
        result = self._default_result()
        assert isinstance(result, Committor1DResult)
        assert isinstance(result, CommittorResult)

    def test_q_shape(self):
        result = self._default_result()
        x = np.linspace(-1.0, 1.0, 50)
        q_vals = result.q(x)
        assert q_vals.shape == (50,)

    def test_dq_shape(self):
        result = self._default_result()
        x = np.linspace(-1.0, 1.0, 20)
        dq_vals = result.dq(x)
        assert dq_vals.shape == (20,)

    def test_boundary_values(self):
        """q should be near 0 at x=-1 and near 1 at x=+1."""
        result = self._default_result()
        assert abs(result.q(np.array([-1.0]))[0]) < 0.02
        assert abs(result.q(np.array([1.0]))[0] - 1.0) < 0.02

    def test_monotone_on_axis(self):
        """The committor should be monotone increasing in x_1."""
        result = self._default_result()
        x = np.linspace(-0.95, 0.95, 50)
        q_vals = result.q(x)
        diffs = np.diff(q_vals)
        assert np.all(diffs > -1e-4), "committor is not monotone"

    def test_accuracy_vs_exact(self):
        """Relative L2 error vs exact ODE solution should be < 1%."""
        result = self._default_result(nbasis=30)
        q_true = exact_committor_1d(double_well_potential, beta=5.0)
        err = relative_error_1d(
            q_approx=result.q, q_true=q_true,
            V=double_well_potential, beta=5.0,
            left=-1.0, right=1.0, nquad=5000,
        )
        assert err < 0.01, f"Relative error {err:.4f} exceeds 1%"

    def test_bad_interval(self):
        with pytest.raises(ValueError, match="a < b"):
            solve_committor_1d(double_well_potential, beta=5.0, a=1.0, b=-1.0)

    def test_bad_rho(self):
        with pytest.raises(ValueError):
            solve_committor_1d(double_well_potential, beta=5.0, a=-2.0, b=2.0, rho=-1.0)

    def test_out_of_range_warning(self):
        result = self._default_result()
        with pytest.warns(UserWarning, match="outside the basis interval"):
            result.q(np.array([-3.0, 3.0]))

    def test_lift_to_d_dimensions(self):
        """Lifting a 1D committor to 3D should only use the first column."""
        result = self._default_result()
        q_nd = lift_to_d_dimensions(result.q)
        X = np.array([[0.0, 1.0, 2.0],
                      [0.5, -1.0, 3.0]])
        q_vals = q_nd(X)
        expected = result.q(np.array([0.0, 0.5]))
        np.testing.assert_allclose(q_vals, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# solve_committor_nd_dense (small d)
# ---------------------------------------------------------------------------


class TestSolveCommittorNDDense:
    def _problem(self, d=2, n=6):
        from committor.basis import tensor_product_legendre_basis
        from committor.problems import double_well_nd_measure_fns
        basis = tensor_product_legendre_basis([n] * d, [(-2.0, 2.0)] * d)
        wfns, wA_fns, wB_fns = double_well_nd_measure_fns(d, beta=2.0)
        return basis, wfns, wA_fns, wB_fns

    def test_returns_correct_type(self):
        basis, wfns, wA_fns, wB_fns = self._problem()
        result = solve_committor_nd_dense(basis, wfns, wA_fns, wB_fns, rho=50.0, nquad=100)
        assert isinstance(result, CommittorNDDenseResult)
        assert isinstance(result, CommittorResult)

    def test_q_shape_2d(self):
        basis, wfns, wA_fns, wB_fns = self._problem(d=2, n=5)
        result = solve_committor_nd_dense(basis, wfns, wA_fns, wB_fns, rho=50.0, nquad=100)
        X = np.random.default_rng(0).uniform(-1.5, 1.5, size=(20, 2))
        q_vals = result.q(X)
        assert q_vals.shape == (20,)

    def test_q_range_roughly_0_1(self):
        basis, wfns, wA_fns, wB_fns = self._problem(d=2, n=5)
        result = solve_committor_nd_dense(basis, wfns, wA_fns, wB_fns, rho=50.0, nquad=100)
        X = np.random.default_rng(1).uniform(-1.0, 1.0, size=(100, 2))
        q_vals = result.q(X)
        assert np.all(q_vals > -0.1)
        assert np.all(q_vals < 1.1)

    def test_1d_dense_matches_1d_solver(self):
        """Dense nd solver at d=1 should agree with solve_committor_1d."""
        from committor.basis import tensor_product_legendre_basis
        from committor.problems import double_well_nd_measure_fns

        n, beta, rho = 15, 5.0, 400.0
        basis = tensor_product_legendre_basis([n], [(-2.0, 2.0)])
        wfns, wA_fns, wB_fns = double_well_nd_measure_fns(1, beta=beta)
        result_nd = solve_committor_nd_dense(basis, wfns, wA_fns, wB_fns,
                                             rho=rho, nquad=300)
        result_1d = solve_committor_1d(double_well_potential, beta=beta,
                                       a=-2.0, b=2.0, nbasis=n, rho=rho, nquad=300)
        x = np.linspace(-1.0, 1.0, 15)
        q_nd = result_nd.q(x.reshape(-1, 1))
        q_1d = result_1d.q(x)
        np.testing.assert_allclose(q_nd, q_1d, atol=1e-8)

    def test_bad_rho(self):
        basis, wfns, wA_fns, wB_fns = self._problem()
        with pytest.raises(ValueError):
            solve_committor_nd_dense(basis, wfns, wA_fns, wB_fns, rho=-1.0)

    def test_wrong_x_shape(self):
        basis, wfns, wA_fns, wB_fns = self._problem(d=2)
        result = solve_committor_nd_dense(basis, wfns, wA_fns, wB_fns, rho=50.0, nquad=100)
        with pytest.raises(ValueError):
            result.q(np.ones((5, 3)))  # wrong d


# ---------------------------------------------------------------------------
# solve_committor_nd_tt (TT-ALS, small d smoke tests)
# ---------------------------------------------------------------------------


class TestSolveCommittorNDTT:
    def test_1d_returns_correct_type(self):
        basis, per_dim = build_double_well_nd_problem(d=1, beta=5.0, nbasis=6, nquad=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_committor_nd_tt(per_dim, basis, rho=50.0, tt_rank=1,
                                           n_sweeps=10, verbose=False)
        assert isinstance(result, CommittorTTResult)
        assert isinstance(result, CommittorResult)

    def test_1d_q_shape(self):
        basis, per_dim = build_double_well_nd_problem(d=1, beta=5.0, nbasis=6, nquad=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_committor_nd_tt(per_dim, basis, rho=50.0, tt_rank=1,
                                           n_sweeps=10, verbose=False)
        x = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)
        q_vals = result.q(x)
        assert q_vals.shape == (10,)

    def test_2d_smoke(self):
        """Smoke test: 2D TT-ALS completes without error."""
        basis, per_dim = build_double_well_nd_problem(d=2, beta=2.0, nbasis=5, nquad=80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_committor_nd_tt(per_dim, basis, rho=20.0, tt_rank=2,
                                           n_sweeps=5, verbose=False)
        X = np.random.default_rng(0).uniform(-1.5, 1.5, size=(20, 2))
        q_vals = result.q(X)
        assert q_vals.shape == (20,)
        assert np.all(np.isfinite(q_vals))

    def test_tt_rank_1_agrees_with_dense_1d(self):
        """TT-ALS with rank-1 in 1D should reproduce the dense solve."""
        beta, rho, n = 5.0, 400.0, 10
        basis, per_dim = build_double_well_nd_problem(d=1, beta=beta, nbasis=n, nquad=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_tt = solve_committor_nd_tt(
                per_dim, basis, rho=rho, tt_rank=1, n_sweeps=50, tol=1e-10
            )
        result_1d = solve_committor_1d(
            double_well_potential, beta=beta, a=-2.0, b=2.0, nbasis=n,
            rho=rho, nquad=300
        )
        x = np.linspace(-0.9, 0.9, 20)
        q_tt = result_tt.q(x.reshape(-1, 1))
        q_1d = result_1d.q(x)
        np.testing.assert_allclose(q_tt, q_1d, atol=1e-4)

    def test_nd_accuracy_d5(self):
        """For d=5, double-well TT-ALS should achieve < 5% relative error."""
        beta, rho, d, n = 5.0, 400.0, 5, 15
        basis, per_dim = build_double_well_nd_problem_weighted(
            d=d, beta=beta, nbasis=n, nquad=300
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_committor_nd_tt(per_dim, basis, rho=rho, tt_rank=4,
                                           n_sweeps=30, tol=1e-8, verbose=False)
        q_true_1d = exact_committor_1d(double_well_potential, beta=beta)
        err = relative_error_nd_mc(
            result, q_true_1d, beta=beta, n_samples=10_000, seed=99
        )
        assert err < 0.05, f"Relative error {err:.4f} exceeds 5%"

    def test_bad_rho(self):
        basis, per_dim = build_double_well_nd_problem(d=1, beta=2.0, nbasis=4, nquad=50)
        with pytest.raises(ValueError):
            solve_committor_nd_tt(per_dim, basis, rho=-1.0)

    def test_bad_tt_rank(self):
        basis, per_dim = build_double_well_nd_problem(d=1, beta=2.0, nbasis=4, nquad=50)
        with pytest.raises(ValueError):
            solve_committor_nd_tt(per_dim, basis, rho=10.0, tt_rank=0)

    def test_wrong_x_shape(self):
        basis, per_dim = build_double_well_nd_problem(d=2, beta=2.0, nbasis=4, nquad=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solve_committor_nd_tt(per_dim, basis, rho=10.0, tt_rank=1, n_sweeps=2)
        with pytest.raises(ValueError):
            result.q(np.ones((5, 3)))  # wrong d
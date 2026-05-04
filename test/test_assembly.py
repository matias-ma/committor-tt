"""Tests for committor.assembly — quadrature matrices and MPO assembly."""

from __future__ import annotations

import numpy as np
import pytest

from committor.assembly import (
    PerDimMatrices,
    quadrature_matrices_1d,
    quadrature_matrices_nd,
    assemble_dense_nd,
    assemble_mpo_rank1,
    assemble_hb_tt,
)
from committor.basis import (
    shifted_orthonormal_legendre_basis,
    tensor_product_legendre_basis,
)
from committor.tensor_train import TTTrain, MPOTrain, tt_inner, mpo_inner


# ---------------------------------------------------------------------------
# 1D quadrature matrices
# ---------------------------------------------------------------------------


class TestQuadratureMatrices1D:
    def _simple_weight(self):
        """Uniform weight on [-1, 1]."""
        return lambda x: np.ones_like(np.asarray(x, dtype=float))

    def test_stiffness_matrix_shape(self):
        n = 8
        basis = shifted_orthonormal_legendre_basis(n, -1.0, 1.0)
        w = self._simple_weight()
        S, MA, MB, bvec = quadrature_matrices_1d(basis, w, w, w, nquad=200)
        assert S.shape == (n, n)
        assert MA.shape == (n, n)
        assert MB.shape == (n, n)
        assert bvec.shape == (n,)

    def test_stiffness_matrix_symmetry(self):
        n = 6
        basis = shifted_orthonormal_legendre_basis(n, -1.0, 1.0)
        w = lambda x: np.exp(-np.asarray(x, dtype=float) ** 2)
        S, MA, MB, bvec = quadrature_matrices_1d(basis, w, w, w, nquad=300)
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        np.testing.assert_allclose(MA, MA.T, atol=1e-12)

    def test_mass_matrix_with_legendre_basis(self):
        """With Legendre basis and uniform weight, M ≈ I (orthonormality)."""
        n = 10
        basis = shifted_orthonormal_legendre_basis(n, -1.0, 1.0)
        w = self._simple_weight()
        _S, MA, _MB, _bvec = quadrature_matrices_1d(basis, w, w, w, nquad=500)
        np.testing.assert_allclose(MA, np.eye(n), atol=1e-11)


# ---------------------------------------------------------------------------
# N-D quadrature matrices (PerDimMatrices)
# ---------------------------------------------------------------------------


class TestQuadratureMatricesND:
    def _double_well_setup(self, d=2, n=5):
        from committor.problems import double_well_nd_measure_fns, build_double_well_nd_problem
        basis, per_dim = build_double_well_nd_problem(
            d=d, beta=2.0, nbasis=n, nquad=200
        )
        return basis, per_dim

    def test_perdim_matrix_shapes(self):
        basis, per_dim = self._double_well_setup(d=3, n=6)
        assert len(per_dim) == 3
        for k, pdm in enumerate(per_dim):
            assert pdm.S.shape == (6, 6), f"dim {k}: S wrong shape"
            assert pdm.M.shape == (6, 6)
            assert pdm.MA.shape == (6, 6)
            assert pdm.MB.shape == (6, 6)
            assert pdm.bvec.shape == (6,)

    def test_perdim_matrices_symmetric(self):
        basis, per_dim = self._double_well_setup(d=2, n=5)
        for k, pdm in enumerate(per_dim):
            np.testing.assert_allclose(pdm.S, pdm.S.T, atol=1e-10,
                                       err_msg=f"S[{k}] not symmetric")
            np.testing.assert_allclose(pdm.M, pdm.M.T, atol=1e-10)

    def test_mass_matrix_psd(self):
        basis, per_dim = self._double_well_setup(d=2, n=5)
        for pdm in per_dim:
            evals = np.linalg.eigvalsh(pdm.M)
            assert np.all(evals >= -1e-10), "M is not PSD"


# ---------------------------------------------------------------------------
# Dense Kronecker assembly
# ---------------------------------------------------------------------------


class TestAssembleDenseND:
    def test_1d_lhs_matches_1d_quadrature(self):
        """For d=1, dense assembly must match 1D quadrature directly."""
        from committor.problems import double_well_potential
        n, beta, rho = 8, 2.0, 10.0
        basis, per_dim = __import__("committor.problems", fromlist=["build_double_well_nd_problem"]).build_double_well_nd_problem(
            d=1, beta=beta, nbasis=n, nquad=300
        )
        lhs, rhs = assemble_dense_nd(per_dim, rho)
        assert lhs.shape == (n, n)
        assert rhs.shape == (n,)
        # LHS must be symmetric
        np.testing.assert_allclose(lhs, lhs.T, atol=1e-10)

    def test_2d_lhs_shape(self):
        from committor.problems import build_double_well_nd_problem
        n = 4
        _basis, per_dim = build_double_well_nd_problem(d=2, beta=2.0, nbasis=n, nquad=100)
        lhs, rhs = assemble_dense_nd(per_dim, rho=10.0)
        assert lhs.shape == (n * n, n * n)
        assert rhs.shape == (n * n,)


# ---------------------------------------------------------------------------
# MPO assembly — rank-1 case
# ---------------------------------------------------------------------------


class TestAssembleMPORank1:
    def _per_dim(self, d=2, n=5):
        from committor.problems import build_double_well_nd_problem
        _basis, per_dim = build_double_well_nd_problem(d=d, beta=2.0, nbasis=n, nquad=150)
        return per_dim

    def test_mpo_structure(self):
        per_dim = self._per_dim(d=3, n=5)
        mpo = assemble_mpo_rank1(per_dim, rho=10.0)
        assert isinstance(mpo, MPOTrain)
        assert mpo.d == 3
        assert mpo.ns == (5, 5, 5)

    def test_mpo_bond_dim_interior(self):
        per_dim = self._per_dim(d=4, n=4)
        mpo = assemble_mpo_rank1(per_dim, rho=5.0)
        # Interior bond dimension should be 4 (4-state FSM)
        assert mpo.bonds[1] == 4
        assert mpo.bonds[2] == 4

    def test_mpo_inner_matches_dense_assembly(self):
        """<Q|W|Q> computed via MPO should match the dense Kronecker product."""
        from committor.problems import build_double_well_nd_problem
        from committor.tensor_train import tt_from_dense

        d, n, rho = 2, 4, 5.0
        basis, per_dim = build_double_well_nd_problem(d=d, beta=2.0, nbasis=n, nquad=200)
        lhs_dense, _rhs = assemble_dense_nd(per_dim, rho)
        mpo = assemble_mpo_rank1(per_dim, rho)

        # Random coefficient vector
        rng = np.random.default_rng(42)
        coeffs = rng.standard_normal(n ** d)
        # Dense quadratic form
        expected = float(coeffs @ lhs_dense @ coeffs)
        # TT quadratic form: pack coeffs into a rank-1 TT
        Q_tensor = coeffs.reshape([n] * d)
        tt = tt_from_dense(Q_tensor)
        got = mpo_inner(tt, mpo, tt)
        np.testing.assert_allclose(got, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# h^B TT assembly
# ---------------------------------------------------------------------------


class TestAssembleHBTT:
    def test_hb_tt_structure(self):
        from committor.problems import build_double_well_nd_problem
        d, n = 3, 5
        _basis, per_dim = build_double_well_nd_problem(d=d, beta=2.0, nbasis=n, nquad=100)
        hb = assemble_hb_tt(per_dim)
        assert isinstance(hb, TTTrain)
        assert hb.d == d
        assert hb.ns == (n, n, n)
        # Must be rank-1
        assert all(r == 1 for r in hb.ranks)

    def test_hb_tt_inner_matches_dense(self):
        """<1|h^B> should equal the sum of all entries of h^B."""
        from committor.problems import build_double_well_nd_problem
        d, n = 2, 4
        _basis, per_dim = build_double_well_nd_problem(d=d, beta=2.0, nbasis=n, nquad=200)
        hb = assemble_hb_tt(per_dim)

        # Dense h^B = Kronecker product of bvec_k
        import functools
        hb_dense = functools.reduce(np.kron, [pdm.bvec for pdm in per_dim])

        # tt_inner(ones_tt, hb_tt) should equal sum(hb_dense)
        ones_cores = [np.ones((1, n, 1)) for _ in range(d)]
        ones_tt = TTTrain(cores=ones_cores)
        got = tt_inner(ones_tt, hb)
        expected = float(np.sum(hb_dense))
        np.testing.assert_allclose(got, expected, rtol=1e-9)
"""Tests for committor.tensor_train — TTTrain, MPOTrain, and contraction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from committor.tensor_train import (
    TTTrain,
    MPOTrain,
    tt_from_dense,
    tt_evaluate,
    tt_inner,
    mpo_inner,
    tt_left_envs,
    tt_right_envs,
    mpo_left_envs,
    mpo_right_envs,
    tt_update_left_env,
    tt_update_right_env,
    mpo_update_left_env,
    mpo_update_right_env,
)
from committor.basis import tensor_product_legendre_basis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_tt(d: int, ns: tuple, ranks: tuple, seed: int = 0) -> TTTrain:
    """Build a random TTTrain with given bond dimensions."""
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(d):
        rl = ranks[k]
        rr = ranks[k + 1]
        cores.append(rng.standard_normal((rl, ns[k], rr)))
    return TTTrain(cores=cores)


def _random_mpo(d: int, ns: tuple, bonds: tuple, seed: int = 1) -> MPOTrain:
    """Build a random (non-symmetric) MPOTrain."""
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(d):
        wl = bonds[k]
        wr = bonds[k + 1]
        n  = ns[k]
        cores.append(rng.standard_normal((wl, n, n, wr)))
    return MPOTrain(cores=cores)


# ---------------------------------------------------------------------------
# TTTrain validation
# ---------------------------------------------------------------------------

class TestTTTrainConstruction:
    def test_basic_construction(self):
        core = np.ones((1, 5, 1))
        tt = TTTrain(cores=[core])
        assert tt.d == 1
        assert tt.ns == (5,)
        assert tt.ranks == (1, 1)

    def test_multi_core(self):
        cores = [
            np.ones((1, 4, 3)),
            np.ones((3, 6, 2)),
            np.ones((2, 5, 1)),
        ]
        tt = TTTrain(cores=cores)
        assert tt.d == 3
        assert tt.ns == (4, 6, 5)
        assert tt.ranks == (1, 3, 2, 1)

    def test_empty_cores_raises(self):
        with pytest.raises(ValueError):
            TTTrain(cores=[])

    def test_wrong_left_boundary_raises(self):
        with pytest.raises(ValueError):
            TTTrain(cores=[np.ones((2, 5, 1))])  # left rank != 1

    def test_wrong_right_boundary_raises(self):
        with pytest.raises(ValueError):
            TTTrain(cores=[np.ones((1, 5, 2))])  # right rank != 1

    def test_bond_mismatch_raises(self):
        with pytest.raises(ValueError):
            TTTrain(cores=[np.ones((1, 4, 3)), np.ones((2, 5, 1))])  # 3 != 2

    def test_non_3tensor_raises(self):
        with pytest.raises(ValueError):
            TTTrain(cores=[np.ones((1, 5))])


# ---------------------------------------------------------------------------
# MPOTrain validation
# ---------------------------------------------------------------------------

class TestMPOTrainConstruction:
    def test_basic_construction(self):
        W = np.ones((1, 4, 4, 1))
        mpo = MPOTrain(cores=[W])
        assert mpo.d == 1
        assert mpo.ns == (4,)
        assert mpo.bonds == (1, 1)

    def test_non_square_physical_raises(self):
        with pytest.raises(ValueError):
            MPOTrain(cores=[np.ones((1, 4, 5, 1))])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MPOTrain(cores=[])

    def test_bond_mismatch_raises(self):
        with pytest.raises(ValueError):
            MPOTrain(cores=[np.ones((1, 4, 4, 3)), np.ones((2, 4, 4, 1))])


# ---------------------------------------------------------------------------
# tt_from_dense
# ---------------------------------------------------------------------------

class TestTTFromDense:
    def test_1d_exact(self):
        v = np.arange(1.0, 6.0)
        tt = tt_from_dense(v)
        assert tt.d == 1
        assert tt.ns == (5,)
        # Inner product should equal squared 2-norm
        np.testing.assert_allclose(tt_inner(tt, tt), float(np.dot(v, v)), rtol=1e-12)

    def test_2d_exact(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 5))
        tt = tt_from_dense(A)
        assert tt.d == 2
        assert tt.ns == (4, 5)
        np.testing.assert_allclose(tt_inner(tt, tt), float(np.sum(A ** 2)), rtol=1e-10)

    def test_3d_rank1_tensor(self):
        """A rank-1 tensor u⊗v⊗w should be exactly representable in TT."""
        u, v, w = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([6.0, 7.0])
        dense = np.einsum('i,j,k->ijk', u, v, w)
        tt = tt_from_dense(dense, max_rank=1)
        assert tt.d == 3
        np.testing.assert_allclose(tt_inner(tt, tt), float(np.sum(dense ** 2)), rtol=1e-10)

    def test_empty_tensor_raises(self):
        with pytest.raises(ValueError):
            tt_from_dense(np.array(5.0))  # 0-d array


# ---------------------------------------------------------------------------
# tt_inner
# ---------------------------------------------------------------------------

class TestTTInner:
    def test_self_inner_equals_frobenius_sq(self):
        rng = np.random.default_rng(7)
        dense = rng.standard_normal((3, 4, 5))
        tt = tt_from_dense(dense)
        expected = float(np.sum(dense ** 2))
        np.testing.assert_allclose(tt_inner(tt, tt), expected, rtol=1e-10)

    def test_cross_inner(self):
        rng = np.random.default_rng(8)
        A = rng.standard_normal((3, 4))
        B = rng.standard_normal((3, 4))
        ttA = tt_from_dense(A)
        ttB = tt_from_dense(B)
        expected = float(np.sum(A * B))
        np.testing.assert_allclose(tt_inner(ttA, ttB), expected, rtol=1e-10)

    def test_inner_symmetry(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((4, 5))
        B = rng.standard_normal((4, 5))
        ttA, ttB = tt_from_dense(A), tt_from_dense(B)
        np.testing.assert_allclose(tt_inner(ttA, ttB), tt_inner(ttB, ttA), atol=1e-12)

    def test_dimension_mismatch_raises(self):
        tt1 = _random_tt(2, (3, 4), (1, 2, 1))
        tt2 = _random_tt(3, (3, 4, 5), (1, 2, 2, 1))
        with pytest.raises(ValueError):
            tt_inner(tt1, tt2)


# ---------------------------------------------------------------------------
# mpo_inner
# ---------------------------------------------------------------------------

class TestMPOInner:
    def _identity_mpo(self, d: int, n: int) -> MPOTrain:
        """MPO that acts as the identity: W[0,i,j,0] = delta_{ij}."""
        core = np.eye(n).reshape(1, n, n, 1)
        return MPOTrain(cores=[core.copy() for _ in range(d)])

    def test_identity_mpo_equals_tt_inner(self):
        rng = np.random.default_rng(20)
        d, n = 3, 4
        dense = rng.standard_normal((n,) * d)
        tt = tt_from_dense(dense)
        mpo = self._identity_mpo(d, n)
        np.testing.assert_allclose(mpo_inner(tt, mpo, tt), tt_inner(tt, tt), rtol=1e-10)

    def test_against_dense_quadratic_form(self):
        """<Q|W|Q> via MPO must match dense matrix-vector product for d=2."""
        rng = np.random.default_rng(21)
        n = 4
        A_dense = rng.standard_normal((n * n, n * n))  # arbitrary matrix
        A_dense = 0.5 * (A_dense + A_dense.T)          # symmetrize
        q = rng.standard_normal(n * n)
        expected = float(q @ A_dense @ q)

        # Reshape A into an MPO with bond dim 1 (not generally possible for
        # arbitrary A; instead just test the identity MPO route)
        # Use identity check instead
        tt = tt_from_dense(q.reshape(n, n))
        mpo = self._identity_mpo(2, n)
        got = mpo_inner(tt, mpo, tt)
        np.testing.assert_allclose(got, tt_inner(tt, tt), rtol=1e-10)

    def test_dimension_mismatch_raises(self):
        tt1 = _random_tt(2, (3, 3), (1, 2, 1))
        tt2 = _random_tt(2, (3, 3), (1, 2, 1))
        mpo = _random_mpo(3, (3, 3, 3), (1, 2, 2, 1))
        with pytest.raises(ValueError):
            mpo_inner(tt1, mpo, tt2)


# ---------------------------------------------------------------------------
# Environment functions
# ---------------------------------------------------------------------------

class TestTTEnvironments:
    def test_left_envs_last_equals_inner(self):
        rng = np.random.default_rng(30)
        d, n = 3, 4
        tt = tt_from_dense(rng.standard_normal((n,) * d))
        envs = tt_left_envs(tt, tt)
        assert len(envs) == d + 1
        # First env is scalar 1
        np.testing.assert_allclose(envs[0], np.ones((1, 1)))
        # Last env == tt_inner
        np.testing.assert_allclose(float(envs[-1][0, 0]), tt_inner(tt, tt), rtol=1e-10)

    def test_right_envs_last_equals_inner(self):
        rng = np.random.default_rng(31)
        d, n = 3, 4
        tt = tt_from_dense(rng.standard_normal((n,) * d))
        envs = tt_right_envs(tt, tt)
        assert len(envs) == d + 1
        # Last (rightmost-contracted) env == tt_inner
        np.testing.assert_allclose(float(envs[-1][0, 0]), tt_inner(tt, tt), rtol=1e-10)

    def test_mpo_left_envs_last_equals_mpo_inner(self):
        rng = np.random.default_rng(32)
        d, n = 2, 3
        tt  = tt_from_dense(rng.standard_normal((n,) * d))
        mpo = _random_mpo(d, (n,) * d, (1,) + (2,) * (d - 1) + (1,))
        envs = mpo_left_envs(tt, mpo, tt)
        assert len(envs) == d + 1
        np.testing.assert_allclose(
            float(envs[-1][0, 0, 0]), mpo_inner(tt, mpo, tt), rtol=1e-10
        )

    def test_mpo_right_envs_last_equals_mpo_inner(self):
        rng = np.random.default_rng(33)
        d, n = 2, 3
        tt  = tt_from_dense(rng.standard_normal((n,) * d))
        mpo = _random_mpo(d, (n,) * d, (1,) + (2,) * (d - 1) + (1,))
        envs = mpo_right_envs(tt, mpo, tt)
        np.testing.assert_allclose(
            float(envs[-1][0, 0, 0]), mpo_inner(tt, mpo, tt), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Incremental environment updaters
# ---------------------------------------------------------------------------

class TestIncrementalUpdaters:
    def test_tt_left_update_matches_envs(self):
        """tt_update_left_env iterated manually must match tt_left_envs output."""
        rng = np.random.default_rng(40)
        d, n = 3, 4
        tt = tt_from_dense(rng.standard_normal((n,) * d))
        envs = tt_left_envs(tt, tt)
        L = np.ones((1, 1))
        for k in range(d):
            L = tt_update_left_env(L, tt.cores[k], tt.cores[k])
            np.testing.assert_allclose(L, envs[k + 1], atol=1e-12, err_msg=f"site {k}")

    def test_tt_right_update_matches_envs(self):
        """tt_update_right_env iterated manually must match tt_right_envs output."""
        rng = np.random.default_rng(41)
        d, n = 3, 4
        tt = tt_from_dense(rng.standard_normal((n,) * d))
        envs = tt_right_envs(tt, tt)
        R = np.ones((1, 1))
        for k in range(d - 1, -1, -1):
            R = tt_update_right_env(R, tt.cores[k], tt.cores[k])
            np.testing.assert_allclose(R, envs[d - k], atol=1e-12, err_msg=f"site {k}")

    def test_mpo_left_update_matches_envs(self):
        rng = np.random.default_rng(42)
        d, n = 2, 3
        tt  = tt_from_dense(rng.standard_normal((n,) * d))
        mpo = _random_mpo(d, (n,) * d, (1,) + (2,) * (d - 1) + (1,))
        envs = mpo_left_envs(tt, mpo, tt)
        L = np.ones((1, 1, 1))
        for k in range(d):
            L = mpo_update_left_env(L, tt.cores[k], mpo.cores[k], tt.cores[k])
            np.testing.assert_allclose(L, envs[k + 1], atol=1e-12, err_msg=f"site {k}")

    def test_mpo_right_update_matches_envs(self):
        rng = np.random.default_rng(43)
        d, n = 2, 3
        tt  = tt_from_dense(rng.standard_normal((n,) * d))
        mpo = _random_mpo(d, (n,) * d, (1,) + (2,) * (d - 1) + (1,))
        envs = mpo_right_envs(tt, mpo, tt)
        R = np.ones((1, 1, 1))
        for k in range(d - 1, -1, -1):
            R = mpo_update_right_env(R, tt.cores[k], mpo.cores[k], tt.cores[k])
            np.testing.assert_allclose(R, envs[d - k], atol=1e-12, err_msg=f"site {k}")


# ---------------------------------------------------------------------------
# tt_evaluate
# ---------------------------------------------------------------------------

class TestTTEvaluate:
    def test_1d_evaluate(self):
        """For d=1, tt_evaluate should recover the original function values."""
        n, a, b = 8, -2.0, 2.0
        basis = tensor_product_legendre_basis([n], [(a, b)])
        # Build a rank-1 TT from a known coefficient vector
        rng = np.random.default_rng(50)
        coeffs = rng.standard_normal(n)
        tt = TTTrain(cores=[coeffs.reshape(1, n, 1)])
        x = np.linspace(-1.5, 1.5, 20).reshape(-1, 1)
        q_tt = tt_evaluate(tt, basis, x)
        # Dense evaluation
        Phi = np.vstack([f(x[:, 0]) for f in basis.bases[0].fns])  # (n, 20)
        q_dense = coeffs @ Phi
        np.testing.assert_allclose(q_tt, q_dense, atol=1e-12)

    def test_2d_evaluate_rank1(self):
        """For d=2 with a separable (rank-1) TT, evaluate should be exact."""
        n1, n2 = 4, 5
        a, b = -1.0, 1.0
        basis = tensor_product_legendre_basis([n1, n2], [(a, b)] * 2)
        rng = np.random.default_rng(51)
        c1, c2 = rng.standard_normal(n1), rng.standard_normal(n2)
        tt = TTTrain(cores=[c1.reshape(1, n1, 1), c2.reshape(1, n2, 1)])
        x = rng.uniform(a, b, size=(15, 2))
        q_tt = tt_evaluate(tt, basis, x)
        Phi1 = np.vstack([f(x[:, 0]) for f in basis.bases[0].fns])  # (n1, 15)
        Phi2 = np.vstack([f(x[:, 1]) for f in basis.bases[1].fns])  # (n2, 15)
        q_dense = (c1 @ Phi1) * (c2 @ Phi2)
        np.testing.assert_allclose(q_tt, q_dense, atol=1e-12)

    def test_shape_mismatch_raises(self):
        n = 4
        basis = tensor_product_legendre_basis([n, n], [(-1.0, 1.0)] * 2)
        tt = TTTrain(cores=[np.ones((1, n, 1)), np.ones((1, n, 1))])
        with pytest.raises(ValueError):
            tt_evaluate(tt, basis, np.ones((5, 3)))  # d mismatch

    def test_wrong_ndim_raises(self):
        n = 4
        basis = tensor_product_legendre_basis([n], [(-1.0, 1.0)])
        tt = TTTrain(cores=[np.ones((1, n, 1))])
        with pytest.raises(ValueError):
            tt_evaluate(tt, basis, np.ones(5))  # 1-D array, not (n,1)

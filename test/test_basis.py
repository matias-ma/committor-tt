"""Tests for committor.basis — univariate and tensor-product basis functions.

These tests are translated from the inline _test_* functions in the original
committor_1d.py, converted to standard pytest format.
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from committor.basis import (
    shifted_orthonormal_legendre_basis,
    fourier_basis,
    tensor_product_legendre_basis,
    density_weighted_orthogonal_basis,
    double_well_density_weighted_basis,
    UnivariateBasis,
    TensorProductBasis,
)


# ---------------------------------------------------------------------------
# UnivariateBasis validation
# ---------------------------------------------------------------------------

class TestUnivariateBasis:
    def test_construction(self):
        basis = shifted_orthonormal_legendre_basis(5, -1.0, 1.0)
        assert isinstance(basis, UnivariateBasis)
        assert basis.n == 5
        assert basis.a == -1.0
        assert basis.b == 1.0
        assert len(basis.fns) == 5
        assert len(basis.dfns) == 5

    def test_bad_interval(self):
        with pytest.raises(ValueError):
            shifted_orthonormal_legendre_basis(5, 1.0, -1.0)

    def test_bad_n(self):
        with pytest.raises(ValueError):
            shifted_orthonormal_legendre_basis(0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Legendre basis orthonormality
# ---------------------------------------------------------------------------

class TestLegendreOrthonormality:
    @pytest.mark.parametrize("n,a,b", [
        (5, -1.0, 1.0),
        (10, -2.0, 2.0),
        (8, 0.0, 3.0),
    ])
    def test_orthonormality(self, n, a, b, tol=1e-12):
        """Gram matrix should be identity to numerical precision."""
        basis = shifted_orthonormal_legendre_basis(n, a, b)
        nquad = 500
        xs_std, ws_std = leggauss(nquad)
        xs = 0.5 * (b - a) * xs_std + 0.5 * (b + a)
        ws = 0.5 * (b - a) * ws_std
        Phi = np.vstack([f(xs) for f in basis.fns])  # (n, nquad)
        G   = (Phi * ws[np.newaxis, :]) @ Phi.T
        err = np.max(np.abs(G - np.eye(n)))
        assert err < tol, f"Gram error {err:.2e} > {tol:.0e} for n={n}, [{a},{b}]"

    def test_derivative_accuracy(self):
        """Analytic derivatives should match finite differences."""
        basis = shifted_orthonormal_legendre_basis(6, -2.0, 2.0)
        x0    = np.linspace(-1.9, 1.9, 50)
        h     = 1e-7
        for i, (f, df) in enumerate(zip(basis.fns, basis.dfns)):
            fd      = (f(x0 + h) - f(x0 - h)) / (2 * h)
            analytic = df(x0)
            assert np.max(np.abs(fd - analytic)) < 1e-5, \
                f"Derivative error too large for basis fn {i}"


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

class TestFourierBasis:
    def test_shape(self):
        basis = fourier_basis(5, 2.6)
        assert basis.n == 5
        assert basis.a == -2.6
        assert basis.b ==  2.6

    def test_bad_n_fns(self):
        with pytest.raises(ValueError):
            fourier_basis(4, 2.6)   # must be odd
        with pytest.raises(ValueError):
            fourier_basis(0, 2.6)

    def test_bad_gamma(self):
        with pytest.raises(ValueError):
            fourier_basis(5, 0.0)

    def test_eval_marginal_shape(self):
        basis = fourier_basis(5, 2.6)
        tp    = TensorProductBasis(bases=(basis,))
        x     = np.linspace(-2.6, 2.6, 50)
        assert tp.eval_marginal(0, x).shape  == (5, 50)
        assert tp.deval_marginal(0, x).shape == (5, 50)

    def test_orthogonality(self, tol=1e-10):
        """Gram matrix should be diag(2γ, γ, γ, …, γ)."""
        n_fns, gamma = 5, 2.6
        basis = fourier_basis(n_fns, gamma)
        nquad = 500
        xs_std, ws_std = leggauss(nquad)
        xs = gamma * xs_std
        ws = gamma * ws_std
        Phi = np.vstack([f(xs) for f in basis.fns])
        G   = Phi @ (ws[:, np.newaxis] * Phi.T)
        G_expected = np.diag([2 * gamma] + [gamma] * (n_fns - 1))
        assert np.max(np.abs(G - G_expected)) < tol

    def test_derivative_accuracy(self):
        basis = fourier_basis(5, 2.6)
        x     = np.linspace(-2.0, 2.0, 100)
        h     = 1e-6
        for i, (f, df) in enumerate(zip(basis.fns, basis.dfns)):
            fd      = (f(x + h) - f(x - h)) / (2 * h)
            analytic = df(x)
            assert np.max(np.abs(fd - analytic)) < 1e-5, \
                f"Fourier derivative error too large for fn {i}"


# ---------------------------------------------------------------------------
# TensorProductBasis
# ---------------------------------------------------------------------------

class TestTensorProductBasis:
    def test_construction(self):
        tp = tensor_product_legendre_basis([10, 8], [(-2.0, 2.0), (-1.0, 1.0)])
        assert tp.d == 2
        assert tp.ns == (10, 8)
        assert tp.lower == (-2.0, -1.0)
        assert tp.upper == ( 2.0,  1.0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            tensor_product_legendre_basis([10], [(-2.0, 2.0), (-1.0, 1.0)])

    def test_empty(self):
        with pytest.raises(ValueError):
            TensorProductBasis(bases=())

    def test_eval_marginal(self):
        tp = tensor_product_legendre_basis([5, 7], [(-1.0, 1.0), (-2.0, 2.0)])
        x0 = np.linspace(-0.9, 0.9, 20)
        x1 = np.linspace(-1.9, 1.9, 20)
        assert tp.eval_marginal(0, x0).shape  == (5, 20)
        assert tp.eval_marginal(1, x1).shape  == (7, 20)
        assert tp.deval_marginal(0, x0).shape == (5, 20)

    def test_out_of_range_k(self):
        tp = tensor_product_legendre_basis([5], [(-1.0, 1.0)])
        with pytest.raises(ValueError):
            tp.eval_marginal(1, np.array([0.0]))


# ---------------------------------------------------------------------------
# Density-weighted basis
# ---------------------------------------------------------------------------

class TestDensityWeightedBasis:
    def test_weighted_orthonormality(self, tol=1e-10):
        """Basis must be orthonormal under the Gaussian weight."""
        n        = 8
        a, b     = -3.0, 3.0
        nquad    = 2000
        weight   = lambda x: np.exp(-np.asarray(x, dtype=float) ** 2)
        uvb      = density_weighted_orthogonal_basis(n, weight, a, b, nquad=nquad)

        xs_std, ws_std = leggauss(nquad)
        x_check  = 0.5 * (b - a) * xs_std + 0.5 * (b + a)
        w_check  = 0.5 * (b - a) * ws_std
        wq_check = w_check * weight(x_check)
        Psi      = np.vstack([f(x_check) for f in uvb.fns])
        G        = (Psi * wq_check[np.newaxis, :]) @ Psi.T
        err      = np.max(np.abs(G - np.eye(n)))
        assert err < tol, f"Weighted Gram error {err:.2e}"

    def test_double_well_basis_smoke(self):
        """double_well_density_weighted_basis should run without errors."""
        uvb = double_well_density_weighted_basis(n=6, k=0, beta=5.0)
        assert uvb.n == 6
        x = np.linspace(-1.9, 1.9, 30)
        vals = np.vstack([f(x) for f in uvb.fns])
        assert vals.shape == (6, 30)
        assert np.all(np.isfinite(vals))
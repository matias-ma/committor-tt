"""Tests for committor.ginzburg_landau — GL kernel, TT density, and solver."""

from __future__ import annotations

import numpy as np
import pytest

from committor.ginzburg_landau import (
    ginzburg_landau_kernel,
    compute_gl_kernel_eigenfunctions,
    compute_gl_tt_cores,
    compute_gl_minimizers,
    solve_gl_committor,
    _gaussian_cheb_cores,
    _assemble_hb_gl,
    _assemble_mpo_gl,
)
from committor.basis import fourier_basis, TensorProductBasis
from committor.tensor_train import TTTrain, MPOTrain
from committor.solvers import CommittorTTResult


# ---------------------------------------------------------------------------
# Transfer kernel
# ---------------------------------------------------------------------------


class TestGinzburgLandauKernel:
    def _default_params(self):
        return dict(lam=0.03, h=1.0 / 51.0, beta=1.0)

    def test_symmetry(self):
        """K(x, y) must equal K(y, x)."""
        p = self._default_params()
        xs = np.linspace(-2.0, 2.0, 10)
        ys = np.linspace(-1.5, 1.5, 10)
        Kxy = ginzburg_landau_kernel(xs, ys, **p)
        Kyx = ginzburg_landau_kernel(ys, xs, **p)
        np.testing.assert_allclose(Kxy, Kyx, atol=1e-14)

    def test_positivity(self):
        p = self._default_params()
        xs = np.linspace(-2.0, 2.0, 20)
        ys = np.linspace(-2.0, 2.0, 20)
        K = ginzburg_landau_kernel(xs, ys, **p)
        assert np.all(K > 0)

    def test_max_on_diagonal(self):
        """For lambda small, the maximum in y should be near x."""
        p = self._default_params()
        x0 = 0.5
        ys = np.linspace(-2.0, 2.0, 200)
        K_row = ginzburg_landau_kernel(x0, ys, **p)
        peak_y = ys[np.argmax(K_row)]
        assert abs(peak_y - x0) < 0.3

    def test_beta_scaling(self):
        """Larger beta should make the kernel more peaked (smaller off-diagonal)."""
        lam, h = 0.03, 1.0 / 51.0
        x0 = 0.5
        y_off = 1.0
        K_low  = ginzburg_landau_kernel(x0, y_off, lam, h, beta=0.5)
        K_high = ginzburg_landau_kernel(x0, y_off, lam, h, beta=4.0)
        assert K_high < K_low


# ---------------------------------------------------------------------------
# Kernel eigenfunctions
# ---------------------------------------------------------------------------


class TestComputeGLKernelEigenfunctions:
    def _compute(self, J=4, nquad=100):
        return compute_gl_kernel_eigenfunctions(
            lam=0.03, h=1.0 / 51.0, R=2.6, J=J, nquad=nquad
        )

    def test_output_shapes(self):
        J, nquad = 4, 80
        xs, ws, eig_fns = compute_gl_kernel_eigenfunctions(
            lam=0.03, h=1.0 / 51.0, R=2.6, J=J, nquad=nquad
        )
        assert xs.shape == (nquad,)
        assert ws.shape == (nquad,)
        assert eig_fns.shape == (J, nquad)

    def test_weights_positive(self):
        xs, ws, _ = self._compute()
        assert np.all(ws > 0)

    def test_quadrature_nodes_in_range(self):
        R = 2.6
        xs, ws, _ = compute_gl_kernel_eigenfunctions(
            lam=0.03, h=1.0 / 51.0, R=R, J=4, nquad=80
        )
        assert np.all(xs >= -R - 1e-12)
        assert np.all(xs <= R + 1e-12)

    def test_mercer_approximation_accuracy(self):
        """sum_j v_j(x)*v_j(y) should approximate K(x,y) to reasonable precision."""
        lam, h, R, J, nquad = 0.03, 1.0 / 51.0, 2.6, 6, 150
        xs, ws, eig_fns = compute_gl_kernel_eigenfunctions(lam, h, R, J, nquad)
        # Reconstruct kernel at a few test pairs
        K_approx = eig_fns.T @ eig_fns  # (nquad, nquad), Mercer approx
        K_exact = ginzburg_landau_kernel(xs[:, None], xs[None, :], lam, h)
        # At the diagonal, the approximation should be good
        diag_err = np.max(np.abs(np.diag(K_approx) - np.diag(K_exact)))
        assert diag_err < 0.01

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match="must all be positive"):
            compute_gl_kernel_eigenfunctions(lam=-0.03, h=0.02, R=2.6, J=4)
        with pytest.raises(ValueError):
            compute_gl_kernel_eigenfunctions(lam=0.03, h=0.02, R=2.6, J=0)


# ---------------------------------------------------------------------------
# GL TT density cores
# ---------------------------------------------------------------------------


class TestComputeGLTTCores:
    def _setup(self, J=4, cheb_N=10, nquad=80):
        lam, h, R = 0.03, 1.0 / 51.0, 2.6
        xs, ws, eig_fns = compute_gl_kernel_eigenfunctions(lam, h, R, J, nquad)
        c_lam = np.exp(-1.0 / (4.0 * lam))
        v0 = np.array([np.interp(0.0, xs, eig_fns[j]) for j in range(J)])
        return xs, ws, eig_fns, c_lam, v0, R

    def test_shape_d1(self):
        xs, ws, eig_fns, c_lam, v0, R = self._setup(J=4, cheb_N=10)
        tt = compute_gl_tt_cores(eig_fns, xs, ws, cheb_N=10, R=R,
                                 c_lam=c_lam, v0=v0, d=1)
        assert tt.d == 1
        assert tt.ns == (11,)
        assert tt.ranks == (1, 1)

    def test_shape_d3(self):
        J = 4
        xs, ws, eig_fns, c_lam, v0, R = self._setup(J=J, cheb_N=8)
        tt = compute_gl_tt_cores(eig_fns, xs, ws, cheb_N=8, R=R,
                                 c_lam=c_lam, v0=v0, d=3)
        assert tt.d == 3
        assert tt.ranks == (1, J, J, 1)
        assert all(n == 9 for n in tt.ns)

    def test_d1_density_accuracy(self):
        """For d=1, the TT must reproduce c_lam * K_trunc(0,x)^2 at quad nodes."""
        J, cheb_N, nquad = 6, 20, 150
        xs, ws, eig_fns, c_lam, v0, R = self._setup(J=J, cheb_N=cheb_N, nquad=nquad)
        tt = compute_gl_tt_cores(eig_fns, xs, ws, cheb_N=cheb_N, R=R,
                                 c_lam=c_lam, v0=v0, d=1)
        # Reconstruct density from Chebyshev coefficients
        core = tt.cores[0][0, :, 0]  # (cheb_N+1,)
        t = xs / R
        cheb = np.empty((cheb_N + 1, nquad))
        cheb[0] = 1.0
        if cheb_N >= 1:
            cheb[1] = t
        for n in range(2, cheb_N + 1):
            cheb[n] = 2.0 * t * cheb[n - 1] - cheb[n - 2]
        p_approx = core @ cheb
        # Reference: c_lam * (sum_j v_j(0) * v_j(xs))^2
        K_trunc = v0 @ eig_fns
        p_ref = c_lam * K_trunc ** 2
        np.testing.assert_allclose(p_approx, p_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# GL minimisers
# ---------------------------------------------------------------------------


class TestComputeGLMinimizers:
    def test_shapes(self):
        U_minus, U_plus = compute_gl_minimizers(d=10, lam=0.03)
        assert U_minus.shape == (10,)
        assert U_plus.shape == (10,)

    def test_antisymmetry(self):
        """U_+ should be approximately -U_-."""
        U_minus, U_plus = compute_gl_minimizers(d=20, lam=0.03)
        np.testing.assert_allclose(U_plus, -U_minus, atol=1e-6)

    def test_interior_close_to_one(self):
        """Interior of U_+ should be close to +1."""
        U_minus, U_plus = compute_gl_minimizers(d=30, lam=0.03)
        interior = U_plus[5:25]
        assert np.all(interior > 0.8)


# ---------------------------------------------------------------------------
# Gaussian Chebyshev cores
# ---------------------------------------------------------------------------


class TestGaussianChebCores:
    def _basis(self, d=2, gamma=2.6):
        fb = fourier_basis(1, gamma)  # 1 function per site (minimal)
        return TensorProductBasis(bases=tuple(fb for _ in range(d)))

    def test_shapes(self):
        d, cheb_N = 3, 10
        basis = self._basis(d)
        mu = np.zeros(d)
        cores = _gaussian_cheb_cores(mu, sigma=0.5, basis=basis, cheb_N=cheb_N, nquad=100)
        assert len(cores) == d
        for c in cores:
            assert c.shape == (1, cheb_N + 1, 1)

    def test_reconstruction_accuracy(self):
        """Reconstructed Gaussian should match the analytic formula."""
        gamma, sigma, cheb_N, nquad = 2.6, 0.3, 30, 200
        basis = self._basis(d=1, gamma=gamma)
        mu = np.array([0.5])
        cores = _gaussian_cheb_cores(mu, sigma, basis, cheb_N, nquad)
        c = cores[0][0, :, 0]
        # Evaluate at a dense grid
        xs_eval = np.linspace(-gamma, gamma, 300)
        t = xs_eval / gamma
        cheb = np.empty((cheb_N + 1, 300))
        cheb[0] = 1.0
        if cheb_N >= 1:
            cheb[1] = t
        for n in range(2, cheb_N + 1):
            cheb[n] = 2.0 * t * cheb[n - 1] - cheb[n - 2]
        p_approx = c @ cheb
        p_exact = np.exp(-0.5 * ((xs_eval - mu[0]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        # Should agree on the central part (away from boundary effects)
        mask = np.abs(xs_eval) < 2.0
        np.testing.assert_allclose(p_approx[mask], p_exact[mask], rtol=0.01)


# ---------------------------------------------------------------------------
# solve_gl_committor — smoke test (tiny d for speed)
# ---------------------------------------------------------------------------


class TestSolveGLCommittor:
    def test_smoke_tiny(self):
        """solve_gl_committor completes on a tiny problem (d=5)."""
        result = solve_gl_committor(
            d=5, lam=0.03, T=8.0,
            gamma=2.6, n_basis=3, cheb_N=10,
            tt_rank=2, nquad=50, rho=10.0,
            n_sweeps=3, J=3, R_ball=2.5,
            seed=0, verbose=False,
        )
        assert isinstance(result, CommittorTTResult)
        assert result.basis.d == 5
        # Evaluate at a random point
        X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(5, 5))
        q_vals = result.q(X)
        assert q_vals.shape == (5,)
        assert np.all(np.isfinite(q_vals))

    def test_wrong_x_shape(self):
        result = solve_gl_committor(
            d=5, lam=0.03, T=8.0, gamma=2.6, n_basis=3, cheb_N=10,
            tt_rank=2, nquad=50, rho=10.0, n_sweeps=2, J=3, seed=0,
        )
        with pytest.raises(ValueError):
            result.q(np.ones((3, 7)))  # d mismatch
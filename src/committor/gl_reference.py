"""Exact (transfer-operator) reference quantities for the discretized
Ginzburg-Landau chain, reusing ``ftde.ginzburg_landau``'s own parametrization
(``ginzburg_landau_log_normalizer`` computes the exact log partition
function the same way).

The chain's energy only couples nearest neighbors, so ANY marginal -- not
just the normalizer -- is exactly computable (up to 1D grid-quadrature
error, not sampling error) via a standard forward/backward transfer-
operator recursion: the same message-passing trick used to get exact edge
marginals of a discrete Markov chain, just with a continuous "spin" at each
site. This gives a genuine ground-truth marginal density to plot and
compare a fitted density against, rather than a held-out-sample histogram
(which is itself just another noisy estimator of the same marginal).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ftde.ginzburg_landau import GL_BETA, GL_LAM


def _default_h(d: int) -> float:
    return 1.0 / (d + 1)


class _TransferOperator:
    """Shared forward/backward message-passing machinery for the GL chain's
    edge kernel ``K(a,b) = exp(-beta*lam*(b-a)^2/(2*h^2))`` and site term
    ``s(x) = exp(-beta*(1-x^2)^2/(4*lam))`` on a 1D quadrature grid -- see
    ``ginzburg_landau_log_normalizer``'s docstring for the same recursion
    used just for the normalizer. Built once per call and reused by both
    the 1D and 2D exact-marginal functions below.
    """

    def __init__(self, d: int, beta: float, lam: float, h: float, n_grid: int, x_max: float):
        self.d = d
        self.grid = np.linspace(-x_max, x_max, n_grid)
        self.w = np.full(n_grid, self.grid[1] - self.grid[0])
        self.w[0] *= 0.5
        self.w[-1] *= 0.5  # trapezoidal rule
        self.edge_kernel = np.exp(-beta * lam * (self.grid[:, None] - self.grid[None, :]) ** 2 / (2.0 * h**2))
        self.site_term = np.exp(-beta * (1.0 - self.grid**2) ** 2 / (4.0 * lam))
        self.boundary_kernel = np.exp(-beta * lam * self.grid**2 / (2.0 * h**2))

    def message(self, n_full: int) -> np.ndarray:
        """Message reaching a site through ``n_full`` OTHER sites fully
        integrated out (each contributing its own site term), stopping
        just short of that site's own term -- i.e. this is the forward
        ``L`` message (or, by symmetry, the backward ``R`` message)
        evaluated at the target site, without double-counting the
        target's own ``s(x)`` factor."""
        if n_full == 0:
            return self.boundary_kernel.copy()
        v = self.boundary_kernel * self.site_term
        for _ in range(n_full - 1):
            v = self.site_term * (self.edge_kernel @ (v * self.w))
        return self.edge_kernel @ (v * self.w)

    def normalizer(self) -> float:
        v = self.boundary_kernel * self.site_term
        for _ in range(self.d - 1):
            v = self.site_term * (self.edge_kernel @ (v * self.w))
        return float(np.sum(v * self.w * self.boundary_kernel))


def ginzburg_landau_exact_1d_marginal(
    d: int,
    i: int,
    grid_x: np.ndarray,
    beta: float = GL_BETA,
    lam: float = GL_LAM,
    h: float | None = None,
    *,
    n_grid: int = 2000,
    x_max: float = 3.0,
) -> np.ndarray:
    """Exact normalized 1D marginal ``p(x_i)`` of the GL chain (0-indexed
    coordinate ``i``), evaluated on ``grid_x``, via the same transfer-
    operator messages as ``ginzburg_landau_exact_adjacent_marginal`` (see
    its docstring), just for a single un-marginalized site instead of two.
    """
    if not (0 <= i < d):
        raise ValueError(f"i={i} must be a valid 0-indexed coordinate, 0 <= i < d={d}")
    if h is None:
        h = _default_h(d)

    op = _TransferOperator(d, beta, lam, h, n_grid, x_max)
    l_pre = op.message(i)
    r_pre = op.message(d - i - 1)
    z = op.normalizer()
    marginal_grid = l_pre * op.site_term * r_pre / z

    return np.interp(grid_x, op.grid, marginal_grid)


def ginzburg_landau_exact_adjacent_marginal(
    d: int,
    i: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    beta: float = GL_BETA,
    lam: float = GL_LAM,
    h: float | None = None,
    *,
    n_grid: int = 2000,
    x_max: float = 3.0,
) -> np.ndarray:
    """Exact normalized 2D marginal ``p(x_i, x_{i+1})`` of the GL chain (0-
    indexed *adjacent* coordinates ``i``, ``i+1``), evaluated on the outer
    product of ``grid_x`` x ``grid_y``.

    Writes the joint as ``L(x_i) * s(x_i) * K(x_i, x_{i+1}) * s(x_{i+1}) *
    R(x_{i+1}) / Z``, where ``K``/``s`` are the same edge/site kernels as
    ``ginzburg_landau_log_normalizer`` and ``L``/``R`` are the forward/
    backward messages reaching site ``i``/``i+1`` from the left/right
    boundary through every *other* site (each of those fully integrated
    out) -- exactly the two-sided message-passing needed to keep one edge
    of a chain un-marginalized while integrating out the rest. Evaluated
    on an internal 1D quadrature grid, then bivariate-spline-interpolated
    onto the requested plotting grid.
    """
    if not (0 <= i < d - 1):
        raise ValueError(f"i={i} must leave a valid adjacent partner i+1 < d={d}")
    if h is None:
        h = _default_h(d)

    op = _TransferOperator(d, beta, lam, h, n_grid, x_max)
    l_pre = op.message(i)          # sites strictly left of site i, each fully integrated
    r_pre = op.message(d - i - 2)  # sites strictly right of site i+1, each fully integrated
    z = op.normalizer()

    joint_grid = (l_pre * op.site_term)[:, None] * op.edge_kernel * (op.site_term * r_pre)[None, :] / z

    spline = RectBivariateSpline(op.grid, op.grid, joint_grid)
    return spline(grid_x, grid_y).T  # .T: rows index y (grid_y), cols index x (grid_x)

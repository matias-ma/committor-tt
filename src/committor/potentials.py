"""Physical systems the committor solver can be pointed at.

Each system is exposed as a small :class:`Potential` bundle: an unnormalized
log-density, a sampler, and the two metastable basin centers used to build
the soft boundary-condition surrogates ``p_A``/``p_B``. This interface is
deliberately generic (nothing here is specific to TT/Tucker or to a
particular system) so a future potential -- e.g. a RECOVAR/cryo-EM latent
free energy -- can be dropped in without touching ``ansatz.py``,
``variational.py``, or ``training.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ftde.ginzburg_landau import (
    GL_BETA,
    GL_LAM,
    ginzburg_landau_log_prob,
    sample_ginzburg_landau,
)

Array = jnp.ndarray


@dataclass(frozen=True)
class Potential:
    """A ``d``-dimensional equilibrium system for the committor problem.

    ``log_density`` need only be correct up to an additive constant (the
    variational loss only ever consumes ``\\|grad q\\|^2`` weighted by ``p``
    and ``p``-derived Monte Carlo samples, never an absolute normalization).
    """

    d: int
    domain: tuple  # ((lo, hi), ...) length d
    log_density: Callable[[Array], Array]  # (..., d) -> (...)
    sample: Callable[[int, int], Array]  # (seed, n) -> (n, d)
    basin_a: Array  # (d,) center of metastable set A (q -> 0)
    basin_b: Array  # (d,) center of metastable set B (q -> 1)


# ---------------------------------------------------------------------------
# Double well: a fast, separable sanity-check system with a closed-form 1D
# reference committor (see reference.exact_committor_1d).
# ---------------------------------------------------------------------------


def double_well(
    d: int,
    beta: float = 1.0,
    alpha: float = 0.3,
    domain_half_width: float = 2.0,
) -> Potential:
    """``V(x) = (x_1^2 - 1)^2 + alpha * sum_{k>=2} x_k^2``.

    Separable across dimensions, so it admits an exact per-dimension sampler
    and (by symmetry) a committor equal to the 1D closed-form committor of
    ``x_1`` alone -- a cheap, dependency-free oracle for regression tests.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    def log_density(x: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        v1 = (x[..., 0] ** 2 - 1.0) ** 2
        vrest = alpha * jnp.sum(x[..., 1:] ** 2, axis=-1)
        return -beta * (v1 + vrest)

    def sample(seed: int, n: int) -> Array:
        return _sample_double_well(seed, n, d, beta, alpha, domain_half_width)

    domain = ((-domain_half_width, domain_half_width),) * d
    basin_a = jnp.zeros(d).at[0].set(-1.0)
    basin_b = jnp.zeros(d).at[0].set(1.0)
    return Potential(d=d, domain=domain, log_density=log_density, sample=sample,
                      basin_a=basin_a, basin_b=basin_b)


def _sample_double_well(
    seed: int, n: int, d: int, beta: float, alpha: float, domain_half_width: float
) -> Array:
    rng = np.random.default_rng(seed)
    a, b = -domain_half_width, domain_half_width

    # x_1 ~ exp(-beta*(x^2-1)^2) via rejection sampling; the unnormalized
    # density has maximum 1 (attained at x = +-1), so a uniform proposal on
    # [a, b] with acceptance probability equal to the density works directly.
    accepted = []
    n_have = 0
    batch = max(4 * n, 4096)
    while n_have < n:
        cands = rng.uniform(a, b, size=batch)
        probs = np.exp(-beta * (cands ** 2 - 1.0) ** 2)
        kept = cands[rng.uniform(size=batch) < probs]
        accepted.append(kept)
        n_have += kept.shape[0]
    x1 = np.concatenate(accepted)[:n]

    if d > 1:
        std = 1.0 / np.sqrt(2.0 * beta * alpha)
        rest = np.clip(rng.normal(0.0, std, size=(n, d - 1)), a, b)
        x = np.concatenate([x1[:, None], rest], axis=1)
    else:
        x = x1[:, None]
    return jnp.asarray(x)


# ---------------------------------------------------------------------------
# Ginzburg-Landau: the flagship example. The density/sampler are reused
# directly from ft-de (no reimplementation) so the two projects describe the
# exact same physical system.
# ---------------------------------------------------------------------------


def ginzburg_landau(
    d: int,
    beta: float = GL_BETA,
    lam: float = GL_LAM,
    h: Optional[float] = None,
    domain_half_width: float = 2.0,
) -> Potential:
    """The discretized Ginzburg-Landau chain from ``ftde.ginzburg_landau``.

    Matches Peng, Khoo & Wang (arXiv:2401.11646) eq. (26) exactly (same
    defaults ``beta=1/8``, ``lam=0.02``, ``h=1/(d+1)``), via ``ftde``'s own
    ``ginzburg_landau_log_prob``/``sample_ginzburg_landau`` -- no
    reimplementation, so this project and ft-de describe the identical
    system.
    """
    if h is None:
        h = 1.0 / (d + 1)

    def log_density(x: Array) -> Array:
        return ginzburg_landau_log_prob(x, beta=beta, lam=lam, h=h)

    def sample(seed: int, n: int) -> Array:
        return sample_ginzburg_landau(seed, n, d, beta=beta, lam=lam, h=h)

    domain = ((-domain_half_width, domain_half_width),) * d
    basin_minus, basin_plus = ginzburg_landau_minimizers(d, beta=beta, lam=lam, h=h)
    return Potential(d=d, domain=domain, log_density=log_density, sample=sample,
                      basin_a=jnp.asarray(basin_minus), basin_b=jnp.asarray(basin_plus))


def find_local_maxima(
    log_density: Callable[[Array], Array], init_points: np.ndarray, domain: np.ndarray | None = None,
) -> np.ndarray:
    """L-BFGS-B search for a local maximum of ``log_density`` from each row of
    ``init_points`` (n_init, d), autodiffed via JAX. General-purpose: works
    for the analytic potential's own log-density just as well as for a
    *fitted* density estimate's ``log_density`` (e.g. ``ftde``'s
    ``log_density_tt``/``log_density_tucker``), which is the point -- finding
    metastable states directly from an estimated density, without knowing
    the underlying analytic potential, is exactly the cryo-EM-style use case
    this project is aimed at. Returns an array of shape ``(n_init, d)``.

    ``domain`` (optional, shape ``(d, 2)``) box-constrains the search to
    stay within it. This matters specifically for a *fitted* Chebyshev-TT
    density: it is a polynomial, only valid on the box it was fit on, and
    an unconstrained L-BFGS-B step can wander outside that box while
    climbing the gradient -- where the polynomial extrapolates to a huge,
    meaningless value rather than actually being a large density. Without
    ``domain``, that failure mode can silently return a "basin" far outside
    where the density estimate means anything, corrupting everything
    downstream (confirmed in practice: it produced a NaN committor loss
    when used as a basin anchor). Pass the density's own fit domain
    whenever ``log_density`` comes from a fitted TT/Tucker estimate; leave
    it ``None`` for an analytic potential with no such restriction.
    """
    from scipy.optimize import minimize

    def neg_log_density(u: Array) -> Array:
        return -log_density(u)

    value_and_grad = jax.jit(jax.value_and_grad(neg_log_density))

    def objective(u_np: np.ndarray):
        value, grad = value_and_grad(jnp.asarray(u_np, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    bounds = None if domain is None else [(float(lo), float(hi)) for lo, hi in np.asarray(domain)]
    optima = []
    for init in np.asarray(init_points, dtype=np.float64):
        result = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000, "ftol": 1e-14})
        optima.append(result.x)
    return np.stack(optima, axis=0)


def ginzburg_landau_minimizers(
    d: int, beta: float = GL_BETA, lam: float = GL_LAM, h: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """The two homogeneous ground states U_- ~= -1 and U_+ ~= +1 of the GL chain.

    Found by :func:`find_local_maxima` from two opposite tanh-kink profiles
    -- the same initialization strategy as the paper's own GL minimizer
    search.
    """
    if h is None:
        h = 1.0 / (d + 1)

    def log_density(u: Array) -> Array:
        return ginzburg_landau_log_prob(u, beta=beta, lam=lam, h=h)

    i_arr = np.arange(1, d + 1, dtype=float)
    x_arr = i_arr / (d + 1)
    sharpness = 10.0 / lam
    kink = np.tanh(sharpness * (x_arr - 0.1)) - np.tanh(sharpness * (x_arr - 0.9))
    kink = kink / (kink.max() + 1e-12)

    minus, plus = find_local_maxima(log_density, np.stack([-kink, kink], axis=0))
    return minus, plus


# ---------------------------------------------------------------------------
# A fitted density p = g^2 / Z with g a Chebyshev TT (ft-de's squared-TT model),
# e.g. a deconvolved RECOVAR latent density: the cryo-EM use case.
# ---------------------------------------------------------------------------
def squared_tt_density(tt, basin_a, basin_b, *, center=None, scale: float = 1.0) -> Potential:
    """:class:`Potential` for ``p = g^2 / Z``, ``g`` a coefficient ChebTT on a box.

    The problem is posed in rescaled coordinates ``x = (z - center) / scale``
    (``z`` the density's own coordinates). One scalar ``scale`` for every
    axis leaves the committor unchanged (``grad q`` scales uniformly, so the
    minimizer of the Dirichlet form is the same), but brings the Dirichlet
    energy to O(1): a latent space whose coordinates span +-1e4, as RECOVAR's
    do, has ``E|grad q|^2 ~ 1e-8``, which ``rho ~ 100`` would swamp.
    ``basin_a``/``basin_b`` are given in ``z`` coordinates.

    ``log_density`` clips into the box, where the polynomial is valid (see
    ``find_local_maxima``). ``sample`` draws exactly from ``g^2`` with
    ft-de's conditional-CDF sampler, so no MCMC is involved.
    """
    from ftde.tt_sampling import make_squared_tt_sampler

    domain_z = np.asarray(tt.domain, dtype=np.float64)
    c = domain_z.mean(axis=1) if center is None else np.asarray(center, dtype=np.float64)
    s = float(scale)
    lo, hi = jnp.asarray(domain_z[:, 0]), jnp.asarray(domain_z[:, 1])
    c_j = jnp.asarray(c)
    sampler = make_squared_tt_sampler(tt)
    params = tuple(core.coeffs for core in tt.cores)
    d = domain_z.shape[0]
    # the conditional-CDF sampler holds (batch, modes, modes) and (rank^2, batch) arrays per
    # step: keep them near 2 GB (32,768 points up to 90 modes, 2,048 at 257)
    n_modes = max(core.coeffs.shape[-1] for core in tt.cores)
    chunk = int(min(32768, 2 ** np.floor(np.log2(2 ** 28 / n_modes ** 2))))

    def log_density(x: Array) -> Array:
        z = jnp.clip(c_j + s * jnp.asarray(x, dtype=jnp.float64), lo, hi)
        return jnp.log(tt(z) ** 2 + 1e-300)

    def sample(seed: int, n: int) -> Array:
        u = jax.random.uniform(jax.random.PRNGKey(seed), (n, d), dtype=jnp.float64,
                               minval=1e-12, maxval=1.0 - 1e-12)
        # fixed-size chunks (see above; one chunk shape also means one compilation)
        pad = (-n) % chunk
        u = jnp.concatenate([u, jnp.full((pad, d), 0.5)]) if pad else u
        z = jnp.concatenate([sampler(params, u[i:i + chunk]) for i in range(0, n + pad, chunk)])
        return (z[:n] - c_j) / s

    domain = tuple(((a - ci) / s, (b - ci) / s) for (a, b), ci in zip(domain_z, c))
    to_x = lambda z: (jnp.asarray(z, dtype=jnp.float64) - c_j) / s  # noqa: E731
    return Potential(d=d, domain=domain, log_density=log_density, sample=sample,
                     basin_a=to_x(basin_a), basin_b=to_x(basin_b))

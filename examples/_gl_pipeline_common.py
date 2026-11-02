"""Shared helpers for the Ginzburg-Landau "density estimation -> committor"
example pipelines (``ginzburg_landau_density_and_committor.py``, the noisy
case, and ``ginzburg_landau_clean_pipeline.py``, the noise-free/higher-d
case). Factored out so the two scripts share one implementation of basin-
finding, committor training, and plotting helpers rather than drifting
apart.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebttge.computations as ttcalc
import optax
from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.cheb_product import square_tt
from ftde.noisy_regularization import tikhonov_penalty_tt
from ftde.sobolev import amplitude_h1, derivative_grams
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import density_tt, rebuild_tt, squared_integral_tt, tt_params

from committor import ansatz, training
from committor.gl_reference import ginzburg_landau_exact_1d_marginal, ginzburg_landau_exact_adjacent_marginal
from committor.potentials import Potential, find_local_maxima

Array = jnp.ndarray

FIG_DPI = 300
PANEL_LABEL_KW = dict(fontsize=12, fontweight="bold", va="top", ha="left")


def relative_l2_error(y_est: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sum((y_est - y_true) ** 2) / np.sum(y_true**2))


def density_at(fit, domain: jnp.ndarray, points: jnp.ndarray) -> np.ndarray:
    """``density_tt(fit, points)``, clipping ``points`` into ``fit``'s own
    domain first (a fitted Chebyshev TT extrapolates catastrophically just
    outside the box it was fit on)."""
    lo, hi = domain[:, 0], domain[:, 1]
    return np.asarray(density_tt(fit, jnp.clip(points, lo, hi)))


def data_domain(points: jnp.ndarray, margin_frac: float = 0.1) -> jnp.ndarray:
    """Domain from the raw min/max of ``points``, padded by ``margin_frac``
    of the range. For clean (noise-free) data only -- see ``robust_domain``
    in the noisy pipeline script for why noisy data needs percentile
    trimming instead."""
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def robust_domain(points: jnp.ndarray, trim_pct: float = 0.5, margin_frac: float = 0.05) -> jnp.ndarray:
    """Domain from the [trim_pct, 100-trim_pct] percentile range per
    dimension (not raw min/max), padded by ``margin_frac`` of that range.

    For NOISY data only (see the noisy pipeline script's own use of this):
    trimming discards genuine heavy-tailed noise outliers that would
    otherwise force the domain, and the Chebyshev degree it buys, to be
    spent on a region with no real density mass. It is NOT a fix for
    CLEAN data at higher d: a d=20 clean-data experiment tried trimming
    here on the (mistaken) theory that it would prevent a cross-
    approximation collapse, as ft-de's cryo-EM pipeline documents for a
    different dataset (``ft-de/scripts/cryoem_common.py``) -- but it made
    the fit *worse* (relative L2 error 2.22 vs. 1.02 untrimmed), because a
    fitted squared-TT renormalizes over whatever domain it is given, so
    shrinking the domain for genuine (non-noise) samples just inflates
    every estimated density value relative to the true density's actual
    scale. The real fix for that d=20 result (confirmed to be an untested
    weak spot in ft-de's own settings, not specific to this project --
    ft-de's own unmodified ``gl_gpu_noise_free.py`` gets the identical
    1.02 at d=20) was to use a dimension ft-de's own settings actually
    validate well (d=10, relative L2 error 0.28), not to change the
    domain -- see ``ginzburg_landau_clean_pipeline.py``'s module docstring.
    """
    lo = jnp.percentile(points, trim_pct, axis=0)
    hi = jnp.percentile(points, 100.0 - trim_pct, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def filter_to_domain(points: jnp.ndarray, domain: jnp.ndarray) -> jnp.ndarray:
    inside = jnp.all((points >= domain[:, 0]) & (points <= domain[:, 1]), axis=1)
    kept = points[inside]
    n_dropped = int(points.shape[0] - kept.shape[0])
    if n_dropped:
        print(f"  (dropped {n_dropped}/{points.shape[0]} points outside the trimmed domain)", flush=True)
    return kept


def exact_marginal_density_tt(g_tt, keep_modes, grid_x, grid_y) -> np.ndarray:
    """2D marginal of a *fitted* squared-Chebyshev-TT density (exact for the
    fit itself, via TT marginalization -- not to be confused with
    ``true_marginal_density`` below, which is the exact marginal of the
    true analytic GL density)."""
    g_sq = square_tt(g_tt)
    z = float(squared_integral_tt(g_tt))
    marginal = ttcalc.marginal(g_sq, keep_modes)
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    pts = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], axis=-1)
    lo = jnp.array([g_tt.domain[keep_modes[0], 0], g_tt.domain[keep_modes[1], 0]])
    hi = jnp.array([g_tt.domain[keep_modes[0], 1], g_tt.domain[keep_modes[1], 1]])
    pts = jnp.clip(pts, lo, hi)
    values = np.asarray(marginal(pts)).reshape(X.shape) / z
    return values


def exact_marginal_density_tt_1d(g_tt, mode: int, grid_x) -> np.ndarray:
    """1D marginal of a *fitted* squared-Chebyshev-TT density -- same idea
    as ``exact_marginal_density_tt``, just keeping one mode instead of two."""
    g_sq = square_tt(g_tt)
    z = float(squared_integral_tt(g_tt))
    marginal = ttcalc.marginal(g_sq, (mode,))
    lo, hi = g_tt.domain[mode, 0], g_tt.domain[mode, 1]
    pts = jnp.clip(jnp.asarray(grid_x)[:, None], lo, hi)
    return np.asarray(marginal(pts)) / z


def true_marginal_density_1d(d: int, mode: int, grid_x) -> np.ndarray:
    """Exact 1D marginal of the TRUE (analytic) Ginzburg-Landau density over
    a single coordinate, via the same transfer-operator recursion as
    ``true_marginal_density``."""
    return ginzburg_landau_exact_1d_marginal(d, mode, np.asarray(grid_x))


def histogram_density_1d(samples: np.ndarray, mode: int, grid_x) -> np.ndarray:
    edges = np.concatenate([[grid_x[0] - (grid_x[1] - grid_x[0]) / 2], (grid_x[:-1] + grid_x[1:]) / 2, [grid_x[-1] + (grid_x[-1] - grid_x[-2]) / 2]])
    counts, _ = np.histogram(samples[:, mode], bins=edges, density=True)
    return counts


def true_marginal_density(d: int, keep_modes: tuple[int, int], grid_x, grid_y) -> np.ndarray:
    """Exact 2D marginal of the TRUE (analytic) Ginzburg-Landau density over
    ``keep_modes`` (must be adjacent, e.g. ``(0, 1)``), via the transfer-
    operator recursion in ``committor.gl_reference`` -- a genuine ground
    truth, unlike a held-out-sample histogram (which is itself only an
    estimator of the same marginal, with its own sampling noise)."""
    i, j = keep_modes
    if j != i + 1:
        raise ValueError(f"true_marginal_density needs adjacent keep_modes, got {keep_modes}")
    return ginzburg_landau_exact_adjacent_marginal(d, i, np.asarray(grid_x), np.asarray(grid_y))


def histogram_density_2d(samples: np.ndarray, keep_modes, grid_x, grid_y) -> np.ndarray:
    x_edges = np.concatenate([[grid_x[0] - (grid_x[1] - grid_x[0]) / 2], (grid_x[:-1] + grid_x[1:]) / 2, [grid_x[-1] + (grid_x[-1] - grid_x[-2]) / 2]])
    y_edges = np.concatenate([[grid_y[0] - (grid_y[1] - grid_y[0]) / 2], (grid_y[:-1] + grid_y[1:]) / 2, [grid_y[-1] + (grid_y[-1] - grid_y[-2]) / 2]])
    counts, _, _ = np.histogram2d(
        samples[:, keep_modes[0]], samples[:, keep_modes[1]], bins=[x_edges, y_edges], density=True
    )
    return counts.T


def committor_slice(committor_ansatz, d: int, keep_modes, grid_x, grid_y):
    """2D slice of the committor over ``keep_modes``, holding every other
    coordinate fixed at 0 (the chain's own natural "center")."""
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    pts = np.zeros((X.size, d))
    pts[:, keep_modes[0]] = X.ravel()
    pts[:, keep_modes[1]] = Y.ravel()
    q = np.asarray(committor_ansatz(jnp.asarray(pts))).reshape(X.shape)
    return q


def domain_clipped_log_density(fit, log_density_fn):
    """Wrap a fitted TT's ``log_density_fn`` (e.g. ``log_density_tt``) so it
    clips its input into ``fit.domain`` before evaluating.

    A fitted Chebyshev-TT is a polynomial, only valid on the box it was fit
    on. ``Potential.sample`` for an estimated-density potential draws from
    the true analytic distribution (there is no other ground-truth sampler
    to use), which is not itself confined to the fitted density's domain --
    clipping here confines every downstream evaluation (training samples,
    basin-adjacent Laplace samples, everything) to the region the estimate
    actually means something on.
    """
    lo = jnp.asarray(fit.domain[:, 0])
    hi = jnp.asarray(fit.domain[:, 1])

    def f(x):
        return log_density_fn(fit, jnp.clip(x, lo, hi))

    return f


def basins_from_data(log_density, train_samples: np.ndarray, domain: np.ndarray, extreme_frac: float = 0.02):
    """Initialize each basin's L-BFGS-B search from the *average* of the
    most extreme ``extreme_frac`` of training points along the mean-field
    direction, rather than the single most extreme point.

    A single most-extreme point is itself a noisy observation. On noisy
    data especially, it is often a large individual noise realization
    rather than a point actually near the true homogeneous state, and
    refining from it can converge to a spurious nearby bump in a rough/
    noisy density estimate instead of the real basin (confirmed in
    practice: this produced basin points far from the true +-1 homogeneous
    states, one of them pinned to the domain edge, which in turn broke the
    downstream Laplace approximation -- see ``committor.sampling.
    laplace_basin_sample``). Averaging over the extreme tail denoises the
    initial guess the same way averaging denoises any other statistic.
    """
    r = np.asarray(jnp.mean(jnp.asarray(train_samples), axis=-1))
    n = train_samples.shape[0]
    k = max(1, int(extreme_frac * n))
    order = np.argsort(r)
    init_minus = np.asarray(np.mean(train_samples[order[:k]], axis=0))
    init_plus = np.asarray(np.mean(train_samples[order[-k:]], axis=0))
    minus, plus = find_local_maxima(log_density, np.stack([init_minus, init_plus], axis=0), domain=domain)
    return minus, plus


def fit_deconvolved_tt_validated(
    init,
    u,
    train: jnp.ndarray,
    valid: jnp.ndarray,
    *,
    degree: int,
    steps: int,
    lr: float,
    batch_size: int,
    noise_std: jnp.ndarray,
    reg_lambda: float = 1.0,
    per_axis_target_budget: float = 0.5,
    smoothness: float = 0.0,
    noise_variance_per_axis: jnp.ndarray | None = None,
    log_every: int = 200,
    seed: int = 0,
    l2_eval=None,
):
    """Fit a noise-deconvolved squared-Chebyshev-TT density, using the
    training recipe validated in ft-de's own
    ``scripts/benchmark_gl_convolution.py`` / ``docs/
    gl_convolution_followup.md`` -- a substantial improvement over a plain
    fixed-LR ``fit_noisy_tt`` call on raw coefficients, which that doc
    documents picking an early checkpoint and then visibly degrading for
    the rest of training (an optimization/regularization gap, not a
    forward-convolution accuracy problem). Three ingredients:

    1. **Gram/Cholesky-preconditioned coefficients**: optimize ``c @
       chol(gram)`` instead of the raw Chebyshev coefficients, so every
       degree gets a comparable effective step size (their Gram-matrix
       eigenvalues otherwise differ by orders of magnitude between low and
       high degree).
    2. **Cosine-decayed learning rate** instead of a fixed one.
    3. **`TTConvolution.prepare()`** (direct contraction of two copies of
       the signal TT against the prescribed noise kernel, not an explicit
       ``g**2`` square-and-round) with integration moments precomputed
       once, plus **validation-NLL checkpoint selection** every
       ``log_every`` steps -- this replaces hand-tuning a fixed early-
       stopping step count (this project's earlier, less principled fix
       for the same ill-posedness) with an automatic one: train long and
       keep whichever checkpoint generalizes best.

    Returns ``(fit, history, best_step, best_nll)`` where ``history`` is a
    list of ``(step, validation_nll)`` pairs.

    ``l2_eval``, if given, is an optional ``callable(tt) -> float`` (e.g.
    relative L2 error against a true/held-out density) logged alongside the
    validation NLL at each ``log_every`` step -- purely diagnostic, and NOT
    used for checkpoint selection (which stays NLL-based, matching what a
    real deployment without ground truth could actually compute). This
    exists because checkpoint selection by *convolved* validation NLL alone
    is not sufficient protection against this problem's ill-posedness: the
    noisy-observation NLL can keep improving while the raw deconvolved
    signal quietly over-sharpens into taller, narrower-than-true peaks
    (invisible to a metric computed on the smoothed/convolved density, but
    visible immediately in ``l2_eval``). Use it to pick a good fixed
    ``steps``/regularization budget for a given noise model, the same way
    an earlier step-count sweep on this exact pipeline did by hand.

    ``smoothness``, if nonzero, adds ft-de's own Sobolev-H1 amplitude
    penalty (``ftde.sobolev.amplitude_h1``, the regularizer validated in
    ``scripts/benchmark_gl_covariance.py`` / ``results/gl_covariance_
    mixture`` for a noisy-GL problem with a comparably heterogeneous,
    correlated noise covariance -- see ``.scratch/cov_r4_n25_smooth.log``
    there, rank=4/degree=24/smoothness=0.001/steps=12000, full relative L2
    0.209 vs. naive 0.916) instead of, or alongside, the per-axis Tikhonov
    penalty above. Unlike that Tikhonov penalty, this one checkpoint-
    selects cleanly by validation NLL alone in ft-de's own benchmark (no
    separate ``l2_eval``-based early-stopping sweep needed there) --
    penalizing the fitted density's own smoothness directly, rather than
    its noise-convolved amplitude budget, evidently suppresses the over-
    sharpening failure mode well enough for NLL to track it.
    ``noise_variance_per_axis`` is the per-axis weight ``amplitude_h1``
    uses (ft-de's own recipe weights each axis by its average noise
    variance); required if ``smoothness`` is nonzero.
    """
    domain = init.domain
    op = TTConvolution.from_tt(init, u)
    all_obs = np.concatenate([np.asarray(train), np.asarray(valid)])
    observation_domain = np.stack([all_obs.min(axis=0) - 0.05, all_obs.max(axis=0) + 0.05], axis=1)
    op = op.prepare(observation_domain, tolerance=1e-14)
    derivatives = derivative_grams(init) if smoothness else None

    factors = [jnp.asarray(np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(degree, tuple(box))))) for box in domain]
    inverse = [jnp.asarray(np.linalg.inv(np.asarray(f))) for f in factors]

    def to_precond(params):
        return tuple(c @ f for c, f in zip(params, factors, strict=True))

    def rebuild(p):
        return rebuild_tt(init, tuple(c @ f for c, f in zip(p, inverse, strict=True)))

    def objective(p, x):
        g = rebuild(p)
        loss = -jnp.mean(op.log_density(g, x))
        if reg_lambda:
            loss = loss + reg_lambda * tikhonov_penalty_tt(square_tt(g), noise_std, per_axis_target_budget=per_axis_target_budget)
        if smoothness:
            loss = loss + smoothness * amplitude_h1(g, op.grams, derivatives, noise_variance_per_axis)
        return loss

    schedule = optax.cosine_decay_schedule(lr, max(steps, 1), alpha=0.1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    params = to_precond(tt_params(init))
    opt_state = optimizer.init(params)

    def step_fn(p, s, x):
        loss, grad = jax.value_and_grad(lambda pp: objective(pp, x))(p)
        updates, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, updates), s, loss

    step_fn = jax.jit(step_fn)
    evaluate = jax.jit(lambda p, x: op.log_density(rebuild(p), x))

    inside = jnp.all((valid >= domain[:, 0]) & (valid <= domain[:, 1]), axis=1)
    validation = valid[inside]

    best, best_nll, best_step, history = params, float(np.inf), 0, []
    key = jax.random.PRNGKey(seed)
    for i in range(steps + 1):
        if i % log_every == 0 or i == steps:
            nll = -float(jnp.mean(evaluate(params, validation)))
            if nll < best_nll:
                best, best_nll, best_step = params, nll, i
            history.append((i, nll))
            if l2_eval is not None:
                l2 = l2_eval(rebuild(params))
                print(f"[deconv] step {i:5d}  validation nll {nll:.4f}  (best {best_nll:.4f} @ step {best_step})  "
                      f"diagnostic rel. L2 vs. true density = {l2:.4f}", flush=True)
            else:
                print(f"[deconv] step {i:5d}  validation nll {nll:.4f}  (best {best_nll:.4f} @ step {best_step})", flush=True)
        if i == steps:
            break
        key, sub = jax.random.split(key)
        batch = train[jax.random.randint(sub, (batch_size,), 0, train.shape[0])]
        params, opt_state, loss = step_fn(params, opt_state, batch)

    return rebuild(best), history, best_step, best_nll


def solve_committor_for_potential(pot: Potential, label: str, *, rank: int = 6, degree: int = 14):
    print(f"[committor] solving for {label} (d={pot.d}) ...", flush=True)
    a0 = ansatz.init(jax.random.PRNGKey(0), d=pot.d, domain=pot.domain, rank=rank, degree=degree, format="tt")
    a, diag = training.train_committor(
        a0, pot, final_rho=100.0, n_rho_stages=4, n_samples=4096,
        batch_size=256, steps_per_stage=400, learning_rate=3e-3, seed=0,
    )
    print(f"[committor] {label}: final loss = {diag.loss_history[-1]:.4f}", flush=True)
    return a, diag


def label_panel(ax, label: str) -> None:
    """Small bold "(a)"-style panel label in the top-left corner, in place
    of dense multi-line titles or gutter-annotation arrows -- keeps every
    panel's own title short and lets the figure caption spell out the
    pipeline order and per-panel stats instead."""
    ax.text(-0.08, 1.12, label, transform=ax.transAxes, **PANEL_LABEL_KW)


def savefig(fig: plt.Figure, path) -> None:
    fig.savefig(path, dpi=FIG_DPI)


def flow_connector(fig, gs_cell, symbol: str = "→", fontsize: int = 20):
    """A bare operator/arrow symbol centered in its own GridSpec cell.

    Putting the connector in a real (narrow) GridSpec column -- rather than
    an ``ax.annotate`` floating in the gutter of a neighboring axes, tried
    in an earlier revision of this figure -- makes its alignment between
    rows automatic instead of something to hand-tune, since it is governed
    by the same shared column grid as every content panel.
    """
    ax = fig.add_subplot(gs_cell)
    ax.axis("off")
    ax.text(0.5, 0.5, symbol, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize)
    return ax

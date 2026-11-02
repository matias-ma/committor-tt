"""Density estimation -> committor, end to end, on Ginzburg-Landau, under
genuinely hard (heteroscedastic, correlated) observation noise.

This is a dry run, on a system with a known ground truth, of the pipeline
this project is ultimately aimed at (estimate a density from data alone,
find its metastable states, solve the committor between them) -- using
ft-de for the density estimation and committor.* for the committor, with
chebdGE underneath both. See ``ginzburg_landau_clean_pipeline.py`` for the
no-noise, higher-dimension counterpart of this same pipeline.

The density-estimation recipe here is the one ft-de itself validates for
noisy Ginzburg-Landau, not the earlier (never fully run) ``gl_gpu_noisy.py``
settings this script started from -- see ``ft-de/docs/
gl_convolution_followup.md`` and ``fit_deconvolved_tt_validated`` in
``_gl_pipeline_common.py`` for exactly what changed and why: Gram/Cholesky-
preconditioned coefficients, a cosine-decayed learning rate, the direct
``TTConvolution`` operator (rather than an explicit ``g**2``-then-round
convolution) with its integration moments precomputed once, and
checkpoint selection by periodic validation NLL. That checkpoint selection
alone is NOT sufficient at this scale, though -- see ``REG_TARGET_BUDGET``'s
own comment below for a real failure it does not catch (the convolved
validation NLL plateaus while the raw deconvolved signal keeps quietly
over-sharpening) and the stronger regularization budget that fixes it.
The naive (no-deconvolution) fit is skipped entirely
(``NAIVE_STEPS = 0``, matching ft-de's own convention: with genuinely hard
noise, training a no-deconvolution model barely improves on its own
untrained KDE cross-approximation init, so that untrained init is reported
directly as the "naive" reference instead of spending a training budget on
it).

1. Fit a squared-Chebyshev-TT density to *clean* Ginzburg-Landau samples.
2. Corrupt the samples with substantial *colored* noise (each observation's
   noise drawn from its own randomly-sampled, correlated covariance -- see
   ``sample_colored_noise``), and fit a noise-deconvolved density with the
   validated recipe above.
3. For the clean fit and the deconvolved-noisy fit, find their two
   metastable basins directly from the *estimated* density alone (no
   knowledge of the analytic GL potential -- see
   ``_gl_pipeline_common.basins_from_data``), and solve the committor
   between them. A committor solved from the true analytic potential is
   included as a reference.
4. Render one flow-chart-style figure: true density (+noise) -> observed
   data -> deconvolved fit -> committor, shown twice -- as 2D marginals on
   top, and as the corresponding 1D marginals/basin-to-basin path below --
   plus a density-estimation-quality bar chart.

Usage: python examples/ginzburg_landau_density_and_committor.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import chebttge.computations as ttcalc
from ftde.ginzburg_landau import ginzburg_landau_density, ginzburg_landau_log_normalizer, sample_ginzburg_landau
from ftde.init import init_tt_from_kde
from ftde.kde import select_gaussian_kde_bandwidth
from ftde.noise_kernel import build_pairwise_correlated_noise_kernel_tt
from ftde.tt_density import fit_tt, log_density_tt

from committor import potentials
from committor.potentials import Potential

from _gl_pipeline_common import (
    FIG_DPI,
    basins_from_data,
    committor_slice,
    data_domain,
    density_at,
    domain_clipped_log_density,
    exact_marginal_density_tt,
    exact_marginal_density_tt_1d,
    filter_to_domain,
    fit_deconvolved_tt_validated,
    flow_connector,
    histogram_density_1d,
    histogram_density_2d,
    relative_l2_error,
    robust_domain,
    solve_committor_for_potential,
    true_marginal_density,
    true_marginal_density_1d,
)

# D=10 is the actual target dimension for this figure (a prior, undocumented
# attempt to push straight to D=10 with the wrong recipe diverged badly --
# relative L2 error 23-108 -- see git history / HANDOFF.txt for that dead
# end; D=6 remains useful as a cheap sanity-check dimension, e.g. via
# ``python examples/_deconv_reg_sweep.py 6 ...``).
D = 10
N_TRAIN, N_TEST, N_VALID = 100_000, 4_000, 2_048


def _rank_degree_for_dim(dim: int) -> tuple[int, int]:
    """ft-de's own dimension-tiered capacity (see the identical function in
    ft-de/scripts/gl_gpu_noise_free.py and gl_gpu_noisy.py)."""
    if dim <= 8:
        return 8, 17
    if dim <= 14:
        return 10, 19
    return 14, 21


RANK, DEGREE = _rank_degree_for_dim(D)
N_STARTS, GD_STEPS = 96, 15
CLEAN_BATCH_SIZE = 2048
CLEAN_STEPS, CLEAN_LR = 2000, 1e-3

# --- Deconvolution recipe -----------------------------------------------
# Two regularizers were tried for this ill-posed deconvolution:
#
# 1. Per-axis Tikhonov budget (ftde.noisy_regularization.tikhonov_penalty_tt,
#    docs/gl_convolution_followup.md's "Training-recipe fix" recipe). This
#    project's own budget/step sweep (examples/_deconv_reg_sweep.py) found
#    it CAN reach a good fit (e.g. relative L2 ~0.17 at D=6, budget=3000)
#    but only inside a narrow, early step window -- validation NLL on the
#    *convolved* observations keeps improving for thousands of steps past
#    the point where the *deconvolved* signal's own L2 error has already
#    bottomed out and started climbing again (e.g. at budget=0.5, the NLL-
#    selected checkpoint at step 6600/10000 gave L2 error 8.16, while the
#    trajectory's true minimum was 0.386 at step 300). Checkpoint selection
#    by convolved NLL alone cannot see this -- a hard-coded step cap tuned
#    per noise model is required, which is fragile and doesn't obviously
#    transfer across dimensions.
# 2. Sobolev-H1 "amplitude smoothness" penalty (ftde.sobolev.amplitude_h1),
#    ft-de's OWN validated recipe for a comparably heterogeneous, correlated
#    noisy-GL covariance-mixture benchmark at D=10 (ft-de/scripts/
#    benchmark_gl_covariance.py, ft-de/.scratch/cov_r4_n25_smooth.log,
#    ft-de/results/gl_covariance_mixture/full_density_errors.png): rank=4,
#    degree=24, smoothness=0.001, steps=12000, lr=5e-4, batch=128, reaching
#    integrated relative L2 0.209 (vs. 0.916 naive) with validation-NLL
#    checkpoint selection landing cleanly near the END of training (step
#    9500/12000) -- i.e., unlike the Tikhonov penalty, this one suppresses
#    the over-sharpening failure mode well enough that plain NLL-based
#    checkpoint selection actually works, no hand-tuned step cap needed.
#
# Adopted here: (2), with ft-de's own rank/degree/steps/lr/batch for the
# deconvolution stage specifically (DECONV_RANK/DECONV_DEGREE, intentionally
# smaller/higher-degree than RANK/DEGREE used for the clean fit and the
# untrained-KDE "naive" reference above) -- adapted to this script's own
# colored-noise kernel (ft-de's benchmark uses a different, fixed
# 3-component rotated-covariance noise mixture, but the regularizer itself
# is noise-kernel-agnostic, penalizing only the fitted signal). ft-de's own
# smoothness=0.001, though, does NOT transfer directly: on our noise (wider
# heteroscedastic range, and approximated by a single averaged Gaussian
# kernel rather than an exactly-matched mixture kernel -- see
# sample_colored_noise's own module comment) it reproduces the same
# catastrophic over-sharpening as the Tikhonov penalty did (relative L2
# error 77 at smoothness=0.001). A dedicated smoothness sweep at D=10 (see
# examples/_deconv_reg_sweep.py; run via ``python examples/
# _deconv_reg_sweep.py 10 <steps> <comma-separated smoothness values>``)
# found smoothness=0.01 is the one value (of 0.001/0.005/0.01/0.02/0.03/0.05/
# 0.1/1/10/100 tried) that behaves the way ft-de's own recipe does: L2 error
# drops fast for ~1000 steps and then stays on a broad, STABLE plateau
# (~0.33-0.39) all the way out to 8000 steps, rather than blowing up --
# meaning plain NLL-based checkpoint selection is safe here (it lands at
# step 1900/8000, already inside the plateau), unlike the fragile early-
# stopping the Tikhonov penalty needed.
DECONV_RANK, DECONV_DEGREE = 4, 24
DECONV_STEPS, DECONV_LR, DECONV_BATCH_SIZE = 8_000, 5e-4, 128
SMOOTHNESS = 0.01
REG_LAMBDA = 0.0  # Tikhonov budget disabled; smoothness replaces it (see above)
REG_TARGET_BUDGET = 0.5  # unused while REG_LAMBDA=0; kept for _deconv_reg_sweep.py compatibility

# "Colored" noise (heteroscedastic AND correlated): each observation's noise
# is drawn from its own covariance, itself sampled from a pool of
# COLORED_N_COVARIANCES distinct pairwise-block covariances -- correlated in
# independent adjacent pairs of coordinates (0,1), (2,3), ..., with no
# correlation across pairs. See ft-de/scripts/gl_gpu_noisy.py's module
# comment for the full derivation of this structure (and why not, e.g.,
# full AR(1) correlation -- that degenerates cross-approximation of the
# noise kernel at d>~3). The deconvolution kernel U approximates this
# genuinely per-point-varying noise with the single pairwise-block
# covariance equal to the *average* of the sampled covariances (exact,
# since every sampled covariance shares the same block structure and
# averaging is linear) -- the same approximation this project's own
# cryo-EM pipeline uses for real per-point-varying noise.
NOISE_STDS_ISO = jnp.array([0.3, 0.9])
NOISE_WEIGHTS_ISO = jnp.array([0.7, 0.3])
NONISO_BASE_STD = float(jnp.sqrt(jnp.sum(NOISE_WEIGHTS_ISO * NOISE_STDS_ISO**2)))
SIGMA = 1.0
COLORED_N_COVARIANCES = 100
COLORED_STD_LOG_RANGE = (0.4, 2.2)
COLORED_RHO_RANGE = (0.35, 0.85)
# 12, not 3: ft-de's own paired (correlated) noise benchmark hit a nonfinite
# training loss at kernel-rank 3 and needed rank 12 for a stable fit (see
# ft-de/.scratch/gl8_paired.log vs. gl8_paired_rank12.log) -- the same fix
# applies here since our noise is likewise pairwise-correlated, not just
# heteroscedastic.
NONISO_KERNEL_RANK = 12

MARGINAL_MODES = (0, 1)
GRID_N = 70
PLOT_HALF_WIDTH = 2.0  # every panel (density, kernel, committor) shares this [-2, 2] view


# ---------------------------------------------------------------------------
# Noise-model helpers specific to this (noisy) pipeline -- not shared with
# the no-noise script, unlike everything in _gl_pipeline_common.
# ---------------------------------------------------------------------------


def _pairwise_block_covariance(dim: int, std: float, rho: float) -> np.ndarray:
    cov = np.eye(dim) * std**2
    for k in range(0, dim - 1, 2):
        cov[k, k + 1] = cov[k + 1, k] = rho * std**2
    return cov


def _colored_covariance_params(sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_std = sigma * NONISO_BASE_STD
    log_lo, log_hi = np.log(COLORED_STD_LOG_RANGE[0]), np.log(COLORED_STD_LOG_RANGE[1])
    stds = base_std * np.exp(rng.uniform(log_lo, log_hi, size=COLORED_N_COVARIANCES))
    rhos = rng.uniform(*COLORED_RHO_RANGE, size=COLORED_N_COVARIANCES)
    return stds, rhos


def _colored_effective_std_rho(stds: np.ndarray, rhos: np.ndarray) -> tuple[float, float]:
    mean_var = float(np.mean(stds**2))
    mean_offdiag = float(np.mean(rhos * stds**2))
    return float(np.sqrt(mean_var)), mean_offdiag / mean_var


def sample_colored_noise(dim: int, sigma: float, n_pts: int, seed: int):
    stds, rhos = _colored_covariance_params(sigma, seed=seed)
    rng = np.random.default_rng(seed + 1)
    labels = rng.integers(0, COLORED_N_COVARIANCES, size=n_pts)
    noise = np.empty((n_pts, dim))
    for k in range(COLORED_N_COVARIANCES):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            continue
        cov_k = _pairwise_block_covariance(dim, stds[k], rhos[k])
        noise[idx] = rng.multivariate_normal(mean=np.zeros(dim), cov=cov_k, size=idx.size)
    return noise, stds, rhos


def kernel_marginal_2d(u, keep_modes, grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    pts = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], axis=-1)
    m = ttcalc.marginal(u, keep_modes)
    return np.asarray(m(pts)).reshape(X.shape)


def kernel_marginal_1d(u, mode, grid_x):
    m = ttcalc.marginal(u, (mode,))
    return np.asarray(m(jnp.asarray(grid_x)[:, None]))


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = D
    print(f"[density] sampling clean Ginzburg-Landau data (d={d}, n_train={N_TRAIN}, n_test={N_TEST}) ...", flush=True)
    log_z_true = ginzburg_landau_log_normalizer(d)
    clean_train = sample_ginzburg_landau(seed=0, n=N_TRAIN, d=d)
    clean_test = sample_ginzburg_landau(seed=1, n=N_TEST, d=d)
    y_true_test = np.asarray(ginzburg_landau_density(clean_test, log_z=log_z_true))

    # --- 1. Clean density fit (same higher-quality settings; no noise) ----
    print("[density] fitting clean TT density ...", flush=True)
    domain_clean = data_domain(clean_train)  # raw min/max, not robust_domain -- see data_domain's docstring
    clean_train_f = filter_to_domain(clean_train, domain_clean)
    bw_clean = select_gaussian_kde_bandwidth(clean_train_f)
    init_clean = init_tt_from_kde(
        clean_train_f, max_rank=RANK, domain=domain_clean, n_coeffs=(DEGREE,), bandwidth=bw_clean,
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )
    clean_fit, clean_hist = fit_tt(
        init_clean, clean_train_f, n_steps=CLEAN_STEPS, lr=CLEAN_LR, batch_size=CLEAN_BATCH_SIZE, log_every=CLEAN_STEPS,
    )
    err_clean = relative_l2_error(density_at(clean_fit, domain_clean, clean_test), y_true_test)
    print(f"[density] clean fit: relative L2 error = {err_clean:.4f}", flush=True)

    # --- 2. Colored (heteroscedastic, correlated) noise -------------------
    print(f"[density] corrupting with colored noise (sigma={SIGMA}, {COLORED_N_COVARIANCES} sampled covariances) ...", flush=True)
    noise, colored_stds, colored_rhos = sample_colored_noise(d, SIGMA, N_TRAIN, seed=2)
    noisy_train_full = jnp.asarray(np.asarray(clean_train) + noise)
    valid_noise, _, _ = sample_colored_noise(d, SIGMA, N_VALID, seed=20)
    clean_valid = sample_ginzburg_landau(seed=21, n=N_VALID, d=d)
    noisy_valid_full = jnp.asarray(np.asarray(clean_valid) + valid_noise)

    print("[density] selecting KDE bandwidth via least-squares cross-validation ...", flush=True)
    bandwidth = select_gaussian_kde_bandwidth(noisy_train_full)

    print("[density] computing a robust (percentile-trimmed) domain ...", flush=True)
    domain = robust_domain(noisy_train_full)
    noisy_train = filter_to_domain(noisy_train_full, domain)

    print("[density] cross-approximating sqrt(KDE) initialization ...", flush=True)
    init_noisy = init_tt_from_kde(
        noisy_train, max_rank=RANK, domain=domain, n_coeffs=(DEGREE,), bandwidth=bandwidth,
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )
    err_naive = relative_l2_error(density_at(init_noisy, domain, clean_test), y_true_test)
    print(f"[density] naive (untrained KDE cross-approx init): relative L2 error = {err_naive:.4f}  "
          f"(NAIVE_STEPS=0 -- see module docstring)", flush=True)

    print("[density] building the known colored-noise kernel U (from the average of the sampled covariances) ...", flush=True)
    eff_std, eff_rho = _colored_effective_std_rho(colored_stds, colored_rhos)
    print(f"  sampled std range=[{colored_stds.min():.3f}, {colored_stds.max():.3f}]  "
          f"rho range=[{colored_rhos.min():.3f}, {colored_rhos.max():.3f}]  "
          f"-> averaged (std={eff_std:.3f}, rho={eff_rho:.3f})", flush=True)
    u = build_pairwise_correlated_noise_kernel_tt(
        eff_std, eff_rho, domain, d, max_rank=NONISO_KERNEL_RANK,
        n_starts=max(N_STARTS, 32), gd_steps=max(GD_STEPS, 8),
    )
    noise_std_per_mode = jnp.full((d,), float(colored_stds.min()))
    noise_variance_per_axis = jnp.full((d,), float(np.mean(colored_stds**2)))

    print(f"[density] cross-approximating a separate (smaller-rank, higher-degree) sqrt(KDE) "
          f"init for the deconvolution (rank={DECONV_RANK}, degree={DECONV_DEGREE}) ...", flush=True)
    init_deconv = init_tt_from_kde(
        noisy_train, max_rank=DECONV_RANK, domain=domain, n_coeffs=(DECONV_DEGREE,), bandwidth=bandwidth,
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )

    print(f"[density] training deconvolved TT ({DECONV_STEPS} steps, ft-de-validated Sobolev-H1 recipe) ...", flush=True)
    deconv_fit, deconv_history, best_step, best_nll = fit_deconvolved_tt_validated(
        init_deconv, u, noisy_train, noisy_valid_full,
        degree=DECONV_DEGREE, steps=DECONV_STEPS, lr=DECONV_LR, batch_size=DECONV_BATCH_SIZE,
        noise_std=noise_std_per_mode, reg_lambda=REG_LAMBDA, per_axis_target_budget=REG_TARGET_BUDGET,
        smoothness=SMOOTHNESS, noise_variance_per_axis=noise_variance_per_axis,
        log_every=100,
        l2_eval=lambda tt: relative_l2_error(density_at(tt, domain, clean_test), y_true_test),
    )
    err_deconv = relative_l2_error(density_at(deconv_fit, domain, clean_test), y_true_test)
    print(f"[density] deconvolved fit: relative L2 error = {err_deconv:.4f}  "
          f"(checkpoint selected at step {best_step}/{DECONV_STEPS}, validation nll {best_nll:.4f})", flush=True)

    # --- 3. Committor from the true potential, the clean fit, and the ------
    #        deconvolved noisy fit -----------------------------------------
    true_pot = potentials.ginzburg_landau(d=d)
    true_ansatz, true_diag = solve_committor_for_potential(true_pot, "true potential")

    domain_clean_t = tuple(map(tuple, np.asarray(clean_fit.domain)))
    minus_clean, plus_clean = basins_from_data(
        lambda x: log_density_tt(clean_fit, x), np.asarray(clean_train_f), np.asarray(clean_fit.domain),
    )
    print(f"  [debug] clean basins: minus={np.round(minus_clean, 3)} plus={np.round(plus_clean, 3)}", flush=True)
    clean_pot = Potential(
        d=d, domain=domain_clean_t, log_density=domain_clipped_log_density(clean_fit, log_density_tt),
        sample=lambda seed, n: sample_ginzburg_landau(seed, n, d),
        basin_a=jnp.asarray(minus_clean), basin_b=jnp.asarray(plus_clean),
    )
    clean_ansatz, clean_diag = solve_committor_for_potential(clean_pot, "clean-estimated density")

    domain_deconv = tuple(map(tuple, np.asarray(deconv_fit.domain)))
    minus_deconv, plus_deconv = basins_from_data(
        lambda x: log_density_tt(deconv_fit, x), np.asarray(noisy_train), np.asarray(deconv_fit.domain),
    )
    print(f"  [debug] deconv basins: minus={np.round(minus_deconv, 3)} plus={np.round(plus_deconv, 3)}"
          f"  domain={np.round(np.asarray(deconv_fit.domain), 3).tolist()}", flush=True)
    deconv_pot = Potential(
        d=d, domain=domain_deconv, log_density=domain_clipped_log_density(deconv_fit, log_density_tt),
        sample=lambda seed, n: sample_ginzburg_landau(seed, n, d),
        basin_a=jnp.asarray(minus_deconv), basin_b=jnp.asarray(plus_deconv),
    )
    deconv_ansatz, deconv_diag = solve_committor_for_potential(deconv_pot, "deconvolved-noisy-estimated density")

    # --- Figure: a flow chart, 2D on top / 1D underneath --------------------
    # underlying density (+ noise) -> observed data -> deconvolved fit ->
    # committor, read left to right, shown twice on one shared 9-column grid
    # (5 content panels + 4 narrow arrow/plus connector columns, so the two
    # rows' panels line up exactly): 2D marginals on top, the matching 1D
    # marginals (and, in the last slot, the already-1D basin-to-basin
    # committor path) directly below. All settings that used to live in a
    # separate run-configuration panel are now in each panel's own title.
    grid = np.linspace(-PLOT_HALF_WIDTH, PLOT_HALF_WIDTH, GRID_N)
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    mode0, mode1 = MARGINAL_MODES

    Z_true = true_marginal_density(d, MARGINAL_MODES, grid, grid)
    Zk = kernel_marginal_2d(u, MARGINAL_MODES, grid, grid)
    Zo = histogram_density_2d(np.asarray(noisy_train), MARGINAL_MODES, grid, grid)
    Zf = exact_marginal_density_tt(deconv_fit, MARGINAL_MODES, grid, grid)
    density_vmax = max(float(np.max(z)) for z in (Z_true, Zo, Zf))
    density_levels = np.linspace(0.0, density_vmax, 41)

    z_true_1d = true_marginal_density_1d(d, mode0, grid)
    zk_1d = kernel_marginal_1d(u, mode0, grid)
    zo_1d = histogram_density_1d(np.asarray(noisy_train), mode0, grid)
    zf_1d = exact_marginal_density_tt_1d(deconv_fit, mode0, grid)

    fig = plt.figure(figsize=(16.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 9, width_ratios=[5, 0.7, 5, 0.7, 5, 0.7, 5, 0.7, 5], height_ratios=[5, 3.2, 3.2])
    content_cols = (0, 2, 4, 6, 8)
    ax_rho, ax_ker, ax_obs, ax_fit, ax_com = (fig.add_subplot(gs[0, c]) for c in content_cols)
    ax_rho1, ax_ker1, ax_obs1, ax_fit1, ax_com1 = (fig.add_subplot(gs[1, c]) for c in content_cols)
    for c, sym in ((1, "+"), (3, "→"), (5, "→"), (7, "→")):
        flow_connector(fig, gs[0, c], sym)
        flow_connector(fig, gs[1, c], sym)
    ax_err = fig.add_subplot(gs[2, 1:8])

    # -- top row: 2D marginals ------------------------------------------------
    ax_rho.contourf(X, Y, np.clip(Z_true, 0, None), levels=density_levels, cmap="viridis", extend="max")
    ax_rho.set_title(f"(a) true density $p$ ($d={d}$)\nexact marginal", fontsize=9.5)

    im_kernel = ax_ker.contourf(X, Y, np.clip(Zk, 0, None), levels=41, cmap="magma")
    ax_ker.set_title("(b) colored noise $U$\n"
                      f"std$\\in$[{colored_stds.min():.2f},{colored_stds.max():.2f}]\n"
                      f"$\\rho\\in$[{colored_rhos.min():.2f},{colored_rhos.max():.2f}]", fontsize=8.5)
    fig.colorbar(im_kernel, ax=ax_ker, orientation="vertical", aspect=14, shrink=0.8, pad=0.08)

    im_obs = ax_obs.contourf(X, Y, np.clip(Zo, 0, None), levels=density_levels, cmap="viridis", extend="max")
    ax_obs.set_title(f"(c) observed $p * U$\nhistogram, n={N_TRAIN:,}", fontsize=9.5)
    fig.colorbar(im_obs, ax=ax_obs, orientation="vertical", aspect=14, shrink=0.8, pad=0.08, label="density")

    ax_fit.contourf(X, Y, np.clip(Zf, 0, None), levels=density_levels, cmap="viridis", extend="max")
    ax_fit.set_title(f"(d) deconvolved fit (same scale)\nrank={DECONV_RANK} deg={DECONV_DEGREE}, rel. L2 err={err_deconv:.3f}", fontsize=9.5)

    contour_levels = np.linspace(0.0, 1.0, 21)
    line_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    Zq = committor_slice(deconv_ansatz, d, MARGINAL_MODES, grid, grid)
    im_com = ax_com.contourf(X, Y, Zq, levels=contour_levels, cmap="viridis", vmin=0, vmax=1, extend="both")
    cs = ax_com.contour(X, Y, Zq, levels=line_levels, colors="white", linewidths=1.0)
    ax_com.clabel(cs, fmt="%.1f", fontsize=7)
    ax_com.scatter([float(minus_deconv[mode0])], [float(minus_deconv[mode1])], marker="o", s=70, color="white", edgecolor="k")
    ax_com.annotate("A", (float(minus_deconv[mode0]), float(minus_deconv[mode1])), xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold")
    ax_com.scatter([float(plus_deconv[mode0])], [float(plus_deconv[mode1])], marker="*", s=160, color="white", edgecolor="k")
    ax_com.annotate("B", (float(plus_deconv[mode0]), float(plus_deconv[mode1])), xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold")
    ax_com.set_title(f"(e) committor\nloss={deconv_diag.loss_history[-1]:.3f}", fontsize=9.5)
    fig.colorbar(im_com, ax=ax_com, orientation="vertical", aspect=20, shrink=0.8, pad=0.08, label="committor q")

    for ax in (ax_rho, ax_ker, ax_obs, ax_fit, ax_com):
        ax.set_xlim(-PLOT_HALF_WIDTH, PLOT_HALF_WIDTH)
        ax.set_ylim(-PLOT_HALF_WIDTH, PLOT_HALF_WIDTH)
        ax.set_xlabel(f"$x_{mode0+1}$", fontsize=9)
        ax.set_ylabel(f"$x_{mode1+1}$", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    # -- bottom row: the matching 1D marginals (+ the basin-to-basin path) --
    ax_rho1.plot(grid, z_true_1d, color="tab:blue")
    ax_rho1.set_title(f"true $p(x_{mode0+1})$", fontsize=9.5)

    ax_ker1.plot(grid, zk_1d, color="tab:purple")
    ax_ker1.set_title(f"noise $U(x_{mode0+1})$", fontsize=9.5)

    ax_obs1.plot(grid, zo_1d, color="tab:blue")
    ax_obs1.set_title(f"observed $x_{mode0+1}$ (hist.)", fontsize=9.5)

    ax_fit1.plot(grid, zf_1d, color="tab:blue")
    ax_fit1.set_title(f"fitted $p(x_{mode0+1})$", fontsize=9.5)

    committor_panels = [
        (true_ansatz, true_pot, "true potential"),
        (clean_ansatz, clean_pot, "clean density"),
        (deconv_ansatz, deconv_pot, "deconvolved density"),
    ]
    for a, pot, panel_label in committor_panels:
        t = np.linspace(0.0, 1.0, 200)
        path = (1 - t[:, None]) * np.asarray(pot.basin_a)[None, :] + t[:, None] * np.asarray(pot.basin_b)[None, :]
        q_path = np.asarray(a(jnp.asarray(path)))
        ax_com1.plot(t, q_path, label=panel_label)
    ax_com1.axhline(0.0, color="gray", lw=0.5)
    ax_com1.axhline(1.0, color="gray", lw=0.5)
    ax_com1.set_xlabel(r"$t$  ($x_A\to x_B$)", fontsize=8.5)
    ax_com1.set_title("committor $q(t)$", fontsize=9.5)
    ax_com1.legend(fontsize=6.5, loc="lower right")

    for ax in (ax_rho1, ax_ker1, ax_obs1, ax_fit1):
        ax.set_xlabel(f"$x_{mode0+1}$", fontsize=8.5)
        ax.set_xlim(-PLOT_HALF_WIDTH, PLOT_HALF_WIDTH)
    for ax in (ax_rho1, ax_ker1, ax_obs1, ax_fit1, ax_com1):
        ax.tick_params(labelsize=7.5)

    # -- density-estimation quality bar chart --------------------------------
    err_names = ["clean\n(no noise)", "naive\n(untrained KDE init)", "deconvolved\n(ft-de, validated recipe)"]
    err_vals = [err_clean, err_naive, err_deconv]
    bars = ax_err.bar(err_names, err_vals, color=["0.6", "indianred", "seagreen"])
    ax_err.set_yscale("log")
    ax_err.set_ylabel("rel. L2 error vs. true $p$", fontsize=9)
    ax_err.set_title(f"(f) density-estimation quality  (deconv. checkpoint @ step {best_step}/{DECONV_STEPS})", fontsize=10.5)
    ax_err.tick_params(labelsize=8)
    ax_err.set_ylim(top=ax_err.get_ylim()[1] * 1.35)
    for b, v in zip(bars, err_vals):
        ax_err.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.3f}", ha="center", va="bottom", fontsize=9, clip_on=True)

    fig.suptitle(f"Ginzburg-Landau (d={d}, n={N_TRAIN:,}): noisy density estimation $\\to$ committor", fontsize=13)
    out_path = out_dir / "ginzburg_landau_noisy_pipeline.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[pipeline] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

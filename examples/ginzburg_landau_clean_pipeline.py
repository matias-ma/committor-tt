"""Density estimation -> committor, end to end, on Ginzburg-Landau, with NO
observation noise, at a higher dimension than the noisy pipeline
(``ginzburg_landau_density_and_committor.py``) uses.

This is the same dry run of the target pipeline (estimate a density from
samples alone, find its metastable basins from that estimate, solve the
committor between them, all checked against a known ground truth) but
isolating "does this scale to higher d" from "does this survive noise" --
the two noisy-pipeline questions are answered separately there. The
density-estimation recipe (dimension-tiered TT rank/degree, training step
budget, learning rate, batch size, KDE-vs-TT baseline comparison, and raw-
min/max domain) is taken directly from the sibling ft-de repository's own
noise-free GPU benchmark, ``ft-de/scripts/gl_gpu_noise_free.py`` -- not
reinvented here.

D=10 (not ft-de's own default "high" d=20) is deliberately chosen: running
ft-de's own unmodified script at d=8/10/14/20 (its own settings, nothing
changed) gives relative L2 errors of 0.083/0.28/0.56/1.02 respectively --
i.e. ft-de's own "high-dimension" d=20 tier is, as shipped, *worse* than
its own KDE baseline (1.02 vs. 0.9997), an untested/unvalidated setting we
confirmed is not specific to this project's adaptation (identical number
from ft-de's own script, unmodified). d=10 is the highest of the dimensions
we actually validated as giving a clearly good fit (0.28, well below the
KDE baseline's 0.97) rather than the highest ft-de's rank/degree table
merely *offers* a tier for.

1. Fit a squared-Chebyshev-TT density to clean Ginzburg-Landau samples at
   d=10 (ft-de's own "moderate/high" tier boundary: rank=10, degree=19),
   alongside a cross-validated Gaussian KDE baseline for reference.
2. Find the two metastable basins directly from the fitted density alone
   (no knowledge of the analytic GL potential -- see
   ``_gl_pipeline_common.basins_from_data``), and solve the committor
   between them. A committor solved from the true analytic potential is
   included as a reference.
3. Render a figure: true density (exact marginal) -> fitted TT density ->
   committor from the fit (2D marginal with level-set contours), plus a
   KDE-vs-TT error comparison and the basin-to-basin committor curves.

Usage: python examples/ginzburg_landau_clean_pipeline.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ftde.ginzburg_landau import ginzburg_landau_density, ginzburg_landau_log_normalizer, sample_ginzburg_landau
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde, select_gaussian_kde_bandwidth
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
    relative_l2_error,
    solve_committor_for_potential,
    true_marginal_density,
)

# ft-de's own dimension-tiered capacity (gl_gpu_noise_free._rank_degree_for_dim):
# dim<=8 -> (8,17), dim<=14 -> (10,19), else -> (14,21). D=10 falls in the
# middle tier -- see the module docstring for why this (not ft-de's own
# default "high" d=20) is the dimension actually used here.
D = 10
RANK, DEGREE = 10, 19
N_TRAIN, N_TEST = 20_000, 4_000
N_STARTS, GD_STEPS = 96, 15
BATCH_SIZE = 2048
N_STEPS, LR = 4000, 1e-3

MARGINAL_MODES = (0, 1)
GRID_N = 70


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = D
    print(f"[density] sampling clean Ginzburg-Landau data (d={d}, n_train={N_TRAIN}, n_test={N_TEST}) ...", flush=True)
    log_z_true = ginzburg_landau_log_normalizer(d)
    train = sample_ginzburg_landau(seed=0, n=N_TRAIN, d=d)
    test = sample_ginzburg_landau(seed=1, n=N_TEST, d=d)
    y_true_test = np.asarray(ginzburg_landau_density(test, log_z=log_z_true))

    print("[density] selecting KDE bandwidth via least-squares cross-validation ...", flush=True)
    bandwidth = select_gaussian_kde_bandwidth(train)

    print("[density] fitting Gaussian KDE baseline ...", flush=True)
    kde = make_gaussian_kde(train, bandwidth=bandwidth)
    err_kde = relative_l2_error(np.asarray(kde(test)), y_true_test)
    print(f"[density] KDE baseline: relative L2 error = {err_kde:.4f}", flush=True)

    # Raw min/max domain, not percentile-trimmed: unlike the noisy pipeline
    # (where trimming discards genuine noise outliers), this is clean data,
    # so every point is a true sample and trimming would just shrink the
    # domain a fitted squared-TT renormalizes over -- confirmed to make
    # things *worse* here, not better (a percentile-trimmed domain inflates
    # estimated density values relative to the true density's actual scale,
    # since the same total probability mass is squeezed into a smaller box).
    domain = data_domain(train)

    print(f"[density] cross-approximating sqrt(KDE) init (rank={RANK}, degree={DEGREE}) ...", flush=True)
    init = init_tt_from_kde(
        train, max_rank=RANK, domain=domain, n_coeffs=(DEGREE,), bandwidth=bandwidth,
        n_starts=N_STARTS, gd_steps=GD_STEPS,
    )
    print(f"[density] fitting TT density via NLL minimization ({N_STEPS} steps) ...", flush=True)
    fit, hist = fit_tt(init, train, n_steps=N_STEPS, lr=LR, batch_size=BATCH_SIZE, log_every=500)
    err_fit = relative_l2_error(density_at(fit, domain, test), y_true_test)
    print(f"[density] TT fit: relative L2 error = {err_fit:.4f}  (KDE baseline = {err_kde:.4f})", flush=True)

    # --- Committor from the true potential and from the fitted density -----
    true_pot = potentials.ginzburg_landau(d=d)
    true_ansatz, true_diag = solve_committor_for_potential(true_pot, "true potential")

    domain_fit_t = tuple(map(tuple, np.asarray(fit.domain)))
    minus_fit, plus_fit = basins_from_data(
        lambda x: log_density_tt(fit, x), np.asarray(train), np.asarray(fit.domain),
    )
    print(f"  [debug] fitted-density basins: minus={np.round(minus_fit, 3)} plus={np.round(plus_fit, 3)}", flush=True)
    fit_pot = Potential(
        d=d, domain=domain_fit_t, log_density=domain_clipped_log_density(fit, log_density_tt),
        sample=lambda seed, n: sample_ginzburg_landau(seed, n, d),
        basin_a=jnp.asarray(minus_fit), basin_b=jnp.asarray(plus_fit),
    )
    fit_ansatz, fit_diag = solve_committor_for_potential(fit_pot, "fitted clean density")

    # --- Figure --------------------------------------------------------------
    # 2 rows x 6 columns, same grid convention as the noisy pipeline figure:
    # row 1 -- true density (exact marginal), fitted TT density, committor
    # from the fit; row 2 -- KDE-vs-TT error comparison, basin-to-basin
    # committor curves, run-config text.
    grid_half_width = float(np.max(np.abs(np.asarray(test[:, MARGINAL_MODES]))) * 1.3)
    grid = np.linspace(-grid_half_width, grid_half_width, GRID_N)
    X, Y = np.meshgrid(grid, grid, indexing="xy")

    Z_true = true_marginal_density(d, MARGINAL_MODES, grid, grid)
    Z_fit = exact_marginal_density_tt(fit, MARGINAL_MODES, grid, grid)
    density_vmax = max(float(np.max(z)) for z in (Z_true, Z_fit))
    density_levels = np.linspace(0.0, density_vmax, 41)

    fig = plt.figure(figsize=(9.5, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    ax_rho, ax_fit_p, ax_com = (fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(3))
    ax_err = fig.add_subplot(gs[1, 0:2])
    ax_path = fig.add_subplot(gs[1, 2:4])
    ax_txt = fig.add_subplot(gs[1, 4:6])

    ax_rho.contourf(X, Y, np.clip(Z_true, 0, None), levels=density_levels, cmap="viridis", extend="max")
    ax_rho.set_title(f"(a) true density $p$ (d={d})\n(exact marginal)", fontsize=10)

    im_fit = ax_fit_p.contourf(X, Y, np.clip(Z_fit, 0, None), levels=density_levels, cmap="viridis", extend="max")
    ax_fit_p.set_title("(b) fitted TT density\n"
                        f"rel. L2 err={err_fit:.3f}\n(KDE baseline={err_kde:.3f})", fontsize=9.5)
    fig.colorbar(im_fit, ax=ax_fit_p, orientation="vertical", aspect=14, shrink=0.85, label="density")

    contour_levels = np.linspace(0.0, 1.0, 21)
    line_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    Zq = committor_slice(fit_ansatz, d, MARGINAL_MODES, grid, grid)
    im_com = ax_com.contourf(X, Y, Zq, levels=contour_levels, cmap="viridis", vmin=0, vmax=1, extend="both")
    cs = ax_com.contour(X, Y, Zq, levels=line_levels, colors="white", linewidths=1.0)
    ax_com.clabel(cs, fmt="%.1f", fontsize=7)
    ax_com.scatter([float(minus_fit[MARGINAL_MODES[0]])], [float(minus_fit[MARGINAL_MODES[1]])],
                   marker="o", s=70, color="white", edgecolor="k", label="basin A")
    ax_com.scatter([float(plus_fit[MARGINAL_MODES[0]])], [float(plus_fit[MARGINAL_MODES[1]])],
                   marker="*", s=160, color="white", edgecolor="k", label="basin B")
    ax_com.set_title("(c) committor from fitted density\n"
                      f"final loss={fit_diag.loss_history[-1]:.3f} (true={true_diag.loss_history[-1]:.3f})", fontsize=10)
    ax_com.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.colorbar(im_com, ax=ax_com, orientation="vertical", aspect=14, shrink=0.85, label="committor q")

    for ax in (ax_rho, ax_fit_p, ax_com):
        ax.set_xlim(float(grid[0]), float(grid[-1]))
        ax.set_ylim(float(grid[0]), float(grid[-1]))
        ax.set_xlabel(f"$x_{MARGINAL_MODES[0]+1}$", fontsize=9)
        ax.set_ylabel(f"$x_{MARGINAL_MODES[1]+1}$", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    err_names, err_vals = ["KDE\nbaseline", "TT fit\n(ft-de)"], [err_kde, err_fit]
    bars = ax_err.bar(err_names, err_vals, color=["0.75", "seagreen"])
    ax_err.set_yscale("log")
    ax_err.set_ylabel("rel. L2 error vs. true $p$", fontsize=9)
    ax_err.set_title("(d) density-estimation quality", fontsize=10.5)
    ax_err.tick_params(labelsize=8)
    ax_err.set_ylim(top=ax_err.get_ylim()[1] * 1.35)
    for b, v in zip(bars, err_vals):
        ax_err.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.3f}", ha="center", va="bottom", fontsize=8, clip_on=True)

    for a, pot, panel_label in [(true_ansatz, true_pot, "true potential"), (fit_ansatz, fit_pot, "fitted density")]:
        t = np.linspace(0.0, 1.0, 200)
        path = (1 - t[:, None]) * np.asarray(pot.basin_a)[None, :] + t[:, None] * np.asarray(pot.basin_b)[None, :]
        q_path = np.asarray(a(jnp.asarray(path)))
        ax_path.plot(t, q_path, label=panel_label)
    ax_path.axhline(0.0, color="gray", lw=0.5)
    ax_path.axhline(1.0, color="gray", lw=0.5)
    ax_path.set_xlabel(r"t  ($x_A\to x_B$)", fontsize=9)
    ax_path.set_ylabel("committor q", fontsize=9)
    ax_path.set_title("(e) basin-to-basin committor", fontsize=10.5)
    ax_path.tick_params(labelsize=8)
    ax_path.legend(fontsize=8)

    ax_txt.axis("off")
    ax_txt.text(
        0.0, 1.0,
        "\n".join([
            "(f) run configuration",
            f"d={d} (no noise)",
            f"n_train={N_TRAIN}, n_test={N_TEST}",
            f"density TT: rank={RANK}, deg={DEGREE}",
            f"  steps={N_STEPS}, lr={LR}",
            "committor TT: rank=6, deg=14",
            "  rho: 0->100 (4 stages)",
            "",
            "final committor loss:",
            f"  true      {true_diag.loss_history[-1]:.4f}",
            f"  fitted    {fit_diag.loss_history[-1]:.4f}",
        ]),
        transform=ax_txt.transAxes, ha="left", va="top", fontsize=9, family="monospace",
    )

    fig.suptitle(f"Ginzburg-Landau (d={d}, no noise): density estimation $\\to$ committor", fontsize=13)
    out_path = out_dir / "ginzburg_landau_clean_pipeline.png"
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[pipeline] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

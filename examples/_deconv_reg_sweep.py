"""TEMPORARY diagnostic: sweep the Sobolev-H1 ``smoothness`` regularizer
(ft-de's own validated recipe, see ginzburg_landau_density_and_committor.py's
DECONV_* comment block) for the noisy-GL deconvolution at a fixed D,
tracking relative L2 error against the true density at every checkpoint (not
just validation NLL) so we can see where/whether a given smoothness value
avoids the over-sharpening failure mode. Delete after use (see
instructions.txt: don't keep scratch scripts around).
"""
from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

from ftde.ginzburg_landau import ginzburg_landau_density, ginzburg_landau_log_normalizer, sample_ginzburg_landau
from ftde.init import init_tt_from_kde
from ftde.kde import select_gaussian_kde_bandwidth
from ftde.noise_kernel import build_pairwise_correlated_noise_kernel_tt

from _gl_pipeline_common import density_at, filter_to_domain, fit_deconvolved_tt_validated, relative_l2_error, robust_domain
from ginzburg_landau_density_and_committor import (
    DECONV_BATCH_SIZE,
    DECONV_DEGREE,
    DECONV_LR,
    DECONV_RANK,
    GD_STEPS,
    N_STARTS,
    N_TEST,
    N_TRAIN,
    N_VALID,
    NONISO_KERNEL_RANK,
    SIGMA,
    _colored_effective_std_rho,
    sample_colored_noise,
)

D = int(sys.argv[1]) if len(sys.argv) > 1 else 10
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SMOOTHNESS_VALUES = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.001, 0.01, 0.1, 1.0]

print(f"[sweep] D={D} STEPS={STEPS} SMOOTHNESS_VALUES={SMOOTHNESS_VALUES}", flush=True)

log_z_true = ginzburg_landau_log_normalizer(D)
clean_train = sample_ginzburg_landau(seed=0, n=N_TRAIN, d=D)
clean_test = sample_ginzburg_landau(seed=1, n=N_TEST, d=D)
y_true_test = np.asarray(ginzburg_landau_density(clean_test, log_z=log_z_true))

noise, colored_stds, colored_rhos = sample_colored_noise(D, SIGMA, N_TRAIN, seed=2)
noisy_train_full = jnp.asarray(np.asarray(clean_train) + noise)
valid_noise, _, _ = sample_colored_noise(D, SIGMA, N_VALID, seed=20)
clean_valid = sample_ginzburg_landau(seed=21, n=N_VALID, d=D)
noisy_valid_full = jnp.asarray(np.asarray(clean_valid) + valid_noise)

bandwidth = select_gaussian_kde_bandwidth(noisy_train_full)
domain = robust_domain(noisy_train_full)
noisy_train = filter_to_domain(noisy_train_full, domain)

init_deconv = init_tt_from_kde(
    noisy_train, max_rank=DECONV_RANK, domain=domain, n_coeffs=(DECONV_DEGREE,), bandwidth=bandwidth,
    n_starts=N_STARTS, gd_steps=GD_STEPS,
)

eff_std, eff_rho = _colored_effective_std_rho(colored_stds, colored_rhos)
u = build_pairwise_correlated_noise_kernel_tt(
    eff_std, eff_rho, domain, D, max_rank=NONISO_KERNEL_RANK,
    n_starts=max(N_STARTS, 32), gd_steps=max(GD_STEPS, 8),
)
noise_std_per_mode = jnp.full((D,), float(colored_stds.min()))
noise_variance_per_axis = jnp.full((D,), float(np.mean(colored_stds**2)))

for smoothness in SMOOTHNESS_VALUES:
    print(f"\n=== smoothness={smoothness} ===", flush=True)

    def l2_eval(tt):
        return relative_l2_error(density_at(tt, domain, clean_test), y_true_test)

    _, history, best_step, best_nll = fit_deconvolved_tt_validated(
        init_deconv, u, noisy_train, noisy_valid_full,
        degree=DECONV_DEGREE, steps=STEPS, lr=DECONV_LR, batch_size=DECONV_BATCH_SIZE,
        noise_std=noise_std_per_mode, reg_lambda=0.0,
        smoothness=smoothness, noise_variance_per_axis=noise_variance_per_axis,
        log_every=100, l2_eval=l2_eval,
    )
    print(f"[sweep] smoothness={smoothness}  NLL-selected checkpoint step={best_step}  "
          f"(see the per-step 'diagnostic rel. L2' lines above for the trajectory and its minimum)",
          flush=True)

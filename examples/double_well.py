"""Double-well committor sanity check.

Trains a committor for the d-dimensional double well
``V(x) = (x_1^2 - 1)^2 + alpha * sum_{k>=2} x_k^2`` at a few dimensions and
compares against the closed-form 1D reference (exact by symmetry), saving a
comparison figure and a loss-curve figure per dimension to ``figs/``.

Usage: python examples/double_well.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from committor import ansatz, potentials, training
from committor.diagnostics import relative_l2_error
from committor.domains import recommended_format
from committor.reference import double_well_1d_committor

BETA = 2.0
ALPHA = 0.3
# 1, 2: TT. 4, 6: Tucker (the 3-7D band). 10: TT again, to show it keeps
# working past the Tucker band -- TT is the format this project emphasizes
# since its cost stays linear in d.
DIMENSIONS = (1, 2, 4, 6, 10)


def run_one(d: int, out_dir: Path) -> None:
    fmt = recommended_format(d)
    print(f"[double_well] d={d} format={fmt}", flush=True)

    pot = potentials.double_well(d=d, beta=BETA, alpha=ALPHA)
    q_true = double_well_1d_committor(beta=BETA)

    a0 = ansatz.init(jax.random.PRNGKey(0), d=d, domain=pot.domain, rank=4, degree=16, format=fmt)
    a, diag = training.train_committor(
        a0, pot,
        final_rho=200.0, n_rho_stages=4, n_samples=4096,
        batch_size=256, steps_per_stage=400, learning_rate=5e-3, seed=0,
    )

    error = relative_l2_error(a, q_true, pot, n_samples=50_000, seed=999)
    print(f"[double_well] d={d}: relative L2 error = {error:.4f}", flush=True)

    xs = np.linspace(-1.8, 1.8, 400)
    x_grid = np.zeros((xs.shape[0], d))
    x_grid[:, 0] = xs
    q_approx = np.asarray(a(x_grid))
    q_ref = q_true(xs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, q_ref, label="exact (1D)", color="black", lw=2)
    axes[0].plot(xs, q_approx, label=f"{fmt} committor", color="tab:orange", ls="--")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("committor $q$")
    axes[0].set_title(f"Double well, d={d} (rel. L2 error={error:.3f})")
    axes[0].legend()

    axes[1].plot(diag.loss_history)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("gradient step")
    axes[1].set_ylabel("soft variational loss")
    axes[1].set_title("training loss")

    fig.tight_layout()
    out_path = out_dir / f"double_well_d{d}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[double_well] saved {out_path}", flush=True)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in DIMENSIONS:
        run_one(d, out_dir)


if __name__ == "__main__":
    main()

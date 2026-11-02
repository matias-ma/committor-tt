"""Ginzburg-Landau committor: the flagship example.

Trains a committor for the discretized Ginzburg-Landau chain (reusing
``ftde.ginzburg_landau`` for the equilibrium density/sampler). TT is the
primary format demonstrated here -- its cost is only linear in d, so it is
the one that scales to the higher-dimensional regime this whole approach is
for -- with a smaller Tucker demo in the 3-7D band alongside it, per this
project's TT(d>3)/Tucker(3-7D) convention. There is no closed-form GL
committor to compare against, so we check the qualitative signature of a
correct committor instead: q ~= 0 at the U_- minimizer, q ~= 1 at the U_+
minimizer, and a smooth monotone-ish transition along the straight-line path
between them (the "reaction coordinate" used in the arXiv paper's own GL
figures).

Usage: python examples/ginzburg_landau.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from committor import ansatz, potentials, training


def run_one(d: int, fmt: str, rank: int, out_dir: Path) -> None:
    print(f"[ginzburg_landau] d={d} format={fmt} rank={rank}", flush=True)

    pot = potentials.ginzburg_landau(d=d)

    a0 = ansatz.init(jax.random.PRNGKey(0), d=d, domain=pot.domain, rank=rank, degree=12, format=fmt)
    a, diag = training.train_committor(
        a0, pot,
        final_rho=100.0, n_rho_stages=4, n_samples=4096,
        batch_size=256, steps_per_stage=400, learning_rate=3e-3, seed=0,
    )
    print(f"[ginzburg_landau] d={d}: final loss = {diag.loss_history[-1]:.4f}", flush=True)

    # Straight-line path between the two homogeneous minimizers U_- and U_+.
    n_path = 200
    t = np.linspace(0.0, 1.0, n_path)
    path = (1 - t[:, None]) * np.asarray(pot.basin_a)[None, :] + t[:, None] * np.asarray(pot.basin_b)[None, :]
    q_path = np.asarray(a(jnp.asarray(path)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(diag.loss_history)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("gradient step")
    axes[0].set_ylabel("soft variational loss")
    axes[0].set_title(f"training loss (d={d}, {fmt})")

    axes[1].plot(t, q_path)
    axes[1].axhline(0.0, color="gray", lw=0.5)
    axes[1].axhline(1.0, color="gray", lw=0.5)
    axes[1].set_xlabel("t  (U_- -> U_+)")
    axes[1].set_ylabel("committor q")
    axes[1].set_title(f"q along the U_- - U_+ path (q(0)={q_path[0]:.3f}, q(1)={q_path[-1]:.3f})")

    if d > 1:
        keep_dims = (0, d // 2)
        model = a.finalize()
        # ``point`` fixes the coordinates of the d-2 dimensions *not* kept,
        # in ascending dimension order (see chebttge.plotting.slice_contour_data).
        model.plot_slice(keep_dims=keep_dims, point=np.zeros(d - 2), n=96, ax=axes[2])
        axes[2].set_title(f"q slice over dims {keep_dims}")

    fig.tight_layout()
    out_path = out_dir / f"ginzburg_landau_{fmt}_d{d}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[ginzburg_landau] saved {out_path}", flush=True)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # TT is the primary format here: its cost is only linear in d, so it is
    # the one that actually scales to the interesting (larger-d) regime.
    run_one(d=20, fmt="tt", rank=6, out_dir=out_dir)
    # Tucker demo in the 3-7D band, per instructions.txt, where its dense
    # core is still cheap -- secondary to the TT showcase above.
    run_one(d=5, fmt="tucker", rank=4, out_dir=out_dir)


if __name__ == "__main__":
    main()

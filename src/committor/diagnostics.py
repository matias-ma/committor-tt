"""Accuracy metrics and plotting hooks for a trained CommittorAnsatz.

The plotting/summary helpers are thin wrappers around chebdGE's own
``chebttge.diagnostics``/``chebttge.plotting`` (and the chebtuckerge
analogues) applied to :meth:`CommittorAnsatz.finalize`, so a trained
committor benefits from chebdGE's existing visualization/summary machinery
rather than reimplementing it.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from committor.ansatz import CommittorAnsatz
from committor.potentials import Potential

Array = jnp.ndarray


def relative_l2_error(
    ansatz: CommittorAnsatz,
    q_true: Callable[[np.ndarray], np.ndarray],
    potential: Potential,
    n_samples: int = 20_000,
    seed: int = 12345,
    reference_column: int = 0,
) -> float:
    """Monte Carlo relative L2(p) error of ``ansatz`` against ``q_true``.

    ``q_true`` is evaluated on ``x[:, reference_column]`` (the double-well
    oracle depends only on the first coordinate by symmetry); pass
    ``reference_column`` to match a different reference-coordinate
    convention if needed.
    """
    x = potential.sample(seed, n_samples)
    q_approx = np.asarray(ansatz(x))
    q_ref = np.asarray(q_true(np.asarray(x[:, reference_column])))

    num = float(np.mean((q_approx - q_ref) ** 2))
    den = float(np.mean(q_ref ** 2))
    if den < 1e-30:
        raise ValueError("relative_l2_error: reference committor is ~0 on the sampled points")
    return float(np.sqrt(num / den))


def summarize(ansatz: CommittorAnsatz):
    """Print chebdGE's own diagnostic summary of the finalized approximation."""
    model = ansatz.finalize()
    if ansatz.format == "tt":
        from chebttge.diagnostics import summarize_chebtt

        return summarize_chebtt(model)
    from chebtuckerge.diagnostics import summarize_chebtucker

    return summarize_chebtucker(model)


def plot_slice(ansatz: CommittorAnsatz, keep_dims, point, n: int = 128, **kwargs):
    """Plot a 2D slice of the trained committor via chebdGE's own plotting."""
    model = ansatz.finalize()
    return model.plot_slice(keep_dims=keep_dims, point=point, n=n, **kwargs)

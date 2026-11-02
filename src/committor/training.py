"""Gradient-based training of a CommittorAnsatz against the soft variational loss.

Follows the same JAX + optax "generate a fixed sample pool, then minibatch
with Adam" convention ft-de uses for its own TT/Tucker density fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from committor.ansatz import CommittorAnsatz
from committor.potentials import Potential
from committor.sampling import laplace_basin_sampler
from committor.variational import rho_schedule as make_rho_schedule
from committor.variational import soft_committor_loss

Array = jnp.ndarray


@dataclass
class TrainingDiagnostics:
    loss_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)


def train_committor(
    ansatz: CommittorAnsatz,
    potential: Potential,
    *,
    final_rho: float = 100.0,
    n_rho_stages: int = 4,
    n_samples: int = 4096,
    batch_size: int = 256,
    steps_per_stage: int = 500,
    learning_rate: float = 3e-3,
    seed: int = 0,
    pool_refresh: Optional[int] = None,
    final_lr_fraction: float = 1.0,
) -> Tuple[CommittorAnsatz, TrainingDiagnostics]:
    """Train ``ansatz`` to approximate the committor of ``potential``.

    One fixed pool of ``n_samples`` samples from p, p_A, p_B is drawn per
    rho-continuation stage, then minibatched through ``steps_per_stage`` Adam
    steps before moving to the next (larger) rho. ``p_A``/``p_B`` are the
    Laplace (Gaussian) approximation of the equilibrium density around each
    basin minimizer (see ``committor.sampling.laplace_basin_sample``), so
    directions unrelated to the transition automatically keep their correct
    local spread instead of being pinned to an arbitrary width.

    ``pool_refresh`` redraws the pool every that many steps instead of once per
    stage. In 10-20 dimensions a rank-8, degree-16 TT has ~10^4 parameters, and
    only a few percent of the equilibrium samples fall in the transition region
    where q varies, so a fixed pool of 16,384 is overfitted: the loss on fresh
    samples ends 20-40% above the pool's, and repeated solves differ by up to
    0.2 per point (confirmed on RECOVAR latent densities of alanine dipeptide).
    ``final_lr_fraction`` < 1 decays the learning rate
    over the last stage (cosine) to that fraction, so the solve ends at a
    converged iterate rather than wherever the last minibatch step left it. The
    defaults reproduce the original fixed-pool, constant-rate schedule.
    """
    # A TT/Tucker ansatz evaluates as a chain of matrix products (one per
    # site/mode), structurally the same as an RNN's hidden-state recurrence,
    # so it is prone to the same exploding-gradient behavior as the chain
    # gets longer -- confirmed empirically (loss blowing up into the
    # hundreds of thousands within a few hundred steps at d=10 without
    # clipping). Clipping the global gradient norm before Adam is the
    # standard fix and costs nothing when gradients are already well-scaled.
    rhos = make_rho_schedule(final_rho, n_rho_stages)
    lr = learning_rate
    if final_lr_fraction != 1.0:
        lr = optax.join_schedules(
            [optax.constant_schedule(learning_rate),
             optax.cosine_decay_schedule(learning_rate, steps_per_stage, alpha=final_lr_fraction)],
            [steps_per_stage * (len(rhos) - 1)])
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    params = eqx.filter(ansatz, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def step(ansatz, opt_state, x_p, x_a, x_b, rho):
        loss, grads = eqx.filter_value_and_grad(soft_committor_loss)(ansatz, x_p, x_a, x_b, rho)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(ansatz, eqx.is_inexact_array)
        )
        ansatz = eqx.apply_updates(ansatz, updates)
        return ansatz, opt_state, loss

    diagnostics = TrainingDiagnostics()
    refresh = pool_refresh or steps_per_stage
    losses = []

    sample_a = laplace_basin_sampler(potential, potential.basin_a)
    sample_b = laplace_basin_sampler(potential, potential.basin_b)
    key = jax.random.PRNGKey(seed)
    for stage, rho in enumerate(rhos):
        key, key_batches = jax.random.split(key)
        for i in range(steps_per_stage):
            if i % refresh == 0:
                # the first pool of each stage is seeded exactly as before pool_refresh existed
                s = seed + 100_003 * (i // refresh)
                x_p = potential.sample(s + 1000 * stage, n_samples)
                x_a = sample_a(s + 2000 * stage, n_samples)
                x_b = sample_b(s + 3000 * stage, n_samples)
            key_batches, subkey = jax.random.split(key_batches)
            idx = jax.random.randint(subkey, (batch_size,), 0, n_samples)
            ansatz, opt_state, loss = step(
                ansatz, opt_state, x_p[idx], x_a[idx], x_b[idx], rho
            )
            losses.append(loss)             # kept on device: a float() per step would sync every step
            diagnostics.rho_history.append(rho)
    diagnostics.loss_history = [float(v) for v in jnp.stack(losses)]

    return ansatz, diagnostics


def heldout_loss(
    ansatz: CommittorAnsatz,
    potential: Potential,
    *,
    rho: float = 100.0,
    n_samples: int = 65536,
    seed: int = 987_654,
    batch_size: int = 8192,
) -> dict:
    """The soft variational loss on fresh samples of p, p_A and p_B, with its standard error.

    The training loss (a minibatch of a few hundred points at the last step) is
    too noisy, and too optimistic once the pool is overfitted, to compare
    solves; this is the quantity to select them by. A fixed ``seed`` gives every
    solve on the same potential the same evaluation points, so differences
    between solves are paired and much more precise than the standard errors.
    """
    x_p = potential.sample(seed, n_samples)
    x_a = laplace_basin_sampler(potential, potential.basin_a)(seed + 1, n_samples)
    x_b = laplace_basin_sampler(potential, potential.basin_b)(seed + 2, n_samples)
    grad2, pen_a, pen_b = [], [], []
    for i in range(0, n_samples, batch_size):
        _, g = ansatz.value_and_grad(x_p[i:i + batch_size])
        grad2.append(jnp.sum(g ** 2, axis=-1))
        pen_a.append(ansatz(x_a[i:i + batch_size]) ** 2)
        pen_b.append((ansatz(x_b[i:i + batch_size]) - 1.0) ** 2)
    grad2, pen_a, pen_b = (jnp.concatenate(v) for v in (grad2, pen_a, pen_b))
    loss = grad2.mean() + rho * (pen_a.mean() + pen_b.mean())
    se = jnp.sqrt((grad2.var() + rho ** 2 * (pen_a.var() + pen_b.var())) / n_samples)
    return {"loss": float(loss), "loss_se": float(se), "dirichlet": float(grad2.mean()),
            "penalty_a": float(pen_a.mean()), "penalty_b": float(pen_b.mean())}

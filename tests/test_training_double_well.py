import jax
import numpy as np
import pytest

from committor import ansatz, potentials, training
from committor.diagnostics import relative_l2_error
from committor.reference import double_well_1d_committor


@pytest.mark.parametrize("d,fmt", [(1, "tt"), (3, "tucker")])
def test_double_well_committor_matches_closed_form(d, fmt):
    beta = 2.0
    pot = potentials.double_well(d=d, beta=beta, alpha=0.3)
    q_true = double_well_1d_committor(beta=beta)

    a = ansatz.init(jax.random.PRNGKey(0), d=d, domain=pot.domain, rank=4, degree=16, format=fmt)
    a, diag = training.train_committor(
        a, pot,
        final_rho=200.0, n_rho_stages=3, n_samples=2048,
        batch_size=256, steps_per_stage=300, learning_rate=5e-3, seed=0,
    )

    assert diag.loss_history[-1] < diag.loss_history[0]

    error = relative_l2_error(a, q_true, pot, n_samples=20_000, seed=999)
    assert error < 0.15


def test_refreshed_pool_decay_heldout_and_fit():
    """pool_refresh + final_lr_fraction still solve the double well, heldout_loss agrees with
    the training loss scale, and fit() reproduces a smooth guess at its own rank and degree."""
    beta = 2.0
    pot = potentials.double_well(d=2, beta=beta, alpha=0.3)
    q_true = double_well_1d_committor(beta=beta)

    a0 = ansatz.init(jax.random.PRNGKey(0), d=2, domain=pot.domain, rank=4, degree=16, format="tt")
    x = pot.sample(5, 4096)
    guess = lambda x: jax.nn.sigmoid(x[:, 0] / 0.3)  # noqa: E731
    a_fit, mse = ansatz.fit(a0, guess, x, steps=500)
    assert mse < 1e-3
    assert a_fit.tt_cores[0].shape == a0.tt_cores[0].shape

    a, diag = training.train_committor(
        a_fit, pot,
        final_rho=200.0, n_rho_stages=3, n_samples=2048,
        batch_size=256, steps_per_stage=300, learning_rate=5e-3, seed=0,
        pool_refresh=100, final_lr_fraction=0.01,
    )
    assert len(diag.loss_history) == 900
    assert relative_l2_error(a, q_true, pot, n_samples=20_000, seed=999) < 0.15
    h = training.heldout_loss(a, pot, rho=200.0, n_samples=8192)
    assert np.isfinite(h["loss"]) and h["loss_se"] < 0.2 * h["loss"]
    assert abs(h["loss"] - np.mean(diag.loss_history[-100:])) < 0.5 * h["loss"]

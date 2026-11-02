import jax
import jax.numpy as jnp
import numpy as np

from committor import ansatz
from committor.variational import rho_schedule, soft_committor_loss


def test_rho_schedule_geometric_and_ends_at_final():
    sched = rho_schedule(100.0, n_stages=4)
    assert len(sched) == 4
    assert sched[-1] == 100.0
    assert sched[0] < sched[1] < sched[2] < sched[3]


def test_loss_is_zero_for_constant_half_when_rho_zero():
    d = 2
    domain = ((-1.0, 1.0),) * d
    a = ansatz.init(jax.random.PRNGKey(0), d=d, domain=domain, rank=2, degree=4, init_scale=0.0)
    x_p = jax.random.uniform(jax.random.PRNGKey(1), (16, d))
    x_a = jax.random.uniform(jax.random.PRNGKey(2), (16, d))
    x_b = jax.random.uniform(jax.random.PRNGKey(3), (16, d))
    loss = soft_committor_loss(a, x_p, x_a, x_b, rho=0.0)
    # q is constant 0.5 -> grad is exactly zero everywhere, and rho=0 drops the
    # boundary terms, so the loss must vanish exactly.
    np.testing.assert_allclose(np.asarray(loss), 0.0, atol=1e-12)


def test_loss_decreases_with_a_gradient_step():
    import equinox as eqx
    import optax

    d = 2
    domain = ((-1.0, 1.0),) * d
    a = ansatz.init(jax.random.PRNGKey(4), d=d, domain=domain, rank=3, degree=6)
    x_p = jax.random.uniform(jax.random.PRNGKey(5), (64, d), minval=-1.0, maxval=1.0)
    x_a = jnp.tile(jnp.array([-0.5, 0.0]), (32, 1))
    x_b = jnp.tile(jnp.array([0.5, 0.0]), (32, 1))

    loss_fn = lambda model: soft_committor_loss(model, x_p, x_a, x_b, rho=10.0)
    loss0 = loss_fn(a)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(eqx.filter(a, eqx.is_inexact_array))
    grads = eqx.filter_grad(loss_fn)(a)
    updates, _ = optimizer.update(grads, opt_state, eqx.filter(a, eqx.is_inexact_array))
    a2 = eqx.apply_updates(a, updates)
    loss1 = loss_fn(a2)

    assert float(loss1) < float(loss0)

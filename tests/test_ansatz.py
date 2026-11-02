import jax
import jax.numpy as jnp
import numpy as np
import pytest

from committor import ansatz


@pytest.mark.parametrize("fmt", ["tt", "tucker"])
def test_init_gives_constant_half_before_training(fmt):
    d = 4
    domain = ((-2.0, 2.0),) * d
    a = ansatz.init(jax.random.PRNGKey(0), d=d, domain=domain, rank=3, degree=6,
                     format=fmt, init_scale=0.0)
    x = jnp.zeros((5, d))
    y = a(x)
    np.testing.assert_allclose(np.asarray(y), 0.5 * np.ones(5), atol=1e-10)


@pytest.mark.parametrize("fmt", ["tt", "tucker"])
def test_call_batched_and_single(fmt):
    d = 3
    domain = ((-1.0, 1.0),) * d
    a = ansatz.init(jax.random.PRNGKey(1), d=d, domain=domain, rank=2, degree=4, format=fmt)
    x_batch = jax.random.uniform(jax.random.PRNGKey(2), (7, d), minval=-1.0, maxval=1.0)
    y_batch = a(x_batch)
    assert y_batch.shape == (7,)
    y_single = a(x_batch[0])
    np.testing.assert_allclose(np.asarray(y_single), np.asarray(y_batch[0]), atol=1e-10)


@pytest.mark.parametrize("fmt", ["tt", "tucker"])
def test_value_and_grad_matches_autodiff(fmt):
    d = 3
    domain = ((-1.5, 1.5),) * d
    a = ansatz.init(jax.random.PRNGKey(3), d=d, domain=domain, rank=2, degree=5, format=fmt)
    x = jax.random.uniform(jax.random.PRNGKey(4), (6, d), minval=-1.0, maxval=1.0)

    value, grad = a.value_and_grad(x)

    def scalar(xi):
        return a(xi[None, :])[0]

    expected_grad = jax.vmap(jax.grad(scalar))(x)
    expected_value = a(x)

    np.testing.assert_allclose(np.asarray(value), np.asarray(expected_value), atol=1e-10)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected_grad), atol=1e-10)


def test_gradient_is_trainable_via_optax_filter():
    import equinox as eqx

    d = 2
    domain = ((-1.0, 1.0),) * d
    a = ansatz.init(jax.random.PRNGKey(5), d=d, domain=domain, rank=2, degree=4, format="tt")
    x = jax.random.uniform(jax.random.PRNGKey(6), (4, d))

    def loss(model):
        return jnp.mean(model(x) ** 2)

    grads = eqx.filter_grad(loss)(a)
    assert grads.tt_cores is not None
    assert any(np.any(np.asarray(g) != 0.0) for g in grads.tt_cores)


def test_warm_start_matches_target_function_reasonably():
    d = 2
    domain = ((-1.0, 1.0),) * d

    def guess_fn(x):
        return jnp.sum(x, axis=-1)

    a = ansatz.warm_start(guess_fn, d=d, domain=domain, rank=4, format="tt",
                           tol=1e-10, n_starts=8)
    x = jax.random.uniform(jax.random.PRNGKey(7), (20, d), minval=-1.0, maxval=1.0)
    got = np.asarray(a(x))
    expected = np.asarray(guess_fn(x))
    np.testing.assert_allclose(got, expected, atol=1e-6)

import jax.numpy as jnp
import numpy as np
import pytest

from committor import potentials


def test_double_well_log_density_shape():
    pot = potentials.double_well(d=3)
    x = jnp.zeros((5, 3))
    logp = pot.log_density(x)
    assert logp.shape == (5,)


def test_double_well_sample_matches_density_ranking():
    pot = potentials.double_well(d=1, beta=2.0)
    x = pot.sample(seed=0, n=5000)
    assert x.shape == (5000, 1)
    # Samples should cluster near the two wells at +-1.
    near_wells = np.mean(np.abs(np.abs(np.asarray(x[:, 0])) - 1.0) < 0.5)
    assert near_wells > 0.5


def test_double_well_basins():
    pot = potentials.double_well(d=2)
    assert np.allclose(pot.basin_a, [-1.0, 0.0])
    assert np.allclose(pot.basin_b, [1.0, 0.0])


def test_ginzburg_landau_matches_ftde_directly():
    from ftde.ginzburg_landau import GL_BETA, GL_LAM, ginzburg_landau_log_prob

    d = 5
    pot = potentials.ginzburg_landau(d=d)
    x = jnp.array(np.random.default_rng(0).normal(size=(10, d)))
    expected = ginzburg_landau_log_prob(x, beta=GL_BETA, lam=GL_LAM, h=1.0 / (d + 1))
    got = pot.log_density(x)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_ginzburg_landau_minimizers_are_near_plus_minus_one():
    d = 8
    pot = potentials.ginzburg_landau(d=d)
    assert np.mean(pot.basin_a) < -0.9
    assert np.mean(pot.basin_b) > 0.9


@pytest.mark.parametrize("d", [1, 4, 8])
def test_ginzburg_landau_sample_shape(d):
    pot = potentials.ginzburg_landau(d=d)
    x = pot.sample(seed=1, n=64)
    assert x.shape == (64, d)


def test_find_local_maxima_on_simple_gaussian():
    center = jnp.array([1.5, -2.0])

    def log_density(x):
        return -jnp.sum((x - center) ** 2)

    optima = potentials.find_local_maxima(log_density, np.array([[0.0, 0.0], [3.0, -3.0]]))
    assert optima.shape == (2, 2)
    np.testing.assert_allclose(optima[0], np.asarray(center), atol=1e-4)
    np.testing.assert_allclose(optima[1], np.asarray(center), atol=1e-4)

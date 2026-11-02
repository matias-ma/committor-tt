import numpy as np
import pytest

from committor.gl_reference import ginzburg_landau_exact_1d_marginal, ginzburg_landau_exact_adjacent_marginal


def test_1d_marginal_integrates_to_one_and_is_bimodal():
    d = 4
    grid = np.linspace(-3.0, 3.0, 400)
    z = ginzburg_landau_exact_1d_marginal(d, 0, grid, n_grid=800, x_max=3.0)
    total = np.trapezoid(z, grid)
    assert total == pytest.approx(1.0, abs=2e-2)

    def value_at(x):
        return z[np.argmin(np.abs(grid - x))]

    # Bistable around +-1, lower in between.
    assert value_at(1.0) > value_at(0.0)
    assert value_at(-1.0) > value_at(0.0)


def test_1d_marginal_rejects_out_of_range_index():
    grid = np.linspace(-2.0, 2.0, 21)
    with pytest.raises(ValueError):
        ginzburg_landau_exact_1d_marginal(4, 4, grid)
    with pytest.raises(ValueError):
        ginzburg_landau_exact_1d_marginal(4, -1, grid)


def test_marginal_integrates_to_one():
    d = 4
    grid = np.linspace(-3.0, 3.0, 400)
    z = ginzburg_landau_exact_adjacent_marginal(d, 0, grid, grid, n_grid=800, x_max=3.0)
    total = np.trapezoid(np.trapezoid(z, grid, axis=1), grid)
    assert total == pytest.approx(1.0, abs=2e-2)


def test_marginal_is_bimodal_and_positively_correlated():
    d = 4
    grid = np.linspace(-2.0, 2.0, 161)
    z = ginzburg_landau_exact_adjacent_marginal(d, 0, grid, grid, n_grid=800, x_max=3.0)

    def value_at(x, y):
        ix = np.argmin(np.abs(grid - x))
        iy = np.argmin(np.abs(grid - y))
        return z[iy, ix]

    # Adjacent coordinates are positively correlated (a domain wall between
    # them costs energy), so the aligned corners (+1,+1)/(-1,-1) should be
    # denser than the anti-aligned corners (+1,-1)/(-1,+1).
    aligned = value_at(1.0, 1.0) + value_at(-1.0, -1.0)
    anti_aligned = value_at(1.0, -1.0) + value_at(-1.0, 1.0)
    assert aligned > anti_aligned

    # Density near the origin (disordered configuration) should be lower
    # than at either homogeneous corner.
    assert value_at(0.0, 0.0) < value_at(1.0, 1.0)


def test_rejects_non_adjacent_or_out_of_range_index():
    grid = np.linspace(-2.0, 2.0, 21)
    with pytest.raises(ValueError):
        ginzburg_landau_exact_adjacent_marginal(4, 3, grid, grid)
    with pytest.raises(ValueError):
        ginzburg_landau_exact_adjacent_marginal(4, -1, grid, grid)

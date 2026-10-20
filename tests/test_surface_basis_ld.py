"""Fixed surface bases receive starry polynomial limb darkening for every law."""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from koala.surface import _surface_polynomial_ld
from models.common import get_I_power2
from models.detrend import _prepare_power2_poly


def test_quadratic_coefficients_pass_through():
    u = np.array([0.3, 0.2])
    np.testing.assert_array_equal(_surface_polynomial_ld(u, "quadratic"), u)
    grid = np.array([[0.3, 0.2], [0.35, 0.15]])
    np.testing.assert_array_equal(_surface_polynomial_ld(grid, "quadratic"), grid)


def test_power2_is_projected_onto_the_builder_polynomial():
    c, alpha = 0.6, 0.7
    poly = _surface_polynomial_ld(np.array([c, alpha]), "power2")
    mus, projection = _prepare_power2_poly()
    expected = np.asarray(projection) @ (1.0 - np.asarray(get_I_power2(c, alpha, mus)))
    assert poly.shape == (12,)
    np.testing.assert_allclose(poly, expected, rtol=1e-12)
    # The polynomial reproduces the power-2 intensity profile on the grid.
    x = 1.0 - np.asarray(mus)
    reconstructed = 1.0 - np.vander(x, N=13, increasing=True)[:, 1:] @ poly
    np.testing.assert_allclose(reconstructed, np.asarray(get_I_power2(c, alpha, mus)), atol=5e-3)
    # It is not the raw (c, alpha) pair that starry would otherwise have seen.
    assert not np.allclose(poly[:2], [c, alpha])


def test_power2_grid_converts_each_channel():
    grid = np.array([[0.6, 0.7], [0.5, 0.9]])
    poly = _surface_polynomial_ld(grid, "power2")
    assert poly.shape == (2, 12)
    np.testing.assert_allclose(poly[1], _surface_polynomial_ld(grid[1], "power2"))

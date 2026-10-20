"""Small CPU parity checks for power-2 recurrence and shared grid derivatives."""
import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.core.limb_dark import light_curve as stock_reference

from models.detrend import _prepare_power2_poly
from models.jaxoplanet import limb_dark_streamed as kernel

jax.config.update("jax_enable_x64", True)


def test_power2_recurrence_flux_and_derivatives():
    mus, projection = _prepare_power2_poly()
    b = jnp.concatenate((jnp.linspace(0.0, 1.4, 91), jnp.array([.86, 1.14])))
    def evaluate(theta, function):
        radius, c, alpha = theta
        u = projection @ (c * (1 - mus ** alpha))
        return function(u, b, radius)
    new = jax.jit(lambda t: (evaluate(t, kernel.stock_recurrence_light_curve), jax.jacfwd(lambda x: evaluate(x, kernel.stock_recurrence_light_curve))(t)))
    old = jax.jit(lambda t: (evaluate(t, stock_reference), jax.jacfwd(lambda x: evaluate(x, stock_reference))(t)))
    for theta in ([.14, .7, .6], [.01, .1, .1], [.4, .95, 1.8], [1.2, .5, .8]):
        actual, actual_grad = new(jnp.array(theta))
        expected, expected_grad = old(jnp.array(theta))
        assert np.all(np.isfinite(actual))
        assert np.all(np.isfinite(actual_grad))
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-9, rtol=1e-8)


def test_grid_shared_sensitivities_match_reference():
    from models.jaxoplanet import transit_grid as grid
    mus, projection = _prepare_power2_poly()
    phases = jnp.linspace(-.065, .065, 131)
    mask = jnp.abs(phases) < .06
    def reference_grid(theta):
        radius, c, alpha = theta
        u = projection @ (c * (1 - mus ** alpha))
        separation = grid.build_duration_transit_grid(impact=.45, radius_ratio=radius, num_nodes=129)
        nodes = stock_reference(u, separation, radius).at[-1].set(0.0)
        return grid.interpolate_duration_transit_grid(nodes, phases, mask, duration=.12, impact=.45, radius_ratio=radius)
    def new(theta):
        radius, c, alpha = theta
        u = projection @ (c * (1 - mus ** alpha))
        return grid._power2_duration_transit_sensitivities(stock_reference, theta, u, phases, mask, duration=.12, impact=.45, num_nodes=129)
    reference_gradient = jax.jit(jax.jacfwd(reference_grid))
    candidate_gradient = jax.jit(new)
    for radius in (.14, .55, .7):
        theta = jnp.array([radius, .7, .6])
        expected = reference_gradient(theta)
        actual = candidate_gradient(theta)
        # At grazing, nearly cancelled individual LD derivatives can have
        # large elementwise relative errors. Bound the full Jacobian's
        # relative norm as well as its absolute error (including zero entries).
        difference = np.asarray(actual - expected)
        assert np.all(np.isfinite(actual))
        assert np.linalg.norm(difference) / np.linalg.norm(expected) < 1e-8
        assert np.max(np.abs(difference)) < 1e-8


def test_grid_custom_reverse_in_maxted_coordinates():
    from models.jaxoplanet import transit_grid as grid
    from models.ld_parameterization import Power2MaxtedTransform
    mus, projection = _prepare_power2_poly()
    phases = jnp.linspace(-.06, .06, 97)
    mask = jnp.abs(phases) < .06
    weights = jnp.sin(jnp.arange(97))
    transform = Power2MaxtedTransform()
    def loss(parameters, custom):
        radius = parameters[0]
        c, alpha = transform.inv(parameters[1:])
        u = projection @ (1 - (1 - c * (1 - mus ** alpha)))
        if custom:
            flux = grid.interpolate_power2_duration_transit(
                stock_reference, c, alpha, u, phases, mask,
                duration=.12, impact=.45, radius_ratio=radius, num_nodes=129,
            )
        else:
            separation = grid.build_duration_transit_grid(impact=.45, radius_ratio=radius, num_nodes=129)
            nodes = stock_reference(u, separation, radius).at[-1].set(0.)
            flux = grid.interpolate_duration_transit_grid(nodes, phases, mask, duration=.12, impact=.45, radius_ratio=radius)
        return weights @ flux
    parameters = jnp.concatenate((jnp.array([.14]), transform(jnp.array([.7, .6]))))
    actual = jax.jit(jax.value_and_grad(lambda p: loss(p, True)))(parameters)
    # Differentiate the entire ordinary primal in forward mode. Its reverse
    # projection accumulates >1e-9 error in the nearly cancelled LD columns;
    # an independent long-double contraction confirms the custom VJP instead.
    reference = lambda p: loss(p, False)
    expected = jax.jit(lambda p: (reference(p), jax.jacfwd(reference)(p)))(parameters)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-9, rtol=1e-8)

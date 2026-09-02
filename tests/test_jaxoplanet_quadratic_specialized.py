import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxoplanet")
numpyro = pytest.importorskip("numpyro")

jax.config.update("jax_enable_x64", True)

from jaxoplanet.core.limb_dark import light_curve as stock_light_curve
from jaxoplanet.light_curves import limb_dark_light_curve as stock_orbit_curve
from jaxoplanet.orbits.transit import TransitOrbit

from models.jaxoplanet.limb_dark_quadratic import (
    light_curve as quadratic_light_curve,
    limb_dark_light_curve as quadratic_orbit_curve,
)
from models.jaxoplanet.limb_dark_quadratic_local_jvp import (
    light_curve as quadratic_local_jvp_light_curve,
    limb_dark_light_curve as quadratic_local_jvp_orbit_curve,
)
from models.jaxoplanet.core import compute_transit_model
from models.jaxoplanet.builder import create_vectorized_model


@pytest.mark.parametrize(
    ("u", "radius"),
    [
        ((0.3, 0.2), 0.12),
        ((0.0, 0.0), 0.12),
        ((0.999999, 0.000001), 0.12),
        ((0.4, -0.2), 0.12),
        ((0.3, 0.2), 1.2),
    ],
)
def test_specialized_quadratic_matches_stock_at_contacts(u, radius):
    inner = abs(1.0 - radius)
    outer = 1.0 + radius
    eps = 1.0e-12
    separations = jnp.asarray(
        [
            0.0,
            max(0.0, inner - eps),
            inner,
            inner + eps,
            outer - eps,
            outer,
            outer + eps,
            2.0,
        ],
        dtype=jnp.float64,
    )
    coefficients = jnp.asarray(u, dtype=jnp.float64)
    expected = stock_light_curve(
        coefficients, separations, jnp.float64(radius), order=10
    )
    actual = quadratic_light_curve(
        coefficients, separations, jnp.float64(radius), order=10
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=5.0e-15)


@pytest.mark.parametrize("impact", [0.0, 0.45, 1.095])
def test_specialized_quadratic_gradients_match_stock(impact):
    separations = jnp.linspace(
        max(0.0, impact), 1.25, 101, dtype=jnp.float64
    )
    weights = jnp.linspace(0.7, 1.3, separations.size, dtype=jnp.float64)

    def objective(theta, evaluator):
        return jnp.sum(
            evaluator(theta[:2], separations, theta[2], order=10) * weights
        )

    theta = jnp.asarray([0.31, 0.18, 0.12], dtype=jnp.float64)
    expected = jax.grad(objective)(theta, stock_light_curve)
    actual = jax.grad(objective)(theta, quadratic_light_curve)
    relative_l2 = np.linalg.norm(np.asarray(actual - expected)) / max(
        np.linalg.norm(np.asarray(expected)), np.finfo(np.float64).tiny
    )
    assert relative_l2 <= 5.0e-11
    np.testing.assert_allclose(actual, expected, rtol=5.0e-10, atol=5.0e-11)


def test_specialized_quadratic_orbit_wrapper_matches_stock():
    duration = jnp.float64(0.12)
    orbit = TransitOrbit(
        period=jnp.float64(3.2),
        duration=duration,
        time_transit=jnp.float64(0.0),
        impact_param=jnp.float64(0.37),
        radius_ratio=jnp.float64(0.11),
    )
    times = jnp.linspace(-0.08, 0.08, 81, dtype=jnp.float64)
    u = jnp.asarray([0.3, 0.2], dtype=jnp.float64)
    expected = stock_orbit_curve(orbit, u, order=10)(times)
    actual = quadratic_orbit_curve(orbit, u, order=10)(times)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=5.0e-15)


def test_coefficients_are_direct_u1_u2_not_kipping():
    # A central transit with u=[0.3, 0.2] must be evaluated as the literal
    # intensity I(mu)=1-u1(1-mu)-u2(1-mu)^2.  Transforming these as q1/q2
    # would produce a visibly different signal.
    separation = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    direct = quadratic_light_curve(
        jnp.asarray([0.3, 0.2]), separation, jnp.float64(0.12)
    )
    q1, q2 = 0.3, 0.2
    kipping_u = jnp.asarray(
        [2.0 * jnp.sqrt(q1) * q2, jnp.sqrt(q1) * (1.0 - 2.0 * q2)]
    )
    transformed = quadratic_light_curve(kipping_u, separation, jnp.float64(0.12))
    assert float(jnp.max(jnp.abs(direct - transformed))) > 1.0e-5


def test_compute_transit_model_opt_in_quadratic_route_matches_stock():
    times = jnp.linspace(-0.08, 0.08, 81, dtype=jnp.float64)
    params = {
        "period": jnp.asarray([3.2], dtype=jnp.float64),
        "duration": jnp.asarray([0.12], dtype=jnp.float64),
        "t0": jnp.asarray([0.0], dtype=jnp.float64),
        "b": jnp.asarray([0.37], dtype=jnp.float64),
        "rors": jnp.asarray([0.11], dtype=jnp.float64),
        "u": jnp.asarray([0.3, 0.2], dtype=jnp.float64),
        "_ld_profile": "quadratic",
    }
    expected = compute_transit_model(params, times, kernel="stock")
    actual = compute_transit_model(
        params, times, kernel="quadratic_specialized"
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=5.0e-15)

    def objective(radius, kernel):
        varied = {**params, "rors": jnp.atleast_1d(radius)}
        return jnp.sum(jnp.square(compute_transit_model(varied, times, kernel=kernel)))

    np.testing.assert_allclose(
        jax.grad(objective)(jnp.float64(0.11), "quadratic_specialized"),
        jax.grad(objective)(jnp.float64(0.11), "stock"),
        rtol=5.0e-10,
        atol=5.0e-12,
    )


def test_fixed_quadratic_builder_opt_in_matches_stock_likelihood():
    times = jnp.linspace(-0.14, 0.14, 101, dtype=jnp.float64)
    yerr = jnp.full((2, times.size), 1.5e-4, dtype=jnp.float64)
    geometry = {
        "mu_duration": jnp.asarray([0.12], dtype=jnp.float64),
        "mu_t0": jnp.asarray([0.0], dtype=jnp.float64),
        "mu_b": jnp.asarray([0.43], dtype=jnp.float64),
        "mu_depths": jnp.asarray([[0.014], [0.014]], dtype=jnp.float64),
        "PERIOD": jnp.asarray([3.1], dtype=jnp.float64),
        "ld_fixed": jnp.asarray(
            [[0.30, 0.20], [0.34, 0.17]], dtype=jnp.float64
        ),
    }

    def trace(kernel):
        model = create_vectorized_model(
            detrend_type="none",
            ld_mode="fixed",
            trend_mode="free",
            n_planets=1,
            ld_profile="quadratic",
            param_method="duration",
            jaxoplanet_kernel=kernel,
        )
        seeded = numpyro.handlers.seed(model, rng_seed=912)
        return numpyro.handlers.trace(seeded).get_trace(
            times,
            yerr,
            y=jnp.ones_like(yerr),
            precomputed_yerr_per_lc=jnp.full(
                (2,), 1.5e-4, dtype=jnp.float64
            ),
            **geometry,
        )

    stock = trace("stock")
    specialized = trace("quadratic_specialized")
    for name in ("rors", "depths", "u", "log_jitter"):
        np.testing.assert_array_equal(
            np.asarray(specialized[name]["value"]),
            np.asarray(stock[name]["value"]),
        )
    np.testing.assert_allclose(
        specialized["obs"]["fn"].loc,
        stock["obs"]["fn"].loc,
        rtol=2.0e-13,
        atol=5.0e-15,
    )


@pytest.mark.parametrize(
    ("u", "radius"),
    [
        ((0.3, 0.2), 0.12),
        ((0.0, 0.0), 0.12),
        ((0.4, -0.2), 0.12),
        ((0.3, 0.2), 1.2),
    ],
)
def test_local_jvp_quadratic_matches_stock_at_contacts(u, radius):
    inner = abs(1.0 - radius)
    outer = 1.0 + radius
    eps = 1.0e-12
    separation = jnp.asarray(
        [
            0.0,
            max(0.0, inner - eps),
            inner,
            inner + eps,
            outer - eps,
            outer,
            outer + eps,
            2.0,
        ],
        dtype=jnp.float64,
    )
    coefficients = jnp.asarray(u, dtype=jnp.float64)
    expected = stock_light_curve(coefficients, separation, radius, order=10)
    actual = quadratic_local_jvp_light_curve(
        coefficients, separation, radius, order=10
    )
    # The primal deliberately retains stock's solution-vector dot and is
    # bitwise equal for these branch/contact cases.
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_local_jvp_supports_free_ld_geometry_gradients_and_hessian():
    separation = jnp.linspace(0.05, 1.09, 89, dtype=jnp.float64)
    weights = jnp.linspace(0.7, 1.3, separation.size, dtype=jnp.float64)
    theta = jnp.asarray([0.31, 0.18, 0.02, 0.12], dtype=jnp.float64)

    def objective(value, evaluator):
        shifted = separation + 0.01 * value[2]
        return jnp.sum(
            evaluator(value[:2], shifted, value[3], order=10) * weights
        )

    expected_value = objective(theta, stock_light_curve)
    actual_value = objective(theta, quadratic_local_jvp_light_curve)
    np.testing.assert_array_equal(
        np.asarray(actual_value), np.asarray(expected_value)
    )

    expected_gradient = jax.grad(objective)(theta, stock_light_curve)
    actual_gradient = jax.grad(objective)(
        theta, quadratic_local_jvp_light_curve
    )
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=5.0e-13, atol=5.0e-13
    )

    expected_hessian = jax.jacrev(jax.grad(objective))(
        theta, stock_light_curve
    )
    actual_hessian = jax.jacrev(jax.grad(objective))(
        theta, quadratic_local_jvp_light_curve
    )
    assert np.all(np.isfinite(np.asarray(actual_hessian)))
    np.testing.assert_allclose(
        actual_hessian, expected_hessian, rtol=5.0e-12, atol=5.0e-12
    )

    _, expected_tangent = jax.jvp(
        lambda value: objective(value, stock_light_curve),
        (theta,),
        (jnp.ones_like(theta),),
    )
    _, actual_tangent = jax.jvp(
        lambda value: objective(value, quadratic_local_jvp_light_curve),
        (theta,),
        (jnp.ones_like(theta),),
    )
    np.testing.assert_allclose(
        actual_tangent, expected_tangent, rtol=5.0e-13, atol=5.0e-13
    )


def test_local_jvp_fixed_ld_geometry_gradient_matches_stock():
    separation = jnp.linspace(0.05, 1.09, 89, dtype=jnp.float64)
    weights = jnp.linspace(0.7, 1.3, separation.size, dtype=jnp.float64)
    fixed_u = jnp.asarray([0.30, 0.20], dtype=jnp.float64)
    geometry = jnp.asarray([0.02, 0.12], dtype=jnp.float64)

    def objective(value, evaluator):
        shifted = separation + 0.01 * value[0]
        return jnp.sum(
            evaluator(fixed_u, shifted, value[1], order=10) * weights
        )

    expected = jax.grad(objective)(geometry, stock_light_curve)
    actual = jax.grad(objective)(
        geometry, quadratic_local_jvp_light_curve
    )
    np.testing.assert_allclose(actual, expected, rtol=5.0e-13, atol=5.0e-13)


def test_compute_transit_model_local_jvp_route_matches_stock():
    times = jnp.linspace(-0.08, 0.08, 81, dtype=jnp.float64)
    params = {
        "period": jnp.asarray([3.2], dtype=jnp.float64),
        "duration": jnp.asarray([0.12], dtype=jnp.float64),
        "t0": jnp.asarray([0.0], dtype=jnp.float64),
        "b": jnp.asarray([0.37], dtype=jnp.float64),
        "rors": jnp.asarray([0.11], dtype=jnp.float64),
        "u": jnp.asarray([0.3, 0.2], dtype=jnp.float64),
        "_ld_profile": "quadratic",
    }
    expected = compute_transit_model(params, times, kernel="stock")
    actual = compute_transit_model(
        params, times, kernel="quadratic_local_jvp"
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_local_jvp_orbit_wrapper_matches_stock():
    orbit = TransitOrbit(
        period=jnp.float64(3.2),
        duration=jnp.float64(0.12),
        time_transit=jnp.float64(0.0),
        impact_param=jnp.float64(0.37),
        radius_ratio=jnp.float64(0.11),
    )
    times = jnp.linspace(-0.08, 0.08, 81, dtype=jnp.float64)
    u = jnp.asarray([0.3, 0.2], dtype=jnp.float64)
    expected = stock_orbit_curve(orbit, u, order=10)(times)
    actual = quadratic_local_jvp_orbit_curve(orbit, u, order=10)(times)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_fixed_quadratic_builder_local_jvp_matches_stock_likelihood():
    times = jnp.linspace(-0.14, 0.14, 101, dtype=jnp.float64)
    yerr = jnp.full((2, times.size), 1.5e-4, dtype=jnp.float64)
    geometry = {
        "mu_duration": jnp.asarray([0.12], dtype=jnp.float64),
        "mu_t0": jnp.asarray([0.0], dtype=jnp.float64),
        "mu_b": jnp.asarray([0.43], dtype=jnp.float64),
        "mu_depths": jnp.asarray([[0.014], [0.014]], dtype=jnp.float64),
        "PERIOD": jnp.asarray([3.1], dtype=jnp.float64),
        "ld_fixed": jnp.asarray(
            [[0.30, 0.20], [0.34, 0.17]], dtype=jnp.float64
        ),
    }

    def trace(kernel):
        model = create_vectorized_model(
            detrend_type="none",
            ld_mode="fixed",
            trend_mode="free",
            n_planets=1,
            ld_profile="quadratic",
            param_method="duration",
            jaxoplanet_kernel=kernel,
        )
        seeded = numpyro.handlers.seed(model, rng_seed=912)
        return numpyro.handlers.trace(seeded).get_trace(
            times,
            yerr,
            y=jnp.ones_like(yerr),
            precomputed_yerr_per_lc=jnp.full(
                (2,), 1.5e-4, dtype=jnp.float64
            ),
            **geometry,
        )

    stock = trace("stock")
    local = trace("quadratic_local_jvp")
    for name in ("rors", "depths", "u", "log_jitter"):
        np.testing.assert_array_equal(
            np.asarray(local[name]["value"]),
            np.asarray(stock[name]["value"]),
        )
    np.testing.assert_array_equal(
        np.asarray(local["obs"]["fn"].loc),
        np.asarray(stock["obs"]["fn"].loc),
    )

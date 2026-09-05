import numpy as np
import pytest


jaxoplanet = pytest.importorskip("jaxoplanet")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from models.jaxoplanet.core import compute_transit_model
import models.jaxoplanet.surface as surface_module
from models.jaxoplanet.surface import (
    build_keplerian_system,
    build_spotted_stellar_surface,
    compute_surface_model,
    validate_phase_curve_fluxes,
)


jax.config.update("jax_enable_x64", True)


def _params():
    return {
        "period": 1.0,
        "t0": 0.0,
        "a_rs": 8.0,
        "b": 0.0,
        "rors": 0.1,
        "ecc": 0.0,
        "omega": 0.0,
        "u": jnp.array([0.3, 0.2]),
        "eclipse_depth": 8.0e-4,
        "dayside_flux": 1.2e-3,
        "nightside_flux": 3.0e-4,
        "hotspot_offset": 0.5 * jnp.pi,
    }


def test_keplerian_system_retains_period_and_a_rs_with_zero_planet_mass():
    params = _params()
    system = build_keplerian_system(
        period=params["period"],
        t0=params["t0"],
        a_rs=params["a_rs"],
        b=params["b"],
        rors=params["rors"],
        ecc=params["ecc"],
        omega=params["omega"],
    )
    body = system.bodies[0]

    assert float(body.mass) == 0.0
    np.testing.assert_allclose(body.period, params["period"], rtol=0, atol=2e-15)
    np.testing.assert_allclose(body.semimajor, params["a_rs"], rtol=0, atol=0)
    xyz0 = body.relative_position(0.123)
    xyz1 = body.relative_position(1.123)
    for position0, position1 in zip(xyz0, xyz1):
        np.testing.assert_allclose(position0, position1, rtol=0, atol=2e-13)


def test_uniform_planet_has_physical_secondary_eclipse():
    params = _params()
    # A/R*=8 and Rp/R*=0.1 make the planet fully hidden at phase 0.5 and
    # completely unobscured at quadrature.
    time = jnp.array([0.25, 0.5, 0.75])
    signal = compute_surface_model(params, time, model="eclipse", order=10)

    np.testing.assert_allclose(
        np.asarray(signal)[[0, 2]], params["eclipse_depth"], rtol=0, atol=3e-14
    )
    np.testing.assert_allclose(signal[1], 0.0, rtol=0, atol=3e-14)


def _circle_overlap_area(radius_star, radius_planet, separation):
    r_star, r_planet, d = radius_star, radius_planet, separation
    first = r_planet**2 * np.arccos(
        (d**2 + r_planet**2 - r_star**2) / (2.0 * d * r_planet)
    )
    second = r_star**2 * np.arccos(
        (d**2 + r_star**2 - r_planet**2) / (2.0 * d * r_star)
    )
    triangle = 0.5 * np.sqrt(
        (-d + r_planet + r_star)
        * (d + r_planet - r_star)
        * (d - r_planet + r_star)
        * (d + r_planet + r_star)
    )
    return first + second - triangle


def test_eclipse_ingress_matches_independent_circle_overlap_geometry():
    params = _params()
    time = 0.519
    system = build_keplerian_system(
        period=params["period"],
        t0=params["t0"],
        a_rs=params["a_rs"],
        b=params["b"],
        rors=params["rors"],
    )
    x, y, z = system.bodies[0].relative_position(time)
    separation = float(jnp.hypot(x, y))
    assert float(z) < 0.0
    assert 1.0 - params["rors"] < separation < 1.0 + params["rors"]
    overlap = _circle_overlap_area(1.0, params["rors"], separation)
    visible_fraction = 1.0 - overlap / (np.pi * params["rors"] ** 2)
    expected = params["eclipse_depth"] * visible_fraction
    calculated = compute_surface_model(
        params, jnp.array([time]), model="eclipse", order=20
    )[0]
    np.testing.assert_allclose(calculated, expected, rtol=0, atol=2e-11)


def _mean_anomaly_at_true_anomaly(true_anomaly, eccentricity):
    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
    )
    return np.mod(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly), 2*np.pi)


def test_eccentric_secondary_eclipse_uses_keplerian_timing():
    params = {**_params(), "ecc": 0.25, "omega": 0.4}
    transit_anomaly = 0.5 * np.pi - params["omega"]
    eclipse_anomaly = 1.5 * np.pi - params["omega"]
    transit_mean = _mean_anomaly_at_true_anomaly(transit_anomaly, params["ecc"])
    eclipse_mean = _mean_anomaly_at_true_anomaly(eclipse_anomaly, params["ecc"])
    delta_mean = np.mod(eclipse_mean - transit_mean, 2.0 * np.pi)
    eclipse_time = params["t0"] + params["period"] * delta_mean / (2.0 * np.pi)

    at_eclipse, at_half_period = compute_surface_model(
        params,
        jnp.array([eclipse_time, params["t0"] + 0.5 * params["period"]]),
        model="eclipse",
        order=20,
    )
    np.testing.assert_allclose(at_eclipse, 0.0, rtol=0, atol=3e-13)
    assert float(at_half_period) > 0.9 * params["eclipse_depth"]


def test_eclipse_flux_and_geometry_gradients_match_finite_differences():
    base = _params()
    time = jnp.array([0.519])

    def model(values):
        a_rs, depth = values
        params = {**base, "a_rs": a_rs, "eclipse_depth": depth}
        return compute_surface_model(params, time, model="eclipse", order=20)[0]

    point = jnp.array([base["a_rs"], base["eclipse_depth"]])
    automatic = np.asarray(jax.grad(model)(point))
    steps = np.array([1e-4, 1e-7])
    finite = np.empty(2)
    for index, step in enumerate(steps):
        delta = np.zeros(2)
        delta[index] = step
        finite[index] = (
            float(model(point + delta)) - float(model(point - delta))
        ) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite, rtol=2e-4, atol=2e-8)


def test_phase_curve_extrema_match_user_day_and_nightside_fluxes():
    params = _params()
    # A positive pi/2 hotspot offset puts the minimum and maximum at the two
    # quadratures, where neither body occults the other.
    signal = compute_surface_model(
        params, jnp.array([0.25, 0.75]), model="phase_curve", order=10
    )

    np.testing.assert_allclose(signal[0], params["nightside_flux"], atol=4e-14)
    np.testing.assert_allclose(signal[1], params["dayside_flux"], atol=4e-14)


def test_spotless_starry_transit_matches_stock_keplerian_light_curve():
    params = _params()
    time = jnp.linspace(-0.025, 0.025, 31)
    starry_signal = compute_surface_model(params, time, model="transit", order=10)
    stock_signal = compute_transit_model(params, time)

    np.testing.assert_allclose(starry_signal, stock_signal, rtol=0, atol=3e-12)


def test_spots_rotate_and_sampled_contrast_is_differentiable():
    spots = (
        {
            "latitude": 0.0,
            "longitude": 0.0,
            "radius": 0.25,
            "contrast": 0.4,
        },
    )
    surface = build_spotted_stellar_surface(
        spots,
        rotation_period=2.0,
        degree=4,
        contrasts=jnp.array([0.4]),
    )
    from jaxoplanet.starry.light_curves import surface_light_curve

    flux = jax.vmap(
        lambda time: surface_light_curve(
            surface, theta=surface.rotational_phase(time), order=10
        )
    )(jnp.array([0.0, 0.5, 1.0]))
    assert float(jnp.ptp(flux)) > 1e-4

    params = {
        **_params(),
        "stellar_rotation_period": 2.0,
        "stellar_spot_contrast": jnp.array([0.4]),
    }

    def at_contrast(contrast):
        changed = {**params, "stellar_spot_contrast": jnp.array([contrast])}
        return compute_surface_model(
            changed,
            jnp.array([0.2]),
            model="transit",
            spots=spots,
            spot_degree=4,
            order=10,
        )[0]

    derivative = jax.grad(at_contrast)(0.4)
    assert bool(jnp.isfinite(derivative))
    assert abs(float(derivative)) > 1e-6


def test_spot_template_cache_never_retains_a_tracer_between_compilations():
    spots = (
        {
            "latitude": 0.1,
            "longitude": -0.2,
            "radius": 0.2,
            "contrast": 0.3,
        },
    )
    surface_module._unit_spot_template.cache_clear()

    def one_coefficient(contrast):
        surface = build_spotted_stellar_surface(
            spots,
            rotation_period=2.0,
            degree=3,
            contrasts=jnp.array([contrast]),
        )
        return surface.y.todense()[1]

    first_trace = jax.jit(one_coefficient)(0.3)
    fresh_trace_gradient = jax.grad(one_coefficient)(0.4)
    assert bool(jnp.isfinite(first_trace))
    assert bool(jnp.isfinite(fresh_trace_gradient))


def test_spot_longitude_is_referenced_to_transit_epoch():
    spots = (
        {
            "latitude": 0.0,
            "longitude": 0.2,
            "radius": 0.2,
            "contrast": 0.3,
        },
    )
    params = {
        **_params(),
        "stellar_rotation_period": 3.7,
        "stellar_spot_contrast": jnp.array([0.3]),
    }
    relative_time = jnp.array([0.12, 0.23])
    at_zero = compute_surface_model(
        params,
        relative_time,
        model="transit",
        spots=spots,
        spot_degree=3,
        order=10,
    )
    epoch = 60_000.123
    shifted = compute_surface_model(
        {**params, "t0": epoch},
        relative_time + epoch,
        model="transit",
        spots=spots,
        spot_degree=3,
        order=10,
    )
    np.testing.assert_allclose(shifted, at_zero, rtol=0, atol=2e-11)


def test_invalid_surface_inputs_fail_with_direct_messages():
    params = _params()
    with pytest.raises(ValueError, match="surface model"):
        compute_surface_model(params, jnp.array([0.0]), model="other")
    with pytest.raises(ValueError, match="stellar_rotation_period"):
        compute_surface_model(
            params,
            jnp.array([0.0]),
            model="transit",
            spots=(
                {
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "radius": 0.2,
                    "contrast": 0.2,
                },
            ),
        )
    with pytest.raises(ValueError, match="five times"):
        validate_phase_curve_fluxes(1.0e-3, 1.0e-4)
    validate_phase_curve_fluxes(1.0e-3, 2.0e-4)

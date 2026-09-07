import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxoplanet")
numpyro = pytest.importorskip("numpyro")

jax.config.update("jax_enable_x64", True)

from jaxoplanet.core.limb_dark import light_curve as stock_core_light_curve
from jaxoplanet.light_curves import limb_dark_light_curve as stock_orbit_light_curve
from jaxoplanet.orbits.transit import TransitOrbit

from models.common import get_I_power2
from models.detrend import _prepare_power2_poly
from models.jaxoplanet.core import (
    build_transit_window_indices,
    build_transit_phase_offsets,
    compute_transit_model,
    resolve_jaxoplanet_kernel,
)
from models.jaxoplanet.builder import create_vectorized_model, create_whitelight_model
from models.jaxoplanet.limb_dark_streamed import (
    light_curve as streamed_core_light_curve,
    limb_dark_light_curve as streamed_orbit_light_curve,
)


MUS, PROJECTION = _prepare_power2_poly(degree=12)


def _power2_coefficients(c=0.62, alpha=0.73):
    return PROJECTION @ (1.0 - get_I_power2(c, alpha, MUS))


@pytest.mark.parametrize(
    ("c", "alpha", "r"),
    [
        (0.62, 0.73, 0.12),
        (1.0e-8, 0.001, 0.12),
        (1.0 - 1.0e-8, 1.0 - 1.0e-8, 0.12),
        (0.4, 0.5, 1.2),
    ],
)
def test_streamed_core_matches_stock_at_contacts_and_edges(c, alpha, r):
    u = _power2_coefficients(c, alpha)
    separations = jnp.asarray(
        [
            0.0,
            abs(1.0 - r),
            max(0.0, abs(1.0 - r) - 1.0e-12),
            abs(1.0 - r) + 1.0e-12,
            1.0 + r - 1.0e-12,
            1.0 + r,
            1.0 + r + 1.0e-12,
            2.0,
        ],
        dtype=jnp.float64,
    )
    expected = stock_core_light_curve(u, separations, jnp.float64(r), order=10)
    actual = streamed_core_light_curve(u, separations, jnp.float64(r), order=10)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=5.0e-12)


def test_streamed_orbit_matches_stock_at_exact_duration_contacts():
    duration = jnp.float64(0.12)
    orbit = TransitOrbit(
        period=jnp.float64(3.2),
        duration=duration,
        time_transit=jnp.float64(0.0),
        impact_param=jnp.float64(0.37),
        radius_ratio=jnp.float64(0.11),
    )
    eps = jnp.float64(1.0e-12)
    times = jnp.asarray(
        [-duration / 2 - eps, -duration / 2, -duration / 2 + eps,
         0.0, duration / 2 - eps, duration / 2, duration / 2 + eps],
        dtype=jnp.float64,
    )
    u = _power2_coefficients()
    expected = stock_orbit_light_curve(orbit, u, order=10)(times)
    actual = streamed_orbit_light_curve(orbit, u, order=10)(times)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-11, atol=5.0e-12)


@pytest.mark.parametrize("impact", [0.05, 0.45, 1.095])
def test_streamed_orbit_gradients_match_stock(impact):
    times = jnp.linspace(-0.055, 0.055, 41, dtype=jnp.float64)

    def objective(theta, light_curve_builder):
        c, alpha, radius = theta
        orbit = TransitOrbit(
            period=jnp.float64(2.7),
            duration=jnp.float64(0.12),
            time_transit=jnp.float64(0.0),
            impact_param=jnp.float64(impact),
            radius_ratio=radius,
        )
        signal = light_curve_builder(
            orbit, _power2_coefficients(c, alpha), order=10
        )(times)
        return jnp.sum(jnp.square(signal))

    theta = jnp.asarray([0.61, 0.71, 0.12], dtype=jnp.float64)
    expected = jax.grad(objective)(theta, stock_orbit_light_curve)
    actual = jax.grad(objective)(theta, streamed_orbit_light_curve)
    relative_l2 = np.linalg.norm(np.asarray(actual - expected)) / max(
        np.linalg.norm(np.asarray(expected)), np.finfo(np.float64).tiny
    )
    assert relative_l2 <= 5.0e-9
    np.testing.assert_allclose(actual, expected, rtol=2.0e-7, atol=1.0e-10)


def test_kernel_routing_is_conservative():
    assert resolve_jaxoplanet_kernel(
        "auto", ld_profile="power2", degree=12
    ) == "streamed"
    assert resolve_jaxoplanet_kernel(
        "streamed", ld_profile="quadratic", degree=2
    ) == "stock"
    assert resolve_jaxoplanet_kernel(
        "streamed", ld_profile="power2", degree=12, keplerian=True
    ) == "stock"
    assert resolve_jaxoplanet_kernel(
        "auto", ld_profile="power2", degree=10
    ) == "stock"
    for removed in (
        "fused",
        "native_power2",
        "quadratic_specialized",
        "quadratic_local_jvp",
        "unknown",
    ):
        with pytest.raises(ValueError, match="jaxoplanet_kernel"):
            resolve_jaxoplanet_kernel(removed)


def test_builders_reject_removed_internal_options():
    for parameterization in ("decorrelated_linear", "latent_gaussian"):
        with pytest.raises(ValueError, match="ld_parameterization"):
            create_whitelight_model(ld_parameterization=parameterization)
        with pytest.raises(ValueError, match="ld_parameterization"):
            create_vectorized_model(ld_parameterization=parameterization)
    with pytest.raises(ValueError, match="two_spot_ordering"):
        create_whitelight_model(two_spot_ordering="ordered")
    with pytest.raises(ValueError, match="jitter_prior"):
        create_vectorized_model(jitter_prior="log_uniform")


def test_precomputed_phase_path_matches_stock_orbit_and_gradients():
    times = jnp.linspace(-0.08, 0.08, 101, dtype=jnp.float64)
    base = {
        "period": jnp.asarray([3.1], dtype=jnp.float64),
        "duration": jnp.asarray([0.12], dtype=jnp.float64),
        "t0": jnp.asarray([0.0], dtype=jnp.float64),
        "b": jnp.asarray([0.42], dtype=jnp.float64),
        "u": _power2_coefficients(),
        "_ld_profile": "power2",
    }
    offsets, mask = build_transit_phase_offsets(
        times, base["period"], base["t0"], base["duration"]
    )

    def objective(radius, shared_phase):
        params = {**base, "rors": jnp.atleast_1d(radius)}
        if shared_phase:
            params["_transit_phase_offsets"] = offsets
            params["_transit_phase_mask"] = mask
        return jnp.sum(jnp.square(compute_transit_model(params, times, kernel="stock")))

    radius = jnp.float64(0.12)
    direct = compute_transit_model({**base, "rors": jnp.atleast_1d(radius)}, times,
                                   kernel="stock")
    shared = compute_transit_model(
        {
            **base,
            "rors": jnp.atleast_1d(radius),
            "_transit_phase_offsets": offsets,
            "_transit_phase_mask": mask,
        },
        times,
        kernel="stock",
    )
    np.testing.assert_allclose(shared, direct, rtol=0.0, atol=5.0e-13)
    np.testing.assert_allclose(
        jax.grad(objective)(radius, True),
        jax.grad(objective)(radius, False),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_power2_builder_auto_trace_matches_stock_with_window_and_shared_phase():
    times = jnp.linspace(-0.18, 0.18, 121, dtype=jnp.float64)
    yerr = jnp.full((2, times.size), 1.5e-4, dtype=jnp.float64)
    geometry = {
        "mu_duration": jnp.asarray([0.12], dtype=jnp.float64),
        "mu_t0": jnp.asarray([0.0], dtype=jnp.float64),
        "mu_b": jnp.asarray([0.43], dtype=jnp.float64),
        "mu_depths": jnp.asarray([[0.014], [0.014]], dtype=jnp.float64),
        "PERIOD": jnp.asarray([3.1], dtype=jnp.float64),
        "ld_fixed": jnp.asarray([[0.58, 0.72], [0.63, 0.67]], dtype=jnp.float64),
    }
    indices = build_transit_window_indices(
        times,
        geometry["PERIOD"],
        geometry["mu_t0"],
        geometry["mu_duration"],
    )

    def trace(kernel):
        model = create_vectorized_model(
            detrend_type="none",
            ld_mode="fixed",
            trend_mode="free",
            n_planets=1,
            ld_profile="power2",
            param_method="duration",
            transit_window="auto",
            transit_window_indices=indices,
            jaxoplanet_kernel=kernel,
        )
        seeded = numpyro.handlers.seed(model, rng_seed=321)
        return numpyro.handlers.trace(seeded).get_trace(
            times,
            yerr,
            y=jnp.ones_like(yerr),
            precomputed_yerr_per_lc=jnp.full((2,), 1.5e-4, dtype=jnp.float64),
            **geometry,
        )

    stock = trace("stock")
    automatic = trace("auto")
    for sample_name in ("rors", "depths", "c1", "c2", "log_jitter"):
        np.testing.assert_array_equal(
            np.asarray(automatic[sample_name]["value"]),
            np.asarray(stock[sample_name]["value"]),
        )
    np.testing.assert_allclose(
        automatic["obs"]["fn"].loc,
        stock["obs"]["fn"].loc,
        rtol=0.0,
        atol=5.0e-11,
    )
    np.testing.assert_array_equal(
        np.asarray(automatic["obs"]["fn"].scale),
        np.asarray(stock["obs"]["fn"].scale),
    )

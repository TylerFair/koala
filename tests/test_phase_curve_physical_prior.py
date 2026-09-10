import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
numpyro = pytest.importorskip("numpyro")
dist = pytest.importorskip("numpyro.distributions")

from models.jaxoplanet.builder import (
    _conditional_phase_flux_from_quantile,
    _positive_normal_interval_log_mass,
    _sample_surface_parameters,
    phase_flux_to_conditional_quantile,
)


jax.config.update("jax_enable_x64", True)


def _spec(name, value, width):
    from koala.config import ParameterSpec

    if width > 0.0:
        return (ParameterSpec(name, "gaussian", value, width),)
    return (ParameterSpec(name, "fixed", value),)


def _config(day_width=2e-4, night_width=8e-5):
    return {
        "model": "phase_curve",
        "spots": (),
        "dayside_flux": np.array([1.2e-3]),
        "dayside_flux_prior_width": np.array([day_width]),
        "dayside_flux_spec": _spec("dayside_flux_ppm", 1.2e-3, day_width),
        "nightside_flux": np.array([2.5e-4]),
        "nightside_flux_prior_width": np.array([night_width]),
        "nightside_flux_spec": _spec("nightside_flux_ppm", 2.5e-4, night_width),
        "hotspot_offset": np.array([0.2]),
        "hotspot_offset_prior_width": np.array([0.1]),
        "hotspot_offset_spec": _spec("hotspot_offset_deg", 0.2, 0.1),
    }


def test_conditional_quantile_round_trip_and_joint_density_identity():
    conditioning = 1.2e-3
    center = 2.5e-4
    width = 8e-5
    value = 2.7e-4
    quantile = phase_flux_to_conditional_quantile(
        value, conditioning, center, width
    )
    rebuilt = _conditional_phase_flux_from_quantile(
        quantile, conditioning, center, width
    )
    np.testing.assert_allclose(rebuilt, value, rtol=0, atol=2e-16)

    # Under the inverse-CDF map, conditional density times its Jacobian is
    # one. Multiplying by the interval mass recovers the original positive-TN
    # density restricted to the physical interval exactly.
    jacobian = jax.grad(
        lambda q: _conditional_phase_flux_from_quantile(
            q, conditioning, center, width
        )
    )(quantile)
    transformed_log_density = _positive_normal_interval_log_mass(
        conditioning, center, width
    )
    original_with_jacobian = (
        dist.TruncatedNormal(center, width, low=0.0).log_prob(value)
        + jnp.log(jacobian)
    )
    np.testing.assert_allclose(
        transformed_log_density, original_with_jacobian, rtol=0, atol=2e-12
    )


def test_quantile_parameterization_is_smooth_and_strictly_physical_near_edges():
    conditioning = 1.2e-3
    center = 2.5e-4
    width = 8e-5
    quantiles = jnp.array([1e-6, 0.01, 0.5, 0.99, 1.0 - 1e-6])
    values = jax.vmap(
        lambda q: _conditional_phase_flux_from_quantile(
            q, conditioning, center, width
        )
    )(quantiles)
    gradients = jax.vmap(
        jax.grad(
            lambda q: _conditional_phase_flux_from_quantile(
                q, conditioning, center, width
            )
        )
    )(quantiles)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(jnp.isfinite(gradients)))
    assert bool(jnp.all(gradients > 0.0))
    assert bool(jnp.all(values > conditioning / 5.0))
    assert bool(jnp.all(values < 5.0 * conditioning))


def test_conditional_mapping_remains_finite_in_extreme_normal_tail():
    # The entire allowed interval is more than 59 sigma above this prior's
    # center. Direct subtraction of two normal CDFs rounds to zero here.
    conditioning = 0.025
    center = 2.5e-4
    width = 8e-5
    quantiles = jnp.array([1e-6, 0.2, 0.8, 1.0 - 1e-6])

    def mapped(q, fixed_flux):
        return _conditional_phase_flux_from_quantile(
            q, fixed_flux, center, width
        )

    values = jax.vmap(mapped, in_axes=(0, None))(quantiles, conditioning)
    quantile_gradients = jax.vmap(
        jax.grad(mapped, argnums=0), in_axes=(0, None)
    )(quantiles, conditioning)
    mass = _positive_normal_interval_log_mass(conditioning, center, width)
    mass_gradient = jax.grad(
        lambda fixed_flux: _positive_normal_interval_log_mass(
            fixed_flux, center, width
        )
    )(conditioning)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert bool(jnp.all(jnp.isfinite(quantile_gradients)))
    assert bool(jnp.isfinite(mass))
    assert bool(jnp.isfinite(mass_gradient))
    assert bool(jnp.all(values >= conditioning / 5.0))
    assert bool(jnp.all(values <= 5.0 * conditioning))


@pytest.mark.parametrize(
    ("day_width", "night_width", "sample_site", "physical_site"),
    [
        (2e-4, 8e-5, "_nightside_flux_quantile_0", "_nightside_flux_0"),
        (2e-4, 0.0, "_dayside_flux_quantile_0", "_dayside_flux_0"),
        (0.0, 8e-5, "_nightside_flux_quantile_0", "_nightside_flux_0"),
    ],
)
def test_free_flux_combinations_keep_physical_deterministics_and_smooth_latent(
    day_width, night_width, sample_site, physical_site
):
    def model():
        return _sample_surface_parameters(
            _config(day_width, night_width), 1, num_lcs=4
        )

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(12))
    ).get_trace()
    assert trace[sample_site]["type"] == "sample"
    assert trace[physical_site]["type"] == "deterministic"
    day = trace["dayside_flux"]["value"]
    night = trace["nightside_flux"]["value"]
    ratio = jnp.maximum(day, night) / jnp.minimum(day, night)
    assert bool(jnp.all(ratio < 5.0))
    assert "phase_curve_map_physicality" not in trace


def test_both_fixed_fluxes_add_no_quantile_site():
    def model():
        return _sample_surface_parameters(_config(0.0, 0.0), 1)

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(4))
    ).get_trace()
    assert not any("_flux_quantile_" in name for name in trace)
    np.testing.assert_allclose(trace["dayside_flux"]["value"], [1.2e-3])
    np.testing.assert_allclose(trace["nightside_flux"]["value"], [2.5e-4])


def test_fixed_zero_counterpart_rejects_a_free_flux():
    config = _config(day_width=2e-4, night_width=0.0)
    config["nightside_flux"] = np.array([0.0])
    with pytest.raises(ValueError, match="fixed zero nightside"):
        numpyro.handlers.trace(
            numpyro.handlers.seed(
                lambda: _sample_surface_parameters(config, 1),
                jax.random.PRNGKey(2),
            )
        ).get_trace()

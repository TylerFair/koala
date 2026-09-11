import jax
import jax.numpy as jnp
import json
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from models.common import compute_transit_model_auto
from models.jaxoplanet.builder import create_vectorized_model, create_whitelight_model
from models.jaxoplanet.config import parse_surface_config
from models.trend_marginal import build_marginalized_trend_design
from models.jaxoplanet.surface_basis import (
    EmissionLightCurveBasis,
    SpotLightCurveBasis,
)
from models.independent_nuts import get_samples_independent


jax.config.update("jax_enable_x64", True)


def _fixed(value):
    return {"value": value, "prior": "fixed"}


def _gaussian(value, sigma, **bounds):
    return {"value": value, "prior": "gaussian", "sigma": sigma, **bounds}


def _geometry_planet(free=False):
    """Orbital specifications; ``free`` opens t0 so the geometry is fitted."""
    planet = {
        "period": _fixed(2.0), "t0": _fixed(60_000.0), "b": _fixed(0.2),
        "rprs": _fixed(0.1), "a_rs": _fixed(6.0),
    }
    if free:
        planet["t0"] = {"value": 60_000.0, "prior": "gaussian", "sigma": 0.02}
        planet["a_rs"] = {"value": 6.0, "prior": "log_uniform", "low": 5.0, "high": 7.0}
    return planet


def _surface_planet(model):
    return {
        "eclipse_depth_ppm": _gaussian(800.0, 100.0, low=0.0),
        "dayside_flux_ppm": _gaussian(800.0, 100.0),
        "nightside_flux_ppm": _gaussian(200.0, 30.0),
        "hotspot_offset_deg": _gaussian(12.0, 4.0),
    }


def _surface_config(model, free_geometry=False):
    from koala.config import parse_planet_parameter_specs

    planet = {**_geometry_planet(free=free_geometry), **_surface_planet(model)}
    specs = parse_planet_parameter_specs(planet)
    return parse_surface_config(
        {"light_curve_model": model}, planet, {}, 1,
        parameter_specs=specs, param_method="a_rs",
    )


def _geometry_specs(free=False):
    from koala.config import parse_planet_parameter_specs

    return parse_planet_parameter_specs(_geometry_planet(free=free))


def _fake_surface(params, time, **kwargs):
    if "eclipse_depth" in params:
        amplitude = params["eclipse_depth"][0]
    else:
        amplitude = params["dayside_flux"][0]
    return jnp.full_like(time, amplitude)


def test_spectroscopic_eclipse_honors_fixed_radius_with_free_white_geometry(monkeypatch):
    from koala.surface import _spectroscopic_surface_config

    monkeypatch.setattr("models.jaxoplanet.surface.compute_surface_model", _fake_surface)
    white_config = _surface_config("eclipse", free_geometry=True)
    config = _spectroscopic_surface_config(white_config, _geometry_specs(free=True))
    assert white_config["fit_geometry"] is True
    assert config["fit_geometry"] is False
    model = create_vectorized_model(ld_mode="fixed", param_method="a_rs", surface_config=config)
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(2))).get_trace(
        jnp.linspace(.4, .6, 7), jnp.full((2, 7), 2e-4),
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.2]),
        mu_depths=jnp.array([0.01]), PERIOD=jnp.array([1.0]),
        mu_a_rs=jnp.array([8.0]), ld_fixed=jnp.array([[0.3, 0.2], [0.3, 0.2]]),
    )
    assert trace['rors']['type'] == 'deterministic'
    np.testing.assert_allclose(trace['rors']['value'], np.full((2, 1), .1))
    assert trace['_eclipse_depth_0']['type'] == 'sample'


def test_vectorized_eclipse_exposes_channel_planet_sites(monkeypatch):
    monkeypatch.setattr("models.jaxoplanet.surface.compute_surface_model", _fake_surface)
    model = create_vectorized_model(
        ld_mode="fixed", param_method="a_rs",
        surface_config=_surface_config("eclipse"),
    )
    time = jnp.linspace(0.4, 0.6, 7)
    error = jnp.full((2, time.size), 2e-4)
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(2))).get_trace(
        time, error,
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.2]),
        mu_depths=jnp.array([[0.01], [0.01]]),
        PERIOD=jnp.array([1.0]), mu_a_rs=jnp.array([8.0]),
        ld_fixed=jnp.array([[0.3, 0.2], [0.3, 0.2]]),
    )
    assert trace["eclipse_depth"]["value"].shape == (2, 1)
    assert trace["eclipse_depth_ppm"]["value"].shape == (2, 1)
    assert trace["obs"]["fn"].loc.shape == (2, time.size)
    np.testing.assert_array_equal(
        trace["_geometry_fixed"]["value"], np.ones(2, dtype=bool)
    )
    np.testing.assert_allclose(trace["rors"]["value"], 0.1)


def test_whitelight_phase_curve_anchors_t0_outside_visit(monkeypatch):
    monkeypatch.setattr("models.jaxoplanet.surface.compute_surface_model", _fake_surface)
    model = create_whitelight_model(
        ld_mode="fixed", param_method="a_rs",
        surface_config=_surface_config("phase_curve", free_geometry=True),
        parameter_priors=_geometry_specs(free=True),
    )
    time = jnp.linspace(60_000.7, 60_001.3, 9)
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(3))).get_trace(
        time, jnp.full(time.shape, 2e-4),
        prior_params={
            "period": jnp.array([2.0]), "duration": jnp.array([0.1]),
            "t0": jnp.array([60_000.0]),
            "ecc": jnp.array([0.0]), "omega": jnp.array([0.0]),
            "u": jnp.array([0.3, 0.2]),
        },
    )
    assert trace["t0_0"]["type"] == "sample"
    assert np.isfinite(float(trace["t0_0"]["fn"].log_prob(60_000.0)))
    assert trace["log_a_rs_0"]["type"] == "sample"
    for name in (
        "dayside_flux", "nightside_flux", "hotspot_offset",
        "dayside_flux_ppm", "nightside_flux_ppm", "hotspot_offset_deg",
    ):
        assert name in trace


def test_surface_signal_is_scaled_by_the_multiplicative_baseline(monkeypatch):
    from models.common import apply_systematics

    monkeypatch.setattr(
        "models.jaxoplanet.core.compute_transit_model",
        lambda params, time: jnp.full_like(time, 0.01),
    )
    params = {"_surface_model": "eclipse", "c": 0.8}
    signal = compute_transit_model_auto(params, jnp.arange(3.0))
    # The raw signal is the normalized system flux minus one ...
    np.testing.assert_allclose(signal, 0.01)
    # ... and the baseline scales the whole system flux multiplicatively.
    np.testing.assert_allclose(apply_systematics(signal, params["c"]), 0.8 * 1.01)


def test_marginalized_design_is_scaled_by_the_transit_factor():
    system_flux = jnp.array([[1.001, 1.000, 1.001], [1.002, 1.000, 1.002]])
    design, names = build_marginalized_trend_design(
        "linear", jnp.arange(3.0), 2, transit_factor=system_flux,
    )
    assert names == ("c", "v")
    np.testing.assert_allclose(design[..., 0], system_flux)
    np.testing.assert_allclose(design[..., 1], system_flux * jnp.arange(3.0))


def test_postfit_surface_metadata_preserves_fitted_values():
    from fit_jwst import _attach_surface_eval_metadata

    config = _surface_config("phase_curve")
    fitted = {
        "dayside_flux": jnp.array([7e-4]),
        "nightside_flux": jnp.array([1.8e-4]),
        "hotspot_offset": jnp.array([0.1]),
    }
    result = _attach_surface_eval_metadata(fitted, config)
    np.testing.assert_allclose(result["dayside_flux"], fitted["dayside_flux"])
    np.testing.assert_allclose(result["nightside_flux"], fitted["nightside_flux"])
    assert result["_surface_model"] == "phase_curve"


def test_native_vectorized_eclipse_builder_has_finite_jitted_log_density_gradient():
    """Exercise the real starry evaluator through the spectroscopic builder."""
    model = create_vectorized_model(
        detrend_type="linear", ld_mode="fixed", param_method="a_rs",
        surface_config=_surface_config("eclipse"),
    )
    time = jnp.array([0.48, 0.50, 0.52])
    error = jnp.full((1, time.size), 2e-4)
    observed = jnp.ones((1, time.size))
    fixed = {
        "log_jitter": jnp.array([jnp.log(1e-5)]),
        "c": jnp.array([1.0]),
        "v": jnp.array([0.0]),
    }

    def log_density(depth):
        conditioned = handlers.condition(
            model, data={**fixed, "_eclipse_depth_0": jnp.array([depth])}
        )
        trace = handlers.trace(handlers.seed(conditioned, jax.random.PRNGKey(9))).get_trace(
            time, error, y=observed,
            mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.0]),
            mu_depths=jnp.array([[0.01]]),
            PERIOD=jnp.array([1.0]), mu_a_rs=jnp.array([8.0]),
            ld_fixed=jnp.array([[0.3, 0.2]]),
        )
        total = jnp.asarray(0.0)
        for site in trace.values():
            if site["type"] == "sample":
                total = total + jnp.sum(site["fn"].log_prob(site["value"]))
        return total

    value, gradient = jax.jit(jax.value_and_grad(log_density))(8e-4)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))


def test_checkpoint_manifest_serializes_surface_array_signature(tmp_path):
    from fit_jwst import _write_or_validate_checkpoint_manifest

    signature = {
        "stage": "lowres",
        "surface_config": {
            "model": "eclipse",
            "fit_geometry": False,
            "eclipse_depth": np.array([9e-4]),
            "eclipse_depth_spec": _surface_config("eclipse")["eclipse_depth_spec"],
            "spots": (),
        },
    }
    path = _write_or_validate_checkpoint_manifest(
        tmp_path, "lr", "lr_deadbeef", "deadbeef", signature,
    )
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["checkpoint_signature"]["surface_config"]["eclipse_depth"] == [9e-4]
    spec = payload["checkpoint_signature"]["surface_config"]["eclipse_depth_spec"][0]
    assert spec["prior"] == "gaussian"
    np.testing.assert_allclose(spec["value"], 8e-4)
    # The normalized in-memory payload also compares equal on resume.
    assert _write_or_validate_checkpoint_manifest(
        tmp_path, "lr", "lr_deadbeef", "deadbeef", signature,
    ) == path


def test_spectroscopic_surface_init_uses_physical_prior_centers():
    from fit_jwst import _seed_surface_spectroscopic_init
    from models.jaxoplanet.builder import _conditional_phase_flux_from_quantile

    config = _surface_config("phase_curve")
    init = _seed_surface_spectroscopic_init({}, config, 4)
    for name in ("dayside_flux", "hotspot_offset"):
        np.testing.assert_allclose(
            init[f"_{name}_0"], np.full(4, config[name][0])
        )
    reconstructed_night = _conditional_phase_flux_from_quantile(
        init["_nightside_flux_quantile_0"],
        init["_dayside_flux_0"],
        config["nightside_flux"][0],
        config["nightside_flux_prior_width"][0],
    )
    np.testing.assert_allclose(
        reconstructed_night, np.full(4, config["nightside_flux"][0])
    )


def test_channel_dependent_spot_basis_slices_and_pads_with_chunks():
    from fit_jwst import _slice_by_channel, _surface_basis_on_time_mask
    from models.independent_nuts import _partition_model_kwargs

    basis = SpotLightCurveBasis(
        baseline=jnp.arange(15.0).reshape(3, 5),
        differences=jnp.arange(15.0).reshape(3, 1, 5) * 0.01,
    )
    masked = _surface_basis_on_time_mask(
        basis, np.array([True, False, True, False, True])
    )
    assert masked.baseline.shape == (3, 3)
    assert masked.differences.shape == (3, 1, 3)
    sliced = _slice_by_channel(basis, slice(1, 3), 3)
    assert sliced.baseline.shape == (2, 5)
    np.testing.assert_allclose(sliced.baseline[0], basis.baseline[1])
    varying, shared = _partition_model_kwargs(
        {"surface_basis_data": sliced},
        ("surface_basis_data",),
        num_channels=2,
        lane_width=4,
    )
    assert not shared
    padded = varying["surface_basis_data"]
    assert padded.baseline.shape == (4, 1, 5)
    assert padded.differences.shape == (4, 1, 1, 5)
    np.testing.assert_allclose(padded.baseline[-1], padded.baseline[-2])


def test_vector_builder_accepts_dynamic_channel_emission_basis():
    config = _surface_config("eclipse")
    model = create_vectorized_model(
        ld_mode="fixed", param_method="a_rs", surface_config=config,
    )
    time = jnp.linspace(0.4, 0.6, 5)
    basis = EmissionLightCurveBasis(
        baseline=jnp.zeros((2, 5)),
        uniform=jnp.stack((jnp.ones(5), 2.0 * jnp.ones(5))),
        dipole_cos=None,
        dipole_sin=None,
    )
    conditioned = handlers.condition(model, data={
        "_eclipse_depth_0": jnp.full(2, 8e-4),
        "c": jnp.ones(2),
        "v": jnp.zeros(2),
        "log_jitter": jnp.full(2, jnp.log(1e-5)),
    })
    trace = handlers.trace(handlers.seed(conditioned, jax.random.PRNGKey(31))).get_trace(
        time, jnp.full((2, 5), 2e-4),
        mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.2]),
        mu_depths=jnp.array([[0.01], [0.01]]),
        PERIOD=jnp.array([1.0]), mu_a_rs=jnp.array([8.0]),
        ld_fixed=jnp.array([[0.3, 0.2], [0.3, 0.2]]),
        surface_basis_data=basis,
    )
    assert trace["obs"]["fn"].loc.shape == (2, 5)
    assert np.all(np.asarray(trace["obs"]["fn"].loc[1]) >=
                  np.asarray(trace["obs"]["fn"].loc[0]))


def test_runtime_surface_basis_rejects_free_geometry():
    import pytest

    # Free t0/a_rs specifications make the surface geometry fitted.
    config = _surface_config("phase_curve", free_geometry=True)
    assert config["fit_geometry"] is True
    model = create_vectorized_model(
        ld_mode="fixed", param_method="a_rs", surface_config=config,
    )
    basis = EmissionLightCurveBasis(
        baseline=jnp.zeros((1, 3)), uniform=jnp.ones((1, 3)),
        dipole_cos=jnp.ones((1, 3)), dipole_sin=jnp.ones((1, 3)),
    )
    with pytest.raises(ValueError, match="fixed geometry"):
        handlers.trace(handlers.seed(model, jax.random.PRNGKey(32))).get_trace(
            jnp.arange(3.0), jnp.full((1, 3), 2e-4),
            mu_t0=jnp.array([0.0]), mu_b=jnp.array([0.2]),
            mu_depths=jnp.array([[0.01]]), PERIOD=jnp.array([1.0]),
            mu_a_rs=jnp.array([8.0]), ld_fixed=jnp.array([[0.3, 0.2]]),
            surface_basis_data=basis,
        )


def test_independent_sampler_preserves_surface_basis_pytree():
    def model(t, yerr, y=None, surface_basis_data=None):
        num_lcs = yerr.shape[0]
        amplitude = numpyro.sample(
            "amplitude", dist.Normal(0.0, 1.0).expand([num_lcs])
        )
        prediction = (
            surface_basis_data.baseline
            + amplitude[:, None] * surface_basis_data.uniform
        )
        numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)

    basis = EmissionLightCurveBasis(
        baseline=jnp.zeros((2, 3)), uniform=jnp.ones((2, 3)),
        dipole_cos=None, dipole_sin=None,
    )
    samples = get_samples_independent(
        model, jax.random.PRNGKey(41), jnp.arange(3.0),
        jnp.ones((2, 3)), jnp.zeros((2, 3)),
        {"amplitude": jnp.zeros(2)},
        mcmc_kwargs={"num_warmup": 2, "num_samples": 2},
        lane_width=2,
        channel_varying_kwargs=("surface_basis_data",),
        surface_basis_data=basis,
    )
    assert samples["amplitude"].shape == (2, 2)

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from models.jaxoplanet.config import parse_surface_config


def _fixed(value):
    return {"value": value, "prior": "fixed"}


def _gaussian(value, sigma, **bounds):
    return {"value": value, "prior": "gaussian", "sigma": sigma, **bounds}


def _geometry(free=False):
    planet = {
        "period": _fixed(2.0), "t0": _fixed(60000.0), "b": _fixed(0.25),
        "rprs": _fixed(0.1), "a_rs": _fixed(6.0),
    }
    if free:
        planet["t0"] = {"value": 60000.0, "prior": "uniform", "low": 59999.9, "high": 60000.1}
    return planet


def test_legacy_transit_needs_no_surface_inputs():
    assert parse_surface_config({}, {}, {}, 2) == {"model": "transit", "spots": ()}


def test_phase_config_converts_public_units():
    result = parse_surface_config({"light_curve_model": "phase_curve"},
                                  {"dayside_flux_ppm": _gaussian(1000., 100.),
                                   "nightside_flux_ppm": _fixed(300.),
                                   "hotspot_offset_deg": _fixed(30.)}, {}, 1)
    np.testing.assert_allclose(result["dayside_flux"], [.001])
    np.testing.assert_allclose(result["dayside_flux_prior_width"], [.0001])
    np.testing.assert_allclose(result["hotspot_offset"], [np.pi / 6])
    np.testing.assert_allclose(result["nightside_flux_prior_width"], [0.])
    assert result["dayside_flux_spec"][0].prior == "gaussian"
    assert result["dayside_flux_spec"][0].sigma == pytest.approx(1e-4)
    assert result["nightside_flux_spec"][0].fixed
    assert result["hotspot_offset_spec"][0].value == pytest.approx(np.pi / 6)
    # Without geometry specifications the surface geometry defaults to fitted.
    assert result["fit_geometry"] is True


def test_eclipse_config_keeps_the_depth_specification():
    result = parse_surface_config(
        {"light_curve_model": "eclipse"},
        {"eclipse_depth_ppm": {"value": 160., "prior": "uniform", "low": 0., "high": 1000.}},
        {}, 1,
    )
    spec = result["eclipse_depth_spec"][0]
    assert spec.prior == "uniform" and spec.high == pytest.approx(1e-3)
    np.testing.assert_allclose(result["eclipse_depth"], [1.6e-4])
    np.testing.assert_allclose(result["eclipse_depth_prior_width"], [0.])


def test_geometry_fixed_is_derived_from_the_specifications():
    from koala.config import parse_planet_parameter_specs

    planet = {**_geometry(), "eclipse_depth_ppm": _fixed(500.)}
    specs = parse_planet_parameter_specs(planet)
    fixed = parse_surface_config({"light_curve_model": "eclipse"}, planet, {}, 1,
                                 parameter_specs=specs, param_method="a_rs")
    assert fixed["fit_geometry"] is False
    planet = {**_geometry(free=True), "eclipse_depth_ppm": _fixed(500.)}
    specs = parse_planet_parameter_specs(planet)
    free = parse_surface_config({"light_curve_model": "eclipse"}, planet, {}, 1,
                                parameter_specs=specs, param_method="a_rs")
    assert free["fit_geometry"] is True
    # A pure transit carries no fit_geometry entry at all.
    assert "fit_geometry" not in parse_surface_config({}, planet, {}, 1,
                                                      parameter_specs=specs, param_method="a_rs")


def test_fit_geometry_flag_is_rejected():
    with pytest.raises(ValueError, match="flags.fit_geometry has been removed"):
        parse_surface_config({"light_curve_model": "eclipse", "fit_geometry": False},
                             {"eclipse_depth_ppm": _fixed(500.)}, {}, 1)


@pytest.mark.parametrize("planet, message", [
    ({"eclipse_depth_ppm": 500.}, "written as a mapping"),
    ({"eclipse_depth_ppm": _fixed(500.), "eclipse_depth_prior_width_ppm": 100.}, "has been removed"),
    ({}, r"planet\.eclipse_depth_ppm is required"),
])
def test_eclipse_config_rejects_old_forms(planet, message):
    with pytest.raises(ValueError, match=message):
        parse_surface_config({"light_curve_model": "eclipse"}, planet, {}, 1)


@pytest.mark.parametrize("planet", [{"dayside_flux_ppm": _fixed(1000.), "nightside_flux_ppm": _fixed(0.)},
                                    {"dayside_flux_ppm": _fixed(1000.), "nightside_flux_ppm": _fixed(100.)}])
def test_negative_intensity_phase_maps_rejected(planet):
    planet = {**planet, "hotspot_offset_deg": _fixed(0.)}
    with pytest.raises(ValueError, match="five times"):
        parse_surface_config({"light_curve_model": "phase_curve"}, planet, {}, 1)


def test_multiple_surface_planets_rejected_early():
    with pytest.raises(ValueError, match="one planet"):
        parse_surface_config({"light_curve_model": "eclipse"}, {"eclipse_depth_ppm": _fixed(1000.)}, {}, 2)


def test_spot_config_validates_and_converts_geometry():
    stellar = {"rotation_period": 5., "spots": [{"latitude_deg": 30., "longitude_deg": -45.,
               "radius_deg": 15., "contrast": .4, "contrast_prior_width": .1}]}
    result = parse_surface_config({}, {}, stellar, 1)
    np.testing.assert_allclose(result["spots"][0]["latitude"], np.pi / 6)
    np.testing.assert_allclose(result["spots"][0]["longitude"], -np.pi / 4)
    stellar["spots"][0]["contrast"] = 1.1
    with pytest.raises(ValueError, match="contrast"):
        parse_surface_config({}, {}, stellar, 1)


def test_science_gate_rejects_stuck_emission_despite_mixed_radius():
    from fit_jwst import _spectro_failed_lanes

    rng = np.random.default_rng(12)
    samples = {"rors": .1 + rng.normal(0., .001, (500, 2, 1)),
               "_eclipse_depth_0": np.column_stack([np.ones(500), rng.normal(size=500)])}
    failed, metrics = _spectro_failed_lanes(samples, None, 20., 0)
    assert 0 in failed
    assert 1 not in failed
    assert "eclipse_depth_0" in metrics["science_ess_per_channel"]


def test_white_science_metrics_include_free_contrast_only():
    from fit_jwst import _geometry_chain_quality

    rng = np.random.default_rng(12)
    ess, _ = _geometry_chain_quality({"_stellar_spot_contrast_0": rng.normal(size=(1, 500)),
                                     "dayside_flux": np.ones((1, 500, 1))})
    assert "stellar_spot_contrast_0" in ess
    assert "dayside_flux" not in ess

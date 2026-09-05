import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from models.jaxoplanet.config import parse_surface_config


def test_legacy_transit_needs_no_surface_inputs():
    assert parse_surface_config({}, {}, {}, 2) == {"model": "transit", "spots": ()}


def test_phase_config_converts_public_units():
    result = parse_surface_config({"light_curve_model": "phase_curve"},
                                  {"dayside_flux_ppm": 1000., "nightside_flux_ppm": 300.,
                                   "dayside_flux_prior_width_ppm": 100., "hotspot_offset_deg": 30.}, {}, 1)
    np.testing.assert_allclose(result["dayside_flux"], [.001])
    np.testing.assert_allclose(result["dayside_flux_prior_width"], [.0001])
    np.testing.assert_allclose(result["hotspot_offset"], [np.pi / 6])
    np.testing.assert_allclose(result["nightside_flux_prior_width"], [0.])


@pytest.mark.parametrize("planet", [{"dayside_flux_ppm": 1000., "nightside_flux_ppm": 0.},
                                    {"dayside_flux_ppm": 1000., "nightside_flux_ppm": 100.}])
def test_negative_intensity_phase_maps_rejected(planet):
    with pytest.raises(ValueError, match="five times"):
        parse_surface_config({"light_curve_model": "phase_curve"}, planet, {}, 1)


def test_multiple_surface_planets_rejected_early():
    with pytest.raises(ValueError, match="one planet"):
        parse_surface_config({"light_curve_model": "eclipse"}, {"eclipse_depth_ppm": 1000.}, {}, 2)


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

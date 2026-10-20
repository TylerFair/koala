import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

from koala.surface_outputs import save_surface_results


def test_emission_output_units_intervals_and_planet_labels(tmp_path):
    flux = np.arange(40, dtype=float).reshape(10, 2, 2) * 1e-6
    samples = {"eclipse_depth": flux, "phase_c1": 0.25 * flux, "phase_d1": 0.0 * flux}
    frame = save_surface_results([1., 2.], [.1, .1], samples, tmp_path / "spectrum.csv")
    assert (tmp_path / "spectrum_emission.png").stat().st_size > 0
    saved = pd.read_csv(tmp_path / "spectrum_emission.csv")
    assert len(frame) == len(saved) == 4
    row = saved[(saved.planet_index == 1) & (saved.wavelength == 2.)].iloc[0]
    low, med, high = np.percentile(flux[:, 1, 1] * 1e6, [15.865, 50, 84.135])
    assert row.eclipse_depth_ppm == pytest.approx(med)
    assert row.eclipse_depth_ppm_err_low == pytest.approx(med - low)
    assert row.eclipse_depth_ppm_err_high == pytest.approx(high - med)
    assert row.nightside_flux_ppm == pytest.approx(0.5 * med)


def test_transit_has_no_emission_output(tmp_path):
    assert save_surface_results([1.], [.1], {"rors": np.ones((10, 1))}, tmp_path / "transit.csv") is None
    assert not list(tmp_path.iterdir())


def test_emission_rejects_mismatched_channels(tmp_path):
    with pytest.raises(ValueError, match="axes"):
        save_surface_results([1., 2.], [.1, .1], {"eclipse_depth": np.ones((10, 3))}, tmp_path / "s.csv")


def test_spot_fit_writes_contrast_spectrum_without_emission(tmp_path):
    values = np.full((10, 2, 1), .3)
    save_surface_results([1., 2.], [.1, .1], {"stellar_spot_contrast": values}, tmp_path / "spot.csv")
    frame = pd.read_csv(tmp_path / "spot_stellar_spots.csv")
    np.testing.assert_allclose(frame.contrast, .3)
    np.testing.assert_allclose(frame.contrast_err_low, 0.)
    assert (tmp_path / "spot_stellar_spots.png").stat().st_size > 0
    assert not (tmp_path / "spot_emission.csv").exists()


def test_postfit_surface_plot_keeps_emission_and_applies_baseline_once():
    from unittest.mock import patch
    from koala.plotting import _map_full_models, _single_curve_transit_signal

    params = {"period": np.array([2.]), "t0": np.array([0.]),
              "a_rs": np.array([8.]), "b": np.array([.2]),
              "rors": np.array([[.1], [.12]]), "u": np.array([[.2, .1], [.3, .1]]),
              "_surface_model": "eclipse", "eclipse_depth": np.array([[.001], [.002]]),
              "c": np.array([.999, .998])}
    with patch("koala.plotting.compute_transit_model_auto", return_value=np.zeros(3)) as evaluate:
        _single_curve_transit_signal(np.arange(3.), params,
                                     {"period": [2.], "transit_engine": "jaxoplanet", "param_method": "a_rs"}, 1)
    actual = evaluate.call_args.args[0]
    assert actual["_surface_model"] == "eclipse"
    np.testing.assert_allclose(actual["eclipse_depth"], [.002])
    # The signal is stellar-normalized. Baseline scaling belongs to the full
    # model, F = (1 + signal) * trend, including when plotting emission.
    assert "c" not in actual
    signal = np.array([.002, 0., .002])
    with patch("koala.plotting.compute_transit_model_auto", return_value=signal):
        full = _map_full_models(np.arange(3.), params,
            {"period": [2.], "transit_engine": "jaxoplanet", "param_method": "a_rs"},
            2, "linear")
    np.testing.assert_allclose(full[1], .998*(1+signal))


def test_normalized_eclipse_retains_planet_to_star_ratio():
    from unittest.mock import patch
    from models.trends import compute_lc_linear

    planet_flux = .02
    baseline = 1. / (1. + planet_flux)
    physical_signal = np.array([planet_flux, 0., planet_flux])
    params = {"_surface_model": "eclipse", "c": baseline, "v": 0.}
    with patch("models.jaxoplanet.core.compute_transit_model", return_value=physical_signal):
        actual = compute_lc_linear(params, np.arange(3.))
    np.testing.assert_allclose(actual, (1. + physical_signal) / (1. + planet_flux))


def test_postfit_basis_selects_channel_and_rejects_a_stale_time_grid():
    from koala.plotting import _map_full_models, _single_curve_transit_signal
    from models.jaxoplanet.surface_basis import EmissionLightCurveBasis

    time = np.arange(3.)
    params = {
        "t0": [0.], "b": [.2], "a_rs": [8.],
        "rors": np.array([[.1], [.1]]), "u": np.array([[.3, .2], [.3, .2]]),
        "_surface_model": "eclipse", "eclipse_depth": np.array([[.001], [.002]]),
        "c": np.array([1., .99]), "_surface_basis_time": time,
        "_surface_basis": EmissionLightCurveBasis(
            baseline=np.zeros((2, 3)), uniform=np.array([[1., 0., 1.], [1., .5, 1.]]),
            phase_cos=None, phase_sin=None,
        ),
    }
    geometry = {"period": [2.], "transit_engine": "jaxoplanet", "param_method": "a_rs"}
    actual = _single_curve_transit_signal(time, params, geometry, 1)
    signal = .002 * np.array([1., .5, 1.])
    np.testing.assert_allclose(actual, signal)
    full = _map_full_models(time, params, geometry, 2, "linear")
    np.testing.assert_allclose(full[1], .99*(1+signal))
    with pytest.raises(ValueError, match="time grid"):
        _single_curve_transit_signal(time + .1, params, geometry, 1)

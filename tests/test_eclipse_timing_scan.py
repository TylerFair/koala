import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from koala.config import ParameterSpec, check_ephemeris_covers_data
from koala.surface import scan_eclipse_time
from models.jaxoplanet.surface_basis import prepare_emission_light_curve_basis

PERIOD, A_RS, B, RORS = 2.6162644, 17.04, 0.208, 0.0318
T0 = 60000.0
ECLIPSE = T0 + 0.5 * PERIOD
GEOMETRY = dict(period=PERIOD, a_rs=A_RS, b=B, rors=RORS, ecc=0.0, omega=0.0)
DEPTH = ParameterSpec("eclipse_depth", "uniform", 1e-4, low=0.0, high=1e-3)


def _eclipse_data(noise=150e-6, depth=80e-6, n=1500, seed=3):
    t = np.linspace(ECLIPSE - 0.14, ECLIPSE + 0.09, n)
    fixed = dict(_surface_model="eclipse", t0=T0, u=np.zeros(2), **GEOMETRY)
    visible = np.asarray(prepare_emission_light_curve_basis(fixed, t, model="eclipse").uniform)
    ramp = 1.0 - 2e-3 * np.exp(-(t - t[0]) / 0.015)
    rng = np.random.default_rng(seed)
    return t, ramp * (1.0 + depth * (visible - 1.0)) + rng.normal(0.0, noise, n), noise


@pytest.mark.parametrize("start_offset", [0.0, 0.06, -0.08])
def test_scan_recovers_eclipse_time_from_distant_start(start_offset):
    t, flux, err = _eclipse_data()
    t0 = ParameterSpec("t0", "uniform", T0 + start_offset, low=T0 - 0.085, high=T0 + 0.085)
    found_t0, depth, delta_chi2 = scan_eclipse_time(
        {"model": "eclipse"}, t, flux, err, t0, DEPTH, "explinear", **GEOMETRY)
    assert abs(found_t0 - T0) * 1440 < 6.0
    assert 50e-6 < depth < 110e-6
    assert delta_chi2 > 25.0


def test_scan_skips_fixed_time():
    t, flux, err = _eclipse_data()
    fixed = ParameterSpec("t0", "fixed", T0)
    assert scan_eclipse_time({"model": "eclipse"}, t, flux, err, fixed, DEPTH,
                             "linear", **GEOMETRY) is None


def _specs(t0_value, low=None, high=None, prior="fixed"):
    return {
        "t0": [ParameterSpec("t0", prior, t0_value, low=low, high=high)],
        "period": [ParameterSpec("period", "fixed", PERIOD)],
    }


def test_ephemeris_check_accepts_covered_eclipse():
    t = np.linspace(ECLIPSE - 0.14, ECLIPSE + 0.09, 100)
    check_ephemeris_covers_data(t, _specs(T0), "eclipse")
    check_ephemeris_covers_data(t, _specs(T0 + 3 * PERIOD), "eclipse")


def test_ephemeris_check_flags_bjd_minus_2400000():
    t = np.linspace(ECLIPSE - 0.14, ECLIPSE + 0.09, 100)
    with pytest.raises(ValueError, match="BJD - 2400000"):
        check_ephemeris_covers_data(
            t, _specs(T0 + 0.5, T0 + 0.5 - 0.08, T0 + 0.5 + 0.08, "uniform"), "eclipse")


def test_ephemeris_check_flags_full_bjd():
    t = np.linspace(ECLIPSE - 0.14, ECLIPSE + 0.09, 100)
    with pytest.raises(ValueError, match="full BJD"):
        check_ephemeris_covers_data(t, _specs(T0 + 2400000.5), "eclipse")


def test_ephemeris_check_uses_transit_for_transit_model():
    t = np.linspace(ECLIPSE - 0.14, ECLIPSE + 0.09, 100)
    with pytest.raises(ValueError, match="no transit time"):
        check_ephemeris_covers_data(t, _specs(T0), "transit")

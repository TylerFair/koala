"""Unified ``planet`` parameter specification and the free white-light period."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import yaml
from numpyro import handlers

import createdatacube
from koala.config import (
    ParameterSpec,
    describe_planet_parameter_specs,
    parse_parameter_spec,
    parse_planet_parameter_specs,
    planet_parameter_centers,
)
from koala.geometry import _geometry_chain_quality, _selected_geometry_primitives
from koala.white_light import _whitelight_geometry_sites
from models.harmonica.core import harmonica_a_rs_from_duration
from models.jaxoplanet.builder import (
    _validate_parameter_priors,
    create_whitelight_model,
    derive_geometry,
)
from plotting import _corner_columns


# --------------------------------------------------------------------------
# Specification grammar
# --------------------------------------------------------------------------

def test_bare_numbers_keep_the_historical_modes():
    specs = parse_planet_parameter_specs({
        'period': 4.05, 't0': 59787.0, 'duration': 0.117, 'b': 0.45,
        'rprs': 0.1457, 'ecc': 0.0, 'omega': 90.0,
    })
    assert {name: spec[0].mode for name, spec in specs.items()} == {
        'period': 'fixed', 't0': 'free', 'duration': 'free', 'b': 'free',
        'rprs': 'free', 'ecc': 'fixed', 'omega': 'fixed',
    }
    assert all(spec[0].prior is None and not spec[0].explicit for spec in specs.values())
    assert specs['t0'][0].center == 59787.0
    np.testing.assert_allclose(planet_parameter_centers(specs, 'period'), [4.05])


def test_ecc_and_omega_default_to_fixed_zero():
    specs = parse_planet_parameter_specs({'period': 1.0, 't0': 0.0})
    assert specs['ecc'] == (ParameterSpec('ecc', 'fixed', None, 0.0),)
    assert specs['omega'] == (ParameterSpec('omega', 'fixed', None, 0.0),)


@pytest.mark.parametrize("raw, expected", [
    (['fixed', 1.05], ParameterSpec('period', 'fixed', None, 1.05, explicit=True)),
    (['free', 'uniform', 0, 10],
     ParameterSpec('period', 'free', 'uniform', None, None, None, 0.0, 10.0, explicit=True)),
    (['free', 'gaussian', 5, 0.2],
     ParameterSpec('period', 'free', 'gaussian', None, 5.0, 0.2, None, None, explicit=True)),
    (['free', 'truncated_gaussian', 5, 0.2, 4, 6],
     ParameterSpec('period', 'free', 'truncated_gaussian', None, 5.0, 0.2, 4.0, 6.0, explicit=True)),
    ({'mode': 'fixed', 'value': 1.05}, ParameterSpec('period', 'fixed', None, 1.05, explicit=True)),
    ({'mode': 'free', 'prior': 'gaussian', 'mu': 5, 'sigma': 0.2},
     ParameterSpec('period', 'free', 'gaussian', None, 5.0, 0.2, None, None, explicit=True)),
    ({'mode': 'free', 'prior': 'uniform', 'lo': 0, 'hi': 10},
     ParameterSpec('period', 'free', 'uniform', None, None, None, 0.0, 10.0, explicit=True)),
    ({'mode': 'free', 'prior': 'truncated_gaussian', 'mu': 5, 'sigma': 0.2, 'low': 4, 'high': 6},
     ParameterSpec('period', 'free', 'truncated_gaussian', None, 5.0, 0.2, 4.0, 6.0, explicit=True)),
])
def test_list_and_mapping_forms(raw, expected):
    assert parse_parameter_spec('period', raw) == expected


def test_yaml_list_form_round_trips_through_the_parser():
    planet = yaml.safe_load(
        "period: [free, gaussian, 4.05528043, 0.001]\n"
        "t0: [free, uniform, 59786.9, 59787.2]\n"
        "b: {mode: free, prior: gaussian, mu: 0.45, sigma: 0.05}\n"
        "rprs: 0.1457\n"
        "duration: [fixed, 0.117]\n"
    )
    specs = parse_planet_parameter_specs(planet)
    assert specs['period'][0].explicit_prior and specs['period'][0].mu == 4.05528043
    assert specs['t0'][0].center == pytest.approx(0.5 * (59786.9 + 59787.2))
    assert specs['duration'][0].fixed and specs['duration'][0].value == 0.117
    assert any('planet.period: free, gaussian' in line
               for line in describe_planet_parameter_specs(specs))


def test_per_planet_entries_and_broadcasting():
    specs = parse_planet_parameter_specs({
        'period': [1.0, ['free', 'gaussian', 2.0, 0.1]],
        't0': [['fixed', 0.5], {'mode': 'free', 'prior': 'uniform', 'low': 0, 'high': 1}],
        'b': 0.3,
    })
    assert len(specs['period']) == 2 and specs['period'][1].free
    assert specs['t0'][0].fixed and specs['t0'][1].prior == 'uniform'
    assert specs['b'] == (specs['b'][0],) * 2
    np.testing.assert_allclose(planet_parameter_centers(specs, 'period'), [1.0, 2.0])


@pytest.mark.parametrize("planet, message", [
    ({'period': ['wobble', 1.0]}, r"planet\.period: unknown mode 'wobble'"),
    ({'period': ['free', 'lognormal', 1, 2]}, r"planet\.period: unknown prior 'lognormal'"),
    ({'period': ['fixed', 1.0, 2.0]}, r"planet\.period: the fixed form is \[fixed, value\]"),
    ({'period': ['free', 'uniform', 1.0]}, r"planet\.period: expected \[free, uniform, low, high\]"),
    ({'period': ['free', 'gaussian', 1.0, 0.1, 5]}, r"planet\.period: expected \[free, gaussian, mu, sigma\]"),
    ({'period': ['free', 'uniform', 10, 0]}, r"planet\.period: prior bounds require low < high"),
    ({'period': ['free', 'gaussian', 1.0, -0.1]}, r"planet\.period: sigma must be > 0"),
    ({'period': ['free', 'truncated_gaussian', 1.0, 0.1, 2.0, 1.0]}, r"planet\.period: prior bounds require low < high"),
    ({'period': {'prior': 'uniform', 'low': 0, 'high': 1}}, r"planet\.period: a mapping specification needs 'mode'"),
    ({'period': {'mode': 'free', 'prior': 'gaussian', 'mu': 1}}, r"planet\.period: prior 'gaussian' takes exactly \['mu', 'sigma'\]"),
    ({'period': {'mode': 'fixed'}}, r"planet\.period: the fixed mapping form"),
    ({'period': 'four'}, r"planet\.period: expected a number"),
    ({'period': ['free', 'uniform', 'a', 1]}, r"planet\.period: low must be a number"),
    ({'period': 1.0, 'ecc': ['free', 'uniform', 0, 0.5]}, r"planet\.ecc may only be fixed"),
    ({'period': 1.0, 'omega': {'mode': 'free', 'prior': 'gaussian', 'mu': 0, 'sigma': 1}}, r"planet\.omega may only be fixed"),
    ({'period': [1.0, 2.0], 't0': [0.0, 1.0, 2.0]}, r"planet\.t0 must be scalar or length 2"),
])
def test_invalid_specifications_name_the_key(planet, message):
    with pytest.raises(ValueError, match=message):
        parse_planet_parameter_specs(planet)


def test_builder_rejects_specification_for_the_unused_geometry_coordinate():
    specs = parse_planet_parameter_specs({'period': 1.0, 'a_rs': ['free', 'uniform', 5, 15]})
    with pytest.raises(ValueError, match=r"planet\.a_rs .* parameterized by duration"):
        _validate_parameter_priors(specs, 1, 'duration')
    specs = parse_planet_parameter_specs({'period': 1.0, 'duration': ['fixed', 0.1]})
    with pytest.raises(ValueError, match=r"planet\.duration .* parameterized by a_rs"):
        _validate_parameter_priors(specs, 1, 'a_rs')
    assert _validate_parameter_priors(None, 1, 'duration') == {}


# --------------------------------------------------------------------------
# White-light model
# --------------------------------------------------------------------------

_T = jnp.linspace(0.88, 1.18, 120)
_YERR = jnp.full(_T.shape, 7.5e-4)
_PRIOR = {"period": jnp.array([3.0]), "u": jnp.array([0.2, 0.1])}


def _trace(model, seed=0):
    seeded = handlers.seed(model, jax.random.PRNGKey(seed))
    return handlers.trace(seeded).get_trace(
        _T, _YERR, y=jnp.ones(_T.shape), prior_params=_PRIOR
    )


def _distribution_signature(trace):
    """(site -> (type, distribution class, constant parameters)) for comparison."""
    signature = {}
    for name, site in trace.items():
        if site["type"] != "sample":
            signature[name] = (site["type"],)
            continue
        fn = site["fn"]
        params = {
            key: np.asarray(getattr(fn, key)).tolist()
            for key in ("loc", "scale", "low", "high")
            if hasattr(fn, key)
        }
        signature[name] = ("sample", type(fn).__name__, params)
    return signature


def _model(**kwargs):
    return create_whitelight_model(
        detrend_type="linear", ld_mode="fixed", ld_profile="quadratic", **kwargs
    )


def test_bare_numbers_reproduce_todays_white_light_priors_exactly():
    bare = parse_planet_parameter_specs({
        'period': 3.0, 't0': 1.0, 'duration': 0.12, 'b': 0.25, 'rprs': 0.095,
    })
    reference = _distribution_signature(_trace(_model()))
    configured = _distribution_signature(_trace(_model(parameter_priors=bare)))
    assert configured == reference
    assert reference["t0_0"] == ("sample", "Uniform", {"low": float(_T.min()), "high": float(_T.max())})
    assert reference["_b_0"] == ("sample", "Uniform", {"low": -2.0, "high": 2.0})
    assert reference["logD_0"][1] == "Uniform"
    np.testing.assert_allclose(reference["logD_0"][2]["low"], np.log(0.0007))
    assert reference["rors_0"][2] == {"low": float(np.sqrt(1e-6)), "high": float(np.sqrt(0.5))}
    assert "period_0" not in reference


def test_free_period_is_sampled_and_drives_the_derived_geometry():
    specs = parse_planet_parameter_specs({
        'period': ['free', 'gaussian', 3.0, 0.001], 't0': 1.0,
        'duration': 0.12, 'b': 0.25, 'rprs': 0.095,
    })
    trace = _trace(_model(parameter_priors=specs), seed=3)
    site = trace["period_0"]
    assert site["type"] == "sample"
    assert isinstance(site["fn"], dist.Normal)
    assert float(site["fn"].loc) == 3.0 and float(site["fn"].scale) == 0.001

    samples = {
        name: jnp.stack([trace[name]["value"]] * 4)
        for name in ("period_0", "t0_0", "rors_0", "_b_0", "logD_0", "b_0", "duration_0", "a_rs_0")
    }
    derived = derive_geometry(samples, jnp.array([2.5]))
    expected = harmonica_a_rs_from_duration(
        samples["period_0"], samples["duration_0"], samples["b_0"], samples["rors_0"]
    )
    np.testing.assert_allclose(np.asarray(derived["a_rs_0"]), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(derived["a_rs_0"]), np.asarray(samples["a_rs_0"]))
    # The fixed period passed to derive_geometry is ignored once period_0 exists.
    wrong = harmonica_a_rs_from_duration(
        jnp.array(2.5), samples["duration_0"], samples["b_0"], samples["rors_0"]
    )
    assert not np.allclose(np.asarray(derived["a_rs_0"]), np.asarray(wrong))


def test_explicit_priors_replace_the_hard_coded_uniforms():
    specs = parse_planet_parameter_specs({
        'period': 3.0,
        't0': ['free', 'uniform', 0.9, 1.1],
        'b': {'mode': 'free', 'prior': 'gaussian', 'mu': 0.25, 'sigma': 0.05},
        'duration': ['free', 'truncated_gaussian', 0.12, 0.01, 0.05, 0.3],
        'rprs': ['fixed', 0.095],
    })
    signature = _distribution_signature(_trace(_model(parameter_priors=specs)))
    assert signature["t0_0"] == ("sample", "Uniform", {"low": 0.9, "high": 1.1})
    assert signature["b_0"] == ("sample", "Normal", {"loc": 0.25, "scale": 0.05})
    assert "_b_0" not in signature and "logD_0" not in signature
    assert signature["duration_0"][1] == "TwoSidedTruncatedDistribution"
    assert signature["duration_0"][2] == {"low": 0.05, "high": 0.3}
    assert signature["rors_0"] == ("deterministic",)
    assert "period_0" not in signature


def test_period_free_and_fixed_geometry_are_incompatible():
    specs = parse_planet_parameter_specs({'period': ['free', 'uniform', 2, 4], 'a_rs': 8.0})
    with pytest.raises(ValueError, match=r"planet\.period is configured as free"):
        create_whitelight_model(
            ld_mode="fixed", ld_profile="quadratic", param_method="a_rs",
            surface_config={"model": "transit", "spots": (), "fit_geometry": False},
            parameter_priors=specs,
        )


# --------------------------------------------------------------------------
# White-light stage plumbing
# --------------------------------------------------------------------------

def test_geometry_sites_follow_the_specifications():
    common = dict(
        period=jnp.array([3.0]), t0=jnp.array([1.0]), b=jnp.array([0.25]),
        rors=jnp.array([0.095]), duration=jnp.array([0.12]), a_rs=jnp.array([9.0]),
    )
    bare = parse_planet_parameter_specs({'period': 3.0, 't0': 1.0, 'duration': 0.12, 'b': 0.25, 'rprs': 0.095})
    sites = _whitelight_geometry_sites(bare, 1, 'duration', **common)
    assert set(sites) == {"logD_0", "_b_0", "t0_0", "rors_0"}
    np.testing.assert_allclose(sites["logD_0"], np.log(0.12))
    assert set(_whitelight_geometry_sites(None, 1, 'a_rs', **common)) == {
        "log_a_rs_0", "_b_0", "t0_0", "rors_0"
    }
    free = parse_planet_parameter_specs({
        'period': ['free', 'gaussian', 3.0, 0.01], 't0': ['fixed', 1.0],
        'duration': ['free', 'uniform', 0.05, 0.3], 'b': ['free', 'uniform', 0, 1],
        'rprs': 0.095,
    })
    sites = _whitelight_geometry_sites(free, 1, 'duration', **common)
    assert set(sites) == {"period_0", "duration_0", "b_0", "rors_0"}
    assert float(sites["period_0"]) == 3.0 and float(sites["duration_0"]) == 0.12


def test_chain_quality_gates_on_period_and_skips_fixed_sites():
    rng = np.random.default_rng(0)
    grouped = {
        "period_0": 3.0 + rng.normal(0, 1e-3, (2, 100)),
        "t0_0": np.full((2, 100), 1.0),
        "rors_0": 0.1 + rng.normal(0, 1e-3, (2, 100)),
        "logD_0": np.log(0.12) + rng.normal(0, 1e-2, (2, 100)),
        "b_0": 0.3 + rng.normal(0, 1e-2, (2, 100)),
    }
    ess, rhat = _geometry_chain_quality(grouped)
    assert set(ess) == {"period", "b", "duration", "rors"}
    assert all(np.isfinite(value) for value in ess.values())
    primitives = _selected_geometry_primitives({"period_0": np.array([3.0]), "rors_0": np.array([0.1])}, 1)
    assert primitives == {"period_0": 3.0, "rors_0": 0.1}


def test_corner_plot_shows_a_sampled_period_first():
    rng = np.random.default_rng(2)
    samples = {
        "t0_0": 1.0 + rng.normal(0, 1e-4, 300),
        "period_0": 3.0 + rng.normal(0, 1e-3, 300),
        "rors_0": 0.1 + rng.normal(0, 1e-3, 300),
    }
    columns, labels = _corner_columns(samples)
    assert labels[0] == r"$P$ [d]"
    assert len(columns) == 3


# --------------------------------------------------------------------------
# Multi-epoch transit masks
# --------------------------------------------------------------------------

def _window(time, center, half_width):
    return (time >= center - half_width) & (time <= center + half_width)


def test_transit_epoch_mask_covers_every_epoch_in_range():
    time = np.linspace(0.0, 10.0, 2001)
    single = createdatacube.transit_epoch_mask(time, 1.0, 0.1)
    np.testing.assert_array_equal(single, _window(time, 1.0, 0.1))
    multi = createdatacube.transit_epoch_mask(time, 1.0, 0.1, period=3.0)
    expected = np.zeros_like(time, dtype=bool)
    for center in (1.0, 4.0, 7.0, 10.0):
        expected |= _window(time, center, 0.1)
    np.testing.assert_array_equal(multi, expected)
    assert multi.sum() > 3 * single.sum()
    # t0 may lie outside the series; epochs are still found.
    shifted = createdatacube.transit_epoch_mask(time, -5.0, 0.1, period=3.0)
    np.testing.assert_array_equal(shifted, multi)
    assert createdatacube.transit_epoch_mask(time, 1.0, 0.1, period=np.nan).sum() == single.sum()


def test_process_spectroscopy_data_masks_all_epochs(monkeypatch):
    time = np.linspace(100.0, 108.0, 801)
    wavelengths = np.array([1.0, 2.0, 3.0])
    wavelength_err = np.full(3, 0.01)
    flux = np.ones((time.size, 3))
    flux_err = np.full((time.size, 3), 0.1)
    captured = {}

    def fake_unpack(*args, **kwargs):
        return wavelengths, wavelength_err, time, flux, flux_err

    def fake_bin(wave, wave_err, native_flux, native_flux_err, cfg, oot_mask):
        captured["oot_mask"] = oot_mask.copy()
        transposed = native_flux.T.copy()
        return {
            "wavelengths_lr": wave.copy(), "wavelengths_err_lr": wave_err.copy(),
            "flux_lr": transposed.copy(), "flux_err_lr": native_flux_err.T.copy(),
            "wavelengths_hr": wave.copy(), "wavelengths_err_hr": wave_err.copy(),
            "flux_hr": transposed.copy(), "flux_err_hr": native_flux_err.T.copy(),
        }

    monkeypatch.setattr(createdatacube, "unpack_exotedrf_spectra", fake_unpack)
    monkeypatch.setattr(createdatacube, "bin_spectroscopy_data", fake_bin)

    cfg = {
        "instrument": "NIRSPEC/PRISM", "nrs": 1,
        "planet": {"period": ["free", "gaussian", 3.0, 0.01], "t0": 101.0, "duration": 0.2},
    }
    createdatacube.process_spectroscopy_data("NIRSPEC/PRISM", "", "", "test", cfg, "unused.fits")
    half_width = 0.6 * 0.2
    expected = np.zeros_like(time, dtype=bool)
    for center in (101.0, 104.0, 107.0):
        expected |= _window(time, center, half_width)
    np.testing.assert_array_equal(~captured["oot_mask"], expected)

    # An explicit ephemeris (what the pipeline passes) takes precedence.
    createdatacube.process_spectroscopy_data(
        "NIRSPEC/PRISM", "", "", "test", cfg, "unused.fits",
        transit_ephemeris={"period": [None], "t0": [101.0], "duration": [0.2]},
    )
    np.testing.assert_array_equal(~captured["oot_mask"], _window(time, 101.0, half_width))

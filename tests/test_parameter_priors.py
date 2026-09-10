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
    geometry_is_fixed,
    parse_parameter_spec,
    parse_planet_parameter_specs,
    parse_planet_surface_specs,
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
from models.priors import latent_init_site, log_site_name
from plotting import _corner_columns


def _fixed(value):
    return {'value': value, 'prior': 'fixed'}


def _planet(**overrides):
    planet = {
        'period': _fixed(3.0), 't0': _fixed(1.0), 'duration': _fixed(0.12),
        'b': _fixed(0.25), 'rprs': _fixed(0.095),
    }
    planet.update(overrides)
    return planet


# --------------------------------------------------------------------------
# Specification grammar
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ({'value': 1.05, 'prior': 'fixed'}, ParameterSpec('period', 'fixed', 1.05)),
    ({'value': 5, 'prior': 'uniform', 'low': 0, 'high': 10},
     ParameterSpec('period', 'uniform', 5.0, None, 0.0, 10.0)),
    ({'value': 5, 'prior': 'log_uniform', 'low': 1, 'high': 10},
     ParameterSpec('period', 'log_uniform', 5.0, None, 1.0, 10.0)),
    ({'value': 5, 'prior': 'gaussian', 'sigma': 0.2},
     ParameterSpec('period', 'gaussian', 5.0, 0.2, None, None)),
    ({'value': 5, 'prior': 'gaussian', 'sigma': 0.2, 'low': 4},
     ParameterSpec('period', 'gaussian', 5.0, 0.2, 4.0, None)),
    ({'value': 5, 'prior': 'Gaussian', 'sigma': 0.2, 'low': 4, 'high': 6},
     ParameterSpec('period', 'gaussian', 5.0, 0.2, 4.0, 6.0)),
])
def test_mapping_form(raw, expected):
    spec = parse_parameter_spec('period', raw)
    assert spec == expected
    assert spec.center == expected.value
    assert spec.free == (expected.prior != 'fixed')
    assert spec.fixed == (expected.prior == 'fixed')


def test_bounded_and_latent_site_helpers():
    plain = parse_parameter_spec('b', {'value': 0.3, 'prior': 'gaussian', 'sigma': 0.1})
    assert not plain.bounded
    truncated = parse_parameter_spec('b', {'value': 0.3, 'prior': 'gaussian', 'sigma': 0.1, 'high': 1})
    assert truncated.bounded
    fixed = parse_parameter_spec('b', _fixed(0.3))
    assert latent_init_site(fixed, 'b_0') is None
    name, value = latent_init_site(plain, 'b_0')
    assert name == 'b_0' and float(value) == 0.3
    logu = parse_parameter_spec('duration', {'value': 0.1, 'prior': 'log_uniform', 'low': 0.01, 'high': 1})
    name, value = latent_init_site(logu, 'duration_0')
    assert name == 'logD_0' and float(value) == pytest.approx(np.log(0.1))
    assert log_site_name('a_rs_2') == 'log_a_rs_2'
    assert log_site_name('rors_0') == 'log_rors_0'


def test_scaled_specification_converts_every_field():
    spec = parse_parameter_spec('eclipse_depth_ppm', {'value': 800, 'prior': 'gaussian', 'sigma': 100, 'low': 0})
    scaled = spec.scaled(1e-6)
    assert scaled.value == pytest.approx(8e-4)
    assert scaled.sigma == pytest.approx(1e-4)
    assert scaled.low == 0.0 and scaled.high is None and scaled.prior == 'gaussian'


def test_yaml_mapping_form_round_trips_through_the_parser():
    planet = yaml.safe_load(
        "period: {value: 4.05528043, prior: gaussian, sigma: 0.001}\n"
        "t0: {value: 59787.05, prior: uniform, low: 59786.9, high: 59787.2}\n"
        "b: {value: 0.45, prior: gaussian, sigma: 0.05}\n"
        "rprs: {value: 0.1457, prior: fixed}\n"
        "duration: {value: 0.117, prior: fixed}\n"
    )
    specs = parse_planet_parameter_specs(planet)
    assert specs['period'][0].free and specs['period'][0].value == 4.05528043
    assert specs['t0'][0].center == 59787.05
    assert specs['duration'][0].fixed and specs['duration'][0].value == 0.117
    assert specs['ecc'] == (ParameterSpec('ecc', 'fixed', 0.0),)
    assert specs['omega'] == (ParameterSpec('omega', 'fixed', 0.0),)
    np.testing.assert_allclose(planet_parameter_centers(specs, 'period'), [4.05528043])
    np.testing.assert_allclose(planet_parameter_centers(specs, 'a_rs', default=9.0), [9.0])


def test_describe_returns_a_table_with_a_title():
    specs = parse_planet_parameter_specs({
        'period': _fixed(3.0),
        't0': {'value': 1.0, 'prior': 'uniform', 'low': 0.9, 'high': 1.1},
        'b': {'value': 0.3, 'prior': 'gaussian', 'sigma': 0.05, 'low': 0.0},
        'rprs': {'value': 0.1, 'prior': 'log_uniform', 'low': 0.01, 'high': 0.5},
    })
    lines = describe_planet_parameter_specs(specs, title="Table")
    assert lines[0] == "Table:"
    body = "\n".join(lines[1:])
    assert "period" in body and "fixed at 3.0" in body
    assert "uniform(0.9, 1.1), start 1.0" in body
    assert "gaussian(mu=0.3, sigma=0.05) truncated to [0.0, None]" in body
    assert "log-uniform(0.01, 0.5), start 0.1" in body


def test_per_planet_entries_and_broadcasting():
    specs = parse_planet_parameter_specs({
        'period': [_fixed(1.0), {'value': 2.0, 'prior': 'gaussian', 'sigma': 0.1}],
        't0': [_fixed(0.5), {'value': 0.5, 'prior': 'uniform', 'low': 0, 'high': 1}],
        'b': _fixed(0.3),
    })
    assert len(specs['period']) == 2 and specs['period'][1].free
    assert specs['t0'][0].fixed and specs['t0'][1].prior == 'uniform'
    assert specs['b'] == (specs['b'][0],) * 2
    np.testing.assert_allclose(planet_parameter_centers(specs, 'period'), [1.0, 2.0])
    lines = describe_planet_parameter_specs(specs)
    assert any(line.strip().startswith("period[1]") for line in lines)


@pytest.mark.parametrize("planet, message", [
    ({'period': 3.0}, r"planet\.period: every planet parameter is written as a mapping"),
    ({'period': [3.0]}, r"planet\.period: a list is only used for several planets"),
    ({'period': ['free', 'uniform', 1, 2]}, r"planet\.period: a list is only used for several planets"),
    ({'period': {'mode': 'fixed', 'value': 1.0}}, r"planet\.period: 'mode' and 'mu' are no longer accepted"),
    ({'period': {'value': 1.0, 'prior': 'gaussian', 'mu': 1.0, 'sigma': 0.1}}, r"'mode' and 'mu' are no longer accepted"),
    ({'period': {'value': 1.0}}, r"planet\.period: 'prior' is required"),
    ({'period': {'prior': 'fixed'}}, r"planet\.period: 'value' is required"),
    ({'period': {'value': 1.0, 'prior': 'lognormal'}}, r"planet\.period: unknown prior 'lognormal'"),
    ({'period': {'value': 1.0, 'prior': 'fixed', 'wobble': 2}}, r"planet\.period: unknown keys \['wobble'\]"),
    ({'period': {'value': 1.0, 'prior': 'fixed', 'sigma': 0.1}}, r"planet\.period: a fixed parameter takes only 'value'"),
    ({'period': {'value': 'four', 'prior': 'fixed'}}, r"planet\.period: 'value' must be a number"),
    ({'period': {'value': 1.0, 'prior': 'gaussian'}}, r"planet\.period: a gaussian prior needs 'sigma'"),
    ({'period': {'value': 1.0, 'prior': 'gaussian', 'sigma': -0.1}}, r"planet\.period: 'sigma' must be > 0"),
    ({'period': {'value': 1.0, 'prior': 'uniform', 'low': 0}}, r"planet\.period: a uniform prior needs both 'low' and 'high'"),
    ({'period': {'value': 1.0, 'prior': 'uniform', 'low': 0, 'high': 2, 'sigma': 1}}, r"'sigma' only applies to a gaussian prior"),
    ({'period': {'value': 1.0, 'prior': 'uniform', 'low': 2, 'high': 0}}, r"planet\.period: bounds require low < high"),
    ({'period': {'value': 1.0, 'prior': 'gaussian', 'sigma': 0.1, 'low': 2, 'high': 0}}, r"bounds require low < high"),
    ({'period': {'value': 5.0, 'prior': 'uniform', 'low': 0, 'high': 2}}, r"planet\.period: 'value' 5\.0 must lie inside"),
    ({'period': {'value': 1.0, 'prior': 'log_uniform', 'low': 0, 'high': 2}}, r"log_uniform prior needs low > 0"),
    ({'period': {'value': 1.0, 'prior': 'uniform', 'low': 'a', 'high': 2}}, r"planet\.period: 'low' must be a number"),
    ({'period': _fixed(1.0), 'ecc': {'value': 0.1, 'prior': 'uniform', 'low': 0, 'high': 0.5}}, r"planet\.ecc may only be fixed"),
    ({'period': _fixed(1.0), 'omega': {'value': 0, 'prior': 'gaussian', 'sigma': 1}}, r"planet\.omega may only be fixed"),
    ({'period': [_fixed(1.0), _fixed(2.0)], 't0': [_fixed(0.0), _fixed(1.0), _fixed(2.0)]},
     r"planet\.t0 must be one mapping or a list of 2"),
    ({'period': _fixed(1.0), 't0_prior_width_days': 0.1}, r"planet\.t0_prior_width_days has been removed; write t0:"),
    ({'period': _fixed(1.0), 'a_rs_prior_min': 2.0}, r"planet\.a_rs_prior_min has been removed"),
    ({'period': _fixed(1.0), 'eclipse_depth_prior_width_ppm': 100.0}, r"eclipse_depth_prior_width_ppm has been removed"),
])
def test_invalid_specifications_name_the_key(planet, message):
    with pytest.raises(ValueError, match=message):
        parse_planet_parameter_specs(planet)


def test_period_is_required():
    with pytest.raises(KeyError, match="planet.period"):
        parse_planet_parameter_specs({'t0': _fixed(1.0)})


def test_surface_specs_scale_units_and_restrict_priors():
    planet = {
        'eclipse_depth_ppm': {'value': 800, 'prior': 'uniform', 'low': 0, 'high': 1000},
        'hotspot_offset_deg': {'value': 30, 'prior': 'gaussian', 'sigma': 5},
    }
    specs = parse_planet_surface_specs(planet, 2, ['eclipse_depth_ppm', 'hotspot_offset_deg'])
    assert len(specs['eclipse_depth_ppm']) == 2
    assert specs['eclipse_depth_ppm'][0].high == pytest.approx(1e-3)
    assert specs['hotspot_offset_deg'][1].value == pytest.approx(np.pi / 6)
    assert specs['hotspot_offset_deg'][1].sigma == pytest.approx(np.deg2rad(5))
    with pytest.raises(ValueError, match=r"planet\.eclipse_depth_ppm is required"):
        parse_planet_surface_specs({}, 1, ['eclipse_depth_ppm'])
    with pytest.raises(ValueError, match=r"planet\.dayside_flux_ppm supports only 'fixed' or 'gaussian'"):
        parse_planet_surface_specs(
            {'dayside_flux_ppm': {'value': 100, 'prior': 'uniform', 'low': 0, 'high': 200}},
            1, ['dayside_flux_ppm'],
        )
    with pytest.raises(ValueError, match="dayside_flux_prior_width_ppm has been removed"):
        parse_planet_surface_specs(
            {'dayside_flux_ppm': _fixed(100), 'dayside_flux_prior_width_ppm': 10}, 1, ['dayside_flux_ppm'],
        )


def test_geometry_is_fixed_follows_the_parameterisation():
    fixed = parse_planet_parameter_specs(_planet(a_rs=_fixed(9.0)))
    assert geometry_is_fixed(fixed, 'duration') and geometry_is_fixed(fixed, 'a_rs')
    free_duration = parse_planet_parameter_specs(
        _planet(duration={'value': 0.12, 'prior': 'log_uniform', 'low': 0.01, 'high': 1}, a_rs=_fixed(9.0))
    )
    assert not geometry_is_fixed(free_duration, 'duration')
    assert geometry_is_fixed(free_duration, 'a_rs')
    free_period_only = parse_planet_parameter_specs(
        _planet(period={'value': 3.0, 'prior': 'gaussian', 'sigma': 0.01})
    )
    assert geometry_is_fixed(free_period_only, 'duration')


def test_builder_validates_the_specifications():
    specs = parse_planet_parameter_specs(
        _planet(a_rs={'value': 9.0, 'prior': 'uniform', 'low': 5, 'high': 15})
    )
    with pytest.raises(ValueError, match=r"planet\.a_rs is free .* parameterized by duration"):
        _validate_parameter_priors(specs, 1, 'duration')
    specs = parse_planet_parameter_specs(
        _planet(duration={'value': 0.12, 'prior': 'uniform', 'low': 0.05, 'high': 0.3}, a_rs=_fixed(9.0))
    )
    with pytest.raises(ValueError, match=r"planet\.duration is free .* parameterized by a_rs"):
        _validate_parameter_priors(specs, 1, 'a_rs')
    # A fixed value for the unused coordinate is informational and allowed.
    specs = parse_planet_parameter_specs(_planet(a_rs=_fixed(9.0)))
    assert set(_validate_parameter_priors(specs, 1, 'duration')) == set(specs)
    with pytest.raises(ValueError, match="parameter_priors is required"):
        _validate_parameter_priors(None, 1, 'duration')
    with pytest.raises(ValueError, match=r"missing planet\.a_rs"):
        _validate_parameter_priors(parse_planet_parameter_specs(_planet()), 1, 'a_rs')
    with pytest.raises(ValueError, match="expected 2 specifications"):
        _validate_parameter_priors(parse_planet_parameter_specs(_planet()), 2, 'duration')


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


def test_builder_requires_the_specifications():
    with pytest.raises(ValueError, match="parameter_priors is required"):
        _model()


def test_every_prior_type_maps_to_its_site():
    specs = parse_planet_parameter_specs(_planet(
        t0={'value': 1.0, 'prior': 'uniform', 'low': 0.9, 'high': 1.1},
        b={'value': 0.25, 'prior': 'gaussian', 'sigma': 0.05},
        duration={'value': 0.12, 'prior': 'log_uniform', 'low': 0.0007, 'high': 1.0},
        rprs={'value': 0.095, 'prior': 'gaussian', 'sigma': 0.01, 'low': 0.05, 'high': 0.3},
    ))
    trace = _trace(_model(parameter_priors=specs))
    signature = _distribution_signature(trace)
    assert signature["t0_0"] == ("sample", "Uniform", {"low": 0.9, "high": 1.1})
    assert signature["b_0"] == ("sample", "Normal", {"loc": 0.25, "scale": 0.05})
    assert signature["logD_0"][1] == "Uniform"
    np.testing.assert_allclose(signature["logD_0"][2]["low"], np.log(0.0007))
    assert signature["duration_0"] == ("deterministic",)
    np.testing.assert_allclose(trace["duration_0"]["value"], np.exp(trace["logD_0"]["value"]))
    assert signature["rors_0"][1] == "TwoSidedTruncatedDistribution"
    assert signature["rors_0"][2] == {"low": 0.05, "high": 0.3}
    assert signature["period_0"] == ("deterministic",)
    assert float(trace["period_0"]["value"]) == 3.0
    for legacy in ("_b_0", "log_rors_0"):
        assert legacy not in signature
    assert signature["a_rs_0"] == ("deterministic",)


def test_fixed_geometry_has_no_sampled_geometry_sites():
    specs = parse_planet_parameter_specs(_planet())
    trace = _trace(_model(parameter_priors=specs))
    sampled = {name for name, site in trace.items()
               if site["type"] == "sample" and not site.get("is_observed")}
    assert not sampled & {"period_0", "t0_0", "b_0", "rors_0", "logD_0", "duration_0"}
    for name, value in (("t0_0", 1.0), ("b_0", 0.25), ("rors_0", 0.095), ("duration_0", 0.12)):
        assert trace[name]["type"] == "deterministic"
        assert float(trace[name]["value"]) == value


def test_a_rs_parameterisation_uses_the_legacy_log_site_name():
    specs = parse_planet_parameter_specs(_planet(
        a_rs={'value': 9.0, 'prior': 'log_uniform', 'low': 2.0, 'high': 100.0},
    ))
    trace = _trace(_model(parameter_priors=specs, param_method="a_rs"))
    assert trace["log_a_rs_0"]["type"] == "sample"
    np.testing.assert_allclose(trace["a_rs_0"]["value"], np.exp(trace["log_a_rs_0"]["value"]))
    assert trace["duration_0"]["type"] == "deterministic"


def test_free_period_is_sampled_and_drives_the_derived_geometry():
    specs = parse_planet_parameter_specs(_planet(
        period={'value': 3.0, 'prior': 'gaussian', 'sigma': 0.001},
        duration={'value': 0.12, 'prior': 'log_uniform', 'low': 0.0007, 'high': 1.0},
        b={'value': 0.25, 'prior': 'uniform', 'low': 0.0, 'high': 1.0},
        rprs={'value': 0.095, 'prior': 'uniform', 'low': 0.01, 'high': 0.5},
        t0={'value': 1.0, 'prior': 'uniform', 'low': 0.9, 'high': 1.1},
    ))
    trace = _trace(_model(parameter_priors=specs), seed=3)
    site = trace["period_0"]
    assert site["type"] == "sample"
    assert isinstance(site["fn"], dist.Normal)
    assert float(site["fn"].loc) == 3.0 and float(site["fn"].scale) == 0.001

    samples = {
        name: jnp.stack([trace[name]["value"]] * 4)
        for name in ("period_0", "t0_0", "rors_0", "logD_0", "b_0", "duration_0", "a_rs_0")
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


def test_fixed_geometry_with_free_period_is_allowed_for_fixed_surface_geometry():
    specs = parse_planet_parameter_specs(_planet(
        period={'value': 3.0, 'prior': 'uniform', 'low': 2, 'high': 4}, a_rs=_fixed(8.0),
    ))
    model = create_whitelight_model(
        ld_mode="fixed", ld_profile="quadratic", param_method="a_rs",
        surface_config={"model": "transit", "spots": (), "fit_geometry": False},
        parameter_priors=specs,
    )
    trace = _trace(model)
    assert trace["period_0"]["type"] == "sample"
    assert trace["_geometry_fixed"]["type"] == "deterministic"


# --------------------------------------------------------------------------
# White-light stage plumbing
# --------------------------------------------------------------------------

def test_geometry_sites_follow_the_specifications():
    fixed = parse_planet_parameter_specs(_planet(a_rs=_fixed(9.0)))
    assert _whitelight_geometry_sites(fixed, 1, 'duration') == {}
    assert _whitelight_geometry_sites(fixed, 1, 'a_rs') == {}
    free = parse_planet_parameter_specs(_planet(
        period={'value': 3.0, 'prior': 'gaussian', 'sigma': 0.01},
        duration={'value': 0.12, 'prior': 'log_uniform', 'low': 0.05, 'high': 0.3},
        b={'value': 0.25, 'prior': 'uniform', 'low': 0, 'high': 1},
        rprs={'value': 0.095, 'prior': 'uniform', 'low': 0.01, 'high': 0.5},
        a_rs={'value': 9.0, 'prior': 'log_uniform', 'low': 2.0, 'high': 20.0},
    ))
    sites = _whitelight_geometry_sites(free, 1, 'duration')
    assert set(sites) == {"period_0", "logD_0", "b_0", "rors_0"}
    assert float(sites["period_0"]) == 3.0
    np.testing.assert_allclose(sites["logD_0"], np.log(0.12))
    assert float(sites["rors_0"]) == 0.095
    sites = _whitelight_geometry_sites(free, 1, 'a_rs')
    assert set(sites) == {"period_0", "log_a_rs_0", "b_0", "rors_0"}
    np.testing.assert_allclose(sites["log_a_rs_0"], np.log(9.0))


def test_geometry_sites_are_indexed_per_planet():
    specs = parse_planet_parameter_specs({
        'period': [_fixed(1.0), _fixed(2.0)],
        't0': [_fixed(0.5), {'value': 1.5, 'prior': 'uniform', 'low': 1, 'high': 2}],
        'b': _fixed(0.1), 'rprs': _fixed(0.1), 'duration': _fixed(0.1),
    })
    assert set(_whitelight_geometry_sites(specs, 2, 'duration')) == {"t0_1"}


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
        "planet": {
            "period": {"value": 3.0, "prior": "gaussian", "sigma": 0.01},
            "t0": _fixed(101.0), "duration": _fixed(0.2),
        },
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

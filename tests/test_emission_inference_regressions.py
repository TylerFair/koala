"""Regressions for sampled geometry, emission priors, starts, and diagnostics."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer.util import log_density
from scipy.stats import truncnorm

from koala.config import parse_planet_parameter_specs
from koala.geometry import _geometry_chain_quality, _whitelight_science_sites
from koala.sampling import _spectro_failed_lanes
from koala.surface import _prepare_fixed_surface_basis, _seed_surface_spectroscopic_init
from models.jaxoplanet.config import parse_surface_config, surface_science_sites
from models.jaxoplanet.builder import (
    create_whitelight_model, create_vectorized_model, _sample_surface_parameters,
)
from models.jaxoplanet.surface import compute_surface_model

jax.config.update('jax_enable_x64', True)


def fixed(value):
    return {'value': value, 'prior': 'fixed'}


def gaussian(value, sigma, **bounds):
    return {'value': value, 'prior': 'gaussian', 'sigma': sigma, **bounds}


def planet(**updates):
    return {
        'period': fixed(1.), 't0': fixed(0.), 'a_rs': fixed(8.),
        'b': fixed(.2), 'rprs': fixed(.0318),
        'eclipse_depth_ppm': {'value': 100., 'prior': 'uniform', 'low': 0., 'high': 1000.},
        'phase_c1_ppm': gaussian(30., 10., low=-1000., high=1000.),
        'phase_d1_ppm': fixed(0.), **updates,
    }


def configuration(model='eclipse', **updates):
    p = planet(**updates)
    specs = parse_planet_parameter_specs(p)
    config = parse_surface_config({'light_curve_model': model}, p, {}, 1,
                                   parameter_specs=specs, param_method='a_rs')
    return specs, config


@pytest.mark.parametrize('model_name', ['eclipse', 'phase_curve'])
def test_free_period_changes_production_surface_likelihood(model_name):
    specs, config = configuration(model_name, period=gaussian(1., .02))
    # Resolve ingress: the rocky planet's ingress is shorter than a 41-point
    # grid here, which can correctly have zero timing derivative at every point.
    time = jnp.linspace(.47, .57, 201)
    base = {'period': 1., 't0': 0., 'a_rs': 8., 'b': .2, 'rors': .0318,
            'ecc': 0., 'omega': 0., 'u': jnp.zeros(2), 'eclipse_depth': 1e-4,
            'phase_c1': 3e-5, 'phase_d1': 0.}
    basis = _prepare_fixed_surface_basis(config, time, np.zeros(2),
        **{key: np.atleast_1d(base[key]) for key in ('period','t0','a_rs','b','rors','ecc','omega')})
    assert basis is None
    model = create_whitelight_model(ld_mode='fixed', param_method='a_rs',
        surface_config=config, surface_basis=basis, parameter_priors=specs)
    init = _seed_surface_spectroscopic_init({}, config, 1)
    init = {key: value[0] for key, value in init.items()}

    @jax.jit
    def prediction(period):
        trace = handlers.trace(handlers.seed(handlers.substitute(model, data={
            **init, 'period_0': period, 'c': 1., 'v': 0., 'log_jitter': jnp.log(2e-5),
        }), 4)).get_trace(time, jnp.full(time.shape, 3e-5), prior_params={'u': jnp.zeros(2)})
        return trace['obs']['fn'].loc

    first, changed = prediction(1.), prediction(1.04)
    assert float(jnp.max(jnp.abs(changed-first))) > 5e-5
    native = compute_surface_model({**base, 'period': 1.04}, time, model=model_name) + 1.
    np.testing.assert_allclose(changed, native, atol=2e-12, rtol=0.)
    derivative = jax.jacfwd(prediction)(1.013)
    finite = (prediction(1.013+1e-6)-prediction(1.013-1e-6))/2e-6
    assert np.isfinite(derivative).all() and np.max(np.abs(derivative)) > 0
    np.testing.assert_allclose(derivative, finite, atol=2e-7, rtol=5e-4)


@pytest.mark.parametrize('prior', ['uniform', 'log_uniform', 'gaussian', 'fixed'])
def test_spectral_eclipse_seed_uses_requested_latent_and_units(prior):
    depth = {'prior': prior, 'value': 100.}
    if prior == 'gaussian':
        depth.update(sigma=30., low=0.)
    elif prior != 'fixed':
        depth.update(low=1., high=1000.)
    _, config = configuration(eclipse_depth_ppm=depth)
    seeds = _seed_surface_spectroscopic_init({}, config, 3)
    expected = ({} if prior == 'fixed' else {
        'log__eclipse_depth_0' if prior == 'log_uniform' else '_eclipse_depth_0':
            np.full(3, np.log(1e-4) if prior == 'log_uniform' else 1e-4)
    })
    assert seeds.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(seeds[key], expected[key], rtol=1e-14)


@pytest.mark.parametrize('num_lcs', [None, 4])
def test_phase_coefficient_priors_are_independent_and_seeded_at_centers(num_lcs):
    # A nightside far below a fifth of the dayside (and negative C1/D1 draws)
    # is inside the sinusoid's support: the priors are plain independent ones.
    _, config = configuration('phase_curve',
        phase_c1_ppm=gaussian(48., 10., low=-1000., high=1000.),
        phase_d1_ppm={'value': -20., 'prior': 'uniform', 'low': -1000., 'high': 1000.})
    def model():
        return _sample_surface_parameters(config, 1, num_lcs=num_lcs)
    n = num_lcs or 1
    init = _seed_surface_spectroscopic_init({}, config, n)
    if num_lcs is None:
        init = {key: value[0] for key, value in init.items()}
    logp, trace = log_density(model, (), {}, init)
    expected = (np.log(1/1000e-6)
                + truncnorm.logpdf(48e-6, -1048/10, 952/10, loc=48e-6, scale=10e-6)
                + np.log(1/2000e-6)) * n
    np.testing.assert_allclose(float(logp), expected, rtol=1e-10)
    np.testing.assert_allclose(trace['nightside_flux_ppm']['value'], 100. - 2*48., rtol=1e-12)
    np.testing.assert_allclose(trace['phase_offset_deg']['value'],
                               np.degrees(np.arctan2(20., 48.)), rtol=1e-12)


@pytest.mark.parametrize('radius,ecc', [(.01, 0.), (.0318, 0.), (.0318, .2)])
def test_small_eclipse_matches_batman_secondary_flux_and_timing(radius, ecc):
    batman = pytest.importorskip('batman')
    bp = batman.TransitParams()
    bp.t0, bp.per, bp.rp, bp.a = 60000., 2.6162644, radius, 17.04
    bp.ecc, bp.w = ecc, np.rad2deg(.4)
    bp.inc = np.rad2deg(np.arccos(.208/bp.a*(1+ecc*np.sin(.4))/(1-ecc**2)))
    bp.limb_dark, bp.u, bp.fp = 'uniform', [], 100e-6
    primary = batman.TransitModel(bp, np.array([bp.t0]))
    bp.t_secondary = primary.get_t_secondary(bp)
    time = np.linspace(bp.t_secondary-.1, bp.t_secondary+.1, 401)
    expected = batman.TransitModel(bp, time, transittype='secondary').light_curve(bp)
    params = {'period': bp.per, 't0': bp.t0, 'rors': radius, 'a_rs': bp.a,
              'b': .208, 'ecc': ecc, 'omega': .4, 'u': jnp.zeros(2), 'eclipse_depth': bp.fp}
    actual = compute_surface_model(params, jnp.asarray(time), model='eclipse') + 1.
    np.testing.assert_allclose(actual, expected, rtol=0., atol=5e-10)
    np.testing.assert_allclose(np.ptp(actual), 100e-6, rtol=0., atol=5e-10)


def test_white_gate_rejects_stuck_free_depth_and_skips_configured_fixed_parameters():
    specs, config = configuration(t0=gaussian(0., .01))
    sites = _whitelight_science_sites(specs, config, 'a_rs')
    assert sites == {'t0': 't0_0', 'eclipse_depth_0': '_eclipse_depth_0'}
    draws = {'t0_0': np.random.default_rng(9).normal(size=(1,1000)),
             'period_0': np.ones((1,1000)), '_eclipse_depth_0': np.full((1,1000),5e-6)}
    ess, _ = _geometry_chain_quality(draws, science_sites=sites)
    assert ess['t0'] > 400
    assert ess['eclipse_depth_0'] == 0.
    assert 'period' not in ess
    # Missing a required physical output cannot silently pass the gate.
    del draws['_eclipse_depth_0']
    ess, _ = _geometry_chain_quality(draws, science_sites=sites)
    assert np.isnan(ess['eclipse_depth_0'])


def test_spectral_gate_checks_only_free_phase_sites_and_still_rejects_stuck_flux():
    _, config = configuration('phase_curve')
    model = create_vectorized_model(ld_mode='fixed', param_method='a_rs', surface_config=config)
    assert model.surface_science_sites == surface_science_sites(config)
    rng = np.random.default_rng(26)
    samples = {'depths': np.full((2000,1,1),.0318**2),
               '_geometry_fixed': np.ones((2000,1),dtype=bool),
               '_eclipse_depth_0': rng.normal(100e-6,1e-7,(2000,1)),
               '_phase_c1_0': rng.normal(30e-6,1e-7,(2000,1)),
               '_phase_d1_0': np.zeros((2000,1))}
    failed, metrics = _spectro_failed_lanes(samples, None, 400., 0,
                                          surface_science_sites=model.surface_science_sites)
    assert failed.size == 0
    assert set(metrics['science_ess_per_channel']) == {'eclipse_depth_0', 'phase_c1_0'}
    samples['_eclipse_depth_0'][:] = 100e-6
    failed, _ = _spectro_failed_lanes(samples, None, 400., 0,
                                     surface_science_sites=model.surface_science_sites)
    assert failed.tolist() == [0]


def test_spectral_gate_all_science_fixed_still_checks_divergences(tmp_path):
    import json
    samples = {'depths': np.ones((20,1,1)), '_geometry_fixed': np.ones((20,1)),
               '_eclipse_depth_0': np.full((20,1),100e-6)}
    failed, _ = _spectro_failed_lanes(samples, None, 400., 0, surface_science_sites=())
    assert failed.size == 0
    path = tmp_path/'diagnostics.json'
    path.write_text(json.dumps({'num_divergences_per_channel': [1]}))
    failed, _ = _spectro_failed_lanes(samples, path, 400., 0, surface_science_sites=())
    assert failed.tolist() == [0]

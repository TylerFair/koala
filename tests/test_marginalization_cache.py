"""Ownership, scientific cache separation, and compatible sweep starts."""
from types import SimpleNamespace
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import jax.numpy as jnp
import numpy as np

from koala.marginalization.fitting import SweepContext
from models.mcmc_diagnostics import PosteriorSamples


def test_prepared_data_shares_immutable_device_buffers_and_copies_host_arrays(tmp_path):
    sweep = SweepContext(tmp_path)
    original = SimpleNamespace(device=jnp.arange(10.), host=np.arange(10.))
    artifact = SimpleNamespace(fingerprint='science-a', is_reusable=lambda: True)
    calls = []
    first = sweep.load_data(artifact, lambda: calls.append('load') or original)
    first.host[:] = -1
    first.added = True
    second = sweep.load_data(artifact, lambda: calls.append('unexpected') or original)
    assert calls == ['load']
    assert first.device is second.device is original.device
    np.testing.assert_array_equal(second.host, np.arange(10.))
    assert not hasattr(second, 'added')
    assert sweep.stats['data_hits'] == 1


def test_factories_and_runners_only_reuse_matching_scientific_programs(tmp_path):
    sweep = SweepContext(tmp_path)
    builds = []
    def builder(**kwargs):
        builds.append(kwargs)
        return object()
    cached = sweep.factory(builder)
    a = cached(prior='uniform', data=np.array([1., 2.]))
    assert cached(prior='uniform', data=np.array([1., 2.])) is a
    b = cached(prior='fixed', data=np.array([1., 2.]))
    c = cached(prior='uniform', data=np.array([1., 3.]))
    assert len(builds) == 3 and b is not a and c is not a
    first = sweep.runner_caches(a, {'target_accept_prob': .95}, {'num_samples': 1000})
    first[0]['program'] = 'compiled'
    assert sweep.runner_caches(a, {'target_accept_prob': .95}, {'num_samples': 1000}) is first
    assert sweep.runner_caches(b, {'target_accept_prob': .95}, {'num_samples': 1000}) != first
    assert sweep.runner_caches(a, {'target_accept_prob': .99}, {'num_samples': 1000}) != first


def test_common_starts_do_not_copy_other_law_or_incompatible_shapes(tmp_path):
    sweep = SweepContext(tmp_path)
    posterior = PosteriorSamples({'rors': np.array([[.1], [.2], [.3]]),
        'u': np.tile([.3, .2], (3, 1)), 'unrelated': np.ones((3, 2))}, sampled_sites=None)
    sweep.remember('highres', posterior, law='quadratic')
    initial = {'rors': np.array([.05]), 'u': np.array([.5, .4]), 'unrelated': np.zeros(2)}
    same = sweep.warm_start('highres', initial, law='quadratic')
    changed = sweep.warm_start('highres', initial, law='power2')
    np.testing.assert_allclose(same['rors'], [.2])
    np.testing.assert_allclose(same['u'], [.3, .2])
    np.testing.assert_allclose(changed['u'], initial['u'])
    np.testing.assert_allclose(same['unrelated'], initial['unrelated'])
    wide = sweep.warm_start('highres', {'rors': np.array([.05, .05])}, law='quadratic')
    np.testing.assert_allclose(wide['rors'], [.05, .05])


def test_matrix_prior_changes_do_not_transfer_starts_outside_new_support(tmp_path):
    from koala.config import ParameterSpec
    sweep = SweepContext(tmp_path)
    sweep.remember('white', {'rors_0': np.array([.2, .3, .4]),
                             'logD_0': np.log(np.array([.1, .2, .3]))}, law='quadratic')
    specs = {'rprs': (ParameterSpec('rprs', 'uniform', .07, low=.05, high=.1),),
             'duration': (ParameterSpec('duration', 'log_uniform', .03, low=.02, high=.05),)}
    initial = {'rors_0': np.array(.07), 'logD_0': np.array(np.log(.03))}
    actual = sweep.warm_start('white', initial, law='quadratic', parameter_specs=specs)
    for name in initial:
        np.testing.assert_array_equal(actual[name], initial[name])


def test_fixed_ld_starts_outside_uniform_prior_are_not_transferred(tmp_path):
    sweep = SweepContext(tmp_path)
    sweep.remember('white', {'c1':np.array([1.2, 1.3]), 'c2':np.array([.2, .3])}, law='power2')
    values = sweep.warm_start('white', {'c1':.5, 'c2':.5}, law='power2',
        site_bounds=sweep.ld_site_bounds('power2', 'uniform', 'coefficients'))
    assert values['c1'] == .5
    np.testing.assert_allclose(values['c2'], .25)


def test_completed_target_eviction_preserves_preparation_and_starts(tmp_path, monkeypatch):
    import jax
    from koala.marginalization import fitting
    sweep = SweepContext(tmp_path)
    sweep._data['shared'] = object()
    sweep._models['old'] = object()
    sweep._runners['old'] = object()
    sweep._starts['white'] = {'values':{'rors_0':.1}}
    events = []
    monkeypatch.setattr(fitting, '_resident_mb', lambda:500.)
    monkeypatch.setattr(jax, 'clear_caches', lambda:events.append('evict'))
    assert not sweep.finish_target(0)
    assert not sweep.finish_target(1000)
    assert sweep.finish_target(400)
    assert not sweep._models and not sweep._runners
    assert 'shared' in sweep._data and sweep._starts['white']['values']['rors_0'] == .1
    assert events == ['evict'] and len(sweep.stats['compilation_evictions']) == 1


def test_repeated_observations_retain_one_cube_and_slice_across_variants():
    from koala.marginalization.batched_fitting import _RepeatedChannels
    original = np.arange(3*7).reshape(3, 7)
    repeated = _RepeatedChannels(original, 4)
    assert repeated.values is original and repeated.shape == (12, 7)
    assert np.shares_memory(repeated[3:5], original)
    tiled = np.tile(original, (4, 1))
    for selection in (slice(0, 4), slice(2, 9), slice(10, 12), slice(None, None, 2),
                      slice(None, None, -1), np.array([11, 3, 4]), -1):
        np.testing.assert_array_equal(repeated[selection], tiled[selection])


def test_batched_white_starts_transfer_geometry_and_trends_across_laws(tmp_path):
    sweep = SweepContext(tmp_path)
    posterior = PosteriorSamples({'b_0':np.tile([.23, .31], (10, 1)),
        'trend_coordinates':np.tile(np.array([[1., .003, .001], [.99, .002, .004]])[None], (10, 1, 1)),
        'ld_coordinates':np.ones((10, 2, 2))*.2},
        sampled_sites=('b_0', 'trend_coordinates', 'ld_coordinates'))
    sweep.remember('white', posterior, law='quadratic')
    initial = {'b_0':np.full(2, .5), 'trend_coordinates':np.zeros((2, 3)),
               'ld_coordinates':np.full((2, 2), .5)}
    warm = sweep.warm_start('white', initial, law='power2')
    np.testing.assert_array_equal(warm['b_0'], posterior['b_0'][0])
    np.testing.assert_array_equal(warm['trend_coordinates'], posterior['trend_coordinates'][0])
    np.testing.assert_array_equal(warm['ld_coordinates'], initial['ld_coordinates'])

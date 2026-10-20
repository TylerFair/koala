"""Disk chunks preserve exact samples, science gates and streamed LOO scores."""
import pickle
import numpy as np
import pytest

from models.mcmc_diagnostics import PosteriorSamples
from koala.marginalization.posterior import PosteriorStore
from koala.marginalization.likelihood import CheckpointPosterior, load_samples
from koala.marginalization.scoring import score_variant
from test_marginalization import normal_stage


def samples(begin, end):
    rng = np.random.default_rng(begin+73)
    return PosteriorSamples(dict(location=rng.normal(0, .1, (32, end-begin)),
        rors=rng.uniform(.08, .12, (32, end-begin, 1))), num_chains=2,
        sampled_sites=('location', 'rors'))


def test_resume_verified_chunks_and_export_member_boundaries(tmp_path):
    store = PosteriorStore(tmp_path/'cache', 'science', lanes=5, chains=2, draws=16)
    first, second = samples(0, 2), samples(2, 5)
    store.save(0, 2, first, dict(passed=True))
    assert not store.complete()
    store = PosteriorStore(tmp_path/'cache', 'science', lanes=5, chains=2, draws=16)
    assert store.accepted(0, 2) and store.accepted(0, 3) is None
    store.save(2, 5, second, dict(passed=True))
    assert store.complete()
    paths = store.export(1, 4, tmp_path/'fit', 'demo')
    assert [p.name for p in paths] == ['demo_chunk_0_1.npz', 'demo_chunk_1_3.npz']
    loaded = load_samples(paths)
    for key in first:
        expected = np.concatenate((first[key], second[key]), axis=1)[:, 1:4]
        np.testing.assert_array_equal(loaded[key], expected)
    full = store.export(0, 2, tmp_path/'copy', 'demo')[0]
    source = store.directory/'posterior_0_2.npz'
    assert full.read_bytes() == source.read_bytes()
    assert full.stat().st_ino != source.stat().st_ino
    source.write_bytes(b'changed')
    assert store.accepted(0, 2) is None and not store.complete()
    with pytest.raises(ValueError, match='changed'):
        store.export(0, 5, tmp_path/'changed', 'demo')
    assert not PosteriorStore(store.directory, 'new science', lanes=5, chains=2, draws=16).entries


def test_failed_malformed_overlapping_chunks_never_accepted(tmp_path):
    store = PosteriorStore(tmp_path, 'science', lanes=4, chains=2, draws=16)
    with pytest.raises(ValueError, match='Unconverged'):
        store.save(0, 2, samples(0, 2), dict(passed=False))
    with pytest.raises(ValueError, match='dimensions'):
        store.save(0, 2, samples(0, 1), dict(passed=True))
    store.save(0, 2, samples(0, 2), dict(passed=True))
    with pytest.raises(ValueError, match='overlap'):
        store.save(1, 3, samples(1, 3), dict(passed=True))
    bad = samples(2, 4)
    bad['rors'][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        store.save(2, 4, bad, dict(passed=True))
    with pytest.raises(ValueError, match='range'):
        store.export(-1, 4, tmp_path/'out', 'demo')


def test_npz_and_legacy_streamed_scoring_equal_pooled_likelihood(tmp_path):
    stage = normal_stage()
    first, second = samples(0, 1), samples(1, 3)
    paths = [tmp_path/'demo_chunk_0_1.pkl', tmp_path/'demo_chunk_1_3.npz']
    paths[0].write_bytes(pickle.dumps(dict(samples=first)))
    np.savez(paths[1], **second)
    pooled = load_samples(paths)
    expected = score_variant(stage, pooled, draw_batch_size=7, point_batch_size=3)
    posterior = CheckpointPosterior(paths, stage)
    actual = score_variant(stage, posterior, draw_batch_size=7, point_batch_size=3)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    assert posterior.depth_complete
    np.testing.assert_array_equal(posterior.fill_depth(), pooled['rors'][..., 0]**2)
    renamed = tmp_path/'demo_chunk_1_2.npz'
    renamed.write_bytes(paths[1].read_bytes())
    with pytest.raises(ValueError, match='filename range'):
        list(CheckpointPosterior([paths[0], renamed], stage).blocks())
    first['t0'] = np.zeros((32, 1))
    paths[0].write_bytes(pickle.dumps(first))
    with pytest.raises(ValueError, match='joint-geometry'):
        CheckpointPosterior(paths, stage)


def test_real_streamed_sampler_reuses_accepted_chunks_without_adaptation(tmp_path, monkeypatch):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    import models.independent_nuts as nuts
    from koala.marginalization.batched_fitting import _sample
    def model(t, yerr, y=None):
        x = numpyro.sample('x', dist.Normal(jnp.zeros(len(yerr)), 1.))
        numpyro.sample('obs', dist.Normal(x[:, None]+jnp.zeros_like(t), yerr), obs=y)
    cache = tmp_path/'accepted.npz'
    args = (model, np.arange(5.), np.ones((3, 5)), np.zeros((3, 5)),
            {'x':np.zeros(3)}, {}, dict(chains=4, draws=500, warmup=200,
                seed=13, min_ess=400, width=2, progress=False), cache, 'science')
    store, report = _sample(*args, stream=True)
    assert store.complete() and report['quality']['passed']
    assert report['chunks'] == 2 and report['sampler_program_builds'] == 1
    monkeypatch.setattr(nuts, 'get_samples_independent', lambda *a, **k:pytest.fail('Accepted chunk sampled again'))
    # Remove only the headline: verified per-chunk progress resumes the run.
    cache.with_suffix('.json').unlink()
    resumed, report = _sample(*args, stream=True)
    assert resumed.complete() and report['quality']['passed']
    cached, report = _sample(*args, stream=True)
    assert cached.complete() and report['cache_reused']


def test_converged_divergent_lane_uses_fresh_strict_laplace_retry(tmp_path, monkeypatch):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    import models.independent_nuts as nuts
    from koala.marginalization.batched_fitting import _sample
    def model(t, yerr, y=None):
        x = numpyro.sample('x', dist.Normal(jnp.zeros(len(yerr)), 1.))
        numpyro.sample('obs', dist.Normal(x[:, None]+jnp.zeros_like(t), yerr), obs=y)
    ordinary = nuts.get_samples_independent
    calls = []
    def marked(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        calls.append((kwargs['nuts_kwargs']['mass_matrix'], kwargs['nuts_kwargs']['target_accept_prob']))
        if len(calls) == 1:
            # Force the failure branch while leaving the independently sampled
            # retry and all of its convergence checks untouched.
            raw = kwargs['_runner']._last_pooled_raw
            raw['diverging'] = np.asarray(raw['diverging']).copy()
            raw['diverging'][0, 0] = True
        return result
    monkeypatch.setattr(nuts, 'get_samples_independent', marked)
    store, report = _sample(model, np.arange(5.), np.ones((2, 5)), np.zeros((2, 5)),
        {'x':np.zeros(2)}, {}, dict(chains=4, draws=500, warmup=200,
            seed=13, min_ess=400, width=2, progress=False), tmp_path/'retried.npz', 'science', stream=True)
    assert store.complete() and report['quality']['passed'] and report['quality']['divergences'] == 0
    assert calls == [('laplace', .95), ('laplace', .99)]
    retry = report['retries'][0]
    assert retry['lanes'] == [0] and retry['accepted']
    assert retry['attempts'][0]['strategy'] == 'laplace_retuned'

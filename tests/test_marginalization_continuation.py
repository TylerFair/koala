"""Continuation preserves real draws, chain identity and divergence evidence."""
import numpy as np
import pytest
from models.mcmc_diagnostics import PosteriorSamples, pool_chains
from koala.marginalization import batched_fitting as fitting


def fixture():
    groups = np.arange(12).reshape(2, 6, 1).astype(float)
    samples = PosteriorSamples(pool_chains({'x': groups}), num_chains=2, sampled_sites=('x',))
    raw = dict(num_steps=np.arange(12).reshape(12, 1), accept_prob=np.ones((12, 1)),
               diverging=np.zeros((12, 1), bool), step_size=np.ones(1), mean_accept_prob=np.ones(1))
    return groups, samples, raw


@pytest.mark.parametrize('divergence', [False, True])
def test_continued_chains_thin_real_history_and_keep_all_divergences(monkeypatch, divergence):
    groups, samples, raw = fixture()
    more = pool_chains({'x': groups+100})
    extra = {k: v.copy() for k, v in raw.items()}
    if divergence:
        # Iteration one is omitted by even thinning; it must still block a
        # successful convergence report.
        extra['diverging'][2] = True
    extra['num_steps'] += 100
    calls = []
    class Runner:
        def continue_raw(self):
            calls.append(True)
            return more, extra
    def quality(samples, diagnostics, **kwargs):
        return dict(passed=len(calls) > 0, divergences=0, failed_lanes=[])
    monkeypatch.setattr(fitting, '_quality', quality)
    result, diagnostic, report, blocks = fitting._extend_to_quality(Runner(), samples, raw,
                                                                    min_ess=400, max_blocks=1)
    positions = np.linspace(0, 11, 6).astype(int)
    expected = np.concatenate((groups, groups+100), axis=1)[:, positions]
    np.testing.assert_array_equal(result.group_by_chain()['x'], expected)
    steps = np.concatenate((raw['num_steps'].reshape(6, 2, 1),
                            extra['num_steps'].reshape(6, 2, 1)), axis=0)[positions]
    np.testing.assert_array_equal(diagnostic['num_steps'], steps.reshape(12, 1))
    assert blocks == 1 and len(calls) == 1
    assert report['passed'] is (not divergence)
    assert report['divergences'] == int(divergence)
    assert report['failed_lanes'] == ([0] if divergence else [])


def test_white_light_ld_offset_initializes_native_bandpasses():
    white = {'u': np.array([[[.4, .3]], [[.5, .4]], [[.6, .5]]])}
    fixed_white = np.array([[.3, .2]])
    fixed_high = np.array([[.2, .1], [.4, .3]])
    actual = fitting._highres_ld_start(white, fixed_white, fixed_high, 2,
                                     'quadratic', 'coefficients', np.ones(2, bool), 'coefficients')
    np.testing.assert_allclose(actual, [[.4, .3], [.6, .5]])


@pytest.mark.parametrize('metric,bad', [('tail_ess', 399.), ('tail_ess', np.nan),
                                       ('rhat', 1.01), ('bulk_ess', 399.)])
def test_quality_rejects_low_tail_ess_and_fiducial_rhat_boundary(monkeypatch, metric, bad):
    import models.mcmc_diagnostics as diagnostics
    from models.independent_nuts import IndependentNUTSDiagnostics
    _, posterior, raw = fixture()
    def ranks(values):
        result = {'bulk_ess': {'x': np.array([800.])},
                  'tail_ess': {'x': np.array([800.])}, 'rhat': {'x': np.array([1.001])}}
        result[metric]['x'][0] = bad
        return result
    monkeypatch.setattr(diagnostics, 'rank_diagnostics', ranks)
    report = fitting._quality(posterior, IndependentNUTSDiagnostics(**raw), min_ess=400.)
    assert not report['passed'] and report['failed_lanes'] == [0]

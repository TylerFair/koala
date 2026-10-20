"""Independent-chain sampling, convergence failures, and chain provenance."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from models.mcmc_diagnostics import PosteriorSamples, pool_chains, rank_diagnostics
from models.independent_nuts import get_samples_independent, build_independent_nuts_runner
from models.independent_hmc import get_samples_independent_hmc
from koala.sampling import _spectro_failed_lanes, _concatenate_chunk_samples, _load_chunk_samples
from koala.config import _resolve_stage_mcmc_kwargs

jax.config.update('jax_enable_x64', True)


def normal_model(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample('x',dist.Normal(prior_loc,.7))
    numpyro.sample('obs',dist.Normal(x[:,None]+jnp.zeros_like(t),yerr),obs=y)


@pytest.mark.parametrize('backend,metric,start_at_map',[
    ('nuts','adaptive',True), ('nuts','laplace',True),
    ('nuts','laplace',False), ('hmc','laplace',True),
])
def test_four_independent_chains_match_analytic_posterior(backend,metric,start_at_map):
    t=jnp.arange(4.)
    y=jnp.array([[1.]*4,[-1.]*4]); error=jnp.full(y.shape,.25)
    prior=jnp.zeros(2)
    kwargs=dict(mcmc_kwargs=dict(num_chains=4,num_warmup=150,num_samples=500,progress_bar=False),
                nuts_kwargs=dict(mass_matrix=metric,laplace_warmup=150,laplace_start_at_map=start_at_map,
                                 laplace_map_iterations=20),lane_width=3,channel_varying_kwargs=('prior_loc',))
    sampler = get_samples_independent if backend == 'nuts' else get_samples_independent_hmc
    samples,diag=sampler(normal_model,jax.random.PRNGKey(265),t,error,y,
        {'x':jnp.zeros(2)},prior_loc=prior,return_diagnostics=True,**kwargs)
    assert samples['x'].shape==(2000,2)
    grouped=samples.group_by_chain()['x']
    assert grouped.shape==(4,500,2)
    assert diag.num_chains==4 and diag.step_size_by_chain.shape==(4,2)
    assert not np.array_equal(grouped[0],grouped[1])
    variance=1/(1/.7**2+4/.25**2)
    mean=variance*4*np.array([1.,-1.])/.25**2
    np.testing.assert_allclose(np.mean(samples['x'],axis=0),mean,atol=.02,rtol=0)
    np.testing.assert_allclose(np.std(samples['x'],axis=0),np.sqrt(variance),atol=.015,rtol=0)
    diagnostics=rank_diagnostics({'x':grouped})
    assert np.all(diagnostics['rhat']['x']<1.03)
    assert np.all(diagnostics['bulk_ess']['x']>400)
    assert int(np.sum(diag.diverging))==0


def good_samples(seed=265):
    grouped=np.random.default_rng(seed).normal(.02,.001,(4,250,2,1))
    return PosteriorSamples(pool_chains({'depths':grouped,'u':np.ones((4,250,2,2))}),
                            num_chains=4,sampled_sites=('rors',))


def test_ess_gate_is_total_across_chains_not_400_per_chain():
    samples=good_samples()
    failed,metrics=_spectro_failed_lanes(samples,None,400,0)
    assert samples.group_by_chain()['depths'].shape[1]==250
    assert not len(failed)
    assert min(metrics['depth_ess_per_channel'])>400
    assert metrics['num_chains']==4


def test_disagreeing_chains_fail_even_with_many_independent_draws():
    grouped=np.random.default_rng(265).normal(0,.001,(4,1000,2,1))
    grouped[:, :, 1, 0]+=np.arange(4)[:,None]*.1
    samples=PosteriorSamples(pool_chains({'depths':grouped}),num_chains=4)
    failed,metrics=_spectro_failed_lanes(samples,None,400,0)
    np.testing.assert_array_equal(failed,[1])
    assert metrics['science_rhat_per_channel']['depths'][1]>1.01


def test_constant_nuisance_site_is_reported_but_not_gated():
    samples=good_samples()
    assert not len(_spectro_failed_lanes(samples,None,400,0)[0])
    samples.sampled_sites=('rors','u')
    failed,metrics=_spectro_failed_lanes(samples,None,400,0)
    assert not len(failed)
    assert 'u' in metrics['nuisance_ess_per_channel']
    assert 'u' not in metrics['science_ess_per_channel']


def test_stuck_depth_still_fails_the_gate():
    samples=good_samples()
    samples['depths']=np.full_like(np.asarray(samples['depths']),.02)
    failed,_=_spectro_failed_lanes(samples,None,400,0)
    np.testing.assert_array_equal(failed,[0,1])


def test_disagreeing_nuisance_chains_are_reported_but_only_depth_is_gated():
    samples = good_samples()
    latent = np.random.default_rng(10).normal(size=(4, 250, 2, 2))
    latent[:, :, 1, :] += np.arange(4)[:, None, None] * 0.6
    samples.update(pool_chains({'ld_uplus_uminus': latent}))
    samples.sampled_sites = ('rors', 'ld_uplus_uminus')
    failed, metrics = _spectro_failed_lanes(samples, None, 400, 0)
    assert not len(failed)
    assert metrics['nuisance_rhat_per_channel']['ld_uplus_uminus'][1] > 1.01
    assert set(metrics['science_rhat_per_channel']) == {'depths'}


def test_divergences_still_fail_lanes_under_the_depth_only_gate(tmp_path):
    import json
    samples = good_samples()
    path = tmp_path / 'diagnostics.json'
    path.write_text(json.dumps({'num_divergences_per_channel': [0, 2]}))
    failed, _ = _spectro_failed_lanes(samples, path, 400, 0)
    np.testing.assert_array_equal(failed, [1])


def test_one_poorly_mixing_chain_fails_the_per_chain_depth_ess_floor():
    rng = np.random.default_rng(7)
    grouped = rng.normal(.02, .001, (4, 1000, 2, 1))
    walk = np.zeros(1000)
    for index in range(1, 1000):
        walk[index] = .995 * walk[index - 1] + np.sqrt(1 - .995**2) * rng.normal()
    grouped[3, :, 1, 0] = .02 + .001 * walk
    samples = PosteriorSamples(pool_chains({'depths': grouped}), num_chains=4)
    failed, metrics = _spectro_failed_lanes(samples, None, 400, 0)
    assert metrics['science_min_chain_ess_per_channel']['depths'][1] < 100
    assert metrics['science_min_chain_ess_per_channel']['depths'][0] > 100
    assert 1 in failed.tolist() and 0 not in failed.tolist()


def test_laplace_start_backtracking_handles_curved_posteriors():
    from types import SimpleNamespace
    from models.independent_nuts import dispersed_laplace_starts
    def potential(z):
        return .5*z['x']**2 + .5*((z['y'] - 10*z['x']**2)/.001)**2
    preparation = SimpleNamespace(unconstrained_map={'x':jnp.array(0.),'y':jnp.array(0.)},
                                  inverse_mass_matrix=jnp.diag(jnp.array([1.,1e-6])),
                                  potential_fn=potential)
    starts, diagnostic = dispersed_laplace_starts(preparation, jax.random.PRNGKey(265), 4)
    energies = jax.vmap(potential)(starts)
    assert np.all(np.isfinite(energies)) and np.all(energies <= 10.)
    assert np.max(diagnostic['chain_start_backtracks']) > 0
    assert len(np.unique(np.asarray(starts['x']))) == 4


def test_chain_identity_survives_device_copy_checkpoints_and_concatenation(tmp_path):
    samples=good_samples()
    copied=jax.device_get(samples)
    assert copied.num_chains==4
    np.testing.assert_array_equal(copied.group_by_chain()['depths'],samples.group_by_chain()['depths'])
    path=tmp_path/'chunk.pkl'
    with path.open('wb') as file:
        pickle.dump(dict(samples=dict(copied),sampler_used=['independent_nuts']*2,
                         num_chains=4,sampled_sites=copied.sampled_sites),file)
    restored=_load_chunk_samples(path)
    joined=_concatenate_chunk_samples([restored,restored])
    assert joined.num_chains==4
    assert joined.sampled_sites == restored.sampled_sites
    assert joined.group_by_chain()['depths'].shape==(4,250,4,1)
    np.testing.assert_array_equal(joined.group_by_chain()['depths'][:,:,:2],samples.group_by_chain()['depths'])


def test_four_chains_are_stage_defaults_and_one_chain_override_is_explicit():
    for stage in ('whitelight','lowres','highres'):
        settings=_resolve_stage_mcmc_kwargs({},stage)
        assert settings['num_chains']==4 and settings['num_samples']==1000
        assert settings['chain_method']=='vectorized'
        assert _resolve_stage_mcmc_kwargs({stage+'_num_chains':1},stage)['num_chains']==1
    with pytest.raises(ValueError):
        _resolve_stage_mcmc_kwargs({'num_chains':0},'highres')


@pytest.mark.parametrize('disagree', [False, True])
def test_white_light_gate_uses_combined_ess_and_chain_agreement(disagree):
    from koala.geometry import _continue_mcmc_until_geometry_gate
    values = np.random.default_rng(265).normal(size=(4, 250))
    if disagree:
        values += np.arange(4)[:, None] * 0.3

    class SamplesOnlyMCMC:
        def get_samples(self, group_by_chain):
            assert group_by_chain
            return {'rors_0': values}

        def get_extra_fields(self, group_by_chain):
            assert group_by_chain
            return {'diverging': np.zeros((4, 250), dtype=bool)}

    samples, grouped, _, diagnostic = _continue_mcmc_until_geometry_gate(
        SamplesOnlyMCMC(), jax.random.PRNGKey(265), (), {},
        science_sites={'rors': 'rors_0'}, max_extra_blocks=0,
    )
    assert diagnostic['whitelight_quality_gate_passed'] is (not disagree)
    assert samples.num_chains == 4 and samples['rors_0'].shape == (1000,)
    np.testing.assert_array_equal(samples.group_by_chain()['rors_0'], grouped['rors_0'])

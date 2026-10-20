"""Continued histories retain chain identity and pass gates after selection."""
import json
import jax.numpy as jnp
import numpy as np
import pytest
from models.mcmc_diagnostics import PosteriorSamples, pool_chains
from koala.sampling import _continue_independent_to_gate, _align_power2_fallback_coordinates


def _posterior(seed, draws=1000):
    rng=np.random.default_rng(seed)
    r=.1+.001*rng.normal(size=(4,draws,2,1))
    return PosteriorSamples(pool_chains({'rors':jnp.asarray(r),'depths':jnp.asarray(r*r)}),
                            num_chains=4,sampled_sites=('rors',))


def _raw(samples):
    shape=samples['rors'].shape[:2]
    return dict(num_steps=jnp.full(shape,7),accept_prob=jnp.full(shape,.95),
        diverging=jnp.zeros(shape,dtype=bool),step_size=jnp.full(shape[1],.1),
        mean_accept_prob=jnp.full(shape[1],.95),num_chains=4,
        step_size_by_chain=jnp.full((4,shape[1]),.1),
        mean_accept_prob_by_chain=jnp.full((4,shape[1]),.95))


class _Runner:
    def __init__(self,first,extra,diverge=False):
        self._last_pooled_raw=_raw(first)
        self.extra=extra
        self.diverge=diverge
        self.calls=0
    def continue_raw(self):
        self.calls+=1
        raw=_raw(self.extra)
        if self.diverge:raw['diverging']=raw['diverging'].at[0,0].set(True)
        return dict(self.extra),raw


def test_extension_keeps_unique_draws_and_gates_retained_chains(tmp_path):
    first,extra=_posterior(12),_posterior(14)
    runner=_Runner(first,extra)
    path=str(tmp_path/'diagnostics.json')
    kept,failed,attempts=_continue_independent_to_gate('independent_nuts',runner,
        first,path,400,0,None,'test',initial_failed=np.array([0]))
    assert runner.calls==1
    assert not failed.size
    assert kept.num_chains==4 and kept.sampled_sites==('rors',)
    assert kept.group_by_chain()['rors'].shape==(4,1000,2,1)
    history=jnp.concatenate((first['rors'],extra['rors']))
    expected=(np.arange(1,2000,2)[:,None]*4+np.arange(4)).reshape(-1)
    np.testing.assert_array_equal(kept['rors'],history[expected])
    assert len(np.unique(np.asarray(kept['rors'])[:,0,0]))==4000
    assert attempts[0]['raw_draws_per_chain']==2000
    assert attempts[0]['retained_draws_per_chain']==1000
    assert json.loads(open(path).read())['num_draws']==4000


def test_divergent_extension_is_not_accepted(tmp_path):
    first,extra=_posterior(12,20),_posterior(14,20)
    runner=_Runner(first,extra,diverge=True)
    kept,failed,attempts=_continue_independent_to_gate('independent_nuts',runner,
        first,str(tmp_path/'diagnostics.json'),400,0,None,'test',initial_failed=np.array([0]))
    assert kept is first
    np.testing.assert_array_equal(failed,[0])
    assert not attempts


def test_failed_chunk_saves_sampler_audit_without_posterior(tmp_path, monkeypatch):
    import koala.sampling as sampling
    posterior = _posterior(12, 20)
    posterior = PosteriorSamples({name:value[:, :1] for name,value in posterior.items()},
                                 num_chains=4, sampled_sites=('rors',))

    def fake_sample(*args, diagnostics_path=None, **kwargs):
        if diagnostics_path is not None:
            with open(diagnostics_path, 'w') as stream:
                json.dump({'num_divergences':0}, stream)
        return posterior

    def fail_gate(*args, **kwargs):
        return np.array([0]), {'bulk_ess_per_channel':[50.],
                              'num_divergences_per_channel':[0]}

    monkeypatch.setattr(sampling, '_sample_independent_chunk', fake_sample)
    monkeypatch.setattr(sampling, 'get_samples', fake_sample)
    monkeypatch.setattr(sampling, '_build_numpyro_mcmc', lambda *a, **k:(None,None,None))
    monkeypatch.setattr(sampling, '_spectro_failed_lanes', fail_gate)
    with pytest.raises(RuntimeError, match='failed the spectroscopic'):
        sampling.get_samples_chunked(lambda *args:None, __import__('jax').random.PRNGKey(11),
            jnp.arange(4.),jnp.full((1,4),.1),jnp.full((1,4),.12),
            {'rors':jnp.array([.1])},1,output_dir=str(tmp_path),checkpoint_prefix='tiny',
            sampler_backend='independent_nuts',mcmc_kwargs={'num_chains':4},
            spectro_min_depth_ess=400)
    audit = next((tmp_path/'chunks').glob('*chunk_0_1.diagnostics.json'))
    payload=json.loads(audit.read_text())
    assert [item['sampler'] for item in payload['sampler_gate_attempts']] == [
        'independent_nuts','independent_hmc','joint_nuts']
    assert not list((tmp_path/'chunks').glob('*chunk_0_1.pkl'))


@pytest.mark.parametrize('mode', ['uniform', 'gaussian'])
def test_coordinate_swap_reconstructs_latents_from_accepted_coefficients(mode):
    from types import SimpleNamespace
    from models.ld_parameterization import Power2MaxtedTransform, power2_smooth_transform
    # Draw and lane axes must be preserved, including all four chain streams.
    c=jnp.linspace(.02,.98,24).reshape(12,2)
    alpha=jnp.linspace(.02,.98,24)[::-1].reshape(12,2)
    physical=jnp.stack((c,alpha),axis=-1)
    fallback=PosteriorSamples({'c1':c,'c2':alpha}, num_chains=4,
                             sampled_sites=('c1','c2'))
    primary={name:jnp.full((12,3,2),999.) for name in (
        'ld_smooth','ld_decorrelated','ld_coefficients')}
    aligned=_align_power2_fallback_coordinates(primary,fallback,
                                             SimpleNamespace(_koala_ld_mode=mode))
    transform=power2_smooth_transform('uniform' if mode=='uniform' else 'gaussian')
    np.testing.assert_allclose(transform(aligned['ld_smooth']), physical, atol=1e-12)
    np.testing.assert_allclose(Power2MaxtedTransform().inv(aligned['ld_decorrelated']),
                               physical, atol=2e-14)
    np.testing.assert_array_equal(aligned['ld_coefficients'],physical)
    assert aligned.num_chains==4 and aligned.sampled_sites==('c1','c2')


def test_sampler_fallback_can_continue_before_another_fresh_fit(tmp_path, monkeypatch):
    import jax
    import koala.sampling as sampling
    def one_lane(seed):
        source=_posterior(seed)
        return PosteriorSamples({name:value[:, :1] for name,value in source.items()},
                                 num_chains=4, sampled_sites=('rors',))
    primary,first,extra=one_lane(11),one_lane(12),one_lane(14)
    runner=_Runner(first,extra)
    original_gate=sampling._spectro_failed_lanes
    def fake_sample(backend, cache, *args, width, varying, diagnostics_path, **kwargs):
        with open(diagnostics_path,'w') as stream:
            json.dump({'num_divergences':0},stream)
        if backend=='independent_hmc':
            cache[(backend,int(width),varying)]=runner
            return first
        assert backend=='independent_nuts'
        return primary
    def gate(samples,*args,**kwargs):
        if samples is primary or samples is first:
            return np.array([0]), {'bulk_ess_per_channel':[50.],
                                  'num_divergences_per_channel':[0]}
        return original_gate(samples,*args,**kwargs)
    monkeypatch.setattr(sampling,'_sample_independent_chunk',fake_sample)
    monkeypatch.setattr(sampling,'_spectro_failed_lanes',gate)
    result=sampling.get_samples_chunked(lambda *args:None,jax.random.PRNGKey(11),
        jnp.arange(4.),jnp.full((1,4),.1),jnp.full((1,4),.12),{'rors':jnp.array([.1])},1,
        output_dir=str(tmp_path),checkpoint_prefix='tiny',sampler_backend='independent_nuts',
        mcmc_kwargs={'num_chains':4},spectro_min_depth_ess=400)
    assert runner.calls==1
    assert result.sampler_used==['independent_hmc']
    assert result.num_chains==4 and result.sampled_sites==('rors',)
    expected=(np.arange(1,2000,2)[:,None]*4+np.arange(4)).reshape(-1)
    np.testing.assert_array_equal(result['rors'],jnp.concatenate((first['rors'],extra['rors']))[expected])
    path=next((tmp_path/'chunks').glob('*chunk_0_1.diagnostics.json'))
    attempts=json.loads(path.read_text())['sampler_gate_attempts']
    assert [item['sampler'] for item in attempts]==[
        'independent_nuts','independent_hmc','independent_hmc']
    assert attempts[-1]['sampling_extension_blocks']==1
    assert attempts[-1]['input_local_lanes']==[0]

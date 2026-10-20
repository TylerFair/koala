"""Compiled retry reuse must continue to sample each channel's own data."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


def test_reused_retry_program_keeps_new_observations_and_independent_chains(tmp_path,monkeypatch):
    from koala.marginalization import batched_fitting as fitting
    from models import independent_nuts
    jax.config.update('jax_enable_x64',True)
    built = []
    original_builder = independent_nuts.build_independent_nuts_runner
    def build(*args,**kwargs):
        runner = original_builder(*args,**kwargs)
        built.append(runner)
        return runner
    monkeypatch.setattr(independent_nuts,'build_independent_nuts_runner',build)
    original_extend = fitting._extend_to_quality
    def extend(runner,samples,raw,**kwargs):
        samples,raw,quality,blocks = original_extend(runner,samples,raw,**kwargs)
        if runner.options['mass_matrix'] == 'laplace':
            quality = dict(quality,passed=False,min_bulk_ess=0.,min_tail_ess=0.,
                           max_rhat=1.1,divergences=1,failed_lanes=[0])
        return samples,raw,quality,blocks
    monkeypatch.setattr(fitting,'_extend_to_quality',extend)
    def model(t,yerr,y=None):
        mu = numpyro.sample('mu',dist.Normal(0.,10.).expand((len(yerr),)))
        numpyro.sample('obs',dist.Normal(mu[:,None],yerr),obs=y)
    model._koala_fixed_initialization_supports = True
    time = np.linspace(0.,1.,16)
    errors = np.stack([np.full(16,.05),np.full(16,.07)])
    observed = np.stack([np.full(16,.25),np.full(16,-.65)])
    samples,report = fitting._sample(model,time,errors,observed,dict(mu=np.zeros(2)),{},
        dict(chains=4,draws=1000,warmup=300,seed=17,min_ess=400.,width=1,prime=False,progress=False),
        tmp_path/'posterior.npz','real-retry-data-gate')
    assert len(built) == 2
    assert built[1].program_build_count == 1
    assert report['quality']['passed']
    assert [item['attempts'][0]['runner_reused'] for item in report['retries']] == [False,True]
    # These are independent conjugate Gaussian targets with widely separated
    # means. A stale-data or stale-chain cache cannot satisfy both.
    expected = np.array([.25,-.65])/(1+np.array([.05,.07])**2/(16*100.))
    np.testing.assert_allclose(np.asarray(samples['mu']).mean(axis=0),expected,atol=.003,rtol=0)
    assert samples.group_by_chain()['mu'].shape == (4,1000,2)

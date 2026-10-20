"""Exact affine target and state-lifecycle checks for optional Laplace coordinates."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_ENABLE_X64','1')
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_value
from models.independent_nuts import (
    _repair_laplace_hessian, _laplace_whitening_factor, _affine_laplace_potential,
    build_independent_nuts_runner, get_samples_independent,
)
from models.independent_hmc import build_independent_hmc_runner, get_samples_independent_hmc


def test_raw_metric_retains_legacy_operations():
    h=jnp.array([[7.,.6],[.6,.2]])
    values,vectors=jnp.linalg.eigh(h)
    repaired=jnp.maximum(values,1e-8*jnp.maximum(jnp.max(jnp.abs(values)),1.))
    expected=(vectors/repaired[None,:])@vectors.T
    actual,_,_=_repair_laplace_hessian(h,1e-8)
    np.testing.assert_array_equal(actual,expected)


def test_correlation_repair_is_invariant_under_coordinate_units():
    h=jnp.array([[4.,.8],[.8,2.]])
    units=jnp.array([1e-5,1e4])
    base,_,_=_repair_laplace_hessian(h,1e-8,'correlation')
    rescaled,condition,_=_repair_laplace_hessian(h/units[:,None]/units[None,:],1e-8,'correlation')
    np.testing.assert_allclose(rescaled/units[:,None]/units[None,:],base,rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(base,np.linalg.inv(h),rtol=2e-12,atol=1e-14)
    factor=_laplace_whitening_factor(rescaled)
    np.testing.assert_allclose(factor@factor.T,rescaled,rtol=2e-12,atol=1e-18)
    assert np.isfinite(condition) and condition<10


def test_affine_density_keeps_support_and_affine_jacobians():
    def model():
        q=numpyro.sample('q',dist.Beta(2.,3.))
        s=numpyro.sample('s',dist.LogNormal(0.,.7))
        numpyro.factor('observed',-.5*((q+.2*s-.8)/.3)**2)
    info=initialize_model(jax.random.PRNGKey(3),model,init_strategy=init_to_value(values={'q':.4,'s':1.2}))
    center,unravel=ravel_pytree(info.param_info.z)
    factor=jnp.array([[.2,0.],[.1,.7]])
    white=jnp.array([.3,-.6]);theta=center+factor@white
    transformed=_affine_laplace_potential(info.potential_fn,unravel,center,factor)
    _,logdet=jnp.linalg.slogdet(factor)
    np.testing.assert_allclose(transformed(white),info.potential_fn(unravel(theta))-logdet,atol=1e-12)
    expected=factor.T@jax.grad(lambda x:info.potential_fn(unravel(x)))(theta)
    np.testing.assert_allclose(jax.grad(transformed)(white),expected,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(jax.jacfwd(transformed)(white),expected,atol=1e-10,rtol=1e-10)


def _normal_model(t,yerr,y=None,prior_loc=None,units=None):
    correlation=jnp.array([[1.,.45],[.45,1.]])
    scale=jnp.array([2.,1.]) * units
    covariance=correlation*scale[:,None]*scale[None,:]
    beta=numpyro.sample('beta',dist.MultivariateNormal(prior_loc,covariance_matrix=covariance))
    numpyro.deterministic('unscaled',beta/units)
    prediction=beta[:,0,None]/units[0]+beta[:,1,None]/units[1]*t
    numpyro.sample('obs',dist.Normal(prediction,yerr),obs=y)


def _problem():
    t=jnp.linspace(-.4,.4,11);units=jnp.array([1e-3,1e2]);loc=jnp.array([[.7,-.8],[-.9,.6]])*units
    y=jnp.array([[1.1,.2],[-1.2,-.3]])[:,0,None]+jnp.array([[1.1,.2],[-1.2,-.3]])[:,1,None]*t
    error=jnp.full(y.shape,.3)
    prior_cov=np.array([[4.,.9],[.9,1.]])
    design=np.stack((np.ones(len(t)),np.asarray(t)),axis=1)
    precision=np.linalg.inv(prior_cov)+design.T@design/.3**2
    covariance=np.linalg.inv(precision)
    mean=np.stack([covariance@(np.linalg.solve(prior_cov,np.asarray(row)/units)+design.T@obs/.3**2) for row,obs in zip(loc,np.asarray(y))])
    return t,units,loc,y,error,mean,np.sqrt(np.diag(covariance))


def _options(coordinates='whitened',samples=4000,fused=False):
    return dict(num_warmup=0,num_samples=samples,lane_width=3,
        channel_varying_kwargs=('prior_loc',),mass_matrix='laplace',laplace_warmup=150,
        laplace_target_accept=.95,laplace_start_at_map=True,
        nuts_kwargs=dict(laplace_coordinates=coordinates,laplace_metric_scaling='correlation',laplace_fuse_program=fused),
        mcmc_kwargs=dict(num_chains=4,progress_bar=False))


@pytest.mark.parametrize('algorithm',['nuts','hmc'])
@pytest.mark.parametrize('coordinates',['unconstrained','whitened'])
@pytest.mark.parametrize('lane_loop,batch',[('vmap',0),('map',0),('map',4),('map',5)])
def test_scaled_laplace_recovers_known_multivariate_posterior(algorithm,coordinates,lane_loop,batch):
    t,units,loc,y,error,mean,sd=_problem()
    build,sample=(build_independent_nuts_runner,get_samples_independent) if algorithm=='nuts' else (build_independent_hmc_runner,get_samples_independent_hmc)
    # Retain enough draws to assess 0.05-sigma means without a short-chain
    # Monte Carlo failure; this analytic problem samples in seconds.
    options=_options(coordinates)
    options['nuts_kwargs'].update(lane_loop=lane_loop,lane_batch_size=batch)
    if algorithm=='hmc':options['num_steps']=8
    runner=build(_normal_model,**options)
    samples,diag=sample(_normal_model,jax.random.PRNGKey(23),t,error,y,{'beta':loc},prior_loc=loc,units=units,_runner=runner,return_diagnostics=True,**options)
    draws=np.asarray(samples['unscaled'])
    np.testing.assert_allclose((draws.mean(axis=0)-mean)/sd,0.,atol=.05,rtol=0)
    np.testing.assert_allclose(draws.std(axis=0),np.broadcast_to(sd,(2,2)),atol=0,rtol=.05)
    assert int(np.asarray(diag.diverging).sum())==0
    assert runner.options['laplace_eigenvalue_floor']==1e-8
    if coordinates=='whitened':
        assert len(runner._continuation[1])==6
        np.testing.assert_array_equal(runner._continuation[0].adapt_state.inverse_mass_matrix,1.)


@pytest.mark.parametrize('fused',[False,True])
def test_whitened_continuation_and_new_target_reuse(fused):
    t,units,loc,y,error,_,_=_problem();options=_options(samples=200,fused=fused)
    runner=build_independent_nuts_runner(_normal_model,**options)
    def run(key,loc,y,runner,options,errors=error):
        return get_samples_independent(_normal_model,key,t,errors,y,{'beta':loc},prior_loc=loc,units=units,_runner=runner,**options)
    key=jax.random.PRNGKey(13);first=run(key,loc,y,runner,options);extra,_=runner.continue_raw()
    whole=run(key,loc,y,None,_options(samples=400,fused=fused))
    for name in first:
        np.testing.assert_allclose(np.concatenate((first[name],extra[name])),whole[name],atol=1e-8,rtol=1e-9)
    moved_loc=loc+jnp.array([.2,-.3])*units;moved_y=y+.7
    # Refresh the affine covariance as well as the center: changing only
    # a Gaussian target's location leaves its Hessian unchanged.
    moved_error=error*jnp.array([.7,1.4])[:,None]
    moved=run(jax.random.PRNGKey(14),moved_loc,moved_y,runner,options,moved_error)
    fresh=run(jax.random.PRNGKey(14),moved_loc,moved_y,None,options,moved_error)
    for name in moved:np.testing.assert_allclose(moved[name],fresh[name],atol=1e-8,rtol=1e-9)
    assert runner.program_build_count==1


@pytest.mark.parametrize('builder',[build_independent_nuts_runner,build_independent_hmc_runner])
def test_laplace_coordinate_options_are_explicit(builder):
    with pytest.raises(ValueError,match='coordinates'):
        builder(_normal_model,lane_width=2,nuts_kwargs={'laplace_coordinates':'invalid'})
    with pytest.raises(ValueError,match='mass_matrix'):
        builder(_normal_model,lane_width=2,nuts_kwargs={'laplace_coordinates':'whitened'})


@pytest.mark.parametrize('algorithm',['nuts','hmc'])
@pytest.mark.parametrize('coordinates',['unconstrained','whitened'])
def test_sequential_lane_execution_preserves_both_coordinate_systems(algorithm,coordinates):
    t,units,loc,y,error,_,_=_problem()
    build,sample=(build_independent_nuts_runner,get_samples_independent) if algorithm=='nuts' else (build_independent_hmc_runner,get_samples_independent_hmc)
    common_state=None;reference=None
    for mode,batch in (('vmap',0),('map',0),('map',4),('map',5)):
        options=_options(coordinates,samples=200)
        options['nuts_kwargs'].update(lane_loop=mode,lane_batch_size=batch)
        if algorithm=='hmc':options['num_steps']=8
        runner=build(_normal_model,**options)
        sample(_normal_model,jax.random.PRNGKey(36),t,error,y,{'beta':loc},prior_loc=loc,units=units,_runner=runner,**options)
        if common_state is None:
            common_state=runner._continuation
        else:
            # Independently compiled adaptive warmups need not choose the
            # same step size. Test the transition law from identical frozen
            # state; the analytic tests separately verify every warmup/mode.
            runner._continuation=common_state
        replay,raw=runner.continue_raw()
        if reference is None:reference=replay
        else:
            for key in reference:
                np.testing.assert_allclose(reference[key],replay[key],atol=1e-7,rtol=1e-6)
        assert raw['num_chains']==4


@pytest.mark.parametrize('algorithm',['nuts','hmc'])
def test_forward_mode_whitened_laplace_recovers_analytic_target(algorithm):
    t,units,loc,y,error,mean,sd=_problem()
    build,sample=(build_independent_nuts_runner,get_samples_independent) if algorithm=='nuts' else (build_independent_hmc_runner,get_samples_independent_hmc)
    options=_options(samples=4000)
    options['nuts_kwargs']['forward_mode_differentiation']=True
    if algorithm=='hmc':options['num_steps']=8
    runner=build(_normal_model,**options)
    samples,diag=sample(_normal_model,jax.random.PRNGKey(23),t,error,y,{'beta':loc},prior_loc=loc,units=units,_runner=runner,return_diagnostics=True,**options)
    draws=np.asarray(samples['unscaled'])
    np.testing.assert_allclose((draws.mean(axis=0)-mean)/sd,0.,atol=.05,rtol=0)
    np.testing.assert_allclose(draws.std(axis=0),np.broadcast_to(sd,(2,2)),atol=0,rtol=.05)
    assert int(np.asarray(diag.diverging).sum())==0


@pytest.mark.parametrize('builder',[build_independent_nuts_runner,build_independent_hmc_runner])
def test_lane_batch_size_requires_explicit_map_execution(builder):
    for options in ({'lane_batch_size':2},{'lane_loop':'map','lane_batch_size':-1}):
        with pytest.raises(ValueError,match='lane_batch_size'):
            builder(_normal_model,lane_width=2,nuts_kwargs=options)

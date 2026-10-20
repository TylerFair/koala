"""Exact marginal integral from OOT sufficient statistics, without binning."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density
from models.cadence_reduction import build_linear_oot_statistics
from models.jaxoplanet.builder import create_vectorized_model
from models.jaxoplanet.core import circular_orbit_separation

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('basis',['elapsed_days','batman_normalized'])
@pytest.mark.parametrize('scale',[.1,3.,10.,100.])
def test_integral_and_conditional_match_every_observation(basis,scale):
    t=np.linspace(-.15,.15,601);dt=t-t.min();X=np.column_stack([np.ones_like(t),dt,dt*dt])
    z=circular_orbit_separation(t,3.3,0.,13.6,.45)
    window=np.flatnonzero(z<1+np.sqrt(.5)+1e-6)
    error=np.array([.001,.002])[:,None]*np.array([1.,1.1,1.3])[np.arange(t.size)%3]
    reference=np.array([[1.001,.003,-.01],[.999,-.002,.007]])
    rng=np.random.default_rng(72);y=reference@X.T-.03*np.exp(-(t/.03)**8)+error*rng.normal(size=error.shape)
    mask=np.ones(y.shape,dtype=bool);mask[0,:5]=False;mask[0,-10:]=False;mask[1,:8]=False;mask[1,240:245]=False
    y=np.where(mask,y,1.);error=np.where(mask,error,1.)
    stats=build_linear_oot_statistics(t,y,error,window,X,reference,likelihood_mask=mask)
    common=dict(detrend_type='quadratic',ld_mode='uniform',ld_uniform_basis='kipping',
        trend_mode='gaussian_marginalized',trend_basis=basis,transit_window='auto',
        transit_window_indices=window,transit_grid='off')
    full=create_vectorized_model(**common,cadence_reduction='off')
    reduced=create_vectorized_model(**common,cadence_reduction='auto')
    kwargs=dict(y=y,PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]),
        mu_b=jnp.array([.45]),transit_separation=jnp.array(z)[None, :],likelihood_mask=mask,
        precomputed_yerr_per_lc=jnp.array([.001,.002]),trend_prior_mean=jnp.array([[1.,0.,0.],[1.,0.,0.]]),
        trend_prior_scale=jnp.full((2,3),scale),**stats)
    def evaluate(model,q):
        return log_density(model,(jnp.array(t),jnp.array(error)),kwargs,
            dict(rors=q[:2].reshape(2,1),log_jitter=q[2:4],ld_q=q[4:].reshape(2,2)))
    q=jnp.array([.175,.19,-7.,-6.5,.3,.3,.4,.2])
    actual,trace=evaluate(reduced,q);expected,original=evaluate(full,q)
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
    np.testing.assert_allclose(jax.grad(lambda q:evaluate(reduced,q)[0])(q),
        jax.grad(lambda q:evaluate(full,q)[0])(q),atol=1e-7,rtol=1e-6)
    mean=np.asarray(trace['trend_beta_mean']['value']);oldmean=np.asarray(original['trend_beta_mean']['value'])
    np.testing.assert_allclose(mean@X.T,oldmean@X.T,atol=1e-9,rtol=0)
    factor=np.asarray(trace['trend_beta_factor']['value']);oldfactor=np.asarray(original['trend_beta_factor']['value'])
    np.testing.assert_allclose(factor@factor.swapaxes(-1,-2),oldfactor@oldfactor.swapaxes(-1,-2),atol=1e-10,rtol=1e-7)

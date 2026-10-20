"""Mixed normalized polynomial/template Gaussian targets versus dense MVN."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from models.jaxoplanet.builder import create_vectorized_model
from models.jaxoplanet.core import compute_transit_model
from models.trend_marginal import marginalized_trend_coefficient_names
jax.config.update('jax_enable_x64',True)
FAMILIES=['spot_spectroscopic','2spot_spectroscopic','linear_discontinuity_spectroscopic',
 'spot_spectroscopic+linear_discontinuity_spectroscopic','spot_spectroscopic+explinear_spectroscopic',
 '2spot_spectroscopic+explinear_spectroscopic','explinear']


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('law',['quadratic','power2'])
def test_actual_normalized_mixed_target_and_all_gradients_match_dense_mvn(family,law):
    t=jnp.linspace(-.06,.06,17);dt=t-t.min();x=(t-t.mean())/jnp.ptp(t)
    spot=.01*jnp.exp(-((t-.01)/.02)**2);spot2=.02*jnp.exp(-((t+.02)/.015)**2)
    jump=.005*jax.nn.sigmoid((t-.01)/.003);ramp=jnp.exp(-dt/.03)
    components=family.split('+');names=['c','v']
    if any('explinear' in c for c in components):names.append('A')
    if 'spot_spectroscopic' in components or '2spot_spectroscopic' in components:names.append('A_spot')
    if '2spot_spectroscopic' in components:names.append('A_spot2')
    if 'linear_discontinuity_spectroscopic' in components:names.append('A_jump')
    assert marginalized_trend_coefficient_names(family)==tuple(names)
    mean=jnp.array([[1. if name=='c' or name.startswith('A_') else .002 for name in names]])
    scale=jnp.array([[.2+.05*i for i in range(len(names))]])
    errors=jnp.full((1,17),.01);y=jnp.ones((1,17))+.03*ramp
    ld=jnp.array([.4,.2] if law=='quadratic' else [.6,.4])
    model=create_vectorized_model(param_method='a_rs',detrend_type=family,ld_profile=law,ld_mode='fixed',
        trend_mode='gaussian_marginalized',trend_basis='batman_normalized')
    kwargs=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),mu_a_rs=jnp.array([13.6]),
        mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ld_fixed=ld[None,:],trend_prior_mean=mean,trend_prior_scale=scale,
        spot_trend=spot,spot_trend2=spot2,jump_trend=jump,exp_trend=ramp,fixed_tau=.03,y=y)
    def actual(p):
        values=dict(rors=p[:1,None],log_jitter=p[1:2],log_tau=p[2:3])
        trace=handlers.trace(handlers.seed(handlers.substitute(model,data=values),709)).get_trace(t,errors,**kwargs)
        assert trace['trend_beta_mean']['value'].shape==(1,len(names))
        return trace['obs_marginalized']['fn'].log_factor
    def reference(p):
        params=dict(period=jnp.array([3.3]),t0=jnp.array([0.]),a_rs=jnp.array([13.6]),
            b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ecc=jnp.array([0.]),omega=jnp.array([0.]),rors=p[:1],u=ld,
            _ld_profile=law,_jaxoplanet_kernel='quadratic_contact_safe' if law=='quadratic' else 'auto',
            _quadratic_radius_bound=.5**.5)
        if law=='power2':params.update(c1=ld[0],c2=ld[1],u=jnp.zeros(12),_power2_exact=True)
        signal=1+compute_transit_model(params,t)
        external=dict(A=(jnp.exp(-dt/jnp.exp(p[2])) if family=='explinear' else ramp),
                      A_spot=spot,A_spot2=spot2,A_jump=jump)
        design=jnp.stack([jnp.ones_like(t),x]+[external[name] for name in names[2:]],axis=-1)*signal[:,None]
        weighted=design*scale[0]
        covariance=jnp.diag(errors[0]**2+jnp.exp(2*p[1]))+weighted@weighted.T
        return jax.scipy.stats.multivariate_normal.logpdf(y[0],design@mean[0],covariance)
    point=jnp.array([.18,-7.,np.log(.03)])
    a,ga=jax.jit(jax.value_and_grad(actual))(point)
    b,gb=jax.jit(jax.value_and_grad(reference))(point)
    np.testing.assert_allclose(a,b,atol=1e-7,rtol=0.)
    np.testing.assert_allclose(ga,gb,atol=1e-7,rtol=1e-6)


def test_white_higher_order_prior_overrides_and_starts():
    from koala.config import parse_trend_priors
    from koala.white_light import _apply_trend_prior_starts
    from models.model_building import sample_whitelight_trend
    specs=parse_trend_priors({f'v{k}':dict(prior='gaussian',value=.02*k,sigma=10.) for k in [2,3,4]})
    initial={f'v{k}':0. for k in [2,3,4]};_apply_trend_prior_starts(initial,specs)
    def prior():sample_whitelight_trend(jnp.linspace(0.,.2,13),{}, {'quartic'},'physical','fixed',trend_priors=specs)
    trace=handlers.trace(handlers.seed(prior,710)).get_trace()
    for k in [2,3,4]:
        fn=trace[f'v{k}']['fn']
        assert fn.loc==.02*k and fn.scale==10.
        assert initial[f'v{k}']==.02*k

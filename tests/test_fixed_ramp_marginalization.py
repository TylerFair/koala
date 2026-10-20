"""Fixed-tau Gaussian integrals retain the ramp amplitude and its density."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from models.trend_marginal import marginalized_trend_coefficient_names,build_marginalized_trend_design
from models.jaxoplanet.builder import create_vectorized_model
from models.jaxoplanet.core import compute_transit_model

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('trend,expected',[
    ('explinear_spectroscopic',('c','v','A')),
    ('spot_spectroscopic+explinear_spectroscopic',('c','v','A','A_spot')),
    ('2spot_spectroscopic+explinear_spectroscopic',('c','v','A','A_spot','A_spot2'))])
def test_fixed_ramp_amplitude_is_a_real_design_column(trend,expected):
    t=jnp.linspace(0.,.2,17);exp=jnp.exp(-t/.03)
    design,names=build_marginalized_trend_design(trend,t,2,exp_trend=exp,
                                              spot_trend=jnp.sin(t),spot_trend2=jnp.cos(t))
    assert names==expected
    np.testing.assert_array_equal(design[:,:,names.index('A')],jnp.broadcast_to(exp,(2,len(t))))


@pytest.mark.parametrize('spots',[0,1,2])
def test_actual_fixed_ramp_marginal_likelihood_and_radius_gradient_match_dense_normal(spots):
    t=jnp.linspace(-.06,.06,17);exp=jnp.exp(-(t-t.min())/.03)
    errors=jnp.full((1,len(t)),.01);data=jnp.ones_like(errors)+.03*exp[None,:]
    mean=jnp.array([[1.,0.,0.]+[1.]*spots]);scale=jnp.array([[.2,.5,.3]+[.4]*spots])
    spot=jnp.exp(-((t-.01)/.02)**2)*.01
    spot2=jnp.exp(-((t+.02)/.015)**2)*.02
    trend=('spot_spectroscopic+' if spots==1 else '2spot_spectroscopic+' if spots==2 else '')+'explinear_spectroscopic'
    model=create_vectorized_model(param_method='a_rs',detrend_type=trend,trend_mode='gaussian_marginalized',
                                  ld_profile='quadratic',ld_mode='fixed')
    kwargs=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),mu_a_rs=jnp.array([13.6]),
                mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ld_fixed=jnp.array([[.4,.2]]),
                exp_trend=exp,fixed_tau=.03,trend_prior_mean=mean,trend_prior_scale=scale,y=data,
                spot_trend=spot,spot_trend2=spot2)
    def actual(radius):
        point=dict(rors=jnp.reshape(radius,(1,1)),log_jitter=jnp.array([-7.]))
        tr=handlers.trace(handlers.seed(handlers.substitute(model,data=point),0)).get_trace(t,errors,**kwargs)
        assert tr['trend_beta_mean']['value'].shape==(1,3+spots)
        assert 'A' not in tr  # integrated, never omitted from the Gaussian target
        return tr['obs_marginalized']['fn'].log_factor
    def expected(radius):
        p=dict(period=jnp.array([3.3]),t0=jnp.array([0.]),a_rs=jnp.array([13.6]),
               b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),ecc=jnp.array([0.]),omega=jnp.array([0.]),
               rors=jnp.atleast_1d(radius),u=jnp.array([.4,.2]),_ld_profile='quadratic',
               _jaxoplanet_kernel='quadratic_contact_safe',_quadratic_radius_bound=.5**.5)
        signal=1+compute_transit_model(p,t)
        design=jnp.stack([jnp.ones_like(t),t-t.min(),exp]+[spot,spot2][:spots],axis=-1)*signal[:,None]
        weighted=design*scale[0]
        covariance=jnp.diag(errors[0]**2+jnp.exp(-14.))+weighted@weighted.T
        return jax.scipy.stats.multivariate_normal.logpdf(data[0],design@mean[0],covariance)
    va,ga=jax.jit(jax.value_and_grad(actual))(.18)
    vb,gb=jax.jit(jax.value_and_grad(expected))(.18)
    np.testing.assert_allclose(va,vb,atol=1e-7,rtol=0)
    np.testing.assert_allclose(ga,gb,atol=1e-7,rtol=1e-6)

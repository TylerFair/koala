"""Gaussian stellar centres outside support must not be used as latent starts."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model
import pytest
from koala.config import ParameterSpec
from koala.limb_darkening import _repair_gaussian_ld_start
import models.jaxoplanet.builder as jp
import models.harmonica.builder as hm

jax.config.update('jax_enable_x64', True)

@pytest.mark.parametrize('engine,stage,law', [
    ('jp','white','quadratic'), ('jp','spectro','quadratic'),
    ('jp','white','power2'), ('jp','spectro','power2'),
    ('hm','white','power2'), ('hm','spectro','power2'),
])
def test_real_builder_initializes_outside_stellar_centre_without_changing_prior(engine,stage,law):
    module=jp if engine=='jp' else hm
    centre=jnp.array([.6,-.07] if law=='quadratic' else [1.1,.4])
    coefficients=centre if stage=='white' else jnp.broadcast_to(centre,(2,2))
    initial=({'u':coefficients} if law=='quadratic' else
             {'c1':coefficients[...,0], 'c2':coefficients[...,1]})
    repaired=_repair_gaussian_ld_start(initial,ld_profile=law,ld_prior_mode='gaussian')
    t=jnp.linspace(-.07,.07,13)
    opts=dict(detrend_type='none',ld_mode='gaussian',ld_profile=law,gaussian_ld_width=.1)
    if stage=='white':
        values=dict(period=3.3,t0=0.,a_rs=13.6,b=.4,ecc=0.,omega=90.,rprs=.18)
        specs={name:(ParameterSpec(name,'fixed',value),) for name,value in values.items()}
        model=module.create_whitelight_model(parameter_priors=specs,param_method='a_rs',**opts)
        error=jnp.full(t.size,.001);kw=dict(prior_params={'period':jnp.array([3.3]),'u':centre},y=jnp.ones(t.size))
    else:
        model=module.create_vectorized_model(**opts,**({'param_method':'a_rs'} if engine=='jp' else {}))
        error=jnp.full((2,t.size),.001)
        kw=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),mu_a_rs=jnp.array([13.6]),
                mu_b=jnp.array([.4]),mu_u_ld=coefficients,y=jnp.ones((2,t.size)))
        if engine=='hm':kw['harmonica_a_rs']=jnp.array([13.6])
    info=initialize_model(jax.random.PRNGKey(16),model,init_strategy=init_to_value(values=repaired),model_args=(t,error),model_kwargs=kw)
    assert np.isfinite(info.param_info.potential_energy)
    trace=handlers.trace(handlers.seed(handlers.substitute(model,data=repaired),16)).get_trace(t,error,**kw)
    if law=='quadratic':
        np.testing.assert_array_equal(trace['u']['fn'].base_dist.base_dist.loc,coefficients)
        expected=dist.TruncatedNormal(coefficients,.1,low=0.,high=1.).log_prob(repaired['u']).sum(-1)
        np.testing.assert_allclose(trace['u']['fn'].log_prob(repaired['u']),expected,atol=1e-12,rtol=0)
    else:
        for index,name in enumerate(['c1','c2']):
            expected=dist.TruncatedNormal(coefficients[...,index],.1,low=0. if index==0 else .001,high=1.).log_prob(repaired[name])
            np.testing.assert_allclose(trace[name]['fn'].log_prob(repaired[name]),expected,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(initial['u'] if law=='quadratic' else initial['c1'], coefficients if law=='quadratic' else coefficients[...,0])


def test_white_laplace_handoff_repairs_optimizer_boundary_without_moving_prior(monkeypatch):
    from koala.white_light import _sample_white_light
    import models.independent_nuts as engine
    centre=jnp.array([.6,-.07]);t=jnp.linspace(-1.,1.,9);error=jnp.full(t.shape,.01)
    def model(time,uncertainty,y=None,prior_params=None):
        u=numpyro.sample('u',dist.TruncatedNormal(prior_params['u'],.1,low=0.,high=1.).to_event(1))
        numpyro.sample('obs',dist.Normal(u.sum(),uncertainty),obs=y)
    class Prepared(Exception):pass
    def prepare(model,key,initial,*args,**kwargs):
        assert np.all(np.asarray(initial['u'])>0) and np.all(np.asarray(initial['u'])<1)
        np.testing.assert_array_equal(kwargs['model_kwargs']['prior_params']['u'],centre)
        assert kwargs['eigenvalue_floor']==1e-12
        info=initialize_model(key,model,init_strategy=init_to_value(values=initial),model_args=args,model_kwargs=kwargs['model_kwargs'])
        assert np.isfinite(info.param_info.potential_energy)
        raise Prepared()
    monkeypatch.setattr(engine,'prepare_laplace_metric',prepare)
    context=SimpleNamespace(cfg={'flags':{}},transit_engine='jaxoplanet',NUTS_KWARGS={},key_master=jax.random.PRNGKey(16),
        ld_profile='quadratic',ld_prior_mode='gaussian',ld_uniform_basis='uplus_uminus',
        whitelight_laplace_options=dict(mass_matrix='laplace',hessian_method='finite_difference',trust_radius=5.),
        data=SimpleNamespace(wl_time=t,wl_flux_err=error,wl_flux=jnp.ones_like(t)),uses_surface_model=False)
    with pytest.raises(Prepared):
        _sample_white_light(context=context,hyper_params_wl={'u':centre},science_sites=('u',),soln={'u':jnp.array([1.,0.])},whitelight_mcmc_kwargs={},whitelight_model_for_run=model)

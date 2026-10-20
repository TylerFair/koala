"""Direct wide quadratic priors and stellar-fixed u1 retain exact fluxes."""
import importlib
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro import handlers
from models.limb_darkening_config import validate_uniform_ld_bounds, validate_ld_prior
from koala.artifacts import _science_artifact_fingerprint
from koala.white_light import _white_light_fingerprint_config
from koala.sampling import _chunk_checkpoint_fingerprint
from koala.spectroscopy import _build_spectroscopic_model
from koala.config import parse_planet_parameter_specs


def trace(engine, stage, mode, bounds=(-3., 3.)):
    module = importlib.import_module('models.' + engine)
    options = dict(ld_profile='quadratic', ld_mode=mode, detrend_type='none',
                   ld_uniform_basis='coefficients', ld_uniform_coefficient_bounds=bounds)
    if engine == 'harmonica':
        options['max_harmonic_order'] = 0
    t = jnp.linspace(-.08, .08, 31)
    if stage == 'white':
        specs = parse_planet_parameter_specs({name:dict(value=value, prior='fixed')
            for name,value in dict(period=3.,t0=0.,a_rs=9.,b=.3,rprs=.1,ecc=0.,omega=90.).items()})
        model = module.create_whitelight_model(param_method='a_rs', parameter_priors=specs, **options)
        u = jnp.array([.4,-.1]); values = dict(u=u,u2=u[1],log_jitter=-10.)
        args = dict(prior_params=dict(period=jnp.array([3.]),ecc=jnp.array([0.]),
                                    omega=jnp.array([90.]),u=u))
        y = jnp.ones(t.size)
    else:
        model = module.create_vectorized_model(**options)
        u = jnp.array([[.4,-.1],[.2,.15]])
        values = dict(u=u,u2=u[:,1],rors=jnp.full((2,1),.1),log_jitter=jnp.full(2,-10.))
        args = dict(PERIOD=jnp.array([3.]),mu_t0=jnp.array([0.]),mu_b=jnp.array([.3]),
                    mu_duration=jnp.array([.1]),ld_fixed=u,mu_u_ld=u,
                    rors_reference=jnp.array([.1]))
        if engine == 'harmonica':
            args.update(mu_cos_i=jnp.array([.3/9]),harmonica_a_rs=jnp.array([9.]))
        else:
            # Exact physical orbit; duration is not an independently varying chord.
            args.update(mu_a_rs=jnp.array([9.]))
            model = module.create_vectorized_model(param_method='a_rs', **options)
        y = jnp.ones((2,t.size))
    tr = handlers.trace(handlers.substitute(handlers.seed(model, 71), data=values)).get_trace(
        t,jnp.full_like(y,.001),y=y,**args)
    return tr,u


@pytest.mark.parametrize('engine', ['jaxoplanet','harmonica'])
@pytest.mark.parametrize('stage', ['white','spectro'])
@pytest.mark.parametrize('mode', ['uniform','fixed_u1'])
def test_actual_quadratic_prior_support_density_and_flux(engine,stage,mode):
    tr,u=trace(engine,stage,mode)
    site='u' if mode=='uniform' else 'u2'
    fn=tr[site]['fn']; value=u if mode=='uniform' else u[...,1]
    np.testing.assert_allclose(fn.log_prob(value), -(2 if mode=='uniform' else 1)*np.log(6.), atol=1e-14)
    assert np.all(np.asarray(fn.support.check(value)))
    assert not np.any(np.asarray(fn.support.check(jnp.full_like(value,3.1))))
    assert not np.any(np.asarray(fn.support.check(jnp.full_like(value,-3.1))))
    if mode=='fixed_u1':
        assert tr['u1']['type']=='deterministic'
        np.testing.assert_array_equal(tr['u1']['value'],u[...,0])
        np.testing.assert_array_equal(tr['u']['value'],u)
    fixed,_=trace(engine,stage,'fixed')
    np.testing.assert_allclose(tr['obs']['fn'].loc,fixed['obs']['fn'].loc,atol=1e-12,rtol=0)


@pytest.mark.parametrize('value', [None,1.,'ab',[0.],[-1.,0.,1.],[1.,1.],[2.,-2.],
                                  [float('nan'),1.],[-1.,float('inf')],[True,2.]])
def test_invalid_bound_options(value):
    with pytest.raises(ValueError,match='two finite numbers'):
        validate_uniform_ld_bounds(value)


@pytest.mark.parametrize('engine',['jaxoplanet','harmonica'])
@pytest.mark.parametrize('stage',['white','spectro'])
def test_nonquadratic_fixed_u1_rejected_by_builder(engine,stage):
    mod=importlib.import_module('models.'+engine)
    factory=mod.create_whitelight_model if stage=='white' else mod.create_vectorized_model
    with pytest.raises(ValueError,match='quadratic'):
        factory(ld_profile='power2',ld_mode='fixed_u1')


def test_public_fixed_u1_validation_and_bounds():
    assert validate_ld_prior('fixed_u1',ld_profile='quadratic')=='fixed_u1'
    for law in ('power2','kipping'):
        with pytest.raises(ValueError,match='quadratic'):
            validate_ld_prior('fixed_u1',ld_profile=law)
    assert validate_uniform_ld_bounds([-2,2])==(-2.,2.)


def test_bounds_and_mode_change_scientific_fingerprints():
    def white(bounds,prior):
        cfg=dict(flags=dict(ld_profile='quadratic',ld_uniform_basis='coefficients',
            ld_prior=prior,ld_uniform_coefficient_bounds=bounds))
        return _science_artifact_fingerprint('white',_white_light_fingerprint_config(cfg))
    assert len({white([-2,2],'uniform'),white([-3,3],'uniform'),white([-3,3],'fixed_u1')})==3
    from models.jaxoplanet import create_vectorized_model
    def spectro(bounds,prior):
        model=_build_spectroscopic_model(create_vectorized_model,ld_mode=prior,
            ld_profile='quadratic',ld_uniform_basis='coefficients',ld_uniform_coefficient_bounds=bounds)
        return _chunk_checkpoint_fingerprint(model,jax.random.PRNGKey(1),jnp.arange(5.),
            jnp.ones((1,5)),jnp.ones((1,5)),{},chunk_size=1,sampler_backend='independent_nuts',
            nuts_kwargs={},mcmc_kwargs={},channel_varying_kwargs=(),checkpoint_signature={},model_kwargs={})
    assert len({spectro((-2,2),'uniform'),spectro((-3,3),'uniform'),spectro((-3,3),'fixed_u1')})==3


def test_fixed_u1_does_not_constrain_model_u1_to_u2_box():
    from models.model_building import sample_quadratic_fixed_u1
    model=lambda: sample_quadratic_fixed_u1(jnp.array([5.,.1]),(),-2.,2.)
    tr=handlers.trace(handlers.substitute(handlers.seed(model,1),data={'u2':-.1})).get_trace()
    np.testing.assert_array_equal(tr['u']['value'],[5.,-.1])
    assert np.isfinite(tr['u2']['fn'].log_prob(-.1))

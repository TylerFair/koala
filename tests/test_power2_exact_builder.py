"""Ensure every physical power-2 prior reaches the exact likelihood."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from models.jaxoplanet.builder import create_vectorized_model, create_whitelight_model
from models.jaxoplanet.core import compute_transit_model, build_transit_phase_offsets
from models.harmonica.core import harmonica_a_rs_from_duration
from koala.config import parse_planet_parameter_specs
from koala.surface import _attach_jaxoplanet_eval_metadata
from models.jaxoplanet.power2_exact import light_curve

jax.config.update('jax_enable_x64',True)

@pytest.mark.parametrize('mode',['fixed','gaussian','uniform','stellarprior'])
def test_spectroscopic_prior_routes_physical_coefficients(mode):
    model=create_vectorized_model(ld_mode=mode,ld_profile='power2',power2_evaluation='exact')
    t=jnp.linspace(-.08,.08,43); err=jnp.full((2,43),1e-3)
    tr=handlers.trace(handlers.seed(model,0)).get_trace(t,err,y=jnp.ones_like(err),
        rors_reference=jnp.array([.1]), mu_duration=jnp.array([.12]),mu_t0=jnp.array([0.]),mu_b=jnp.array([.45]),
        PERIOD=jnp.array([3.]),ld_fixed=jnp.array([[.6,.6],[.9,.1]]),
        mu_u_ld=jnp.array([[.6,.6],[.9,.1]]),sigma_u_ld=jnp.full((2,2),.1))
    r=tr['rors']['value'][:,0]; c=tr['c1']['value']; a=tr['c2']['value']
    phase,mask=build_transit_phase_offsets(t,jnp.array([3.]),jnp.array([0.]),jnp.array([.12]))
    axis=harmonica_a_rs_from_duration(3.,.12,.45,.1)
    angle=2*jnp.pi*t/3.
    z=jnp.sqrt(axis**2*jnp.sin(angle)**2+.45**2*jnp.cos(angle)**2)
    transit=light_curve(c[:,None],a[:,None],z[None,:],r[:,None])
    # Trend convention is anchored at the earliest cadence.
    expected=(1+transit)*(tr['c']['value'][:,None]+tr['v']['value'][:,None]*(t-jnp.min(t)))
    np.testing.assert_allclose(tr['obs']['fn'].loc,expected,atol=2e-12,rtol=0)

@pytest.mark.parametrize('mode',['fixed','gaussian','uniform','stellarprior'])
def test_whitelight_prior_routes_physical_coefficients(mode):
    specs=parse_planet_parameter_specs({k:dict(value=v,prior='fixed') for k,v in
        dict(period=3.,duration=.12,t0=0.,b=.45,rprs=.146).items()})
    model=create_whitelight_model(ld_mode=mode,ld_profile='power2',power2_evaluation='exact',
                                  parameter_priors=specs)
    t=jnp.linspace(-.08,.08,43)
    tr=handlers.trace(handlers.seed(model,0)).get_trace(t,jnp.full(t.shape,1e-3),y=jnp.ones_like(t),
        prior_params=dict(u=jnp.array([.6,.6]),u_sigma=jnp.array([.1,.1]),c=1.,v=0.))
    params=dict(period=jnp.array([3.]),duration=jnp.array([.12]),t0=jnp.array([0.]),
        b=jnp.array([.45]),rors=jnp.array([.146]),u=jnp.zeros(12),
        c1=tr['c1']['value'],c2=tr['c2']['value'],_power2_exact=True)
    expected=(1+compute_transit_model(params,t))*(tr['c']['value']+tr['v']['value']*(t-jnp.min(t)))
    np.testing.assert_allclose(tr['obs']['fn'].loc,expected,atol=2e-12,rtol=0)

def test_postfit_keeps_exact_law_and_stellar_units():
    params=dict(period=jnp.array([3.]),duration=jnp.array([.12]),t0=jnp.array([0.]),
        b=jnp.array([.45]),rors=jnp.array([.146]),u=jnp.full(12,123.),c1=.9,c2=.1)
    t=jnp.linspace(-.08,.08,41)
    attached=_attach_jaxoplanet_eval_metadata(params,t,jaxoplanet_kernel='auto',ld_profile='power2')
    assert attached['_power2_exact']
    r=.146; axis=harmonica_a_rs_from_duration(3.,.12,.45,r)
    angle=2*jnp.pi*t/3.
    z=jnp.sqrt(axis**2*jnp.sin(angle)**2+.45**2*jnp.cos(angle)**2)
    expected=light_curve(.9,.1,z,r)
    np.testing.assert_allclose(compute_transit_model(attached,t),expected,atol=2e-12,rtol=0)


@pytest.mark.parametrize('impact',[0.,1.4])
def test_duration_gradients_central_and_nontransiting(impact):
    params=dict(period=jnp.array([3.]),duration=jnp.array([.12]),t0=jnp.array([0.]),
        b=jnp.array([impact]),u=jnp.zeros(12),c1=.6,c2=.4,_power2_exact=True)
    t=jnp.array([-.01,0.,.01])
    def evaluate(r):
        return compute_transit_model(dict(params,rors=jnp.array([r])),t)
    values,gradient=jax.jit(lambda r:(evaluate(r),jax.jacfwd(evaluate)(r)))(.146)
    assert np.all(np.isfinite(gradient))
    if impact>1.146:
        np.testing.assert_array_equal(values,np.zeros(3))
        np.testing.assert_array_equal(gradient,np.zeros(3))


def test_keplerian_uniform_and_linear_law_unit_parity(monkeypatch):
    import jaxoplanet.light_curves as curves
    reference=curves.limb_dark_light_curve
    monkeypatch.setattr(curves,'limb_dark_light_curve',lambda orbit,*u: reference(orbit,*u,order=40))
    params=dict(period=jnp.array([3.]),a_rs=jnp.array([10.]),t0=jnp.array([0.]),
        b=jnp.array([.45]),rors=jnp.array([.146]),ecc=jnp.array([.1]),omega=jnp.array([.3]))
    t=jnp.linspace(-.08,.08,101)
    for c in [0.,.6,1.]:
        expected=compute_transit_model(dict(params,u=jnp.array([c])),t)
        actual=compute_transit_model(dict(params,u=jnp.zeros(12),c1=c,c2=1.,_power2_exact=True),t)
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)


@pytest.mark.parametrize('failure', ['validated_grid_size', 'prepare_table'])
def test_duration_interpolation_is_never_used_for_physical_orbits(monkeypatch, failure):
    from types import SimpleNamespace
    import koala.spectroscopy as spectroscopy
    import models.jaxoplanet.power2_table as tables
    monkeypatch.setattr(tables, 'validated_grid_size', lambda **kwargs:2049)
    def reject(**kwargs):
        raise AssertionError('duration interpolation must not be prepared for a Keplerian orbit')
    monkeypatch.setattr(tables, failure, reject)
    monkeypatch.setattr(spectroscopy, '_build_spectroscopic_model',
                        lambda factory, **kwargs:kwargs)
    geometry=SimpleNamespace(param_method='duration', n_planets=1,
                             B_BASE=np.array([.45]), RORS_BASE=np.array([.146]),
                             PERIOD_FIXED=np.array([3.3]), T0_BASE=np.array([0.]),
                             A_RS_BASE=np.array([13.6]), HARMONICA_ECC=np.array([0.]),
                             HARMONICA_OMEGA=np.array([np.pi/2]))
    model=SimpleNamespace(transit_engine='jaxoplanet', transit_window_optimization='off',
        ld_profile='power2', spectro_ld_parameterization='coefficients',
        surface_config={'model':'transit','spots':()},
        create_vectorized_model=None, _engine_spectro_kw=dict(
            power2_evaluation='exact', transit_grid='auto'))
    initial=dict(rors=jnp.array([[.146]]))
    fit=spectroscopy._spectroscopic_model(geometry,model,None,jnp.linspace(-.1,.1,5001),
        jnp.array([[.6,.6]]),initial,None,'uniform','free','linear','test')
    assert fit.builder_kwargs['power2_evaluation']=='exact'
    assert fit.builder_kwargs['ld_mode']=='uniform'
    assert fit.builder_kwargs['transit_grid']=='off'
    assert 'power2_table' not in fit.builder_kwargs
    assert fit.init_params is initial
    assert fit.transit_separation.shape == (1,5001)

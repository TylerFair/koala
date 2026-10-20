"""Quadratic two-template spectra preserve observer-time, multiplicative semantics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from koala.config import _spectro_detrend_type
from koala.plotting import _summary_channel_trend
from koala.outputs import _trend_from_params_np
from models import trends
from models.detrend import COMPUTE_KERNELS, COMPOSITE_KERNELS, resolve_detrend_kernel
from models.cadence_reduction import linear_spectro_trend_coefficient_names
from models.trend_marginal import build_marginalized_trend_design
from models.jaxoplanet.builder import create_vectorized_model
from models.jaxoplanet.core import compute_transit_model

NAME = 'quadratic+2spot_spectroscopic'
NAMES = ('c', 'v', 'v2', 'A_spot', 'A_spot2')


def fixture():
    t = jnp.linspace(-.09, .09, 31)
    # Templates stay in observer time, with opposite signed amplitudes.
    s1 = .002*jnp.exp(-.5*((t+.017)/.012)**2)
    s2 = .003*jnp.exp(-.5*((t-.024)/.009)**2)
    point = dict(rors=jnp.array([[.12]]), dt0=jnp.array([[20./86400]]),
                 log_jitter=jnp.array([-10.]), c=jnp.array([1.001]),
                 v=jnp.array([-.003]), v2=jnp.array([.02]),
                 A_spot=jnp.array([.7]), A_spot2=jnp.array([-.6]))
    kwargs = dict(PERIOD=jnp.array([4.]), mu_t0=jnp.array([0.]),
                  mu_a_rs=jnp.array([9.]), mu_b=jnp.array([.35]),
                  ld_fixed=jnp.array([[.3,.2]]), spot_trend=s1, spot_trend2=s2)
    return t,s1,s2,point,kwargs


@pytest.mark.parametrize('name', [NAME, '2spot_spectroscopic+quadratic'])
def test_registered_kernel_numpy_flux_multiplicative_and_reconstruction(monkeypatch,name):
    t,s1,s2,point,_ = fixture()
    p = {key:np.asarray(value).ravel()[0] for key,value in point.items()}
    signal = np.where(np.abs(t)<.04,-.02,0.)
    monkeypatch.setattr(trends,'compute_transit_model_auto',lambda params,time: jnp.asarray(signal))
    kernel = resolve_detrend_kernel(_spectro_detrend_type(name))
    assert COMPUTE_KERNELS[NAME] is COMPOSITE_KERNELS[frozenset(name.split('+'))]
    dt=np.asarray(t)-np.min(t)
    baseline=p['c']+p['v']*dt+p['v2']*dt**2+p['A_spot']*s1+p['A_spot2']*s2
    flux=np.asarray(kernel(p,t,s1,s2))
    np.testing.assert_allclose(flux,(1+signal)*baseline,atol=1e-12,rtol=0)
    monkeypatch.setattr(trends,'compute_transit_model_auto',lambda params,time: jnp.zeros_like(time))
    np.testing.assert_allclose(flux,(1+signal)*kernel(p,t,s1,s2),atol=1e-12,rtol=0)
    for evaluate in (_summary_channel_trend,_trend_from_params_np):
        if evaluate is _summary_channel_trend:
            actual=evaluate(t,point,0,name,spot_trend=s1,spot_trend2=s2)
        else:
            actual=evaluate(name,np.asarray(t),point,idx=0,spot_trend=s1,spot_trend2=s2)
        np.testing.assert_allclose(actual,baseline,atol=1e-12,rtol=0)
    assert linear_spectro_trend_coefficient_names(name)==NAMES
    design,names=build_marginalized_trend_design(name,t,1,spot_trend=s1,spot_trend2=s2)
    assert names==NAMES
    np.testing.assert_allclose(design[0]@np.array([p[n] for n in names]),baseline,atol=1e-12,rtol=0)


def make_model():
    return create_vectorized_model(detrend_type=NAME,param_method='a_rs',ld_mode='fixed',
        jaxoplanet_kernel='quadratic_contact_safe',surface_config={'model':'transit','spectro_free_t0':True},
        trend_bounds={'A_spot':(-2.,2.),'A_spot2':(-2.,2.)})


def test_free_epoch_moves_only_transit_with_signed_template_sites():
    t,s1,s2,point,kwargs=fixture();model=make_model()
    yerr=jnp.full((1,t.size),.001)
    def trace(p):
        return handlers.trace(handlers.substitute(handlers.seed(model,1),data=p)).get_trace(t,yerr,**kwargs)
    result=trace(point)
    dt=np.asarray(t)-np.min(t)
    baseline=point['c'][0]+point['v'][0]*dt+point['v2'][0]*dt**2+point['A_spot'][0]*s1+point['A_spot2'][0]*s2
    for offset in (20.,-30.):
        p=dict(point,dt0=jnp.array([[offset/86400]]));tr=trace(p)
        params=dict(period=jnp.array([4.]),t0=jnp.array([offset/86400]),a_rs=jnp.array([9.]),
                    b=jnp.array([.35]),rors=jnp.array([.12]),u=jnp.array([.3,.2]),
                    _jaxoplanet_kernel='quadratic_contact_safe')
        expected=(1+compute_transit_model(params,t))*baseline
        np.testing.assert_allclose(tr['obs']['fn'].loc[0],expected,atol=1e-12,rtol=0)
    for name in NAMES:
        assert name in result and result[name]['type']=='sample'
    assert bool(result['A_spot2']['fn'].support.check(jnp.array([-.6]))[0])
    assert 'dt0' in model.surface_science_sites


def test_free_epoch_two_spot_normal_laplace_sampler_smoke():
    """Finite posterior draws through the normal sampler; no convergence claim."""
    from models.independent_nuts import get_samples_independent
    t,s1,s2,point,kwargs=fixture();model=make_model()
    nuisance={k:v for k,v in point.items() if k not in {'rors','dt0'}}
    conditional=handlers.substitute(model,data=nuisance)
    tr=handlers.trace(handlers.substitute(handlers.seed(model,1),data=point)).get_trace(
        t,jnp.full((1,t.size),.001),**kwargs)
    y=tr['obs']['fn'].loc
    draws=get_samples_independent(conditional,jax.random.PRNGKey(181),t,jnp.full_like(y,.001),y,
        {k:point[k] for k in ('rors','dt0')},num_warmup=8,num_samples=16,
        laplace_warmup=8,max_tree_depth=3,laplace_max_tree_depth=3,
        channel_varying_kwargs=('ld_fixed',),**kwargs)
    for name in ('rors','dt0','t0'):
        assert np.all(np.isfinite(draws[name]))
        assert draws[name].shape[0]>=16

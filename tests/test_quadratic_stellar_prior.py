"""The stellar-informed quadratic mode must propagate the quadratic law."""
from types import SimpleNamespace
import numpy as np
import pytest
from koala import limb_darkening as ld
from models.limb_darkening_config import validate_ld_prior, resolve_ld_prior


def synthetic_grid(monkeypatch, tmp_path, *, uncertainties=True):
    class QuadraticStar(ld.CachedStellarLimbDarkening):
        calls = 0
        def __init__(self, M_H, Teff, logg, **kwargs):
            type(self).calls += 1
            self.verbose = 0
            self.mus = np.linspace(1., 0., 30)
            self.stellar_wavelengths = np.linspace(6000., 53000., 1200)
            u1 = .4 + .0001*(Teff-3800.) + .02*(logg-4.6)
            u2 = .2 + .03*(M_H-.3) - .00005*(Teff-3800.)
            self.stellar_intensities = np.broadcast_to(
                1-u1*(1-self.mus)-u2*(1-self.mus)**2, (1200,30)).copy()
        def _read_sensitivity_data(self, mode):
            return np.linspace(6000.,53000.,100), np.ones(100)
        def compute_power2_ld_coeffs(self, *args, **kwargs):
            raise AssertionError('Quadratic stellar prior cannot use power-2 fits')
    monkeypatch.setattr(ld, 'CachedStellarLimbDarkening', QuadraticStar)
    monkeypatch.setattr(ld, '_get_ld_mode_bounds', lambda *a: ('test',6000.,53000.))
    grid_path=tmp_path/'grid';grid_path.mkdir()
    cfg = dict(teff=3800., logg=4.6, feh=.3, teff_sigma=50. if uncertainties else 0.,
               logg_sigma=.05 if uncertainties else 0., feh_sigma=.1 if uncertainties else 0.,
               ld_prior_n_grid=3, ld_prior_nsigma=2., ld_data_path=str(grid_path),
               ld_prior_cache_dir=str(tmp_path/'cache'), ld_prior_min_sigma=1e-4)
    return cfg, QuadraticStar


@pytest.mark.parametrize('resolved_profile', ['quadratic','power2'])
def test_stellar_mode_is_explicit_and_quadratic_default_is_uniform(resolved_profile):
    assert validate_ld_prior('stellarprior',ld_profile=resolved_profile)=='stellarprior'
    assert resolve_ld_prior(None,ld_profile='quadratic',has_stellar_uncertainties=True)=='uniform'


@pytest.mark.parametrize('vector', [False, True])
def test_actual_quadratic_grid_propagates_stellar_uncertainty_and_cache(monkeypatch,tmp_path,vector):
    cfg, star = synthetic_grid(monkeypatch,tmp_path)
    wavelengths=np.array([1.2,2.4]);widths=np.array([.05,.05]) if vector else 0.
    mu,sigma=ld.get_or_build_quadratic_ld_prior(cfg,wavelengths,widths,'NIRSPEC/PRISM')
    weights=np.exp(-.5*np.array([-2.,0.,2.])**2);weights/=weights.sum()
    variance=np.sum(weights*np.array([-2.,0.,2.])**2)
    expected_sigma=np.sqrt(variance*np.array([(.0001*50.)**2+(.02*.05)**2,
                                             (.03*.1)**2+(.00005*50.)**2]))
    expected_mu=np.broadcast_to([.4,.2],(2,2)) if vector else np.array([.4,.2])
    np.testing.assert_allclose(mu,expected_mu,atol=2e-8,rtol=0.)
    np.testing.assert_allclose(sigma,np.broadcast_to(expected_sigma,np.shape(mu)),atol=2e-8,rtol=1e-5)
    calls=star.calls
    cached=ld.get_or_build_quadratic_ld_prior(cfg,wavelengths,widths,'NIRSPEC/PRISM')
    assert star.calls==calls
    np.testing.assert_allclose(cached[0],mu,atol=1e-15,rtol=0.)
    np.testing.assert_allclose(cached[1],sigma,atol=1e-15,rtol=0.)
    assert 'u1_sigma_star' in next((tmp_path/'cache').glob('*.csv')).read_text()
    changed=dict(cfg,ld_mu_min=.3)
    ld.get_or_build_quadratic_ld_prior(changed,wavelengths,widths,'NIRSPEC/PRISM')
    assert star.calls>calls, 'Quadratic mu cut must invalidate the cache'


def test_zero_uncertainty_uses_stated_floor_fresh_and_cached(monkeypatch,tmp_path):
    cfg,star=synthetic_grid(monkeypatch,tmp_path,uncertainties=False)
    args=(cfg,np.array([1.2,2.4]),np.array([.05,.05]),'NIRSPEC/PRISM')
    fresh=ld.get_or_build_quadratic_ld_prior(*args)
    cached=ld.get_or_build_quadratic_ld_prior(*args)
    np.testing.assert_array_equal(fresh[1],np.full((2,2),1e-4))
    np.testing.assert_array_equal(fresh[1],cached[1])


def test_spectral_resolver_uses_quadratic_grid_sigmas(monkeypatch,tmp_path):
    from koala import spectroscopy as spec
    cfg,star=synthetic_grid(monkeypatch,tmp_path)
    monkeypatch.setattr(spec,'get_or_build_power2_ld_prior',lambda *a,**k:pytest.fail('Wrong law'))
    inputs=SimpleNamespace(explicit_ld=None,stellar_cfg=cfg,instrument='NIRSPEC/PRISM',
                           order=None,output_dir=str(tmp_path))
    model=SimpleNamespace(ld_prior_mode='stellarprior',ld_profile='quadratic')
    mu,sigma,sing=spec._spectroscopic_ld_uncached(model,inputs,np.array([1.2,2.4]),np.array([.05,.05]),'hr')
    assert sing is None
    assert np.shape(mu)==np.shape(sigma)==(2,2)
    assert np.all(np.asarray(sigma)<.01), 'Must use propagated scatter, not Gaussian width0.2'


@pytest.mark.parametrize('stage',['white','spectro'])
def test_actual_quadratic_target_retains_propagated_density_and_gradient(stage):
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from scipy.stats import truncnorm
    from koala.config import ParameterSpec
    from models.jaxoplanet import builder
    jax.config.update('jax_enable_x64',True)
    t=jnp.linspace(-.07,.07,13);center=jnp.array([.4,.2]);sigma=jnp.array([.03,.04])
    options=dict(param_method='a_rs',detrend_type='none',ld_mode='stellarprior',ld_profile='quadratic')
    if stage=='white':
        values=dict(period=3.3,t0=0.,a_rs=13.6,b=float(13.6*np.cos(np.deg2rad(88.))),ecc=0.,omega=90.,rprs=.18)
        specs={name:(ParameterSpec(name,'fixed',value),) for name,value in values.items()}
        model=builder.create_whitelight_model(parameter_priors=specs,**options)
        kw=dict(prior_params=dict(period=jnp.array([3.3]),u=center,u_sigma=sigma));errors=jnp.full(13,.001)
    else:
        model=builder.create_vectorized_model(**options)
        errors=jnp.full((2,13),.001)
        kw=dict(PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),
                mu_a_rs=jnp.array([13.6]),mu_b=jnp.array([13.6*jnp.cos(jnp.deg2rad(88.))]),
                mu_u_ld=jnp.broadcast_to(center,(2,2)),sigma_u_ld=jnp.broadcast_to(sigma,(2,2)))
    trace=handlers.trace(handlers.seed(model,1807)).get_trace(t,errors,**kw)
    distribution=trace['u']['fn'];value=trace['u']['value']
    expected=truncnorm.logpdf(np.asarray(value),-np.asarray(center/sigma),
                             np.asarray((1-center)/sigma),loc=center,scale=sigma).sum(-1)
    np.testing.assert_allclose(distribution.log_prob(value),expected,atol=1e-12,rtol=0.)
    derivative=jax.grad(lambda u:distribution.log_prob(u).sum())(value)
    np.testing.assert_allclose(derivative,(center-value)/sigma**2,atol=1e-10,rtol=1e-12)

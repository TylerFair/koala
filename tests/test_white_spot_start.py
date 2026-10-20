"""The staged white-light optimizer must see the configured spot start."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model
import pytest

from koala.white_light import _initialize_white_light, _whitelight_geometry_sites
from planet_specs import default_planet_specs


@pytest.mark.parametrize('engine,law', [
    ('jaxoplanet', 'quadratic'), ('jaxoplanet', 'power2'), ('harmonica', 'power2'),
])
@pytest.mark.parametrize('coordinates', ['physical', 'cadence'])
def test_configured_spot_start_reaches_actual_white_model(engine, law, coordinates):
    if engine == 'jaxoplanet':
        from models.jaxoplanet import create_whitelight_model
    else:
        from models.harmonica.builder import create_whitelight_model
    specs = default_planet_specs(period=3., t0=1.05, b=.2, rprs=.1, duration=.12)
    t = jnp.linspace(.95, 1.15, 31)
    context = SimpleNamespace(
        PRIOR_DUR=jnp.array([.12]), PRIOR_T0=jnp.array([1.05]),
        PERIOD_FIXED=jnp.array([3.]), PRIOR_B=jnp.array([.2]),
        PRIOR_RPRS=jnp.array([.1]), HARMONICA_A_RS=jnp.array([9.]),
        HARMONICA_ECC=jnp.array([0.]), HARMONICA_OMEGA=jnp.array([0.]),
        ld_prior_mode='stellarprior', ld_profile=law, ld_uniform_basis='coefficients',
        whitelight_ld_parameterization='coefficients', transit_engine=engine,
        HARMONICA_ODD_HARMONICS=(), detrending_type='spot', spot_mu=1.06,
        spot_sigma=.005, spot_amp=.001, surface_config={},
        planet_parameter_specs=specs, uses_surface_model=False, n_planets=1,
        step_width_mode='fixed',
        whitelight_trend_parameterization=coordinates, data=SimpleNamespace(wl_time=t),
        create_whitelight_model=create_whitelight_model, _engine_wl_kw={},
    )
    prior, initial, _, model = _initialize_white_light(
        context=context, U_mu_wl=jnp.array([.4, .3]), U_sigma_wl=jnp.array([.1, .1]),
        _engine_wl_kw=dict(ld_profile=law, parameter_priors=specs,
                           trend_parameterization=coordinates),
        geometry_sites=_whitelight_geometry_sites(specs, 1, 'duration'),
    )
    if coordinates == 'physical':
        assert not any(name.endswith('cadences') for name in initial)
    else:
        assert float(initial['spot_mu_offset_cadences']) == 0.
    # Changing the tracing RNG must not replace the user's spot initialization.
    for seed in (1, 17):
        info = initialize_model(
            jax.random.PRNGKey(seed), model, init_strategy=init_to_value(values=initial),
            model_args=(t, jnp.full_like(t, .001)),
            model_kwargs=dict(y=jnp.ones_like(t), prior_params=prior),
        )
        physical = info.postprocess_fn(info.param_info.z)
        np.testing.assert_allclose(physical['spot_mu'], context.spot_mu, rtol=0, atol=1e-14)
        np.testing.assert_allclose(physical['spot_sigma'], context.spot_sigma, rtol=0, atol=1e-14)
        assert np.isfinite(info.param_info.potential_energy)
        assert all(np.all(np.isfinite(v)) for v in jax.tree.leaves(info.param_info.z_grad))

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace
from models.ld_parameterization import Power2SmoothTransform, power2_smooth_transform
from models.model_building import sample_power2_coefficients


def test_smooth_roundtrip_and_jacobian():
    rng = np.random.default_rng(72)
    z = jnp.asarray(rng.uniform(-7, 7, (1000, 2)))
    for low, high in [(0., 1.), (.001, 1.), (0., 3.)]:
        transform = Power2SmoothTransform(low, high)
        y = transform(z)
        np.testing.assert_allclose(transform.inv(y), z, atol=1e-10, rtol=0)
        jac = jax.vmap(jax.jacfwd(transform))(z)
        actual = jnp.linalg.slogdet(jac)[1]
        np.testing.assert_allclose(transform.log_abs_det_jacobian(z, y), actual, atol=1e-10, rtol=0)
        assert np.all((y[:, 0] > 0) & (y[:, 0] < 1))
        assert np.all((y[:, 1] > low) & (y[:, 1] < high))


def test_smooth_density_is_original_prior_with_jacobian():
    for mode in ['uniform', 'gaussian']:
        def model():
            return sample_power2_coefficients(jnp.array([.4, .5]), None, mode, maxted_smooth=True)
        tr = trace(seed(model, 3)).get_trace()
        site = tr['ld_smooth']
        transform = power2_smooth_transform(mode)
        z = jnp.array([-.4, .6])
        coefficients = transform(z)
        prior = site['fn'].base_dist
        np.testing.assert_allclose(site['fn'].log_prob(z), prior.log_prob(coefficients)+transform.log_abs_det_jacobian(z, coefficients), atol=1e-12)
        assert 'ld_decorrelated_support' not in tr
        # Exercise pytree reconstruction used by compiled NumPyro programs.
        np.testing.assert_allclose(jax.jit(lambda t, x: t(x))(transform, z), coefficients)


def test_smooth_pipeline_sites_and_config():
    from koala.config import _resolve_ld_parameterization
    from koala.limb_darkening import _power2_ld_initial_sites, _power2_ld_optimization_sites
    import pytest
    for mode in ['uniform', 'gaussian']:
        assert _resolve_ld_parameterization(mode, requested='maxted_smooth') == 'maxted_smooth'
        assert _resolve_ld_parameterization(mode) == 'maxted_smooth'
        physical = jnp.array([[.4, .5], [.7, .3]])
        initial = _power2_ld_initial_sites(physical, 'maxted_smooth', mode)
        t = power2_smooth_transform(mode)
        np.testing.assert_allclose(t(initial['ld_smooth']), physical, atol=1e-12)
    assert _power2_ld_optimization_sites('maxted_smooth') == ['ld_smooth']
    with pytest.raises(ValueError):
        _resolve_ld_parameterization('uniform', 'quadratic', 'maxted_smooth')


def test_smooth_independent_initial_values_derive_from_physical_sites():
    from models.independent_nuts import _prepare_unconstrained_initial_values
    def model():
        sample_power2_coefficients(jnp.array([[.4, .5]]), None, 'uniform', shape=(1,), maxted_smooth=True)
    tr = trace(seed(model, 8)).get_trace()
    physical = jnp.array([[.4, .5], [.7, .3]])
    initial = _prepare_unconstrained_initial_values(
        {'c1':physical[:, 0], 'c2':physical[:, 1]}, tr, 2, 2)
    np.testing.assert_allclose(power2_smooth_transform('uniform')(initial['ld_smooth'][:,0]), physical, atol=1e-12)


def test_harmonica_smooth_sites_reach_physical_likelihood(monkeypatch):
    from models.harmonica.builder import create_vectorized_model
    captured = {}
    def transit(params, t):
        captured.update(params)
        return jnp.zeros((1, t.size), dtype=jnp.float64)
    monkeypatch.setattr('models.harmonica.builder.compute_transit_model_harmonica_batched', transit)
    for mode in ['uniform', 'gaussian']:
        model = create_vectorized_model(ld_mode=mode, ld_profile='power2',
                                        ld_parameterization='maxted_smooth', fit_jitter=False)
        tr = trace(seed(model, 3)).get_trace(
            jnp.linspace(0., .1, 8), jnp.full((1, 8), 1e-4), y=jnp.ones((1, 8)),
            mu_t0=jnp.array([.05]), mu_b=jnp.array([.2]), mu_cos_i=jnp.array([.02]),
            PERIOD=jnp.array([3.]), harmonica_a_rs=jnp.array([10.]),
            mu_u_ld=jnp.array([[.4, .5]]))
        assert tr['ld_smooth']['type'] == 'sample'
        np.testing.assert_allclose(captured['c_ld'], tr['c1']['value'])
        np.testing.assert_allclose(captured['alpha_ld'], tr['c2']['value'])

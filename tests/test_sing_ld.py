import numpy as np

from models.sing_ld import estimate_gray_offset, quadratic_to_sing, sing_to_quadratic


def test_sing_conversion_round_trip():
    c1 = np.array([0.2, 0.4, 0.7])
    c2 = np.array([0.3, -0.05, 0.1])
    l, delta = quadratic_to_sing(c1, c2)
    got_c1, got_c2 = sing_to_quadratic(l, delta)
    np.testing.assert_allclose(got_c1, c1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(got_c2, c2, rtol=0, atol=1e-14)


def test_gray_offset_estimator_recovers_synthetic_star():
    model = np.array([[0.30, 0.20], [0.35, 0.15], [0.40, 0.10]])
    ml, md = quadratic_to_sing(model[:, 0], model[:, 1])
    fitted = np.column_stack(sing_to_quadratic(ml + 0.02, md - 0.003))
    result = estimate_gray_offset(fitted, model, np.full_like(model, 0.01))
    np.testing.assert_allclose(result['l'], 0.02, atol=1e-14)
    np.testing.assert_allclose(result['delta'], -0.003, atol=1e-14)


def test_gray_offset_estimator_downweights_low_ess_channel():
    model = np.array([[0.30, 0.20], [0.30, 0.20]])
    ml, md = quadratic_to_sing(model[:, 0], model[:, 1])
    fitted = np.column_stack(sing_to_quadratic(
        ml + np.array([0.02, 0.20]), md - 0.003))
    sigma = np.full_like(model, 0.01)
    result = estimate_gray_offset(
        fitted, model, sigma, ess=np.array([[1000.0, 1000.0], [10.0, 10.0]]),
        n_draws=1000,
    )
    # The second channel has 100 times less inverse-variance weight after
    # sqrt(N/ESS) uncertainty inflation.
    expected_l = (0.02 * 100.0 + 0.20) / 101.0
    np.testing.assert_allclose(result['l'], expected_l, atol=1e-14)
    assert result['ess_inflation_applied']


def test_sing_prior_trace_has_physical_quadratic_coefficients():
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from models.jaxoplanet import create_vectorized_model

    model = create_vectorized_model(ld_mode='sing', ld_profile='quadratic')
    kwargs = dict(
        t=jnp.linspace(-0.03, 0.03, 15),
        yerr=jnp.full((2, 15), 1e-3),
        y=jnp.ones((2, 15)),
        mu_duration=jnp.array([0.06]), mu_t0=jnp.array([0.0]),
        mu_b=jnp.array([0.3]), mu_depths=jnp.full((2, 1), 0.01),
        PERIOD=jnp.array([3.0]),
        mu_u_ld=jnp.array([[0.50, 0.04], [0.55, 0.03]]),
        sigma_u_ld=jnp.array([[0.031, 0.016], [0.031, 0.016]]),
    )
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace(**kwargs)
    c1, c2 = np.asarray(trace['c1']['value']), np.asarray(trace['c2']['value'])
    assert np.all(c1 >= 0)
    assert np.all(c1 + 2 * c2 >= -1e-12)
    assert np.all(c1 + c2 <= 1 + 1e-12)


def _tiny_sing_problem():
    import jax.numpy as jnp
    from models.jaxoplanet import create_vectorized_model

    nchan, ntime = 1, 11
    t = jnp.linspace(-0.04, 0.04, ntime)
    yerr = jnp.full((nchan, ntime), 2e-3)
    y = jnp.ones_like(yerr)
    model = create_vectorized_model(
        detrend_type='linear', ld_mode='sing', trend_mode='free',
        n_planets=1, ld_profile='quadratic', param_method='duration',
        transit_window='off',
    )
    kwargs = dict(
        mu_duration=jnp.array([0.06]), mu_t0=jnp.array([0.0]),
        mu_b=jnp.array([0.3]), mu_depths=jnp.full((nchan, 1), 0.01),
        PERIOD=jnp.array([3.0]),
        mu_u_ld=jnp.array([[0.5, 0.04]]),
        sigma_u_ld=jnp.array([[0.031, 0.016]]),
        precomputed_yerr_per_lc=jnp.full((nchan,), 2e-3),
    )
    init = dict(
        rors=jnp.full((nchan, 1), 0.1), limb_l=jnp.array([0.5]),
        limb_delta=jnp.array([0.04]), c=jnp.ones(nchan), v=jnp.zeros(nchan),
    )
    varying = ('mu_depths', 'mu_u_ld', 'sigma_u_ld', 'precomputed_yerr_per_lc')
    return model, t, yerr, y, init, kwargs, varying


def test_sing_end_to_end_independent_nuts_laplace_metric():
    import jax
    import jax.numpy as jnp
    from models.independent_nuts import get_samples_independent

    model, t, yerr, y, init, kwargs, varying = _tiny_sing_problem()
    samples, diagnostics = get_samples_independent(
        model, jax.random.PRNGKey(31), t, yerr, y, init,
        num_warmup=2, num_samples=2, lane_width=1,
        nuts_kwargs={
            'mass_matrix': 'laplace', 'laplace_warmup': 2,
            'laplace_hessian_method': 'finite_difference',
            'laplace_max_tree_depth': 3,
        },
        channel_varying_kwargs=varying,
        return_diagnostics=True, **kwargs,
    )
    assert samples['c1'].shape == (2, 1)
    assert samples['limb_l'].shape == (2, 1)
    assert jnp.all(jnp.isfinite(samples['c1']))
    assert diagnostics.num_steps.shape == (2, 1)


def test_sing_end_to_end_laplace_is():
    import jax
    import jax.numpy as jnp
    from models.laplace_is import get_samples_laplace_is

    model, t, yerr, y, init, kwargs, varying = _tiny_sing_problem()
    samples, diagnostics = get_samples_laplace_is(
        model, jax.random.PRNGKey(32), t, yerr, y, init,
        num_warmup=2, num_samples=8, lane_width=1,
        laplace_is_num_draws=32, laplace_is_rounds=0,
        laplace_is_draw_chunk_size=8, laplace_is_map_maxiter=30,
        laplace_is_fallback=False, channel_varying_kwargs=varying,
        return_diagnostics=True, **kwargs,
    )
    assert samples['c2'].shape == (8, 1)
    assert samples['limb_delta'].shape == (8, 1)
    assert jnp.all(jnp.isfinite(samples['c2']))
    assert diagnostics.pareto_k.shape == (1,)

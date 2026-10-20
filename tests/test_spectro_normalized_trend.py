"""Reference normalized polynomial prior: values, derivatives and padding."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density
from models.jaxoplanet.builder import create_vectorized_model
from models.spectro_trends import normalized_trend_initial_values

jax.config.update('jax_enable_x64', True)


def test_normalized_trend_preserves_reference_box_and_flux_gradients():
    t = jnp.array([60324.18, 60324.2, 60324.25, 60324.32, 60324.4])
    z = jnp.array([1.4, 1.1, .7, .5, 1.5])
    error = jnp.full((1, 5), .001)
    obs = jnp.array([[1., .995, .97, .968, 1.001]])
    kw = dict(rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]), mu_t0=jnp.array([60324.31]),
              mu_b=jnp.array([.45]), PERIOD=jnp.array([3.325271]),
              ld_fixed=jnp.array([[.4, .2]]), transit_separation=z[None, :],
              precomputed_yerr_per_lc=jnp.array([.001]), y=obs)
    model = create_vectorized_model(detrend_type='quadratic', ld_mode='fixed',
                                    trend_basis='batman_normalized')
    def evaluate(beta, time=t, err=error, extra=None):
        values=dict(rors=jnp.array([[.18]]), log_jitter=jnp.array([-8.]),
                    trend_c=beta[:1], trend_v=beta[1:2], trend_v2=beta[2:3])
        return log_density(model,(time,err),{**kw,**(extra or {})},values)
    beta = jnp.array([1.0001,.002,-.004])
    value, trace = evaluate(beta)
    from models.jaxoplanet.core import _fixed_separation_transit
    # Use the production flux kernel on both sides of this basis identity.
    # Independent kernel parity is covered by the physical-orbit tests.
    transit = 1 + _fixed_separation_transit(dict(u=jnp.array([.4,.2]),rors=jnp.array([.18])),z,'quadratic_contact_safe')
    x=(t-jnp.mean(t))/(t.max()-t.min())
    def expected(b):return transit*(b[0]+b[1]*x+b[2]*x*x)
    actual=lambda b:evaluate(b)[1]['obs']['fn'].loc[0]
    np.testing.assert_allclose(actual(beta),expected(beta),atol=1e-9,rtol=0)
    np.testing.assert_allclose(jax.jacfwd(actual)(beta),jax.jacfwd(expected)(beta),atol=1e-7,rtol=1e-6)
    for name,bounds in [('trend_c',(.5,1.5)),('trend_v',(-1.,1.)),('trend_v2',(-1.,1.))]:
        np.testing.assert_array_equal(trace[name]['fn'].support.lower_bound,bounds[0])
        np.testing.assert_array_equal(trace[name]['fn'].support.upper_bound,bounds[1])
    padded_t=jnp.concatenate([jnp.array([60320.]),t,jnp.array([60330.])]);padded_err=jnp.pad(error,((0,0),(1,1)),constant_values=1.)
    extra=dict(likelihood_mask=jnp.array([False]+[True]*5+[False]),y=jnp.pad(obs,((0,0),(1,1)),constant_values=1.),transit_separation=jnp.concatenate([jnp.array([1.e3]),z,jnp.array([1.e3])])[None, :])
    padded,_=evaluate(beta,padded_t,padded_err,extra)
    np.testing.assert_allclose(value,padded,atol=1e-8,rtol=0)


def test_each_channel_normalizes_only_its_observed_times():
    time = jnp.array([60324.18, 60324.20, 60324.25, 60324.32, 60324.40])
    mask = jnp.array([[False, True, True, True, True],
                      [True, True, True, True, False]])
    model = create_vectorized_model(detrend_type='quadratic', ld_mode='fixed',
                                    trend_basis='batman_normalized')
    error = jnp.full((2, 5), .001)
    values = dict(rors=jnp.array([[.18], [.18]]), log_jitter=jnp.array([-8., -8.]),
                  trend_c=jnp.array([1.001, .999]), trend_v=jnp.array([.002, -.003]),
                  trend_v2=jnp.array([-.004, .005]))
    _, trace = log_density(model, (time, error), dict(
        y=jnp.ones((2, 5)), rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]), mu_t0=jnp.array([60324.31]),
        mu_b=jnp.array([.45]), PERIOD=jnp.array([3.325271]),
        ld_fixed=jnp.array([[.4, .2], [.4, .2]]), likelihood_mask=mask,
        transit_separation=jnp.full((1, 5), 1.e3),
        precomputed_yerr_per_lc=jnp.array([.001, .001])), values)
    distribution = trace['obs']['fn']
    while not hasattr(distribution, 'loc'):
        distribution = distribution.base_dist
    expected = []
    for lane in range(2):
        selected = np.asarray(time)[np.asarray(mask[lane])]
        x = (np.asarray(time) - selected.mean()) / np.ptp(selected)
        expected.append(values['trend_c'][lane] + values['trend_v'][lane]*x
                        + values['trend_v2'][lane]*x*x)
    np.testing.assert_allclose(distribution.loc, expected, atol=1e-9, rtol=0)


def test_normalized_initial_values_round_trip():
    t=np.linspace(60000.,60000.24,21);initial=dict(c=np.array([1.001]),v=np.array([.003]),v2=np.array([-.01]))
    normalized=normalized_trend_initial_values(t,initial);x=(t-t.mean())/np.ptp(t);dt=t-t.min()
    expected=initial['c']+initial['v']*dt+initial['v2']*dt**2
    actual=normalized['trend_c']+normalized['trend_v']*x+normalized['trend_v2']*x*x
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


@pytest.mark.parametrize('mode,trend',[('fixed','quadratic'),('free','none')])
def test_normalized_basis_rejects_unsupported_models(mode,trend):
    with pytest.raises(ValueError,match='batman_normalized'):
        create_vectorized_model(trend_basis='batman_normalized',trend_mode=mode,detrend_type=trend)


@pytest.mark.slow
def test_wide_marginalized_normalized_matches_direct_integration_and_gradients():
    from models.linear_marginalization import marginalized_log_likelihood_and_conditional
    from models.jaxoplanet.core import _fixed_separation_transit
    t = jnp.linspace(60324., 60324.25, 51)
    x = (t-t.mean()) / jnp.ptp(t)
    z = jnp.sqrt(.45**2 + (25.*(t-t.mean()))**2)
    error = jnp.full((1,t.size), .001)
    y = jnp.array([1. + .002*x - .003*x*x])
    model = create_vectorized_model(detrend_type='quadratic', ld_mode='fixed',
        trend_basis='batman_normalized', trend_mode='gaussian_marginalized')
    kwargs=dict(rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]),mu_t0=jnp.array([60324.125]),
        mu_b=jnp.array([.45]), PERIOD=jnp.array([3.325271]),ld_fixed=jnp.array([[.4,.2]]),
        transit_separation=z[None, :],precomputed_yerr_per_lc=jnp.array([.001]),y=y)
    for scale in [3.,10.,30.,100.]:
        def traced(radius):
            return log_density(model,(t,error),dict(**kwargs,trend_prior_scale=jnp.full((1,3),scale)),
                dict(rors=radius.reshape(1,1),log_jitter=jnp.array([-8.])))
        def actual(radius):
            site=traced(radius)[1]['obs_marginalized']
            return site['fn'].log_prob(site['value'])
        def expected(radius):
            flux=1+_fixed_separation_transit(dict(u=jnp.array([.4,.2]),rors=radius.reshape(1)),z,'quadratic_contact_safe')
            design=flux[:,None]*jnp.stack([jnp.ones_like(x),x,x*x],axis=-1)
            return marginalized_log_likelihood_and_conditional(y[0],design,
                jnp.sqrt(error[0]**2+jnp.exp(-16.)),jnp.array([1.,0.,0.]),
                prior_scale=jnp.full(3,scale))[0]
        r=jnp.array(.18)
        np.testing.assert_allclose(actual(r),expected(r),atol=1e-7,rtol=0)
        np.testing.assert_allclose(jax.grad(actual)(r),jax.grad(expected)(r),atol=1e-7,rtol=1e-6)
        trace=traced(r)[1]
        physical=trace['trend_beta_mean']['value'][0]
        normalized=trace['trend_normalized_mean']['value'][0]
        dt=t-t.min()
        np.testing.assert_allclose(physical[0]+physical[1]*dt+physical[2]*dt**2,
            normalized[0]+normalized[1]*x+normalized[2]*x*x,atol=1e-9,rtol=0)


def test_configurable_sampled_bounds_and_validation():
    bounds=dict(c=(-10.,10.),v=(-10.,10.),v2=(-10.,10.))
    model=create_vectorized_model(detrend_type='quadratic',trend_basis='batman_normalized',
                                  ld_mode='fixed',trend_bounds=bounds)
    t=jnp.linspace(0.,.25,5)
    _,trace=log_density(model,(t,jnp.ones((1,5))*.001),dict(
        y=jnp.ones((1,5)),PERIOD=jnp.array([3.3]),mu_t0=jnp.array([.1]),
        mu_b=jnp.array([.45]),rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]),
        ld_fixed=jnp.array([[.4,.2]]),transit_separation=jnp.ones((1, 5))*1.e3),
        dict(rors=jnp.array([[.18]]),log_jitter=jnp.array([-8.]),
             trend_c=jnp.array([1.]),trend_v=jnp.array([0.]),trend_v2=jnp.array([0.])))
    for name in bounds:
        np.testing.assert_array_equal(trace['trend_'+name]['fn'].support.lower_bound,-10.)
        np.testing.assert_array_equal(trace['trend_'+name]['fn'].support.upper_bound,10.)
    with pytest.raises(ValueError,match='ordered'):
        create_vectorized_model(trend_bounds={'v2':(1.,-1.)})


@pytest.mark.slow
def test_normalized_marginalized_transit_window_has_full_likelihood_gradients():
    from models.jaxoplanet.core import circular_orbit_separation
    time=np.linspace(-.15,.15,301)
    z=circular_orbit_separation(time,3.3,0.,13.6,.45)
    indices=np.flatnonzero(z<1+np.sqrt(.5)+1e-6)
    kw=dict(y=jnp.ones((1,301)),PERIOD=jnp.array([3.3]),mu_t0=jnp.array([0.]),
        mu_b=jnp.array([.45]),rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]),
        ld_fixed=jnp.array([[.4,.2]]),transit_separation=jnp.array(z)[None, :],
        trend_prior_scale=jnp.full((1,3),100.))
    common=dict(detrend_type='quadratic',ld_mode='fixed',trend_mode='gaussian_marginalized',
                trend_basis='batman_normalized')
    full=create_vectorized_model(**common)
    window=create_vectorized_model(**common,transit_window='auto',transit_window_indices=indices)
    def potential(model,q):
        return log_density(model,(jnp.array(time),jnp.full((1,301),.001)),kw,
            dict(rors=q[:1].reshape(1,1),log_jitter=q[1:]))[0]
    for r in [.1,.18,.35,.6]:
        q=jnp.array([r,-8.])
        np.testing.assert_allclose(potential(full,q),potential(window,q),atol=1e-8,rtol=0)
        np.testing.assert_allclose(jax.grad(lambda x:potential(full,x))(q),
            jax.grad(lambda x:potential(window,x))(q),atol=1e-7,rtol=1e-6)


def test_pipeline_gaussian_prior_uses_selected_coordinate_system():
    from koala.config import _build_gaussian_trend_prior
    initial=dict(c=np.array([1.01]),v=np.array([.4]),v2=np.array([-.3]))
    mean,scale,names=_build_gaussian_trend_prior('quadratic',1,initial,
        dict(spectro_trend_basis='batman_normalized',trend_prior_scale=10.))
    assert names == ('c','v','v2')
    np.testing.assert_array_equal(mean,[[1.,0.,0.]])
    np.testing.assert_array_equal(scale,[[10.,10.,10.]])


@pytest.mark.parametrize('engine', ['jaxoplanet', 'harmonica'])
@pytest.mark.parametrize('trend,order', [('cubic', 3), ('quartic', 4)])
def test_higher_order_polynomials_preserve_values_gradients_and_initialization(trend, order, engine, monkeypatch):
    time = jnp.linspace(60000., 60000.3, 19)
    mask = np.ones((2, time.size), dtype=bool)
    mask[0, :2] = False
    mask[1, -3:] = False
    names = ['c', 'v', 'v2', 'v3', 'v4'][:order + 1]
    coefficients = jnp.array([[1.001, .002, -.003, .004, -.005],
                              [.999, -.003, .004, -.005, .006]])[:, :order + 1]
    normalized_times = jnp.array([
        (np.asarray(time) - np.asarray(time)[valid].mean()) /
        np.ptp(np.asarray(time)[valid]) for valid in mask])
    factory = create_vectorized_model
    engine_options = {}
    if engine == 'harmonica':
        from models.harmonica.builder import create_vectorized_model as factory
        # Isolate polynomial coordinates; native kernel parity is tested separately.
        monkeypatch.setattr('models.harmonica.builder.compute_transit_model_harmonica_batched',
                            lambda params, t: jnp.zeros((2, t.size)))
        engine_options = dict(ld_profile='quadratic', odd_parameterization='delta_r')
    model = factory(detrend_type=trend, ld_mode='fixed',
                    trend_basis='batman_normalized', **engine_options)
    kwargs = dict(y=jnp.ones((2, time.size)), PERIOD=jnp.array([3.3]),
                  mu_t0=jnp.array([60000.15]), mu_b=jnp.array([.45]),
                  rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]), ld_fixed=jnp.array([[.4, .2]] * 2),
                  transit_separation=jnp.full((1, time.size), 1.e3),
                  likelihood_mask=mask)
    if engine == 'harmonica':
        kwargs['harmonica_a_rs'] = jnp.array([13.6])

    def actual(beta):
        values = dict(rors=jnp.full((2, 1), .18), log_jitter=jnp.full(2, -8.))
        if engine == 'harmonica':
            values['delta_r'] = jnp.zeros((2, 1))
        values.update({'trend_' + name: beta[:, index] for index, name in enumerate(names)})
        _, sites = log_density(model, (time, jnp.full((2, time.size), .001)), kwargs, values)
        distribution = sites['obs']['fn']
        while not hasattr(distribution, 'loc'):
            distribution = distribution.base_dist
        return distribution.loc

    def expected(beta):
        return sum(beta[:, index, None] * normalized_times**index
                   for index in range(order + 1))

    np.testing.assert_allclose(actual(coefficients), expected(coefficients), atol=1e-9, rtol=0)
    np.testing.assert_allclose(jax.jacfwd(actual)(coefficients),
                               jax.jacfwd(expected)(coefficients), atol=1e-7, rtol=1e-6)
    initial = {name: np.asarray(coefficients[:, index]) for index, name in enumerate(names)}
    converted = normalized_trend_initial_values(np.asarray(time), initial)
    x = (time - time.mean()) / jnp.ptp(time)
    elapsed = time - time.min()
    np.testing.assert_allclose(
        sum(converted['trend_' + name][:, None] * x**index for index, name in enumerate(names)),
        sum(initial[name][:, None] * elapsed**index for index, name in enumerate(names)),
        atol=1e-9, rtol=0)


def test_harmonica_masked_rows_carry_no_likelihood_or_gradient(monkeypatch):
    from models.harmonica.builder import create_vectorized_model as factory
    monkeypatch.setattr('models.harmonica.builder.compute_transit_model_harmonica_batched',
                        lambda params, t: jnp.zeros((1, t.size)))
    model = factory(detrend_type='quadratic', ld_mode='fixed', ld_profile='quadratic',
                    odd_parameterization='delta_r', trend_basis='batman_normalized')
    t = jnp.linspace(0., .25, 13)
    mask = jnp.array([[False] + [True] * 10 + [False] * 2])
    kwargs = dict(PERIOD=jnp.array([3.3]), mu_t0=jnp.array([.12]),
                  mu_b=jnp.array([.45]), harmonica_a_rs=jnp.array([13.6]),
                  ld_fixed=jnp.array([[.4, .2]]), likelihood_mask=mask)
    def potential(beta, bad_value):
        values = dict(rors=jnp.full((1, 1), .18), delta_r=jnp.zeros((1, 1)),
                      log_jitter=jnp.array([-8.]), trend_c=beta[:1],
                      trend_v=beta[1:2], trend_v2=beta[2:])
        y = jnp.where(mask, 1., bad_value)
        return log_density(model, (t, jnp.full((1, t.size), .001)),
                           dict(y=y, **kwargs), values)[0]
    beta = jnp.array([1., .002, -.003])
    np.testing.assert_allclose(potential(beta, 1.), potential(beta, 1.e6), atol=1e-9, rtol=0)
    np.testing.assert_allclose(jax.grad(potential)(beta, 1.),
                               jax.grad(potential)(beta, 1.e6), atol=1e-7, rtol=1e-6)


def test_public_default_is_the_wide_normalized_target_before_fingerprinting():
    from models.spectro_trends import resolve_spectral_trend_defaults
    from numpyro import handlers
    flags = resolve_spectral_trend_defaults({'detrending_type': 'quadratic'})
    assert flags['spectro_trend_basis'] == 'batman_normalized'
    model = create_vectorized_model(detrend_type='quadratic', ld_mode='fixed',
        trend_basis=flags['spectro_trend_basis'], trend_bounds=flags['spectro_trend_bounds'])
    t = jnp.array([0., .03, .1, .2, .3])
    trace = handlers.trace(handlers.seed(model, 505)).get_trace(t, jnp.full((1,5), .001),
        y=jnp.ones((1,5)), rors_reference=jnp.array([.18]), mu_duration=jnp.array([.085]),
        mu_t0=jnp.array([.15]), mu_b=jnp.array([.45]), PERIOD=jnp.array([3.325271]),
        ld_fixed=jnp.array([[.4,.2]]), transit_separation=jnp.full((1,5), 1.e3))
    for name in ['c', 'v', 'v2']:
        fn = trace['trend_'+name]['fn']
        while not hasattr(fn, 'low'):
            fn = fn.base_dist
        np.testing.assert_allclose(fn.low, -10., atol=0., rtol=0.)
        np.testing.assert_allclose(fn.high, 10., atol=0., rtol=0.)
    x = (t-t.mean())/jnp.ptp(t)
    expected = sum(trace['trend_'+name]['value'][:,None]*x**i
                   for i,name in enumerate(['c','v','v2']))
    fn = trace['obs']['fn']
    while not hasattr(fn, 'loc'):
        fn = fn.base_dist
    np.testing.assert_allclose(fn.loc, expected, atol=1e-9, rtol=0.)


def test_public_defaults_preserve_explicit_coordinates_and_bound_overrides():
    from models.spectro_trends import resolve_spectral_trend_defaults as resolve
    original = {'detrending_type':'quartic', 'spectro_trend_bounds':{'v4':[-30.,30.]}}
    actual = resolve(original)
    assert 'spectro_trend_basis' not in original
    assert actual['spectro_trend_bounds']['v4'] == [-30.,30.]
    assert actual['spectro_trend_bounds']['c'] == [-10.,10.]
    for flags in [{'spectro_trend_basis':'elapsed_days'}, {'detrending_type':'explinear'},
                  {'trend_inference':'gaussian_marginalized'}]:
        actual = resolve(flags)
        assert actual['spectro_trend_basis'] == 'elapsed_days'
        assert 'spectro_trend_bounds' not in actual

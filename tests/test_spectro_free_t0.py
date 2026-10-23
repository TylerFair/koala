"""Independent channel epochs keep the white-light Keplerian orbit fixed."""
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro import handlers
from models.jaxoplanet.builder import create_vectorized_model
from models.jaxoplanet.core import fixed_orbit_separation, compute_transit_model
from koala.time_shift import analytic_time_shift_coefficient, time_shift_table, save_time_shift_products


def traced(free=None, offsets=(-20., 30.), ld_fixed=((.3,.2), (.4,.1))):
    kw = {} if free is None else {'surface_config': {'model':'transit', 'spectro_free_t0':free}}
    model = create_vectorized_model(
        detrend_type='none', ld_mode='fixed', param_method='a_rs',
        jaxoplanet_kernel='quadratic_contact_safe', **kw)
    t = jnp.linspace(-.11, .11, 151)
    values = dict(rors=jnp.array([[.12], [.18]]), log_jitter=jnp.full(2, -12.))
    if free:
        values['dt0'] = jnp.asarray(offsets)[:, None]/86400
    trace = handlers.trace(handlers.substitute(handlers.seed(model, 7), data=values)).get_trace(
        t, jnp.full((2, t.size), .0002), y=jnp.ones((2, t.size)),
        mu_t0=jnp.array([0.]), mu_b=jnp.array([.35]), mu_a_rs=jnp.array([9.]),
        mu_duration=jnp.array([.1]), PERIOD=jnp.array([4.]),
        ld_fixed=jnp.asarray(ld_fixed))
    return model, t, trace


def test_channel_epochs_match_independent_batman_and_keep_orbit():
    import batman
    model, t, tr = traced(True)
    _, _, uniform = traced(True, ld_fixed=((0.,0.), (0.,0.)))
    np.testing.assert_allclose(tr['t0']['value'][:,0]*86400, [-20,30], atol=1e-12)
    assert 'dt0' in model.surface_science_sites
    for i, (r, offset, u) in enumerate(zip([.12,.18], [-20,30], [[.3,.2],[.4,.1]])):
        p=batman.TransitParams();p.t0=offset/86400;p.per=4.;p.a=9.;p.rp=r
        p.inc=np.degrees(np.arccos(.35/9));p.ecc=0.;p.w=90.;p.u=[0.,0.];p.limb_dark='quadratic'
        expected=batman.TransitModel(p,np.asarray(t)).light_curve(p)
        # Uniform-star batman independently checks exact shifted geometry.
        # Quadratic batman has a pre-existing few-e-9 elliptic-kernel offset;
        # check nonzero LD against the unchanged fixed-epoch path at shifted t.
        np.testing.assert_allclose(uniform['obs']['fn'].loc[i], expected, atol=1e-9, rtol=0)
        fixed = dict(period=jnp.array([4.]), t0=jnp.array([0.]),
                     a_rs=jnp.array([9.]), b=jnp.array([.35]),
                     rors=jnp.array([r]), u=jnp.asarray(u),
                     _jaxoplanet_kernel='quadratic_contact_safe')
        shifted = 1 + compute_transit_model(fixed, t-offset/86400)
        np.testing.assert_allclose(tr['obs']['fn'].loc[i], shifted, atol=1e-12, rtol=0)


def test_flag_off_is_bitwise_identical_and_has_no_epoch_sites():
    _, _, old = traced()
    _, _, off = traced(False)
    assert old.keys() == off.keys()
    assert 'dt0' not in old and 't0' not in old
    for name in old:
        np.testing.assert_array_equal(old[name]['value'], off[name]['value'])
    np.testing.assert_array_equal(old['obs']['fn'].loc, off['obs']['fn'].loc)


@pytest.mark.parametrize('offset_s', [-1800., 0., 1800.])
def test_exact_shifted_orbit_epoch_gradient(offset_s):
    import batman
    t=jnp.linspace(-.12,.12,181);weights=jnp.cos(13*t)
    def signal(offset):
        p=dict(period=jnp.array([4.]),t0=jnp.array([offset]),a_rs=jnp.array([9.]),
               b=jnp.array([.35]),rors=jnp.array([.12]),u=jnp.array([.3,.2]),
               _jaxoplanet_kernel='quadratic_contact_safe')
        return jnp.sum(weights*compute_transit_model(p,t))
    x=offset_s/86400
    def reference(offset):
        p=batman.TransitParams();p.t0=offset;p.per=4.;p.a=9.;p.rp=.12
        p.inc=np.degrees(np.arccos(.35/9));p.ecc=0.;p.w=90.;p.u=[.3,.2];p.limb_dark='quadratic'
        return np.dot(np.asarray(weights),batman.TransitModel(p,np.asarray(t)).light_curve(p)-1)
    h=2e-7
    fd=(reference(x-2*h)-8*reference(x-h)+8*reference(x+h)-reference(x+2*h))/(12*h)
    np.testing.assert_allclose(jax.grad(signal)(x), fd, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('options', [dict(joint_geometry=True), dict(t0_half_width_days=0),
                                      dict(surface_config={'model':'eclipse'})])
def test_free_t0_invalid_options_fail_explicitly(options):
    with pytest.raises(ValueError):
        create_vectorized_model(surface_config={
            'model':'transit', 'spectro_free_t0':True,
            **options.pop('surface_config', {}),
            **({'spectro_t0_half_width_days':options.pop('t0_half_width_days')} if 't0_half_width_days' in options else {}),
        }, **options)


def test_prepared_separation_cannot_hide_a_free_epoch():
    model, t, _ = traced(True)
    with pytest.raises(ValueError, match='Prepared separations'):
        handlers.seed(model, 0)(t, jnp.ones((1,t.size)), mu_t0=jnp.array([0.]),
            mu_a_rs=jnp.array([9.]),mu_b=jnp.array([.35]),PERIOD=jnp.array([4.]),
            ld_fixed=jnp.array([[.3,.2]]),transit_separation=jnp.ones((1,t.size)))


def test_spherical_zero_shift_and_finite_timing_products(tmp_path):
    rng=np.random.default_rng(22);n=2000
    dt=rng.normal(0, 2/86400, (n//2,2,1))
    dt=np.concatenate([dt,-dt]) # exact zero posterior centre, nonzero uncertainty
    samples={'rors':rng.normal(.12,.0001,(n,2,1)), 'dt0':dt}
    geometry=SimpleNamespace(T0_BASE=[60000.125], PERIOD_FIXED=[4.], A_RS_BASE=[9.],
                             B_BASE=[.35], HARMONICA_ECC=[0.])
    response = dict(time_shift_coefficient=5000., d_dt0_dRL_s_per_Rstar=-5000.,
        d_dt0_dRT_s_per_Rstar=5000., d_depth_dRL=.12, d_depth_dRT=.12,
        antisymmetry_check=0., radius_response_relative_error=1e-12, n_retained=151)
    responses = [[response], [response]]
    table=time_shift_table([1.,2.],[.1,.1],samples,geometry,responses=responses)
    np.testing.assert_array_equal(table.time_shift_coefficient, 5000.)
    np.testing.assert_allclose(table.time_shift_coefficient_analytic,
                               analytic_time_shift_coefficient(4.,9.,.35), rtol=1e-14)
    np.testing.assert_array_equal(table.dt0_s,0.)
    np.testing.assert_array_equal(table.t0,2460000.125)
    assert np.all(np.isfinite(table.to_numpy()))
    prefix=str(tmp_path/'test_R10')
    import pandas as pd
    pd.DataFrame({'wavelength':[1.,2.]}).to_csv(prefix+'_spectrum.csv', index=False)
    save_time_shift_products([1.,2.],[.1,.1],samples,prefix,geometry,responses=responses)
    output=pd.read_csv(prefix+'_spectrum.csv')
    np.testing.assert_array_equal(output.time_shift_coefficient, 5000.)
    np.testing.assert_array_equal(output.d_dt0_dRL_s_per_Rstar, -5000.)
    assert set(('depth','depth_err','t0','t0_err_s','dt0_s','corr_depth_t0',
                'time_shift_coefficient')) <= set(output)
    assert (tmp_path/'figures'/'R10_timing.png').exists()
    assert (tmp_path/'test_R10_timing.csv').exists()
    assert (tmp_path/'test_R10_timing.json').exists()
    assert not list(tmp_path.rglob('*.pdf'))


def test_platon_coefficient_central_chord_and_orientation():
    # Independent substitution of theta=0 into the published half-disk formula.
    period=4.;a=9.;velocity=2*np.pi*np.sqrt(a*a-1)/(period*86400)
    expected=3*(np.pi*np.pi/4+1)/16/velocity
    np.testing.assert_allclose(analytic_time_shift_coefficient(period,a,0),expected,rtol=1e-14)
    np.testing.assert_array_equal(analytic_time_shift_coefficient(period,a,.35),
                                  analytic_time_shift_coefficient(period,a,-.35))
    with pytest.raises(ValueError):
        analytic_time_shift_coefficient(period,a,1.)


def test_free_epoch_widens_window_and_avoids_prepared_orbit(monkeypatch):
    import koala.spectroscopy as sp
    from models.jaxoplanet.core import build_transit_window_indices
    monkeypatch.setattr(sp,'_build_spectroscopic_model',lambda factory,**kw: kw)
    geometry=SimpleNamespace(PERIOD_FIXED=jnp.array([4.]),T0_BASE=jnp.array([0.]),
        A_RS_BASE=jnp.array([9.]),B_BASE=jnp.array([.35]),RORS_BASE=jnp.array([.12]),
        HARMONICA_ECC=jnp.array([0.]),HARMONICA_OMEGA=jnp.array([np.pi/2]),n_planets=1)
    model=SimpleNamespace(transit_engine='jaxoplanet',transit_window_optimization='auto',
        _engine_spectro_kw={},surface_config={'model':'transit','spots':()},
        ld_profile='quadratic',spectro_ld_parameterization='coefficients',create_vectorized_model=None)
    inputs=SimpleNamespace(flags={'spectro_free_t0':True},build_transit_window_indices=build_transit_window_indices)
    t=jnp.linspace(-.3,.3,1001)
    fit=sp._spectroscopic_model(geometry,model,inputs,t,jnp.array([[.3,.2]]),
        {'rors':jnp.array([[.12]])},None,'fixed','free','none','test')
    assert fit.transit_separation is None
    assert fit.surface_basis is None
    assert fit.builder_kwargs['transit_grid']=='off'
    keep=np.zeros(t.size,bool);keep[np.asarray(fit.transit_window_indices)]=True
    for epoch in [-1/48,1/48]:
        z=fixed_orbit_separation(t,4.,epoch,9.,.35)
        assert np.all(keep[np.asarray(z)<1+np.sqrt(.5)])


def test_two_limb_injection_linear_timing_recovery():
    """Project an independently integrated half-disk injection onto timing.

    This is a deterministic linear-response unit check, not a posterior fit.
    The bundled GPU check additionally uses the actual Laplace sampler.
    """
    from numpy.polynomial.legendre import leggauss
    period,a,b,r=4.,9.,.2,.08
    t=np.linspace(-.095,.095,1601)
    phase=2*np.pi*t/period
    xp=a*np.sin(phase);yp=b*np.cos(phase)
    nodes,weights=leggauss(1024)
    def half_area(radius,sign):
        x=sign*radius*(nodes+1)/2
        y=np.sqrt(np.maximum(0,radius*radius-x*x))
        stellar_y=np.sqrt(np.maximum(0,1-(xp[:,None]+x)**2))
        height=np.maximum(0,np.minimum(yp[:,None]+y,stellar_y)
                            -np.maximum(yp[:,None]-y,-stellar_y))
        height=np.where(np.abs(xp[:,None]+x)<1,height,0)
        return height@(weights*radius/2)
    # xp increases with time: +x is leading and -x is trailing.
    contrast=.002
    asymmetric=1-(half_area(r+contrast/2,-1)+half_area(r-contrast/2,+1))/np.pi
    spherical=1-(half_area(r,-1)+half_area(r,+1))/np.pi
    derivative=np.gradient(spherical,t*86400,edge_order=2)
    recovered=-np.dot(asymmetric-spherical,derivative)/np.dot(derivative,derivative)
    prediction=float(analytic_time_shift_coefficient(period,a,b))*contrast
    assert recovered>0
    np.testing.assert_allclose(recovered,prediction,rtol=.03,atol=0)
    assert -np.dot(spherical-spherical,derivative)==0


def test_full_plot_curve_uses_each_channels_epoch():
    from koala.plotting import _single_curve_transit_signal
    _, t, trace = traced(True)
    params=dict(t0=trace['t0']['value'],b=jnp.array([.35]),a_rs=jnp.array([9.]),
        rors=trace['rors']['value'],u=jnp.array([[.3,.2],[.4,.1]]),
        _jaxoplanet_kernel='quadratic_contact_safe',_ld_profile='quadratic')
    orbit=dict(period=jnp.array([4.]),transit_engine='jaxoplanet',param_method='a_rs')
    for channel in range(2):
        actual=_single_curve_transit_signal(t,params,orbit,channel)
        np.testing.assert_allclose(actual,trace['obs']['fn'].loc[channel]-1,atol=1e-12,rtol=0)

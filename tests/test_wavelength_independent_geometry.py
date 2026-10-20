"""Fixed-orbit spectroscopy: independent references, derivatives and recovery.

A frozen test-only linear fixed-T14 helper measures the incompatible legacy
target; production models always construct a physical shared orbit.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from scipy.integrate import quad
from scipy.optimize import least_squares
from models.harmonica.core import harmonica_a_rs_from_duration, harmonica_duration_from_geometry
from models.jaxoplanet.core import (compute_transit_model, orbit_separation,
    compute_transit_model_window)
from models.jaxoplanet.builder import create_vectorized_model

jax.config.update('jax_enable_x64', True)

P, A, B, KREF = 3.325271, 13.61, .447, .182
D = P/np.pi*np.arcsin(np.sqrt(((1+KREF)**2-B**2)/(A*A-B*B)))
T = np.linspace(-.065, .065, 151)


def _compute_transit_model_duration(params, t):
    """Frozen pre-fix linear target, only for regression/recovery evidence."""
    from jaxoplanet.orbits.transit import TransitOrbit
    from jaxoplanet.light_curves import limb_dark_light_curve
    orbit=TransitOrbit(period=params['period'][0],duration=params['duration'][0],
        time_transit=params['t0'][0],impact_param=params['b'][0],radius_ratio=params['rors'][0])
    return limb_dark_light_curve(orbit, params['u'])(t)


def parameters(k=.195, law='quadratic', ecc=0.):
    return dict(period=jnp.array([P]),t0=jnp.array([0.]),b=jnp.array([B]),
        a_rs=jnp.array([A]),rors=jnp.array([k]),ecc=jnp.array([ecc]),omega=jnp.array([.7]),
        u=jnp.array([.3,.2]) if law=='quadratic' else jnp.zeros(12),
        **(dict(c1=.6,c2=.6,_power2_exact=True) if law=='power2' else {}))


def independent_quadratic(z, k, u):
    """Adaptive NumPy/SciPy annular integral, independent of transit kernels."""
    if z>=1+k: return 0.
    u1,u2=u
    def intensity(r):
        mu=np.sqrt(max(0.,1-r*r))
        return 1-u1*(1-mu)-u2*(1-mu)**2
    inner=min(1.,max(0.,k-z))
    total=quad(lambda r:2*np.pi*r*intensity(r),0,inner,epsabs=2e-13)[0]
    if z>0:
        total+=quad(lambda r:2*r*intensity(r)*np.arccos(np.clip(
            (r*r+z*z-k*k)/(2*r*z),-1,1)),abs(z-k),min(1.,z+k),
            epsabs=2e-13,epsrel=2e-12,limit=400)[0]
    return -total/(np.pi*(1-u1/3-u2/6))


@pytest.mark.parametrize('law',['quadratic','power2'])
@pytest.mark.parametrize('ecc',[0.,.05,.1])
def test_fixed_orbit_batman(law,ecc):
    batman=pytest.importorskip('batman')
    p=parameters(law=law,ecc=ecc)
    bp=batman.TransitParams();bp.per=P;bp.t0=0.;bp.a=A;bp.rp=.195
    bp.ecc=ecc;bp.w=np.degrees(.7)
    bp.inc=np.degrees(np.arccos(B*(1+ecc*np.sin(.7))/(A*(1-ecc**2))))
    bp.limb_dark=law;bp.u=[.3,.2] if law=='quadratic' else [.6,.6]
    expected=batman.TransitModel(bp,T,max_err=.001).light_curve(bp)-1
    actual=jax.jit(lambda t: compute_transit_model(p,t))(jnp.array(T))
    print('batman_max_flux_error',law,ecc,float(np.max(np.abs(np.asarray(actual)-expected))))
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)


@pytest.mark.parametrize("law",["quadratic","power2"])
def test_numpy_exact_quadratic_and_finite_difference_gradients(law):
    # Times avoid nondifferentiable exact contacts; include ingress and egress.
    t=np.linspace(-.045,.045,81)
    phase=2*np.pi*t/P
    z=np.sqrt(A*A*np.sin(phase)**2+B*B*np.cos(phase)**2)
    expected=np.array([independent_quadratic(x,.195,[.3,.2]) for x in z])
    p=parameters(law=law);actual=compute_transit_model(p,jnp.array(t))
    if law == "quadratic":
        print('numpy_quadratic_max_flux_error',float(np.max(np.abs(np.asarray(actual)-expected))))
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
    def evaluate(q):
        update = dict(u=q[3:]) if law == 'quadratic' else dict(c1=q[3],c2=q[4])
        return compute_transit_model(dict(p,rors=q[:1],a_rs=q[1:2],b=q[2:3],**update),jnp.array(t))
    q=np.array([.195,A,B,*([.3,.2] if law=="quadratic" else [.6,.6])]);jac=np.asarray(jax.jit(jax.jacfwd(evaluate))(q))
    fd=np.column_stack([(np.asarray(evaluate(q+np.eye(5)[i]*1e-6))-
                         np.asarray(evaluate(q-np.eye(5)[i]*1e-6)))/(2e-6) for i in range(5)])
    print('gradient_gate',law,'max_abs',float(np.max(np.abs(jac-fd))),
          'max_tolerance_fraction',float(np.max(np.abs(jac-fd)/(1e-7+1e-6*np.abs(fd)))))
    assert np.all(np.abs(jac-fd)<=1e-7+1e-6*np.abs(fd))


def test_reference_radius_conversion_and_legacy_difference():
    # Equality of old/new physical a/Rs models at k_ref, including WL conversion.
    p=parameters(KREF);duration=dict(p,duration=jnp.array([D]));duration.pop('a_rs')
    np.testing.assert_allclose(harmonica_a_rs_from_duration(P,D,B,KREF),A,atol=1e-12,rtol=0)
    np.testing.assert_allclose(compute_transit_model(duration,T),compute_transit_model(p,T),atol=1e-12,rtol=0)
    # A linear TransitOrbit is not an exact circular orbit even at k_ref.
    # Thus the user's requested old-linear WL equality cannot coexist with
    # their exact-orbit requirement. Keep the discrepancy visible.
    delta=np.max(np.abs(np.asarray(_compute_transit_model_duration(duration,T))-
                        np.asarray(compute_transit_model(p,T))))
    print('legacy_linear_white_max_flux_difference',delta)
    assert delta>1e-9


def test_builder_shares_orbit_across_channels_and_joint_duration():
    for joint in (False,True):
        model=create_vectorized_model(detrend_type='none',ld_mode='fixed',joint_geometry=joint)
        kw=dict(mu_duration=jnp.array([D]),mu_t0=jnp.array([0.]),mu_b=jnp.array([B]),
            rors_reference=jnp.array([KREF]),PERIOD=jnp.array([P]),
            ld_fixed=jnp.array([[.3,.2],[.3,.2]]))
        fixed=dict(rors=jnp.array([[.17],[.195]]))
        if joint:
            kw.update(sigma_duration=jnp.array([.001]),sigma_t0=jnp.array([.001]),sigma_b=jnp.array([.01]))
            fixed.update(duration=jnp.array([D]),t0=jnp.array([0.]),b=jnp.array([B]))
        tr=handlers.trace(handlers.seed(handlers.substitute(model,data=fixed),0)).get_trace(
            jnp.array(T),jnp.full((2,len(T)),1e-4),**kw)
        for lane,k in enumerate((.17,.195)):
            np.testing.assert_allclose(tr['obs']['fn'].loc[lane]-1,
                compute_transit_model(parameters(k),T),atol=1e-12,rtol=0)


def test_expanded_window_and_exact_active_evaluation():
    from models.jaxoplanet.core import build_transit_window_indices
    duration=float(harmonica_duration_from_geometry(P,A,B,np.sqrt(.5)))
    t=np.linspace(-.1,.1,301)
    indices=build_transit_window_indices(t,P,0.,duration)
    for k in (KREF,.195,np.sqrt(.5)):
        p=parameters(k);full=compute_transit_model(p,t)
        window=dict(p,_transit_window_indices=indices)
        np.testing.assert_allclose(compute_transit_model(window,t),full,atol=1e-12,rtol=0)
        np.testing.assert_allclose(compute_transit_model_window(window,t),full[indices],atol=1e-12,rtol=0)


def test_injection_recovery_free_kipping_and_old_failure():
    ktrue=KREF*np.sqrt(1.05);t=np.linspace(-.065,.065,401)
    truth=np.asarray(compute_transit_model(parameters(ktrue),t))
    p=parameters()
    def flux(q,old=False):
        root=jnp.sqrt(q[1]);u=jnp.array([2*root*q[2],root*(1-2*q[2])])
        params=dict(p,rors=q[:1],u=u)
        if old:
            params.pop('a_rs');params['duration']=jnp.array([D])
            return _compute_transit_model_duration(params,t)
        return compute_transit_model(params,t)
    estimates=[]
    for old in (False,True):
        fn=jax.jit(lambda q:flux(q,old));deriv=jax.jit(jax.jacfwd(lambda q:flux(q,old)))
        fit=least_squares(lambda q:np.asarray(fn(q))-truth,[KREF,.25,.3],
            jac=lambda q:np.asarray(deriv(q)),bounds=([.1,0.,0.],[.3,1.,1.]),
            xtol=1e-14,ftol=1e-14,gtol=1e-14)
        estimates.append((fit.x[0]**2-ktrue**2)/(ktrue**2-KREF**2))
    print('depth_change_relative_bias_new_old',estimates)
    assert abs(estimates[0])<.02
    assert abs(estimates[1])>.02


@pytest.mark.parametrize('ecc,omega',[(.05,.7),(.1,-.4)])
def test_eccentric_duration_is_exact_contact_interval_and_has_implicit_gradients(ecc,omega):
    from scipy.optimize import brentq
    from models.orbital_geometry import eccentric_duration
    duration=harmonica_duration_from_geometry(P,A,B,KREF,ecc=ecc,omega=omega)
    recovered=harmonica_a_rs_from_duration(P,duration,B,KREF,ecc=ecc,omega=omega)
    np.testing.assert_allclose(recovered,A,atol=1e-9,rtol=0)
    # Independent NumPy Kepler solve (no shared contact-equation code).
    f0=np.pi/2-omega
    E0=2*np.arctan2(np.sqrt(1-ecc)*np.sin(f0/2),np.sqrt(1+ecc)*np.cos(f0/2))
    M0=E0-ecc*np.sin(E0)
    cosi=B*(1+ecc*np.sin(omega))/(A*(1-ecc*ecc))
    def numpy_distance(time):
        M=M0+2*np.pi*time/P
        E=brentq(lambda x:x-ecc*np.sin(x)-M,M-ecc-1e-12,M+ecc+1e-12,xtol=1e-14)
        f=2*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2),np.sqrt(1-ecc)*np.cos(E/2))
        r=A*(1-ecc*np.cos(E))
        return r*np.sqrt(np.cos(f+omega)**2+cosi**2*np.sin(f+omega)**2)
    before=brentq(lambda t:numpy_distance(t)-1-KREF,-.1,0.,xtol=1e-14)
    after=brentq(lambda t:numpy_distance(t)-1-KREF,0.,.1,xtol=1e-14)
    np.testing.assert_allclose(duration,after-before,atol=1e-12,rtol=0)
    q=np.array([P,float(duration),B,KREF,ecc,omega])
    def inverse(q):
        return harmonica_a_rs_from_duration(q[0],q[1],q[2],q[3],ecc=q[4],omega=q[5])
    actual=np.asarray(jax.jit(jax.grad(inverse))(q))
    steps=np.array([1e-5,1e-7,1e-5,1e-5,1e-5,1e-5])
    expected=np.array([(float(inverse(q+np.eye(6)[i]*h))-float(inverse(q-np.eye(6)[i]*h)))/(2*h)
                       for i,h in enumerate(steps)])
    print('eccentric_inverse_gradient_gate',ecc,omega,'max_abs',float(np.max(np.abs(actual-expected))),
          'max_tolerance_fraction',float(np.max(np.abs(actual-expected)/(1e-7+1e-6*np.abs(expected)))))
    assert np.all(np.abs(actual-expected)<=1e-7+1e-6*np.abs(expected))


def test_missing_reference_is_rejected_instead_of_using_channel_k():
    model=create_vectorized_model(detrend_type='none',ld_mode='fixed')
    with pytest.raises(ValueError,match='rors_reference'):
        handlers.trace(handlers.seed(model,0)).get_trace(jnp.array(T),jnp.full((1,len(T)),1e-4),
            mu_duration=jnp.array([D]),mu_t0=jnp.array([0.]),mu_b=jnp.array([B]),
            PERIOD=jnp.array([P]),ld_fixed=jnp.array([[.3,.2]]))


def test_harmonica_channels_use_one_physical_orbit(monkeypatch):
    """Backend-independent wiring check; native FFI parity is in suite tests."""
    import models.harmonica.core as core
    from models.harmonica.builder import create_vectorized_model as harmonica_builder
    def backend(t,t0,period,a_rs,inc,ecc,omega,*,u1,u2,r):
        # A spherical Harmonica silhouette must use the same orbit when r[0]
        # changes. An independent exact projected-distance expression exposes
        # accidental dependence of a/Rs or inclination on the channel radius.
        from models.jaxoplanet.quadratic_safe import light_curve
        def lane(time,first,second,radius):
            phase=2*jnp.pi*(time-t0)/period
            z=jnp.sqrt(a_rs*a_rs*jnp.sin(phase)**2+(a_rs*jnp.cos(inc))**2*jnp.cos(phase)**2)
            return 1+light_curve(jnp.stack([first.reshape(()),second.reshape(())]),z,radius.reshape(-1)[0])
        times = jnp.broadcast_to(t, (len(u1), jnp.shape(t)[-1]))
        return jax.vmap(lane)(times,u1,u2,r)
    monkeypatch.setattr(core,'harmonica_transit_quad_ld',backend)
    model=harmonica_builder(detrend_type='none',ld_mode='fixed',ld_profile='quadratic')
    fixed=dict(rors=jnp.array([[.17],[.195]]))
    trace=handlers.trace(handlers.seed(handlers.substitute(model,data=fixed),0)).get_trace(
        jnp.array(T),jnp.full((2,len(T)),1e-4),PERIOD=jnp.array([P]),
        mu_t0=jnp.array([0.]),mu_b=jnp.array([B]),harmonica_a_rs=jnp.array([A]),
        ld_fixed=jnp.array([[.3,.2],[.3,.2]]))
    for lane,k in enumerate((.17,.195)):
        np.testing.assert_allclose(trace['obs']['fn'].loc[lane]-1,
            compute_transit_model(parameters(k),T),atol=1e-12,rtol=0)

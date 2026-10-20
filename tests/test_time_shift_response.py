"""Independent validation of post-fit half-disk coefficient products."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from koala.time_shift_response import (
    projected_position, half_radius_responses, solve_response, fitted_bin_response)
from models.jaxoplanet.core import compute_transit_model, orbit_separation
from models.jaxoplanet.builder import create_vectorized_model

jax.config.update('jax_enable_x64', True)


def params(orbit, radius, u):
    period,t0,a,b,e,w=orbit
    return dict(period=jnp.array([period]),t0=jnp.array([t0]),a_rs=jnp.array([a]),
                b=jnp.array([b]),ecc=jnp.array([e]),omega=jnp.array([w]),
                rors=jnp.array([radius]),u=jnp.asarray(u),_jaxoplanet_kernel='quadratic_contact_safe')


def halves(t, orbit, r, u, **kwargs):
    p,v=jax.jvp(lambda tt:projected_position(tt,*orbit)[0],(jnp.asarray(t),),(jnp.ones(len(t)),))
    return half_radius_responses(p,v,r,u,foreground=projected_position(t,*orbit)[1],**kwargs)


@pytest.mark.parametrize('b,e,w',[(0.,0.,0.),(.35,0.,0.),(1.04,0.,0.),(.6,.43,.8)])
def test_boundary_sum_is_autodiff_and_orbit_matches(b,e,w):
    orbit=(4.,0.,9.,b,e,w);r=.12;u=[.3,.2]
    t=jnp.linspace(-.13,.13,431)
    h=halves(t,orbit,r,u)
    d=np.asarray(jax.jacfwd(lambda rr:compute_transit_model(params(orbit,rr,u),t))(r))
    assert np.linalg.norm(h.sum(1)-d)/np.linalg.norm(d)<1e-9
    p,fg=projected_position(t,*orbit)
    z,fg0=orbit_separation(t,*orbit)
    np.testing.assert_allclose(np.linalg.norm(p,axis=1),z,rtol=2e-14,atol=1e-14)
    np.testing.assert_array_equal(fg,fg0)


def test_symmetric_change_and_leading_sign_with_weighted_nuisances():
    orbit=(4.,0.,9.,.3,0.,0.);r=.1;t=jnp.linspace(-.13,.13,501)
    def model(x):
        return (1+compute_transit_model(params((4.,x[1],9.,.3,0.,0.),x[0],[.3,.2]),t))*(x[2]+x[3]*t)
    theta=jnp.array([r,0.,1.,.003]);jac=jax.jacfwd(model)(theta)
    h=halves(t,orbit,r,[.3,.2])*(1+.003*np.asarray(t))[:,None]
    err=.0002*(1+.4*np.sin(np.arange(len(t))))
    keep=np.arange(len(t))%7!=0
    result=solve_response(np.asarray(jac)[keep],h[keep],err[keep])
    assert result[1,0]<0<result[1,1]
    assert abs(result[1].sum()/result[1,1])<1e-9
    np.testing.assert_allclose(result[0].sum(),1.,rtol=1e-9)


def test_uniform_central_small_planet_analytic_limit():
    # C is the small-planet, straight stellar-limb continuum limit. Sample
    # ingress/egress finely at their actual times, with no time averaging.
    r=.0002;period=4.;a=90.
    center=np.arcsin(1/a)*period/(2*np.pi)
    dt=r/(2*np.pi*a/period)
    t=jnp.asarray(np.concatenate([np.linspace(-center-1.05*dt,-center+1.05*dt,2001),
                                 np.linspace(center-1.05*dt,center+1.05*dt,2001)]))
    orbit=(period,0.,a,0.,0.,0.)
    def model(x):return compute_transit_model(params((period,x[1],a,0.,0.,0.),x[0],[0.,0.]),t)
    jac=jax.jacfwd(model)(jnp.array([r,0.]))
    response=solve_response(jac,halves(t,orbit,r,[0.,0.]),np.ones(len(t)))
    velocity=2*np.pi*np.sqrt(a*a-1)/period
    c=response[1,1]*velocity
    print('uniform central C:',float(c),'analytic:',float(3*(np.pi**2+4)/64))
    np.testing.assert_allclose(c,3*(np.pi**2+4)/64,rtol=2e-6)


@pytest.mark.parametrize('b,e,w',[(1.04,0.,0.),(.55,.43,.8)])
def test_independent_2d_nonlinear_refits(b,e,w):
    from reference_half_disk import two_half_flux
    from scipy.optimize import least_squares
    orbit=(4.,0.,9.,b,e,w);r=.12;u=[.3,.2]
    t=np.linspace(-.14,.14,351);keep=np.arange(len(t))%9!=0
    t=jnp.asarray(t[keep]);x=np.asarray(t)/.28
    err=.0002*(1+.3*np.cos(np.arange(len(t))*.7))
    def model(theta):
        return (1+compute_transit_model(params((4.,theta[1]/86400,9.,b,e,w),theta[0],[.3,theta[2]]),t))*(theta[3]+theta[4]*x)
    theta=np.array([r,0.,.2,1.,.001]);signal=jax.jit(model);derivative=jax.jit(jax.jacfwd(model))
    jac=np.asarray(derivative(theta));h=halves(t,orbit,r,u)*(1+.001*x)[:,None]
    response=solve_response(jac,h,err)
    def refit(rl,rt):
        flux=two_half_flux(np.asarray(t),orbit,rl,rt,u)*(1+.001*x)
        fit=least_squares(lambda z:(np.asarray(signal(z))-flux)/err,theta,
                          jac=lambda z:np.asarray(derivative(z))/err[:,None],
                          xtol=1e-12,ftol=1e-12,gtol=1e-10,x_scale='jac')
        assert fit.success
        return np.array([fit.x[0]**2,fit.x[1]])
    epsilon=2e-5
    for limb in range(2):
        radii=np.array([r,r]);delta=np.zeros(2);delta[limb]=epsilon
        fd=(refit(*(radii+delta))-refit(*(radii-delta)))/(2*epsilon)
        expected=[2*r*response[0,limb],response[1,limb]]
        print('independent refit:',dict(b=b,ecc=e,omega=w,limb=limb,relative_error=(fd/expected-1).tolist()))
        np.testing.assert_allclose(fd,expected,rtol=3e-4,atol=1e-7)


@pytest.mark.parametrize('ld_mode',['fixed','fixed_u1','gaussian','uniform'])
def test_replayed_fit_includes_original_priors_and_masks(ld_mode):
    model=create_vectorized_model(detrend_type='linear',ld_mode=ld_mode,param_method='a_rs',
        jaxoplanet_kernel='quadratic_contact_safe',ld_uniform_basis='coefficients',
        surface_config={'model':'transit','spectro_free_t0':True})
    t=np.linspace(-.13,.13,301)
    kw=dict(mu_t0=jnp.array([0.]),mu_b=jnp.array([.35]),mu_a_rs=jnp.array([9.]),PERIOD=jnp.array([4.]),
            ld_fixed=jnp.array([[.3,.2]]),mu_u_ld=jnp.array([[.3,.2]]),
            likelihood_mask=jnp.array([np.arange(len(t))%7!=0]))
    s=dict(rors=np.full((3,1,1),.12),dt0=np.zeros((3,1,1)),u=np.tile([.3,.2],(3,1,1)),
           u2=np.full((3,1),.2),c=np.ones((3,1)),v=np.full((3,1),.001),log_jitter=np.full((3,1),-10.))
    result=fitted_bin_response(model,t,np.full((1,len(t)),.0002),s,kw,kernel='quadratic_contact_safe')[0]
    assert result['d_dt0_dRL_s_per_Rstar']<0<result['d_dt0_dRT_s_per_Rstar']
    assert result['n_retained']==int(np.sum(kw['likelihood_mask']))
    assert result['radius_response_relative_error']<1e-9
    assert result['antisymmetry_check']<1e-9


@pytest.mark.parametrize('alpha',[.2,.6,2.])
def test_power2_boundary_sum(alpha):
    orbit=(4.,0.,9.,.4,.3,.7);r=.12;c=.4;t=jnp.linspace(-.12,.12,371)
    def flux(radius):
        p=params(orbit,radius,[0.,0.]);p.update(c1=c,c2=alpha,_power2_exact=True,_ld_profile='power2',_jaxoplanet_kernel='auto')
        return compute_transit_model(p,t)
    d=np.asarray(jax.jacfwd(flux)(r))
    h=halves(t,orbit,r,[c,alpha],law='power2')
    assert np.linalg.norm(h.sum(1)-d)/np.linalg.norm(d)<1e-9


@pytest.mark.parametrize('basis',['elapsed_days','batman_normalized'])
def test_marginalized_gaussian_trend_uses_original_prior(basis):
    model=create_vectorized_model(detrend_type='linear',ld_mode='fixed',param_method='a_rs',
        trend_mode='gaussian_marginalized',trend_basis=basis,jaxoplanet_kernel='quadratic_contact_safe',
        surface_config={'model':'transit','spectro_free_t0':True})
    t=jnp.linspace(-.13,.13,251);r=.12
    kw=dict(mu_t0=jnp.array([0.]),mu_b=jnp.array([.35]),mu_a_rs=jnp.array([9.]),PERIOD=jnp.array([4.]),
            ld_fixed=jnp.array([[.3,.2]]),trend_prior_mean=jnp.array([[1.,0.]]),
            trend_prior_scale=jnp.array([[.01,.003]]),_response_normalized_trend=basis=='batman_normalized')
    s=dict(rors=np.full((3,1,1),r),dt0=np.zeros((3,1,1)),c=np.ones((3,1)),
           v=np.zeros((3,1)),log_jitter=np.full((3,1),-10.))
    result=fitted_bin_response(model,t,np.full((1,len(t)),.0002),s,kw,kernel='quadratic_contact_safe',
                              detrend_type='linear',marginalized=True)[0]
    # Independently construct the four physical-coordinate columns and Gaussian precision.
    def explicit(theta):
        tr=compute_transit_model(params((4.,theta[1],9.,.35,0.,0.),theta[0],[.3,.2]),t)
        return (1+tr)*(theta[2]+theta[3]*(t-t.min()))
    jac=np.asarray(jax.jacfwd(explicit)(jnp.array([r,0.,1.,0.])))
    transform=np.eye(2)
    if basis=='batman_normalized':
        # normalized intercept = elapsed intercept + mean(elapsed)*slope
        transform=np.array([[1.,float(jnp.mean(t-t.min()))],[0.,float(jnp.ptp(t))]])
    p=np.zeros((4,4));p[2:,2:]=transform.T@np.diag(1/np.array([.01,.003])**2)@transform
    h=halves(t,(4.,0.,9.,.35,0.,0.),r,[.3,.2])
    sigma=np.full(len(t),np.hypot(.0002,np.exp(-10.)))
    # A direct unscaled normal solve is an independent check of the replay adapter.
    w=1/sigma**2
    expected=np.linalg.solve(jac.T@(w[:,None]*jac)+p,jac.T@(w[:,None]*h))
    np.testing.assert_allclose(result['time_shift_coefficient'],expected[1,1]*86400,rtol=1e-10)
    np.testing.assert_allclose(result['d_depth_dRT'],expected[0,1]*2*r,rtol=1e-10)


def test_cached_replay_does_not_capture_first_bins_mask_or_ld():
    model=create_vectorized_model(detrend_type='none',ld_mode='fixed',param_method='a_rs',
        jaxoplanet_kernel='quadratic_contact_safe',surface_config={'model':'transit','spectro_free_t0':True})
    t=np.linspace(-.13,.13,231)
    kw=dict(mu_t0=jnp.array([0.]),mu_b=jnp.array([.65]),mu_a_rs=jnp.array([9.]),PERIOD=jnp.array([4.]),
            ld_fixed=jnp.array([[.1,.1]]))
    s=dict(rors=np.full((3,1,1),.12),dt0=np.zeros((3,1,1)),log_jitter=np.full((3,1),-10.))
    error=np.full((1,len(t)),.0002)
    first=fitted_bin_response(model,t,error,s,kw,kernel='quadratic_contact_safe')[0]
    kw.update(ld_fixed=jnp.array([[.7,.1]]),likelihood_mask=jnp.array([np.arange(len(t))%3!=0]))
    second=fitted_bin_response(model,t,error,s,kw,kernel='quadratic_contact_safe')[0]
    assert second['n_retained']<first['n_retained']
    assert abs(first['time_shift_coefficient']/second['time_shift_coefficient']-1)>1e-4


@pytest.mark.parametrize('ld_mode',['fixed','gaussian','uniform'])
def test_replayed_power2_uses_physical_intensity(ld_mode):
    model=create_vectorized_model(detrend_type='linear',ld_profile='power2',ld_mode=ld_mode,
        param_method='a_rs',surface_config={'model':'transit','spectro_free_t0':True})
    t=np.linspace(-.13,.13,251)
    kw=dict(mu_t0=jnp.array([0.]),mu_b=jnp.array([.35]),mu_a_rs=jnp.array([9.]),PERIOD=jnp.array([4.]),
            ld_fixed=jnp.array([[.4,.6]]),mu_u_ld=jnp.array([[.4,.6]]))
    s=dict(rors=np.full((3,1,1),.12),dt0=np.zeros((3,1,1)),c=np.ones((3,1)),
           v=np.zeros((3,1)),log_jitter=np.full((3,1),-10.),c1=np.full((3,1),.4),
           c2=np.full((3,1),.6),ld_coefficients=np.tile([.4,.6],(3,1,1)))
    result=fitted_bin_response(model,t,np.full((1,len(t)),.0002),s,kw,ld_profile='power2')[0]
    assert result['d_dt0_dRL_s_per_Rstar']<0<result['d_dt0_dRT_s_per_Rstar']
    assert result['radius_response_relative_error']<1e-9


@pytest.mark.parametrize('basis',['uplus_uminus','kipping'])
def test_replayed_quadratic_latent_coordinates(basis):
    model=create_vectorized_model(detrend_type='none',ld_mode='uniform',ld_uniform_basis=basis,
        param_method='a_rs',surface_config={'model':'transit','spectro_free_t0':True})
    t=np.linspace(-.13,.13,251)
    kw=dict(mu_t0=jnp.array([0.]),mu_b=jnp.array([.35]),mu_a_rs=jnp.array([9.]),PERIOD=jnp.array([4.]))
    s=dict(rors=np.full((3,1,1),.12),dt0=np.zeros((3,1,1)),log_jitter=np.full((3,1),-10.),
           ld_q=np.tile([.25,.3],(3,1,1)),ld_uplus_uminus=np.tile([.5,.1],(3,1,1)))
    result=fitted_bin_response(model,t,np.full((1,len(t)),.0002),s,kw)[0]
    assert result['d_dt0_dRL_s_per_Rstar']<0<result['d_dt0_dRT_s_per_Rstar']
    assert result['radius_response_relative_error']<1e-9


def test_multiplanet_replay_keeps_each_planets_own_orbit_and_radius():
    model=create_vectorized_model(detrend_type='none',ld_mode='fixed',n_planets=2,
        param_method='a_rs',surface_config={'model':'transit','spectro_free_t0':True})
    t=np.linspace(-.15,.42,401)
    kw=dict(mu_t0=jnp.array([0.,.25]),mu_b=jnp.array([.35,.7]),mu_a_rs=jnp.array([9.,13.]),
            PERIOD=jnp.array([4.,7.]),mu_ecc=jnp.array([0.,.3]),mu_omega=jnp.array([0.,.7]),
            ld_fixed=jnp.array([[.3,.2]]))
    s=dict(rors=np.tile([.12,.08],(3,1,1)),dt0=np.zeros((3,1,2)),log_jitter=np.full((3,1),-10.))
    rows=fitted_bin_response(model,t,np.full((1,len(t)),.0002),s,kw)
    assert len(rows)==2
    for row,r in zip(rows,[.12,.08]):
        assert row['d_dt0_dRL_s_per_Rstar']<0<row['d_dt0_dRT_s_per_Rstar']
        assert row['antisymmetry_check']<1e-9
        np.testing.assert_allclose(row['d_depth_dRL']+row['d_depth_dRT'],2*r,rtol=1e-9)
    assert rows[0]['time_shift_coefficient']!=rows[1]['time_shift_coefficient']

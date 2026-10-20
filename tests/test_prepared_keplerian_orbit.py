"""Prepared separations use the same exact orbit and flux as direct calls."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet import core

@pytest.mark.parametrize('ecc', [0., .05, .1])
@pytest.mark.parametrize('power2', [False, True])
def test_prepared_values_radius_ld_gradients_and_windows(ecc, power2):
    t=jnp.linspace(-.075,.075,151)
    z=core.fixed_orbit_separation(t,3.325271,0.,13.6,.45,ecc,60.)
    params=dict(period=jnp.array([3.325271]),t0=jnp.array([0.]),a_rs=jnp.array([13.6]),
                b=jnp.array([.45]),ecc=jnp.array([ecc]),omega=jnp.array([jnp.pi/3]))
    def flux(q, prepared, window):
        p=dict(params,rors=q[:1],u=q[1:])
        if power2:p.update(_power2_exact=True,c1=q[1],c2=q[2],u=jnp.zeros(12))
        if prepared:p['_transit_separation']=z[None,:]
        if window:p['_transit_window_indices']=jnp.flatnonzero(z<1.7072)
        return core.compute_transit_model(p,t)
    q=jnp.array([.18,.4,.2]) if not power2 else jnp.array([.18,.6,.7])
    weights=jnp.cos(t*13.)
    reference=jax.jit(lambda q:flux(q,False,False))(q)
    gradient=jax.jit(jax.grad(lambda q:jnp.sum(weights*flux(q,False,False))))(q)
    for window in [False,True]:
        np.testing.assert_allclose(flux(q,True,window),reference,atol=1e-12,rtol=0)
        actual=jax.grad(lambda q:jnp.sum(weights*flux(q,True,window)))(q)
        np.testing.assert_allclose(actual,gradient,atol=1e-7,rtol=1e-6)
    p=dict(params,rors=q[:1],u=jnp.array([.4,.2]),_transit_separation=z[None,:],_transit_window_indices=jnp.array([],dtype=int))
    np.testing.assert_array_equal(core.compute_transit_model(p,t),jnp.zeros_like(t))

def test_prepared_far_side_and_shape_guard():
    t=jnp.array([0.,1.65]);z=core.circular_orbit_separation(t,3.3,0.,13.6,.45)
    assert z[1]==1e3
    p=dict(rors=jnp.array([.18]),u=jnp.array([.4,.2]),_transit_separation=z[None,:])
    assert core.compute_transit_model(p,t)[1]==0.
    with pytest.raises(ValueError,match='shape'):
        core.compute_transit_model(dict(p,_transit_separation=z[:1]),t)


def test_fixed_kepler_orbit_matches_independent_uniform_flux():
    from scipy.optimize import brentq
    from models.jaxoplanet.power2_exact import uniform_overlap
    t=np.linspace(-.065,.065,301);period=3.325271;a=13.606882266688462
    b=.4473242676414512;k=.18187297632416788
    for e in [0.,.05,.1]:
        for omega in [0.,45.,90.,150.,270.]:
            w=np.deg2rad(omega);f0=np.pi/2-w
            E0=2*np.arctan2(np.sqrt(1-e)*np.sin(f0/2),np.sqrt(1+e)*np.cos(f0/2))
            mean=(E0-e*np.sin(E0)+2*np.pi*t/period+np.pi)%(2*np.pi)-np.pi
            E=np.array([brentq(lambda E:E-e*np.sin(E)-m,-np.pi,np.pi,xtol=1e-14) for m in mean])
            # Independent eccentric-anomaly Cartesian coordinates avoid the
            # production true-anomaly/radial projection and Newton solver.
            xp=a*(np.cos(E)-e);yp=a*np.sqrt(1-e*e)*np.sin(E)
            ci=b*(1+e*np.sin(w))/(a*(1-e*e))
            z=np.sqrt((xp*np.cos(w)-yp*np.sin(w))**2+ci**2*(xp*np.sin(w)+yp*np.cos(w))**2)
            actual=core.fixed_orbit_separation(t,period,0.,a,b,e,omega)
            np.testing.assert_allclose(actual,z,atol=1e-11,rtol=0)
            alpha=np.arccos(np.clip((z*z+k*k-1)/(2*z*k),-1,1))
            beta=np.arccos(np.clip((z*z+1-k*k)/(2*z),-1,1))
            radicand=np.maximum(0.,(-z+k+1)*(z+k-1)*(z-k+1)*(z+k+1))
            area=np.where(z>=1+k,0.,np.where(z<=1-k,np.pi*k*k,k*k*alpha+beta-.5*np.sqrt(radicand)))
            flux=1-np.asarray(uniform_overlap(jnp.array(actual),k))/np.pi
            np.testing.assert_allclose(flux,1-area/np.pi,atol=1e-9,rtol=0)


"""Exact contact durations for a radius-independent Keplerian orbit.

Angles are radians. b is the projected conjunction distance, consistent with
batman and jaxoplanet (conjunction is f=pi/2-omega).
"""
import jax
import jax.numpy as jnp


def _contact_equation(delta, a, b, k, e, w):
    f = .5*jnp.pi-w+delta
    radius = a*(1-e*e)/(1+e*jnp.cos(f))
    cosi = b*(1+e*jnp.sin(w))/(a*(1-e*e))
    return radius**2*(jnp.sin(delta)**2+cosi**2*jnp.cos(delta)**2)-(1+k)**2


@jax.custom_jvp
def _contact(a, b, k, e, w, side):
    # Each branch brackets one contact around inferior conjunction. The
    # derivative comes from the contact equation, not bisection decisions.
    lo = jnp.zeros_like(a+b+k+e+w)
    hi = jnp.full_like(lo, .5*jnp.pi)
    def step(_, bounds):
        low, high = bounds
        mid = (low+high)*.5
        outside = _contact_equation(side*mid,a,b,k,e,w)>0
        return jnp.where(outside,low,mid), jnp.where(outside,mid,high)
    lo, hi = jax.lax.fori_loop(0,60,step,(lo,hi))
    return side*(lo+hi)*.5


@_contact.defjvp
def _contact_jvp(primals,tangents):
    a,b,k,e,w,side=primals
    root=_contact(*primals)
    args=(a,b,k,e,w)
    _, numerator=jax.jvp(lambda *q:_contact_equation(root,*q),args,tangents[:5])
    _, denominator=jax.jvp(lambda d:_contact_equation(d,*args),(root,),(jnp.ones_like(root),))
    return root,-numerator/denominator


def eccentric_duration(period,a,b,k,e,w):
    before=_contact(a,b,k,e,w,-1.)
    after=_contact(a,b,k,e,w,1.)
    def mean(delta):
        f=.5*jnp.pi-w+delta
        E=2*jnp.arctan2(jnp.sqrt(1-e)*jnp.sin(f/2),jnp.sqrt(1+e)*jnp.cos(f/2))
        return E-e*jnp.sin(E)
    return period/(2*jnp.pi)*jnp.mod(mean(after)-mean(before),2*jnp.pi)


@jax.custom_jvp
def eccentric_a_from_duration(period,duration,b,k,e,w):
    shape=jnp.broadcast_arrays(period,duration,b,k,e,w)[0]
    fac=(1-e*e)/(1+e*jnp.sin(w))
    lower=jnp.maximum(jnp.maximum(1+k,b/fac),1.)*(1+1e-12)
    upper=jnp.full_like(shape,1e6)
    def step(_,bounds):
        lo,hi=bounds
        mid=.5*(lo+hi)
        longer=eccentric_duration(period,mid,b,k,e,w)>duration
        return jnp.where(longer,mid,lo),jnp.where(longer,hi,mid)
    lo,hi=jax.lax.fori_loop(0,60,step,(lower+jnp.zeros_like(shape),upper))
    return .5*(lo+hi)


@eccentric_a_from_duration.defjvp
def _a_jvp(primals,tangents):
    p,d,b,k,e,w=primals
    a=eccentric_a_from_duration(*primals)
    _,numerator=jax.jvp(lambda pp,dd,bb,kk,ee,ww:eccentric_duration(pp,a,bb,kk,ee,ww)-dd,
                        primals,tangents)
    _,denominator=jax.jvp(lambda aa:eccentric_duration(p,aa,b,k,e,w),
                          (a,),(jnp.ones_like(a),))
    return a,-numerator/denominator

"""Closed-form quadratic flux and analytic local partials.

The returned signal is normalized stellar flux minus one, evaluated directly
from the occulted disk moments. Float64 is required for numerical accuracy;
callers must enable JAX x64. The fixed eight-step CEL evaluation
is checked against twelve steps and independent numerical disk integrals.
Exact central, coincident and contact limits are evaluated analytically,
without shifting the geometry or snapping nearby points to a contact.

The CEL iteration and mu moment are adapted from exoplanet-core's ellip.h
and limbdark.h at commit 156cca94f455541f3d37d04af97479798022937b, under
the following license.

MIT License

Copyright (c) 2018, 2019, 2020, 2021 Dan Foreman-Mackey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
from jaxoplanet.core.limb_dark import kappas

from .power2_exact import uniform_overlap


def _cel3(kc, p, a, b, *, iterations):
    """Three positive-p Bulirsch CEL integrals sharing one modulus iteration."""
    p = jnp.sqrt(p)
    b = b/p
    ee = kc
    m = jnp.ones_like(kc)
    p1 = jnp.ones_like(kc)
    a2 = jnp.ones_like(kc)
    a3 = jnp.ones_like(kc)
    b2 = kc*kc
    b3 = jnp.zeros_like(kc)
    # The first duplication uses ee=kc; later ones use 2*sqrt(ee).
    for i in range(iterations):
        if i:
            kc = 2*jnp.sqrt(ee)
            ee = kc*m
        g, g1 = ee/p, ee/p1
        a, b = a+b/p, 2*(b+a*g)
        a2, b2 = a2+b2/p1, 2*(b2+a2*g1)
        a3, b3 = a3+b3/p1, 2*(b3+a3*g1)
        p, p1 = p+g, p1+g1
        m = m+kc
    common = .5*jnp.pi/(m*(m+p1))
    return (.5*jnp.pi*(a*m+b)/(m*(m+p)),
            common*(a2*m+b2), common*(a3*m+b3))


def _mu_moment(b, r, *, iterations):
    """Occulted integral of mu and its radius/separation derivatives."""
    central = b == 0.
    coincident = b == r
    contact = b+r == 1.
    special = central | contact
    bb, rr = jnp.where(special, .5, b), jnp.where(special, .1, r)
    bm, bp = bb-rr, bb+rr
    diff = (1+bm)*(1-bm)
    outside = (1+bp)*(1-bp)
    fourbr = 4*bb*rr
    contained = outside > 0.
    kc2 = jnp.where(contained, outside/diff, -outside/fourbr)
    kc = jnp.sqrt(jnp.maximum(kc2, jnp.finfo(jnp.float64).tiny))
    ratio = bm/bp
    mu = 3*ratio/diff
    p = jnp.where(contained, ratio*ratio*outside/diff, bm*bm*kc2)
    aa = jnp.where(contained, 1+mu, 0.)
    zz = jnp.where(contained, p+mu, 3*kc2*bm*bp)
    p = jnp.where(coincident, 1., p)
    aa = jnp.where(coincident, 1., aa)
    zz = jnp.where(coincident, kc2, zz)
    pi, E, C = _cel3(kc, p, aa, zz, iterations=iterations)
    root = jnp.sqrt(diff)
    invbr = 1/jnp.sqrt(bb*rr)
    lam = jnp.where(
        contained,
        2*root*(outside*pi-(4-7*rr*rr-bb*bb)*E)/3,
        diff*(pi+(-3+6*rr*rr+2*bb*rr)*C-fourbr*E)*invbr/3,
    )
    moment = (2*jnp.pi*(rr > bb)+lam)/3
    dr = jnp.where(contained, 4*rr*root*E, 2*rr*diff*C*invbr)
    db = jnp.where(contained, 4*rr*root*(E-2*C)/3,
                   2*rr*diff*(C-2*E)*invbr/3)

    # b=r has p=0 in the general CEL; evaluate its finite analytic limit.
    re = jnp.where(coincident & (r != .5), r, .1)
    m = 4*re*re
    le = jnp.where(re < .5, jnp.pi+2*((2*m-3)*E-m*C)/3,
                   jnp.pi+(-m*E+(2*m-3)*C)/(3*re))
    me = le/3
    dre = jnp.where(re < .5, 4*re*E, 2*C)
    dbe = jnp.where(re < .5, 4*re*(E-2*C)/3, 2*(C-2*E)/3)
    me = jnp.where(r == .5, (jnp.pi-4/3)/3, me)
    dre = jnp.where(r == .5, 2., dre)
    dbe = jnp.where(r == .5, -2/3, dbe)
    moment, dr, db = (jnp.where(coincident, x, y)
                      for x, y in [(me, moment), (dre, dr), (dbe, db)])

    # Exact inner tangency, also for r>1/2: the indicator terms cancel.
    rc = jnp.where(contact, r, .1)
    rootc = jnp.sqrt(rc*(1-rc))
    mc = (2*jnp.arccos(1-2*rc)-4*(3+2*rc-8*rc*rc)*rootc/3)/3
    drc = 8*rc*rootc
    moment, dr, db = (jnp.where(contact, x, y)
                      for x, y in [(mc, moment), (drc, dr), (-drc/3, db)])

    r0 = jnp.where(central, r, .1)
    m0 = -2*jnp.pi*jnp.expm1(1.5*jnp.log1p(-r0*r0))/3
    dr0 = 2*jnp.pi*r0*jnp.sqrt(1-r0*r0)
    return (jnp.where(central, m0, moment), jnp.where(central, dr0, dr),
            jnp.where(central, 0., db))


def partials(u, radius, separation, *, iterations=8):
    """Return signal, d/dr, d/dz, d/du1 and d/du2 at each cadence.

    Radius and separation broadcast in stellar-radius units. The two limb
    coefficients use I(mu)=1-u1*(1-mu)-u2*(1-mu)**2, normalized by its disk
    integral. Radius/separation signs follow the absolute-value geometry.
    ``iterations`` is static and is exposed for convergence checks.
    """
    raw_b, raw_r = jnp.broadcast_arrays(separation, radius)
    b, r = jnp.abs(raw_b), jnp.abs(raw_r)
    full, empty = r >= 1+b, (b >= 1+r) | (r == 0.)
    inactive = full | empty
    b, r = jnp.where(inactive, .5, b), jnp.where(inactive, .1, r)
    kite, k0, _ = kappas(b, r)
    area = uniform_overlap(b, r)
    s2 = 2*r*r*(r*r+2*b*b-1)*k0+.5*kite*(1-5*r*r-b*b)
    m1, m1r, m1b = _mu_moment(b, r, iterations=iterations)
    u1, u2 = u
    norm = jnp.pi*(1-u1/3-u2/6)
    occulted = (1-u1-1.5*u2)*area+(u1+2*u2)*m1+.25*u2*s2
    value = -occulted/norm
    ar = 2*r*k0
    ab = jnp.where(b > 0., -kite/jnp.where(b > 0., b, 1.), 0.)
    sine, cosine = jnp.sin(k0), jnp.cos(k0)
    m2r = 2*r*((1-r*r-b*b)*k0+2*b*r*sine)
    m2b = -2*r*((1-r*r-b*b)*sine+b*r*(k0+sine*cosine))
    dr = -((1-u1-u2)*ar+(u1+2*u2)*m1r-u2*m2r)/norm
    db = -((1-u1-u2)*ab+(u1+2*u2)*m1b-u2*m2b)/norm
    du1 = (area-m1+value*jnp.pi/3)/norm
    du2 = (1.5*area-2*m1-.25*s2+value*jnp.pi/6)/norm
    value = jnp.where(full, -1., jnp.where(empty, 0., value))
    dr *= jnp.where(raw_r < 0., -1., 1.)
    db *= jnp.where(raw_b < 0., -1., 1.)
    return value, *(jnp.where(inactive, 0., a) for a in (dr, db, du1, du2))


@lru_cache(maxsize=None)
def fixed_orbit_evaluator(*, iterations=8):
    """Return the closed form with an analytic-column JVP.

    Higher derivatives differentiate the analytic columns with JAX.
    """
    @jax.custom_jvp
    def evaluate(u, radius, separation):
        return partials(u, radius, separation, iterations=iterations)[0]

    def rule(primals, tangents):
        value, dr, db, du1, du2 = partials(*primals, iterations=iterations)
        u_dot, r_dot, b_dot = tangents
        zero = jax.custom_derivatives.SymbolicZero
        tangent = jnp.zeros_like(value)
        if not isinstance(u_dot, zero):
            tangent += du1*u_dot[0]+du2*u_dot[1]
        if not isinstance(r_dot, zero):
            tangent += dr*r_dot
        if not isinstance(b_dot, zero):
            tangent += db*b_dot
        return value, tangent

    evaluate.defjvp(rule, symbolic_zeros=True)
    return evaluate

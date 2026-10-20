"""Opt-in quadratic kernel with analytic full/no-occultation limits.

The stock kernel evaluates the doubly degenerate b=0,r=1 geometry through
an inactive integral branch, yielding an incorrect flux and NaN derivatives.
Use finite placeholder geometry for inactive integrals, then restore their
analytic values. A sine-squared endpoint map regularizes the S1 integral;
stable disk moments avoid subtracting full stellar flux from visible flux.
"""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.core.limb_dark import kappas
from .power2_exact import occulted_power2_moment, uniform_overlap
from .power2_table import _all_lanes_in_domain


@lru_cache(maxsize=8)
def _endpoint_quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = (nodes+1)/2
    return np.sin(.5*np.pi*x)**2, weights*.25*np.pi*np.sin(np.pi*x)


def _partial_flux(u, b, r, order):
    area, kappa0, kappa1 = kappas(b, r)
    fractions, weights = (jnp.asarray(v) for v in _endpoint_quadrature(order))
    cosine = jnp.cos(kappa0[..., None]*fractions)
    rho2 = jnp.maximum(0., r[..., None]**2+b[..., None]**2
                       -2*b[..., None]*r[..., None]*cosine)
    mu2 = 1-rho2
    mu = jnp.where(mu2 > 0., jnp.sqrt(jnp.where(mu2 > 0., mu2, 1.)), 0.)
    # (1-mu**3)/(1-mu**2) = 1+mu**2/(1+mu), including the pole mu=1.
    ratio = 1+jnp.maximum(0., mu2)/(1+mu)
    p1 = kappa0*jnp.sum(weights*2*r[..., None]*(r[..., None]-b[..., None]*cosine)*ratio/3, axis=-1)
    s2 = (2*r**2*(r**2+2*b**2-1)*kappa0
          +.5*area*(1-5*r**2-b**2))
    overlap = uniform_overlap(b, r)
    u1, u2 = u
    occulted = ((1-u1-1.5*u2)*overlap+(u1+2*u2)*(p1+2*kappa1/3)+.25*u2*s2)
    return -occulted/(jnp.pi*(1-u1/3-u2/6))


def light_curve(u, b, r, *, order=10, equal_disk_fallback=True):
    order = max(int(order), 20)
    b, r = jnp.broadcast_arrays(jnp.abs(b), jnp.abs(r))
    full = r >= 1+b
    empty = b >= 1+r
    inactive = full | empty
    near_equal_disks = ((jnp.abs(r-1) < 1e-5) & (b < 1e-4) & ~inactive
                        if equal_disk_fallback else jnp.zeros_like(inactive))
    def stock(_):
        skip = inactive | near_equal_disks
        signal = _partial_flux(u, jnp.where(skip, .5, b),
                               jnp.where(skip, .1, r), order)
        return jnp.where(full, -1., jnp.where(empty, 0., signal))
    if not equal_disk_fallback:
        return stock(None)
    def fallback(_):
        # I(mu)=(1-u1-u2)+(u1+2u2)*mu-u2*mu**2. Independent
        # disk moments regularize the almost coincident stellar/planet limbs.
        z = jnp.where(near_equal_disks, b, .5)
        radius = jnp.where(near_equal_disks, r, .1)
        u1, u2 = u
        moment = ((1-u1-u2)*uniform_overlap(z, radius)
                  +(u1+2*u2)*occulted_power2_moment(z, radius, 1.)
                  -u2*occulted_power2_moment(z, radius, 2.))
        direct = -moment/(jnp.pi*(1-u1/3-u2/6))
        return jnp.where(near_equal_disks, direct, stock(None))
    return jax.lax.cond(_all_lanes_in_domain(jnp.all(~near_equal_disks)), stock, fallback, None)


@lru_cache(maxsize=2)
def fixed_orbit_evaluator(equal_disk_fallback=True):
    """Return the same quadratic flux with a memory-bounded fixed-orbit JVP.

    Spectroscopy differentiates radius and two LD coefficients while the
    separation is data. A radius JVP plus the linear LD JVP avoids retaining
    the annular integration's full reverse graph for every channel/chain.
    Geometry uses the diagonal separation derivative: each cadence's flux
    depends only on its own separation. All columns are JVPs of the original
    integral, without approximation.
    """
    equal_disk_fallback = bool(equal_disk_fallback)

    def original(u, radius, separation):
        return light_curve(u, separation, radius,
                           equal_disk_fallback=equal_disk_fallback)

    @jax.custom_jvp
    def evaluate(u, radius, separation):
        return original(u, radius, separation)

    def rule(primals, tangents):
        u, radius, separation = primals
        u_dot, radius_dot, separation_dot = tangents
        zero = jax.custom_derivatives.SymbolicZero
        value = None
        tangent = 0.0
        if not isinstance(radius_dot, zero):
            value, column = jax.jvp(
                lambda r: original(u, r, separation),
                (radius,), (jnp.ones_like(radius),))
            tangent = tangent + column * radius_dot
        if not isinstance(separation_dot, zero):
            # The separation Jacobian is diagonal. A unit direction gives
            # every diagonal entry, avoiding a full reverse integral graph.
            primal, column = jax.jvp(
                lambda z: original(u, radius, z),
                (separation,), (jnp.ones_like(separation),))
            value = primal if value is None else value
            tangent = tangent + column * separation_dot
        if not isinstance(u_dot, zero):
            primal, column = jax.jvp(
                lambda coeff: original(coeff, radius, separation),
                (u,), (u_dot,))
            value = primal if value is None else value
            tangent = tangent + column
        if value is None:
            value = original(*primals)
            tangent = jnp.zeros_like(value)
        return value, tangent

    evaluate.defjvp(rule, symbolic_zeros=True)
    return evaluate

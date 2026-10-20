"""Direct disk integration of the power-2 law, without a polynomial profile.

The fully covered central disk is analytic. The remaining annuli use fixed
Gauss-Legendre quadrature with a fourth-order endpoint map. This regularizes
both the circle intersections and the noninteger power at the stellar limb.
Flux has the same convention as jaxoplanet: zero outside transit.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_batching.custom_vmap
def _all_lanes_in_domain(predicate):
    """Make one quadrature/table decision across nested channel/chain maps."""
    return predicate


@_all_lanes_in_domain.def_vmap
def _all_lanes_in_domain_batch(axis_size, in_batched, predicate):
    del axis_size
    # Re-enter to reduce any outer vmap too, so cond never becomes a select
    # that evaluates both the fast integral and its fallback for every lane.
    return (_all_lanes_in_domain(jnp.all(predicate)) if in_batched[0] else predicate), False


@lru_cache(maxsize=None)
def _quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    x = (nodes + 1.0) / 2.0
    denominator = x**4 + (1.0 - x)**4
    fraction = x**4 / denominator
    # Store the complementary fraction separately: 1-fraction loses precision
    # at the limb, precisely where small alpha needs accurate log(mu).
    complement = (1.0 - x)**4 / denominator
    weights = weights * 2.0 * x**3 * (1.0 - x)**3 / denominator**2
    return fraction, complement, weights


def occulted_power2_moment(separation, radius, exponent, *, order=128):
    """Return integral(mu**exponent dA) over the overlap of two disks.

    Inputs broadcast; radius and separation are nonnegative stellar units.
    Exponent must be nonnegative. Numerical precision requires JAX x64.
    """
    z, r, alpha = jnp.broadcast_arrays(
        jnp.abs(jnp.asarray(separation)), jnp.abs(jnp.asarray(radius)),
        jnp.asarray(exponent),
    )
    active = (z < 1.0 + r) & (r > 0.0)
    full_star = r >= 1.0 + z
    central = z == 0.0
    partial = active & ~full_star & ~central
    # Keep unused branches differentiable and finite under vmap/where.
    zz = jnp.where(partial, z, 0.5)
    rr = jnp.where(partial, r, 0.1)
    # Use the symmetric derivative at coincident lower endpoints. jnp.abs
    # selects +1 at zero, inconsistent with the symmetric max factors below.
    lo = jnp.maximum(zz-rr, 0.)+jnp.maximum(rr-zz, 0.)
    hi = jnp.minimum(1.0, zz + rr)
    span = hi - lo
    fraction, complement, weights = (jnp.asarray(v) for v in _quadrature(order))
    rho = lo[..., None] + span[..., None] * fraction
    # Accurate distance to the stellar limb even when rho rounds to one.
    one_minus_rho = (1.0 - hi)[..., None] + span[..., None] * complement
    mu_squared = one_minus_rho * (1.0 + rho)
    # Heron's factors, evaluated without subtracting almost equal radii.
    lower_distance = span[..., None] * fraction
    upper_distance = (zz + rr - hi)[..., None] + span[..., None] * complement
    # Express both lower-end factors in endpoint distances. At z == r,
    # subtracting z and r from a tiny rho can round a factor to zero and
    # create enormous artificial derivatives of the Heron square root.
    first = ((rho + zz[..., None] + rr[..., None])
             * (lower_distance+2*jnp.maximum(zz-rr, 0.)[..., None]))
    second = upper_distance*(lower_distance+2*jnp.maximum(rr-zz, 0.)[..., None])
    sine_numerator = jnp.sqrt(jnp.maximum(first * second, 1e-300))
    rho_minus_radius = (hi-rr)[..., None] - span[..., None]*complement
    cosine_numerator = rho_minus_radius*(rho+rr[..., None]) + zz[..., None]**2
    angle = jnp.arctan2(sine_numerator, cosine_numerator)
    moment = jnp.sum(
        2.0 * angle * rho * jnp.exp(0.5 * alpha[..., None] * jnp.log(mu_squared))
        * span[..., None] * weights, axis=-1,
    )
    inner_radius = jnp.minimum(1.0, jnp.maximum(0.0, r - z))
    # Avoid log(0) in inactive analytic branches (full star handled below).
    inner_safe = jnp.minimum(inner_radius, 1.0 - jnp.finfo(jnp.float64).eps)
    inner = -2.0 * jnp.pi / (alpha + 2.0) * jnp.expm1(
        (1.0 + alpha / 2.0) * jnp.log1p(-inner_safe**2)
    )
    total = 2.0 * jnp.pi / (alpha + 2.0)
    central_radius = jnp.minimum(r, 1.0 - jnp.finfo(jnp.float64).eps)
    central_moment = -total * jnp.expm1(
        (1.0 + alpha / 2.0) * jnp.log1p(-central_radius**2)
    )
    return jnp.where(full_star, total,
                     jnp.where(central, central_moment,
                               jnp.where(active, inner + moment, 0.0)))


@jax.custom_jvp
def uniform_overlap(separation, radius):
    """Analytic overlapping area, with finite derivatives at exact contacts."""
    z, r = jnp.broadcast_arrays(jnp.abs(jnp.asarray(separation)),
                                jnp.abs(jnp.asarray(radius)))
    partial = (z > jnp.abs(1.0 - r)) & (z < 1.0 + r) & (r > 0.0)
    zz, rr = jnp.where(partial, z, 1.0), jnp.where(partial, r, 0.5)
    product = (1.0 + rr - zz) * (zz + rr - 1.0) * (zz + 1.0 - rr) * (zz + 1.0 + rr)
    chord = jnp.sqrt(jnp.maximum(product, 0.))
    first = jnp.arctan2(chord, zz**2+(1.0-rr)*(1.0+rr))
    second = jnp.arctan2(chord, zz**2+(rr-1.0)*(rr+1.0))
    def segment(angle):
        # angle-sin(angle)*cos(angle) is O(angle**3). Avoid cancellation
        # near external tangency with its rapidly convergent Taylor series.
        squared = angle**2
        polynomial = -4096./6227020800.
        for coefficient in (1024./39916800., -256./362880., 64./5040., -16./120., 4./6.):
            polynomial = coefficient+squared*polynomial
        small = angle**3*polynomial
        return jnp.where(angle < .1, small, angle-jnp.sin(angle)*jnp.cos(angle))
    area = segment(first)+rr**2*segment(second)
    return jnp.where(z >= 1.0 + r, 0.0,
                     jnp.where(partial, area, jnp.pi * jnp.minimum(1.0, r)**2))


@uniform_overlap.defjvp
def _uniform_overlap_jvp(primals, tangents):
    separation, radius = primals
    separation_dot, radius_dot = tangents
    z, r = jnp.broadcast_arrays(jnp.abs(jnp.asarray(separation)), jnp.abs(jnp.asarray(radius)))
    partial = (z > jnp.abs(1-r)) & (z < 1+r) & (r > 0.)
    zz, rr = jnp.where(partial, z, 1.), jnp.where(partial, r, .5)
    product = (1+rr-zz)*(zz+rr-1)*(zz+1-rr)*(zz+1+rr)
    chord = jnp.sqrt(jnp.maximum(product, 0.))
    # Geometric boundary derivatives avoid cancellation of the two acos
    # derivatives and the Heron-root derivative near either tangency.
    dz = jnp.where(partial, -chord/zz, 0.)
    planet_angle = jnp.arctan2(chord, zz**2+rr**2-1.)
    contained = (r < 1.) & (z <= 1-r)
    dr = jnp.where(partial, 2*rr*planet_angle,
                   jnp.where(contained, 2*jnp.pi*r, 0.))
    tangent = (dz*jnp.sign(separation)*separation_dot
               +dr*jnp.sign(radius)*radius_dot)
    return uniform_overlap(separation, radius), tangent


def _light_curve(coefficient, exponent, separation, radius, order):
    area = uniform_overlap(separation, radius)
    moment = occulted_power2_moment(separation, radius, exponent, order=order)
    normalization = jnp.pi * (1.0 - coefficient * exponent / (exponent + 2.0))
    return -((1.0 - coefficient) * area + coefficient * moment) / normalization


@lru_cache(maxsize=None)
def _light_curve_evaluator(order):
    """Transpose pointwise JVP columns instead of the full annular graph.

    Each output depends only on the corresponding broadcast input point.
    A unit JVP therefore gives every diagonal Jacobian entry, including for
    separation when t0 is free. Multiplication by the original tangent lets
    JAX reduce broadcast scalar/channel inputs correctly in reverse mode.
    These are derivatives of the integral itself, without a derivative fit.
    """
    def original(*args):
        return _light_curve(*args, order=order)

    @jax.custom_jvp
    def evaluate(*args):
        return original(*args)

    def rule(primals, tangents):
        value = None
        tangent = 0.0
        for index, dot in enumerate(tangents):
            if isinstance(dot, jax.custom_derivatives.SymbolicZero):
                continue

            def column_fn(x):
                args = list(primals)
                args[index] = x
                return original(*args)

            primal, column = jax.jvp(
                column_fn, (primals[index],), (jnp.ones_like(primals[index]),))
            value = primal if value is None else value
            tangent = tangent + column * dot
        if value is None:
            value = original(*primals)
            tangent = jnp.zeros_like(value)
        return value, tangent

    evaluate.defjvp(rule, symbolic_zeros=True)
    return evaluate


def light_curve(coefficient, exponent, separation, radius, *, order=None):
    """Normalized power-2 transit flux minus one (no intensity projection).

    Automatic order uses 64 nodes in the audited planet/limb-darkening
    domain (0.01 <= |radius| <= 0.3, 0.05 <= exponent <= 1.5, 0 <= c <= 1).
    Other inputs retain 128 nodes. An explicit order always overrides this
    choice. No temporal interpolation or cadence reduction is involved.
    """
    args = coefficient, exponent, separation, radius
    if order is not None:
        return _light_curve_evaluator(int(order))(*args)
    r, a, c = jnp.abs(jnp.asarray(radius)), jnp.asarray(exponent), jnp.asarray(coefficient)
    within = jnp.all((r >= .01) & (r <= .3)) & jnp.all((a >= .05) & (a <= 1.5))
    within = within & jnp.all((c >= 0.) & (c <= 1.))
    return jax.lax.cond(
        _all_lanes_in_domain(within),
        lambda _: _light_curve_evaluator(64)(*args),
        lambda _: _light_curve_evaluator(128)(*args), None)

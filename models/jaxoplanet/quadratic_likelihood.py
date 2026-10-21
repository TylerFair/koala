"""Gaussian likelihood and analytic score for a fixed quadratic orbit.

The sampled vector is radius, u1, u2, epoch, trend coefficients, jitter.
Cadences, observations, trend design and shared orbit are fixed during a
spectroscopic fit. The orbit arithmetic is the same as the ordinary model.
"""
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from .core import orbit_separation
from .quadratic_closed import fixed_orbit_evaluator, partials


def value_and_grad(theta, t, y, variance, mask, design, orbit):
    """Return log likelihood and its score in physical parameter coordinates."""
    radius, u, epoch = theta[0], theta[1:3], theta[3]
    beta, jitter = theta[4:-1], theta[-1]
    period, a_rs, impact, ecc, omega = orbit

    def separation(t0):
        return orbit_separation(t, period, t0, a_rs, impact, ecc, omega)

    (z, front), (z_dot, _) = jax.jvp(
        separation, (epoch,), (jnp.ones_like(epoch),)
    )
    signal, dr, dz, du1, du2 = (
        jnp.where(front, column, 0.)
        for column in partials(u, radius, z)
    )
    flux = 1 + signal
    trend = design @ beta
    model = flux * trend
    total_variance = variance + jitter * jitter
    log_prob = dist.Normal(model, jnp.sqrt(total_variance)).log_prob(y)
    value = jnp.sum(jnp.where(mask, log_prob, 0.))

    residual = y - model
    score = jnp.where(mask, residual / total_variance, 0.)
    transit_score = score * trend
    transit_gradient = jnp.stack([
        jnp.sum(transit_score * dr), jnp.sum(transit_score * du1),
        jnp.sum(transit_score * du2), jnp.sum(transit_score * dz * z_dot),
    ])
    trend_gradient = jnp.sum((score * flux)[:, None] * design, axis=0)
    jitter_gradient = jnp.sum(jnp.where(
        mask, jitter * (residual * residual / total_variance - 1) / total_variance, 0.
    ))
    gradient = jnp.concatenate([
        transit_gradient, trend_gradient, jitter_gradient[None],
    ])
    return value, gradient


def _ordinary(theta, t, y, variance, mask, design, orbit):
    """Unfused expression for derivatives with respect to normally fixed data."""
    period, a_rs, impact, ecc, omega = orbit
    z, front = orbit_separation(t, period, theta[3], a_rs, impact, ecc, omega)
    signal = fixed_orbit_evaluator()(theta[1:3], theta[0], z)
    flux = 1 + jnp.where(front, signal, 0.)
    model = flux * (design @ theta[4:-1])
    log_prob = dist.Normal(model, jnp.sqrt(variance + theta[-1]**2)).log_prob(y)
    return jnp.sum(jnp.where(mask, log_prob, 0.))


@jax.custom_jvp
def log_likelihood(theta, t, y, variance, mask, design, orbit):
    """Fixed-orbit likelihood with a fused first derivative and JAX Hessians."""
    return value_and_grad(theta, t, y, variance, mask, design, orbit)[0]


def _rule(primals, tangents):
    zero = jax.custom_derivatives.SymbolicZero
    value, gradient = value_and_grad(*primals)
    tangent = (jnp.zeros_like(value) if isinstance(tangents[0], zero)
               else jnp.sum(gradient * tangents[0]))
    # Usually only theta is differentiated. Keep the normal model's derivative
    # for callers differentiating observations, trend bases or shared geometry.
    indices = [i for i in range(1, len(primals)) if not isinstance(tangents[i], zero)]
    if indices:
        def fixed_data(*args):
            values = list(primals)
            for i, argument in zip(indices, args):
                values[i] = argument
            return _ordinary(*values)

        tangent += jax.jvp(
            fixed_data, tuple(primals[i] for i in indices),
            tuple(tangents[i] for i in indices),
        )[1]
    return value, tangent


log_likelihood.defjvp(_rule, symbolic_zeros=True)

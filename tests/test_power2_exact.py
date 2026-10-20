"""Independent integration, limiting cases, and derivative checks for power-2."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from models.jaxoplanet.power2_exact import light_curve, occulted_power2_moment

jax.config.update("jax_enable_x64", True)


def numerical_flux(z, r, c, alpha):
    """Independent adaptive annular integral (no endpoint map or JAX code)."""
    if r == 0.0 or z >= 1.0 + r:
        return 0.0
    total = np.pi * (1.0 - c * alpha / (alpha + 2.0))
    if r >= 1.0 + z:
        return -1.0
    def intensity(rho):
        return 1.0 - c + c * max(0.0, 1.0 - rho*rho)**(alpha / 2.0)
    inner_radius = min(1.0, max(0.0, r - z))
    inner = quad(lambda rho: 2*np.pi*rho*intensity(rho), 0, inner_radius,
                 epsabs=2e-13, epsrel=2e-12)[0]
    if z == 0:
        return -inner / total
    lo, hi = abs(z - r), min(1.0, z + r)
    def annulus(rho):
        angle = np.arccos(np.clip((rho*rho + z*z - r*r)/(2*rho*z), -1, 1))
        return 2*angle*rho*intensity(rho)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        partial = quad(annulus, lo, hi, epsabs=2e-13, epsrel=2e-12, limit=500)[0]
    return -(inner + partial) / total


def test_randomized_disk_integration():
    rng = np.random.default_rng(265)
    radius = np.exp(rng.uniform(np.log(.001), np.log(2.0), 180))
    z = rng.uniform(0, 1.0 + radius)
    c = rng.uniform(0.0, 1.0, len(z))
    alpha = np.exp(rng.uniform(np.log(.001), np.log(10.0), len(z)))
    parameters = np.column_stack((z, radius, c, alpha))
    actual = jax.jit(jax.vmap(lambda p: light_curve(p[2], p[3], p[0], p[1])))(parameters)
    expected = np.array([numerical_flux(*p) for p in parameters])
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize("radius", [.001, .05, .146, .5, 1.0, 1.2, 2.0])
def test_contacts_and_limiting_cases(radius):
    z = np.array([0.0, radius, abs(1-radius), 1+radius,
                  abs(1-radius)-1e-8, abs(1-radius)+1e-8,
                  1+radius-1e-8, 1+radius+1e-8])
    z = np.maximum(z, 0)
    for c, alpha in [(0., .2), (.6, .6), (.95, .001), (1., 0.), (1., 1.), (1., 10.)]:
        actual = jax.jit(lambda z: light_curve(c, alpha, z, radius))(z)
        expected = [numerical_flux(zz, radius, c, alpha) for zz in z]
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
        jac = jax.jit(jax.vmap(jax.jacfwd(lambda p: light_curve(p[2], p[3], p[0], p[1]))))(
            np.column_stack((z, np.full_like(z, radius), np.full_like(z, c), np.full_like(z, alpha))))
        assert np.all(np.isfinite(jac))


def test_gradients_against_independent_finite_differences():
    rng = np.random.default_rng(84)
    parameters = np.column_stack((rng.uniform(.2, 1.0, 18), rng.uniform(.02, .4, 18),
                                  rng.uniform(.1, .9, 18), rng.uniform(.01, 3., 18)))
    actual = jax.jit(jax.vmap(jax.jacfwd(lambda p: light_curve(p[2], p[3], p[0], p[1]))))(parameters)
    expected = np.zeros_like(parameters)
    step = 2e-5
    for i, p in enumerate(parameters):
        for j in range(4):
            d = np.eye(4)[j] * step
            expected[i, j] = (numerical_flux(*(p-2*d))-8*numerical_flux(*(p-d))
                               +8*numerical_flux(*(p+d))-numerical_flux(*(p+2*d)))/(12*step)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    assert float(jax.grad(lambda z: light_curve(.7, .4, z, .146))(0.)) == 0.0


def test_quadrature_order_convergence():
    z = jnp.linspace(.45, 1.146, 311)
    for alpha in (.0, .001, .02, .1, .6, 1., 3., 10.):
        low = jax.jit(lambda z: occulted_power2_moment(z, .146, alpha, order=128))(z)
        high = jax.jit(lambda z: occulted_power2_moment(z, .146, alpha, order=256))(z)
        np.testing.assert_allclose(low, high, atol=1e-10, rtol=0)


def test_equal_separation_radius_has_physical_derivatives_at_tiny_annuli():
    points = jnp.array([[r, r, 1., alpha] for r in (.001, .1, .5, 1., 1.2, 2.)
                        for alpha in (0., .001, .1, 1., 10.)])
    def evaluate(p):
        return light_curve(p[2], p[3], p[0], p[1], order=512)
    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(evaluate)))(points)
    assert np.isfinite(values).all() and np.isfinite(derivatives).all()
    # A finite-only check misses cancellation artifacts of order 1e100.
    assert np.max(np.abs(derivatives)) < 5
    point = np.array([2., 2., 1., 10.])
    # Independent shape derivatives integrate intensity around the planet
    # boundary. Annular finite differences lose accuracy at the tiny central
    # disk that appears on either side of coincident radius/separation.
    z, r, c, alpha = point
    angle = np.arccos((z*z+r*r-1)/(2*z*r))
    intensity = lambda theta:1-c+c*max(0., 1-z*z-r*r+2*z*r*np.cos(theta))**(alpha/2)
    normalization = np.pi*(1-c*alpha/(alpha+2))
    dz = 2*r/normalization*quad(lambda theta:intensity(theta)*np.cos(theta), 0., angle,
                               epsabs=1e-13, epsrel=1e-13)[0]
    dr = -2*r/normalization*quad(intensity, 0., angle, epsabs=1e-13, epsrel=1e-13)[0]
    step = 2e-5
    numerical = [dz, dr]+[(numerical_flux(*(point+np.eye(4)[k]*step))
                          -numerical_flux(*(point-np.eye(4)[k]*step)))/(2*step) for k in (2, 3)]
    np.testing.assert_allclose(jax.grad(evaluate)(point), numerical, atol=2e-7, rtol=2e-6)


def test_batched_64_node_rule_passes_full_contact_flux_gradient_gate():
    rows = []
    for radius in (.001, .01, .1, .5, .9, 1., 1.2, 2.):
        separations = [0., radius, abs(1-radius), 1+radius]
        separations += [max(0, boundary+eps) for boundary in (radius, abs(1-radius), 1+radius)
                        for eps in (-1e-8, -1e-10, 1e-10, 1e-8)]
        rows += [[z, radius, c, alpha] for z in separations for c in (0., .6, 1.)
                 for alpha in (0., .001, .1, 1., 3., 10.)]
    points = jnp.asarray(rows)
    def program(order):
        return jax.jit(jax.vmap(jax.value_and_grad(
            lambda p:light_curve(p[2], p[3], p[0], p[1], order=order))))
    low, high = program(64)(points), program(512)(points)
    np.testing.assert_allclose(low[0], high[0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(low[1], high[1], atol=2e-7, rtol=2e-6)

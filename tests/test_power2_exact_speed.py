"""Strict flux/gradient parity of the accelerated direct power-2 integral."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from models.jaxoplanet.power2_exact import light_curve, _light_curve


def _contact_grid():
    rows = []
    for r in (.01, .02, .04, .07, .1, .1356, .146, .2, .25, .3):
        zs = [0., .25*r, .5*r, 2*r, .5, .8, .95, 1., 1.+.5*r, 1.+2*r]
        for boundary in (r, 1-r, 1+r):
            zs.extend(max(0., boundary+eps) for eps in
                      (-1e-4, -1e-8, -1e-12, 0., 1e-12, 1e-8, 1e-4))
        rows.extend([z, r, c, a] for z in sorted(set(zs))
                    for c in (0., .2, .6, 1.)
                    for a in (.05, .075, .1, .2, .4, .6, 1., 1.5))
    return np.array(rows)


def test_strict_contact_grid_parity_with_original_and_high_order_reference():
    def program(order):
        # Raw original AD for references also audits the custom derivative.
        fn = (lambda p: light_curve(p[2], p[3], p[0], p[1])) if order is None else (
            lambda p: _light_curve(p[2], p[3], p[0], p[1], order))
        return jax.jit(jax.vmap(jax.value_and_grad(fn)))
    candidate, baseline, reference = map(program, (None, 128, 1024))
    for points in np.array_split(_contact_grid(), 32):
        flux, gradient = candidate(points)
        for expected in (baseline(points), reference(points)):
            tolerance = np.maximum(1e-12, 1e-9*np.abs(expected[0]))
            assert np.all(np.abs(np.asarray(flux)-expected[0]) <= tolerance)
            np.testing.assert_allclose(gradient, expected[1], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('radius', [.01, .1356, .3])
def test_contact_shape_gradients_against_independent_boundary_integral(radius):
    # Reynolds transport theorem on the planet boundary; independent of the
    # annular endpoint map and its automatic derivatives.
    for alpha in (.05, .6, 1.5):
        for z in (0., radius, 1-radius, 1., 1+radius):
            c = .8
            if z >= 1+radius:
                dz = dr = 0.
            else:
                angle = (np.pi if z <= 1-radius else
                         np.arccos(np.clip((z*z+radius*radius-1)/(2*z*radius), -1, 1)))
                def intensity(theta):
                    mu2 = max(0., (1-z+radius)*(1+z-radius)
                              -4*z*radius*np.sin(theta/2)**2)
                    return 1-c+c*mu2**(alpha/2)
                norm = np.pi*(1-c*alpha/(alpha+2))
                dz = 2*radius/norm*quad(lambda t:intensity(t)*np.cos(t), 0., angle,
                                      epsabs=2e-12, epsrel=2e-12)[0]
                dr = -2*radius/norm*quad(intensity, 0., angle,
                                       epsabs=2e-12, epsrel=2e-12)[0]
                if z == 0.:
                    dz = 0.
            gradient = jax.jit(jax.grad(lambda p:light_curve(c, alpha, p[0], p[1])))(
                jnp.array([z, radius]))
            np.testing.assert_allclose(gradient, [dz, dr], atol=1e-7, rtol=1e-6)


def test_broadcast_reverse_derivative_and_nested_vmap_fallback():
    # Shared LD scalars, per-channel radii, and per-cadence geometry require
    # different transpose reductions of the same pointwise JVP columns.
    z = jnp.linspace(.01, 1.31, 31)[None, :]
    radius = jnp.array([.01, .1356, .3])[:, None]
    args = (jnp.array(.7), jnp.array(.4), z, radius)
    weights = jnp.linspace(.8, 1.2, 93).reshape(3, 31)
    candidate = lambda *p:jnp.sum(weights*light_curve(*p))
    original = lambda *p:jnp.sum(weights*_light_curve(*p, order=128))
    actual = jax.jit(jax.value_and_grad(candidate, argnums=(0, 1, 2, 3)))(*args)
    expected = jax.jit(jax.value_and_grad(original, argnums=(0, 1, 2, 3)))(*args)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=1e-9)
    for a, b in zip(actual[1], expected[1]):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)

    # One lane outside the audited domain makes the entire nested map use
    # 128 nodes; the branch must remain shared after differentiation too.
    points = jnp.array([[[.700000000001, .3, 1., .05], [.3, .3, .8, .6]],
                        [[1e-5, 1., .9, .001], [.2, .2, .7, 2.]]])
    fun = lambda p:light_curve(p[2], p[3], p[0], p[1])
    ref = lambda p:_light_curve(p[2], p[3], p[0], p[1], order=128)
    nested = lambda f:jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(f))))(points)
    actual, expected = nested(fun), nested(ref)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-12, rtol=1e-12)

"""Independent numerical and derivative checks for the production closed form."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.jaxoplanet import quadratic_closed, quadratic_safe
from models.jaxoplanet.power2_exact import occulted_power2_moment, uniform_overlap


def _reference(q, z, order):
    radius, u1, u2, offset = q
    z = z+offset
    area = uniform_overlap(z, radius)
    mu = occulted_power2_moment(z, radius, 1., order=order)
    mu2 = occulted_power2_moment(z, radius, 2., order=order)
    return -((1-u1-u2)*area+(u1+2*u2)*mu-u2*mu2)/(jnp.pi*(1-u1/3-u2/6))


def _closed(q, z):
    return quadratic_closed.fixed_orbit_evaluator()(q[1:3], q[0], z+q[3])


def _original(q, z):
    return quadratic_safe.fixed_orbit_evaluator(False)(q[1:3], q[0], z+q[3])


def _pair(fn):
    value = jax.jit(fn)
    derivative = jax.jit(jax.jacfwd(fn))
    return lambda q, z: (np.asarray(value(q, z)), np.asarray(derivative(q, z)))


_CLOSED = _pair(_closed)
_ORIGINAL = _pair(_original)
_REFERENCE_512 = _pair(partial(_reference, order=512))
_REFERENCE_1024 = _pair(partial(_reference, order=1024))


def _assert_parity(actual, reference):
    flux, gradient = actual
    ref_flux, ref_gradient = reference
    assert np.isfinite(flux).all() and np.isfinite(gradient).all()
    # The relative flux gate applies to normalized stellar flux, not deficit.
    assert np.all(np.abs(flux-ref_flux) <= np.maximum(1e-12, 1e-9*np.abs(1+ref_flux)))
    assert np.all(np.abs(gradient-ref_gradient) <= 1e-7+1e-6*np.abs(ref_gradient))


@pytest.mark.parametrize('radius', [np.sqrt(1e-5), .14683, .5, np.sqrt(.5)])
@pytest.mark.parametrize('plus,minus', [(.22, .024), (-1., -2.), (-1., 2.),
                                         (2., -2.), (2., 2.)])
def test_contacts_and_prior_edges_against_both_references(radius, plus, minus):
    offsets = np.array([-1e-4, -1e-8, -1e-12, 0., 1e-12, 1e-8, 1e-4])
    z = jnp.asarray(np.r_[np.linspace(0., 2., 61), radius,
                         1-radius+offsets, 1+radius+offsets])
    q = jnp.array([radius, (plus+minus)/2, (plus-minus)/2, 0.])
    independent = _REFERENCE_1024(q, z)
    _assert_parity(_REFERENCE_512(q, z), independent)
    actual = _CLOSED(q, z)
    _assert_parity(actual, _ORIGINAL(q, z))
    _assert_parity(actual, independent)


def test_eight_cel_iterations_converge_to_twelve():
    radius = jnp.asarray([np.sqrt(1e-5), .14683, .5, np.sqrt(.5)])[:, None]
    offsets = jnp.asarray([-1e-12, -1e-8, -1e-4, 0., 1e-12, 1e-8, 1e-4])
    separation = jnp.concatenate([
        jnp.broadcast_to(jnp.linspace(0., 2., 129), (4, 129)),
        1-radius+offsets, 1+radius+offsets, radius,
    ], axis=1)
    fn = jax.jit(quadratic_closed.partials, static_argnames=('iterations',))
    for u in [jnp.array([.3, .2]), jnp.array([2., 0.]), jnp.array([0., 2.])]:
        actual = fn(u, radius, separation, iterations=8)
        reference = fn(u, radius, separation, iterations=12)
        for a, b in zip(actual, reference):
            np.testing.assert_allclose(a, b, atol=2e-13, rtol=2e-13)


def test_exact_inactive_and_uniform_central_limits():
    value, *columns = quadratic_closed.partials(
        jnp.array([.3, .2]), jnp.array([0., .2, 1., 1.2]),
        jnp.array([.2, 1.2, 0., .1]),
    )
    np.testing.assert_array_equal(value, [0., 0., -1., -1.])
    for column in columns:
        np.testing.assert_array_equal(column, np.zeros(4))
    value, dr, dz, _, _ = quadratic_closed.partials(jnp.zeros(2), .2, 0.)
    np.testing.assert_allclose([value, dr, dz], [-.04, -.4, 0.], atol=1e-15, rtol=0.)


def test_radius_and_separation_signs_and_broadcasting():
    z = jnp.asarray([.11, .3, .7, .9, 1.1, 1.3])
    positive = quadratic_closed.partials(jnp.array([.3, .2]), .2, z)
    negative = quadratic_closed.partials(jnp.array([.3, .2]), -.2, -z)
    for index, (a, b) in enumerate(zip(positive, negative)):
        np.testing.assert_array_equal(a, -b if index in (1, 2) else b)


def test_batched_reverse_mode_and_mixed_hessian():
    z = jnp.asarray([.12, .37, .64, .76, .91, 1.04, 1.11])
    q = jnp.array([[.195, .3, .2, .001], [.17, .2, .1, -.001]])
    closed = lambda q: jnp.sum(_closed(q, z)**2)
    original = lambda q: jnp.sum(_original(q, z)**2)
    actual = jax.jit(jax.vmap(jax.value_and_grad(closed)))(q)
    reference = jax.jit(jax.vmap(jax.value_and_grad(original)))(q)
    np.testing.assert_allclose(actual[0], reference[0], atol=1e-12, rtol=0.)
    np.testing.assert_allclose(actual[1], reference[1], atol=1e-7, rtol=1e-6)
    actual = jax.jit(jax.hessian(closed))(q[0])
    reference = jax.jit(jax.hessian(original))(q[0])
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-6)



# Each float.hex literal specifies the exact binary64 input used by the
# independent 70-decimal-digit boundary integral (u1=2, u2=0). These four
# derivative values are regressions for the old quadrature's contact error.
# For phi=acos((z*z+r*r-1)/(2*z*r)), integrate I(theta) on [0, phi]:
# dF/dr=-2*r*integral(I)/N, dF/dz=2*r*integral(I*cos(theta))/N,
# I=2*sqrt(1-z*z-r*r+2*z*r*cos(theta))-1, N=pi/3.
@pytest.mark.parametrize('z_hex,expected_dr,expected_dz', [
    ('0x1.fe618391bccc9p-1', '0.01626093935179058597209721547351132718258007992254608723252141997970791', '0.0009042219598674003412086645982787534476238339192772624408110481798403217'),
    ('0x1.00cf3e372199bp+0', '0.0000000152186095759387817439675973089435483932396769607535679730227069856082', '-0.00000001521860957592267635793524038827018311415766722881755409064751810579603'),
 ])
def test_extreme_contacts_against_70_digit_boundary_integrals(z_hex, expected_dr, expected_dz):
    radius = float.fromhex('0x1.9e7c6e43390b7p-9')
    separation = float.fromhex(z_hex)
    _, dr, dz, _, _ = quadratic_closed.partials(jnp.array([2., 0.]), radius, separation)
    np.testing.assert_allclose([dr, dz], [float(expected_dr), float(expected_dz)],
                               atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize('radius', [1.-1e-8, 1., 1.+1e-8, 1.2])
def test_large_occultor_and_nearly_coincident_disks(radius):
    z = jnp.asarray([0., 1e-8, 1e-4, .3, 1.2, 2.1, 2.3])
    q = jnp.array([radius, .3, .2, 0.])
    def reference(q, z):
        return quadratic_safe.fixed_orbit_evaluator(True)(q[1:3], q[0], z+q[3])
    _assert_parity(_CLOSED(q, z), _pair(reference)(q, z))


def test_model_default_routes_to_closed_form(monkeypatch):
    from models.jaxoplanet.core import compute_transit_model
    def old_is_not_default(*args, **kwargs):
        raise AssertionError('default transit called the old quadrature')
    monkeypatch.setattr(quadratic_safe, 'fixed_orbit_evaluator', old_is_not_default)
    params = dict(period=jnp.array([4.]), t0=jnp.array([0.]), a_rs=jnp.array([9.]),
                  b=jnp.array([.35]), rors=jnp.array([.12]), u=jnp.array([.3, .2]))
    result = jax.jit(compute_transit_model)(params, jnp.linspace(-.1, .1, 151))
    assert np.isfinite(result).all()
    assert float(result.min()) < -.01


def _orbit_curve(q, t, impact, evaluator):
    from models.jaxoplanet.core import orbit_separation
    z, front = orbit_separation(t, 4.05528, 59770.8339+q[3], 11.4, impact)
    return jnp.where(front, evaluator(q[1:3], q[0], z), 0.)


def _orbit_reference(q, t, impact, order):
    from models.jaxoplanet.core import orbit_separation
    z, front = orbit_separation(t, 4.05528, 59770.8339+q[3], 11.4, impact)
    local = q.at[3].set(0.)
    return jnp.where(front, _reference(local, z, order), 0.)


def _orbit_pair(function):
    value, derivative = jax.jit(function), jax.jit(jax.jacfwd(function))
    return lambda *args: tuple(np.asarray(fn(*args)) for fn in (value, derivative))


_ORBIT_CLOSED = _orbit_pair(partial(_orbit_curve, evaluator=quadratic_closed.fixed_orbit_evaluator()))
_ORBIT_OLD = _orbit_pair(partial(_orbit_curve, evaluator=quadratic_safe.fixed_orbit_evaluator(True)))
_ORBIT_REF512 = _orbit_pair(partial(_orbit_reference, order=512))
_ORBIT_REF1024 = _orbit_pair(partial(_orbit_reference, order=1024))


@pytest.mark.parametrize('radius,impact', [(.14683, .45), (.14683, .99),
    (.14683, 1.14682), (np.sqrt(1e-5), .45), (np.sqrt(.5), .45)])
@pytest.mark.parametrize('epoch_offset', [-1/48, 0., 1/48])
def test_composed_orbit_grazing_and_epoch_prior_endpoints(radius, impact, epoch_offset):
    offsets = np.array([-1e-4, -1e-8, -1e-12, 0., 1e-12, 1e-8, 1e-4])
    z = np.maximum(impact, np.r_[1-radius+offsets, 1+radius+offsets])
    phase = np.arcsin(np.sqrt((z*z-impact*impact)/(11.4**2-impact*impact)))
    contacts = 59770.8339+epoch_offset+np.r_[-phase, phase]*4.05528/(2*np.pi)
    t = jnp.asarray(np.r_[np.linspace(59770.65, 59771.02, 257), contacts])
    for plus, minus in [(.22, .024), (-1., -2.), (-1., 2.), (2., -2.), (2., 2.)]:
        q = jnp.asarray([radius, (plus+minus)/2, (plus-minus)/2, epoch_offset])
        independent = _ORBIT_REF1024(q, t, impact)
        _assert_parity(_ORBIT_REF512(q, t, impact), independent)
        actual = _ORBIT_CLOSED(q, t, impact)
        _assert_parity(actual, independent)
        _assert_parity(actual, _ORBIT_OLD(q, t, impact))

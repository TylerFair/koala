"""Strict parity of the memory-bounded derivative and the original integral."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet.quadratic_safe import light_curve, fixed_orbit_evaluator

jax.config.update('jax_enable_x64',True)

@pytest.mark.parametrize('fallback',[False,True])
def test_fixed_orbit_value_radius_ld_and_geometry_gradients(fallback):
    radius=.195
    z=jnp.asarray(np.unique(np.r_[np.linspace(0.,1.3,71),radius,1-radius,1+radius,
                                     1-radius-1e-9,1-radius+1e-9,1+radius-1e-9]))
    wrapped=fixed_orbit_evaluator(fallback)
    def original(q):return light_curve(q[1:],z,q[0],equal_disk_fallback=fallback)
    def candidate(q):return wrapped(q[1:],q[0],z)
    q=jnp.array([radius,.3,.2])
    np.testing.assert_array_equal(candidate(q),original(q))
    reference=jax.jit(jax.jacrev(original))(q)
    actual=jax.jit(jax.jacrev(candidate))(q)
    assert np.isfinite(actual).all()
    assert np.all(np.abs(actual-reference)<=1e-7+1e-6*np.abs(reference))
    # A geometry tangent must invoke the full original derivative.
    one=lambda q:light_curve(q[1:3],z+q[3],q[0],equal_disk_fallback=fallback)
    two=lambda q:wrapped(q[1:3],q[0],z+q[3])
    q=jnp.array([radius,.3,.2,.0])
    np.testing.assert_allclose(jax.jit(jax.jacrev(two))(q),jax.jit(jax.jacrev(one))(q),
                               atol=1e-7,rtol=1e-6)


def test_batched_reverse_gradients_and_second_derivatives():
    z=jnp.linspace(.1,1.3,43)
    q=jnp.array([[.1,.3,.2],[.195,.5,.1],[.6,.2,.3]])
    fn=fixed_orbit_evaluator(False)
    def reference(q):return jnp.sum(light_curve(q[1:],z,q[0],equal_disk_fallback=False)**2)
    def wrapped(q):return jnp.sum(fn(q[1:],q[0],z)**2)
    a=jax.jit(jax.vmap(jax.value_and_grad(wrapped)))(q)
    b=jax.jit(jax.vmap(jax.value_and_grad(reference)))(q)
    np.testing.assert_allclose(a[0],b[0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(a[1],b[1],atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(jax.hessian(wrapped)(q[1]),jax.hessian(reference)(q[1]),
                               atol=1e-7,rtol=1e-6)


def test_mixed_geometry_hessian_preserves_original_integral():
    z=jnp.linspace(.1,1.3,43)
    q=jnp.array([.195,.3,.2,.001])
    fn=fixed_orbit_evaluator(False)
    original=lambda q:jnp.sum(light_curve(q[1:3],z+q[3],q[0],equal_disk_fallback=False)**2)
    wrapped=lambda q:jnp.sum(fn(q[1:3],q[0],z+q[3])**2)
    a=jax.jit(jax.hessian(wrapped))(q)
    b=jax.jit(jax.hessian(original))(q)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('radius',[1.-1e-8,1.,1.+1e-8])
def test_equal_disk_safety_fallback_keeps_original_derivative(radius):
    z=jnp.array([0.,1e-8,1e-4,.3,1.2,2.1])
    q=jnp.array([radius,.3,.2])
    fn=fixed_orbit_evaluator(True)
    original=lambda p:light_curve(p[1:],z,p[0],equal_disk_fallback=True)
    wrapped=lambda p:fn(p[1:],p[0],z)
    np.testing.assert_allclose(wrapped(q),original(q),atol=1e-12,rtol=0)
    actual=jax.jit(jax.jacrev(wrapped))(q)
    expected=jax.jit(jax.jacrev(original))(q)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)


def test_separation_jacobian_is_diagonal_and_matches_finite_difference():
    z=jnp.linspace(.1,1.3,43)
    u=jnp.array([.3,.2]);radius=.195
    fn=fixed_orbit_evaluator(False)
    original=lambda z:light_curve(u,z,radius,equal_disk_fallback=False)
    wrapped=lambda z:fn(u,radius,z)
    actual=jax.jit(jax.jacfwd(wrapped))(z)
    reference=jax.jit(jax.jacfwd(original))(z)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,reference,atol=1e-7,rtol=1e-6)
    np.testing.assert_array_equal(actual-jnp.diag(jnp.diag(actual)),jnp.zeros_like(actual))
    direction=jnp.linspace(.8,1.2,len(z));step=1e-6
    fd=(original(z+step*direction)-original(z-step*direction))/(2*step)
    np.testing.assert_allclose(actual@direction,fd,atol=1e-7,rtol=1e-6)

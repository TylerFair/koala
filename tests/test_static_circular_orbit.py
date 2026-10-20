"""A configured e = 0 skips the Kepler solve and the eccentric T14 branch without changing the orbit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.harmonica.core import harmonica_a_rs_from_duration, harmonica_duration_from_geometry
from models.jaxoplanet.core import compute_transit_model, orbit_separation

jax.config.update('jax_enable_x64', True)

T = jnp.linspace(-0.12, 0.12, 241)
IMPACTS = [0.0, 0.45, 0.98, 1.1]


@pytest.mark.parametrize('b', IMPACTS)
@pytest.mark.parametrize('convert,x', [
    (harmonica_a_rs_from_duration, (3.325271, 0.0852, None, 0.18)),
    (harmonica_duration_from_geometry, (3.325271, 13.6, None, 0.18)),
])
def test_circular_contact_conversion_matches_the_cond_path(convert, x, b):
    x = jnp.array([x[0], x[1], b, x[3]])
    fast = lambda p: convert(*p, 0.0, np.pi/2, circular=True)
    general = lambda p: convert(*p, 0.0, np.pi/2)
    np.testing.assert_allclose(fast(x), general(x), rtol=1e-14, atol=0)
    np.testing.assert_allclose(jax.grad(fast)(x), jax.grad(general)(x), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(jax.hessian(fast)(x), jax.hessian(general)(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('omega', [np.pi/2, 0.3])
@pytest.mark.parametrize('b', IMPACTS)
def test_circular_separation_matches_kepler(b, omega):
    fast = orbit_separation(T, 3.325271, 0.001, 13.6, b, 0.0, omega, circular=True)
    general = orbit_separation(T, 3.325271, 0.001, 13.6, b, 0.0, omega)
    np.testing.assert_array_equal(fast[1], general[1])
    np.testing.assert_allclose(fast[0], general[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize('b', IMPACTS)
def test_circular_flag_matches_flux_and_gradients(b):
    def flux(p, circular):
        period, t0, a_rs, impact, rors, u1, u2 = p
        params = dict(period=jnp.array([period]), t0=jnp.array([t0]), a_rs=jnp.array([a_rs]),
                      b=jnp.array([impact]), rors=jnp.array([rors]), ecc=jnp.array([0.0]),
                      omega=jnp.array([np.pi/2]), u=jnp.array([u1, u2]),
                      _fixed_circular_orbit=circular)
        return compute_transit_model(params, T)
    p = jnp.array([3.325271, 0.001, 13.6, b, 0.18, 0.4, 0.2])
    np.testing.assert_allclose(flux(p, True), flux(p, False), rtol=0, atol=1e-12)
    np.testing.assert_allclose(jax.jacfwd(lambda q: flux(q, True))(p),
                               jax.jacfwd(lambda q: flux(q, False))(p), rtol=1e-6, atol=1e-7)

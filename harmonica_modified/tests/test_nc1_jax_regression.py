"""Regression tests for the pure-JAX N_c=1 Harmonica implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from harmonica.jax import harmonica_transit_power2_ld, harmonica_transit_quad_ld
from harmonica.jax.power2_nc1_jax import (
    _compute_flux_nc1_power2,
    harmonica_transit_power2_nc1_jax,
    harmonica_transit_quadratic_nc1_jax,
)


jax.config.update("jax_enable_x64", True)


def _case(a1=0.003, b1=-0.001):
    period = 3.9502001
    t0 = 60595.21518
    a_rs = 7.3
    inc = np.arccos(0.15 / a_rs)
    times = jnp.linspace(t0 - 0.14, t0 + 0.14, 193, dtype=jnp.float64)
    return times, t0, period, a_rs, inc, 0.0, 0.0, 0.107, a1, b1


@pytest.mark.parametrize(
    ("a1", "b1"),
    [(0.0, 0.0), (1e-8, 0.0), (-1e-8, 0.0), (0.006, -0.002)],
)
def test_power2_flux_matches_cpp_reference(a1, b1):
    times, t0, period, a_rs, inc, ecc, omega, r0, _, _ = _case(a1, b1)
    c_ld, alpha = 0.54, 0.72
    actual = harmonica_transit_power2_nc1_jax(
        times, t0, period, a_rs, inc, ecc, omega,
        c_ld, alpha, r0, a1, b1,
    )
    reference = harmonica_transit_power2_ld(
        times, t0, period, a_rs, inc, ecc, omega,
        c=c_ld, alpha=alpha, r=jnp.array([r0, a1, b1]),
    )
    # The legacy CPU wrapper regularizes exactly zero trailing harmonics at
    # 1e-9, which changes symmetric fluxes by at most ~7e-11.
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("u1", "u2"),
    [(0.0, 0.0), (0.35, 0.18), (0.60, -0.20)],
)
def test_quadratic_flux_matches_cpp_reference(u1, u2):
    times, t0, period, a_rs, inc, ecc, omega, r0, a1, b1 = _case()
    actual = harmonica_transit_quadratic_nc1_jax(
        times, t0, period, a_rs, inc, ecc, omega,
        u1, u2, r0, a1, b1,
    )
    reference = harmonica_transit_quad_ld(
        times, t0, period, a_rs, inc, ecc, omega,
        u1=u1, u2=u2, r=jnp.array([r0, a1, b1]),
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("a1", [-1e-8, 1e-8, 0.004])
def test_power2_shape_gradient_matches_cpp_near_symmetry(a1):
    times, t0, period, a_rs, inc, ecc, omega, r0, _, b1 = _case(a1, 0.0)
    weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)

    def loss_jax(a1_value):
        flux = harmonica_transit_power2_nc1_jax(
            times, t0, period, a_rs, inc, ecc, omega,
            0.54, 0.72, r0, a1_value, b1,
        )
        return jnp.vdot(weights, flux)

    def loss_cpp(a1_value):
        flux = harmonica_transit_power2_ld(
            times, t0, period, a_rs, inc, ecc, omega,
            c=0.54, alpha=0.72, r=jnp.array([r0, a1_value, b1]),
        )
        return jnp.vdot(weights, flux)

    actual = jax.grad(loss_jax)(jnp.float64(a1))
    reference = jax.grad(loss_cpp)(jnp.float64(a1))
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-8)


def test_power2_gradient_matches_independent_central_difference():
    times, t0, period, a_rs, inc, ecc, omega, _, _, _ = _case()
    weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)

    def loss(theta):
        r0, a1, c_ld, alpha = theta
        flux = harmonica_transit_power2_nc1_jax(
            times, t0, period, a_rs, inc, ecc, omega,
            c_ld, alpha, r0, a1, -0.001,
        )
        return jnp.vdot(weights, flux)

    theta = jnp.array([0.107, 0.003, 0.54, 0.72], dtype=jnp.float64)
    autodiff = np.asarray(jax.grad(loss)(theta))
    step = np.array([2e-7, 2e-7, 2e-6, 2e-6])
    finite_difference = np.empty(4)
    for index in range(4):
        offset = np.zeros(4)
        offset[index] = step[index]
        finite_difference[index] = (
            float(loss(theta + offset)) - float(loss(theta - offset))
        ) / (2.0 * step[index])
    np.testing.assert_allclose(
        autodiff, finite_difference, rtol=3e-5, atol=3e-7
    )


def test_power2_and_quadratic_equivalent_special_case():
    """alpha=2 power-2 is quadratic with direct u1=2c and u2=-c."""
    times, t0, period, a_rs, inc, ecc, omega, r0, a1, b1 = _case()
    c_ld = 0.3
    power2 = harmonica_transit_power2_nc1_jax(
        times, t0, period, a_rs, inc, ecc, omega,
        c_ld, 2.0, r0, a1, b1,
    )
    quadratic = harmonica_transit_quadratic_nc1_jax(
        times, t0, period, a_rs, inc, ecc, omega,
        2.0 * c_ld, -c_ld, r0, a1, b1,
    )
    np.testing.assert_allclose(power2, quadratic, rtol=2e-12, atol=2e-12)


def test_reverse_mode_does_not_leak_nan_from_padded_roots():
    """A masked 0/0 at invalid padded roots previously poisoned this gradient."""
    def flux_at_separation(d):
        return _compute_flux_nc1_power2(
            d, 1.0, 0.0, 0.3, 0.7, 0.1, 0.005, 0.0
        )

    reverse = float(jax.grad(flux_at_separation)(jnp.float64(1.0)))
    _, forward = jax.jvp(
        flux_at_separation, (jnp.float64(1.0),), (jnp.float64(1.0),)
    )
    step = 1e-6
    finite_difference = (
        float(flux_at_separation(1.0 + step))
        - float(flux_at_separation(1.0 - step))
    ) / (2.0 * step)
    assert np.isfinite(reverse)
    np.testing.assert_allclose(reverse, forward, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        reverse, finite_difference, rtol=3e-5, atol=3e-7
    )


def test_exact_outer_tangency_is_no_occultation_with_finite_gradient():
    """An odd one-root tangency must not be integrated as a full loop."""
    r0, a1 = 0.1, 0.005
    contact = 1.0 + r0 + a1

    def flux_at_separation(d):
        return _compute_flux_nc1_power2(
            d, 1.0, 0.0, 0.3, 0.7, r0, a1, 0.0
        )

    values = np.array([
        float(flux_at_separation(contact - 1e-8)),
        float(flux_at_separation(contact)),
        float(flux_at_separation(contact + 1e-8)),
    ])
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[1:], 1.0, rtol=0.0, atol=2e-13)
    assert values[0] <= 1.0
    assert np.isfinite(jax.grad(flux_at_separation)(jnp.float64(contact)))


def test_high_eccentricity_orbit_matches_cpp_reference():
    """The old E0=M Newton starter selected a wrong branch above e~0.98."""
    times = jnp.linspace(-0.15, 0.15, 301, dtype=jnp.float64)
    common = (times, 0.0, 3.95, 10.0, 1.53, 0.99, 0.7)
    actual = harmonica_transit_power2_nc1_jax(
        *common, 0.4, 0.8, 0.1, 0.003, 0.0
    )
    reference = harmonica_transit_power2_ld(
        *common, c=0.4, alpha=0.8, r=jnp.array([0.1, 0.003, 0.0])
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-11)


def test_shared_time_channel_batch_matches_replicated_time_batch():
    """The optimized spectral path must preserve the former nested-vmap result."""
    times, t0, period, a_rs, inc, ecc, omega, _, _, _ = _case()
    n_channels = 6
    c_ld = jnp.linspace(0.3, 0.7, n_channels)
    alpha = jnp.linspace(0.5, 1.1, n_channels)
    r0 = jnp.linspace(0.102, 0.112, n_channels)
    a1 = jnp.linspace(-1e-8, 0.006, n_channels)
    b1 = jnp.zeros(n_channels)
    shared = harmonica_transit_power2_nc1_jax(
        times, t0, period, a_rs, inc, ecc, omega,
        c_ld, alpha, r0, a1, b1,
    )
    replicated = harmonica_transit_power2_nc1_jax(
        jnp.broadcast_to(times[None, :], (n_channels, times.size)),
        t0, period, a_rs, inc, ecc, omega,
        c_ld, alpha, r0, a1, b1,
    )
    np.testing.assert_allclose(shared, replicated, rtol=2e-12, atol=2e-12)


def test_two_dimensional_times_accept_scalar_channel_parameters():
    """The public batching API previously claimed, but lacked, this broadcast."""
    times, t0, period, a_rs, inc, ecc, omega, r0, a1, b1 = _case()
    batched_times = jnp.broadcast_to(times[None, :], (3, times.size))
    flux = harmonica_transit_quadratic_nc1_jax(
        batched_times, t0, period, a_rs, inc, ecc, omega,
        0.35, 0.18, r0, a1, b1,
    )
    assert flux.shape == batched_times.shape
    np.testing.assert_allclose(flux[0], flux[1], rtol=0.0, atol=0.0)


def test_jitted_reverse_gradient_has_no_inactive_segment_nan():
    """XLA reverse-mode used to leak a singular padded-segment derivative."""
    times = jnp.linspace(-0.12, 0.12, 300, dtype=jnp.float64)
    period, a_rs = 3.95, 7.3
    inc = np.arccos(0.15 / a_rs)

    def loss(a1):
        return jnp.sum(
            harmonica_transit_power2_nc1_jax(
                times, 0.0, period, a_rs, inc, 0.0, 0.0,
                0.2338983, 0.6508475, 0.107, a1, 0.0,
            )
        )

    a1 = jnp.float64(-0.00198305)
    eager = jax.grad(loss)(a1)
    compiled = jax.jit(jax.grad(loss))(a1)
    assert np.isfinite(eager)
    assert np.isfinite(compiled)
    np.testing.assert_allclose(compiled, eager, rtol=2e-10, atol=2e-10)

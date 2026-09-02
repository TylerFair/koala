"""Whole-lane custom-VJP prototype for the native Power-2 evaluator.

This module is evaluation-only and is not imported by the fitter.  Its goal
is to test the memory/performance architecture needed by a future analytic
VJP: retain only the local light-curve Jacobian ``(channel, time, parameter)``
and contract the incoming likelihood cotangent directly to one three-vector
per channel.  In particular, it does not retain the much larger
``(channel, time, quadrature_node)`` reverse-mode tape.

The local partials in this prototype are generated with forward-mode JAX.
That makes it a correctness-preserving upper-bound experiment for hand-coded
shape derivatives, not the proposed final implementation.  A useful timing
result would justify replacing these generated partials with the analytic
area/contact formulas; a slow result closes the architecture before doing
that delicate work.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .experimental_power2_native import light_curve


def make_fixed_geometry_light_curve_local_vjp(
    phase_offsets,
    phase_mask,
    *,
    impact_parameter: float,
    duration: float,
    order: int = 16,
):
    """Return ``flux(theta)`` with a whole-lane local-Jacobian VJP.

    ``theta`` has shape ``(n_channels, 3)`` with columns
    ``(radius_ratio, c, alpha)``.  Geometry and cadence offsets are fixed, as
    they are during the optimized spectroscopic fit.
    """

    offsets_np = np.asarray(phase_offsets, dtype=np.float64)
    mask_np = np.asarray(phase_mask, dtype=bool)
    if offsets_np.ndim != 1 or offsets_np.size == 0:
        raise ValueError("phase_offsets must be a non-empty one-dimensional array")
    if mask_np.shape != offsets_np.shape:
        raise ValueError("phase_mask must have the same shape as phase_offsets")
    if not np.all(np.isfinite(offsets_np)):
        raise ValueError("phase_offsets must be finite")
    impact_parameter = float(impact_parameter)
    duration = float(duration)
    if not np.isfinite(impact_parameter) or impact_parameter < 0.0:
        raise ValueError("impact_parameter must be finite and non-negative")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if int(order) < 2:
        raise ValueError("order must be at least two")

    offsets = jnp.asarray(offsets_np, dtype=jnp.float64)
    mask = jnp.asarray(mask_np)

    def one_lane(one_theta):
        radius, c_value, alpha = one_theta
        speed = 2.0 * jnp.sqrt(
            jnp.maximum(0.0, (1.0 + radius) ** 2 - impact_parameter**2)
        ) / duration
        separation = jnp.sqrt(
            jnp.square(speed * offsets) + impact_parameter**2
        )
        signal = light_curve(
            c_value, alpha, separation, radius, order=int(order)
        )
        return jnp.where(mask, signal, 0.0)

    def ordinary(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        return jax.vmap(one_lane)(theta)

    @jax.custom_vjp
    def evaluate(theta):
        return ordinary(theta)

    def evaluate_fwd(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)

        def value_and_local_jacobian(one_theta):
            def with_aux(values):
                flux = one_lane(values)
                return flux, flux

            # Shape: (n_times, 3).  No quadrature-node axis survives this
            # boundary, so the outer likelihood reverse pass retains O(BT)
            # rather than O(BTQ) data.
            return jax.jacfwd(with_aux, has_aux=True)(one_theta)

        local_jacobian, value = jax.vmap(value_and_local_jacobian)(theta)
        return value, local_jacobian

    def evaluate_bwd(local_jacobian, cotangent):
        cotangent = jnp.asarray(cotangent, dtype=jnp.float64)
        theta_cotangent = jnp.einsum(
            "bt,btp->bp", cotangent, local_jacobian
        )
        return (theta_cotangent,)

    evaluate.defvjp(evaluate_fwd, evaluate_bwd)
    return evaluate


__all__ = ["make_fixed_geometry_light_curve_local_vjp"]

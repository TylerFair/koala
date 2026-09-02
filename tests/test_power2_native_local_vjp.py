import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from models.jaxoplanet.experimental_power2_native import light_curve
from models.jaxoplanet.experimental_power2_native_local_vjp import (
    make_fixed_geometry_light_curve_local_vjp,
)


class TestNativePower2LocalVjp(unittest.TestCase):
    def test_value_and_weighted_gradient_match_ordinary_reverse_mode(self):
        duration = 0.12
        impact = 0.35
        radius = 0.2
        speed = 2.0 * np.sqrt((1.0 + radius) ** 2 - impact**2) / duration
        inner_time = np.sqrt((1.0 - radius) ** 2 - impact**2) / speed
        offsets = np.asarray(
            [
                -0.5 * duration,
                -inner_time - 1e-10,
                -inner_time,
                -inner_time + 1e-10,
                0.0,
                inner_time - 1e-10,
                inner_time,
                inner_time + 1e-10,
                0.5 * duration,
            ]
        )
        mask = np.abs(offsets) < 0.5 * duration
        theta = jnp.asarray(
            [[radius, 0.7, 0.4], [0.35, 1.0, 0.001]], dtype=jnp.float64
        )

        def one_lane(one_theta):
            one_radius, c_value, alpha = one_theta
            lane_speed = 2.0 * jnp.sqrt(
                (1.0 + one_radius) ** 2 - impact**2
            ) / duration
            separation = jnp.sqrt(
                (lane_speed * jnp.asarray(offsets)) ** 2 + impact**2
            )
            signal = light_curve(
                c_value, alpha, separation, one_radius, order=16
            )
            return jnp.where(jnp.asarray(mask), signal, 0.0)

        ordinary = lambda values: jax.vmap(one_lane)(values)
        custom = make_fixed_geometry_light_curve_local_vjp(
            offsets,
            mask,
            impact_parameter=impact,
            duration=duration,
            order=16,
        )
        weights = jnp.arange(theta.shape[0] * offsets.size, dtype=jnp.float64)
        weights = weights.reshape(theta.shape[0], offsets.size) + 0.5
        ordinary_vg = jax.jit(
            jax.value_and_grad(lambda values: jnp.vdot(ordinary(values), weights))
        )(theta)
        custom_vg = jax.jit(
            jax.value_and_grad(lambda values: jnp.vdot(custom(values), weights))
        )(theta)

        np.testing.assert_allclose(custom(theta), ordinary(theta), atol=0.0, rtol=0.0)
        np.testing.assert_allclose(custom_vg[0], ordinary_vg[0], atol=2e-12, rtol=0)
        np.testing.assert_allclose(custom_vg[1], ordinary_vg[1], atol=2e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()

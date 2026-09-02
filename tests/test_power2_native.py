import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from models.jaxoplanet.experimental_power2_native import light_curve


def _radial_reference(separation, radius_ratio, alpha):
    """Independent area integral of the blocked ``mu**alpha`` flux."""

    separation = float(separation)
    radius_ratio = float(radius_ratio)
    alpha = float(alpha)
    if separation == 0.0:
        upper = min(radius_ratio, 1.0)
        return (
            2.0
            * np.pi
            / (alpha + 2.0)
            * (1.0 - (1.0 - upper**2) ** (1.0 + 0.5 * alpha))
        )

    def integrand(stellar_radius):
        if radius_ratio >= separation + stellar_radius:
            occulted_angle = 2.0 * np.pi
        elif radius_ratio <= abs(separation - stellar_radius):
            occulted_angle = 0.0
        else:
            cosine = (
                stellar_radius**2 + separation**2 - radius_ratio**2
            ) / (2.0 * stellar_radius * separation)
            occulted_angle = 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))
        return (
            stellar_radius
            * (1.0 - stellar_radius**2) ** (0.5 * alpha)
            * occulted_angle
        )

    breakpoints = sorted(
        {
            0.0,
            1.0,
            np.clip(abs(separation - radius_ratio), 0.0, 1.0),
            np.clip(separation + radius_ratio, 0.0, 1.0),
        }
    )
    return sum(
        quad(integrand, low, high, epsabs=2.0e-13, epsrel=2.0e-13)[0]
        for low, high in zip(breakpoints[:-1], breakpoints[1:])
        if high > low
    )


class TestNativePower2(unittest.TestCase):
    def test_selected_points_match_independent_radial_integral(self):
        cases = (
            (0.0, 0.10, 0.001),
            (0.30, 0.20, 0.10),
            (0.80, 0.25, 0.35),
            (0.75, 0.28, 1.00),
            (0.999, 0.10, 0.001),
            (1.02, 0.10, 0.30),
        )
        for separation, radius_ratio, alpha in cases:
            blocked = _radial_reference(separation, radius_ratio, alpha)
            stellar_flux = 2.0 * np.pi / (alpha + 2.0)
            expected = -blocked / stellar_flux
            actual = float(
                light_curve(
                    1.0, alpha, separation, radius_ratio, order=16
                )
            )
            self.assertLess(abs(actual - expected), 2.0e-6)

    def test_order_16_is_below_two_ppm_on_contact_focused_full_prior_grid(self):
        radii = np.geomspace(np.sqrt(1.0e-5), np.sqrt(0.5), 9)
        alphas = np.geomspace(0.001, 1.0, 8)
        rows = []
        for radius in radii:
            inner = 1.0 - radius
            outer = 1.0 + radius
            separations = np.concatenate(
                (
                    np.linspace(0.0, inner, 9),
                    inner + 2.0 * radius * np.linspace(0.0, 1.0, 17),
                    inner + np.array([-1.0, 1.0]) * 1.0e-10,
                    outer + np.array([-1.0, 1.0]) * 1.0e-10,
                )
            )
            for alpha in alphas:
                rows.extend((1.0, alpha, z, radius) for z in separations)

        theta = jnp.asarray(rows, dtype=jnp.float64)

        def evaluate(order):
            return jax.vmap(
                lambda values: light_curve(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    order=order,
                )
            )(theta)

        candidate = np.asarray(evaluate(16))
        reference = np.asarray(evaluate(128))
        self.assertLess(np.max(np.abs(candidate - reference)), 2.0e-6)

    def test_values_and_gradients_are_finite_around_contacts(self):
        c = 0.7
        alpha = 0.4
        radius = 0.2
        separations = jnp.asarray(
            [
                1.0 - radius - 1.0e-8,
                1.0 - radius,
                1.0 - radius + 1.0e-8,
                1.0 + radius - 1.0e-8,
                1.0 + radius,
                1.0 + radius + 1.0e-8,
            ]
        )

        def scalar(theta):
            return light_curve(theta[0], theta[1], theta[2], theta[3])

        theta = jnp.stack(
            (
                jnp.full_like(separations, c),
                jnp.full_like(separations, alpha),
                separations,
                jnp.full_like(separations, radius),
            ),
            axis=1,
        )
        values = jax.vmap(scalar)(theta)
        gradients = jax.vmap(jax.grad(scalar))(theta)
        self.assertTrue(np.all(np.isfinite(np.asarray(values))))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradients))))


if __name__ == "__main__":
    unittest.main()

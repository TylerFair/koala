import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from models.jaxoplanet.experimental_power2_grid import (
    FixedPower2TransitGeometry,
    Power2TransitGrid,
)
from models.jaxoplanet.experimental_power2_hermite import (
    Power2RadiusCubicGrid,
    Power2RadiusHermiteGrid,
    augment_power2_grid_with_radius_derivatives,
)


class TestPower2RadiusHermiteGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(-0.02, 0.02, 5)
        cls.geometry = FixedPower2TransitGeometry(2.0, 0.12, 0.0, 0.3)
        rors = np.array([0.08, 0.10, 0.13])
        c_axis = np.array([0.2, 0.7])
        alpha = np.array([0.4, 0.9])
        rr, cc, aa = np.meshgrid(rors, c_axis, alpha, indexing="ij")
        flux = (
            rr**3 + 2.0 * cc + 3.0 * aa
        )[..., None] + 0.5 * cls.times[None, None, None, :]
        derivative = np.broadcast_to(
            (3.0 * rr**2)[..., None], flux.shape
        ).copy()
        base = Power2TransitGrid(
            times=cls.times,
            geometry=cls.geometry,
            rors_knots=rors,
            c_knots=c_axis,
            alpha_knots=alpha,
            flux_grid=flux,
        )
        cls.grid = Power2RadiusHermiteGrid(base, derivative)

    def test_cubic_radius_and_parameter_gradient_are_exact(self):
        theta = jnp.asarray([0.094, 0.43, 0.61])

        def evaluate(x):
            return self.grid.evaluate_with_status(x[0], x[1], x[2])[0]

        expected = (
            theta[0] ** 3
            + 2.0 * theta[1]
            + 3.0 * theta[2]
            + 0.5 * self.times
        )
        np.testing.assert_allclose(evaluate(theta), expected, atol=2.0e-15, rtol=0.0)
        expected_gradient = np.broadcast_to(
            np.array([3.0 * float(theta[0]) ** 2, 2.0, 3.0]),
            (self.times.size, 3),
        )
        np.testing.assert_allclose(
            jax.jacrev(evaluate)(theta), expected_gradient, atol=5.0e-14, rtol=0.0
        )

    def test_radius_derivative_is_continuous_at_an_internal_knot(self):
        def scalar(radius):
            return self.grid._interpolate_raw(radius, 0.43, 0.61)[0][2]

        derivative = jax.grad(scalar)
        knot = 0.10
        left = float(derivative(knot - 1.0e-10))
        at_knot = float(derivative(knot))
        right = float(derivative(knot + 1.0e-10))
        self.assertAlmostEqual(left, at_knot, places=9)
        self.assertAlmostEqual(right, at_knot, places=9)

    def test_out_of_domain_is_flagged_and_memory_includes_derivatives(self):
        flux, valid = self.grid.evaluate_with_status(0.2, 0.4, 0.6)
        self.assertFalse(bool(valid))
        self.assertTrue(np.all(np.isnan(np.asarray(flux))))
        self.assertEqual(
            self.grid.estimated_bytes, 2 * self.grid.base_grid.estimated_bytes
        )

    def test_exact_radius_jvp_builder_is_finite(self):
        # Use an already tiny exact grid to exercise the stored-JVP path without
        # adding a large benchmark artifact to the test suite.
        from models.jaxoplanet.experimental_power2_grid import (
            build_power2_transit_grid,
        )

        base = build_power2_transit_grid(
            np.linspace(-0.04, 0.04, 9),
            self.geometry,
            [0.09, 0.11],
            [0.3, 0.6],
            [0.5, 0.8],
            build_batch_size=4,
        )
        augmented = augment_power2_grid_with_radius_derivatives(
            base, derivative_batch_size=4
        )
        self.assertEqual(augmented.rors_derivative_grid.shape, base.flux_grid.shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(augmented.rors_derivative_grid))))

    def test_finite_difference_cubic_is_c1_without_extra_storage(self):
        rors = np.array([0.07, 0.09, 0.11, 0.13])
        c_axis = np.array([0.2, 0.7])
        alpha = np.array([0.4, 0.9])
        rr, cc, aa = np.meshgrid(rors, c_axis, alpha, indexing="ij")
        values = (rr**2 + 2.0 * cc + 3.0 * aa)[..., None]
        values = values + 0.5 * self.times[None, None, None, :]
        base = Power2TransitGrid(
            times=self.times,
            geometry=self.geometry,
            rors_knots=rors,
            c_knots=c_axis,
            alpha_knots=alpha,
            flux_grid=values,
        )
        cubic = Power2RadiusCubicGrid(base)

        def scalar(radius):
            return cubic._interpolate_raw(radius, 0.43, 0.61)[0][2]

        expected = 0.10**2 + 2.0 * 0.43 + 3.0 * 0.61 + 0.5 * self.times[2]
        self.assertAlmostEqual(float(scalar(0.10)), expected, places=13)
        self.assertAlmostEqual(float(jax.grad(scalar)(0.10)), 0.20, places=12)
        left = float(jax.grad(scalar)(0.11 - 1.0e-10))
        right = float(jax.grad(scalar)(0.11 + 1.0e-10))
        self.assertAlmostEqual(left, right, delta=1.0e-8)
        self.assertEqual(cubic.estimated_bytes, base.estimated_bytes)


if __name__ == "__main__":
    unittest.main()

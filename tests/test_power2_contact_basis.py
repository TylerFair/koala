import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jaxoplanet.core.limb_dark import solution_vector

from models.jaxoplanet.experimental_power2_contact_basis import (
    POWER2_ALPHA_PRIOR_BOUNDS,
    POWER2_C_PRIOR_BOUNDS,
    RORS_PRIOR_BOUNDS,
    build_power2_contact_basis_grid,
    make_contact_basis_axes,
)
from models.jaxoplanet.experimental_power2_grid import FixedPower2TransitGeometry


class TestPower2ContactBasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = FixedPower2TransitGeometry(2.0, 0.12, 0.0, 0.35)
        cls.grid = build_power2_contact_basis_grid(
            cls.geometry,
            n_rors=5,
            n_full=7,
            n_partial=7,
            build_batch_size=16,
        )

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()

    def test_axes_cover_the_complete_current_spectroscopic_prior(self):
        rors, full_q, partial_q = make_contact_basis_axes(
            n_rors=9, n_full=9, n_partial=9
        )
        self.assertEqual((rors[0], rors[-1]), RORS_PRIOR_BOUNDS)
        self.assertEqual((full_q[0], full_q[-1]), (0.0, 1.0))
        self.assertEqual((partial_q[0], partial_q[-1]), (0.0, 1.0))
        self.assertLess(1.0 - full_q[-2], full_q[1] - full_q[0])
        self.assertLess(partial_q[1], partial_q[4] - partial_q[3])
        self.assertEqual(POWER2_C_PRIOR_BOUNDS, (0.0, 1.0))
        self.assertEqual(POWER2_ALPHA_PRIOR_BOUNDS, (0.001, 1.0))

    def test_solution_is_exact_at_full_and_partial_tensor_knots(self):
        exact = solution_vector(12, order=10)
        radius = self.grid.rors_knots[2]
        full_q = self.grid.full_q_knots[3]
        full_z = full_q * (1.0 - radius)
        actual, valid = self.grid.solution_from_separation(radius, full_z)
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(actual, exact(full_z, radius), atol=2e-12, rtol=0)

        partial_q = self.grid.partial_q_knots[3]
        partial_z = 1.0 - radius + 2.0 * radius * partial_q
        actual, valid = self.grid.solution_from_separation(radius, partial_z)
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(
            actual, exact(partial_z, radius), atol=2e-12, rtol=0
        )

    def test_inner_contact_is_shared_and_outer_contact_is_exact_zero(self):
        radius = self.grid.rors_knots[2]
        c_value, alpha = 0.73, 0.27
        inner = 1.0 - radius
        left = self.grid.signal_from_separation(
            radius, c_value, alpha, inner - 1e-11
        )[0]
        right = self.grid.signal_from_separation(
            radius, c_value, alpha, inner + 1e-11
        )[0]
        self.assertAlmostEqual(float(left), float(right), places=8)
        outer = self.grid.signal_from_separation(
            radius, c_value, alpha, 1.0 + radius
        )[0]
        self.assertAlmostEqual(float(outer), 0.0, delta=1.0e-9)

    def test_limb_darkening_is_exact_not_interpolated(self):
        radius = self.grid.rors_knots[2]
        q = self.grid.partial_q_knots[3]
        separation = 1.0 - radius + 2.0 * radius * q
        exact_solution = solution_vector(12, order=10)(separation, radius)
        for c_value, alpha in ((0.0, 0.001), (1.0, 0.001), (1.0, 1.0)):
            green = self.grid._power2_green(c_value, alpha)
            expected = exact_solution @ green - 1.0
            actual, valid = self.grid.signal_from_separation(
                radius, c_value, alpha, separation
            )
            self.assertTrue(bool(valid))
            # Batched table construction and a scalar exact call can use
            # slightly different XLA contractions; this is <0.001 ppm.
            self.assertAlmostEqual(float(actual), float(expected), delta=1.0e-9)

    def test_limb_darkening_transform_is_jittable(self):
        cases = jnp.asarray([[0.0, 0.001], [0.5, 0.5], [1.0, 1.0]])
        transformed = jax.jit(
            jax.vmap(lambda pair: self.grid._power2_green(pair[0], pair[1]))
        )(cases)
        self.assertEqual(transformed.shape, (3, 13))
        self.assertTrue(bool(jnp.all(jnp.isfinite(transformed))))

    def test_out_of_domain_parameters_are_flagged(self):
        signal, valid = self.grid.signal_from_separation(
            RORS_PRIOR_BOUNDS[0] / 2.0, 0.5, 0.5, 1.0
        )
        self.assertFalse(bool(valid))
        self.assertTrue(np.isnan(float(signal)))
        signal, valid = self.grid.signal_from_separation(0.1, 0.5, 1e-4, 1.0)
        self.assertFalse(bool(valid))
        self.assertTrue(np.isnan(float(signal)))


if __name__ == "__main__":
    unittest.main()

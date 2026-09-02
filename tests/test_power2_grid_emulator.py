import tempfile
import unittest
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from models.jaxoplanet.experimental_power2_grid import (
    FixedPower2TransitGeometry,
    Power2GridOutOfDomainError,
    Power2GridValidationError,
    Power2TransitGrid,
    UnvalidatedPower2GridError,
    build_power2_transit_grid,
    load_power2_grid,
    make_exact_power2_evaluator,
    save_power2_grid,
    validate_power2_grid,
)


class TestPower2GridInterpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(-0.03, 0.03, 7)
        cls.geometry = FixedPower2TransitGeometry(
            period=2.0, duration=0.12, t0=0.0, b=0.3
        )
        rors = np.array([0.08, 0.10, 0.13])
        c_axis = np.array([0.2, 0.7])
        alpha = np.array([0.4, 0.9])
        rr, cc, aa = np.meshgrid(rors, c_axis, alpha, indexing="ij")
        base = rr + 2.0 * cc + 3.0 * aa
        values = base[..., None] + 0.5 * cls.times[None, None, None, :]
        cls.affine_grid = Power2TransitGrid(
            times=cls.times,
            geometry=cls.geometry,
            rors_knots=rors,
            c_knots=c_axis,
            alpha_knots=alpha,
            flux_grid=values,
            exact_kernel="stock",
        )

    def test_multilinear_interpolation_and_gradient_are_exact_for_affine_grid(self):
        theta = jnp.array([0.095, 0.43, 0.61])

        def evaluate(x):
            return self.affine_grid.evaluate_with_status(
                x[0], x[1], x[2], allow_unvalidated=True
            )[0]

        actual = evaluate(theta)
        expected = theta[0] + 2.0 * theta[1] + 3.0 * theta[2] + 0.5 * self.times
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)
        jacobian = jax.jacrev(evaluate)(theta)
        np.testing.assert_allclose(
            jacobian,
            np.broadcast_to(np.array([1.0, 2.0, 3.0]), jacobian.shape),
            rtol=0.0,
            atol=3.0e-14,
        )

    def test_unvalidated_and_out_of_domain_calls_never_silently_return_flux(self):
        flux, valid = self.affine_grid.evaluate_with_status(0.1, 0.4, 0.6)
        self.assertFalse(bool(valid))
        self.assertTrue(np.all(np.isnan(np.asarray(flux))))
        with self.assertRaises(UnvalidatedPower2GridError):
            self.affine_grid.evaluate_checked(0.1, 0.4, 0.6)
        with self.assertRaises(Power2GridOutOfDomainError):
            self.affine_grid.evaluate_checked(
                0.2, 0.4, 0.6, allow_unvalidated=True
            )

        jitted = jax.jit(
            lambda radius: self.affine_grid.evaluate_with_status(
                radius, 0.4, 0.6, allow_unvalidated=True
            )
        )
        flux, valid = jitted(0.2)
        self.assertFalse(bool(valid))
        self.assertTrue(np.all(np.isnan(np.asarray(flux))))

    def test_invalid_axes_and_memory_guard_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            Power2TransitGrid(
                times=self.times,
                geometry=self.geometry,
                rors_knots=[0.1, 0.1],
                c_knots=[0.2, 0.7],
                alpha_knots=[0.4, 0.9],
                flux_grid=np.zeros((2, 2, 2, self.times.size)),
            )
        with self.assertRaises(MemoryError):
            build_power2_transit_grid(
                self.times,
                self.geometry,
                [0.08, 0.1],
                [0.2, 0.7],
                [0.4, 0.9],
                max_grid_bytes=1,
            )


class TestPower2GridExactValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(-0.075, 0.075, 25)
        cls.geometry = FixedPower2TransitGeometry(
            period=2.0, duration=0.12, t0=0.0, b=0.35
        )
        cls.grid = build_power2_transit_grid(
            cls.times,
            cls.geometry,
            [0.085, 0.10, 0.115],
            [0.25, 0.65],
            [0.45, 0.85],
            exact_kernel="stock",
            build_batch_size=4,
        )

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()

    def test_grid_knot_matches_existing_exact_jaxoplanet_implementation(self):
        theta = jnp.array([0.10, 0.65, 0.45])
        exact = jax.jit(
            make_exact_power2_evaluator(
                self.times, self.geometry, kernel="stock"
            )
        )
        expected = jax.block_until_ready(exact(theta))
        actual = self.grid.evaluate_checked(
            theta[0], theta[1], theta[2], allow_unvalidated=True
        )
        # Batched grid construction and a scalar exact call can choose slightly
        # different XLA contraction schedules; the observed difference is far
        # below 0.001 ppm and the validator accounts for it.
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-10)

    def test_strict_validation_rejects_a_coarse_grid(self):
        report = validate_power2_grid(
            self.grid,
            samples_per_cell=1,
            gradient_points=2,
            flux_tolerance_ppm=0.01,
            gradient_abs_tolerance_ppm_per_unit=0.01,
            gradient_relative_tolerance=1.0e-8,
            validation_batch_size=4,
        )
        self.assertFalse(report.passed)
        self.assertGreater(report.max_abs_error_ppm, report.flux_tolerance_ppm)
        with self.assertRaises(Power2GridValidationError):
            self.grid.with_validation(report)

    def test_passing_report_is_bound_to_artifact_and_survives_round_trip(self):
        report = validate_power2_grid(
            self.grid,
            samples_per_cell=1,
            gradient_points=2,
            flux_tolerance_ppm=1.0e9,
            gradient_abs_tolerance_ppm_per_unit=1.0e9,
            gradient_relative_tolerance=1.0e9,
            validation_batch_size=4,
        )
        validated = self.grid.with_validation(report)
        self.assertTrue(validated.is_validated)
        flux = validated.evaluate_checked(0.10, 0.5, 0.65)
        self.assertTrue(np.all(np.isfinite(np.asarray(flux))))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.npz"
            save_power2_grid(path, validated)
            loaded = load_power2_grid(path)
        self.assertEqual(loaded.fingerprint, validated.fingerprint)
        self.assertTrue(loaded.is_validated)
        np.testing.assert_array_equal(loaded.flux_grid, validated.flux_grid)


if __name__ == "__main__":
    unittest.main()

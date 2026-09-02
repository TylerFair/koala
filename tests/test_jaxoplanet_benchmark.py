import unittest
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from tools.benchmark_jaxoplanet_gpu import (
    assert_streamed_fidelity,
    build_parser,
    fidelity_metrics,
    load_time_grid,
    make_case,
    make_production_comparison_case,
    make_streamed_comparison_case,
    parse_int_list,
    streamed_kernel_status,
    streamed_stress_metrics,
    transit_gradient_metrics,
)


jax.config.update("jax_enable_x64", True)


class TestJaxoplanetBenchmark(unittest.TestCase):
    def tearDown(self):
        # Each test intentionally compiles different high-order kernels. Keep
        # the complete suite usable on memory-constrained CPU CI runners.
        jax.clear_caches()

    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("1, 4,8"), [1, 4, 8])

    def test_gpu_saturation_batch_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.batches, [1, 8, 40, 60, 80, 100])

    def test_static_window_power2_parity(self):
        _, _, theta, full_fwd, window_fwd, full_obj, window_obj = make_case(
            129, 3, ld_profile="power2"
        )
        metrics = fidelity_metrics(
            theta, full_fwd, window_fwd, full_obj, window_obj
        )
        self.assertLessEqual(metrics["flux_max_abs"], 5.0e-13)
        self.assertLessEqual(metrics["objective_abs"], 1.0e-10)
        self.assertLessEqual(metrics["gradient_relative_l2"], 1.0e-11)

    def test_static_window_quadratic_parity(self):
        case, _, theta, full_fwd, window_fwd, full_obj, window_obj = make_case(
            97, 2, ld_profile="quadratic"
        )
        self.assertLess(case.window_size, case.n_times)
        metrics = fidelity_metrics(
            theta, full_fwd, window_fwd, full_obj, window_obj
        )
        self.assertTrue(np.all(np.isfinite(list(metrics.values()))))
        self.assertLessEqual(metrics["flux_max_abs"], 5.0e-13)
        self.assertLessEqual(metrics["gradient_relative_l2"], 1.0e-11)

    def test_real_irregular_time_grid_parity(self):
        raw = np.cumsum(np.linspace(0.007, 0.009, 101))
        centered = raw - np.median(raw)
        case, _, theta, full_fwd, window_fwd, full_obj, window_obj = make_case(
            centered.size,
            2,
            ld_profile="power2",
            duration=0.11693087083333333,
            time_grid=centered,
        )
        self.assertEqual(case.n_times, centered.size)
        metrics = fidelity_metrics(
            theta, full_fwd, window_fwd, full_obj, window_obj
        )
        self.assertTrue(metrics["all_finite"])
        self.assertLessEqual(metrics["flux_max_abs"], 5.0e-13)

    def test_time_grid_rejects_non_monotonic_input(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            make_case(3, 1, time_grid=np.array([0.0, 1.0, 0.5]))

    def test_load_spectrodata_pickle_time_attribute(self):
        spectroscopic = np.linspace(60000.0, 60000.2, 9)
        white_light = np.linspace(60000.0, 60000.2, 5)
        payload = SimpleNamespace(time=spectroscopic, wl_time=white_light)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectroscopy_data.pkl"
            with path.open("wb") as stream:
                pickle.dump(payload, stream)
            centered, metadata = load_time_grid(
                str(path), attribute="wl_time", transit_t0=60000.1
            )
        np.testing.assert_allclose(centered, white_light - 60000.1)
        self.assertEqual(metadata["attribute"], "wl_time")
        self.assertEqual(metadata["center_source"], "--time-t0")

    def test_streamed_stress_parity_when_available(self):
        streamed, reason = streamed_kernel_status()
        if streamed is None:
            self.skipTest(reason)
        metrics = streamed_stress_metrics()
        assert_streamed_fidelity(
            metrics,
            flux_tolerance=1.0e-10,
            gradient_tolerance=5.0e-10,
        )
        self.assertTrue(metrics["all_finite"])
        self.assertEqual(
            [case["name"] for case in metrics["cases"]],
            [
                "nominal_contacts",
                "grazing_contacts",
                "power2_low_c_alpha",
                "power2_high_c_alpha",
            ],
        )

    def test_streamed_production_window_parity_when_available(self):
        streamed, reason = streamed_kernel_status()
        if streamed is None:
            self.skipTest(reason)
        (
            _,
            theta,
            stock_forward,
            streamed_forward,
            stock_objective,
            streamed_objective,
        ) = make_streamed_comparison_case(97, 2)
        metrics = fidelity_metrics(
            theta,
            stock_forward,
            streamed_forward,
            stock_objective,
            streamed_objective,
        )
        metrics.update(
            transit_gradient_metrics(theta, stock_forward, streamed_forward)
        )
        assert_streamed_fidelity(
            metrics,
            flux_tolerance=1.0e-10,
            objective_tolerance=1.0e-8,
            gradient_tolerance=5.0e-10,
        )

    def test_stock_shared_phase_production_parity_when_available(self):
        streamed, reason = streamed_kernel_status()
        if streamed is None:
            self.skipTest(reason)
        irregular = np.cumsum(np.linspace(0.0018, 0.0022, 67))
        irregular -= np.median(irregular)
        (
            _,
            theta,
            production_stock_forward,
            shared_stock_forward,
            shared_streamed_forward,
            production_stock_objective,
            shared_stock_objective,
            _,
        ) = make_production_comparison_case(
            irregular.size, 2, time_grid=irregular
        )
        phase_metrics = fidelity_metrics(
            theta,
            production_stock_forward,
            shared_stock_forward,
            production_stock_objective,
            shared_stock_objective,
        )
        phase_metrics.update(
            transit_gradient_metrics(
                theta, production_stock_forward, shared_stock_forward
            )
        )
        self.assertTrue(phase_metrics["all_finite"])
        self.assertLessEqual(phase_metrics["flux_max_abs"], 5.0e-13)
        self.assertLessEqual(
            phase_metrics["transit_gradient_max_lane_relative_l2"], 1.0e-10
        )

        stock_flux = jax.block_until_ready(jax.jit(shared_stock_forward)(theta))
        streamed_flux = jax.block_until_ready(
            jax.jit(shared_streamed_forward)(theta)
        )
        self.assertLessEqual(
            float(np.max(np.abs(np.asarray(stock_flux - streamed_flux)))),
            1.0e-10,
        )


if __name__ == "__main__":
    unittest.main()

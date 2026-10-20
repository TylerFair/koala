"""Shared numerical precision and bounded CPU parallelism for the test suite."""
import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    # Every worker compiles JAX programs; unbounded workers exhaust runner RAM.
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return min(4, cores or 1)


# Limit XLA's compilation thread pools as well as BLAS threads. JAX sizes some
# pools from CPU affinity rather than OMP_NUM_THREADS. Do this before imports.
worker = os.environ.get("PYTEST_XDIST_WORKER")
if worker and hasattr(os, "sched_getaffinity"):
    cores = sorted(os.sched_getaffinity(0))
    index = int(worker.removeprefix("gw"))
    count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "4"))
    assigned = cores[index::count]
    os.sched_setaffinity(0, assigned or {cores[index % len(cores)]})


def pytest_collection_modifyitems(items):
    # Per-case timing marks retain faster variants in the CI selection.
    # PR 05: 30.40 s; fresh full suite: 229.68 s / 287.29 s for the C++ checks.
    slow_cases = {
        "tests/test_flux_derivatives.py::TestFlux::test_flux_derivative_quad_ld",
        "tests/test_flux_derivatives.py::TestFlux::test_flux_derivative_power2_ld",
        "tests/test_marginalization_batched.py::test_all_four_candidate_targets_flux_density_and_gradients[power2-coefficients-coefficients-white]",
    }
    for item in items:
        if item.nodeid in slow_cases:
            item.add_marker(pytest.mark.slow)
        name = item.path.stem
        if "harmonica" in name or name in {
            "test_flux", "test_flux_derivatives", "test_power2_vs_jaxoplanet"
        }:
            item.add_marker(pytest.mark.harmonica)

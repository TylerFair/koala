"""Independent samplers split across CPU host devices keep the same posterior.

The host-device count is fixed when the JAX backend starts, so the sharded
runs execute in a fresh interpreter (this file run as a script) with eight
host devices and are compared there against ``lane_devices=1``. Lanes have
well-separated analytic posteriors, so a permuted, dropped, or padded lane
fails its own analytic check rather than only a pooled statistic.
"""
import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.independent_hmc import build_independent_hmc_runner, get_samples_independent_hmc
from models.independent_nuts import (
    _lane_device_layout, build_independent_nuts_runner, get_samples_independent,
)
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

HOST_DEVICES = 8
NUM_TIMES = 20
NOISE = 0.5
PRIOR_SCALE = 2.0


def _normal_model(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, PRIOR_SCALE))
    numpyro.deterministic("twice_x", 2.0 * x)
    numpyro.sample("obs", dist.Normal(x[:, None] + jnp.zeros_like(t), yerr), obs=y)


def _problem(num_lanes):
    t = jnp.linspace(-1.0, 1.0, NUM_TIMES)
    means = jnp.linspace(-3.0, 3.0, num_lanes)
    offsets = jnp.sin(7.0 * jnp.arange(NUM_TIMES))[None, :] * NOISE
    y = means[:, None] + offsets
    yerr = jnp.full(y.shape, NOISE)
    prior_loc = 0.1 * means
    precision = NUM_TIMES / NOISE**2 + 1.0 / PRIOR_SCALE**2
    exact_mean = (jnp.sum(y, axis=1) / NOISE**2 + prior_loc / PRIOR_SCALE**2) / precision
    return t, y, yerr, prior_loc, np.asarray(exact_mean), float(precision**-0.5)


SCENARIOS = {
    # 37 lanes on 8 devices: blocks of 5 and three padded lanes.
    "nuts_adaptive": dict(backend="nuts", lanes=37, width=37, chains=1, mass_matrix="adaptive",
                          num_warmup=300, num_samples=1500),
    "nuts_laplace_4chains": dict(backend="nuts", lanes=37, width=40, chains=4,
                                 mass_matrix="laplace", num_warmup=0, num_samples=1000),
    "hmc_laplace_2chains": dict(backend="hmc", lanes=13, width=13, chains=2,
                                mass_matrix="laplace", num_warmup=0, num_samples=600),
}


def _run(name, lane_devices, *, num_samples=None, runner=None):
    spec = SCENARIOS[name]
    t, y, yerr, prior_loc, _, _ = _problem(spec["lanes"])
    options = dict(
        num_warmup=spec["num_warmup"], num_samples=num_samples or spec["num_samples"],
        lane_width=spec["width"], channel_varying_kwargs=("prior_loc",),
        mass_matrix=spec["mass_matrix"], laplace_warmup=150,
        mcmc_kwargs=dict(num_chains=spec["chains"], progress_bar=False),
    )
    if spec["backend"] == "hmc":
        options["num_steps"] = 5
        build, sample = build_independent_hmc_runner, get_samples_independent_hmc
    else:
        build, sample = build_independent_nuts_runner, get_samples_independent
    if runner is None:
        runner = build(_normal_model, lane_devices=lane_devices, **options)
    samples, diagnostics = sample(
        _normal_model, jax.random.PRNGKey(17), t, yerr, y, {"x": prior_loc},
        prior_loc=prior_loc, return_diagnostics=True, _runner=runner, **options,
    )
    return runner, samples, diagnostics


def _summaries(name, samples, diagnostics):
    chains = SCENARIOS[name]["chains"]
    x = np.asarray(samples["x"])
    by_chain = np.stack([x[chain::chains] for chain in range(chains)])
    return {
        "mean": x.mean(axis=0).tolist(),
        "ess": [float(effective_sample_size(by_chain[:, :, lane])) for lane in range(x.shape[1])],
        "rhat": ([float(split_gelman_rubin(by_chain[:, :, lane])) for lane in range(x.shape[1])]
                 if chains > 1 else None),
        "divergences": int(np.asarray(diagnostics.diverging).sum()),
        "twice_x_consistent": bool(np.allclose(np.asarray(samples["twice_x"]), 2.0 * x)),
        "shape": list(x.shape),
        "step_size_shape": list(np.asarray(diagnostics.step_size).shape),
    }


def _worker(output):
    """Runs in an interpreter started with HOST_DEVICES host devices."""
    assert jax.device_count() == HOST_DEVICES, jax.devices()
    report = {"layouts": {}}
    for lanes in (1, 3, 8, 13, 37, 40):
        mesh, width = _lane_device_layout(lanes)
        report["layouts"][lanes] = [0 if mesh is None else mesh.devices.size, width]
    mesh, width = _lane_device_layout(37, 1)
    report["layouts"]["37_one_device"] = [0 if mesh is None else mesh.devices.size, width]
    real_backend = jax.default_backend
    jax.default_backend = lambda: "gpu"
    try:
        mesh, width = _lane_device_layout(37)
    finally:
        jax.default_backend = real_backend
    report["layouts"]["37_accelerator"] = [0 if mesh is None else mesh.devices.size, width]

    for name in SCENARIOS:
        for devices in (1, 2, 4, HOST_DEVICES):
            runner, samples, diagnostics = _run(name, devices)
            summary = _summaries(name, samples, diagnostics)
            summary["mesh_devices"] = 0 if runner._mesh is None else runner._mesh.devices.size
            summary["single_device_result"] = (len(samples["x"].sharding.device_set) == 1
                                               and not samples["x"].committed)
            report[f"{name}/{devices}"] = summary

    # Continuation and runner reuse with the lanes split across devices.
    runner, first, _ = _run("nuts_laplace_4chains", HOST_DEVICES, num_samples=200)
    before = np.asarray(runner._continuation[0].i)
    extra, raw = runner.continue_raw()
    _, whole, _ = _run("nuts_laplace_4chains", HOST_DEVICES, num_samples=400)
    combined = np.concatenate((np.asarray(first["x"]), np.asarray(extra["x"])))
    report["continuation_identical"] = bool(np.array_equal(combined, np.asarray(whole["x"])))
    report["continuation_steps"] = int(np.min(np.asarray(runner._continuation[0].i) - before))
    report["continuation_chains"] = int(raw["num_chains"])
    _, reused, _ = _run("nuts_laplace_4chains", HOST_DEVICES, num_samples=200, runner=runner)
    report["reuse_identical"] = bool(np.array_equal(np.asarray(reused["x"]), np.asarray(first["x"])))
    report["program_build_count"] = runner.program_build_count
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream)


@pytest.fixture(scope="module")
def sharded_report(tmp_path_factory):
    output = tmp_path_factory.mktemp("lane_sharding") / "report.json"
    environment = dict(os.environ)
    environment.update(
        JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={HOST_DEVICES}",
        PYTHONPATH=os.pathsep.join(filter(None, (ROOT, environment.get("PYTHONPATH")))),
    )
    subprocess.run([sys.executable, os.path.abspath(__file__), str(output)],
                   cwd=ROOT, env=environment, check=True, timeout=3600)
    with open(output, encoding="utf-8") as stream:
        return json.load(stream)


def test_one_device_process_keeps_single_device_programs():
    assert jax.device_count() == 1
    assert _lane_device_layout(37) == (None, 37)
    runner = build_independent_nuts_runner(_normal_model, lane_width=6, num_samples=2,
                                           mcmc_kwargs=dict(num_chains=4))
    assert runner._mesh is None and runner._device_width == runner._execution_width == 24
    with pytest.raises(ValueError, match="CPU devices"):
        _lane_device_layout(37, 2)


_ENTRY_POINT = """
import sys
from koala.host_devices import configure_cpu_host_devices, verify_cpu_host_devices
configure_cpu_host_devices(["-c", sys.argv[1]])
import jax
verify_cpu_host_devices(jax, "cpu")
import os
print(os.environ["XLA_FLAGS"].count("--xla_cpu_multi_thread_eigen"), jax.device_count())
"""


@pytest.mark.parametrize("flags, xla_flags, expected, eigen_flags, allocated_cores", [
    ({}, "", 3, 1, 3),
    ({}, "", 32, 1, 32),
    ({"cpu_host_devices": 2}, "", 2, 1, 3),
    ({"cpu_host_devices": 1}, "", 1, 0, 3),
    ({}, "--xla_force_host_platform_device_count=5", 5, 0, 3),
    ({}, "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=1", 3, 1, 3),
])
def test_cpu_entry_point_starts_one_host_device_per_core(
    tmp_path, flags, xla_flags, expected, eigen_flags, allocated_cores,
):
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"host_device": "cpu", "flags": flags}), encoding="utf-8")
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"XLA_FLAGS", "KOALA_CPU_HOST_DEVICES"}}
    environment.update(JAX_PLATFORMS="cpu", SLURM_CPUS_PER_TASK=str(allocated_cores), XLA_FLAGS=xla_flags,
                       PYTHONPATH=os.pathsep.join(filter(None, (ROOT, environment.get("PYTHONPATH")))))
    result = subprocess.run([sys.executable, "-c", _ENTRY_POINT, str(config)], cwd=ROOT,
                            env=environment, check=True, capture_output=True, text=True)
    # Automatic startup respects affinity even if Slurm advertises more CPUs.
    # A CPU dispatcher may share an A100 allocation with a GPU dispatcher.
    if not flags and "--xla_force_host_platform_device_count" not in xla_flags:
        expected = min(expected, len(os.sched_getaffinity(0)))
        # With one affinity core, startup leaves XLA_FLAGS untouched: no
        # multi-device configuration or additional Eigen flag is needed.
        if expected == 1 and "--xla_cpu_multi_thread_eigen" not in xla_flags:
            eigen_flags = 0
    # Koala adds single-threaded Eigen only with its own devices, never twice.
    assert list(map(int, result.stdout.split()[-2:])) == [eigen_flags, expected]



def test_single_affinity_core_leaves_automatic_device_flags_untouched(tmp_path):
    config = tmp_path / "one-core.yaml"
    config.write_text(json.dumps({"host_device": "cpu", "flags": {}}))
    environment = {k: v for k, v in os.environ.items()
                   if k not in {"XLA_FLAGS", "KOALA_CPU_HOST_DEVICES"}}
    environment.update(JAX_PLATFORMS="cpu", SLURM_CPUS_PER_TASK="32", XLA_FLAGS="")
    core = min(os.sched_getaffinity(0))
    result = subprocess.run(
        ["taskset", "-c", str(core), sys.executable, "-c", _ENTRY_POINT, str(config)], cwd=ROOT,
        env=environment, check=True, capture_output=True, text=True,
    )
    assert list(map(int, result.stdout.split()[-2:])) == [0, 1]


@pytest.mark.parametrize("affinity, slurm, expected", [
    (32, "32", 32),
    (10, "32", 10),
    (32, "16", 16),
    (32, None, 32),
])
def test_available_cpu_cores_respects_allocation_and_affinity(monkeypatch, affinity, slurm, expected):
    from koala import host_devices
    monkeypatch.setattr(host_devices.os, "sched_getaffinity", lambda _: set(range(affinity)))
    if slurm is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurm)
    assert host_devices.available_cpu_cores() == expected


@pytest.mark.slow
def test_layout_uses_fewest_devices_and_never_shards_accelerators(sharded_report):
    layouts = sharded_report["layouts"]
    assert layouts["1"] == [0, 1]
    assert layouts["3"] == [3, 3]
    assert layouts["8"] == [8, 8]
    assert layouts["13"] == [7, 14]
    assert layouts["37"] == [8, 40]
    assert layouts["40"] == [8, 40]
    assert layouts["37_one_device"] == [0, 37]
    assert layouts["37_accelerator"] == [0, 37]


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_sharded_lanes_match_analytic_and_single_device_posteriors(sharded_report, name):
    spec = SCENARIOS[name]
    *_, exact_mean, exact_sd = _problem(spec["lanes"])
    reference = sharded_report[f"{name}/1"]
    assert reference["mesh_devices"] == 0
    for devices in (1, 2, 4, HOST_DEVICES):
        run = sharded_report[f"{name}/{devices}"]
        per_device = -(-spec["width"] // devices)
        assert run["mesh_devices"] == (0 if devices == 1 else -(-spec["width"] // per_device))
        assert run["single_device_result"]
        assert run["shape"] == [spec["num_samples"] * spec["chains"], spec["lanes"]]
        assert run["step_size_shape"] == [spec["lanes"]]
        assert run["twice_x_consistent"]
        assert run["divergences"] == 0
        mean, ess = np.asarray(run["mean"]), np.asarray(run["ess"])
        assert np.all(ess > 300), (devices, ess.min())
        # Each lane against its own exact posterior: five Monte Carlo errors.
        assert np.all(np.abs(mean - exact_mean) < 5.0 * exact_sd / np.sqrt(ess))
        if run["rhat"] is not None:
            assert max(run["rhat"]) < 1.01, (devices, max(run["rhat"]))
        shift = np.abs(mean - np.asarray(reference["mean"])) / exact_sd
        joint_error = np.sqrt(1.0 / ess + 1.0 / np.asarray(reference["ess"]))
        assert np.all(shift < 5.0 * joint_error), shift.max()
        assert np.median(shift) < 0.1
        assert 0.7 < np.median(ess / np.asarray(reference["ess"])) < 1.4


@pytest.mark.slow
def test_sharded_continuation_and_reuse_are_exact(sharded_report):
    assert sharded_report["continuation_identical"]
    assert sharded_report["continuation_steps"] == 200
    assert sharded_report["continuation_chains"] == 4
    assert sharded_report["reuse_identical"]
    assert sharded_report["program_build_count"] == 1


if __name__ == "__main__":
    _worker(sys.argv[1])

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

import diagnose_nuts_gradient
import fit_jwst


def _factorized_normal_model(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, 1.0))
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def test_reused_runner_matches_fresh_runner_after_prior_chunk(monkeypatch):
    """A completed chunk must not leak its state or adaptation to the next."""
    monkeypatch.setattr(fit_jwst, "_save_mcmc_diagnostics", lambda *a, **k: {})

    t = jnp.arange(3.0, dtype=jnp.float64)
    yerr = jnp.full((2, 3), 0.2, dtype=jnp.float64)
    first_y = jnp.asarray([[-2.1] * 3, [-1.9] * 3])
    second_y = jnp.asarray([[2.1] * 3, [1.9] * 3])
    first_loc = jnp.asarray([-2.0, -2.0])
    second_loc = jnp.asarray([2.0, 2.0])
    first_init = {"x": jnp.asarray([-2.2, -1.8])}
    second_init = {"x": jnp.asarray([2.2, 1.8])}
    mcmc_kwargs = {
        "num_warmup": 8,
        "num_samples": 8,
        "progress_bar": False,
        "jit_model_args": True,
    }

    runner, _, _ = fit_jwst._build_numpyro_mcmc(
        _factorized_normal_model, first_init, {}, mcmc_kwargs
    )
    fit_jwst.get_samples(
        _factorized_normal_model,
        jax.random.PRNGKey(1),
        t,
        yerr,
        first_y,
        first_init,
        mcmc_kwargs=mcmc_kwargs,
        _mcmc_runner=runner,
        prior_loc=first_loc,
    )
    reused = fit_jwst.get_samples(
        _factorized_normal_model,
        jax.random.PRNGKey(2),
        t,
        yerr,
        second_y,
        second_init,
        mcmc_kwargs=mcmc_kwargs,
        _mcmc_runner=runner,
        prior_loc=second_loc,
    )
    fresh = fit_jwst.get_samples(
        _factorized_normal_model,
        jax.random.PRNGKey(2),
        t,
        yerr,
        second_y,
        second_init,
        mcmc_kwargs=mcmc_kwargs,
        prior_loc=second_loc,
    )

    np.testing.assert_array_equal(np.asarray(reused["x"]), np.asarray(fresh["x"]))
    assert runner._warmup_state is None
    assert len(runner._init_state_cache) == 0


def test_chunk_cache_is_partitioned_by_resident_width(monkeypatch):
    builds = []
    calls = []

    def fake_build(model, init_params, nuts_kwargs, mcmc_kwargs):
        runner = object()
        builds.append((int(init_params["x"].shape[0]), runner))
        return runner, {}, {}

    def fake_get_samples(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        _mcmc_runner=None,
        **kwargs,
    ):
        calls.append((int(init_params["x"].shape[0]), _mcmc_runner))
        return {"x": jnp.asarray(init_params["x"])[None, :]}

    monkeypatch.setattr(fit_jwst, "_build_numpyro_mcmc", fake_build)
    monkeypatch.setattr(fit_jwst, "get_samples", fake_get_samples)

    samples = fit_jwst.get_samples_chunked(
        lambda *a, **k: None,
        jax.random.PRNGKey(0),
        jnp.arange(3.0),
        jnp.ones((5, 3)),
        jnp.zeros((5, 3)),
        {"x": jnp.arange(5.0)},
        chunk_size=2,
        mcmc_kwargs={"jit_model_args": True},
    )

    assert [width for width, _ in builds] == [2, 1]
    assert [width for width, _ in calls] == [2, 2, 1]
    assert calls[0][1] is calls[1][1]
    assert calls[2][1] is not calls[0][1]
    np.testing.assert_array_equal(np.asarray(samples["x"]), np.arange(5.0)[None, :])


def test_selective_fallback_programs_reuse_fixed_resident_width(monkeypatch):
    import models.independent_hmc as independent_hmc
    import models.independent_nuts as independent_nuts

    builds = {"nuts": 0, "hmc": 0, "joint": 0}
    joint_widths = []
    gate_call = 0

    def fake_build_nuts(*args, **kwargs):
        builds["nuts"] += 1
        return object()

    def fake_build_hmc(*args, **kwargs):
        builds["hmc"] += 1
        return object()

    def fake_independent(model, key, t, yerr, y, init, **kwargs):
        return {"depths": jnp.ones((8, yerr.shape[0], 1))}

    def fake_gate(samples, path, min_ess, max_divergences):
        nonlocal gate_call
        phase = gate_call % 3
        gate_call += 1
        failed = (
            np.arange(samples["depths"].shape[1], dtype=int)
            if phase < 2 else np.asarray([], dtype=int)
        )
        return failed, {
            "depth_ess_per_channel": [999.0] * samples["depths"].shape[1],
            "num_divergences_per_channel": [0] * samples["depths"].shape[1],
        }

    def fake_build_joint(*args, **kwargs):
        builds["joint"] += 1
        return object(), {}, {}

    def fake_joint(model, key, t, yerr, y, init, **kwargs):
        joint_widths.append(int(yerr.shape[0]))
        return {"depths": jnp.ones((8, yerr.shape[0], 1))}

    monkeypatch.setattr(independent_nuts, "build_independent_nuts_runner", fake_build_nuts)
    monkeypatch.setattr(independent_nuts, "get_samples_independent", fake_independent)
    monkeypatch.setattr(independent_hmc, "build_independent_hmc_runner", fake_build_hmc)
    monkeypatch.setattr(independent_hmc, "get_samples_independent_hmc", fake_independent)
    monkeypatch.setattr(fit_jwst, "_spectro_failed_lanes", fake_gate)
    monkeypatch.setattr(fit_jwst, "_build_numpyro_mcmc", fake_build_joint)
    monkeypatch.setattr(fit_jwst, "get_samples", fake_joint)

    independent_cache = {}
    joint_cache = {}
    common = dict(
        model=lambda *args, **kwargs: None,
        key=jax.random.PRNGKey(7),
        t=jnp.arange(3.0),
        yerr=jnp.ones((8, 3)),
        indiv_y=jnp.zeros((8, 3)),
        init_params={"depths": jnp.full((8, 1), 0.01)},
        chunk_size=8,
        sampler_backend="independent_nuts",
        mcmc_kwargs={"num_samples": 8},
        spectro_min_depth_ess=400,
        adaptive_fallback_resident_width=4,
        _independent_mcmc_runner_cache=independent_cache,
        _joint_mcmc_runner_cache=joint_cache,
    )
    fit_jwst.get_samples_chunked(**common)
    fit_jwst.get_samples_chunked(**{**common, "key": jax.random.PRNGKey(8)})

    assert builds == {"nuts": 1, "hmc": 1, "joint": 1}
    assert joint_widths == [4, 4, 4, 4]


def test_reused_runner_respects_explicit_init_strategy(monkeypatch):
    class FakeSampler:
        _init_strategy = None

    class FakeMCMC:
        sampler = FakeSampler()

        def run(self, *args, **kwargs):
            return None

        def get_samples(self):
            return {"x": jnp.zeros((1, 1))}

    monkeypatch.setattr(fit_jwst, "_save_mcmc_diagnostics", lambda *a, **k: {})
    explicit_strategy = numpyro.infer.init_to_feasible()
    runner = FakeMCMC()

    fit_jwst.get_samples(
        _factorized_normal_model,
        jax.random.PRNGKey(0),
        jnp.zeros(1),
        jnp.ones((1, 1)),
        jnp.zeros((1, 1)),
        {"x": jnp.asarray([4.0])},
        nuts_kwargs={"init_strategy": explicit_strategy},
        _mcmc_runner=runner,
        prior_loc=jnp.zeros(1),
    )

    assert runner.sampler._init_strategy is explicit_strategy


def test_checkpoint_fingerprint_changes_with_data_and_science_signature():
    common = {
        "model": _factorized_normal_model,
        "rng_key": jax.random.key_data(jax.random.PRNGKey(0)),
        "t": jnp.arange(3.0),
        "yerr": jnp.ones((2, 3)),
        "indiv_y": jnp.zeros((2, 3)),
        "init_params": {"x": jnp.zeros(2)},
        "chunk_size": 2,
        "sampler_backend": "joint_nuts",
        "nuts_kwargs": {"dense_mass": True},
        "mcmc_kwargs": {"num_samples": 10},
        "channel_varying_kwargs": ("prior_loc",),
        "checkpoint_signature": {"ld_profile": "quadratic"},
        "model_kwargs": {"prior_loc": jnp.zeros(2)},
    }
    reference = fit_jwst._chunk_checkpoint_fingerprint(**common)
    changed_data = fit_jwst._chunk_checkpoint_fingerprint(
        **{**common, "indiv_y": jnp.ones((2, 3))}
    )
    changed_signature = fit_jwst._chunk_checkpoint_fingerprint(
        **{
            **common,
            "checkpoint_signature": {"ld_profile": "power2"},
        }
    )
    changed_seed = fit_jwst._chunk_checkpoint_fingerprint(
        **{
            **common,
            "rng_key": jax.random.key_data(jax.random.PRNGKey(1)),
        }
    )
    assert reference != changed_data
    assert reference != changed_signature
    assert reference != changed_seed
    changed_fallback_width = fit_jwst._chunk_checkpoint_fingerprint(
        **{**common, "adaptive_fallback_resident_width": 4}
    )
    assert reference != changed_fallback_width


def test_checkpoint_fingerprint_changes_with_ld_parameterization():
    def model():
        return None

    common = {
        "rng_key": jax.random.key_data(jax.random.PRNGKey(0)),
        "t": jnp.arange(3.0),
        "yerr": jnp.ones((1, 3)),
        "indiv_y": jnp.zeros((1, 3)),
        "init_params": {"x": jnp.zeros(1)},
        "chunk_size": 1,
        "sampler_backend": "independent_nuts",
        "nuts_kwargs": {"mass_matrix": "laplace"},
        "mcmc_kwargs": {"num_samples": 10},
        "channel_varying_kwargs": (),
        "checkpoint_signature": {},
        "model_kwargs": {},
    }
    model.__sampler_input_builder__ = {
        "identity": "tests.fake_builder",
        "kwargs": {"ld_parameterization": "coefficients"},
        "callable_is_model": False,
    }
    coefficients = fit_jwst._chunk_checkpoint_fingerprint(model=model, **common)
    model.__sampler_input_builder__["kwargs"]["ld_parameterization"] = "decorrelated"
    decorrelated = fit_jwst._chunk_checkpoint_fingerprint(model=model, **common)
    assert coefficients != decorrelated


def test_chunk_keys_are_global_indexed_and_resume_independent(monkeypatch, tmp_path):
    observed_keys = {}

    def fake_get_samples(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        **kwargs,
    ):
        start = int(np.asarray(init_params["x"])[0])
        observed_keys[start] = np.asarray(jax.random.key_data(key)).copy()
        return {"x": jnp.asarray(init_params["x"])[None, :]}

    monkeypatch.setattr(fit_jwst, "get_samples", fake_get_samples)
    base_key = jax.random.PRNGKey(71)
    common = dict(
        model=lambda *a, **k: None,
        key=base_key,
        t=jnp.arange(3.0),
        yerr=jnp.ones((6, 3)),
        indiv_y=jnp.zeros((6, 3)),
        init_params={"x": jnp.arange(6.0)},
        chunk_size=2,
        mcmc_kwargs={"jit_model_args": False},
        output_dir=str(tmp_path),
        checkpoint_prefix="rng",
    )

    fit_jwst.get_samples_chunked(**common)
    first_run = {name: value.copy() for name, value in observed_keys.items()}
    assert len({tuple(value) for value in first_run.values()}) == 3
    for chunk_idx, start in enumerate((0, 2, 4)):
        np.testing.assert_array_equal(
            first_run[start],
            np.asarray(jax.random.key_data(jax.random.fold_in(base_key, chunk_idx))),
        )

    # Remove only the last checkpoint.  A resume must give that chunk the
    # same stream it had when all previous chunks were computed.
    last_checkpoint = next((tmp_path / "chunks").glob("rng_cfg*_chunk_4_6.pkl"))
    last_checkpoint.unlink()
    observed_keys.clear()
    fit_jwst.get_samples_chunked(**common)
    assert set(observed_keys) == {4}
    np.testing.assert_array_equal(observed_keys[4], first_run[4])

    # A preempted/direct partial pickle is detected and recomputed; it is not
    # trusted merely because the filename exists.
    corrupt_checkpoint = next(
        (tmp_path / "chunks").glob("rng_cfg*_chunk_2_4.pkl")
    )
    corrupt_checkpoint.write_bytes(b"partial pickle")
    observed_keys.clear()
    fit_jwst.get_samples_chunked(**common)
    assert set(observed_keys) == {2}
    np.testing.assert_array_equal(observed_keys[2], first_run[2])


def test_each_chunk_gradient_diagnostic_is_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit_jwst,
        "get_samples",
        lambda model, key, t, yerr, indiv_y, init_params, **kwargs: {
            "x": jnp.asarray(init_params["x"])[None, :]
        },
    )
    t = jnp.arange(3.0, dtype=jnp.float64)
    fit_jwst.get_samples_chunked(
        _factorized_normal_model,
        jax.random.PRNGKey(3),
        t,
        jnp.full((2, 3), 0.2),
        jnp.zeros((2, 3)),
        {"x": jnp.zeros(2)},
        chunk_size=1,
        mcmc_kwargs={"jit_model_args": False},
        output_dir=str(tmp_path),
        checkpoint_prefix="gradient",
        gradient_diagnostic_mode="each",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=jnp.zeros(2),
    )

    paths = sorted((tmp_path / "chunks").glob("gradient_cfg*_chunk_*.gradient.json"))
    assert len(paths) == 2
    for path in paths:
        payload = __import__("json").loads(path.read_text())
        assert payload["passed"]
        assert payload["all_finite"]


def test_first_gradient_diagnostic_runs_for_cached_chunk_and_is_reused(
    monkeypatch, tmp_path
):
    calls = []

    monkeypatch.setattr(
        fit_jwst,
        "get_samples",
        lambda model, key, t, yerr, indiv_y, init_params, **kwargs: {
            "x": jnp.asarray(init_params["x"])[None, :]
        },
    )

    def fake_diagnostic(*args, **kwargs):
        calls.append(True)
        return {
            "passed": True,
            "all_finite": True,
            "max_directional_relative_error": 1e-12,
        }

    monkeypatch.setattr(
        diagnose_nuts_gradient, "diagnose_gradient_quality", fake_diagnostic
    )
    common = dict(
        model=_factorized_normal_model,
        key=jax.random.PRNGKey(9),
        t=jnp.arange(3.0),
        yerr=jnp.full((1, 3), 0.2),
        indiv_y=jnp.zeros((1, 3)),
        init_params={"x": jnp.zeros(1)},
        chunk_size=1,
        mcmc_kwargs={"jit_model_args": False},
        output_dir=str(tmp_path),
        checkpoint_prefix="cached_gradient",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=jnp.zeros(1),
    )

    # Produce the checkpoint without a diagnostic, then request one. The
    # checkpoint must not bypass the requested safety check.
    fit_jwst.get_samples_chunked(**common)
    fit_jwst.get_samples_chunked(
        **common,
        gradient_diagnostic_mode="first",
        gradient_diagnostic_strict=True,
    )
    assert len(calls) == 1

    # A fingerprinted successful result can be reused without recompilation.
    fit_jwst.get_samples_chunked(
        **common,
        gradient_diagnostic_mode="first",
        gradient_diagnostic_strict=True,
    )
    assert len(calls) == 1


def test_strict_gradient_diagnostic_rejects_cached_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fit_jwst,
        "get_samples",
        lambda model, key, t, yerr, indiv_y, init_params, **kwargs: {
            "x": jnp.asarray(init_params["x"])[None, :]
        },
    )
    monkeypatch.setattr(
        diagnose_nuts_gradient,
        "diagnose_gradient_quality",
        lambda *args, **kwargs: {
            "passed": False,
            "all_finite": False,
            "max_directional_relative_error": float("inf"),
        },
    )
    common = dict(
        model=_factorized_normal_model,
        key=jax.random.PRNGKey(10),
        t=jnp.arange(3.0),
        yerr=jnp.full((1, 3), 0.2),
        indiv_y=jnp.zeros((1, 3)),
        init_params={"x": jnp.zeros(1)},
        chunk_size=1,
        mcmc_kwargs={"jit_model_args": False},
        output_dir=str(tmp_path),
        checkpoint_prefix="failed_gradient",
        gradient_diagnostic_mode="first",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=jnp.zeros(1),
    )

    fit_jwst.get_samples_chunked(**common)
    with pytest.raises(RuntimeError, match="gradient diagnostic failed"):
        fit_jwst.get_samples_chunked(
            **common, gradient_diagnostic_strict=True
        )

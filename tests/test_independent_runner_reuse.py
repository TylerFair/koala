import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

import fit_jwst
import models.independent_nuts as independent_nuts
from models.channel_batching import PilotDiagnostics, build_difficulty_batch_plan
from models.independent_hmc import (
    build_independent_hmc_runner,
    get_samples_independent_hmc,
)
from models.independent_nuts import (
    build_independent_nuts_runner,
    get_samples_independent,
)


def _dynamic_normal_model(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, 0.7))
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.deterministic("twice_x", 2.0 * x)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


@pytest.mark.parametrize("backend", ["nuts", "hmc"])
def test_reused_independent_runner_has_no_stale_data_or_initial_state(backend):
    t = jnp.linspace(-1.0, 1.0, 4)
    yerr = jnp.full((2, 4), 0.25)
    first_y = jnp.asarray([[-2.2] * 4, [-1.8] * 4])
    second_y = jnp.asarray([[2.3] * 4, [1.7] * 4])
    first_loc = jnp.asarray([-2.0, -2.0])
    second_loc = jnp.asarray([2.0, 2.0])
    first_init = {"x": jnp.asarray([-2.4, -1.6])}
    second_init = {"x": jnp.asarray([2.4, 1.6])}
    common = {
        "num_warmup": 8,
        "num_samples": 10,
        "lane_width": 3,
        "channel_varying_kwargs": ("prior_loc",),
    }
    if backend == "nuts":
        sampler = get_samples_independent
        builder = build_independent_nuts_runner
        common["max_tree_depth"] = 5
    else:
        sampler = get_samples_independent_hmc
        builder = build_independent_hmc_runner
        common["num_steps"] = 5

    runner = builder(_dynamic_normal_model, **common)
    sampler(
        _dynamic_normal_model,
        jax.random.PRNGKey(20),
        t,
        yerr,
        first_y,
        first_init,
        prior_loc=first_loc,
        _runner=runner,
        **common,
    )
    reused = sampler(
        _dynamic_normal_model,
        jax.random.PRNGKey(21),
        t,
        yerr,
        second_y,
        second_init,
        prior_loc=second_loc,
        _runner=runner,
        **common,
    )
    fresh = sampler(
        _dynamic_normal_model,
        jax.random.PRNGKey(21),
        t,
        yerr,
        second_y,
        second_init,
        prior_loc=second_loc,
        **common,
    )

    assert runner.program_build_count == 1
    np.testing.assert_array_equal(np.asarray(reused["x"]), np.asarray(fresh["x"]))
    np.testing.assert_array_equal(
        np.asarray(reused["twice_x"]), np.asarray(fresh["twice_x"])
    )
    assert float(np.asarray(reused["x"]).mean()) > 1.0


def test_fit_chunk_router_reuses_one_independent_runner_per_padded_width(
    monkeypatch,
):
    built = []
    calls = []

    def fake_builder(model, *, lane_width, **kwargs):
        runner = object()
        built.append((model, lane_width, runner, kwargs))
        return runner

    def fake_sampler(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        *,
        lane_width,
        _runner,
        **kwargs,
    ):
        del model, key, t, yerr, kwargs
        calls.append((lane_width, _runner, np.asarray(init_params["x"])))
        return {"x": jnp.asarray(indiv_y[:, 0])[None, :]}

    monkeypatch.setattr(
        independent_nuts, "build_independent_nuts_runner", fake_builder
    )
    monkeypatch.setattr(
        independent_nuts, "get_samples_independent", fake_sampler
    )
    samples = fit_jwst.get_samples_chunked(
        _dynamic_normal_model,
        jax.random.PRNGKey(3),
        jnp.arange(3.0),
        jnp.ones((5, 3)),
        jnp.broadcast_to(jnp.arange(5.0)[:, None], (5, 3)),
        {"x": jnp.arange(5.0)},
        chunk_size=2,
        sampler_backend="independent_nuts",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=jnp.arange(5.0),
        mcmc_kwargs={"num_warmup": 1, "num_samples": 1},
    )

    assert len(built) == 1
    assert built[0][1] == 2
    assert [width for width, _, _ in calls] == [2, 2, 2]
    assert all(runner is built[0][2] for _, runner, _ in calls)
    np.testing.assert_array_equal(
        np.asarray(samples["x"]), np.arange(5.0)[None, :]
    )


def test_difficulty_plan_shares_same_width_runner_across_inner_calls(
    monkeypatch,
):
    pilot = PilotDiagnostics(
        channel_indices=(0, 1, 2, 3),
        mean_num_steps=(8.0, 9.0, 200.0, 7.0),
        max_num_steps=(15.0, 17.0, 511.0, 13.0),
        num_divergences=(0, 0, 1, 0),
        step_size=(0.2, 0.19, 0.01, 0.21),
        num_draws=20,
    )
    plan = build_difficulty_batch_plan(
        pilot,
        nominal_width=2,
        quarantine_width=1,
    )
    builds = []
    calls = []

    def fake_builder(model, *, lane_width, **kwargs):
        del model, kwargs
        runner = object()
        builds.append((lane_width, runner))
        return runner

    def fake_sampler(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        *,
        lane_width,
        _runner,
        **kwargs,
    ):
        del model, key, t, yerr, init_params, kwargs
        calls.append((lane_width, _runner))
        return {"x": jnp.asarray(indiv_y[:, 0])[None, :]}

    monkeypatch.setattr(
        independent_nuts, "build_independent_nuts_runner", fake_builder
    )
    monkeypatch.setattr(
        independent_nuts, "get_samples_independent", fake_sampler
    )
    channel_id = jnp.arange(4.0)
    samples = fit_jwst.get_samples_channel_plan(
        _dynamic_normal_model,
        jax.random.PRNGKey(9),
        jnp.arange(3.0),
        jnp.ones((4, 3)),
        jnp.broadcast_to(channel_id[:, None], (4, 3)),
        {"x": channel_id},
        plan,
        sampler_backend="independent_nuts",
        mcmc_kwargs={"num_warmup": 1, "num_samples": 1},
    )

    assert [width for width, _ in builds] == [1, 2]
    assert [width for width, _ in calls] == [1, 2, 2]
    width_two_runner = next(runner for width, runner in builds if width == 2)
    assert all(
        runner is width_two_runner
        for width, runner in calls
        if width == 2
    )
    np.testing.assert_array_equal(
        np.asarray(samples["x"]), np.arange(4.0)[None, :]
    )


def test_difficulty_plan_also_shares_joint_runner_cache(monkeypatch):
    pilot = PilotDiagnostics(
        channel_indices=(0, 1, 2, 3),
        mean_num_steps=(8.0, 9.0, 200.0, 7.0),
        max_num_steps=(15.0, 17.0, 511.0, 13.0),
        num_divergences=(0, 0, 1, 0),
        step_size=(0.2, 0.19, 0.01, 0.21),
        num_draws=20,
    )
    plan = build_difficulty_batch_plan(
        pilot,
        nominal_width=2,
        quarantine_width=1,
    )
    builds = []
    calls = []

    def fake_build(model, init_params, nuts_kwargs, mcmc_kwargs):
        del model, nuts_kwargs, mcmc_kwargs
        width = int(init_params["x"].shape[0])
        runner = object()
        builds.append((width, runner))
        return runner, {}, {}

    def fake_get_samples(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        *,
        _mcmc_runner,
        **kwargs,
    ):
        del model, key, t, yerr, init_params, kwargs
        calls.append((int(indiv_y.shape[0]), _mcmc_runner))
        return {"x": jnp.asarray(indiv_y[:, 0])[None, :]}

    monkeypatch.setattr(fit_jwst, "_build_numpyro_mcmc", fake_build)
    monkeypatch.setattr(fit_jwst, "get_samples", fake_get_samples)
    channel_id = jnp.arange(4.0)
    samples = fit_jwst.get_samples_channel_plan(
        _dynamic_normal_model,
        jax.random.PRNGKey(10),
        jnp.arange(3.0),
        jnp.ones((4, 3)),
        jnp.broadcast_to(channel_id[:, None], (4, 3)),
        {"x": channel_id},
        plan,
        sampler_backend="joint_nuts",
        mcmc_kwargs={
            "num_warmup": 1,
            "num_samples": 1,
            "jit_model_args": True,
        },
    )

    assert [width for width, _ in builds] == [1, 2]
    assert [width for width, _ in calls] == [1, 2, 1]
    width_one_runner = next(runner for width, runner in builds if width == 1)
    assert calls[0][1] is width_one_runner
    assert calls[2][1] is width_one_runner
    np.testing.assert_array_equal(
        np.asarray(samples["x"]), np.arange(4.0)[None, :]
    )

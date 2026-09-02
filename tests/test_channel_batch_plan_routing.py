import numpy as np

import fit_jwst
from models.channel_batching import PilotDiagnostics, build_difficulty_batch_plan


def _plan():
    pilot = PilotDiagnostics(
        channel_indices=(0, 1, 2, 3),
        mean_num_steps=(8.0, 9.0, 200.0, 7.0),
        max_num_steps=(15.0, 17.0, 511.0, 13.0),
        num_divergences=(0, 0, 1, 0),
        step_size=(0.2, 0.19, 0.01, 0.21),
        num_draws=20,
    )
    return build_difficulty_batch_plan(
        pilot,
        nominal_width=2,
        quarantine_width=1,
    )


def test_planned_batches_take_arbitrary_channels_and_restore_order(monkeypatch):
    plan = _plan()
    calls = []

    def fake_get_samples_chunked(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        chunk_size,
        **kwargs,
    ):
        del model, key, t
        calls.append(
            {
                "channel_values": np.asarray(indiv_y)[:, 0].copy(),
                "errors": np.asarray(yerr)[:, 0].copy(),
                "init": np.asarray(init_params["theta"]).copy(),
                "prior": np.asarray(kwargs["trend_prior_mean"]).copy(),
                "shared": np.asarray(kwargs["mu_t0"]).copy(),
                "width": chunk_size,
            }
        )
        # The real independent runner trims padding before returning.
        return {"theta": np.asarray(indiv_y)[:, 0][None, :]}

    monkeypatch.setattr(fit_jwst, "get_samples_chunked", fake_get_samples_chunked)
    channel_id = np.arange(4.0)
    samples = fit_jwst.get_samples_channel_plan(
        object(),
        fit_jwst.jax.random.PRNGKey(3),
        np.arange(3.0),
        np.broadcast_to(channel_id[:, None] + 0.1, (4, 3)),
        np.broadcast_to(channel_id[:, None], (4, 3)),
        {"theta": channel_id + 10.0},
        plan,
        chunk_mode="serial",
        sampler_backend="independent_nuts",
        channel_varying_kwargs=("trend_prior_mean",),
        trend_prior_mean=np.column_stack((channel_id, channel_id + 20.0)),
        # Its leading dimension happens to equal num_channels, but this is a
        # shared geometry vector and must never be inferred as channel-varying.
        mu_t0=channel_id + 100.0,
    )

    assert np.array_equal(samples["theta"], channel_id[None, :])
    assert [call["width"] for call in calls] == [
        batch.lane_width for batch in plan.batches
    ]
    for call, batch in zip(calls, plan.batches):
        indices = np.asarray(batch.channel_indices)
        assert np.array_equal(call["channel_values"], channel_id[indices])
        assert np.array_equal(call["init"], channel_id[indices] + 10.0)
        assert np.array_equal(call["prior"][:, 0], channel_id[indices])
        assert np.array_equal(call["shared"], channel_id + 100.0)


def test_plain_chunks_slice_only_explicit_channel_kwargs(monkeypatch):
    calls = []

    def fake_get_samples(
        model, key, t, yerr, y, init_params, **kwargs
    ):
        del model, key, t, yerr, init_params
        calls.append(
            {
                "shared": np.asarray(kwargs["shared_geometry"]).copy(),
                "varying": np.asarray(kwargs["varying_prior"]).copy(),
            }
        )
        return {"theta": np.asarray(y)[:, 0][None, :]}

    monkeypatch.setattr(fit_jwst, "get_samples", fake_get_samples)
    channel_id = np.arange(4.0)
    samples = fit_jwst.get_samples_chunked(
        object(),
        fit_jwst.jax.random.PRNGKey(8),
        np.arange(3.0),
        np.ones((4, 3)),
        np.broadcast_to(channel_id[:, None], (4, 3)),
        {"theta": channel_id},
        2,
        mcmc_kwargs={"jit_model_args": False},
        channel_varying_kwargs=("varying_prior",),
        varying_prior=channel_id + 10.0,
        shared_geometry=channel_id + 100.0,
    )

    assert np.array_equal(samples["theta"], channel_id[None, :])
    assert len(calls) == 2
    for index, call in enumerate(calls):
        assert np.array_equal(
            call["varying"], channel_id[index * 2:(index + 1) * 2] + 10.0
        )
        assert np.array_equal(call["shared"], channel_id + 100.0)


def test_planned_parallel_job_only_executes_assigned_batches(monkeypatch, tmp_path):
    plan = _plan()
    called = []

    def fake_get_samples_chunked(*args, **kwargs):
        called.append(kwargs["checkpoint_prefix"])
        active = np.asarray(args[5]["theta"]).shape[0]
        return {"theta": np.zeros((1, active))}

    monkeypatch.setattr(fit_jwst, "get_samples_chunked", fake_get_samples_chunked)
    result = fit_jwst.get_samples_channel_plan(
        object(),
        fit_jwst.jax.random.PRNGKey(4),
        np.arange(3.0),
        np.ones((4, 3)),
        np.ones((4, 3)),
        {"theta": np.arange(4.0)},
        plan,
        chunk_mode="parallel",
        parallel_job_count=2,
        parallel_job_index=1,
        output_dir=tmp_path,
        checkpoint_prefix="test",
    )
    assert result is None
    assert len(called) == len(plan.batches[1::2])

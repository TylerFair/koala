import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from fit_jwst import _run_sampling_stage


def _factorized_normal_model(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, 1.0))
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def test_fit_sampling_router_independent_chunks_and_padding(tmp_path):
    num_channels, num_times = 3, 4
    t = jnp.arange(num_times, dtype=jnp.float64)
    yerr = jnp.ones((num_channels, num_times), dtype=jnp.float64)
    y = jnp.asarray([[0.1] * 4, [0.2] * 4, [0.3] * 4])
    prior_loc = jnp.zeros(num_channels)

    samples = _run_sampling_stage(
        _factorized_normal_model,
        jax.random.PRNGKey(1),
        t,
        yerr,
        y,
        {"x": prior_loc},
        nuts_kwargs={"dense_mass": True},
        mcmc_kwargs={"num_warmup": 5, "num_samples": 5},
        use_chunked=True,
        chunk_size=2,
        output_dir=str(tmp_path),
        checkpoint_prefix="toy",
        sampler_backend="independent_nuts",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=prior_loc,
    )

    assert samples["x"].shape == (5, num_channels)
    assert jnp.all(jnp.isfinite(samples["x"]))
    checkpoint_dir = tmp_path / "chunks"
    assert len(
        list(checkpoint_dir.glob("toy_independent_nuts_cfg*_chunk_*.pkl"))
    ) == 2
    assert len(
        list(
            checkpoint_dir.glob(
                "toy_independent_nuts_cfg*_chunk_*.diagnostics.json"
            )
        )
    ) == 2
    for diagnostic_path in checkpoint_dir.glob(
        "toy_independent_nuts_cfg*_chunk_*.diagnostics.json"
    ):
        payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        assert payload["diagnostic_provenance_schema_version"] == 1
        assert len(payload["sampling_workload_fingerprint_sha256"]) == 64
        assert payload["sampler_backend"] == "independent_nuts"
    manifests = list(
        checkpoint_dir.glob("toy_independent_nuts_cfg*.manifest.json")
    )
    assert len(manifests) == 1

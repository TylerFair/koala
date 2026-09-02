import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import fit_jwst
from models.independent_hmc import get_samples_independent_hmc


def _normal_channels(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, 2.0))
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.deterministic("twice_x", 2.0 * x)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


def test_independent_hmc_matches_normal_posterior_and_fixed_work(tmp_path):
    channels, times = 3, 5
    t = jnp.linspace(-1.0, 1.0, times)
    means = jnp.array([-1.5, 0.25, 2.0])
    y = jnp.broadcast_to(means[:, None], (channels, times))
    yerr = jnp.ones_like(y)
    prior_loc = jnp.array([-0.5, 0.0, 0.5])
    diagnostics_path = tmp_path / "hmc.json"

    samples, diagnostics = get_samples_independent_hmc(
        _normal_channels,
        jax.random.PRNGKey(31),
        t,
        yerr,
        y,
        {"x": prior_loc},
        prior_loc=prior_loc,
        lane_width=4,
        num_warmup=250,
        num_samples=1000,
        num_steps=12,
        channel_varying_kwargs=("prior_loc",),
        diagnostics_path=str(diagnostics_path),
        return_diagnostics=True,
    )

    expected = (times * means + 0.25 * prior_loc) / (times + 0.25)
    assert samples["x"].shape == (1000, channels)
    assert jnp.allclose(samples["x"].mean(axis=0), expected, atol=0.12)
    assert jnp.allclose(samples["twice_x"], 2.0 * samples["x"])
    assert diagnostics.num_steps.shape == (1000, channels)
    assert jnp.all(diagnostics.num_steps == 12)
    assert not jnp.any(diagnostics.diverging)
    assert jnp.all(diagnostics.step_size > 0)
    assert diagnostics_path.exists()


def test_independent_hmc_rejects_invalid_step_count():
    t = jnp.arange(3.0)
    y = jnp.zeros((1, 3))
    yerr = jnp.ones_like(y)
    try:
        get_samples_independent_hmc(
            _normal_channels,
            jax.random.PRNGKey(0),
            t,
            yerr,
            y,
            {"x": jnp.zeros(1)},
            prior_loc=jnp.zeros(1),
            num_warmup=1,
            num_samples=1,
            num_steps=0,
            channel_varying_kwargs=("prior_loc",),
        )
    except ValueError as error:
        assert "num_steps" in str(error)
    else:
        raise AssertionError("num_steps=0 should fail")


def test_laplace_hmc_uses_fd_metric_and_jittered_fixed_work():
    channels, times = 2, 4
    t = jnp.arange(float(times))
    yerr = jnp.full((channels, times), 0.25)
    y = jnp.asarray([[0.8] * times, [-1.1] * times])
    prior_loc = jnp.asarray([0.0, -0.5])

    samples, diagnostics = get_samples_independent_hmc(
        _normal_channels,
        jax.random.PRNGKey(32),
        t,
        yerr,
        y,
        {"x": prior_loc},
        prior_loc=prior_loc,
        channel_varying_kwargs=("prior_loc",),
        num_samples=20,
        nuts_kwargs={
            "mass_matrix": "laplace",
            "laplace_hessian_method": "finite_difference",
            "laplace_compare_exact_hessian": True,
            "laplace_map_iterations": 3,
            "laplace_warmup": 6,
            "num_steps": 8,
            "trajectory_jitter": 0.25,
        },
        return_diagnostics=True,
    )

    assert samples["x"].shape == (20, channels)
    assert diagnostics.map_gradient_norm.shape == (channels,)
    assert jnp.all(diagnostics.map_hessian_relative_error < 1.0e-8)
    assert int(jnp.min(diagnostics.num_steps)) >= 6
    assert int(jnp.max(diagnostics.num_steps)) <= 10
    assert jnp.unique(diagnostics.num_steps).size > 1
    assert not jnp.any(diagnostics.diverging)


def test_pipeline_resolves_laplace_hmc_options():
    resolved = fit_jwst._resolve_jaxoplanet_spectro_nuts_kwargs(
        {
            "spectro_mass_matrix": "laplace",
            "spectro_laplace_hessian_method": "finite_difference",
            "spectro_laplace_warmup": 100,
            "spectro_laplace_target_accept": 0.88,
            "spectro_laplace_fuse_program": True,
            "spectro_hmc_num_steps": 16,
            "spectro_hmc_trajectory_jitter": 0.25,
        },
        "highres",
        {"dense_mass": False, "target_accept_prob": 0.8},
        independent=True,
        hmc=True,
    )

    assert resolved["mass_matrix"] == "laplace"
    assert resolved["laplace_hessian_method"] == "finite_difference"
    assert resolved["laplace_warmup"] == 100
    assert resolved["laplace_target_accept"] == 0.88
    assert resolved["laplace_fuse_program"] is True
    assert resolved["num_steps"] == 16
    assert resolved["trajectory_jitter"] == 0.25
    assert "max_tree_depth" not in resolved

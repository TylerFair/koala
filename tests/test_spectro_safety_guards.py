import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest
import numpyro
import numpyro.distributions as dist

import fit_jwst


@pytest.mark.parametrize("prior", ["gaussian", "uniform"])
def test_wide_power2_ld_uses_maxted_coordinates(prior):
    assert fit_jwst._resolve_ld_parameterization(prior, "power2") == "decorrelated"


@pytest.mark.parametrize("prior", ["gaussian", "uniform"])
def test_wide_quadratic_ld_keeps_coefficients(prior):
    assert fit_jwst._resolve_ld_parameterization(prior, "quadratic") == "coefficients"


@pytest.mark.parametrize("prior", ["fixed", "stellarprior", "sing"])
def test_other_ld_priors_keep_coefficients(prior):
    assert fit_jwst._resolve_ld_parameterization(prior) == "coefficients"


@pytest.mark.parametrize(
    ("detrend", "expected"),
    [
        ("linear", "laplace"),
        ("quadratic", "laplace"),
        ("explinear", "laplace"),
        ("spot", "laplace"),
        ("2spot", "adaptive"),
        ("linear_discontinuity", "laplace"),
        ("spot+linear_discontinuity", "adaptive"),
    ],
)
def test_whitelight_metric_uses_production_family_default(detrend, expected):
    assert fit_jwst._resolve_whitelight_mass_matrix(detrend) == expected


@pytest.mark.parametrize(
    ("detrend", "expected"),
    [
        ("linear", "physical"),
        ("quadratic", "physical"),
        ("explinear", "physical"),
        ("spot", "cadence"),
        ("2spot", "physical"),
        ("linear_discontinuity", "cadence"),
    ],
)
def test_whitelight_trend_parameterization_is_inferred(detrend, expected):
    assert fit_jwst._resolve_whitelight_trend_parameterization(detrend) == expected


def test_persistent_compile_cache_is_off_unless_path_is_set():
    assert fit_jwst._resolve_compile_cache_options({}) == {"cache_dir": None}
    assert fit_jwst._resolve_compile_cache_options({
        "jax_compilation_cache_dir": "/tmp/compile-cache",
    }) == {"cache_dir": "/tmp/compile-cache"}
    assert fit_jwst._resolve_ld_prior_cache_options({}) == {
        "enabled": True,
        "cache_dir": "/scratch/midway3/tfairnington/ld_prior_cache",
    }


def test_sampler_swap_order_is_directional_and_ends_adaptive():
    assert fit_jwst._spectro_sampler_swap_order("independent_nuts") == (
        "independent_hmc", "joint_nuts"
    )
    assert fit_jwst._spectro_sampler_swap_order("independent_hmc") == (
        "independent_nuts", "joint_nuts"
    )


def _tiny_radius_model(t, yerr, y=None):
    rors = numpyro.sample("rors", dist.Uniform(0.02, 0.3))
    numpyro.sample("obs", dist.Normal(rors[:, None], yerr), obs=y)


def test_tiny_exact_model_selectively_swaps_nuts_to_hmc(tmp_path, monkeypatch):
    gate_calls = []

    def forced_gate(samples, diagnostics_path, min_depth_ess, max_divergences):
        gate_calls.append(diagnostics_path)
        failed = jnp.asarray([0], dtype=int) if len(gate_calls) == 1 else jnp.asarray([], dtype=int)
        return failed, {
            "depth_ess_per_channel": [999.0],
            "num_divergences_per_channel": [0],
        }

    monkeypatch.setattr(fit_jwst, "_spectro_failed_lanes", forced_gate)
    t = jnp.arange(4.0)
    samples = fit_jwst.get_samples_chunked(
        _tiny_radius_model, jax.random.PRNGKey(92), t,
        jnp.full((1, 4), 0.08), jnp.full((1, 4), 0.12),
        {"rors": jnp.asarray([0.1])}, chunk_size=1,
        nuts_kwargs={"mass_matrix": "laplace", "laplace_warmup": 10,
                     "laplace_map_iterations": 4},
        mcmc_kwargs={"num_warmup": 10, "num_samples": 20},
        output_dir=str(tmp_path), checkpoint_prefix="tiny",
        sampler_backend="independent_nuts", spectro_min_depth_ess=1,
    )
    assert samples["rors"].shape == (20, 1)
    assert samples.sampler_used == ["independent_hmc"]
    checkpoint = next((tmp_path / "chunks").glob("*.pkl"))
    loaded = fit_jwst._load_chunk_samples(checkpoint)
    assert loaded.sampler_used == ["independent_hmc"]

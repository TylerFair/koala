import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from fit_jwst import (
    _geometry_from_white_light_samples,
    _load_whitelight_geometry_handoff,
    _select_max_likelihood_retained_draw,
    _sum_data_log_likelihood_per_draw,
    _write_whitelight_geometry_handoff,
    _resolve_whitelight_laplace_options,
    _geometry_chain_quality,
)
from models.independent_nuts import prepare_laplace_metric


def _normal_location_model(t, yerr, y=None):
    location = numpyro.sample("location", dist.Normal(0.0, 0.1))
    numpyro.deterministic("location_squared", location**2)
    numpyro.sample("obs", dist.Normal(location + jnp.zeros_like(t), yerr), obs=y)


def test_whitelight_laplace_flag_defaults_and_validation():
    defaults = _resolve_whitelight_laplace_options({})
    assert defaults == {
        "mass_matrix": "adaptive",
        "warmup": 200,
        "target_accept": 0.9,
        "max_tree_depth": 10,
        "trust_radius": 5.0,
        "hessian_method": "finite_difference",
    }
    selected = _resolve_whitelight_laplace_options(
        {"whitelight_mass_matrix": "laplace", "whitelight_laplace_warmup": 7}
    )
    assert selected["mass_matrix"] == "laplace"
    assert selected["warmup"] == 7


def test_single_lane_laplace_nuts_preserves_sample_dict_layout():
    import jax

    time = jnp.arange(8.0)
    error = jnp.full(8, 0.2)
    flux = jnp.full(8, 0.03)
    key = jax.random.PRNGKey(9)
    preparation = prepare_laplace_metric(
        _normal_location_model,
        key,
        {"location": 0.0},
        time,
        error,
        model_kwargs={"y": flux},
        max_iterations=4,
    )
    mcmc = MCMC(
        NUTS(
            _normal_location_model,
            dense_mass=True,
            inverse_mass_matrix=preparation.inverse_mass_matrix,
            adapt_mass_matrix=False,
            max_tree_depth=3,
        ),
        num_warmup=5,
        num_samples=6,
        progress_bar=False,
    )
    mcmc.run(
        key,
        time,
        error,
        y=flux,
        init_params=preparation.unconstrained_map,
    )
    samples = mcmc.get_samples()
    assert set(samples) == {"location", "location_squared"}
    assert samples["location"].shape == (6,)
    assert samples["location_squared"].shape == (6,)


def test_whitelight_geometry_quality_uses_required_sites():
    draws = np.linspace(-1.0, 1.0, 512)[None, :]
    grouped = {
        "t0_0": draws,
        "b_0": draws + 2.0,
        "logD_0": draws * 0.01,
        "rors_0": draws * 0.001 + 0.1,
        "unrelated": np.ones_like(draws),
    }
    ess, rhat = _geometry_chain_quality(grouped)
    assert set(ess) == {"t0", "b", "duration", "rors"}
    assert all(np.isfinite(value) and value > 0 for value in ess.values())
    assert rhat == {}


def test_summed_data_log_likelihood_uses_observation_distribution_only():
    samples = {
        "location": jnp.array([-1.0, 0.0, 1.0]),
        "location_squared": jnp.array([1.0, 0.0, 1.0]),
    }
    time = jnp.arange(2.0)
    yerr = jnp.array([0.5, 0.25])
    flux = jnp.array([0.2, -0.1])

    actual = _sum_data_log_likelihood_per_draw(
        _normal_location_model,
        samples,
        time,
        yerr,
        y=flux,
        batch_size=2,
    )
    expected = np.asarray(
        dist.Normal(samples["location"][:, None], yerr[None, :])
        .log_prob(flux[None, :])
        .sum(axis=1)
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    # The prior strongly prefers zero, but is intentionally absent from this
    # observed-data score.
    prior_log_prob = np.asarray(dist.Normal(0.0, 0.1).log_prob(samples["location"]))
    assert not np.allclose(actual, expected + prior_log_prob)


def test_max_likelihood_draw_is_finite_coherent_and_maps_chain_index():
    posterior = {
        "t0_0": jnp.array([10.0, 20.0, 30.0, 40.0]),
        "rors_0": jnp.array([0.10, 0.20, jnp.nan, 0.40]),
        "duration_0": jnp.array([0.01, 0.02, 0.03, 0.04]),
        "b_0": jnp.array([0.1, 0.2, 0.3, 0.4]),
        "a_rs_0": jnp.array([10.0, 20.0, 30.0, 40.0]),
        "cos_i_0": jnp.array([0.01, 0.02, 0.03, 0.04]),
        "inc_0": jnp.array([1.56, 1.55, 1.54, 1.53]),
    }
    # Draw 2 has the largest nominal score but contains a non-finite sample.
    selected = _select_max_likelihood_retained_draw(
        posterior,
        np.array([1.0, 5.0, 9.0, 4.0]),
        num_chains=2,
    )

    assert selected["flat_draw_index"] == 1
    assert selected["chain_index"] == 0
    assert selected["draw_index"] == 1
    assert selected["summed_data_log_likelihood"] == 5.0

    def derive_geometry(samples, period, ecc=0.0, omega=0.0):
        del period, ecc, omega
        return {
            name: samples[name]
            for name in (
                "duration_0", "b_0", "a_rs_0", "cos_i_0", "inc_0"
            )
        }

    geometry = _geometry_from_white_light_samples(
        selected["samples"], derive_geometry, period=[3.0]
    )
    assert geometry == {
        "period": [3.0],
        "duration": [0.02],
        "t0": [20.0],
        "b": [0.2],
        "a_rs": [20.0],
        "cos_i": [0.02],
        "inclination": [1.55],
        "rors": [0.2],
    }


def test_geometry_handoff_round_trip_and_tamper_invalidation(tmp_path):
    path = tmp_path / "whitelight_geometry_handoff.json"
    posterior_fingerprint = "posterior-target-123"
    payload = {
        "estimator": "max_likelihood_draw",
        "posterior_fingerprint_sha256": posterior_fingerprint,
        "transit_engine": "jaxoplanet",
        "param_method": "duration",
        "num_retained_draws": 4,
        "num_chains": 2,
        "selected_flat_draw_index": 1,
        "selected_chain_index": 0,
        "selected_draw_index": 1,
        "summed_data_log_likelihood": 42.0,
        "selected_primitive_parameters": {
            "t0_0": 20.0,
            "rors_0": 0.2,
            "logD_0": float(np.log(0.02)),
        },
        "geometry": {
            "period": [3.0],
            "duration": [0.02],
            "t0": [20.0],
            "b": [0.2],
            "a_rs": [20.0],
            "cos_i": [0.02],
            "inclination": [1.55],
            "rors": [0.2],
        },
    }

    written = _write_whitelight_geometry_handoff(path, payload)
    loaded = _load_whitelight_geometry_handoff(
        path,
        expected_posterior_fingerprint=posterior_fingerprint,
        expected_estimator="max_likelihood_draw",
    )
    assert loaded == written
    assert loaded["artifact_fingerprint_sha256"]
    assert not list(tmp_path.glob("*.tmp.*"))

    assert _load_whitelight_geometry_handoff(
        path,
        expected_posterior_fingerprint="different-target",
        expected_estimator="max_likelihood_draw",
    ) is None

    tampered = json.loads(path.read_text())
    tampered["geometry"]["t0"] = [999.0]
    path.write_text(json.dumps(tampered))
    assert _load_whitelight_geometry_handoff(
        path,
        expected_posterior_fingerprint=posterior_fingerprint,
        expected_estimator="max_likelihood_draw",
    ) is None

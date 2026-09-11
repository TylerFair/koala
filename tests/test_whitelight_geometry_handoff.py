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
    _load_whitelight_geometry_handoff,
    _write_whitelight_geometry_handoff,
    _resolve_whitelight_laplace_options,
    _geometry_chain_quality,
)
from models.independent_nuts import prepare_laplace_metric


def _normal_location_model(t, yerr, y=None):
    location = numpyro.sample("location", dist.Normal(0.0, 0.1))
    numpyro.deterministic("location_squared", location**2)
    numpyro.sample("obs", dist.Normal(location + jnp.zeros_like(t), yerr), obs=y)


def test_whitelight_laplace_production_constants():
    defaults = _resolve_whitelight_laplace_options()
    assert defaults == {
        "mass_matrix": "laplace",
        "warmup": 200,
        "target_accept": 0.9,
        "max_tree_depth": 10,
        "trust_radius": 5.0,
        "hessian_method": "finite_difference",
    }
    assert _resolve_whitelight_laplace_options(True)["target_accept"] == 0.99
    assert _resolve_whitelight_laplace_options(False, "adaptive")["mass_matrix"] == "adaptive"


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




def test_geometry_quality_ignores_nonbinary_fixed_period():
    rng = np.random.default_rng(7)
    grouped = {
        "period": np.full((1, 1000, 1), 2.6162644),
        "t0_0": rng.normal(size=(1, 1000)),
        "_eclipse_depth_0": rng.normal(1e-4, 1e-5, size=(1, 1000)),
    }
    ess, _ = _geometry_chain_quality(grouped)
    assert set(ess) == {"t0", "eclipse_depth_0"}
    # Resolve even tiny real variation; no tolerance should erase it.
    grouped["period"][0, 0, 0] = np.nextafter(2.6162644, np.inf)
    ess, _ = _geometry_chain_quality(grouped)
    assert "period" in ess


def test_geometry_handoff_round_trip_and_tamper_invalidation(tmp_path):
    path = tmp_path / "whitelight_geometry_handoff.json"
    posterior_fingerprint = "posterior-target-123"
    payload = {
        "estimator": "posterior_median",
        "posterior_fingerprint_sha256": posterior_fingerprint,
        "transit_engine": "jaxoplanet",
        "param_method": "duration",
        "num_retained_draws": 4,
        "selected_flat_draw_index": None,
        "selected_chain_index": None,
        "selected_draw_index": None,
        "summed_data_log_likelihood": None,
        "selected_primitive_parameters": {},
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
        expected_estimator="posterior_median",
    )
    assert loaded == written
    assert loaded["artifact_fingerprint_sha256"]
    assert not list(tmp_path.glob("*.tmp.*"))

    assert _load_whitelight_geometry_handoff(
        path,
        expected_posterior_fingerprint="different-target",
        expected_estimator="posterior_median",
    ) is None

    tampered = json.loads(path.read_text())
    tampered["geometry"]["t0"] = [999.0]
    path.write_text(json.dumps(tampered))
    assert _load_whitelight_geometry_handoff(
        path,
        expected_posterior_fingerprint=posterior_fingerprint,
        expected_estimator="posterior_median",
    ) is None

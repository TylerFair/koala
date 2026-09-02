import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

import fit_jwst
from tools.run_sampler_on_stage_inputs import _base_key_for_seed
from tools.spectro_stage_inputs import load_stage_inputs


def _make_factorized_model(prior_scale=1.0):
    def model(t, yerr, y=None, prior_loc=None):
        x = numpyro.sample("x", dist.Normal(prior_loc, prior_scale))
        prediction = x[:, None] + jnp.zeros_like(t)
        numpyro.deterministic("depths", x[:, None] ** 2)
        numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)

    return model


def _tiny_inputs():
    t = jnp.arange(4, dtype=jnp.float64)
    yerr = jnp.full((3, 4), 0.2, dtype=jnp.float64)
    y = jnp.asarray([[0.1] * 4, [0.2] * 4, [0.3] * 4])
    prior_loc = jnp.asarray([0.0, 0.1, 0.2], dtype=jnp.float64)
    return t, yerr, y, prior_loc


def test_dump_hook_loader_and_initial_potential(monkeypatch, tmp_path):
    monkeypatch.setenv("FIT_JWST_DUMP_SAMPLER_INPUTS", str(tmp_path))
    monkeypatch.delenv("FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT", raising=False)
    t, yerr, y, prior_loc = _tiny_inputs()
    model = fit_jwst._build_spectroscopic_model(
        _make_factorized_model,
        prior_scale=1.25,
    )

    monkeypatch.setattr(
        fit_jwst,
        "get_samples",
        lambda *args, **kwargs: {"x": prior_loc[None, :]},
    )
    result = fit_jwst._run_sampling_stage(
        model,
        jax.random.PRNGKey(17),
        t,
        yerr,
        y,
        {"x": prior_loc},
        nuts_kwargs={"dense_mass": False},
        mcmc_kwargs={"num_warmup": 2, "num_samples": 2},
        sampler_backend="joint_nuts",
        channel_varying_kwargs=("prior_loc",),
        checkpoint_signature={"stage": "low_resolution"},
        dump_metadata={
            "stage_kind": "low_resolution",
            "stage_label": "tiny low-res",
            "config_path": "/tmp/tiny.yaml",
            "wavelength": np.asarray([1.0, 1.1, 1.2]),
            "active_window_cadences": 3,
        },
        prior_loc=prior_loc,
    )

    np.testing.assert_array_equal(np.asarray(result["x"]), prior_loc[None, :])
    dump_path = tmp_path / "tiny_low-res_inputs.pkl"
    assert dump_path.is_file()
    with open(dump_path, "rb") as stream:
        raw = pickle.load(stream)
    assert raw["model_builder"]["kwargs"] == {"prior_scale": 1.25}
    assert raw["meta"]["num_channels"] == 3
    assert raw["meta"]["num_cadences"] == 4
    assert raw["meta"]["active_window_cadences"] == 3

    loaded = load_stage_inputs(dump_path)
    measured = loaded.recompute_initial_potential()
    assert abs(measured - raw["initial_potential"]) <= 1.0e-8
    selected = loaded.select(1, 3)
    assert selected.y.shape == (2, 4)
    np.testing.assert_array_equal(
        selected.model_kwargs["prior_loc"], np.asarray([0.1, 0.2])
    )
    np.testing.assert_array_equal(
        selected.meta["wavelength"], np.asarray([1.1, 1.2])
    )


def test_dump_hook_is_a_noop_when_environment_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("FIT_JWST_DUMP_SAMPLER_INPUTS", raising=False)
    monkeypatch.setattr(
        fit_jwst,
        "_sampler_initial_potential",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dump work ran while disabled")
        ),
    )
    t, yerr, y, prior_loc = _tiny_inputs()
    monkeypatch.setattr(
        fit_jwst,
        "get_samples",
        lambda *args, **kwargs: {"x": prior_loc[None, :]},
    )
    result = fit_jwst._run_sampling_stage(
        _make_factorized_model(),
        jax.random.PRNGKey(18),
        t,
        yerr,
        y,
        {"x": prior_loc},
        mcmc_kwargs={"num_warmup": 1, "num_samples": 1},
        prior_loc=prior_loc,
    )
    assert result["x"].shape == (1, 3)
    assert not list(tmp_path.iterdir())


def test_reference_seed_uses_pipeline_stage_split():
    class Stage:
        meta = {"stage_kind": "low_resolution"}
        rng_key = jax.random.split(jax.random.PRNGKey(555), 7)[3]

    baseline, baseline_source = _base_key_for_seed(Stage(), 0)
    replicate, replicate_source = _base_key_for_seed(Stage(), 1)
    np.testing.assert_array_equal(baseline, Stage.rng_key)
    np.testing.assert_array_equal(
        replicate,
        jax.random.split(jax.random.PRNGKey(556), 7)[3],
    )
    assert baseline_source == "dumped_pipeline_key"
    assert replicate_source == "pipeline_master_seed_split"

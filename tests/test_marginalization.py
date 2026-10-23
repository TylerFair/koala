"""Grid isolation, exact likelihood replay, statistical and cache gates."""
import itertools
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import pytest
import yaml

from koala.marginalization.grid import prepare_grid, run_grid
from koala.marginalization.likelihood import (Stage, CompiledLikelihood, score_loglik,
    observation_identity, require_same_observations, load_stage)
from koala.marginalization.scoring import score_grid, score_variant, score_fingerprint
from koala.marginalization.weights import (stacking_weights, hierarchical_map_weights,
    model_weights, mix_depths)
from models.stacking import pointwise_loglik, psis_loo

jax.config.update("jax_enable_x64", True)


def normal_model(t, yerr, y=None, likelihood_mask=None):
    location = numpyro.sample("location", dist.Normal(jnp.zeros(yerr.shape[0]), 1))
    with handlers.mask(mask=True if likelihood_mask is None else likelihood_mask):
        numpyro.sample("obs", dist.Normal(location[:, None]+jnp.zeros_like(t), yerr), obs=y)


def normal_stage():
    t = np.arange(7.)
    return Stage(normal_model, t, np.ones((3, 7)), np.zeros((3, 7)),
                 {"likelihood_mask": np.array([True, True, False, True, True, False, True])},
                 dict(wavelength=np.arange(3.)+1))


def test_eight_grid_isolation_and_resume_environment(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    original = dict(path=".", output_dir="fiducial", planet=dict(name="demo"),
                    flags=dict(spectro_min_depth_ess=777, num_chains=4),
                    outlier_clip=dict(mask_integrations_start=3, mask_integrations_end=2))
    base.write_text(yaml.safe_dump(original))
    manifest = prepare_grid(base, tmp_path / "grid")
    rows = yaml.safe_load(manifest.read_text())["variants"]
    assert len(rows) == 8
    assert yaml.safe_load(base.read_text()) == original
    calls = []
    monkeypatch.setenv("FIT_JWST_DUMP_SAMPLER_INPUTS", "original")
    def run(cfg, config_path):
        calls.append((cfg, os.environ["FIT_JWST_DUMP_SAMPLER_INPUTS"]))
        assert cfg["flags"]["spectro_min_depth_ess"] == 777
        assert cfg["flags"]["num_chains"] == 4
        assert cfg["flags"]["spectro_cadence_reduction"] == "off"
        assert np.isinf(cfg["outlier_clip"]["whitelight_sigma"])
        assert cfg['outlier_clip']['mask_integrations_start'] == 3
        assert cfg['outlier_clip']['mask_integrations_end'] == 2
        assert cfg["output_dir"] != "fiducial"
    run_grid(manifest, runner=run)
    assert len(calls) == 8
    assert os.environ["FIT_JWST_DUMP_SAMPLER_INPUTS"] == "original"
    assert len({cfg["flags"]["jax_compilation_cache_dir"] for cfg, _ in calls}) == 1
    with pytest.raises(FileExistsError):
        prepare_grid(base, tmp_path / "grid")
    def failed(*args, **kwargs):
        raise RuntimeError("failed fit")
    with pytest.raises(RuntimeError):
        run_grid(manifest, runner=failed)
    assert os.environ["FIT_JWST_DUMP_SAMPLER_INPUTS"] == "original"


def test_jit_batches_and_streamed_psis_match_existing_evaluation():
    stage = normal_stage()
    samples = {"location": np.random.default_rng(23).normal(0, .15, (103, 3))}
    expected = pointwise_loglik(samples, stage)[..., stage.model_kwargs["likelihood_mask"]]
    evaluator = CompiledLikelihood(stage, batch_size=16)
    actual = np.concatenate([values for _, _, values in evaluator.batches(samples)])
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert evaluator.evaluate._cache_size() == 1  # final short batch was padded
    with pytest.raises(ValueError, match="missing sampled"):
        list(evaluator.batches({"unrelated": np.zeros((16, 3))}))
    scores, khat = score_loglik(actual, point_batch_size=4)
    baseline, _, expected_khat = psis_loo(expected)
    np.testing.assert_allclose(scores, baseline, atol=2e-14, rtol=0)
    np.testing.assert_allclose(khat, expected_khat, atol=2e-12, rtol=0)
    streamed, streamed_khat = score_variant(stage, samples, draw_batch_size=16, point_batch_size=4)
    np.testing.assert_allclose(streamed, baseline, atol=2e-14, rtol=0)
    np.testing.assert_allclose(streamed_khat, expected_khat, atol=2e-12, rtol=0)


@pytest.mark.parametrize("changed", ["time", "flux", "uncertainty", "wavelength"])
def test_observation_gate_checks_exact_values(changed):
    stage = normal_stage()
    args = dict(model=stage.model, t=stage.t.copy(), y=stage.y.copy(), yerr=stage.yerr.copy(),
                model_kwargs=stage.model_kwargs, meta=dict(stage.meta))
    if changed == "wavelength":
        args["meta"]["wavelength"] = args["meta"]["wavelength"] + 1e-12
    else:
        key = dict(time="t", flux="y", uncertainty="yerr")[changed]
        args[key].flat[0] += 1e-12
    with pytest.raises(ValueError, match=changed):
        require_same_observations(observation_identity(stage), Stage(**args))


def test_stacking_and_pooling_have_correct_predictive_behavior():
    probabilities = np.array([.9, .1])
    outcomes = np.arange(1000) < 700
    scores = np.log(np.where(outcomes, probabilities[:, None], 1-probabilities[:, None]))
    np.testing.assert_allclose(stacking_weights(scores), [.75, .25], atol=1e-6)
    # Two channels with opposing preference: pooling shrinks their differences.
    elpd = np.stack((scores, scores[:, ::-1]), axis=1)
    elpd[:, 1] = elpd[::-1, 0]
    independent = hierarchical_map_weights(elpd, strength=0)
    pooled = hierarchical_map_weights(elpd, strength=1e5)
    np.testing.assert_allclose(independent, [[.75, .25], [.25, .75]], atol=1e-6)
    assert np.max(np.abs(pooled[:, 0]-pooled[:, 1])) < .01
    np.testing.assert_allclose(pooled.sum(axis=0), 1, atol=1e-14)
    for method, pooling in itertools.product(("stacking", "pseudo_bma_plus", "equal"), ("global", "channel")):
        weights, _, _ = model_weights(elpd, method=method, pooling=pooling, n_boot=100, seed=8)
        np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-14)
    with pytest.raises(ValueError):
        stacking_weights(np.full((2, 3), np.nan))


def test_joint_mixture_keeps_model_identity_and_same_draw_across_channels():
    x = np.linspace(.01, .02, 2000)
    first = np.column_stack((x, 2*x))
    second = first+.02
    mixture, _ = mix_depths([first, second], [.6, .4], n_out=20000, seed=71)
    # Channel relation identifies model identity and holds exactly within a row.
    delta = mixture[:, 1]-2*mixture[:, 0]
    assert np.all(np.isclose(delta, 0, atol=1e-16) | np.isclose(delta, -.02, atol=1e-16))
    np.testing.assert_allclose(np.mean(delta < -.01), .4, atol=.012)
    marginal, _ = mix_depths([first, second], np.full((2, 2), .5), n_out=2000, seed=71)
    assert not np.all(np.isclose(marginal[:, 1]-2*marginal[:, 0], 0)
                      | np.isclose(marginal[:, 1]-2*marginal[:, 0], -.02))


def test_score_cache_and_products_roundtrip(tmp_path, monkeypatch):
    from koala.marginalization import scoring
    # Conjugate Normal posterior, with rors retained as a science product.
    stage_dir, fit_dir = tmp_path / "stage", tmp_path / "fit"
    stage_dir.mkdir()
    (fit_dir / "state" / "chunks").mkdir(parents=True)
    stage = normal_stage()
    stage_file = stage_dir / "demo_high_resolution_inputs.pkl"
    payload = dict(schema_version=1, model_builder=dict(identity=__name__+".normal_model", callable_is_model=True),
                   t=stage.t, yerr=stage.yerr, indiv_y=stage.y, model_kwargs=stage.model_kwargs,
                   meta=dict(stage.meta, stage_kind="high_resolution", stage_label="demo_high_resolution"))
    stage_file.write_bytes(pickle.dumps(payload))
    rng = np.random.default_rng(61)
    samples = dict(location=rng.normal(0, 1/np.sqrt(6), (2000, 3)),
                   rors=rng.normal(.1, .001, (2000, 3, 1)))
    chunk = fit_dir / "state" / "chunks" / "demo_chunk_0_3.pkl"
    chunk.write_bytes(pickle.dumps(samples))
    manifest = tmp_path / "analysis.yaml"
    manifest.write_text(yaml.safe_dump(dict(variants=[dict(name="normal", fit_dir="fit", stage_inputs="stage")])))
    output = tmp_path / "products"
    path = score_grid(manifest, output, n_out=1000)
    assert path.is_file()
    with np.load(output / "state" / "posterior.npz") as result:
        assert result["mixture"].shape == (1000, 3)
        np.testing.assert_array_equal(result["weights"], np.ones((1, 3)))
    fingerprint = score_fingerprint(stage_file, [chunk], {})
    monkeypatch.setattr(scoring, "score_variant", lambda *a, **k: pytest.fail("Cache was not reused"))
    score_grid(manifest, output, n_out=1000)
    samples["rors"][0, 0, 0] += .001
    chunk.write_bytes(pickle.dumps(samples))
    assert score_fingerprint(stage_file, [chunk], {}) != fingerprint
    # A poor importance tail prevents overwriting a previously delivered spectrum.
    monkeypatch.setattr(scoring, "score_variant", lambda *a, **k: (np.zeros((3, 5)), np.ones((3, 5))))
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="reliability gate"):
        score_grid(manifest, output)
    assert path.read_bytes() == previous


@pytest.mark.parametrize("law,prior,trend", list(itertools.product(
    ("quadratic", "power2"), ("fixed", "uniform"), ("linear", "quadratic"))))
def test_eight_production_models_replay_dynamic_channels_and_mask(law, prior, trend):
    from models.jaxoplanet.builder import create_vectorized_model
    model = create_vectorized_model(ld_profile=law, ld_mode=prior, detrend_type=trend,
                                     cadence_reduction="off", transit_window="off")
    t = np.linspace(-.08, .08, 35)
    # Geometry, LD and errors vary across wavelength, exposing incorrect cache captures.
    error = np.stack([np.full(35, v) for v in (.001, .002, .003)])
    y = np.ones_like(error)
    coefficients = np.array([[.3, .4], [.8, .1], [.2, .9]])
    kwargs = dict(rors_reference=np.array([.1]), mu_duration=np.array([.12]), mu_t0=np.array([0.]), mu_b=np.array([1.03]),
                  PERIOD=np.array([3.]), ld_fixed=coefficients, mu_u_ld=coefficients,
                  sigma_u_ld=np.full((3, 2), .1), likelihood_mask=np.arange(35)%4 != 0)
    trace = handlers.trace(handlers.seed(model, 4)).get_trace(t, error, y=y, **kwargs)
    draw = {key: site["value"] for key, site in trace.items()
            if site.get("type") == "sample" and not site.get("is_observed")}
    samples = {key: np.repeat(np.asarray(value)[None], 7, axis=0) for key, value in draw.items()}
    stage = Stage(model, t, error, y, kwargs, dict(wavelength=np.arange(3)))
    actual = np.concatenate([v for _, _, v in CompiledLikelihood(stage, 4).batches(samples)])
    expected = pointwise_loglik(samples, stage)[..., kwargs["likelihood_mask"]]
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-13)
    # Streamed single-channel replay must preserve the full vectorized model.
    from koala.marginalization.likelihood import channel_stage, channel_samples
    evaluator = CompiledLikelihood(stage, 4)
    for channel in range(3):
        values = np.concatenate([v for _, _, v in evaluator.batches(
            channel_samples(samples, stage, channel), channel_stage(stage, channel))])
        np.testing.assert_allclose(values[:, 0], expected[:, channel], atol=2e-8, rtol=2e-13)
    assert evaluator.evaluate._cache_size() == 1

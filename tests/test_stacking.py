import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from models.stacking import (
    aligned_stack_posteriors,
    laplace_log_evidence,
    pseudo_bma_plus_weights,
    psis_loo,
    stacking_weights,
)
from tools.stacking.stack_spectra import render_stacking_figures_from_arrays


def test_stacking_weights_recover_known_predictive_mixture():
    rng = np.random.default_rng(4)
    true_weights = np.array([0.7, 0.3])
    component = rng.choice(2, size=20000, p=true_weights)
    # Two calibrated binary predictive models. Stacking their probabilities
    # recovers the population mixture in the large synthetic point sample.
    outcome = rng.random(component.size) < np.where(component == 0, 0.9, 0.1)
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    elpd = np.stack([
        np.log(np.where(outcome, probabilities[m, 0], probabilities[m, 1]))
        for m in range(2)
    ])
    actual = stacking_weights(elpd)
    np.testing.assert_allclose(actual, true_weights, atol=0.025)


def test_psis_loo_gaussian_matches_exact_conjugate_loo():
    rng = np.random.default_rng(9)
    y = np.array([-0.8, -0.1, 0.15, 0.7, 1.2])
    prior_var = 2.0
    post_var = 1.0 / (1.0 / prior_var + y.size)
    post_mean = post_var * y.sum()
    theta = rng.normal(post_mean, np.sqrt(post_var), size=200000)
    loglik = (-0.5 * ((y[None, :] - theta[:, None]) ** 2 + np.log(2 * np.pi)))[:, None, :]
    elpd_i, elpd, khat = psis_loo(loglik)
    exact = []
    for index in range(y.size):
        train = np.delete(y, index)
        variance = 1.0 / (1.0 / prior_var + train.size)
        mean = variance * train.sum()
        predictive_var = 1.0 + variance
        exact.append(-0.5 * ((y[index] - mean) ** 2 / predictive_var
                            + np.log(2 * np.pi * predictive_var)))
    np.testing.assert_allclose(elpd_i[0], exact, atol=0.012)
    np.testing.assert_allclose(elpd[0], np.sum(exact), atol=0.025)
    assert np.max(khat) < 0.7


def test_pseudo_bma_plus_sums_to_one():
    weights = pseudo_bma_plus_weights(np.array([[-1.0, -2.0, -1.5],
                                                [-1.2, -1.8, -1.7]]),
                                      n_boot=1000, rng=12)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14)


def test_laplace_evidence_matches_analytic_gaussian_integral():
    precision = np.array([[2.0, 0.3], [0.3, 1.5]])
    map_point = np.array([0.2, -0.4])
    log_at_map = -3.7
    actual = laplace_log_evidence(map_point, precision, log_at_map)
    expected = log_at_map + np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(precision)[1]
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_achromatic_alignment_removes_constant_model_offset_from_width():
    rng = np.random.default_rng(91)
    base = rng.normal(0.02, 5e-5, size=(50000, 6))
    result = aligned_stack_posteriors(
        [base, base + 2e-4], np.array([0.5, 0.5]), n_out=200000, rng=17
    )
    single_sigma = (np.percentile(base, 84, axis=0)
                    - np.percentile(base, 16, axis=0)) / 2
    aligned_sigma = ((result["aligned_summary"]["q84"]
                      - result["aligned_summary"]["q16"]) / 2)
    absolute_sigma = ((result["absolute_summary"]["q84"]
                       - result["absolute_summary"]["q16"]) / 2)
    np.testing.assert_allclose(aligned_sigma, single_sigma, rtol=0.025)
    np.testing.assert_allclose(result["aligned_summary"]["disagreement"], 1.0,
                               rtol=0.025)
    assert np.all(absolute_sigma > 1.5 * single_sigma)


def test_saved_arrays_render_publication_and_diagnostics_figures(tmp_path):
    rng = np.random.default_rng(20260904)
    wavelength = np.linspace(2.9, 5.1, 18)
    model_names = np.array(["fixed_power2_linear", "uniform_linear", "stellarprior_linear_step"])
    baseline = 0.019 + 1.8e-4 * np.sin(2.4 * wavelength)
    model_depth_median = np.vstack((
        baseline - 3.0e-5,
        baseline + 5.0e-5 * np.cos(4.0 * wavelength),
        baseline + 4.0e-5,
    ))
    weights = rng.dirichlet(np.ones(model_names.size), size=wavelength.size).T
    aligned_mixture = rng.normal(baseline, 8.0e-5, size=(500, wavelength.size))
    absolute_mixture = rng.normal(
        baseline + 2.0e-5, 1.1e-4, size=(500, wavelength.size)
    )
    khat = rng.uniform(0.05, 0.58, size=(model_names.size, wavelength.size, 9))
    arrays_path = tmp_path / "synthetic_stacking_arrays.npz"
    np.savez_compressed(
        arrays_path,
        wavelength=wavelength,
        stacking_weights=weights,
        khat=khat,
        aligned_mixture=aligned_mixture,
        absolute_mixture=absolute_mixture,
        model_names=model_names,
        model_depth_median=model_depth_median,
        aligned_disagreement=np.linspace(1.0, 1.35, wavelength.size),
        absolute_disagreement=np.linspace(1.1, 1.7, wavelength.size),
    )

    publication_path, diagnostics_path = render_stacking_figures_from_arrays(
        arrays_path
    )

    assert publication_path == tmp_path / "synthetic_stacking.png"
    assert diagnostics_path == tmp_path / "synthetic_stacking_diagnostics.png"
    for path in (publication_path, diagnostics_path):
        assert path.is_file()
        assert path.stat().st_size > 1_000

import itertools

import numpy as np

from koala.limb_darkening import _build_axis, _gaussian_pdf, _stellar_grid_combos


def _weighted_moments(combos, response):
    w = np.array([c[3] for c in combos])
    w = w / w.sum()
    values = np.array([response(*c[:3]) for c in combos])
    mean = np.sum(w * values)
    return mean, np.sqrt(np.sum(w * (values - mean) ** 2))


def test_default_grid_spans_five_sigma_and_skips_negligible_points():
    combos = _stellar_grid_combos(5000.0, 100.0, 4.5, 0.1, 0.0, 0.1, 9, 5.0)
    assert len(combos) == 305
    teff = np.array([c[1] for c in combos])
    assert np.isclose(teff.min(), 4500.0) and np.isclose(teff.max(), 5500.0)


def test_default_grid_reproduces_gaussian_moments():
    # A linear response in each parameter: the propagated scatter must equal the
    # quadrature sum of the input sigmas times the slopes.
    combos = _stellar_grid_combos(5000.0, 100.0, 4.5, 0.1, 0.0, 0.1, 9, 5.0)
    mean, sd = _weighted_moments(combos, lambda feh, teff, logg: 1e-3 * teff + 0.2 * logg - 0.3 * feh)
    assert np.isclose(mean, 5.0 + 0.9, rtol=0, atol=1e-10)
    # Nine nodes over +/-5 sigma reproduce the continuous Gaussian scatter to ~1e-4.
    assert np.isclose(sd, np.sqrt(0.1 ** 2 + 0.02 ** 2 + 0.03 ** 2), rtol=5e-4)


def test_skipping_changes_nonlinear_moments_negligibly():
    response = lambda feh, teff, logg: np.sin((teff - 5000.0) / 150.0) + (logg - 4.5) ** 2 * 30.0 + (feh * 10.0) ** 3
    full = []
    for mh, te, lg in itertools.product(*(_build_axis(m, s, 9, 5.0) for m, s in ((0.0, 0.1), (5000.0, 100.0), (4.5, 0.1)))):
        full.append((mh, te, lg, _gaussian_pdf(mh, 0.0, 0.1) * _gaussian_pdf(te, 5000.0, 100.0) * _gaussian_pdf(lg, 4.5, 0.1)))
    kept = _stellar_grid_combos(5000.0, 100.0, 4.5, 0.1, 0.0, 0.1, 9, 5.0)
    mean_full, sd_full = _weighted_moments(full, response)
    mean_kept, sd_kept = _weighted_moments(kept, response)
    assert abs(mean_kept - mean_full) < 1e-4 * sd_full
    assert abs(sd_kept / sd_full - 1.0) < 1e-3


def test_zero_sigma_parameter_collapses_to_one_value():
    combos = _stellar_grid_combos(5000.0, 100.0, 4.5, 0.0, 0.0, 0.0, 9, 5.0)
    assert len(combos) == 9
    assert {c[2] for c in combos} == {4.5} and {c[0] for c in combos} == {0.0}

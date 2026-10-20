"""Regression checks for the local, NumPy-only spectral binning port."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from koala.binning import bin_at_bins, bin_at_pixel, bin_at_resolution


@pytest.mark.parametrize('method', ['sum', 'average'])
@pytest.mark.parametrize('one_dimensional', [False, True])
def test_resolution_grid_matches_upstream_fixture(method, one_dimensional):
    fixture = json.loads((Path(__file__).parent / 'data' /
                          'binning_exotedrf_2_2_0.json').read_text())
    wave, flux, err = [np.asarray(fixture[key]) for key in ('wave', 'flux', 'err')]
    expected = [np.asarray(a) for a in fixture['expected'][method]]
    if one_dimensional:
        flux, err = flux[:, 0], err[:, 0]
    # Descending inputs must sort wavelengths and observations together.
    centers, widths, binned, errors = bin_at_resolution(
        wave[::-1], flux[::-1], err[::-1], fixture['res'], method=method)
    # The wavelength grid is the upstream one; fluxes and errors are not (see below).
    np.testing.assert_allclose(centers, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(widths, expected[1], rtol=1e-13, atol=1e-13)
    assert binned.shape == errors.shape == expected[2].shape[:1] + flux.shape[1:]


def _centre_assignment(wave, centers, widths):
    owner = np.full(wave.size, -1)
    for j, (c, w) in enumerate(zip(centers, widths)):
        owner[(owner < 0) & (wave >= c - w) & (wave < c + w)] = j
    return owner


@pytest.mark.parametrize('method', ['sum', 'average'])
def test_resolution_uses_each_pixel_once_by_its_centre(method):
    rng = np.random.default_rng(3)
    wave = np.sort(rng.uniform(1., 2., 300))
    flux = rng.uniform(1., 2., (300, 4))
    err = rng.uniform(.01, .02, (300, 4))
    centers, widths, binned, errors = bin_at_resolution(wave, flux, err, 40, method=method)
    owner = _centre_assignment(wave, centers, widths)
    assert np.all(owner >= 0)
    for j in range(len(centers)):
        rows = owner == j
        n = rows.sum()
        scale = 1. if method == 'sum' else 1. / n
        np.testing.assert_allclose(binned[j], scale * flux[rows].sum(axis=0), rtol=1e-12)
        np.testing.assert_allclose(errors[j], scale * np.sqrt((err[rows]**2).sum(axis=0)), rtol=1e-12)


def test_resolution_errors_match_scatter_for_coarse_pixels():
    # Pixels 0.6 of a bin wide, like a detector already summed in wavelength. Splitting edge
    # pixels with weights that did not match the errors inflated some bins several times.
    res, sigma = 100, 1e-3
    wave = np.exp(np.arange(np.log(1.), np.log(1.5), 0.6 / res))
    rng = np.random.default_rng(4)
    flux = 1. + sigma * rng.standard_normal((wave.size, 4000))
    err = np.full_like(flux, sigma)
    _, _, binned, errors = bin_at_resolution(wave, flux, err, res, method='average')
    finite = np.isfinite(binned).all(axis=1)
    assert finite.sum() > 0.9 * len(binned)
    ratio = errors[finite, 0] / binned[finite].std(axis=1)
    np.testing.assert_allclose(ratio, 1., atol=0.06)


def test_resolution_handles_missing_values_per_integration():
    wave = np.linspace(1., 2., 60)
    flux = np.ones((60, 3)); err = np.full((60, 3), .1)
    flux[10, 1] = np.nan
    centers, widths, binned, errors = bin_at_resolution(wave, flux, err, 10, method='average')
    np.testing.assert_allclose(binned, 1.)
    owner = _centre_assignment(wave, centers, widths)
    n = (owner == owner[10]).sum()
    np.testing.assert_allclose(errors[owner[10]], [.1 / np.sqrt(n), .1 / np.sqrt(n - 1), .1 / np.sqrt(n)])


@pytest.mark.parametrize('res', [0, -1, np.nan, np.inf, 0.1, 1000])
def test_resolution_rejects_invalid_grid(res):
    wave = np.linspace(1, 2, 20)
    with pytest.raises(ValueError):
        bin_at_resolution(wave, np.ones(20), np.ones(20), res)


@pytest.mark.parametrize('one_dimensional', [True, False])
def test_resolution_leaves_bins_without_a_pixel_centre_empty(one_dimensional):
    # The first coarse input pixel is wider than several R=200 output bins. Only the bin
    # holding its centre gets it; the others are NaN and the datacube drops them.
    wave = np.r_[2.0, np.linspace(2.03, 3.0, 195)]
    flux = np.ones(wave.size) if one_dimensional else np.ones((wave.size, 3))
    centers, widths, binned, errors = bin_at_resolution(
        wave, flux, flux * 0.01, 200, method='average')
    assert binned.shape == errors.shape
    assert len(centers) == len(widths) == binned.shape[0]
    rows = binned if one_dimensional else binned[:, 0]
    empty = ~np.isfinite(rows)
    assert 0 < empty.sum() < 0.1 * len(rows)
    assert np.all(centers[empty] < 2.03)
    np.testing.assert_allclose(binned[~empty], 1.0)
    assert np.all(np.isfinite(errors[~empty]) & (errors[~empty] > 0))


def test_pixel_binning_trims_and_preserves_time_axis():
    wave = np.arange(1., 12.)
    flux = np.column_stack((wave, 10 * wave))
    err = np.ones_like(flux)
    centers, half, binned, uncertainty = bin_at_pixel(wave, flux, err, 4)
    np.testing.assert_allclose(centers, [3.5, 7.5])
    np.testing.assert_allclose(half, [2, 2])
    np.testing.assert_allclose(binned, [[14, 140], [30, 300]])
    np.testing.assert_allclose(uncertainty, np.full((2, 2), 2))


def test_pixel_binning_single_bin():
    result = bin_at_pixel(np.arange(1., 6.), np.ones(5), np.ones(5), 5)
    for actual, expected in zip(result, ([3], [2.5], [5], [np.sqrt(5)])):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('npix', [0, -1, 1.5, True, 6])
def test_pixel_rejects_invalid_width(npix):
    with pytest.raises(ValueError, match='npix'):
        bin_at_pixel(np.arange(1., 6.), np.ones(5), np.ones(5), npix)


def test_preset_bins_keep_upstream_boundaries_and_layout():
    wave = np.arange(1., 5.)
    low, high, flux, err = bin_at_bins(
        wave - .5, wave + .5, np.array([[1., 2., 3., 4.]]),
        np.ones((1, 4)), np.array([1., 3., 8.]), np.array([3., 5., 9.]))
    np.testing.assert_array_equal(low, [[1, 3, 8]])
    np.testing.assert_array_equal(high, [[3, 5, 9]])
    np.testing.assert_array_equal(flux, [[3, 7, 0]])
    np.testing.assert_allclose(err, [[np.sqrt(2), np.sqrt(2), 0]])


def test_reference_wrapper_retains_quadrature_errors():
    from koala.reference_grid import bin_to_reference_grid

    wave = np.arange(1., 5.)
    result = bin_to_reference_grid(
        wave, np.column_stack((wave, 10 * wave)), np.ones((4, 2)),
        np.array([1.5, 3.5]), np.array([1., 1.]))
    np.testing.assert_allclose(result[2], [[3, 30], [7, 70]])
    np.testing.assert_allclose(result[3], np.full((2, 2), np.sqrt(2)))


def test_pipeline_runs_with_exotedrf_imports_blocked():
    script = '''
import importlib.abc
import sys
class BlockExotedrf(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'exotedrf', 'spectres'}:
            raise ImportError('Dependency must not be imported: ' + fullname)
sys.meta_path.insert(0, BlockExotedrf())
import numpy as np
from koala import datacube
from koala import reference_grid
wave = np.linspace(1, 2.5, 80)
flux = np.ones((7, 80))
for mode in ({'resolution': {'low': 10, 'high': 20}},
             {'pixels': {'low': 4, 'high': 2}},
             {'resolution': {'high': 20}},
             {'pixels': {'high': 2}}):
    result = datacube.bin_spectroscopy_data(
        wave, np.full(80, .01), flux, flux * .1,
        {'instrument': 'NIRISS/SOSS', **mode}, np.ones(7, dtype=bool))
    for stage in ('lr', 'hr'):
        assert result['flux_' + stage].shape[1] == 7
        assert np.isfinite(result['flux_' + stage]).all()
        assert (result['flux_err_' + stage] > 0).all()
    has_low = 'low' in next(iter(mode.values()))
    assert (result['flux_lr'].shape[0] > 0) == has_low
    datacube._validate_binned_spectroscopy(result, expected_n_time=7)
assert 'exotedrf' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', script], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True,
                   text=True)

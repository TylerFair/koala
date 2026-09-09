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
def test_resolution_matches_upstream_fixture(method, one_dimensional):
    fixture = json.loads((Path(__file__).parent / 'data' /
                          'binning_exotedrf_2_2_0.json').read_text())
    wave, flux, err = [np.asarray(fixture[key]) for key in ('wave', 'flux', 'err')]
    expected = [np.asarray(a) for a in fixture['expected'][method]]
    if one_dimensional:
        flux, err = flux[:, 0], err[:, 0]
        expected[2:] = [a[:, 0] for a in expected[2:]]
    # Descending inputs must sort wavelengths and observations together.
    result = bin_at_resolution(wave[::-1], flux[::-1], err[::-1],
                               fixture['res'], method=method)
    for actual, reference in zip(result, expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('res', [0, -1, np.nan, np.inf, 0.1, 1000])
def test_resolution_rejects_invalid_grid(res):
    wave = np.linspace(1, 2, 20)
    with pytest.raises(ValueError):
        bin_at_resolution(wave, np.ones(20), np.ones(20), res)


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
    # The wrapper replaces this legacy linear error sum with quadrature.
    np.testing.assert_array_equal(err, [[2, 2, 0]])


def test_reference_wrapper_retains_quadrature_errors():
    from bin_to_reference_grid import bin_to_reference_grid

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
import createdatacube
import bin_to_reference_grid
wave = np.linspace(1, 2.5, 80)
flux = np.ones((7, 80))
for mode in ({'resolution': {'low': 10, 'high': 20}},
             {'pixels': {'low': 4, 'high': 2}},
             {'resolution': {'high': 20}},
             {'pixels': {'high': 2}}):
    result = createdatacube.bin_spectroscopy_data(
        wave, np.full(80, .01), flux, flux * .1,
        {'instrument': 'NIRISS/SOSS', **mode}, np.ones(7, dtype=bool))
    for stage in ('lr', 'hr'):
        assert result['flux_' + stage].shape[1] == 7
        assert np.isfinite(result['flux_' + stage]).all()
        assert (result['flux_err_' + stage] > 0).all()
    has_low = 'low' in next(iter(mode.values()))
    assert (result['flux_lr'].shape[0] > 0) == has_low
    createdatacube._validate_binned_spectroscopy(result, expected_n_time=7)
assert 'exotedrf' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', script], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True,
                   text=True)

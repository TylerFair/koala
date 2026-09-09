"""The low-resolution grid is optional: omit it and the bridge stage is skipped."""

import numpy as np
import pytest

import createdatacube
from koala.pipeline import _resolve_need_lowres


def _bin(cfg):
    wave = np.linspace(1.0, 2.5, 60)
    flux = np.ones((5, 60))
    return createdatacube.bin_spectroscopy_data(
        wave, np.full(60, 0.01), flux, flux * 0.1,
        {'instrument': 'NIRISS/SOSS', **cfg}, np.ones(5, dtype=bool))


@pytest.mark.parametrize('cfg', [
    {'resolution': {'high': 20}},
    {'resolution': {'high': 'native'}},
    {'pixels': {'high': 3}},
])
def test_missing_low_grid_yields_empty_low_resolution_arrays(cfg):
    result = _bin(cfg)
    assert result['wavelengths_lr'].shape == (0,)
    assert result['wavelengths_err_lr'].shape == (0,)
    assert result['flux_lr'].shape == (0, 5)
    assert result['flux_err_lr'].shape == (0, 5)
    assert result['flux_hr'].shape[0] > 0
    assert result['flux_hr'].shape[1] == 5
    # The shared validation still passes with zero low-resolution channels.
    createdatacube._validate_binned_spectroscopy(result, expected_n_time=5)


def test_present_low_grid_is_unchanged():
    result = _bin({'resolution': {'low': 10, 'high': 20}})
    assert result['flux_lr'].shape[0] > 0
    assert result['flux_lr'].shape[0] < result['flux_hr'].shape[0]


def test_need_lowres_defaults_follow_the_grid():
    assert _resolve_need_lowres({}, 20) is True
    assert _resolve_need_lowres({'need_lowres': False}, 20) is False
    assert _resolve_need_lowres({}, None) is False
    assert _resolve_need_lowres({'need_lowres': False}, None) is False


def test_explicit_need_lowres_without_grid_is_an_error():
    with pytest.raises(ValueError, match='resolution.low'):
        _resolve_need_lowres({'need_lowres': True}, None)

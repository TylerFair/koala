import numpy as np
import pytest

from koala.exclusions import (
    has_cut_phase_directive,
    integration_keep_mask,
    normalize_ranges,
    resolve_exclusions,
    time_keep_mask,
)


def test_normalize_accepts_single_pair_and_list_of_pairs():
    assert normalize_ranges(None, "x") == []
    assert normalize_ranges([1, 2], "x") == [(1, 2)]
    assert normalize_ranges([[1, 2], [5, None]], "x") == [(1, 2), (5, None)]
    assert normalize_ranges([None, 3.5], "x") == [(None, 3.5)]


@pytest.mark.parametrize("bad", ["1,2", [1, 2, 3], [[1, 2, 3]], [1, [2, 3]]])
def test_normalize_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        normalize_ranges(bad, "exclude_times")


def test_integration_ranges_are_inclusive_and_support_open_and_negative_ends():
    keep = integration_keep_mask(10, [(0, 1), (4, 5), (-2, None)])
    assert keep.tolist() == [False, False, True, True, False, False, True, True, False, False]
    keep = integration_keep_mask(10, [(None, 2)])
    assert keep.tolist() == [False, False, False] + [True] * 7


def test_integration_range_outside_series_warns_and_drops_nothing():
    with pytest.warns(RuntimeWarning):
        keep = integration_keep_mask(5, [(7, 9)])
    assert keep.all()


def test_time_ranges_are_inclusive_and_accept_expressions_and_open_ends():
    t = np.linspace(0.0, 1.0, 11)
    keep = time_keep_mask(t, [(0.15, 0.35), ("max(t) - 0.15", None)])
    assert keep.tolist() == [True, True, False, False, True, True, True, True, True, False, False]
    keep = time_keep_mask(t, [(None, "min(t) + 0.05")])
    assert (~keep).sum() == 1


def test_time_range_that_removes_nothing_warns():
    t = np.linspace(0.0, 1.0, 11)
    with pytest.warns(RuntimeWarning):
        keep = time_keep_mask(t, [(5.0, 6.0)])
    assert keep.all()


def test_resolve_merges_new_keys_with_legacy_keys():
    cfg = {
        "flags": {
            "exclude_times": [[1.0, 2.0]],
            "mask_start": 3.0,
            "mask_end": 4.0,
            "exclude_integrations": [[10, 12]],
        },
        "outlier_clip": {"mask_integrations_start": 5, "mask_integrations_end": 2},
    }
    times, integrations = resolve_exclusions(cfg)
    assert times == [(1.0, 2.0), (3.0, 4.0)]
    assert integrations == [(10, 12), (0, 4), (-2, None)]
    keep = integration_keep_mask(20, integrations)
    assert (~keep).sum() == 3 + 5 + 2


def test_resolve_reads_top_level_keys_and_legacy_lists():
    cfg = {
        "exclude_times": [5.0, 6.0],
        "flags": {"mask_start": [1.0, 2.0], "mask_end": [1.5, "cut_phase_to_transit"]},
    }
    times, integrations = resolve_exclusions(cfg)
    assert times == [(5.0, 6.0), (1.0, 1.5), (2.0, "cut_phase_to_transit")]
    assert integrations == []
    assert has_cut_phase_directive(times)
    # The directive is skipped by the range mask rather than evaluated.
    assert time_keep_mask(np.array([1.0, 1.2, 3.0]), times).tolist() == [False, False, True]


def test_resolve_rejects_non_integer_integration_bounds():
    with pytest.raises(ValueError):
        resolve_exclusions({"flags": {"exclude_integrations": [[0, 1.5]]}})


def test_process_spectroscopy_data_applies_both_exclusions(tmp_path):
    from astropy.io import fits
    import createdatacube

    n_time = 40
    wave = np.linspace(3.0, 4.0, 30)
    flux = np.full((n_time, 30), 100.0)
    time = 60000.0 + np.arange(n_time) * 0.01
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in (
        wave, np.full(30, 0.01), flux, flux * 0.01)]
    hdus.append(fits.ImageHDU(time))
    path = tmp_path / "spectrum.fits"
    fits.HDUList(hdus).writeto(path)

    cfg = {
        "instrument": "NIRCAM/F322W2",
        "planet": {"name": "TEST", "period": 3.0, "t0": 60000.2, "duration": 0.05},
        "resolution": {"high": "native"},
        "flags": {
            "exclude_integrations": [[0, 4], [-3, None]],
            "exclude_times": [[60000.10, 60000.12]],
        },
    }
    data = createdatacube.process_spectroscopy_data(
        "NIRCAM/F322W2", str(tmp_path), str(tmp_path), "TEST", cfg, str(path)
    )
    kept = np.asarray(data.time)
    assert kept.size == n_time - 5 - 3 - 3
    assert kept.min() == pytest.approx(60000.05)
    assert kept.max() == pytest.approx(60000.36)
    assert not np.any((kept >= 60000.10) & (kept <= 60000.12))

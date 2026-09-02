import numpy as np
import pytest

from models.temporal_binning import (
    TemporalBinningConfig,
    TransitContacts,
    validate_temporal_binning,
)


def _model_ensemble(time, n_models=8, n_channels=2, center=None):
    """Linear models are represented exactly at arithmetic-mean bin times."""

    time = np.asarray(time)
    if center is None:
        center = float(np.mean(time))
    models = np.empty((n_models, n_channels, time.size))
    for draw in range(n_models):
        for channel in range(n_channels):
            slope = (draw - 3.5) * 2.0e-7 + channel * 1.0e-7
            models[draw, channel] = 1.0 + slope * (time - center)
    return models


def _basic_data(n_time=101, n_channels=2):
    time = np.arange(n_time, dtype=float)
    model = _model_ensemble(time, n_channels=n_channels)
    flux = model[0].copy()
    error = np.full_like(flux, 2.0e-4)
    return time, flux, error, model


def test_disabled_is_identity_and_cannot_be_applied():
    time, flux, error, _ = _basic_data()
    result = validate_temporal_binning(time, flux, error)

    assert not result.accepted
    assert result.report.decision == "disabled"
    np.testing.assert_array_equal(result.candidate.time, time)
    np.testing.assert_array_equal(result.candidate.flux, flux)
    with pytest.raises(RuntimeError, match="not accepted"):
        result.validated_data()


def test_contacts_and_buffer_are_never_binned_and_gaps_are_not_crossed():
    time = np.concatenate((np.arange(0.0, 45.0), np.arange(50.0, 101.0)))
    model = _model_ensemble(time)
    flux = model[0].copy()
    error = np.full_like(flux, 2.0e-4)
    contacts = TransitContacts(40.0, 44.0, 56.0, 60.0)
    config = TemporalBinningConfig(
        enabled=True,
        max_points_per_bin=8,
        protection_buffer_cadences=2.0,
        min_compression_ratio=1.0,
        min_projected_speedup=1.0,
        min_projected_memory_reduction=1.0,
    )

    result = validate_temporal_binning(
        time,
        flux,
        error,
        contacts=contacts,
        unbinned_model=model,
        model_evaluator=lambda candidate_time: _model_ensemble(
            candidate_time, center=float(np.mean(time))
        ),
        config=config,
    )

    protected_indices = np.flatnonzero((time >= 38.0) & (time <= 62.0))
    assert result.report.n_protected == protected_indices.size
    for index in protected_indices:
        bin_index = result.candidate.original_to_bin[index]
        members = result.candidate.bin_members[bin_index]
        np.testing.assert_array_equal(members, [index])
        assert result.candidate.time[bin_index] == time[index]

    # The five-cadence data gap must be a boundary between two bins.
    left = np.flatnonzero(time == 44.0)[0]
    right = np.flatnonzero(time == 50.0)[0]
    assert result.candidate.original_to_bin[left] != result.candidate.original_to_bin[right]


def test_smooth_candidate_passes_likelihood_and_gain_gates():
    time, flux, error, _ = _basic_data()
    levels = 1.0 + np.arange(8, dtype=float)[:, None, None] * 2.0e-5
    model = np.broadcast_to(levels, (8, flux.shape[0], time.size)).copy()
    flux = model[0].copy()

    def evaluate_constant(candidate_time):
        return np.broadcast_to(
            levels, (8, flux.shape[0], candidate_time.size)
        ).copy()

    config = TemporalBinningConfig(
        enabled=True,
        max_points_per_bin=5,
        protection_buffer_cadences=2.0,
        max_rms_model_error_ppm=1.0e-8,
        max_abs_model_error_ppm=1.0e-7,
        max_abs_model_error_sigma=1.0e-8,
        max_abs_loglike_delta_per_channel=1.0e-8,
        min_compression_ratio=2.0,
        min_projected_speedup=1.5,
        min_projected_memory_reduction=1.5,
    )

    result = validate_temporal_binning(
        time,
        flux,
        error,
        contacts=TransitContacts(45.0, 47.0, 53.0, 55.0),
        unbinned_model=model,
        model_evaluator=evaluate_constant,
        config=config,
    )

    assert result.accepted, result.report.reasons
    assert result.validated_data() is result.candidate
    assert result.report.compression_ratio > 2.0
    assert result.report.projected_speedup > 1.5
    assert result.report.max_likelihood_identity_error < 1.0e-8
    assert np.max(np.abs(result.loglike_delta)) < 1.0e-8


def test_curved_model_is_rejected_even_when_compression_is_large():
    time = np.linspace(0.0, 10.0, 201)
    phases = np.linspace(0.0, 1.0, 8)

    def evaluate(at_time):
        curves = [
            1.0 + 2.0e-4 * np.sin(4.0 * at_time + phase)
            for phase in phases
        ]
        return np.asarray(curves)[:, np.newaxis, :]

    model = evaluate(time)
    flux = model[0]
    error = np.full_like(flux, 1.0e-4)
    config = TemporalBinningConfig(
        enabled=True,
        max_points_per_bin=12,
        protection_buffer_cadences=1.0,
        min_compression_ratio=2.0,
        min_projected_speedup=1.0,
        min_projected_memory_reduction=1.0,
    )

    result = validate_temporal_binning(
        time,
        flux,
        error,
        contacts=TransitContacts(4.5, 4.7, 5.3, 5.5),
        unbinned_model=model,
        model_evaluator=evaluate,
        config=config,
    )

    assert not result.accepted
    assert result.report.compression_ratio > 2.0
    assert result.report.max_abs_model_error_ppm > 5.0
    assert result.report.max_abs_loglike_delta_per_channel > 0.5
    assert any("model error" in reason for reason in result.report.reasons)
    with pytest.raises(RuntimeError, match="not accepted"):
        result.validated_data()


def test_low_gain_and_too_small_model_ensemble_are_rejected():
    time, flux, error, model = _basic_data()
    one_model = model[:1]
    config = TemporalBinningConfig(
        enabled=True,
        max_points_per_bin=4,
        protection_buffer_cadences=10.0,
        min_validation_models=8,
        min_compression_ratio=2.0,
        min_projected_speedup=1.5,
        min_projected_memory_reduction=1.5,
    )

    result = validate_temporal_binning(
        time,
        flux,
        error,
        contacts=TransitContacts(10.0, 20.0, 80.0, 90.0),
        unbinned_model=one_model,
        model_evaluator=lambda candidate_time: _model_ensemble(
            candidate_time, n_models=1, center=float(np.mean(time))
        ),
        config=config,
    )

    assert not result.accepted
    assert result.report.compression_ratio < 2.0
    assert any("only 1 model curves" in reason for reason in result.report.reasons)
    assert any("compression ratio" in reason for reason in result.report.reasons)


def test_missing_contacts_or_model_validation_fails_closed():
    time, flux, error, model = _basic_data()
    config = TemporalBinningConfig(
        enabled=True,
        min_compression_ratio=1.0,
        min_projected_speedup=1.0,
        min_projected_memory_reduction=1.0,
    )

    no_contacts = validate_temporal_binning(
        time,
        flux,
        error,
        unbinned_model=model,
        model_evaluator=lambda candidate_time: _model_ensemble(
            candidate_time, center=float(np.mean(time))
        ),
        config=config,
    )
    assert not no_contacts.accepted
    assert any("no transit contacts" in reason for reason in no_contacts.report.reasons)

    no_model = validate_temporal_binning(
        time,
        flux,
        error,
        contacts=TransitContacts(45.0, 47.0, 53.0, 55.0),
        config=config,
    )
    assert not no_model.accepted
    assert np.isnan(no_model.report.rms_model_error_ppm)
    assert any("scientific validation" in reason for reason in no_model.report.reasons)


def test_report_is_json_friendly_and_contacts_must_be_ordered():
    with pytest.raises(ValueError, match="t1 < t2 < t3 < t4"):
        TransitContacts(1.0, 3.0, 2.0, 4.0)

    time, flux, error, _ = _basic_data()
    result = validate_temporal_binning(time, flux, error)
    report = result.report.to_dict()
    assert isinstance(report["reasons"], list)
    assert report["decision"] == "disabled"

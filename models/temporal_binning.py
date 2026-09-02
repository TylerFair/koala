"""Conservative, opt-in temporal binning for spectroscopic light curves.

This module deliberately does not alter the fitting pipeline.  It builds and
validates a candidate time grid, and returns ``accepted=False`` whenever the
candidate is scientifically unsafe or the projected gain is too small.  A
caller must explicitly enable the proposal and then check ``accepted`` (or use
``validated_data``) before using the binned arrays.

The validation compares an ensemble of full forward models on the original
cadence with the same models evaluated at the candidate bin times.  For
independent Gaussian data, a data-only within-bin likelihood correction makes
the binned and expanded constant-within-bin likelihoods exactly equivalent.
Consequently, the reported corrected likelihood error measures the temporal
model approximation instead of the irrelevant likelihood-normalisation change
caused by reducing the number of data points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


Array = np.ndarray
ModelEvaluator = Callable[[Array], Array]


@dataclass(frozen=True)
class TransitContacts:
    """Four ordered transit contact times for one event."""

    t1: float
    t2: float
    t3: float
    t4: float

    def __post_init__(self) -> None:
        values = np.asarray((self.t1, self.t2, self.t3, self.t4), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Transit contact times must all be finite")
        if not np.all(np.diff(values) > 0.0):
            raise ValueError("Transit contacts must satisfy t1 < t2 < t3 < t4")

    @property
    def protected_interval(self) -> tuple[float, float]:
        """Return the full first-to-fourth-contact interval."""

        return float(self.t1), float(self.t4)


@dataclass(frozen=True)
class TemporalBinningConfig:
    """Scientific and performance gates for a temporal-binning proposal.

    The default is disabled.  The error limits assume continuum-normalised
    light curves, so an absolute flux error of ``1e-6`` is one ppm.

    ``time_dependent_runtime_fraction`` and ``time_dependent_memory_fraction``
    are Amdahl-law estimates, not benchmark results.  They should be replaced
    by measured fractions when available.
    """

    enabled: bool = False
    max_points_per_bin: int = 4
    max_time_span: float | None = None
    max_gap_cadences: float = 3.0
    protection_buffer_cadences: float = 2.0
    protection_buffer_time: float = 0.0
    require_transit_contacts: bool = True

    min_validation_models: int = 8
    max_rms_model_error_ppm: float = 1.0
    max_abs_model_error_ppm: float = 5.0
    max_abs_model_error_sigma: float = 0.05
    max_abs_loglike_delta_per_channel: float = 0.5

    min_compression_ratio: float = 1.5
    time_dependent_runtime_fraction: float = 0.9
    time_dependent_memory_fraction: float = 0.9
    min_projected_speedup: float = 1.25
    min_projected_memory_reduction: float = 1.25

    def __post_init__(self) -> None:
        if self.max_points_per_bin < 2:
            raise ValueError("max_points_per_bin must be at least 2")
        if self.max_time_span is not None and self.max_time_span <= 0.0:
            raise ValueError("max_time_span must be positive when supplied")
        if self.max_gap_cadences <= 0.0:
            raise ValueError("max_gap_cadences must be positive")
        if self.protection_buffer_cadences < 0.0:
            raise ValueError("protection_buffer_cadences cannot be negative")
        if self.protection_buffer_time < 0.0:
            raise ValueError("protection_buffer_time cannot be negative")
        if self.min_validation_models < 1:
            raise ValueError("min_validation_models must be at least one")

        nonnegative = (
            "max_rms_model_error_ppm",
            "max_abs_model_error_ppm",
            "max_abs_model_error_sigma",
            "max_abs_loglike_delta_per_channel",
        )
        for name in nonnegative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")

        if self.min_compression_ratio < 1.0:
            raise ValueError("min_compression_ratio must be at least one")
        if self.min_projected_speedup < 1.0:
            raise ValueError("min_projected_speedup must be at least one")
        if self.min_projected_memory_reduction < 1.0:
            raise ValueError(
                "min_projected_memory_reduction must be at least one"
            )
        for name in (
            "time_dependent_runtime_fraction",
            "time_dependent_memory_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class BinnedLightCurves:
    """Candidate channel-first light curves and their cadence mapping."""

    time: Array
    flux: Array
    flux_err: Array
    original_to_bin: Array
    bin_members: tuple[Array, ...]
    protected_original_mask: Array

    @property
    def n_original(self) -> int:
        return int(self.original_to_bin.size)

    @property
    def n_binned(self) -> int:
        return int(self.time.size)


@dataclass(frozen=True)
class TemporalBinningReport:
    """Serializable decision and validation metrics."""

    enabled: bool
    accepted: bool
    decision: str
    reasons: tuple[str, ...]

    n_original: int
    n_binned: int
    n_protected: int
    n_validation_models: int
    compression_ratio: float
    projected_speedup: float
    projected_memory_reduction: float

    rms_model_error_ppm: float
    max_abs_model_error_ppm: float
    max_abs_model_error_sigma: float
    max_abs_loglike_delta_per_channel: float
    rms_loglike_delta_per_channel: float
    max_likelihood_identity_error: float

    median_cadence: float
    protection_buffer: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly report contents."""

        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class TemporalBinningValidation:
    """A candidate grid together with its scientific validation result."""

    candidate: BinnedLightCurves
    report: TemporalBinningReport
    model_at_bin_time: Array | None
    reference_loglike: Array | None
    corrected_binned_loglike: Array | None
    loglike_delta: Array | None

    @property
    def accepted(self) -> bool:
        return self.report.accepted

    def validated_data(self) -> BinnedLightCurves:
        """Return the candidate only after all gates pass.

        Raising here prevents a rejected or merely inspected candidate from
        being wired into a fit by accident.
        """

        if not self.accepted:
            reasons = "; ".join(self.report.reasons) or self.report.decision
            raise RuntimeError(f"Temporal binning was not accepted: {reasons}")
        return self.candidate


def _normalise_channel_data(
    values: Array | Sequence[float], name: str, n_time: int
) -> tuple[Array, bool]:
    array = np.asarray(values, dtype=float)
    was_one_dimensional = array.ndim == 1
    if was_one_dimensional:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[-1] != n_time:
        raise ValueError(
            f"{name} must have shape (time,) or (channel, time); got "
            f"{array.shape} for {n_time} times"
        )
    return array, was_one_dimensional


def _normalise_model_ensemble(
    values: Array,
    *,
    n_channels: int,
    n_time: int,
    name: str,
    expected_models: int | None = None,
) -> Array:
    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        if n_channels != 1 or array.shape[0] != n_time:
            raise ValueError(
                f"One-dimensional {name} is only valid for one channel"
            )
        array = array[np.newaxis, np.newaxis, :]
    elif array.ndim == 2:
        if array.shape == (n_channels, n_time):
            array = array[np.newaxis, :, :]
        elif n_channels == 1 and array.shape[1] == n_time:
            # An ensemble for a single channel: (model, time).
            array = array[:, np.newaxis, :]
        else:
            raise ValueError(
                f"Two-dimensional {name} must have shape "
                f"({n_channels}, {n_time}), or (model, {n_time}) for one "
                f"channel; got {array.shape}"
            )
    elif array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (channel, time) or "
            f"(model, channel, time); got {array.shape}"
        )

    if array.shape[1:] != (n_channels, n_time):
        raise ValueError(
            f"{name} has trailing shape {array.shape[1:]}, expected "
            f"({n_channels}, {n_time})"
        )
    if expected_models is not None and array.shape[0] != expected_models:
        raise ValueError(
            f"{name} returned {array.shape[0]} validation models; expected "
            f"{expected_models}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalise_contacts(
    contacts: TransitContacts | Iterable[TransitContacts] | None,
) -> tuple[TransitContacts, ...]:
    if contacts is None:
        return ()
    if isinstance(contacts, TransitContacts):
        return (contacts,)
    result = tuple(contacts)
    if not all(isinstance(item, TransitContacts) for item in result):
        raise TypeError("contacts must contain only TransitContacts values")
    return result


def _cadence_and_protected_mask(
    time: Array,
    contacts: tuple[TransitContacts, ...],
    config: TemporalBinningConfig,
) -> tuple[float, float, Array]:
    differences = np.diff(time)
    positive = differences[differences > 0.0]
    if positive.size == 0:
        raise ValueError("time must be strictly increasing")
    median_cadence = float(np.median(positive))
    protection_buffer = max(
        float(config.protection_buffer_time),
        float(config.protection_buffer_cadences) * median_cadence,
    )
    protected = np.zeros(time.size, dtype=bool)
    for event in contacts:
        start, end = event.protected_interval
        protected |= (time >= start - protection_buffer) & (
            time <= end + protection_buffer
        )
    return median_cadence, protection_buffer, protected


def _build_bin_members(
    time: Array,
    protected: Array,
    median_cadence: float,
    config: TemporalBinningConfig,
) -> tuple[Array, ...]:
    groups: list[Array] = []
    index = 0
    n_time = time.size
    maximum_gap = config.max_gap_cadences * median_cadence

    while index < n_time:
        if protected[index]:
            groups.append(np.asarray((index,), dtype=np.int64))
            index += 1
            continue

        start = index
        stop = start + 1
        while stop < n_time:
            if protected[stop]:
                break
            if stop - start >= config.max_points_per_bin:
                break
            if time[stop] - time[stop - 1] > maximum_gap:
                break
            if (
                config.max_time_span is not None
                and time[stop] - time[start] > config.max_time_span
            ):
                break
            stop += 1
        groups.append(np.arange(start, stop, dtype=np.int64))
        index = stop

    # This is a safety invariant, not only a unit-test expectation.
    if any(group.size != 1 and np.any(protected[group]) for group in groups):
        raise RuntimeError("Internal error: a protected cadence was binned")
    return tuple(groups)


def _bin_light_curves(
    time: Array,
    flux: Array,
    flux_err: Array,
    groups: tuple[Array, ...],
    protected: Array,
) -> BinnedLightCurves:
    n_channels = flux.shape[0]
    n_bins = len(groups)
    binned_time = np.empty(n_bins, dtype=float)
    binned_flux = np.empty((n_channels, n_bins), dtype=float)
    binned_error = np.empty((n_channels, n_bins), dtype=float)
    original_to_bin = np.empty(time.size, dtype=np.int64)

    for bin_index, members in enumerate(groups):
        binned_time[bin_index] = float(np.mean(time[members]))
        weights = 1.0 / np.square(flux_err[:, members])
        weight_sum = np.sum(weights, axis=1)
        binned_flux[:, bin_index] = np.sum(
            weights * flux[:, members], axis=1
        ) / weight_sum
        binned_error[:, bin_index] = 1.0 / np.sqrt(weight_sum)
        original_to_bin[members] = bin_index

    return BinnedLightCurves(
        time=binned_time,
        flux=binned_flux,
        flux_err=binned_error,
        original_to_bin=original_to_bin,
        bin_members=groups,
        protected_original_mask=protected.copy(),
    )


def _identity_candidate(
    time: Array, flux: Array, flux_err: Array, protected: Array
) -> BinnedLightCurves:
    groups = tuple(
        np.asarray((index,), dtype=np.int64) for index in range(time.size)
    )
    return BinnedLightCurves(
        time=time.copy(),
        flux=flux.copy(),
        flux_err=flux_err.copy(),
        original_to_bin=np.arange(time.size, dtype=np.int64),
        bin_members=groups,
        protected_original_mask=protected.copy(),
    )


def _projected_reduction(compression: float, fraction: float) -> float:
    """Amdahl-law reduction for a compressed time-dependent component."""

    denominator = (1.0 - fraction) + fraction / compression
    return float(1.0 / denominator)


def _gaussian_loglike(data: Array, model: Array, error: Array) -> Array:
    """Return log likelihood with shape ``(model, channel)``."""

    variance = np.square(error)[np.newaxis, :, :]
    residual = data[np.newaxis, :, :] - model
    return -0.5 * np.sum(
        np.square(residual) / variance + np.log(2.0 * np.pi * variance),
        axis=-1,
    )


def _within_bin_likelihood_offset(
    flux: Array,
    flux_err: Array,
    candidate: BinnedLightCurves,
) -> Array:
    """Data-only offset from binned to expanded constant-model likelihood."""

    offset = np.zeros(flux.shape[0], dtype=float)
    for bin_index, members in enumerate(candidate.bin_members):
        y = flux[:, members]
        sigma = flux_err[:, members]
        y_bar = candidate.flux[:, bin_index][:, np.newaxis]
        sigma_bar = candidate.flux_err[:, bin_index]
        scatter = np.sum(np.square((y - y_bar) / sigma), axis=1)
        normalisation = np.sum(
            np.log(2.0 * np.pi * np.square(sigma)), axis=1
        ) - np.log(2.0 * np.pi * np.square(sigma_bar))
        offset += -0.5 * (scatter + normalisation)
    return offset


def _empty_metrics() -> dict[str, float]:
    return {
        "rms_model_error_ppm": float("nan"),
        "max_abs_model_error_ppm": float("nan"),
        "max_abs_model_error_sigma": float("nan"),
        "max_abs_loglike_delta_per_channel": float("nan"),
        "rms_loglike_delta_per_channel": float("nan"),
        "max_likelihood_identity_error": float("nan"),
    }


def validate_temporal_binning(
    time: Array | Sequence[float],
    flux: Array,
    flux_err: Array,
    *,
    contacts: TransitContacts | Iterable[TransitContacts] | None = None,
    unbinned_model: Array | None = None,
    model_evaluator: ModelEvaluator | None = None,
    config: TemporalBinningConfig | None = None,
) -> TemporalBinningValidation:
    """Build and scientifically validate a conservative candidate time grid.

    Parameters
    ----------
    time, flux, flux_err
        Increasing times and channel-first light curves.  One-dimensional flux
        and error arrays are accepted for a single channel.
    contacts
        One or more four-contact transit windows.  Every cadence from first to
        fourth contact, plus the configured buffer on both sides, is retained
        individually.  By default a proposal without contacts is rejected.
    unbinned_model
        Full forward-model values with shape ``(model, channel, time)``.  A
        single ``(channel, time)`` model is accepted, but the default gate asks
        for at least eight curves spanning posterior/plausible parameter space.
    model_evaluator
        Callable evaluated on the candidate times.  It must return the same
        model ensemble with the final dimension replaced by ``n_binned``.
    config
        Explicit opt-in and all validation/performance tolerances.

    Notes
    -----
    The proposal estimates speed and memory reduction from time-axis
    compression and user-supplied Amdahl fractions.  It does not claim a GPU
    benchmark.  Rebenchmark the accepted configuration in the real sampler.
    """

    if config is None:
        config = TemporalBinningConfig()

    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("time must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values")
    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("time must be strictly increasing")

    flux_array, _ = _normalise_channel_data(flux, "flux", time_array.size)
    error_array, _ = _normalise_channel_data(
        flux_err, "flux_err", time_array.size
    )
    if flux_array.shape != error_array.shape:
        raise ValueError(
            f"flux and flux_err shapes differ: {flux_array.shape} versus "
            f"{error_array.shape}"
        )
    if not np.all(np.isfinite(flux_array)):
        raise ValueError("flux must contain only finite values")
    if not np.all(np.isfinite(error_array)) or np.any(error_array <= 0.0):
        raise ValueError("flux_err must contain only finite positive values")

    contact_values = _normalise_contacts(contacts)
    median_cadence, protection_buffer, protected = (
        _cadence_and_protected_mask(time_array, contact_values, config)
    )

    if not config.enabled:
        candidate = _identity_candidate(
            time_array, flux_array, error_array, protected
        )
        report = TemporalBinningReport(
            enabled=False,
            accepted=False,
            decision="disabled",
            reasons=(
                "temporal binning is opt-in; set config.enabled=True to test "
                "a candidate",
            ),
            n_original=time_array.size,
            n_binned=time_array.size,
            n_protected=int(np.sum(protected)),
            n_validation_models=0,
            compression_ratio=1.0,
            projected_speedup=1.0,
            projected_memory_reduction=1.0,
            median_cadence=median_cadence,
            protection_buffer=protection_buffer,
            **_empty_metrics(),
        )
        return TemporalBinningValidation(
            candidate=candidate,
            report=report,
            model_at_bin_time=None,
            reference_loglike=None,
            corrected_binned_loglike=None,
            loglike_delta=None,
        )

    groups = _build_bin_members(
        time_array, protected, median_cadence, config
    )
    candidate = _bin_light_curves(
        time_array, flux_array, error_array, groups, protected
    )
    compression = float(time_array.size / candidate.n_binned)
    projected_speedup = _projected_reduction(
        compression, config.time_dependent_runtime_fraction
    )
    projected_memory = _projected_reduction(
        compression, config.time_dependent_memory_fraction
    )

    reasons: list[str] = []
    if config.require_transit_contacts and not contact_values:
        reasons.append(
            "no transit contacts were supplied, so ingress/egress protection "
            "cannot be verified"
        )
    if unbinned_model is None or model_evaluator is None:
        reasons.append(
            "both unbinned_model and model_evaluator are required for "
            "scientific validation"
        )

    model_at_bins: Array | None = None
    reference_loglike: Array | None = None
    corrected_binned_loglike: Array | None = None
    loglike_delta: Array | None = None
    n_models = 0
    metrics = _empty_metrics()

    if unbinned_model is not None and model_evaluator is not None:
        model_unbinned = _normalise_model_ensemble(
            unbinned_model,
            n_channels=flux_array.shape[0],
            n_time=time_array.size,
            name="unbinned_model",
        )
        n_models = int(model_unbinned.shape[0])
        model_at_bins = _normalise_model_ensemble(
            model_evaluator(candidate.time),
            n_channels=flux_array.shape[0],
            n_time=candidate.n_binned,
            name="model_evaluator output",
            expected_models=n_models,
        )

        expanded_model = model_at_bins[:, :, candidate.original_to_bin]
        model_error = expanded_model - model_unbinned
        model_error_ppm = 1.0e6 * model_error
        model_error_sigma = model_error / error_array[np.newaxis, :, :]

        reference_loglike = _gaussian_loglike(
            flux_array, model_unbinned, error_array
        )
        expanded_loglike = _gaussian_loglike(
            flux_array, expanded_model, error_array
        )
        binned_loglike = _gaussian_loglike(
            candidate.flux, model_at_bins, candidate.flux_err
        )
        offset = _within_bin_likelihood_offset(
            flux_array, error_array, candidate
        )
        corrected_binned_loglike = binned_loglike + offset[np.newaxis, :]
        loglike_delta = corrected_binned_loglike - reference_loglike

        metrics = {
            "rms_model_error_ppm": float(
                np.sqrt(np.mean(np.square(model_error_ppm)))
            ),
            "max_abs_model_error_ppm": float(
                np.max(np.abs(model_error_ppm))
            ),
            "max_abs_model_error_sigma": float(
                np.max(np.abs(model_error_sigma))
            ),
            "max_abs_loglike_delta_per_channel": float(
                np.max(np.abs(loglike_delta))
            ),
            "rms_loglike_delta_per_channel": float(
                np.sqrt(np.mean(np.square(loglike_delta)))
            ),
            "max_likelihood_identity_error": float(
                np.max(np.abs(corrected_binned_loglike - expanded_loglike))
            ),
        }

        if n_models < config.min_validation_models:
            reasons.append(
                f"only {n_models} model curves were validated; at least "
                f"{config.min_validation_models} are required"
            )
        if metrics["rms_model_error_ppm"] > config.max_rms_model_error_ppm:
            reasons.append(
                "RMS temporal model error exceeds the configured ppm limit"
            )
        if metrics["max_abs_model_error_ppm"] > config.max_abs_model_error_ppm:
            reasons.append(
                "maximum temporal model error exceeds the configured ppm limit"
            )
        if (
            metrics["max_abs_model_error_sigma"]
            > config.max_abs_model_error_sigma
        ):
            reasons.append(
                "maximum temporal model error is too large relative to the "
                "photometric uncertainty"
            )
        if (
            metrics["max_abs_loglike_delta_per_channel"]
            > config.max_abs_loglike_delta_per_channel
        ):
            reasons.append(
                "corrected binned likelihood differs too much from the "
                "unbinned likelihood"
            )
        # This identity should be roundoff-limited.  A looser scale-aware gate
        # catches implementation/data-shape mistakes without rejecting long
        # light curves because of ordinary floating-point summation order.
        identity_scale = max(
            1.0,
            float(np.max(np.abs(expanded_loglike))),
        )
        if metrics["max_likelihood_identity_error"] > 1.0e-10 * identity_scale:
            reasons.append(
                "the Gaussian within-bin likelihood identity failed its "
                "numerical consistency check"
            )

    if compression < config.min_compression_ratio:
        reasons.append(
            f"compression ratio {compression:.3g} is below the required "
            f"{config.min_compression_ratio:.3g}"
        )
    if projected_speedup < config.min_projected_speedup:
        reasons.append(
            f"projected speedup {projected_speedup:.3g}x is below the required "
            f"{config.min_projected_speedup:.3g}x"
        )
    if projected_memory < config.min_projected_memory_reduction:
        reasons.append(
            f"projected memory reduction {projected_memory:.3g}x is below the "
            f"required {config.min_projected_memory_reduction:.3g}x"
        )

    accepted = len(reasons) == 0
    report = TemporalBinningReport(
        enabled=True,
        accepted=accepted,
        decision="accept" if accepted else "reject",
        reasons=tuple(reasons),
        n_original=time_array.size,
        n_binned=candidate.n_binned,
        n_protected=int(np.sum(protected)),
        n_validation_models=n_models,
        compression_ratio=compression,
        projected_speedup=projected_speedup,
        projected_memory_reduction=projected_memory,
        median_cadence=median_cadence,
        protection_buffer=protection_buffer,
        **metrics,
    )
    return TemporalBinningValidation(
        candidate=candidate,
        report=report,
        model_at_bin_time=model_at_bins,
        reference_loglike=reference_loglike,
        corrected_binned_loglike=corrected_binned_loglike,
        loglike_delta=loglike_delta,
    )


__all__ = [
    "BinnedLightCurves",
    "TemporalBinningConfig",
    "TemporalBinningReport",
    "TemporalBinningValidation",
    "TransitContacts",
    "validate_temporal_binning",
]

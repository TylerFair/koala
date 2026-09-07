"""Exact out-of-transit sufficient statistics for sampled linear trends."""

from __future__ import annotations

import numpy as np


STATISTIC_KEYS = (
    "oot_reference_beta",
    "oot_group_yerr",
    "oot_group_count",
    "oot_group_reference_sse",
    "oot_group_x_reference_residual",
    "oot_group_xx",
)


_LINEAR_SPECTRO_TREND_NAMES = {
    frozenset({"linear"}): ("c", "v"),
    frozenset({"quadratic"}): ("c", "v", "v2"),
    frozenset({"cubic"}): ("c", "v", "v2", "v3"),
    frozenset({"quartic"}): ("c", "v", "v2", "v3", "v4"),
    frozenset({"explinear_spectroscopic"}): ("c", "v", "A"),
    frozenset({"spot_spectroscopic"}): ("c", "A_spot"),
    frozenset({"quadratic", "spot_spectroscopic"}): (
        "c", "v", "v2", "A_spot"
    ),
    frozenset({"2spot_spectroscopic"}): (
        "c", "A_spot", "A_spot2"
    ),
    frozenset({"linear_discontinuity_spectroscopic"}): ("c", "A_jump"),
    frozenset({
        "spot_spectroscopic", "linear_discontinuity_spectroscopic"
    }): ("c", "A_spot", "A_jump"),
}


def linear_spectro_trend_coefficient_names(detrend_type):
    """Return sampled coefficients when a spectroscopic trend is linear.

    ``None`` deliberately excludes GP trends, free-timescale exponentials,
    Gaussian-marginalized inference, and any unrecognized composition.
    """
    canonical = {
        "quadratic_spot_spectroscopic": "quadratic+spot_spectroscopic",
    }.get(str(detrend_type), str(detrend_type))
    return _LINEAR_SPECTRO_TREND_NAMES.get(
        frozenset(canonical.split("+"))
    )


def build_linear_spectro_trend_design(
    detrend_type,
    time,
    *,
    exp_trend=None,
    spot_trend=None,
    spot_trend2=None,
    jump_trend=None,
):
    """Build the fixed basis matrix for an eligible additive trend."""
    names = linear_spectro_trend_coefficient_names(detrend_type)
    if names is None:
        raise ValueError(
            f"Trend {detrend_type!r} is not an eligible linear spectroscopic "
            "trend."
        )
    time = np.asarray(time, dtype=np.float64)
    centered = time - np.min(time)
    external = {
        "A": exp_trend,
        "A_spot": spot_trend,
        "A_spot2": spot_trend2,
        "A_jump": jump_trend,
    }
    columns = []
    for name in names:
        if name == "c":
            column = np.ones_like(time)
        elif name == "v":
            column = centered
        elif name.startswith("v") and name[1:].isdigit():
            column = centered ** int(name[1:])
        else:
            column = external[name]
            if column is None:
                raise ValueError(
                    f"Trend {detrend_type!r} requires the fixed {name} basis."
                )
            column = np.asarray(column, dtype=np.float64)
            if column.shape != time.shape:
                raise ValueError(
                    f"The fixed {name} basis must have the same shape as time."
                )
        columns.append(column)
    return names, np.column_stack(columns)


def build_linear_oot_statistics(
    time,
    flux,
    error,
    transit_window_indices,
    design,
    reference_beta,
    likelihood_mask=None,
):
    """Precompute exact grouped Gaussian quadratics outside transit windows.

    Accumulation uses platform extended precision before the sufficient
    statistics are rounded once to the float64 representation consumed by
    JAX. Grouping is by the exact float64 reported-error bit pattern.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.atleast_2d(np.asarray(flux, dtype=np.float64))
    error = np.atleast_2d(np.asarray(error, dtype=np.float64))
    if error.shape[0] == 1 and flux.shape[0] > 1:
        error = np.broadcast_to(error, flux.shape)
    if flux.shape != error.shape or flux.shape[1] != time.size:
        raise ValueError("flux/error must have shape [channel, time].")
    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != time.size:
        raise ValueError("design must have shape [time, coefficient].")
    reference_beta = np.asarray(reference_beta, dtype=np.float64)
    expected_beta_shape = (flux.shape[0], design.shape[1])
    if reference_beta.shape != expected_beta_shape:
        raise ValueError(
            f"reference_beta must have shape {expected_beta_shape}."
        )

    if likelihood_mask is None:
        valid = np.ones(flux.shape, dtype=bool)
    else:
        valid = np.asarray(likelihood_mask, dtype=bool)
        valid = np.broadcast_to(valid, flux.shape).copy()
    in_window = np.zeros(time.size, dtype=bool)
    in_window[np.asarray(transit_window_indices, dtype=np.int64)] = True
    valid &= ~in_window[None, :]
    valid &= np.isfinite(flux) & np.isfinite(error)

    extended = np.longdouble
    design = np.asarray(design, dtype=extended)
    num_coefficients = design.shape[1]

    lane_payloads = []
    maximum_groups = 0
    for lane in range(flux.shape[0]):
        lane_indices = np.flatnonzero(valid[lane])
        group_errors, group_ids = np.unique(
            error[lane, lane_indices], return_inverse=True
        )
        group_count = len(group_errors)
        maximum_groups = max(maximum_groups, group_count)
        beta = np.asarray(reference_beta[lane], dtype=extended)
        x = design[lane_indices]
        residual = np.asarray(flux[lane, lane_indices], dtype=extended) - x @ beta
        counts = np.zeros(group_count, dtype=np.int64)
        reference_sse = np.zeros(group_count, dtype=extended)
        x_residual = np.zeros(
            (group_count, num_coefficients), dtype=extended
        )
        xx = np.zeros(
            (group_count, num_coefficients, num_coefficients), dtype=extended
        )
        for group in range(group_count):
            selected = group_ids == group
            group_x = x[selected]
            group_residual = residual[selected]
            counts[group] = int(np.count_nonzero(selected))
            reference_sse[group] = np.sum(
                group_residual * group_residual, dtype=extended
            )
            for left in range(num_coefficients):
                x_residual[group, left] = np.sum(
                    group_x[:, left] * group_residual, dtype=extended
                )
                for right in range(num_coefficients):
                    xx[group, left, right] = np.sum(
                        group_x[:, left] * group_x[:, right], dtype=extended
                    )
        lane_payloads.append((
            group_errors, counts, reference_sse, x_residual, xx
        ))

    result = {
        "oot_reference_beta": reference_beta,
        "oot_group_yerr": np.ones((flux.shape[0], maximum_groups)),
        "oot_group_count": np.zeros((flux.shape[0], maximum_groups)),
        "oot_group_reference_sse": np.zeros(
            (flux.shape[0], maximum_groups)
        ),
        "oot_group_x_reference_residual": np.zeros(
            (flux.shape[0], maximum_groups, num_coefficients)
        ),
        "oot_group_xx": np.zeros(
            (
                flux.shape[0], maximum_groups,
                num_coefficients, num_coefficients,
            )
        ),
    }
    for lane, payload in enumerate(lane_payloads):
        group_errors, counts, reference_sse, x_residual, xx = payload
        size = len(group_errors)
        result["oot_group_yerr"][lane, :size] = group_errors
        result["oot_group_count"][lane, :size] = counts
        result["oot_group_reference_sse"][lane, :size] = reference_sse
        result["oot_group_x_reference_residual"][lane, :size] = x_residual
        result["oot_group_xx"][lane, :size] = xx
    return result


def build_explinear_oot_statistics(
    time,
    flux,
    error,
    transit_window_indices,
    exp_trend,
    reference_beta,
    likelihood_mask=None,
):
    """Compatibility wrapper for the original fixed-timescale path."""
    _, design = build_linear_spectro_trend_design(
        "explinear_spectroscopic", time, exp_trend=exp_trend
    )
    return build_linear_oot_statistics(
        time,
        flux,
        error,
        transit_window_indices,
        design,
        reference_beta,
        likelihood_mask=likelihood_mask,
    )


__all__ = [
    "STATISTIC_KEYS",
    "build_explinear_oot_statistics",
    "build_linear_oot_statistics",
    "build_linear_spectro_trend_design",
    "linear_spectro_trend_coefficient_names",
]

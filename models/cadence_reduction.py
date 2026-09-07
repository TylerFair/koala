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


def build_explinear_oot_statistics(
    time,
    flux,
    error,
    transit_window_indices,
    exp_trend,
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
    exp_trend = np.asarray(exp_trend, dtype=np.float64)
    if exp_trend.shape != time.shape:
        raise ValueError("exp_trend must have the same shape as time.")
    reference_beta = np.asarray(reference_beta, dtype=np.float64)
    if reference_beta.shape != (flux.shape[0], 3):
        raise ValueError("reference_beta must have shape [channel, 3].")

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
    centered_time = np.asarray(time - np.min(time), dtype=extended)
    design = np.stack((
        np.ones(time.size, dtype=extended),
        centered_time,
        np.asarray(exp_trend, dtype=extended),
    ), axis=1)

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
        x_residual = np.zeros((group_count, 3), dtype=extended)
        xx = np.zeros((group_count, 3, 3), dtype=extended)
        for group in range(group_count):
            selected = group_ids == group
            group_x = x[selected]
            group_residual = residual[selected]
            counts[group] = int(np.count_nonzero(selected))
            reference_sse[group] = np.sum(
                group_residual * group_residual, dtype=extended
            )
            for left in range(3):
                x_residual[group, left] = np.sum(
                    group_x[:, left] * group_residual, dtype=extended
                )
                for right in range(3):
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
            (flux.shape[0], maximum_groups, 3)
        ),
        "oot_group_xx": np.zeros(
            (flux.shape[0], maximum_groups, 3, 3)
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


__all__ = ["STATISTIC_KEYS", "build_explinear_oot_statistics"]

"""Integration and time-range exclusions for the spectroscopic data cube.

Two configuration keys, each optional, remove data before any fitting:

``exclude_integrations``
    A list of ``[first, last]`` integration-index pairs (0-based, both ends
    inclusive) to drop from the file, for example ``[[0, 19], [400, 405]]``.
    ``null`` at either end makes the range open, and a negative index counts
    from the end of the series, so ``[[-5, null]]`` drops the last five.

``exclude_times``
    A list of ``[start, end]`` time pairs in the file's time system (BMJD_TDB),
    both ends inclusive, for example ``[[60795.15, 60795.26]]``. Either end may
    be ``null`` for an open range, or a string expression in ``t`` such as
    ``"min(t) + 0.007"``.

A single pair such as ``exclude_times: [60795.15, 60795.26]`` is accepted as
shorthand for a one-element list. Both keys may live under ``flags`` or at the
top level of the configuration.

The older keys ``flags.mask_start``/``flags.mask_end`` and
``outlier_clip.mask_integrations_start``/``mask_integrations_end`` are still
read and merged with the new ones. ``cut_phase_to_transit`` remains a valid
value of ``mask_start``/``mask_end``.
"""

from __future__ import annotations

import numbers
import warnings

import numpy as np

CUT_PHASE_DIRECTIVE = "cut_phase_to_transit"

_TIME_NAMESPACE_KEYS = ("t", "time")


def _is_scalar_like(value):
    return value is None or isinstance(value, (str, numbers.Number, np.generic))


def normalize_ranges(value, name):
    """Return ``value`` as a list of ``(start, end)`` pairs.

    Accepts ``None`` (no ranges), a single ``[start, end]`` pair, or a list
    of such pairs. Elements may be numbers, ``None``, or strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        raise ValueError(
            f"{name} must be a list of [start, end] pairs, got the string "
            f"{value!r}."
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a list of [start, end] pairs, got {value!r}."
        ) from exc
    if len(items) == 2 and all(_is_scalar_like(item) for item in items):
        items = [items]
    pairs = []
    for item in items:
        if _is_scalar_like(item):
            raise ValueError(
                f"{name} entries must be [start, end] pairs, got {item!r}. "
                f"Write a single range as [[start, end]]."
            )
        pair = list(item)
        if len(pair) != 2:
            raise ValueError(
                f"{name} entries must be [start, end] pairs, got {item!r}."
            )
        pairs.append((pair[0], pair[1]))
    return pairs


def _pairs_from_legacy_masks(mask_start, mask_end):
    """Convert ``mask_start``/``mask_end`` (scalars or lists) to pairs."""
    if mask_start is None and mask_end is None:
        return []
    if mask_start is False:
        mask_start = None
    if mask_end is False:
        mask_end = None
    if mask_start is None and mask_end is None:
        return []

    def _as_list(value):
        if value is None:
            return None
        if _is_scalar_like(value):
            return [value]
        return list(value)

    starts = _as_list(mask_start)
    ends = _as_list(mask_end)
    if starts is None:
        return [(None, end) for end in ends]
    if ends is None:
        return [(start, None) for start in starts]
    if len(starts) != len(ends):
        raise ValueError(
            "flags.mask_start and flags.mask_end lists must have the same length."
        )
    return list(zip(starts, ends))


def _pairs_from_legacy_trim(mask_integrations_start, mask_integrations_end):
    pairs = []
    if mask_integrations_start:
        pairs.append((0, int(mask_integrations_start) - 1))
    if mask_integrations_end:
        pairs.append((-int(mask_integrations_end), None))
    return pairs


def _lookup(cfg, key):
    flags = cfg.get("flags") or {}
    if key in flags:
        return flags[key]
    return cfg.get(key)


def resolve_exclusions(cfg, *, mask_start=None, mask_end=None,
                       mask_integrations_start=None, mask_integrations_end=None):
    """Return ``(time_ranges, integration_ranges)`` from a configuration.

    The keyword arguments let callers supply the legacy values directly;
    otherwise they are read from ``cfg``. New-style keys and legacy keys are
    merged, so either or both may be present.
    """
    flags = cfg.get("flags") or {}
    outlier_clip = cfg.get("outlier_clip") or {}
    if mask_start is None:
        mask_start = flags.get("mask_start")
    if mask_end is None:
        mask_end = flags.get("mask_end")
    if mask_integrations_start is None:
        mask_integrations_start = outlier_clip.get("mask_integrations_start")
    if mask_integrations_end is None:
        mask_integrations_end = outlier_clip.get("mask_integrations_end")

    time_ranges = normalize_ranges(_lookup(cfg, "exclude_times"), "exclude_times")
    time_ranges += _pairs_from_legacy_masks(mask_start, mask_end)
    integration_ranges = normalize_ranges(
        _lookup(cfg, "exclude_integrations"), "exclude_integrations"
    )
    integration_ranges += _pairs_from_legacy_trim(
        mask_integrations_start, mask_integrations_end
    )
    for start, end in integration_ranges:
        for bound in (start, end):
            if bound is not None and (
                isinstance(bound, bool) or int(bound) != bound
            ):
                raise ValueError(
                    f"exclude_integrations bounds must be integers or null, "
                    f"got {bound!r}."
                )
    return time_ranges, integration_ranges


def is_cut_phase_directive(value):
    return isinstance(value, str) and value.strip().lower() == CUT_PHASE_DIRECTIVE


def has_cut_phase_directive(time_ranges):
    return any(
        is_cut_phase_directive(start) or is_cut_phase_directive(end)
        for start, end in time_ranges
    )


def integration_keep_mask(n_integrations, integration_ranges):
    """Boolean mask of integrations to keep after dropping the given ranges."""
    n = int(n_integrations)
    keep = np.ones(n, dtype=bool)
    for start, end in integration_ranges:
        first = 0 if start is None else int(start)
        last = n - 1 if end is None else int(end)
        if first < 0:
            first += n
        if last < 0:
            last += n
        first = max(first, 0)
        last = min(last, n - 1)
        if last < first:
            warnings.warn(
                f"exclude_integrations range [{start}, {end}] selects no "
                f"integrations out of {n}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        keep[first:last + 1] = False
    return keep


def evaluate_time_bound(value, time):
    """Evaluate a time bound: a number, ``None``, or an expression in ``t``."""
    if value is None:
        return None
    if isinstance(value, str):
        expression = value.strip()
        namespace = {
            "np": np,
            "jnp": np,
            "min": np.min,
            "max": np.max,
            "median": np.median,
        }
        for key in _TIME_NAMESPACE_KEYS:
            namespace[key] = time
        try:
            return float(eval(expression, {"__builtins__": {}}, namespace))
        except Exception as exc:
            raise ValueError(
                f"Could not evaluate the exclude_times bound {value!r}: {exc}"
            ) from exc
    return float(value)


def time_keep_mask(time, time_ranges):
    """Boolean mask of cadences to keep after dropping the given time ranges.

    ``cut_phase_to_transit`` directives are ignored here; callers handle them
    separately with :func:`has_cut_phase_directive`.
    """
    time = np.asarray(time, dtype=float)
    keep = np.ones(time.shape, dtype=bool)
    if time.size == 0:
        return keep
    for start, end in time_ranges:
        if is_cut_phase_directive(start) or is_cut_phase_directive(end):
            continue
        if start is None and end is None:
            continue
        lower = evaluate_time_bound(start, time)
        upper = evaluate_time_bound(end, time)
        if lower is None:
            lower = float(np.min(time))
        if upper is None:
            upper = float(np.max(time))
        if upper < lower:
            lower, upper = upper, lower
        inside = (time >= lower) & (time <= upper)
        if not np.any(inside):
            warnings.warn(
                f"exclude_times range [{start}, {end}] removes no cadences "
                f"(data span {time.min():.5f} to {time.max():.5f}).",
                RuntimeWarning,
                stacklevel=2,
            )
        keep &= ~inside
    return keep

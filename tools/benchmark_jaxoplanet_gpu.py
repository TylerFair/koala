#!/usr/bin/env python3
"""GPU-oriented jaxoplanet benchmark with exact full/window parity checks.

This is deliberately independent of NumPyro while exercising the same
``models.jaxoplanet.core`` routes used by the production model builder.  It
measures the two kernels that matter for HMC: the batched forward model and a
scalar Gaussian log likelihood together with its gradient.  The optional
static window evaluates the transit only on a conservative, compile-time slice
while retaining the trend and likelihood on every cadence.  Production mode
separately times the original per-channel stock wrapper, stock occultation
with shared phases, and streamed occultation with shared phases.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import jax

jax.config.update("jax_enable_x64", True)
# The polynomial constants below use JAX, so honor the CLI platform before
# they can initialize a backend.  Full argparse validation still happens in
# main().
if __name__ == "__main__":
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == "--platform" and index + 1 < len(sys.argv):
            requested_platform = sys.argv[index + 1]
            if requested_platform in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", requested_platform)
            break
        if token.startswith("--platform="):
            requested_platform = token.split("=", 1)[1]
            if requested_platform in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", requested_platform)
            break

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.common import get_I_power2
from models.detrend import _prepare_power2_poly
from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    compute_transit_model,
)


Array = jax.Array
POLY_MUS, POLY_PROJECTION = _prepare_power2_poly()


@dataclass(frozen=True)
class Timing:
    compile_first_ms: float
    median_ms: float
    p05_ms: float
    p95_ms: float
    repeats: int


@dataclass(frozen=True)
class Case:
    n_times: int
    batch_size: int
    ld_profile: str
    period: float
    duration: float
    window_start: int
    window_end: int
    transit_kernel: str = "stock"

    @property
    def window_size(self) -> int:
        return self.window_end - self.window_start


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_hdu(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _one_dimensional_time_array(data, *, column: str | None = None) -> np.ndarray:
    """Extract a finite 1-D time vector from an array or FITS table."""
    names = getattr(getattr(data, "dtype", None), "names", None)
    if names:
        if column is None:
            time_names = [name for name in names if "time" in name.lower()]
            if len(time_names) != 1:
                raise ValueError(
                    "FITS table needs --time-column because it does not have "
                    "exactly one column containing 'time'"
                )
            column = time_names[0]
        matching = [name for name in names if name.lower() == column.lower()]
        if len(matching) != 1:
            raise ValueError(f"FITS table has no column named {column!r}")
        data = data[matching[0]]
    elif column is not None:
        raise ValueError("--time-column was supplied for a non-table time HDU")

    values = np.asarray(data, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size < 2:
        raise ValueError("time grid must be a one-dimensional array with >= 2 rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("time grid contains non-finite values")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("time grid must be strictly increasing")
    return values


def load_time_grid(
    path: str,
    *,
    hdu: int | str | None = None,
    column: str | None = None,
    attribute: str = "time",
    transit_t0: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Load and center real timestamps without adding a mandatory dependency.

    NPY and one-column text files use NumPy. FITS support imports Astropy only
    when requested. If no FITS HDU is supplied, an HDU whose name contains
    ``time`` must be uniquely identifiable. Trusted SpectroData pickle files
    expose either the spectroscopic ``.time`` or white-light ``.wl_time``
    attribute; ``attribute`` selects which one is benchmarked.
    """
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    selected_hdu: int | str | None = hdu
    selected_column = column
    selected_attribute: str | None = None
    if suffix == ".npy":
        raw = np.load(source, allow_pickle=False)
    elif suffix in {".txt", ".dat", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        raw = np.loadtxt(source, delimiter=delimiter)
    elif suffix in {".fits", ".fit", ".fts"}:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("Astropy is required to read --time-file FITS data") from exc
        with fits.open(source, memmap=True) as hdul:
            if hdu is None:
                candidates = [
                    index
                    for index, item in enumerate(hdul)
                    if "time" in (item.name or "").lower() and item.data is not None
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "FITS file needs --time-hdu because no unique HDU name "
                        "containing 'time' was found"
                    )
                selected_hdu = candidates[0]
            raw = np.array(hdul[selected_hdu].data, copy=True)
    elif suffix in {".pkl", ".pickle"}:
        # Pickle is intentionally limited to an explicit, user-supplied local
        # file. Loading can import createdatacube.SpectroData automatically;
        # do not use this option for untrusted files.
        with source.open("rb") as stream:
            container = pickle.load(stream)
        if attribute not in {"time", "wl_time"}:
            raise ValueError(
                "SpectroData --time-attribute must be 'time' or 'wl_time'"
            )
        if not hasattr(container, attribute):
            raise ValueError(
                f"pickle object has no {attribute!r} timestamp attribute"
            )
        raw = getattr(container, attribute)
        selected_attribute = attribute
    else:
        raise ValueError(
            "--time-file must be .npy, .txt, .dat, .csv, .fits, .fit, .fts, "
            ".pkl, or .pickle"
        )

    values = _one_dimensional_time_array(raw, column=selected_column)
    center = float(np.median(values) if transit_t0 is None else transit_t0)
    centered = values - center
    return centered, {
        "path": str(source.resolve()),
        "hdu": selected_hdu,
        "column": selected_column,
        "attribute": selected_attribute,
        "center": center,
        "center_source": "median" if transit_t0 is None else "--time-t0",
        "n_times": int(values.size),
        "centered_min": float(centered[0]),
        "centered_max": float(centered[-1]),
    }


def streamed_kernel_status() -> tuple[Callable | None, str | None]:
    """Return the optional streamed light-curve entry point and any skip reason.

    Keeping this import lazy lets the benchmark and its existing window tests
    run while the experimental module is absent or being developed on another
    branch.
    """
    module_name = "models.jaxoplanet.limb_dark_streamed"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        return None, f"{module_name} is unavailable: {exc}"
    light_curve = getattr(module, "limb_dark_light_curve", None)
    if light_curve is None:
        return None, f"{module_name} has no limb_dark_light_curve symbol"
    return light_curve, None


def _limb_darkening(theta: Array, ld_profile: str) -> Array:
    if ld_profile == "quadratic":
        return theta[3:5]
    if ld_profile == "power2":
        c, alpha = theta[3], theta[4]
        profile = get_I_power2(c, alpha, POLY_MUS)
        return POLY_PROJECTION @ (1.0 - profile)
    raise ValueError(f"unknown limb-darkening profile: {ld_profile}")


def _transit_signal(
    theta: Array,
    t: Array,
    duration: float,
    ld_profile: str,
    transit_kernel: str = "stock",
    period: float = 4.05528043,
) -> Array:
    params = {
        "period": jnp.asarray([period], dtype=jnp.float64),
        "duration": jnp.asarray([duration], dtype=jnp.float64),
        "t0": theta[0:1],
        "b": theta[1:2],
        "rors": theta[2:3],
        "u": _limb_darkening(theta, ld_profile),
    }
    if transit_kernel == "stock":
        return compute_transit_model(params, t)
    if transit_kernel != "streamed":
        raise ValueError(f"unknown transit kernel: {transit_kernel}")
    if ld_profile != "power2":
        raise ValueError("the streamed benchmark currently supports only power2 LD")

    streamed_light_curve, reason = streamed_kernel_status()
    if streamed_light_curve is None:
        raise RuntimeError(reason)
    from jaxoplanet.orbits.transit import TransitOrbit

    orbit = TransitOrbit(
        period=params["period"][0],
        duration=params["duration"][0],
        time_transit=params["t0"][0],
        impact_param=params["b"][0],
        radius_ratio=params["rors"][0],
    )
    return jnp.asarray(streamed_light_curve(orbit, params["u"])(t))


def make_case(
    n_times: int,
    batch_size: int,
    *,
    ld_profile: str = "power2",
    period: float = 4.05528043,
    duration: float = 0.11693087083333333,
    observation_half_span_durations: float = 4.0,
    window_half_width_durations: float = 0.75,
    transit_kernel: str = "stock",
    time_grid: Array | np.ndarray | None = None,
) -> tuple[
    Case,
    Array,
    Array,
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Construct matched full and static-window forward/objective functions."""
    if ld_profile not in {"power2", "quadratic"}:
        raise ValueError("ld_profile must be 'power2' or 'quadratic'")
    if window_half_width_durations < 0.5:
        raise ValueError("window must be at least half a transit duration")

    if time_grid is None:
        t = jnp.linspace(
            -observation_half_span_durations * duration,
            observation_half_span_durations * duration,
            n_times,
            dtype=jnp.float64,
        )
    else:
        t_host_input = _one_dimensional_time_array(time_grid)
        if t_host_input.size != n_times:
            raise ValueError(
                f"n_times={n_times} does not match time grid size {t_host_input.size}"
            )
        t = jnp.asarray(t_host_input, dtype=jnp.float64)
    t_host = np.asarray(t)
    half_width = window_half_width_durations * duration
    selected = np.flatnonzero(np.abs(t_host) <= half_width)
    if selected.size == 0:
        raise ValueError("static window contains no cadences")
    window_start = int(selected[0])
    window_end = int(selected[-1]) + 1
    t_window = t[window_start:window_end]

    # [t0, b, rprs, LD1, LD2, offset, slope, log_jitter].
    if ld_profile == "power2":
        ld1, ld2 = 0.55, 0.40
    else:
        ld1, ld2 = 0.30, 0.20
    base = np.array(
        [0.0, 0.45, 0.1457, ld1, ld2, 1.0, 2.0e-3, np.log(7.0e-4)],
        dtype=np.float64,
    )
    theta_host = np.repeat(base[None, :], batch_size, axis=0)
    if batch_size > 1:
        lane = np.linspace(-1.0, 1.0, batch_size)
        theta_host[:, 0] += lane * 0.02 * duration
        theta_host[:, 1] += lane * 0.02
        theta_host[:, 2] += lane * 0.002
    theta0 = jnp.asarray(theta_host)

    def single_full(theta: Array) -> Array:
        trend = theta[5] + theta[6] * (t - t[0])
        return _transit_signal(
            theta, t, duration, ld_profile, transit_kernel, period
        ) + trend

    def single_window(theta: Array) -> Array:
        trend = theta[5] + theta[6] * (t - t[0])
        transit = jnp.zeros_like(t)
        transit = transit.at[window_start:window_end].set(
            _transit_signal(
                theta, t_window, duration, ld_profile, transit_kernel, period
            )
        )
        return transit + trend

    forward_full = jax.vmap(single_full)
    forward_window = jax.vmap(single_window)

    truth = jax.lax.stop_gradient(forward_full(theta0))
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, n_times, dtype=jnp.float64)
    y = truth + 0.1 * jnp.exp(theta0[:, 7:8]) * jnp.sin(phase)[None, :]

    def objective(forward: Callable[[Array], Array], theta: Array) -> Array:
        model = forward(theta)
        sigma = jnp.exp(theta[:, 7:8])
        residual = (model - y) / sigma
        return 0.5 * jnp.sum(residual * residual) + n_times * jnp.sum(theta[:, 7])

    objective_full = lambda theta: objective(forward_full, theta)
    objective_window = lambda theta: objective(forward_window, theta)
    case = Case(
        n_times=n_times,
        batch_size=batch_size,
        ld_profile=ld_profile,
        period=period,
        duration=duration,
        window_start=window_start,
        window_end=window_end,
        transit_kernel=transit_kernel,
    )
    return (
        case,
        t,
        theta0,
        forward_full,
        forward_window,
        objective_full,
        objective_window,
    )


def make_production_comparison_case(
    n_times: int,
    batch_size: int,
    *,
    period: float = 4.05528043,
    duration: float = 0.11693087083333333,
    observation_half_span_durations: float = 4.0,
    window_half_width_durations: float = 0.75,
    time_grid: Array | np.ndarray | None = None,
) -> tuple[
    Case,
    Array,
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Build the three production power-2 paths used by spectroscopy.

    The reference path calls :func:`compute_transit_model` independently for
    every spectral channel, including the ``TransitOrbit`` phase calculation.
    The other paths reproduce ``create_vectorized_model``: phase offsets and
    the duration mask are calculated once from the shared geometry, outside
    the channel ``vmap``, then consumed by either the stock or streamed
    occultation kernel.  Keeping the stock shared-phase path separate makes it
    possible to attribute speedups to phase sharing and kernel streaming
    independently.

    In the compact benchmark parameter array, the shared ``t0`` and impact
    parameter live in the first row.  Radius ratio, limb darkening, trend, and
    noise remain channel-specific, as they are in the production model.
    """
    streamed_light_curve, reason = streamed_kernel_status()
    if streamed_light_curve is None:
        raise RuntimeError(reason)

    case, t, theta, _, _, _, _ = make_case(
        n_times,
        batch_size,
        ld_profile="power2",
        period=period,
        duration=duration,
        observation_half_span_durations=observation_half_span_durations,
        window_half_width_durations=window_half_width_durations,
        transit_kernel="stock",
        time_grid=time_grid,
    )
    # The window is deliberately static, matching fit_jwst's host-side window
    # construction.  The phase mask remains dynamic because it depends on the
    # shared transit center and is evaluated inside each HMC model call.
    window_indices = jnp.arange(
        case.window_start, case.window_end, dtype=jnp.int32
    )
    periods = jnp.asarray([period], dtype=jnp.float64)
    durations = jnp.asarray([duration], dtype=jnp.float64)

    def channel_params(
        value: Array,
        shared_t0: Array,
        shared_b: Array,
        *,
        phase_offsets: Array | None = None,
        phase_mask: Array | None = None,
    ) -> dict[str, Array | str]:
        params: dict[str, Array | str] = {
            "period": periods,
            "duration": durations,
            "t0": shared_t0[None],
            "b": shared_b[None],
            "rors": value[2:3],
            "u": _limb_darkening(value, "power2"),
            "_transit_window_indices": window_indices,
            "_ld_profile": "power2",
        }
        if phase_offsets is not None:
            params["_transit_phase_offsets"] = phase_offsets
            params["_transit_phase_mask"] = phase_mask
        return params

    def forward(value: Array, *, kernel: str, share_phase: bool) -> Array:
        shared_t0 = value[0, 0]
        shared_b = value[0, 1]
        if share_phase:
            phase_offsets, phase_mask = build_transit_phase_offsets(
                t, periods, shared_t0[None], durations
            )
        else:
            phase_offsets = phase_mask = None

        def single_channel(channel: Array) -> Array:
            params = channel_params(
                channel,
                shared_t0,
                shared_b,
                phase_offsets=phase_offsets,
                phase_mask=phase_mask,
            )
            transit = compute_transit_model(
                params, t, kernel=kernel, ld_profile="power2"
            )
            trend = channel[5] + channel[6] * (t - t[0])
            return transit + trend

        return jax.vmap(single_channel)(value)

    production_stock_forward = lambda value: forward(
        value, kernel="stock", share_phase=False
    )
    shared_stock_forward = lambda value: forward(
        value, kernel="stock", share_phase=True
    )
    shared_streamed_forward = lambda value: forward(
        value, kernel="streamed", share_phase=True
    )

    truth = jax.lax.stop_gradient(production_stock_forward(theta))
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, n_times, dtype=jnp.float64)
    y = truth + 0.1 * jnp.exp(theta[:, 7:8]) * jnp.sin(phase)[None, :]

    def objective(forward: Callable[[Array], Array], value: Array) -> Array:
        model = forward(value)
        sigma = jnp.exp(value[:, 7:8])
        residual = (model - y) / sigma
        return 0.5 * jnp.sum(residual * residual) + n_times * jnp.sum(value[:, 7])

    production_stock_objective = lambda value: objective(
        production_stock_forward, value
    )
    shared_stock_objective = lambda value: objective(shared_stock_forward, value)
    shared_streamed_objective = lambda value: objective(
        shared_streamed_forward, value
    )
    comparison_case = Case(
        n_times=case.n_times,
        batch_size=case.batch_size,
        ld_profile=case.ld_profile,
        period=case.period,
        duration=case.duration,
        window_start=case.window_start,
        window_end=case.window_end,
        transit_kernel="stock_vs_streamed",
    )
    return (
        comparison_case,
        theta,
        production_stock_forward,
        shared_stock_forward,
        shared_streamed_forward,
        production_stock_objective,
        shared_stock_objective,
        shared_streamed_objective,
    )


def make_streamed_comparison_case(
    n_times: int,
    batch_size: int,
    *,
    period: float = 4.05528043,
    duration: float = 0.11693087083333333,
    observation_half_span_durations: float = 4.0,
    window_half_width_durations: float = 0.75,
    time_grid: Array | np.ndarray | None = None,
) -> tuple[
    Case,
    Array,
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
    Callable[[Array], Array],
]:
    """Backward-compatible combined production-vs-streamed comparison."""
    (
        case,
        theta,
        production_stock_forward,
        _,
        shared_streamed_forward,
        production_stock_objective,
        _,
        shared_streamed_objective,
    ) = make_production_comparison_case(
        n_times,
        batch_size,
        period=period,
        duration=duration,
        observation_half_span_durations=observation_half_span_durations,
        window_half_width_durations=window_half_width_durations,
        time_grid=time_grid,
    )
    return (
        case,
        theta,
        production_stock_forward,
        shared_streamed_forward,
        production_stock_objective,
        shared_streamed_objective,
    )


def block_ready(value):
    return jax.block_until_ready(value)


def timed_jit(
    fn: Callable[[Array], Array],
    argument: Array,
    *,
    warmup: int,
    repeats: int,
) -> tuple[Timing, Callable[[Array], Array]]:
    compiled = jax.jit(fn)
    start = time.perf_counter()
    block_ready(compiled(argument))
    compile_first_ms = 1.0e3 * (time.perf_counter() - start)

    for _ in range(warmup):
        block_ready(compiled(argument))

    elapsed_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_ready(compiled(argument))
        elapsed_ms.append(1.0e3 * (time.perf_counter() - start))

    elapsed_ms.sort()
    timing = Timing(
        compile_first_ms=compile_first_ms,
        median_ms=float(statistics.median(elapsed_ms)),
        p05_ms=float(np.percentile(elapsed_ms, 5)),
        p95_ms=float(np.percentile(elapsed_ms, 95)),
        repeats=repeats,
    )
    return timing, compiled


def fidelity_metrics(
    theta: Array,
    full_forward: Callable[[Array], Array],
    window_forward: Callable[[Array], Array],
    full_objective: Callable[[Array], Array],
    window_objective: Callable[[Array], Array],
) -> dict[str, float]:
    full_flux = block_ready(jax.jit(full_forward)(theta))
    window_flux = block_ready(jax.jit(window_forward)(theta))
    full_value, full_grad = block_ready(jax.jit(jax.value_and_grad(full_objective))(theta))
    window_value, window_grad = block_ready(
        jax.jit(jax.value_and_grad(window_objective))(theta)
    )

    flux_diff = np.asarray(full_flux - window_flux)
    grad_diff = np.asarray(full_grad - window_grad)
    grad_ref = np.asarray(full_grad)
    return {
        "flux_max_abs": float(np.max(np.abs(flux_diff))),
        "objective_abs": float(np.abs(np.asarray(full_value - window_value))),
        "objective_abs_per_datum": float(
            np.abs(np.asarray(full_value - window_value)) / full_flux.size
        ),
        "gradient_max_abs": float(np.max(np.abs(grad_diff))),
        "gradient_relative_l2": float(
            np.linalg.norm(grad_diff) / max(np.linalg.norm(grad_ref), 1.0e-300)
        ),
        "all_finite": bool(
            np.all(np.isfinite(np.asarray(full_flux)))
            and np.all(np.isfinite(np.asarray(window_flux)))
            and np.isfinite(np.asarray(full_value)).all()
            and np.isfinite(np.asarray(window_value)).all()
            and np.all(np.isfinite(np.asarray(full_grad)))
            and np.all(np.isfinite(np.asarray(window_grad)))
        ),
    }


def transit_gradient_metrics(
    theta: Array,
    stock_forward: Callable[[Array], Array],
    streamed_forward: Callable[[Array], Array],
) -> dict[str, float | bool]:
    """Compare the transit-model VJP without likelihood cancellation.

    A deterministic unit cotangent exercises every cadence and every fitted
    parameter.  This is more stable than comparing a Gaussian objective whose
    offset gradient can be nearly zero through residual cancellation.
    """
    stock_flux = block_ready(jax.jit(stock_forward)(theta))
    cotangent = jnp.reshape(
        jnp.linspace(
            0.7,
            1.3,
            stock_flux.size,
            dtype=stock_flux.dtype,
        ),
        stock_flux.shape,
    )

    def projection(forward: Callable[[Array], Array], value: Array) -> Array:
        return jnp.sum(forward(value) * cotangent)

    stock_grad = block_ready(
        jax.jit(jax.grad(lambda value: projection(stock_forward, value)))(theta)
    )
    streamed_grad = block_ready(
        jax.jit(jax.grad(lambda value: projection(streamed_forward, value)))(theta)
    )
    # Trend and jitter derivatives are identical and often much larger, so
    # exclude them: they would hide discrepancies in the five transit/LD
    # derivatives that the benchmark is intended to validate.
    grad_diff = np.asarray(streamed_grad - stock_grad)[..., :5]
    grad_ref = np.asarray(stock_grad)[..., :5]
    lane_relative = np.linalg.norm(grad_diff, axis=-1) / np.maximum(
        np.linalg.norm(grad_ref, axis=-1), 1.0e-300
    )
    return {
        "transit_gradient_max_abs": float(np.max(np.abs(grad_diff))),
        "transit_gradient_relative_l2": float(
            np.linalg.norm(grad_diff) / max(np.linalg.norm(grad_ref), 1.0e-300)
        ),
        "transit_gradient_max_lane_relative_l2": float(np.max(lane_relative)),
        "transit_gradient_all_finite": bool(
            np.all(np.isfinite(np.asarray(stock_grad)))
            and np.all(np.isfinite(np.asarray(streamed_grad)))
        ),
    }


def streamed_stress_metrics() -> dict:
    """Compare stock and streamed kernels at difficult physical boundaries."""
    streamed_light_curve, reason = streamed_kernel_status()
    if streamed_light_curve is None:
        raise RuntimeError(reason)

    duration = 0.11693087083333333
    contact = 0.5 * duration
    contact_points = np.array(
        [
            -contact,
            np.nextafter(-contact, -np.inf),
            np.nextafter(-contact, np.inf),
            contact,
            np.nextafter(contact, -np.inf),
            np.nextafter(contact, np.inf),
        ],
        dtype=np.float64,
    )
    dense = np.linspace(-0.75 * duration, 0.75 * duration, 257)
    times = jnp.asarray(np.unique(np.concatenate((dense, contact_points))))
    names = (
        "nominal_contacts",
        "grazing_contacts",
        "power2_low_c_alpha",
        "power2_high_c_alpha",
    )
    # [t0, b, rprs, power2 c, power2 alpha, offset, slope, log_jitter].
    theta = jnp.asarray(
        [
            [0.0, 0.45, 0.1457, 0.55, 0.40, 1.0, 0.0, np.log(7.0e-4)],
            [0.0, 1.05, 0.1200, 0.55, 0.40, 1.0, 0.0, np.log(7.0e-4)],
            [0.0, 0.45, 0.1457, 1.0e-8, 0.001, 1.0, 0.0, np.log(7.0e-4)],
            [0.0, 0.45, 0.1457, 1.0 - 1.0e-8, 1.0, 1.0, 0.0, np.log(7.0e-4)],
        ],
        dtype=jnp.float64,
    )
    weights = jnp.linspace(0.7, 1.3, times.size, dtype=jnp.float64)

    def single_forward(value: Array, kernel: str) -> Array:
        return _transit_signal(value, times, duration, "power2", kernel)

    stock_forward = jax.vmap(lambda value: single_forward(value, "stock"))
    streamed_forward = jax.vmap(lambda value: single_forward(value, "streamed"))
    stock_flux = block_ready(jax.jit(stock_forward)(theta))
    streamed_flux = block_ready(jax.jit(streamed_forward)(theta))

    def single_loss(value: Array, kernel: str) -> Array:
        return jnp.sum(weights * single_forward(value, kernel))

    stock_grad_fn = jax.vmap(jax.grad(lambda value: single_loss(value, "stock")))
    streamed_grad_fn = jax.vmap(
        jax.grad(lambda value: single_loss(value, "streamed"))
    )
    stock_grad = block_ready(jax.jit(stock_grad_fn)(theta))
    streamed_grad = block_ready(jax.jit(streamed_grad_fn)(theta))

    stock_flux_host = np.asarray(stock_flux)
    streamed_flux_host = np.asarray(streamed_flux)
    stock_grad_host = np.asarray(stock_grad)
    streamed_grad_host = np.asarray(streamed_grad)
    cases = []
    for index, name in enumerate(names):
        flux_diff = streamed_flux_host[index] - stock_flux_host[index]
        grad_diff = streamed_grad_host[index] - stock_grad_host[index]
        cases.append(
            {
                "name": name,
                "flux_max_abs": float(np.max(np.abs(flux_diff))),
                "gradient_max_abs": float(np.max(np.abs(grad_diff))),
                "gradient_relative_l2": float(
                    np.linalg.norm(grad_diff)
                    / max(np.linalg.norm(stock_grad_host[index]), 1.0e-300)
                ),
                "stock_finite": bool(
                    np.all(np.isfinite(stock_flux_host[index]))
                    and np.all(np.isfinite(stock_grad_host[index]))
                ),
                "streamed_finite": bool(
                    np.all(np.isfinite(streamed_flux_host[index]))
                    and np.all(np.isfinite(streamed_grad_host[index]))
                ),
            }
        )
    return {
        "n_times": int(times.size),
        "contact_time": contact,
        "cases": cases,
        "flux_max_abs": max(case["flux_max_abs"] for case in cases),
        "gradient_max_abs": max(case["gradient_max_abs"] for case in cases),
        "gradient_relative_l2": max(
            case["gradient_relative_l2"] for case in cases
        ),
        "all_finite": all(
            case["stock_finite"] and case["streamed_finite"] for case in cases
        ),
    }


def assert_streamed_fidelity(
    metrics: dict,
    *,
    flux_tolerance: float,
    gradient_tolerance: float,
    objective_tolerance: float | None = None,
) -> None:
    if not metrics.get("all_finite", True) or not metrics.get(
        "transit_gradient_all_finite", True
    ):
        raise RuntimeError("streamed comparison produced non-finite flux or gradients")
    if metrics["flux_max_abs"] > flux_tolerance:
        raise RuntimeError(
            "streamed flux fidelity gate failed: "
            f"{metrics['flux_max_abs']:.3e} > {flux_tolerance:.3e}"
        )
    gradient_metric = metrics.get(
        "transit_gradient_max_lane_relative_l2",
        metrics.get("transit_gradient_relative_l2", metrics["gradient_relative_l2"]),
    )
    if gradient_metric > gradient_tolerance:
        raise RuntimeError(
            "streamed gradient fidelity gate failed: "
            f"{gradient_metric:.3e} > {gradient_tolerance:.3e}"
        )
    if (
        objective_tolerance is not None
        and metrics.get(
            "objective_abs_per_datum", metrics.get("objective_abs", 0.0)
        )
        > objective_tolerance
    ):
        raise RuntimeError(
            "streamed objective fidelity gate failed: "
            f"{metrics.get('objective_abs_per_datum', metrics['objective_abs']):.3e} "
            f"> {objective_tolerance:.3e} per datum"
        )


def device_memory() -> dict[str, int]:
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except (AttributeError, RuntimeError):
        return {}
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    return {key: int(stats[key]) for key in keys if key in stats}


def _speedup(reference: Timing, candidate: Timing) -> float:
    return reference.median_ms / candidate.median_ms


def run_case(
    n_times: int,
    batch_size: int,
    *,
    ld_profile: str,
    warmup: int,
    repeats: int,
    window_half_width_durations: float,
    flux_tolerance: float,
    objective_tolerance: float,
    gradient_tolerance: float,
    time_grid: Array | np.ndarray | None = None,
    period: float = 4.05528043,
    duration: float = 0.11693087083333333,
) -> dict:
    case, _, theta, full_fwd, window_fwd, full_obj, window_obj = make_case(
        n_times,
        batch_size,
        ld_profile=ld_profile,
        period=period,
        duration=duration,
        window_half_width_durations=window_half_width_durations,
        time_grid=time_grid,
    )
    # Time before the parity helper JITs these callables, so compile+first is
    # genuinely cold for each benchmark kernel.
    full_forward_timing, _ = timed_jit(
        full_fwd, theta, warmup=warmup, repeats=repeats
    )
    window_forward_timing, _ = timed_jit(
        window_fwd, theta, warmup=warmup, repeats=repeats
    )
    full_vg_timing, _ = timed_jit(
        jax.value_and_grad(full_obj), theta, warmup=warmup, repeats=repeats
    )
    window_vg_timing, _ = timed_jit(
        jax.value_and_grad(window_obj), theta, warmup=warmup, repeats=repeats
    )

    parity = fidelity_metrics(theta, full_fwd, window_fwd, full_obj, window_obj)
    if (
        not parity["all_finite"]
        or parity["flux_max_abs"] > flux_tolerance
        or parity["objective_abs"] > objective_tolerance
        or parity["gradient_relative_l2"] > gradient_tolerance
    ):
        raise RuntimeError(
            "static window failed fidelity gate: "
            f"flux={parity['flux_max_abs']:.3e}, "
            f"objective={parity['objective_abs']:.3e}, "
            f"gradient_rel_l2={parity['gradient_relative_l2']:.3e}"
        )

    return {
        "case": asdict(case) | {"window_size": case.window_size},
        "fidelity": parity,
        "timings": {
            "full_forward": asdict(full_forward_timing),
            "window_forward": asdict(window_forward_timing),
            "full_value_grad": asdict(full_vg_timing),
            "window_value_grad": asdict(window_vg_timing),
        },
        "speedup": {
            "forward": _speedup(full_forward_timing, window_forward_timing),
            "value_grad": _speedup(full_vg_timing, window_vg_timing),
        },
        "device_memory": device_memory(),
    }


def run_streamed_case(
    n_times: int,
    batch_size: int,
    *,
    warmup: int,
    repeats: int,
    window_half_width_durations: float,
    flux_tolerance: float,
    objective_tolerance: float,
    gradient_tolerance: float,
    time_grid: Array | np.ndarray | None = None,
    period: float = 4.05528043,
    duration: float = 0.11693087083333333,
    shared_phase_flux_tolerance: float = 5.0e-13,
    shared_phase_objective_tolerance: float = 1.0e-8,
    shared_phase_gradient_tolerance: float = 1.0e-8,
) -> dict:
    """Benchmark phase sharing and streaming in the production power-2 path."""
    (
        case,
        theta,
        production_stock_forward,
        shared_stock_forward,
        shared_streamed_forward,
        production_stock_objective,
        shared_stock_objective,
        shared_streamed_objective,
    ) = make_production_comparison_case(
        n_times,
        batch_size,
        period=period,
        duration=duration,
        window_half_width_durations=window_half_width_durations,
        time_grid=time_grid,
    )
    production_stock_forward_timing, _ = timed_jit(
        production_stock_forward, theta, warmup=warmup, repeats=repeats
    )
    shared_stock_forward_timing, _ = timed_jit(
        shared_stock_forward, theta, warmup=warmup, repeats=repeats
    )
    shared_streamed_forward_timing, _ = timed_jit(
        shared_streamed_forward, theta, warmup=warmup, repeats=repeats
    )
    production_stock_vg_timing, _ = timed_jit(
        jax.value_and_grad(production_stock_objective),
        theta,
        warmup=warmup,
        repeats=repeats,
    )
    shared_stock_vg_timing, _ = timed_jit(
        jax.value_and_grad(shared_stock_objective),
        theta,
        warmup=warmup,
        repeats=repeats,
    )
    shared_streamed_vg_timing, _ = timed_jit(
        jax.value_and_grad(shared_streamed_objective),
        theta,
        warmup=warmup,
        repeats=repeats,
    )
    shared_phase_parity = fidelity_metrics(
        theta,
        production_stock_forward,
        shared_stock_forward,
        production_stock_objective,
        shared_stock_objective,
    )
    shared_phase_parity.update(
        transit_gradient_metrics(
            theta, production_stock_forward, shared_stock_forward
        )
    )
    if (
        not shared_phase_parity["all_finite"]
        or shared_phase_parity["flux_max_abs"] > shared_phase_flux_tolerance
        or shared_phase_parity["objective_abs"]
        > shared_phase_objective_tolerance
        or shared_phase_parity["transit_gradient_max_lane_relative_l2"]
        > shared_phase_gradient_tolerance
    ):
        raise RuntimeError(
            "production shared-phase fidelity gate failed: "
            f"flux={shared_phase_parity['flux_max_abs']:.3e}, "
            "objective="
            f"{shared_phase_parity['objective_abs']:.3e}, "
            "gradient_lane_rel_l2="
            f"{shared_phase_parity['transit_gradient_max_lane_relative_l2']:.3e}"
        )

    parity = fidelity_metrics(
        theta,
        shared_stock_forward,
        shared_streamed_forward,
        shared_stock_objective,
        shared_streamed_objective,
    )
    parity.update(
        transit_gradient_metrics(
            theta, shared_stock_forward, shared_streamed_forward
        )
    )
    assert_streamed_fidelity(
        parity,
        flux_tolerance=flux_tolerance,
        gradient_tolerance=gradient_tolerance,
        objective_tolerance=objective_tolerance,
    )
    return {
        "case": asdict(case) | {"window_size": case.window_size},
        "fidelity": parity,
        "shared_phase_fidelity": shared_phase_parity,
        "timings": {
            "production_stock_forward": asdict(production_stock_forward_timing),
            "shared_stock_forward": asdict(shared_stock_forward_timing),
            "shared_streamed_forward": asdict(shared_streamed_forward_timing),
            "production_stock_value_grad": asdict(production_stock_vg_timing),
            "shared_stock_value_grad": asdict(shared_stock_vg_timing),
            "shared_streamed_value_grad": asdict(shared_streamed_vg_timing),
        },
        "speedup": {
            "shared_phase_forward": _speedup(
                production_stock_forward_timing, shared_stock_forward_timing
            ),
            "streamed_kernel_forward": _speedup(
                shared_stock_forward_timing, shared_streamed_forward_timing
            ),
            "combined_forward": _speedup(
                production_stock_forward_timing, shared_streamed_forward_timing
            ),
            "shared_phase_value_grad": _speedup(
                production_stock_vg_timing, shared_stock_vg_timing
            ),
            "streamed_kernel_value_grad": _speedup(
                shared_stock_vg_timing, shared_streamed_vg_timing
            ),
            "combined_value_grad": _speedup(
                production_stock_vg_timing, shared_streamed_vg_timing
            ),
            # Compatibility aliases: these are the combined production-stock
            # to shared-phase-streamed gains reported by earlier JSON readers.
            "forward": _speedup(
                production_stock_forward_timing, shared_streamed_forward_timing
            ),
            "value_grad": _speedup(
                production_stock_vg_timing, shared_streamed_vg_timing
            ),
        },
        "device_memory": device_memory(),
    }


def _print_result(result: dict) -> None:
    case = result["case"]
    print(
        f"\n{case['ld_profile']} n_times={case['n_times']} "
        f"batch={case['batch_size']} window={case['window_size']}/{case['n_times']}"
    )
    print(
        "  parity "
        f"flux={result['fidelity']['flux_max_abs']:.3e} "
        f"objective={result['fidelity']['objective_abs']:.3e} "
        f"grad={result['fidelity']['gradient_max_abs']:.3e}"
    )
    for name, timing in result["timings"].items():
        print(
            f"  {name:20s} median={timing['median_ms']:9.4f} ms "
            f"p05={timing['p05_ms']:9.4f} p95={timing['p95_ms']:9.4f} "
            f"compile+first={timing['compile_first_ms']:9.2f} ms"
        )
    print(
        f"  speedup forward={result['speedup']['forward']:.3f}x "
        f"value+grad={result['speedup']['value_grad']:.3f}x"
    )


def _print_streamed_stress(result: dict) -> None:
    print("\nstock vs streamed power-2 stress parity")
    for case in result["cases"]:
        print(
            f"  {case['name']:24s} flux={case['flux_max_abs']:.3e} "
            f"grad={case['gradient_max_abs']:.3e} "
            f"grad_rel_l2={case['gradient_relative_l2']:.3e} "
            f"finite={case['stock_finite'] and case['streamed_finite']}"
        )


def _print_streamed_result(result: dict) -> None:
    case = result["case"]
    print(
        f"\nproduction vs shared-phase power2 n_times={case['n_times']} "
        f"batch={case['batch_size']} window={case['window_size']}/{case['n_times']}"
    )
    print(
        "  stock shared-phase parity "
        f"flux={result['shared_phase_fidelity']['flux_max_abs']:.3e} "
        f"objective={result['shared_phase_fidelity']['objective_abs']:.3e} "
        f"grad={result['shared_phase_fidelity']['gradient_max_abs']:.3e}"
    )
    print(
        "  stock vs streamed parity   "
        f"flux={result['fidelity']['flux_max_abs']:.3e} "
        f"objective={result['fidelity']['objective_abs']:.3e} "
        f"grad={result['fidelity']['gradient_max_abs']:.3e}"
    )
    for name, timing in result["timings"].items():
        print(
            f"  {name:22s} median={timing['median_ms']:9.4f} ms "
            f"p05={timing['p05_ms']:9.4f} p95={timing['p95_ms']:9.4f} "
            f"compile+first={timing['compile_first_ms']:9.2f} ms"
        )
    print(
        "  speedup phase-sharing "
        f"forward={result['speedup']['shared_phase_forward']:.3f}x "
        f"value+grad={result['speedup']['shared_phase_value_grad']:.3f}x"
    )
    print(
        "  speedup streamed-kernel "
        f"forward={result['speedup']['streamed_kernel_forward']:.3f}x "
        f"value+grad={result['speedup']['streamed_kernel_value_grad']:.3f}x"
    )
    print(
        "  speedup combined        "
        f"forward={result['speedup']['combined_forward']:.3f}x "
        f"value+grad={result['speedup']['combined_value_grad']:.3f}x"
    )


def _is_resource_exhausted(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "resource_exhausted",
        "out of memory",
        "cuda_error_out_of_memory",
        "failed to allocate",
    )
    return any(marker in message for marker in markers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadences", type=parse_int_list, default=[254, 537, 1458])
    parser.add_argument(
        "--batches", type=parse_int_list, default=[1, 8, 40, 60, 80, 100]
    )
    parser.add_argument(
        "--ld-profile", choices=("power2", "quadratic", "both"), default="power2"
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("all", "window", "production"),
        default="all",
        help=(
            "'window' measures the legacy full-vs-window benchmark; "
            "'production' measures stock-wrapper vs shared-stock vs "
            "shared-streamed; 'all' runs both (default)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--period",
        type=float,
        default=4.05528043,
        help="Orbital period in the same time units as the timestamp grid",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.11693087083333333,
        help="Transit duration in the same time units as the timestamp grid",
    )
    parser.add_argument(
        "--window-half-width-durations",
        type=float,
        default=0.5,
        help=(
            "Half-width of the benchmark transit window in fitted-duration "
            "units (default: 0.5, matching the production TransitOrbit mask)"
        ),
    )
    parser.add_argument("--flux-tolerance", type=float, default=5.0e-13)
    parser.add_argument("--objective-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--streamed",
        choices=("auto", "off", "require"),
        default="auto",
        help=(
            "Benchmark the optional streamed power-2 kernel when available; "
            "'require' fails instead of skipping when it is absent."
        ),
    )
    parser.add_argument("--streamed-flux-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--streamed-objective-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--streamed-gradient-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--time-file",
        help=(
            "Optional real timestamp grid (.npy, text/CSV, FITS, or a trusted "
            "SpectroData pickle). This replaces --cadences and is centered "
            "on --time-t0 or its median."
        ),
    )
    parser.add_argument(
        "--time-hdu",
        type=_parse_hdu,
        help="FITS extension index or name containing the timestamp array",
    )
    parser.add_argument(
        "--time-column", help="Timestamp column name when the FITS HDU is a table"
    )
    parser.add_argument(
        "--time-attribute",
        choices=("time", "wl_time"),
        default="time",
        help=(
            "SpectroData pickle timestamp attribute: 'time' for spectroscopy "
            "or 'wl_time' for the white-light grid (default: time)"
        ),
    )
    parser.add_argument(
        "--time-t0",
        type=float,
        help="Transit center in the same units as --time-file (default: median)",
    )
    parser.add_argument("--json", dest="json_path")
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="GPU is recommended; CPU is useful for smoke tests.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("--warmup must be >= 0 and --repeats must be >= 1")
    if not np.isfinite(args.period) or args.period <= 0.0:
        raise SystemExit("--period must be finite and positive")
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        raise SystemExit("--duration must be finite and positive")
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)

    print(
        f"JAX {jax.__version__}; backend={jax.default_backend()}; "
        f"device={jax.devices()[0]}; x64={jax.config.read('jax_enable_x64')}"
    )
    profiles = ("power2", "quadratic") if args.ld_profile == "both" else (args.ld_profile,)
    time_metadata = None
    if args.time_file:
        time_grid, time_metadata = load_time_grid(
            args.time_file,
            hdu=args.time_hdu,
            column=args.time_column,
            attribute=args.time_attribute,
            transit_t0=args.time_t0,
        )
        cadence_grids = ((int(time_grid.size), time_grid),)
        print(
            "Using real timestamps: "
            f"n={time_grid.size}, center={time_metadata['center']:.12g} "
            f"({time_metadata['center_source']})"
        )
    else:
        cadence_grids = tuple((n_times, None) for n_times in args.cadences)
    results = []
    window_profiles = () if args.benchmark_mode == "production" else profiles
    for profile in window_profiles:
        for n_times, time_grid in cadence_grids:
            for batch_size in args.batches:
                try:
                    result = run_case(
                        n_times,
                        batch_size,
                        ld_profile=profile,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        window_half_width_durations=args.window_half_width_durations,
                        flux_tolerance=args.flux_tolerance,
                        objective_tolerance=args.objective_tolerance,
                        gradient_tolerance=args.gradient_tolerance,
                        time_grid=time_grid,
                        period=args.period,
                        duration=args.duration,
                    )
                except Exception as exc:
                    if not _is_resource_exhausted(exc):
                        raise
                    result = {
                        "case": {
                            "n_times": n_times,
                            "batch_size": batch_size,
                            "ld_profile": profile,
                        },
                        "status": "oom",
                        "error": str(exc),
                    }
                    print(
                        f"\n{profile} n_times={n_times} batch={batch_size}: OOM"
                    )
                    jax.clear_caches()
                results.append(result)
                if result.get("status") != "oom":
                    _print_result(result)

    streamed_entry, streamed_reason = streamed_kernel_status()
    streamed_payload = {
        "benchmark_mode": args.benchmark_mode,
        "requested": args.streamed,
        "available": streamed_entry is not None,
        "skip_reason": streamed_reason,
        "stress": None,
        "results": [],
    }
    if args.benchmark_mode == "window":
        streamed_payload["skip_reason"] = (
            "production comparison disabled by --benchmark-mode window"
        )
    elif args.streamed == "require" and streamed_entry is None:
        raise SystemExit(streamed_reason)
    elif args.streamed != "off" and streamed_entry is None:
        print(f"\nSkipping streamed comparison: {streamed_reason}")
    elif args.streamed != "off" and "power2" not in profiles:
        streamed_payload["skip_reason"] = (
            "streamed comparison requires --ld-profile power2 or both"
        )
        print(f"\nSkipping streamed comparison: {streamed_payload['skip_reason']}")
    elif args.streamed != "off":
        stress = streamed_stress_metrics()
        assert_streamed_fidelity(
            stress,
            flux_tolerance=args.streamed_flux_tolerance,
            gradient_tolerance=args.streamed_gradient_tolerance,
        )
        streamed_payload["stress"] = stress
        _print_streamed_stress(stress)
        for n_times, time_grid in cadence_grids:
            for batch_size in args.batches:
                try:
                    result = run_streamed_case(
                        n_times,
                        batch_size,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        window_half_width_durations=args.window_half_width_durations,
                        flux_tolerance=args.streamed_flux_tolerance,
                        objective_tolerance=args.streamed_objective_tolerance,
                        gradient_tolerance=args.streamed_gradient_tolerance,
                        time_grid=time_grid,
                        period=args.period,
                        duration=args.duration,
                        shared_phase_flux_tolerance=args.flux_tolerance,
                        shared_phase_objective_tolerance=args.objective_tolerance,
                        shared_phase_gradient_tolerance=args.gradient_tolerance,
                    )
                except Exception as exc:
                    if not _is_resource_exhausted(exc):
                        raise
                    result = {
                        "case": {"n_times": n_times, "batch_size": batch_size},
                        "status": "oom",
                        "error": str(exc),
                    }
                    print(
                        "\nstock vs streamed power2 "
                        f"n_times={n_times} batch={batch_size}: OOM"
                    )
                    jax.clear_caches()
                streamed_payload["results"].append(result)
                if result.get("status") != "oom":
                    _print_streamed_result(result)

    if args.json_path:
        payload = {
            "environment": {
                "jax_version": jax.__version__,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "x64": bool(jax.config.read("jax_enable_x64")),
                "pid": os.getpid(),
                "benchmark_mode": args.benchmark_mode,
                "time_grid": time_metadata,
            },
            "results": results,
            "streamed": streamed_payload,
        }
        with open(args.json_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

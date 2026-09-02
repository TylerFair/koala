#!/usr/bin/env python3
"""End-to-end audit of the opt-in native Power-2 spectroscopic fit path.

The accuracy case uses real HAT-P-65 cadences, fluxes, uncertainties and the
fixed white-light geometry.  It compares the order-16 native kernel with an
order-256 direct reference, and separately characterizes the scientific-model
change from the existing degree-12 projection.  Timing cases compile and run
the actual NumPyro ``explinear`` spectroscopic potential at one resident width.

The tool never changes fitter defaults and never runs NUTS.  Run timing kernels
in separate processes because an OOM in one XLA executable should not hide the
result from another.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import pickle
import resource
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common import get_I_power2
from models.detrend import _prepare_power2_poly
from models.jaxoplanet import build_transit_window_indices, create_vectorized_model
from models.jaxoplanet.core import build_transit_phase_offsets
from models.jaxoplanet.experimental_power2_native import light_curve as native_light_curve
from models.jaxoplanet.limb_dark_streamed import light_curve as streamed_light_curve


POWER2_MUS, POWER2_PROJECTION = _prepare_power2_poly(degree=12, n_mu=300)


DEFAULT_DATA = (
    "ASYMM_HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/"
    "HAT-P-65_NIRSPEC_PRISM_nrs1_spectroscopy_data_10LR_50HR.pkl"
)
DEFAULT_WHITE = (
    "ASYMM_HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/"
    "HAT-P-65_NIRSPEC_PRISM_nrs1_whitelight_bestfit_params.csv"
)
DEFAULT_MASK = (
    "ASYMM_HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/"
    "HAT-P-65_NIRSPEC_PRISM_nrs1_whitelight_outlier_mask.npy"
)


class _SpectroPayload:
    """Import-light stand-in for ``createdatacube.SpectroData``."""


class _PayloadUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "createdatacube" and name == "SpectroData":
            return _SpectroPayload
        return super().find_class(module, name)


def _ready(value):
    jax.block_until_ready(value)


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _write_report(path, report):
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


def _load_white_row(path):
    with Path(path).open(newline="") as stream:
        row = next(csv.DictReader(stream))

    def value(name, default=None):
        raw = row.get(name)
        if raw not in (None, ""):
            parsed = float(raw)
            if np.isfinite(parsed):
                return parsed
        if default is None:
            raise ValueError(f"White-light CSV has no finite {name!r} value")
        return float(default)

    return {
        "period": value("period"),
        "duration": value("duration"),
        "t0": value("t0"),
        "b": value("b"),
        "rors": value("rors"),
        "ld_c": value("u1", 0.3),
        "ld_alpha": value("u2", 0.5),
        "trend_c": value("c", 1.0),
        "trend_v": value("v", 0.0),
        "trend_A": value("A", 0.0),
        "trend_tau": value("tau", 0.05),
    }


def _load_problem_arrays(data_path, mask_path, width, max_cadences=0):
    with Path(data_path).open("rb") as stream:
        payload = _PayloadUnpickler(stream).load()
    time_values = np.asarray(payload.time, dtype=np.float64)
    flux = np.asarray(payload.flux_hr, dtype=np.float64)
    flux_err = np.asarray(payload.flux_err_hr, dtype=np.float64)
    wavelengths = np.asarray(payload.wavelengths_hr, dtype=np.float64)
    outlier = np.asarray(np.load(mask_path), dtype=bool)
    if outlier.shape == time_values.shape:
        keep = ~outlier
        time_values = time_values[keep]
        flux = flux[:, keep]
        flux_err = flux_err[:, keep]
    if width > flux.shape[0]:
        raise ValueError(f"width {width} exceeds {flux.shape[0]} high-resolution channels")

    # A centered contiguous block has the same resident layout as one fitter
    # chunk while avoiding the noisier detector edges.
    start = (flux.shape[0] - width) // 2
    channel_indices = np.arange(start, start + width, dtype=np.int64)
    flux = flux[channel_indices]
    flux_err = flux_err[channel_indices]
    wavelengths = wavelengths[channel_indices]
    source_cadences = int(time_values.size)
    if max_cadences and max_cadences < time_values.size:
        cadence_indices = np.linspace(
            0, time_values.size - 1, max_cadences, dtype=np.int64
        )
        time_values = time_values[cadence_indices]
        flux = flux[:, cadence_indices]
        flux_err = flux_err[:, cadence_indices]
    return {
        "time": time_values,
        "flux": flux,
        "flux_err": flux_err,
        "wavelengths": wavelengths,
        "channel_indices": channel_indices,
        "source_channels": int(np.asarray(payload.flux_hr).shape[0]),
        "source_cadences": source_cadences,
    }


def _nominal_physical_parameters(arrays, white):
    wavelength = arrays["wavelengths"]
    if np.ptp(wavelength) > 0:
        coordinate = (wavelength - np.mean(wavelength)) / np.ptp(wavelength)
    else:
        coordinate = np.zeros_like(wavelength)
    radius = white["rors"] * (1.0 + 0.01 * coordinate)
    c_value = np.clip(white["ld_c"] + 0.04 * coordinate, 0.01, 0.99)
    alpha = np.clip(white["ld_alpha"] + 0.04 * coordinate, 0.002, 0.99)
    median_error = np.median(arrays["flux_err"], axis=1)
    jitter = np.clip(0.25 * median_error, 1.1e-6, 0.2)
    width = wavelength.size
    return np.column_stack(
        [
            radius,
            c_value,
            alpha,
            np.log(jitter),
            np.full(width, np.clip(white["trend_c"], 0.901, 1.099)),
            np.full(width, np.clip(white["trend_v"], -0.099, 0.099)),
            np.full(width, np.clip(white["trend_A"], -0.099, 0.099)),
            np.full(
                width,
                np.log(np.clip(white["trend_tau"], 1.001e-3, 0.0999)),
            ),
        ]
    )


def _native_flux(theta, time_values, white, order):
    radius, c_value, alpha = theta[:, 0], theta[:, 1], theta[:, 2]
    offsets, mask = build_transit_phase_offsets(
        time_values,
        jnp.asarray([white["period"]]),
        jnp.asarray([white["t0"]]),
        jnp.asarray([white["duration"]]),
    )
    speed = (
        2.0
        * jnp.sqrt(jnp.maximum(0.0, (1.0 + radius) ** 2 - white["b"] ** 2))
        / white["duration"]
    )
    separation = jnp.sqrt(
        (speed[:, None] * offsets[0][None, :]) ** 2 + white["b"] ** 2
    )
    result = native_light_curve(
        c_value[:, None],
        alpha[:, None],
        separation,
        radius[:, None],
        order=order,
    )
    return jnp.where(mask[0][None, :], result, 0.0)


def _degree12_flux(theta, time_values, white):
    radius, c_value, alpha = theta[:, 0], theta[:, 1], theta[:, 2]
    offsets, mask = build_transit_phase_offsets(
        time_values,
        jnp.asarray([white["period"]]),
        jnp.asarray([white["t0"]]),
        jnp.asarray([white["duration"]]),
    )
    speed = (
        2.0
        * jnp.sqrt(jnp.maximum(0.0, (1.0 + radius) ** 2 - white["b"] ** 2))
        / white["duration"]
    )
    separation = jnp.sqrt(
        (speed[:, None] * offsets[0][None, :]) ** 2 + white["b"] ** 2
    )
    profile = get_I_power2(
        c_value[:, None], alpha[:, None], POWER2_MUS[None, :]
    )
    coefficients = (POWER2_PROJECTION @ (1.0 - profile).T).T
    result = jax.vmap(
        lambda u, z, r: streamed_light_curve(u, z, r, order=10)
    )(coefficients, separation, radius)
    return jnp.where(mask[0][None, :], result, 0.0)


def _negative_log_likelihood(theta, time_values, flux, flux_err, white, flux_fn):
    signal = flux_fn(theta, time_values, white)
    shifted = time_values - jnp.min(time_values)
    trend = (
        theta[:, 4, None]
        + theta[:, 5, None] * shifted[None, :]
        + theta[:, 6, None]
        * jnp.exp(-shifted[None, :] / jnp.exp(theta[:, 7, None]))
    )
    scale = jnp.sqrt(jnp.exp(2.0 * theta[:, 3, None]) + flux_err**2)
    standardized = (flux - (signal + trend)) / scale
    return jnp.sum(
        0.5 * standardized**2 + jnp.log(scale) + 0.5 * math.log(2.0 * math.pi)
    )


def _tree_difference(left, right):
    left_flat, _ = ravel_pytree(left)
    right_flat, _ = ravel_pytree(right)
    difference = np.asarray(left_flat - right_flat)
    right_values = np.asarray(right_flat)
    return {
        "size": int(difference.size),
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(right_values), 1.0e-300)
        ),
    }


def _model_inputs(arrays, white):
    theta = _nominal_physical_parameters(arrays, white)
    width = theta.shape[0]
    time_values = jnp.asarray(arrays["time"], dtype=jnp.float64)
    yerr = jnp.asarray(arrays["flux_err"], dtype=jnp.float64)
    y = jnp.asarray(arrays["flux"], dtype=jnp.float64)
    ld_mean = jnp.asarray(theta[:, 1:3], dtype=jnp.float64)
    init = {
        "rors": jnp.asarray(theta[:, 0, None]),
        "c1": ld_mean[:, 0],
        "c2": ld_mean[:, 1],
        "log_jitter": jnp.asarray(theta[:, 3]),
        "c": jnp.asarray(theta[:, 4]),
        "v": jnp.asarray(theta[:, 5]),
        "A": jnp.asarray(theta[:, 6]),
        "log_tau": jnp.asarray(theta[:, 7]),
    }
    transit_indices = build_transit_window_indices(
        arrays["time"],
        np.asarray([white["period"]]),
        np.asarray([white["t0"]]),
        np.asarray([white["duration"]]),
    )
    kwargs = {
        "y": y,
        "mu_duration": jnp.asarray([white["duration"]]),
        "mu_t0": jnp.asarray([white["t0"]]),
        "mu_b": jnp.asarray([white["b"]]),
        "mu_depths": jnp.full((width, 1), white["rors"] ** 2),
        "PERIOD": jnp.asarray([white["period"]]),
        "mu_u_ld": ld_mean,
        # A conservative nominal scale; performance is shape-dependent, not
        # sensitive to the exact stellar-grid uncertainty value.
        "sigma_u_ld": jnp.full_like(ld_mean, 0.03),
        "precomputed_yerr_per_lc": jnp.nanmedian(yerr, axis=1),
    }
    return time_values, yerr, init, kwargs, transit_indices, theta


def _initialize_potential(kernel, arrays, white):
    time_values, yerr, init, kwargs, transit_indices, theta = _model_inputs(
        arrays, white
    )
    model = create_vectorized_model(
        detrend_type="explinear",
        ld_mode="informed",
        trend_mode="free",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=transit_indices,
        jaxoplanet_kernel=kernel,
    )
    info = initialize_model(
        jax.random.PRNGKey(9817),
        model,
        model_args=(time_values, yerr),
        model_kwargs=kwargs,
        init_strategy=init_to_value(values=init),
    )
    return info.potential_fn, info.param_info.z, theta, int(transit_indices.size)


def _timing_summary(function, argument, repeats):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    _ready(compiled(argument))
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compiled(argument)
        _ready(result)
        elapsed.append(time.perf_counter() - start)
    return {
        "compile_seconds": float(compile_seconds),
        "median_ms": float(np.median(elapsed) * 1.0e3),
        "min_ms": float(np.min(elapsed) * 1.0e3),
        "p95_ms": float(np.percentile(elapsed, 95.0) * 1.0e3),
    }


def run_accuracy(args):
    arrays = _load_problem_arrays(
        args.data_pickle, args.outlier_mask, args.accuracy_width, args.accuracy_cadences
    )
    white = _load_white_row(args.white_csv)
    theta = jnp.asarray(_nominal_physical_parameters(arrays, white))
    times = jnp.asarray(arrays["time"])
    flux = jnp.asarray(arrays["flux"])
    flux_err = jnp.asarray(arrays["flux_err"])

    native16 = jax.jit(lambda values: _native_flux(values, times, white, 16))(theta)
    native256 = jax.jit(lambda values: _native_flux(values, times, white, 256))(theta)
    degree12 = jax.jit(lambda values: _degree12_flux(values, times, white))(theta)
    _ready((native16, native256, degree12))

    objective16 = lambda values: _negative_log_likelihood(
        values,
        times,
        flux,
        flux_err,
        white,
        lambda v, t, w: _native_flux(v, t, w, 16),
    )
    objective256 = lambda values: _negative_log_likelihood(
        values,
        times,
        flux,
        flux_err,
        white,
        lambda v, t, w: _native_flux(v, t, w, 256),
    )
    objective12 = lambda values: _negative_log_likelihood(
        values, times, flux, flux_err, white, _degree12_flux
    )
    value_gradient16 = jax.jit(jax.value_and_grad(objective16))(theta)
    value_gradient256 = jax.jit(jax.value_and_grad(objective256))(theta)
    value_gradient12 = jax.jit(jax.value_and_grad(objective12))(theta)
    _ready((value_gradient16, value_gradient256, value_gradient12))

    native_potential, native_z, _, active = _initialize_potential(
        "native_power2", arrays, white
    )
    stock_potential, stock_z, _, stock_active = _initialize_potential(
        "streamed", arrays, white
    )
    if jax.tree.structure(native_z) != jax.tree.structure(stock_z):
        raise RuntimeError("Native and streamed potentials have different latent trees")
    native_integrated = jax.jit(jax.value_and_grad(native_potential))(native_z)
    stock_integrated = jax.jit(jax.value_and_grad(stock_potential))(native_z)
    _ready((native_integrated, stock_integrated))

    error16 = np.asarray(native16 - native256)
    model_delta = np.asarray(native16 - degree12)
    vg16_value, vg16_gradient = value_gradient16
    vg256_value, vg256_gradient = value_gradient256
    vg12_value, vg12_gradient = value_gradient12
    integrated_native_value, integrated_native_gradient = native_integrated
    integrated_stock_value, integrated_stock_gradient = stock_integrated
    report = {
        "case": "accuracy",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "production_defaults_changed": False,
        "data": {
            "source": str(args.data_pickle),
            "source_channels": arrays["source_channels"],
            "source_cadences_after_white_mask": arrays["source_cadences"],
            "channels": int(theta.shape[0]),
            "cadences": int(theta.shape[0] * 0 + times.size),
            "channel_indices": arrays["channel_indices"].tolist(),
            "first_time": float(times[0]),
            "last_time": float(times[-1]),
        },
        "fixed_white_geometry": {
            key: white[key] for key in ("period", "duration", "t0", "b", "rors")
        },
        "geometry_source_note": (
            "Existing white-light best-fit CSV; this legacy artifact predates the "
            "new max-likelihood-draw handoff and must not be labelled as that estimator."
        ),
        "native_order16_vs_order256": {
            "max_abs_flux_error_ppm": float(np.max(np.abs(error16)) * 1.0e6),
            "rms_flux_error_ppm": float(np.sqrt(np.mean(error16**2)) * 1.0e6),
            "negative_log_likelihood_delta": float(vg16_value - vg256_value),
            "gradient": _tree_difference(vg16_gradient, vg256_gradient),
        },
        "native_order16_vs_degree12_projection": {
            "max_abs_flux_difference_ppm": float(np.max(np.abs(model_delta)) * 1.0e6),
            "p99_abs_flux_difference_ppm": float(
                np.percentile(np.abs(model_delta), 99.0) * 1.0e6
            ),
            "rms_flux_difference_ppm": float(np.sqrt(np.mean(model_delta**2)) * 1.0e6),
            "negative_log_likelihood_delta": float(vg16_value - vg12_value),
            "gradient": _tree_difference(vg16_gradient, vg12_gradient),
        },
        "integrated_numpyro_native_vs_streamed": {
            "native_potential": float(integrated_native_value),
            "streamed_potential_at_same_unconstrained_point": float(
                integrated_stock_value
            ),
            "potential_delta": float(integrated_native_value - integrated_stock_value),
            "gradient": _tree_difference(
                integrated_native_gradient, integrated_stock_gradient
            ),
            "active_transit_cadences": active,
            "streamed_active_transit_cadences": stock_active,
        },
    }
    return report


def run_timing(args):
    arrays = _load_problem_arrays(
        args.data_pickle, args.outlier_mask, args.width, args.max_cadences
    )
    white = _load_white_row(args.white_csv)
    potential, z, _, active = _initialize_potential(args.kernel, arrays, white)
    potential_timing = _timing_summary(potential, z, args.repeats)
    value_gradient_timing = _timing_summary(
        jax.value_and_grad(potential), z, args.repeats
    )
    device = jax.devices()[0]
    try:
        memory = device.memory_stats() or {}
    except Exception:
        memory = {}
    return {
        "case": "timing",
        "kernel": args.kernel,
        "backend": jax.default_backend(),
        "device": str(device),
        "production_defaults_changed": False,
        "width": args.width,
        "cadences": int(arrays["time"].size),
        "source_cadences_after_white_mask": arrays["source_cadences"],
        "active_transit_cadences": active,
        "channel_indices": arrays["channel_indices"].tolist(),
        "potential": potential_timing,
        "value_and_gradient": value_gradient_timing,
        "memory": {
            key: int(memory[key])
            for key in (
                "peak_bytes_in_use",
                "bytes_in_use",
                "bytes_limit",
            )
            if key in memory
        },
        "process_max_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "fixed_white_geometry": {
            key: white[key] for key in ("period", "duration", "t0", "b", "rors")
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("accuracy", "timing"), required=True)
    parser.add_argument(
        "--kernel", choices=("native_power2", "streamed", "fused")
    )
    parser.add_argument("--width", type=int, choices=(40, 80), default=40)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-cadences", type=int, default=0)
    parser.add_argument("--accuracy-width", type=int, default=8)
    parser.add_argument("--accuracy-cadences", type=int, default=513)
    parser.add_argument("--data-pickle", default=DEFAULT_DATA)
    parser.add_argument("--white-csv", default=DEFAULT_WHITE)
    parser.add_argument("--outlier-mask", default=DEFAULT_MASK)
    parser.add_argument("--json-output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.case == "timing" and args.kernel is None:
        raise ValueError("--kernel is required for a timing case")
    if args.repeats < 1 or args.accuracy_cadences < 16 or args.accuracy_width < 1:
        raise ValueError("repeats/accuracy sizes are too small")
    started = time.perf_counter()
    report = run_accuracy(args) if args.case == "accuracy" else run_timing(args)
    report["wall_seconds"] = time.perf_counter() - started
    _write_report(args.json_output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

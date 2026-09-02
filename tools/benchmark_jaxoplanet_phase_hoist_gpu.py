#!/usr/bin/env python3
"""Benchmark hoisting fixed spectroscopic transit phases out of HMC evaluations.

The production spectroscopic model fits channel-specific radius ratio, limb
darkening, trend, and noise while taking period, transit center, impact
parameter, and duration from the white-light fit.  With ``jit_model_args=True``
those fixed geometry values are nevertheless dynamic arguments to the compiled
NumPyro transition.  This benchmark reproduces that distinction:

* ``phase_inside`` computes the shared phase offsets and duration mask from
  dynamic, fixed geometry on every forward-model evaluation.
* ``phase_hoisted`` closes over offsets and mask computed once before JIT.

Both phase-placement routes use the same static transit window and
differentiate only with respect to channel-specific fitted parameters. Real
SpectroData pickle timestamp grids can be selected with ``--preset hatp12`` or
``--preset hatp65``. The streamed selection calls the production
``models.jaxoplanet.core`` route; ``--kernel fused`` substitutes the
experimental algebraically contracted evaluator at the same precomputed
separation, while ``--kernel both`` reports fused-versus-streamed fidelity and
speedups in one run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, Iterable

import jax

jax.config.update("jax_enable_x64", True)
# Select the backend before any imported module can initialize JAX.
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
    build_transit_window_indices,
    compute_transit_model,
)
from models.jaxoplanet.limb_dark_fused import light_curve as fused_light_curve
from tools.benchmark_jaxoplanet_gpu import (
    Timing,
    _is_resource_exhausted,
    device_memory,
    load_time_grid,
    parse_int_list,
)


Array = jax.Array
POLY_MUS, POLY_PROJECTION = _prepare_power2_poly()

PRESETS = {
    "hatp12": {
        "path": (
            "/scratch/midway3/tfairnington/"
            "ASYMM_HAT-P-12_SOSS_ORDER1_FIXEDLD_QUADRATIC_SPOT/"
            "HAT-P-12_NIRISS_SOSS_order1_spectroscopy_data_20LR_50HR.pkl"
        ),
        "t0": 60115.45,
        "period": 3.21305762,
        "duration": 0.090375,
    },
    "hatp65": {
        "path": (
            "/scratch/midway3/tfairnington/STELLARINFORMED/"
            "HAT-P-65_PRISM_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/"
            "HAT-P-65_NIRSPEC_PRISM_nrs1_spectroscopy_data_20LR_nativeHR.pkl"
        ),
        "t0": 60470.72,
        "period": 2.60544751,
        "duration": 0.17688375,
    },
}


def _block(value):
    return jax.block_until_ready(value)


def _timed_jit(
    fn: Callable[[Array, dict[str, Array]], Array],
    theta: Array,
    runtime: dict[str, Array],
    *,
    warmup: int,
    repeats: int,
) -> tuple[Timing, Callable]:
    """Compile and synchronously time a two-argument JAX callable."""
    compiled = jax.jit(fn)
    start = time.perf_counter()
    _block(compiled(theta, runtime))
    compile_first_ms = 1.0e3 * (time.perf_counter() - start)

    for _ in range(warmup):
        _block(compiled(theta, runtime))

    elapsed_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block(compiled(theta, runtime))
        elapsed_ms.append(1.0e3 * (time.perf_counter() - start))
    elapsed_ms.sort()
    return (
        Timing(
            compile_first_ms=compile_first_ms,
            median_ms=float(statistics.median(elapsed_ms)),
            p05_ms=float(np.percentile(elapsed_ms, 5)),
            p95_ms=float(np.percentile(elapsed_ms, 95)),
            repeats=repeats,
        ),
        compiled,
    )


def _power2_coefficients(theta: Array) -> Array:
    profile = get_I_power2(theta[1], theta[2], POLY_MUS)
    return POLY_PROJECTION @ (1.0 - profile)


def make_phase_hoist_case(
    time_grid: np.ndarray,
    batch_size: int,
    *,
    period: float,
    duration: float,
    impact: float,
    kernel: str = "streamed",
) -> tuple[
    Array,
    dict[str, Array],
    np.ndarray,
    Callable[[Array, dict[str, Array]], Array],
    Callable[[Array, dict[str, Array]], Array],
    Callable[[Array, dict[str, Array]], Array],
    Callable[[Array, dict[str, Array]], Array],
]:
    """Construct fixed-geometry inside-phase and hoisted-phase objectives."""
    if kernel not in {"streamed", "fused"}:
        raise ValueError("kernel must be 'streamed' or 'fused'")
    times = np.asarray(time_grid, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time_grid must be one-dimensional with >=2 cadences")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError("time_grid must be finite and strictly increasing")

    t = jnp.asarray(times, dtype=jnp.float64)
    runtime = {
        "t": t,
        "period": jnp.asarray([period], dtype=jnp.float64),
        # Real grids are centered on the fixed white-light transit center.
        "t0": jnp.asarray([0.0], dtype=jnp.float64),
        "duration": jnp.asarray([duration], dtype=jnp.float64),
        "b": jnp.asarray([impact], dtype=jnp.float64),
    }
    window_indices_host = build_transit_window_indices(
        times,
        np.asarray([period]),
        np.asarray([0.0]),
        np.asarray([duration]),
    )
    if window_indices_host.size == 0:
        raise ValueError("the fixed transit window contains no cadences")
    window_indices = jnp.asarray(window_indices_host, dtype=jnp.int32)

    hoisted_offsets, hoisted_mask = build_transit_phase_offsets(
        runtime["t"], runtime["period"], runtime["t0"], runtime["duration"]
    )
    hoisted_offsets, hoisted_mask = _block((hoisted_offsets, hoisted_mask))

    # [rprs, power2_c, power2_alpha, offset, slope, log_jitter].
    base = np.asarray(
        [0.1457, 0.55, 0.40, 1.0, 2.0e-3, np.log(7.0e-4)],
        dtype=np.float64,
    )
    theta_host = np.repeat(base[None, :], batch_size, axis=0)
    if batch_size > 1:
        lane = np.linspace(-1.0, 1.0, batch_size)
        theta_host[:, 0] += 0.002 * lane
        theta_host[:, 1] += 0.02 * lane
        theta_host[:, 2] += 0.02 * lane
    theta0 = jnp.asarray(theta_host, dtype=jnp.float64)

    def forward(
        theta: Array,
        dynamic: dict[str, Array],
        *,
        hoisted: bool,
        selected_kernel: str,
    ) -> Array:
        if hoisted:
            phase_offsets, phase_mask = hoisted_offsets, hoisted_mask
        else:
            # These operands are dynamic JIT inputs, matching NumPyro's
            # jit_model_args=True potential rather than compile-time constants.
            phase_offsets, phase_mask = build_transit_phase_offsets(
                dynamic["t"],
                dynamic["period"],
                dynamic["t0"],
                dynamic["duration"],
            )

        def single_channel(channel: Array) -> Array:
            params = {
                "period": dynamic["period"],
                "duration": dynamic["duration"],
                "t0": dynamic["t0"],
                "b": dynamic["b"],
                "rors": channel[0:1],
                "u": _power2_coefficients(channel),
                "_transit_window_indices": window_indices,
                "_transit_phase_offsets": phase_offsets,
                "_transit_phase_mask": phase_mask,
                "_ld_profile": "power2",
            }
            if selected_kernel == "streamed":
                transit = compute_transit_model(
                    params,
                    dynamic["t"],
                    kernel="streamed",
                    ld_profile="power2",
                )
            else:
                # This is the precomputed-phase duration branch in
                # models.jaxoplanet.core, with only its low-level evaluator
                # replaced because compute_transit_model does not route the
                # experimental fused kernel.
                active_offsets = phase_offsets[:, window_indices]
                active_mask = phase_mask[:, window_indices]

                def get_lc_from_phase(
                    channel_duration: Array,
                    channel_b: Array,
                    channel_rors: Array,
                    dt: Array,
                    mask: Array,
                ) -> Array:
                    speed = 2 * jnp.sqrt(
                        jnp.maximum(
                            0,
                            jnp.square(1 + channel_rors)
                            - jnp.square(channel_b),
                        )
                    ) / channel_duration
                    separation = jnp.sqrt(
                        jnp.square(speed * dt) + jnp.square(channel_b)
                    )
                    lc = fused_light_curve(
                        params["u"], separation, channel_rors, order=10
                    )
                    return jnp.where(mask, lc, 0)

                active_lcs = jax.vmap(get_lc_from_phase)(
                    dynamic["duration"],
                    dynamic["b"],
                    params["rors"],
                    active_offsets,
                    active_mask,
                )
                active_transit = jnp.sum(active_lcs, axis=0)
                transit = jnp.zeros_like(
                    dynamic["t"], dtype=active_transit.dtype
                ).at[window_indices].set(active_transit)
            trend = channel[3] + channel[4] * (
                dynamic["t"] - dynamic["t"][0]
            )
            return transit + trend

        return jax.vmap(single_channel)(theta)

    forward_inside = lambda theta, dynamic: forward(
        theta, dynamic, hoisted=False, selected_kernel=kernel
    )
    forward_hoisted = lambda theta, dynamic: forward(
        theta, dynamic, hoisted=True, selected_kernel=kernel
    )

    # Use the same streamed synthetic observation for either benchmark kernel,
    # making the likelihood objective directly comparable in --kernel both.
    truth = _block(
        forward(
            theta0,
            runtime,
            hoisted=True,
            selected_kernel="streamed",
        )
    )
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, times.size, dtype=jnp.float64)
    y = jax.lax.stop_gradient(
        truth + 0.1 * jnp.exp(theta0[:, 5:6]) * jnp.sin(phase)[None, :]
    )

    def objective(
        theta: Array,
        dynamic: dict[str, Array],
        forward_fn: Callable[[Array, dict[str, Array]], Array],
    ) -> Array:
        model = forward_fn(theta, dynamic)
        sigma = jnp.exp(theta[:, 5:6])
        residual = (model - y) / sigma
        return 0.5 * jnp.sum(residual * residual) + times.size * jnp.sum(
            theta[:, 5]
        )

    objective_inside = lambda theta, dynamic: objective(
        theta, dynamic, forward_inside
    )
    objective_hoisted = lambda theta, dynamic: objective(
        theta, dynamic, forward_hoisted
    )
    return (
        theta0,
        runtime,
        window_indices_host,
        forward_inside,
        forward_hoisted,
        objective_inside,
        objective_hoisted,
    )


def fidelity_metrics(
    theta: Array,
    runtime: dict[str, Array],
    forward_inside: Callable,
    forward_hoisted: Callable,
    objective_inside: Callable,
    objective_hoisted: Callable,
    *,
    compiled_forward_inside: Callable | None = None,
    compiled_forward_hoisted: Callable | None = None,
    compiled_value_grad_inside: Callable | None = None,
    compiled_value_grad_hoisted: Callable | None = None,
) -> dict[str, float | bool]:
    compiled_forward_inside = compiled_forward_inside or jax.jit(forward_inside)
    compiled_forward_hoisted = compiled_forward_hoisted or jax.jit(forward_hoisted)
    compiled_value_grad_inside = compiled_value_grad_inside or jax.jit(
        jax.value_and_grad(objective_inside, argnums=0)
    )
    compiled_value_grad_hoisted = compiled_value_grad_hoisted or jax.jit(
        jax.value_and_grad(objective_hoisted, argnums=0)
    )
    inside_flux = _block(compiled_forward_inside(theta, runtime))
    hoisted_flux = _block(compiled_forward_hoisted(theta, runtime))
    inside_value, inside_grad = _block(
        compiled_value_grad_inside(theta, runtime)
    )
    hoisted_value, hoisted_grad = _block(
        compiled_value_grad_hoisted(theta, runtime)
    )

    flux_diff = np.asarray(inside_flux - hoisted_flux)
    grad_diff = np.asarray(inside_grad - hoisted_grad)
    grad_ref = np.asarray(inside_grad)
    # Use a deterministic transit-only VJP for the mathematical gradient gate.
    # A Gaussian likelihood can amplify sub-1e-10 flux rounding through its
    # small sigma, even when the actual rprs/LD Jacobian is unchanged.
    weights = jnp.reshape(
        jnp.linspace(0.7, 1.3, inside_flux.size, dtype=inside_flux.dtype),
        inside_flux.shape,
    )
    inside_transit_grad = _block(
        jax.jit(
            jax.grad(
                lambda value, dynamic: jnp.sum(
                    forward_inside(value, dynamic) * weights
                ),
                argnums=0,
            )
        )(theta, runtime)
    )
    hoisted_transit_grad = _block(
        jax.jit(
            jax.grad(
                lambda value, dynamic: jnp.sum(
                    forward_hoisted(value, dynamic) * weights
                ),
                argnums=0,
            )
        )(theta, runtime)
    )
    # Only rprs, c, and alpha affect the transit. Trend and log-jitter are
    # excluded so identical nuisance derivatives cannot hide a discrepancy.
    transit_grad_diff = np.asarray(
        inside_transit_grad - hoisted_transit_grad
    )[..., :3]
    transit_grad_ref = np.asarray(inside_transit_grad)[..., :3]
    return {
        "flux_max_abs": float(np.max(np.abs(flux_diff))),
        "objective_abs": float(
            np.abs(np.asarray(inside_value - hoisted_value))
        ),
        "objective_abs_per_datum": float(
            np.abs(np.asarray(inside_value - hoisted_value)) / inside_flux.size
        ),
        "gradient_max_abs": float(np.max(np.abs(grad_diff))),
        "gradient_relative_l2": float(
            np.linalg.norm(grad_diff) / max(np.linalg.norm(grad_ref), 1.0e-300)
        ),
        "transit_gradient_max_abs": float(
            np.max(np.abs(transit_grad_diff))
        ),
        "transit_gradient_relative_l2": float(
            np.linalg.norm(transit_grad_diff)
            / max(np.linalg.norm(transit_grad_ref), 1.0e-300)
        ),
        "all_finite": bool(
            np.all(np.isfinite(np.asarray(inside_flux)))
            and np.all(np.isfinite(np.asarray(hoisted_flux)))
            and np.isfinite(np.asarray(inside_value)).all()
            and np.isfinite(np.asarray(hoisted_value)).all()
            and np.all(np.isfinite(np.asarray(inside_grad)))
            and np.all(np.isfinite(np.asarray(hoisted_grad)))
            and np.all(np.isfinite(np.asarray(inside_transit_grad)))
            and np.all(np.isfinite(np.asarray(hoisted_transit_grad)))
        ),
    }


def run_case(
    time_grid: np.ndarray,
    batch_size: int,
    *,
    period: float,
    duration: float,
    impact: float,
    kernel: str,
    warmup: int,
    repeats: int,
    _return_artifacts: bool = False,
) -> dict | tuple[dict, dict]:
    (
        theta,
        runtime,
        window_indices,
        forward_inside,
        forward_hoisted,
        objective_inside,
        objective_hoisted,
    ) = make_phase_hoist_case(
        time_grid,
        batch_size,
        period=period,
        duration=duration,
        impact=impact,
        kernel=kernel,
    )

    inside_forward_timing, compiled_forward_inside = _timed_jit(
        forward_inside, theta, runtime, warmup=warmup, repeats=repeats
    )
    hoisted_forward_timing, compiled_forward_hoisted = _timed_jit(
        forward_hoisted, theta, runtime, warmup=warmup, repeats=repeats
    )
    inside_vg_timing, compiled_value_grad_inside = _timed_jit(
        jax.value_and_grad(objective_inside, argnums=0),
        theta,
        runtime,
        warmup=warmup,
        repeats=repeats,
    )
    hoisted_vg_timing, compiled_value_grad_hoisted = _timed_jit(
        jax.value_and_grad(objective_hoisted, argnums=0),
        theta,
        runtime,
        warmup=warmup,
        repeats=repeats,
    )
    fidelity = fidelity_metrics(
        theta,
        runtime,
        forward_inside,
        forward_hoisted,
        objective_inside,
        objective_hoisted,
        compiled_forward_inside=compiled_forward_inside,
        compiled_forward_hoisted=compiled_forward_hoisted,
        compiled_value_grad_inside=compiled_value_grad_inside,
        compiled_value_grad_hoisted=compiled_value_grad_hoisted,
    )
    fidelity_pass = not (
        not fidelity["all_finite"]
        or fidelity["flux_max_abs"] > 1.0e-10
        or fidelity["objective_abs_per_datum"] > 1.0e-8
        or fidelity["transit_gradient_relative_l2"] > 1.0e-8
    )

    result = {
        "case": {
            "n_times": int(len(time_grid)),
            "window_size": int(window_indices.size),
            "batch_size": int(batch_size),
            "period": float(period),
            "duration": float(duration),
            "impact": float(impact),
            "kernel": kernel,
        },
        "fidelity": fidelity,
        "fidelity_pass": fidelity_pass,
        "timings": {
            "phase_inside_forward": asdict(inside_forward_timing),
            "phase_hoisted_forward": asdict(hoisted_forward_timing),
            "phase_inside_value_grad": asdict(inside_vg_timing),
            "phase_hoisted_value_grad": asdict(hoisted_vg_timing),
        },
        "speedup": {
            "forward": (
                inside_forward_timing.median_ms / hoisted_forward_timing.median_ms
            ),
            "value_grad": (
                inside_vg_timing.median_ms / hoisted_vg_timing.median_ms
            ),
        },
        "device_memory": device_memory(),
    }
    if not _return_artifacts:
        return result
    return result, {
        "theta": theta,
        "runtime": runtime,
        "forward_hoisted": forward_hoisted,
        "objective_hoisted": objective_hoisted,
        "compiled_forward_hoisted": compiled_forward_hoisted,
        "compiled_value_grad_hoisted": compiled_value_grad_hoisted,
    }


def _print_result(result: dict) -> None:
    case = result["case"]
    print(
        f"\nkernel={case['kernel']} n_times={case['n_times']} "
        f"batch={case['batch_size']} "
        f"window={case['window_size']}/{case['n_times']}"
    )
    fidelity = result["fidelity"]
    print(
        f"  parity {'PASS' if result['fidelity_pass'] else 'FAIL'} "
        f"flux={fidelity['flux_max_abs']:.3e} "
        f"objective/datum={fidelity['objective_abs_per_datum']:.3e} "
        f"transit_grad_rel_l2={fidelity['transit_gradient_relative_l2']:.3e}"
    )
    for name, timing in result["timings"].items():
        print(
            f"  {name:26s} median={timing['median_ms']:9.4f} ms "
            f"p05={timing['p05_ms']:9.4f} p95={timing['p95_ms']:9.4f} "
            f"compile+first={timing['compile_first_ms']:9.2f} ms"
        )
    print(
        f"  hoist speedup forward={result['speedup']['forward']:.4f}x "
        f"value+grad={result['speedup']['value_grad']:.4f}x"
    )


def compare_kernels(
    streamed_result: dict,
    streamed_artifacts: dict,
    fused_result: dict,
    fused_artifacts: dict,
) -> dict:
    """Compare fused against streamed using their shared hoisted-phase path."""
    theta = streamed_artifacts["theta"]
    runtime = streamed_artifacts["runtime"]
    fidelity = fidelity_metrics(
        theta,
        runtime,
        streamed_artifacts["forward_hoisted"],
        fused_artifacts["forward_hoisted"],
        streamed_artifacts["objective_hoisted"],
        fused_artifacts["objective_hoisted"],
        compiled_forward_inside=streamed_artifacts[
            "compiled_forward_hoisted"
        ],
        compiled_forward_hoisted=fused_artifacts[
            "compiled_forward_hoisted"
        ],
        compiled_value_grad_inside=streamed_artifacts[
            "compiled_value_grad_hoisted"
        ],
        compiled_value_grad_hoisted=fused_artifacts[
            "compiled_value_grad_hoisted"
        ],
    )
    fidelity_pass = not (
        not fidelity["all_finite"]
        or fidelity["flux_max_abs"] > 1.0e-10
        or fidelity["objective_abs_per_datum"] > 1.0e-8
        or fidelity["transit_gradient_relative_l2"] > 1.0e-8
    )

    timing_pairs = {
        "phase_inside_forward": "phase_inside_forward",
        "phase_hoisted_forward": "phase_hoisted_forward",
        "phase_inside_value_grad": "phase_inside_value_grad",
        "phase_hoisted_value_grad": "phase_hoisted_value_grad",
    }
    speedup = {
        name: (
            streamed_result["timings"][streamed_name]["median_ms"]
            / fused_result["timings"][streamed_name]["median_ms"]
        )
        for name, streamed_name in timing_pairs.items()
    }
    return {
        "case": {
            "n_times": streamed_result["case"]["n_times"],
            "window_size": streamed_result["case"]["window_size"],
            "batch_size": streamed_result["case"]["batch_size"],
        },
        "reference": "streamed",
        "candidate": "fused",
        "phase_path": "hoisted",
        "fidelity": fidelity,
        "fidelity_pass": fidelity_pass,
        "fused_speedup_over_streamed": speedup,
    }


def _print_kernel_comparison(result: dict) -> None:
    case = result["case"]
    fidelity = result["fidelity"]
    print(
        "  fused vs streamed "
        f"{'PASS' if result['fidelity_pass'] else 'FAIL'} "
        f"flux={fidelity['flux_max_abs']:.3e} "
        f"objective/datum={fidelity['objective_abs_per_datum']:.3e} "
        f"transit_grad_rel_l2={fidelity['transit_gradient_relative_l2']:.3e}"
    )
    speedup = result["fused_speedup_over_streamed"]
    print(
        f"  fused speedup n_times={case['n_times']} batch={case['batch_size']} "
        f"inside-forward={speedup['phase_inside_forward']:.4f}x "
        f"hoisted-forward={speedup['phase_hoisted_forward']:.4f}x "
        f"inside-valuegrad={speedup['phase_inside_value_grad']:.4f}x "
        f"hoisted-valuegrad={speedup['phase_hoisted_value_grad']:.4f}x"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--time-file")
    parser.add_argument(
        "--time-attribute", choices=("time", "wl_time"), default="time"
    )
    parser.add_argument("--time-t0", type=float)
    parser.add_argument("--period", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--impact", type=float, default=0.45)
    parser.add_argument(
        "--kernel",
        choices=("streamed", "fused", "both"),
        default="streamed",
        help=(
            "occultation contraction to benchmark; 'both' additionally "
            "reports fused-vs-streamed parity and speedups"
        ),
    )
    parser.add_argument("--cadences", type=int, default=1458)
    parser.add_argument(
        "--batches", type=parse_int_list, default=[1, 8, 40, 60, 80, 100]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument(
        "--require-fidelity",
        action="store_true",
        help="exit nonzero after writing results if any parity gate fails",
    )
    parser.add_argument(
        "--platform", choices=("auto", "cpu", "gpu"), default="auto"
    )
    return parser


def _resolve_inputs(args) -> tuple[np.ndarray, float, float, dict | None]:
    preset = PRESETS.get(args.preset, {})
    path = args.time_file or preset.get("path")
    t0 = args.time_t0 if args.time_t0 is not None else preset.get("t0")
    period = args.period if args.period is not None else preset.get("period")
    duration = (
        args.duration if args.duration is not None else preset.get("duration")
    )
    if period is None:
        period = 4.05528043
    if duration is None:
        duration = 0.11693087083333333
    if not np.isfinite(period) or period <= 0.0:
        raise SystemExit("period must be finite and positive")
    if not np.isfinite(duration) or duration <= 0.0:
        raise SystemExit("duration must be finite and positive")

    metadata = None
    if path is not None:
        if not Path(path).expanduser().exists():
            raise SystemExit(f"timestamp file does not exist: {path}")
        time_grid, metadata = load_time_grid(
            path,
            attribute=args.time_attribute,
            transit_t0=t0,
        )
    else:
        if args.cadences < 2:
            raise SystemExit("--cadences must be >= 2")
        time_grid = np.linspace(
            -4.0 * duration,
            4.0 * duration,
            args.cadences,
            dtype=np.float64,
        )
    return time_grid, float(period), float(duration), metadata


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preset and args.time_file:
        raise SystemExit("use either --preset or --time-file, not both")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("--warmup must be >=0 and --repeats must be >=1")
    if not np.isfinite(args.impact) or args.impact < 0.0:
        raise SystemExit("--impact must be finite and non-negative")
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)

    time_grid, period, duration, time_metadata = _resolve_inputs(args)
    print(
        f"JAX {jax.__version__}; backend={jax.default_backend()}; "
        f"device={jax.devices()[0]}; x64={jax.config.read('jax_enable_x64')}"
    )
    if time_metadata is not None:
        print(
            f"timestamps={time_metadata['path']} "
            f"attribute={args.time_attribute} center={time_metadata['center']:.12g}"
        )

    kernels = ("streamed", "fused") if args.kernel == "both" else (args.kernel,)
    results = []
    kernel_comparisons = []
    for batch_size in args.batches:
        batch_results = {}
        batch_artifacts = {}
        for kernel in kernels:
            try:
                result, artifacts = run_case(
                    time_grid,
                    batch_size,
                    period=period,
                    duration=duration,
                    impact=args.impact,
                    kernel=kernel,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    _return_artifacts=True,
                )
            except Exception as exc:
                if not _is_resource_exhausted(exc):
                    raise
                result = {
                    "case": {
                        "n_times": int(len(time_grid)),
                        "batch_size": int(batch_size),
                        "kernel": kernel,
                    },
                    "status": "oom",
                    "error": str(exc),
                }
                artifacts = None
                print(
                    f"\nkernel={kernel} n_times={len(time_grid)} "
                    f"batch={batch_size}: OOM"
                )
                jax.clear_caches()
            results.append(result)
            batch_results[kernel] = result
            if artifacts is not None:
                batch_artifacts[kernel] = artifacts
            if result.get("status") != "oom":
                _print_result(result)

        if args.kernel == "both" and set(batch_artifacts) == {
            "streamed",
            "fused",
        }:
            comparison = compare_kernels(
                batch_results["streamed"],
                batch_artifacts["streamed"],
                batch_results["fused"],
                batch_artifacts["fused"],
            )
            kernel_comparisons.append(comparison)
            _print_kernel_comparison(comparison)

    if args.json_path:
        payload = {
            "environment": {
                "jax_version": jax.__version__,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "x64": bool(jax.config.read("jax_enable_x64")),
                "requested_kernel": args.kernel,
                "time_grid": time_metadata,
            },
            "results": results,
            "kernel_comparisons": kernel_comparisons,
        }
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        print(f"\nWrote {destination}")
    fidelity_failed = any(
        result.get("status") != "oom" and not result.get("fidelity_pass", False)
        for result in results
    ) or any(not result["fidelity_pass"] for result in kernel_comparisons)
    return 2 if args.require_fidelity and fidelity_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

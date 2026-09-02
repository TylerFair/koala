#!/usr/bin/env python3
"""GPU benchmark for the specialized direct-quadratic jaxoplanet kernel.

This compares the stock :mod:`jaxoplanet.core.limb_dark` polynomial evaluator
against ``models.jaxoplanet.limb_dark_quadratic.light_curve`` at identical
fixed-geometry separations, duration masks, and static production transit
windows.  Timings cover both the batched forward model and a complete scalar
Gaussian objective with its gradient with respect to every channel-specific
fitted parameter.

The limb-darkening coordinates are the literal quadratic intensity
coefficients ``u1,u2`` in

``I(mu) = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2``.

No Kipping ``q1,q2`` transformation is used anywhere in this benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, Iterable

import jax

jax.config.update("jax_enable_x64", True)
# The backend must be selected before imports initialize any JAX computation.
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

from jaxoplanet.core.limb_dark import light_curve as stock_light_curve

from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    build_transit_window_indices,
)
from tools.benchmark_jaxoplanet_gpu import (
    Timing,
    _is_resource_exhausted,
    device_memory,
    load_time_grid,
    parse_int_list,
)


Array = jax.Array

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


def specialized_light_curve() -> Callable:
    """Load the current experimental kernel only when the benchmark runs."""
    module = importlib.import_module("models.jaxoplanet.limb_dark_quadratic")
    evaluator = getattr(module, "light_curve", None)
    if evaluator is None:
        raise RuntimeError(
            "models.jaxoplanet.limb_dark_quadratic has no light_curve symbol"
        )
    return evaluator


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


def make_case(
    time_grid: np.ndarray,
    batch_size: int,
    *,
    period: float,
    duration: float,
    impact: float,
    u1: float,
    u2: float,
) -> tuple[
    Array,
    dict[str, Array],
    np.ndarray,
    Callable,
    Callable,
    Callable,
    Callable,
]:
    """Construct matched stock and specialized quadratic objectives."""
    times = np.asarray(time_grid, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time_grid must be one-dimensional with >=2 cadences")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError("time_grid must be finite and strictly increasing")

    t = jnp.asarray(times, dtype=jnp.float64)
    runtime = {
        "t": t,
        "period": jnp.asarray([period], dtype=jnp.float64),
        # load_time_grid centers real timestamps on the fixed WL transit time.
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

    phase_offsets, phase_mask = build_transit_phase_offsets(
        runtime["t"], runtime["period"], runtime["t0"], runtime["duration"]
    )
    phase_offsets, phase_mask = _block((phase_offsets, phase_mask))
    active_offsets = phase_offsets[:, window_indices]
    active_mask = phase_mask[:, window_indices]

    # [rprs, direct_u1, direct_u2, offset, slope, log_jitter].
    base = np.asarray(
        [0.1457, u1, u2, 1.0, 2.0e-3, np.log(7.0e-4)],
        dtype=np.float64,
    )
    theta_host = np.repeat(base[None, :], batch_size, axis=0)
    if batch_size > 1:
        lane = np.linspace(-1.0, 1.0, batch_size)
        theta_host[:, 0] += 0.002 * lane
        theta_host[:, 1] += 0.03 * lane
        theta_host[:, 2] += 0.02 * lane
    theta0 = jnp.asarray(theta_host, dtype=jnp.float64)

    quadratic_light_curve = specialized_light_curve()

    def build_forward(evaluator: Callable) -> Callable:
        def forward(theta: Array, dynamic: dict[str, Array]) -> Array:
            def single_channel(channel: Array) -> Array:
                channel_rors = channel[0]
                speed = 2 * jnp.sqrt(
                    jnp.maximum(
                        0,
                        jnp.square(1 + channel_rors)
                        - jnp.square(dynamic["b"][0]),
                    )
                ) / dynamic["duration"][0]
                separation = jnp.sqrt(
                    jnp.square(speed * active_offsets[0])
                    + jnp.square(dynamic["b"][0])
                )
                # These are literal u1,u2 values. There is deliberately no
                # q1/q2-to-u1/u2 (Kipping) transformation here.
                direct_u = channel[1:3]
                active_transit = evaluator(
                    direct_u, separation, channel_rors, order=10
                )
                active_transit = jnp.where(
                    active_mask[0], active_transit, 0
                )
                transit = jnp.zeros_like(
                    dynamic["t"], dtype=active_transit.dtype
                ).at[window_indices].set(active_transit)
                trend = channel[3] + channel[4] * (
                    dynamic["t"] - dynamic["t"][0]
                )
                return transit + trend

            return jax.vmap(single_channel)(theta)

        return forward

    stock_forward = build_forward(stock_light_curve)
    specialized_forward = build_forward(quadratic_light_curve)

    truth = _block(stock_forward(theta0, runtime))
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, times.size, dtype=jnp.float64)
    observations = jax.lax.stop_gradient(
        truth + 0.1 * jnp.exp(theta0[:, 5:6]) * jnp.sin(phase)[None, :]
    )

    def objective(
        theta: Array,
        dynamic: dict[str, Array],
        forward: Callable,
    ) -> Array:
        model = forward(theta, dynamic)
        sigma = jnp.exp(theta[:, 5:6])
        residual = (model - observations) / sigma
        return 0.5 * jnp.sum(residual * residual) + times.size * jnp.sum(
            theta[:, 5]
        )

    stock_objective = lambda theta, dynamic: objective(
        theta, dynamic, stock_forward
    )
    specialized_objective = lambda theta, dynamic: objective(
        theta, dynamic, specialized_forward
    )
    return (
        theta0,
        runtime,
        window_indices_host,
        stock_forward,
        specialized_forward,
        stock_objective,
        specialized_objective,
    )


def fidelity_metrics(
    theta: Array,
    runtime: dict[str, Array],
    stock_forward: Callable,
    specialized_forward: Callable,
    stock_objective: Callable,
    specialized_objective: Callable,
    *,
    compiled_stock_forward: Callable,
    compiled_specialized_forward: Callable,
    compiled_stock_value_grad: Callable,
    compiled_specialized_value_grad: Callable,
) -> dict[str, float | bool]:
    stock_flux = _block(compiled_stock_forward(theta, runtime))
    specialized_flux = _block(compiled_specialized_forward(theta, runtime))
    stock_value, stock_grad = _block(
        compiled_stock_value_grad(theta, runtime)
    )
    specialized_value, specialized_grad = _block(
        compiled_specialized_value_grad(theta, runtime)
    )

    flux_diff = np.asarray(specialized_flux - stock_flux)
    objective_grad_diff = np.asarray(specialized_grad - stock_grad)
    objective_grad_ref = np.asarray(stock_grad)

    weights = jnp.reshape(
        jnp.linspace(0.7, 1.3, stock_flux.size, dtype=stock_flux.dtype),
        stock_flux.shape,
    )

    def transit_projection(forward: Callable, value: Array) -> Array:
        return jnp.sum(forward(value, runtime) * weights)

    stock_transit_grad = _block(
        jax.jit(
            jax.grad(lambda value: transit_projection(stock_forward, value))
        )(theta)
    )
    specialized_transit_grad = _block(
        jax.jit(
            jax.grad(
                lambda value: transit_projection(specialized_forward, value)
            )
        )(theta)
    )
    # Only rprs,u1,u2 affect the transit. Identical nuisance derivatives must
    # not dilute the direct light-curve Jacobian comparison.
    transit_grad_diff = np.asarray(
        specialized_transit_grad - stock_transit_grad
    )[..., :3]
    transit_grad_ref = np.asarray(stock_transit_grad)[..., :3]

    return {
        "flux_max_abs": float(np.max(np.abs(flux_diff))),
        "objective_abs": float(
            np.abs(np.asarray(specialized_value - stock_value))
        ),
        "objective_abs_per_datum": float(
            np.abs(np.asarray(specialized_value - stock_value)) / stock_flux.size
        ),
        "objective_gradient_max_abs": float(
            np.max(np.abs(objective_grad_diff))
        ),
        "objective_gradient_relative_l2": float(
            np.linalg.norm(objective_grad_diff)
            / max(np.linalg.norm(objective_grad_ref), 1.0e-300)
        ),
        "transit_gradient_max_abs": float(
            np.max(np.abs(transit_grad_diff))
        ),
        "transit_gradient_relative_l2": float(
            np.linalg.norm(transit_grad_diff)
            / max(np.linalg.norm(transit_grad_ref), 1.0e-300)
        ),
        "all_finite": bool(
            np.all(np.isfinite(np.asarray(stock_flux)))
            and np.all(np.isfinite(np.asarray(specialized_flux)))
            and np.isfinite(np.asarray(stock_value)).all()
            and np.isfinite(np.asarray(specialized_value)).all()
            and np.all(np.isfinite(np.asarray(stock_grad)))
            and np.all(np.isfinite(np.asarray(specialized_grad)))
            and np.all(np.isfinite(np.asarray(stock_transit_grad)))
            and np.all(np.isfinite(np.asarray(specialized_transit_grad)))
        ),
    }


def run_case(
    time_grid: np.ndarray,
    batch_size: int,
    *,
    period: float,
    duration: float,
    impact: float,
    u1: float,
    u2: float,
    warmup: int,
    repeats: int,
) -> dict:
    (
        theta,
        runtime,
        window_indices,
        stock_forward,
        specialized_forward,
        stock_objective,
        specialized_objective,
    ) = make_case(
        time_grid,
        batch_size,
        period=period,
        duration=duration,
        impact=impact,
        u1=u1,
        u2=u2,
    )

    stock_forward_timing, compiled_stock_forward = _timed_jit(
        stock_forward, theta, runtime, warmup=warmup, repeats=repeats
    )
    specialized_forward_timing, compiled_specialized_forward = _timed_jit(
        specialized_forward, theta, runtime, warmup=warmup, repeats=repeats
    )
    stock_value_grad_timing, compiled_stock_value_grad = _timed_jit(
        jax.value_and_grad(stock_objective, argnums=0),
        theta,
        runtime,
        warmup=warmup,
        repeats=repeats,
    )
    specialized_value_grad_timing, compiled_specialized_value_grad = _timed_jit(
        jax.value_and_grad(specialized_objective, argnums=0),
        theta,
        runtime,
        warmup=warmup,
        repeats=repeats,
    )
    fidelity = fidelity_metrics(
        theta,
        runtime,
        stock_forward,
        specialized_forward,
        stock_objective,
        specialized_objective,
        compiled_stock_forward=compiled_stock_forward,
        compiled_specialized_forward=compiled_specialized_forward,
        compiled_stock_value_grad=compiled_stock_value_grad,
        compiled_specialized_value_grad=compiled_specialized_value_grad,
    )
    fidelity_pass = not (
        not fidelity["all_finite"]
        or fidelity["flux_max_abs"] > 1.0e-10
        or fidelity["objective_abs_per_datum"] > 1.0e-8
        or fidelity["transit_gradient_relative_l2"] > 1.0e-8
    )
    return {
        "case": {
            "n_times": int(len(time_grid)),
            "window_size": int(window_indices.size),
            "batch_size": int(batch_size),
            "period": float(period),
            "duration": float(duration),
            "impact": float(impact),
            "u1": float(u1),
            "u2": float(u2),
            "limb_darkening_parameterization": "direct_u1_u2_no_kipping",
        },
        "fidelity": fidelity,
        "fidelity_pass": fidelity_pass,
        "timings": {
            "stock_forward": asdict(stock_forward_timing),
            "specialized_forward": asdict(specialized_forward_timing),
            "stock_value_grad": asdict(stock_value_grad_timing),
            "specialized_value_grad": asdict(specialized_value_grad_timing),
        },
        "specialized_speedup_over_stock": {
            "forward": (
                stock_forward_timing.median_ms
                / specialized_forward_timing.median_ms
            ),
            "value_grad": (
                stock_value_grad_timing.median_ms
                / specialized_value_grad_timing.median_ms
            ),
        },
        "device_memory": device_memory(),
    }


def _print_result(result: dict) -> None:
    case = result["case"]
    fidelity = result["fidelity"]
    print(
        f"\nn_times={case['n_times']} batch={case['batch_size']} "
        f"window={case['window_size']}/{case['n_times']} direct_u="
        f"[{case['u1']:.4f}, {case['u2']:.4f}] (no Kipping)"
    )
    print(
        f"  parity {'PASS' if result['fidelity_pass'] else 'FAIL'} "
        f"flux={fidelity['flux_max_abs']:.3e} "
        f"objective/datum={fidelity['objective_abs_per_datum']:.3e} "
        f"transit_grad_rel_l2={fidelity['transit_gradient_relative_l2']:.3e}"
    )
    for name, timing in result["timings"].items():
        print(
            f"  {name:24s} median={timing['median_ms']:9.4f} ms "
            f"p05={timing['p05_ms']:9.4f} p95={timing['p95_ms']:9.4f} "
            f"compile+first={timing['compile_first_ms']:9.2f} ms"
        )
    speedup = result["specialized_speedup_over_stock"]
    print(
        f"  specialized speedup forward={speedup['forward']:.4f}x "
        f"value+grad={speedup['value_grad']:.4f}x"
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
    parser.add_argument("--u1", type=float, default=0.30)
    parser.add_argument("--u2", type=float, default=0.20)
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
        help="exit nonzero after writing JSON if any parity gate fails",
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
            raise SystemExit("--cadences must be >=2")
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
    for name in ("impact", "u1", "u2"):
        if not np.isfinite(getattr(args, name)):
            raise SystemExit(f"--{name} must be finite")
    if args.impact < 0.0:
        raise SystemExit("--impact must be non-negative")
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)

    time_grid, period, duration, time_metadata = _resolve_inputs(args)
    print(
        f"JAX {jax.__version__}; backend={jax.default_backend()}; "
        f"device={jax.devices()[0]}; x64={jax.config.read('jax_enable_x64')}"
    )
    print(
        "Quadratic LD parameterization: direct physical u1,u2; "
        "no Kipping q1,q2 transformation."
    )
    if time_metadata is not None:
        print(
            f"timestamps={time_metadata['path']} "
            f"attribute={args.time_attribute} center={time_metadata['center']:.12g}"
        )

    results = []
    for batch_size in args.batches:
        try:
            result = run_case(
                time_grid,
                batch_size,
                period=period,
                duration=duration,
                impact=args.impact,
                u1=args.u1,
                u2=args.u2,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        except Exception as exc:
            if not _is_resource_exhausted(exc):
                raise
            result = {
                "case": {
                    "n_times": int(len(time_grid)),
                    "batch_size": int(batch_size),
                    "limb_darkening_parameterization": (
                        "direct_u1_u2_no_kipping"
                    ),
                },
                "status": "oom",
                "error": str(exc),
            }
            print(f"\nn_times={len(time_grid)} batch={batch_size}: OOM")
            jax.clear_caches()
        results.append(result)
        if result.get("status") != "oom":
            _print_result(result)

    if args.json_path:
        payload = {
            "environment": {
                "jax_version": jax.__version__,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]),
                "x64": bool(jax.config.read("jax_enable_x64")),
                "time_grid": time_metadata,
            },
            "limb_darkening_parameterization": "direct_u1_u2_no_kipping",
            "results": results,
        }
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        print(f"\nWrote {destination}")

    fidelity_failed = any(
        result.get("status") != "oom" and not result.get("fidelity_pass", False)
        for result in results
    )
    return 2 if args.require_fidelity and fidelity_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

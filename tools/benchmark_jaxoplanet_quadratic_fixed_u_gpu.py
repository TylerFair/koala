#!/usr/bin/env python3
"""GPU benchmark for fixed direct quadratic limb darkening.

This benchmark matches the requested production parameterization: ``u1,u2``
are literal, fixed model inputs (never Kipping coordinates), while gradients
are taken with respect to radius ratio, impact parameter, duration, and the
per-channel nuisance parameters.  It compares stock jaxoplanet, the current
hand-contracted quadratic kernel, a stock-parity explicit-dot specialization,
and the higher-order-safe local-JVP prototype on real HAT-P-12/HAT-P-65 time
grids with the production static transit window.
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

import jax

jax.config.update("jax_enable_x64", True)
if __name__ == "__main__":
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == "--platform" and index + 1 < len(sys.argv):
            requested = sys.argv[index + 1]
            if requested in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", requested)
            break
        if token.startswith("--platform="):
            requested = token.split("=", 1)[1]
            if requested in {"cpu", "gpu"}:
                jax.config.update("jax_platform_name", requested)
            break

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jaxoplanet.core.limb_dark import light_curve as stock_light_curve

from models.jaxoplanet.experimental_quadratic_dot import (
    light_curve as dot_light_curve,
)
from models.jaxoplanet.experimental_quadratic_local_jvp import (
    light_curve as local_jvp_light_curve,
)
from models.jaxoplanet.limb_dark_quadratic import (
    light_curve as direct_light_curve,
)
from models.jaxoplanet.core import (
    build_transit_phase_offsets,
    build_transit_window_indices,
)
from tools.benchmark_jaxoplanet_gpu import Timing, device_memory, load_time_grid
from tools.benchmark_jaxoplanet_quadratic_gpu import PRESETS as _TIME_PRESETS


PRESETS = {
    "hatp12": {
        **_TIME_PRESETS["hatp12"],
        "impact": 0.343,
        "radius": 0.1406,
    },
    "hatp65": {
        **_TIME_PRESETS["hatp65"],
        "impact": 0.464,
        "radius": 0.1006,
    },
}


CANDIDATES = {
    "stock": stock_light_curve,
    "current_direct": direct_light_curve,
    "direct_dot": dot_light_curve,
    "local_jvp": local_jvp_light_curve,
}


def _block(value):
    return jax.block_until_ready(value)


def _timed(fn, theta, runtime, *, warmup, repeats):
    compiled = jax.jit(fn)
    start = time.perf_counter()
    _block(compiled(theta, runtime))
    compile_first_ms = 1.0e3 * (time.perf_counter() - start)
    for _ in range(warmup):
        _block(compiled(theta, runtime))
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        _block(compiled(theta, runtime))
        elapsed.append(1.0e3 * (time.perf_counter() - start))
    return (
        Timing(
            compile_first_ms=compile_first_ms,
            median_ms=float(statistics.median(elapsed)),
            p05_ms=float(np.percentile(elapsed, 5)),
            p95_ms=float(np.percentile(elapsed, 95)),
            repeats=repeats,
        ),
        compiled,
    )


def make_case(times, batch_size, preset, *, u1, u2):
    times = np.asarray(times, dtype=np.float64)
    t = jnp.asarray(times)
    period = jnp.asarray([preset["period"]])
    nominal_duration = jnp.asarray([preset["duration"]])
    t0 = jnp.asarray([0.0])
    indices_host = build_transit_window_indices(
        times, np.asarray(period), np.asarray(t0), np.asarray(nominal_duration)
    )
    indices = jnp.asarray(indices_host, dtype=jnp.int32)
    phase_offsets, phase_mask = build_transit_phase_offsets(
        t, period, t0, nominal_duration
    )
    active_offsets = phase_offsets[0, indices]
    active_mask = phase_mask[0, indices]

    lanes = np.linspace(-1.0, 1.0, batch_size)
    theta_host = np.column_stack(
        (
            preset["radius"] + 0.002 * lanes,
            preset["impact"] + 0.02 * lanes,
            preset["duration"] * (1.0 + 0.01 * lanes),
            np.ones(batch_size),
            2.0e-3 * np.ones(batch_size),
            np.log(7.0e-4) * np.ones(batch_size),
        )
    )
    theta = jnp.asarray(theta_host)
    runtime = {
        "t": t,
        "u": jnp.asarray(
            np.column_stack((u1 + 0.03 * lanes, u2 + 0.02 * lanes))
        ),
    }

    def build_forward(evaluator):
        def forward(value, dynamic):
            def one(channel, fixed_u):
                radius, impact, duration = channel[:3]
                speed = 2 * jnp.sqrt(
                    jnp.maximum(0, jnp.square(1 + radius) - impact**2)
                ) / duration
                separation = jnp.sqrt(
                    jnp.square(speed * active_offsets) + impact**2
                )
                active = evaluator(fixed_u, separation, radius, order=10)
                active = jnp.where(active_mask, active, 0)
                transit = jnp.zeros_like(dynamic["t"]).at[indices].set(active)
                trend = channel[3] + channel[4] * (
                    dynamic["t"] - dynamic["t"][0]
                )
                return transit + trend

            return jax.vmap(one)(value, dynamic["u"])

        return forward

    forwards = {name: build_forward(fn) for name, fn in CANDIDATES.items()}
    truth = _block(forwards["stock"](theta, runtime))
    phase = jnp.linspace(0, 2 * jnp.pi, times.size)
    observations = jax.lax.stop_gradient(
        truth + 0.1 * jnp.exp(theta[:, 5:6]) * jnp.sin(phase)[None, :]
    )

    def build_objective(forward):
        def objective(value, dynamic):
            model = forward(value, dynamic)
            sigma = jnp.exp(value[:, 5:6])
            residual = (model - observations) / sigma
            return 0.5 * jnp.sum(residual**2) + times.size * jnp.sum(
                value[:, 5]
            )

        return objective

    objectives = {
        name: build_objective(forward) for name, forward in forwards.items()
    }
    return theta, runtime, indices_host, forwards, objectives


def _hlo_metrics(fn, theta, runtime):
    lowered = jax.jit(fn).lower(theta, runtime)
    text = str(lowered.compiler_ir(dialect="stablehlo"))
    cost = lowered.compile().cost_analysis()
    return {
        "stablehlo_reduce_count": text.count("stablehlo.reduce"),
        "stablehlo_dot_general_count": text.count("stablehlo.dot_general"),
        "bytes_accessed_estimate": float(cost.get("bytes accessed", np.nan)),
        "flops_estimate": float(cost.get("flops", np.nan)),
        "transcendentals_estimate": float(
            cost.get("transcendentals", np.nan)
        ),
    }


def run_case(times, batch_size, preset, *, u1, u2, warmup, repeats):
    theta, runtime, indices, forwards, objectives = make_case(
        times, batch_size, preset, u1=u1, u2=u2
    )
    compiled = {}
    results = {}
    for name in CANDIDATES:
        forward_timing, forward_fn = _timed(
            forwards[name], theta, runtime, warmup=warmup, repeats=repeats
        )
        vg = jax.value_and_grad(objectives[name], argnums=0)
        vg_timing, vg_fn = _timed(
            vg, theta, runtime, warmup=warmup, repeats=repeats
        )
        compiled[name] = (forward_fn, vg_fn)
        results[name] = {
            "forward": asdict(forward_timing),
            "value_grad": asdict(vg_timing),
            "value_grad_hlo": _hlo_metrics(vg, theta, runtime),
        }

    reference_flux = _block(compiled["stock"][0](theta, runtime))
    reference_value, reference_grad = _block(
        compiled["stock"][1](theta, runtime)
    )
    reference_grad_host = np.asarray(reference_grad)
    for name in CANDIDATES:
        flux = _block(compiled[name][0](theta, runtime))
        value, gradient = _block(compiled[name][1](theta, runtime))
        flux_difference = np.asarray(flux - reference_flux)
        gradient_difference = np.asarray(gradient - reference_grad)
        fidelity = {
            "flux_max_abs": float(np.max(np.abs(flux_difference))),
            "objective_abs_per_datum": float(
                np.abs(np.asarray(value - reference_value)) / flux.size
            ),
            "gradient_max_abs": float(
                np.max(np.abs(gradient_difference), initial=0.0)
            ),
            "gradient_relative_l2": float(
                np.linalg.norm(gradient_difference)
                / max(np.linalg.norm(reference_grad_host), 1.0e-300)
            ),
            "all_finite": bool(
                np.all(np.isfinite(np.asarray(flux)))
                and np.isfinite(np.asarray(value)).all()
                and np.all(np.isfinite(np.asarray(gradient)))
            ),
        }
        results[name]["fidelity"] = fidelity
        results[name]["fidelity_pass"] = bool(
            fidelity["all_finite"]
            and fidelity["flux_max_abs"] <= 1.0e-10
            and fidelity["objective_abs_per_datum"] <= 1.0e-8
            and fidelity["gradient_relative_l2"] <= 1.0e-8
        )
        results[name]["speedup_over_stock"] = {
            "forward": (
                results["stock"]["forward"]["median_ms"]
                / results[name]["forward"]["median_ms"]
            ),
            "value_grad": (
                results["stock"]["value_grad"]["median_ms"]
                / results[name]["value_grad"]["median_ms"]
            ),
        }
    return {
        "case": {
            "n_times": int(len(times)),
            "window_size": int(indices.size),
            "batch_size": int(batch_size),
            "period": float(preset["period"]),
            "nominal_duration": float(preset["duration"]),
            "nominal_impact": float(preset["impact"]),
            "u1": float(u1),
            "u2": float(u2),
            "gradient_variables": [
                "radius_ratio",
                "impact_parameter",
                "duration",
                "offset",
                "slope",
                "log_jitter",
            ],
            "fixed_dynamic_variables": ["u1", "u2"],
            "limb_darkening_parameterization": "direct_u1_u2_no_kipping",
        },
        "results": results,
        "device_memory": device_memory(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--u1", type=float, default=0.30)
    parser.add_argument("--u2", type=float, default=0.20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--require-fidelity", action="store_true")
    args = parser.parse_args()
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)
    preset = PRESETS[args.preset]
    times, metadata = load_time_grid(
        preset["path"], attribute="time", transit_t0=preset["t0"]
    )
    print(
        f"JAX {jax.__version__}; backend={jax.default_backend()}; "
        f"device={jax.devices()[0]}; fixed direct u1,u2 (no Kipping)"
    )
    result = run_case(
        times,
        args.batch,
        preset,
        u1=args.u1,
        u2=args.u2,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    for name, item in result["results"].items():
        print(
            f"{name:15s} forward={item['forward']['median_ms']:.6f} ms "
            f"value+grad={item['value_grad']['median_ms']:.6f} ms "
            f"speedup={item['speedup_over_stock']['value_grad']:.4f}x "
            f"grad_rel={item['fidelity']['gradient_relative_l2']:.3e} "
            f"{'PASS' if item['fidelity_pass'] else 'FAIL'}"
        )
    payload = {
        "environment": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "time_grid": metadata,
        },
        **result,
    }
    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        print(f"Wrote {destination}")
    failed = any(
        not item["fidelity_pass"] for item in result["results"].values()
    )
    return 2 if args.require_fidelity and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

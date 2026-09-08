#!/usr/bin/env python3
"""Benchmark Koala's serial and parallel quasiseparable GP implementations.

The benchmark deliberately builds the GP inside every jitted workload and
passes all GP and linear-mean parameters as dynamic arguments.  It can be
imported safely; use :func:`main` to invoke it from Python.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_SIZES = (1024, 4096, 16384)
WORKLOADS = ("log_likelihood", "likelihood_gradient", "training_prediction")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--cadences", type=int, action="append", default=[],
        help="add one cadence count (repeat the option to add more)",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense-cap", type=int, default=128)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument(
        "--solver", choices=("serial", "parallel", "both"), default="both"
    )
    parser.add_argument(
        "--duplicates", action="store_true",
        help="include deterministic duplicate timestamps",
    )
    return parser


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _tinygp_revision() -> tuple[str, str | None]:
    version = importlib.metadata.version("tinygp")
    revision = None
    try:
        dist = importlib.metadata.distribution("tinygp")
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        revision = direct_url.get("vcs_info", {}).get("commit_id")
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass
    return version, revision


def _memory_stats(device: Any) -> dict[str, Any] | None:
    try:
        stats = device.memory_stats()
    except (AttributeError, RuntimeError):
        return None
    if stats is None:
        return None
    wanted = ("bytes_in_use", "peak_bytes_in_use")
    return {key: _jsonable(stats.get(key)) for key in wanted if key in stats}


def _block(value: Any, jax: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _timed_call(function: Callable[..., Any], args: tuple[Any, ...], jax: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = _block(function(*args), jax)
    return value, time.perf_counter() - started


def _measure(function: Callable[..., Any], args: tuple[Any, ...], repeats: int, jax: Any) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    compiled = function.lower(*args).compile()
    compile_seconds = time.perf_counter() - started
    first_value, first_seconds = _timed_call(compiled, args, jax)
    samples = [_timed_call(compiled, args, jax)[1] for _ in range(repeats)]
    return first_value, {
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_seconds,
        "steady_state_seconds": {
            "median": float(np.median(samples)),
            "min": float(np.min(samples)),
            "runs": samples,
        },
    }


def _data(n: int, dtype: Any, duplicates: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1907 + n)
    increments = rng.uniform(0.65, 1.35, size=n)
    times = np.cumsum(increments)
    times = (times - times[0]) / max(times[-1] - times[0], 1.0)
    if duplicates and n > 3:
        times[np.arange(17, n, 53)] = times[np.arange(17, n, 53) - 1]
    errors = 0.025 + 0.005 * (1.0 + np.sin(9.0 * times))
    signal = 0.2 - 0.08 * times + 0.035 * np.sin(13.0 * times)
    noise = rng.normal(scale=errors)
    return times.astype(dtype), errors.astype(dtype), (signal + noise).astype(dtype)


def _differences(left: Any, right: Any) -> dict[str, float]:
    left_flat = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in left])
    right_flat = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in right])
    absolute = np.abs(left_flat - right_flat)
    scale = np.maximum(np.maximum(np.abs(left_flat), np.abs(right_flat)), 1e-300)
    return {
        "max_abs": float(np.max(absolute, initial=0.0)),
        "max_rel": float(np.max(absolute / scale, initial=0.0)),
    }


def _flatten_result(name: str, result: Any, jax: Any) -> list[Any]:
    if name == "likelihood_gradient":
        likelihood, gradient = result
        leaves = [likelihood]
        leaves.extend(jax.tree_util.tree_leaves(gradient))
        return leaves
    if name == "training_prediction":
        mean, variance = result
        return [mean, variance]
    return [result]


def _dense_reference(times: np.ndarray, errors: np.ndarray, y: np.ndarray, params: dict[str, float]) -> dict[str, Any]:
    rho = np.exp(params["GP_log_rho"])
    sigma = np.exp(params["GP_log_sigma"])
    delta = np.abs(times[:, None] - times[None, :])
    scaled = np.sqrt(3.0) * delta / rho
    covariance = sigma**2 * (1.0 + scaled) * np.exp(-scaled)
    covariance[np.diag_indices_from(covariance)] += errors**2
    mean = params["c"] + params["v"] * (times - times[0])
    residual = y - mean
    chol = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
    log_likelihood = -0.5 * (
        residual @ alpha
        + 2.0 * np.log(np.diag(chol)).sum()
        + len(times) * np.log(2.0 * np.pi)
    )
    inverse = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(times))))
    inverse_diag = np.diag(inverse)
    covariance_weight = np.outer(alpha, alpha) - inverse
    gradient = {
        "GP_log_rho": 0.5 * np.sum(
            covariance_weight * (sigma**2 * scaled**2 * np.exp(-scaled))
        ),
        "GP_log_sigma": 0.5 * np.sum(covariance_weight * (2.0 * (
            covariance - np.diag(errors**2)
        ))),
        "c": np.sum(alpha),
        "v": np.dot(times - times[0], alpha),
    }
    prediction_mean = y - errors**2 * alpha
    prediction_variance = errors**2 * (1.0 - errors**2 * inverse_diag)
    prediction_variance += np.sqrt(np.finfo(times.dtype).eps)
    return {
        "log_likelihood": float(log_likelihood),
        "likelihood_gradient": gradient,
        "training_prediction": (prediction_mean, np.maximum(prediction_variance, 0.0)),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    # Configure precision before creating any JAX arrays.
    import jax

    jax.config.update("jax_enable_x64", not args.float32)
    import jax.numpy as jnp
    import jaxlib

    from models import gp as gp_module

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.dense_cap < 2:
        raise ValueError("--dense-cap must be at least 2")
    sizes = list(dict.fromkeys([*args.sizes, *args.cadences]))
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("all cadence counts must be at least 2")
    backend = jax.default_backend()
    if args.require_gpu and backend == "cpu":
        raise RuntimeError(
            "--require-gpu was requested, but JAX selected the CPU backend; "
            "run inside a GPU allocation with a CUDA-enabled JAX installation"
        )

    device = jax.devices()[0]
    tinygp_version, tinygp_revision = _tinygp_revision()
    dtype = np.float32 if args.float32 else np.float64
    selected = [args.solver] if args.solver != "both" else ["serial", "parallel"]
    availability: dict[str, dict[str, Any]] = {}
    for solver in selected:
        if solver == "serial":
            availability[solver] = {"available": True, "reason": None}
        else:
            try:
                supported = bool(gp_module.gp_parallel_supported())
            except AttributeError:
                supported = False
            availability[solver] = {
                "available": supported,
                "reason": None if supported else "installed tinygp has no parallel QuasisepSolver support",
            }

    metadata = {
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "tinygp_version": tinygp_version,
        "tinygp_git_revision": tinygp_revision,
        "platform": backend,
        "device_kind": getattr(device, "device_kind", None),
        "device_name": str(device),
        "dtype": np.dtype(dtype).name,
        "compiler": {
            "jax_backend_platform_version": getattr(device.client, "platform_version", None),
        },
        "memory_before": _memory_stats(device),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "metadata": metadata,
        "options": {
            "sizes": sizes,
            "repeats": args.repeats,
            "dense_cap": args.dense_cap,
            "duplicates": args.duplicates,
            "solver": args.solver,
        },
        "solver_availability": availability,
        "benchmarks": {},
        "agreement": {},
        "dense_reference": None,
    }

    def linear_mean(params: dict[str, Any], t: Any, t_ref: Any = None) -> Any:
        reference = jnp.min(t) if t_ref is None else t_ref
        return params["c"] + params["v"] * (t - reference)

    def make_functions(solver: str) -> dict[str, Callable[..., Any]]:
        def likelihood(params: dict[str, Any], t: Any, error: Any, y: Any) -> Any:
            model = gp_module.build_gp_model(
                params, t, error, mean_function=linear_mean,
                gp_solver=solver, assume_sorted=True, t_ref=t[0],
            )
            return model.log_probability(y)

        def likelihood_gradient(params: dict[str, Any], t: Any, error: Any, y: Any) -> Any:
            value, gradient = jax.value_and_grad(
                lambda dynamic_params: gp_module.build_gp_model(
                    dynamic_params, t, error, mean_function=linear_mean,
                    gp_solver=solver, assume_sorted=True, t_ref=t[0],
                ).log_probability(y)
            )(params)
            return value, gradient

        def prediction(params: dict[str, Any], t: Any, error: Any, y: Any) -> Any:
            model = gp_module.build_gp_model(
                params, t, error, mean_function=linear_mean,
                gp_solver=solver, assume_sorted=True, t_ref=t[0],
            )
            return gp_module.predict_gp_training_points(model, y)

        return {
            "log_likelihood": jax.jit(likelihood),
            "likelihood_gradient": jax.jit(likelihood_gradient),
            "training_prediction": jax.jit(prediction),
        }

    functions = {
        solver: make_functions(solver)
        for solver in selected if availability[solver]["available"]
    }
    outputs: dict[int, dict[str, dict[str, Any]]] = {}
    params_np = {
        "GP_log_rho": float(np.log(0.075)),
        "GP_log_sigma": float(np.log(0.04)),
        "c": 0.2,
        "v": -0.08,
    }
    for size in sizes:
        times_np, errors_np, y_np = _data(size, dtype, args.duplicates)
        call_args = (
            {key: jnp.asarray(value, dtype=dtype) for key, value in params_np.items()},
            jnp.asarray(times_np), jnp.asarray(errors_np), jnp.asarray(y_np),
        )
        report["benchmarks"][str(size)] = {}
        outputs[size] = {}
        for solver, solver_functions in functions.items():
            report["benchmarks"][str(size)][solver] = {}
            outputs[size][solver] = {}
            for name in WORKLOADS:
                value, timing = _measure(solver_functions[name], call_args, args.repeats, jax)
                outputs[size][solver][name] = value
                report["benchmarks"][str(size)][solver][name] = timing
        if "serial" in outputs[size] and "parallel" in outputs[size]:
            report["agreement"][str(size)] = {
                name: _differences(
                    _flatten_result(name, outputs[size]["serial"][name], jax),
                    _flatten_result(name, outputs[size]["parallel"][name], jax),
                )
                for name in WORKLOADS
            }

    if args.dense_cap > 0:
        dense_size = min(128, args.dense_cap)
        times_np, errors_np, y_np = _data(dense_size, np.float64, args.duplicates)
        dense = _dense_reference(times_np, errors_np, y_np, params_np)
        dense_report: dict[str, Any] = {"size": dense_size, "solvers": {}}
        dense_args = (
            {key: jnp.asarray(value, dtype=dtype) for key, value in params_np.items()},
            jnp.asarray(times_np, dtype=dtype), jnp.asarray(errors_np, dtype=dtype),
            jnp.asarray(y_np, dtype=dtype),
        )
        for solver, solver_functions in functions.items():
            ll = _block(solver_functions["log_likelihood"](*dense_args), jax)
            _, outputs_gradient = _block(
                solver_functions["likelihood_gradient"](*dense_args), jax
            )
            prediction = _block(solver_functions["training_prediction"](*dense_args), jax)
            dense_report["solvers"][solver] = {
                "log_likelihood": _differences([ll], [dense["log_likelihood"]]),
                "likelihood_gradient": _differences(
                    jax.tree_util.tree_leaves(outputs_gradient),
                    [dense["likelihood_gradient"][key] for key in sorted(params_np)],
                ),
                "training_prediction": _differences(
                    _flatten_result("training_prediction", prediction, jax),
                    list(dense["training_prediction"]),
                ),
            }
        report["dense_reference"] = dense_report

    report["metadata"]["memory_after"] = _memory_stats(device)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_benchmark(args)
    except RuntimeError as exc:
        print(f"benchmark_gp: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(_jsonable(report), indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

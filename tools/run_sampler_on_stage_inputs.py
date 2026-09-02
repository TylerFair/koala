#!/usr/bin/env python3
"""Replay a dumped production spectroscopic stage with a selected backend."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path


def _early_platform(argv):
    for index, value in enumerate(argv):
        if value == "--platform" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--platform="):
            return value.split("=", 1)[1]
    return "cpu"


os.environ.setdefault("JAX_PLATFORMS", _early_platform(sys.argv[1:]))
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro

import fit_jwst
from tools.spectro_stage_inputs import load_stage_inputs


_PIPELINE_MASTER_SEED = 555
_PIPELINE_STAGE_KEY_INDEX = {
    "low_resolution": 3,
    "low-resolution": 3,
    "lowres": 3,
    "high_resolution": 5,
    "high-resolution": 5,
    "highres": 5,
}


def _base_key_for_seed(stage, seed):
    """Return a stage key using the production master-key split.

    Production splits ``PRNGKey(555)`` into seven stage keys and uses entries
    3 and 5 for low- and high-resolution MCMC. Seed index zero therefore uses
    the exact dumped baseline. Positive indices use master seeds 556, 557,
    and so on while retaining the production split convention.
    """
    if seed is None or int(seed) == 0:
        return stage.rng_key, "dumped_pipeline_key"
    stage_kind = str(stage.meta.get("stage_kind", "")).lower()
    try:
        key_index = _PIPELINE_STAGE_KEY_INDEX[stage_kind]
    except KeyError as error:
        raise ValueError(
            "A positive --seed requires stage_kind low_resolution or "
            f"high_resolution; received {stage_kind!r}."
        ) from error
    keys = jax.random.split(
        jax.random.PRNGKey(_PIPELINE_MASTER_SEED + int(seed)), 7
    )
    return keys[key_index], "pipeline_master_seed_split"


DIAGNOSTIC_SITES = (
    "A_spot",
    "rors",
    "depths",
    "c",
    "v",
    "log_jitter",
    "c1",
    "c2",
    "total_error",
    "delta_r",
    "a1",
    "q",
    "limb_l",
    "limb_delta",
)

COMPILE_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "jaxpr_trace_seconds",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "mlir_lowering_seconds",
    "/jax/core/compile/backend_compile_duration": "backend_compile_seconds",
}


class _CompilationEvents:
    def __init__(self):
        self.events = []

    def listener(self, event, duration_secs, **_):
        if event in COMPILE_EVENTS:
            self.events.append((event, float(duration_secs)))

    def mark(self):
        return len(self.events)

    def since(self, mark):
        result = {name: 0.0 for name in COMPILE_EVENTS.values()}
        result.update(
            {
                name.replace("_seconds", "_count"): 0
                for name in COMPILE_EVENTS.values()
            }
        )
        for event, duration in self.events[mark:]:
            name = COMPILE_EVENTS[event]
            result[name] += duration
            result[name.replace("_seconds", "_count")] += 1
        result["total_recorded_compile_seconds"] = sum(
            result[name] for name in COMPILE_EVENTS.values()
        )
        return result


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _slice_value(value, sl, num_channels):
    if (
        hasattr(value, "ndim")
        and value.ndim > 0
        and value.shape[0] == num_channels
    ):
        return value[sl]
    return value


def _chunk_inputs(stage, start, end):
    sl = slice(start, end)
    varying = frozenset(stage.channel_varying_kwargs)
    return {
        "yerr": stage.yerr[sl],
        "y": stage.y[sl],
        "init_params": {
            name: _slice_value(value, sl, stage.num_channels)
            for name, value in stage.init_params.items()
        },
        "model_kwargs": {
            name: (
                _slice_value(value, sl, stage.num_channels)
                if name in varying
                else value
            )
            for name, value in stage.model_kwargs.items()
        },
    }


def _diagnostic_arrays(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {
            name: np.asarray(jax.device_get(array))
            for name, array in value.items()
        }
    result = {}
    for name in (
        "num_steps",
        "accept_prob",
        "diverging",
        "step_size",
        "mean_accept_prob",
        "map_gradient_norm",
        "map_newton_decrement",
        "map_iterations",
        "map_condition_number",
        "map_hessian_min_eigenvalue",
        "map_hessian_relative_error",
        "converged",
        "pareto_k",
        "is_ess",
        "imh_acceptance",
        "max_rejection_run",
        "hessian_condition_number",
        "hessian_min_eigenvalue",
        "gate_passed",
        "fell_back",
        "effective_draw_chunk_size",
        "map_seconds",
        "adaptation_seconds",
        "importance_round_seconds",
        "sampling_seconds",
        "postprocess_seconds",
        "fallback_seconds",
    ):
        if hasattr(value, name) and getattr(value, name) is not None:
            result[name] = np.asarray(jax.device_get(getattr(value, name)))
    return result


def _diagnostic_summary(arrays):
    result = {}
    if "num_steps" in arrays:
        values = np.asarray(arrays["num_steps"])
        result.update(
            {
                "num_steps_median": float(np.median(values)),
                "num_steps_mean": float(np.mean(values)),
                "num_steps_max": int(np.max(values)),
            }
        )
    if "diverging" in arrays:
        result["num_divergences"] = int(
            np.count_nonzero(arrays["diverging"])
        )
    if "accept_prob" in arrays:
        result["accept_prob_mean"] = float(
            np.mean(arrays["accept_prob"])
        )
    if "step_size" in arrays:
        result["step_size_median"] = float(
            np.median(arrays["step_size"])
        )
    for name in (
        "map_gradient_norm",
        "map_newton_decrement",
        "map_iterations",
        "map_condition_number",
        "map_hessian_min_eigenvalue",
        "map_hessian_relative_error",
    ):
        if name in arrays:
            values = np.asarray(arrays[name])
            result[f"{name}_median"] = float(np.median(values))
            result[f"{name}_max"] = float(np.max(values))
    for name in (
        "pareto_k",
        "is_ess",
        "imh_acceptance",
        "max_rejection_run",
        "hessian_condition_number",
        "hessian_min_eigenvalue",
        "effective_draw_chunk_size",
        "map_seconds",
        "adaptation_seconds",
        "sampling_seconds",
        "postprocess_seconds",
        "fallback_seconds",
    ):
        if name in arrays:
            values = np.asarray(arrays[name])
            result[f"{name}_min"] = float(np.min(values))
            result[f"{name}_median"] = float(np.median(values))
            result[f"{name}_max"] = float(np.max(values))
    for name in ("converged", "gate_passed", "fell_back"):
        if name in arrays:
            values = np.asarray(arrays[name], dtype=bool)
            result[f"{name}_count"] = int(np.count_nonzero(values))
            result[f"{name}_total"] = int(values.size)
    if "importance_round_seconds" in arrays:
        values = np.asarray(arrays["importance_round_seconds"])
        result["importance_round_seconds"] = values[0].tolist()
    return result


def _load_external_backend(name):
    module = importlib.import_module(f"models.{name}")
    candidates = (
        f"get_samples_{name}",
        "get_samples_independent",
        "get_samples",
    )
    for function_name in candidates:
        if hasattr(module, function_name):
            return getattr(module, function_name)
    raise AttributeError(
        f"models.{name} must expose one of {candidates} with the "
        "independent-NUTS call signature."
    )


def _run_joint_chunk(stage, chunk, key, runners, width, nuts_kwargs, mcmc_kwargs):
    runner = runners.get(width)
    if runner is None:
        runner, _, _ = fit_jwst._build_numpyro_mcmc(
            stage.model,
            fit_jwst._tree_to_f64(chunk["init_params"]),
            nuts_kwargs,
            mcmc_kwargs,
        )
        runners[width] = runner
    else:
        runner.sampler._init_strategy = nuts_kwargs.get(
            "init_strategy",
            numpyro.infer.init_to_value(values=chunk["init_params"]),
        )
    runner.run(
        key,
        stage.t,
        chunk["yerr"],
        y=chunk["y"],
        extra_fields=(
            "diverging",
            "accept_prob",
            "potential_energy",
            "num_steps",
        ),
        **chunk["model_kwargs"],
    )
    samples = runner.get_samples(group_by_chain=False)
    diagnostics = runner.get_extra_fields(group_by_chain=False)
    return samples, diagnostics


def _run_independent_chunk(
    backend,
    stage,
    chunk,
    key,
    runners,
    lane_width,
    nuts_kwargs,
    mcmc_kwargs,
    backend_kwargs,
):
    if backend == "independent_nuts":
        from models.independent_nuts import (
            build_independent_nuts_runner,
            get_samples_independent,
        )

        sampler = get_samples_independent
        builder = build_independent_nuts_runner
    elif backend == "independent_hmc":
        from models.independent_hmc import (
            build_independent_hmc_runner,
            get_samples_independent_hmc,
        )

        sampler = get_samples_independent_hmc
        builder = build_independent_hmc_runner
    elif backend == "laplace_is":
        from models.laplace_is import (
            build_laplace_is_runner,
            get_samples_laplace_is,
        )

        sampler = get_samples_laplace_is
        builder = build_laplace_is_runner
    else:
        sampler = _load_external_backend(backend)
        builder = None

    runner = None
    if builder is not None:
        runner = runners.get(lane_width)
        if runner is None:
            runner = builder(
                stage.model,
                nuts_kwargs=nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                lane_width=lane_width,
                channel_varying_kwargs=tuple(
                    name
                    for name in stage.channel_varying_kwargs
                    if name in chunk["model_kwargs"]
                ),
                **backend_kwargs,
            )
            runners[lane_width] = runner

    kwargs = dict(
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        lane_width=lane_width,
        channel_varying_kwargs=tuple(
            name
            for name in stage.channel_varying_kwargs
            if name in chunk["model_kwargs"]
        ),
        return_diagnostics=True,
        **backend_kwargs,
    )
    if runner is not None:
        kwargs["_runner"] = runner
    result = sampler(
        stage.model,
        key,
        stage.t,
        chunk["yerr"],
        chunk["y"],
        chunk["init_params"],
        **kwargs,
        **chunk["model_kwargs"],
    )
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def _concatenate_samples(chunks, widths):
    names = set(chunks[0])
    if any(set(chunk) != names for chunk in chunks[1:]):
        raise ValueError("Sampler chunks returned different sample sites.")
    result = {}
    for name in sorted(names):
        arrays = [np.asarray(jax.device_get(chunk[name])) for chunk in chunks]
        if all(
            array.ndim >= 2 and array.shape[1] == width
            for array, width in zip(arrays, widths)
        ):
            result[name] = np.concatenate(arrays, axis=1)
        else:
            reference = arrays[0]
            if not all(np.array_equal(reference, item) for item in arrays[1:]):
                raise ValueError(
                    f"Sample site {name!r} has no channel axis and differs "
                    "between chunks."
                )
            result[name] = reference
    return result


def _component_series(array, num_channels):
    array = np.asarray(array)
    if array.ndim < 2 or array.shape[1] != num_channels:
        return []
    trailing = array.shape[2:]
    if not trailing:
        return [("", array)]
    result = []
    for component in np.ndindex(trailing):
        suffix = "[" + ",".join(str(index) for index in component) + "]"
        result.append((suffix, array[(slice(None), slice(None), *component)]))
    return result


def _arviz_diagnostics(samples, num_channels):
    rows = []
    for site in DIAGNOSTIC_SITES:
        if site not in samples:
            continue
        for suffix, values in _component_series(samples[site], num_channels):
            for channel in range(num_channels):
                series = np.asarray(values[:, channel], dtype=np.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        ess = float(
                            np.asarray(az.ess(series[None, :], method="bulk"))
                        )
                    except Exception:
                        ess = float("nan")
                rows.append(
                    {
                        "site": f"{site}{suffix}",
                        "channel": channel,
                        "ess_bulk": ess,
                        # Production returns one chain. A valid rank R-hat
                        # cannot be estimated from it; do not manufacture
                        # pseudo-independent chains by splitting one trace.
                        "r_hat": None,
                    }
                )
    finite_ess = [row["ess_bulk"] for row in rows if np.isfinite(row["ess_bulk"])]
    return {
        "rows": rows,
        "ess_bulk_min": min(finite_ess) if finite_ess else None,
        "r_hat_note": (
            "Unavailable: these stage samplers produce one chain. ArviZ "
            "R-hat requires at least two independent chains."
        ),
    }


def _load_sample_mapping(path):
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    with open(path, "rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Reference pickle must contain a sample dictionary.")
    return {name: np.asarray(array) for name, array in value.items()}


def _reference_channels(array, start, end, candidate_channels):
    if array.ndim < 2:
        return array
    if array.shape[1] == candidate_channels:
        return array
    if array.shape[1] >= end:
        return array[:, start:end]
    raise ValueError(
        f"Reference channel axis {array.shape[1]} cannot provide {start}:{end}."
    )


def _safe_scaled_shift(value, reference, sigma):
    difference = float(value - reference)
    if sigma > 0.0:
        return difference / sigma
    return 0.0 if difference == 0.0 else float(np.copysign(np.inf, difference))


def compare_samples(candidate, reference, start, end):
    num_channels = end - start
    rows = []
    common_sites = sorted(set(candidate) & set(reference))
    for site in common_sites:
        candidate_array = np.asarray(candidate[site])
        reference_array = _reference_channels(
            np.asarray(reference[site]), start, end, num_channels
        )
        candidate_components = dict(
            _component_series(candidate_array, num_channels)
        )
        reference_components = dict(
            _component_series(reference_array, num_channels)
        )
        for suffix in sorted(set(candidate_components) & set(reference_components)):
            candidate_values = candidate_components[suffix]
            reference_values = reference_components[suffix]
            for channel in range(num_channels):
                cand = np.asarray(candidate_values[:, channel], dtype=np.float64)
                ref = np.asarray(reference_values[:, channel], dtype=np.float64)
                cand_q16, cand_median, cand_q84 = np.percentile(cand, [16, 50, 84])
                ref_q16, ref_median, ref_q84 = np.percentile(ref, [16, 50, 84])
                ref_sigma = float(np.std(ref, ddof=1)) if ref.size > 1 else 0.0
                cand_sigma = float(np.std(cand, ddof=1)) if cand.size > 1 else 0.0
                median_z = abs(_safe_scaled_shift(cand_median, ref_median, ref_sigma))
                if ref_sigma > 0.0:
                    sigma_ratio = cand_sigma / ref_sigma
                elif cand_sigma == 0.0:
                    sigma_ratio = 1.0
                else:
                    sigma_ratio = float("inf")
                passed = bool(median_z < 0.1 and 0.9 <= sigma_ratio <= 1.1)
                rows.append(
                    {
                        "site": f"{site}{suffix}",
                        "channel": start + channel,
                        "abs_median_shift_ref_sigma": median_z,
                        "sigma_ratio": sigma_ratio,
                        "p16_shift_ref_sigma": _safe_scaled_shift(
                            cand_q16, ref_q16, ref_sigma
                        ),
                        "p84_shift_ref_sigma": _safe_scaled_shift(
                            cand_q84, ref_q84, ref_sigma
                        ),
                        "pass": passed,
                    }
                )
    if not rows:
        raise ValueError("Candidate and reference have no comparable per-channel sites.")
    worst_median = max(rows, key=lambda row: row["abs_median_shift_ref_sigma"])
    worst_ratio = max(rows, key=lambda row: abs(np.log(row["sigma_ratio"])) if row["sigma_ratio"] > 0 else np.inf)
    return {
        "gate": {
            "abs_median_shift_ref_sigma": "< 0.1",
            "sigma_ratio": "[0.9, 1.1]",
        },
        "pass": all(row["pass"] for row in rows),
        "num_rows": len(rows),
        "num_failed": sum(not row["pass"] for row in rows),
        "rows": rows,
        "worst_median_shift": worst_median,
        "worst_sigma_ratio": worst_ratio,
    }


def _print_comparison(result):
    print("\nFidelity gate: |dmedian| < 0.1 reference sigma; sigma ratio in [0.9, 1.1]")
    print("site                 ch   |dmed|/sig   sig-ratio   p16-shift   p84-shift   gate")
    for row in result["rows"]:
        print(
            f"{row['site'][:20]:20s} {row['channel']:4d} "
            f"{row['abs_median_shift_ref_sigma']:12.4g} "
            f"{row['sigma_ratio']:11.4g} "
            f"{row['p16_shift_ref_sigma']:11.4g} "
            f"{row['p84_shift_ref_sigma']:11.4g} "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )
    print(
        f"Overall: {'PASS' if result['pass'] else 'FAIL'} "
        f"({result['num_failed']}/{result['num_rows']} failed); "
        f"worst median={result['worst_median_shift']['site']} "
        f"channel {result['worst_median_shift']['channel']} "
        f"({result['worst_median_shift']['abs_median_shift_ref_sigma']:.4g} sigma)."
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="Path to a *_inputs.pkl stage dump.")
    parser.add_argument("--backend", default="joint_nuts")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--potential-atol",
        type=float,
        help=(
            "Initial-potential absolute tolerance; default 1e-8 on CPU and "
            "1e-4 on GPU for cross-accelerator float64 reduction order."
        ),
    )
    parser.add_argument("--output-prefix")
    parser.add_argument("--compare")
    parser.add_argument(
        "--builder-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a recorded model-builder keyword (repeatable), e.g. "
            "jitter_prior=lognormal or jitter_prior_scale=2.0. Values are "
            "parsed as Python literals when possible. The stored initial "
            "potential is validated with the recorded builder before the "
            "override is applied."
        ),
    )
    parser.add_argument(
        "--nuts-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a recorded NUTS keyword (repeatable), e.g. "
            "mass_matrix=laplace or laplace_warmup=150. Values are parsed "
            "as Python literals when possible."
        ),
    )
    parser.add_argument(
        "--backend-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a backend keyword (repeatable), e.g. "
            "laplace_is_scale_inflation=1.2 or laplace_is_fallback=False."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be >= 1.")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be a non-negative replicate index.")
    potential_atol = (
        float(args.potential_atol)
        if args.potential_atol is not None
        else (1.0e-4 if args.platform == "gpu" else 1.0e-8)
    )
    if potential_atol < 0.0:
        raise ValueError("--potential-atol must be >= 0.")

    builder_overrides = {}
    for item in args.builder_override:
        if "=" not in item:
            raise ValueError(f"--builder-override expects KEY=VALUE; got {item!r}.")
        key, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        builder_overrides[key.strip()] = value
    backend_overrides = {}
    for item in args.backend_override:
        if "=" not in item:
            raise ValueError(
                f"--backend-override expects KEY=VALUE; got {item!r}."
            )
        key, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        backend_overrides[key.strip()] = value
    stage = load_stage_inputs(
        args.dump,
        validate_potential=True,
        potential_atol=potential_atol,
        builder_overrides=builder_overrides or None,
    )
    end = stage.num_channels if args.end is None else int(args.end)
    if not (0 <= args.start < end <= stage.num_channels):
        raise ValueError(
            f"Invalid channel range {args.start}:{end} for "
            f"{stage.num_channels} channels."
        )
    selected = stage.select(args.start, end)
    chunk_size = int(args.chunk_size or selected.chunk_size)
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1.")

    nuts_kwargs = dict(selected.nuts_kwargs)
    # Every replay starts from the selected physical values; retaining a
    # serialized init_to_value over the full stage would defeat channel slicing.
    nuts_kwargs.pop("init_strategy", None)
    for item in args.nuts_override:
        if "=" not in item:
            raise ValueError(f"--nuts-override expects KEY=VALUE; got {item!r}.")
        name, raw = item.split("=", 1)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        nuts_kwargs[name.strip()] = value
    mcmc_kwargs = dict(selected.mcmc_kwargs)
    mcmc_kwargs.update(
        {
            "num_warmup": int(
                args.warmup
                if args.warmup is not None
                else mcmc_kwargs.get("num_warmup", 1000)
            ),
            "num_samples": int(
                args.samples
                if args.samples is not None
                else mcmc_kwargs.get("num_samples", 1000)
            ),
            "progress_bar": False,
        }
    )
    base_key, rng_source = _base_key_for_seed(selected, args.seed)

    output_prefix = Path(
        args.output_prefix
        or f"{Path(args.dump).stem}_{args.backend}_{args.start}_{end}"
    ).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    compilation = _CompilationEvents()
    try:
        jax.monitoring.register_event_duration_secs_listener(
            compilation.listener
        )
    except ValueError:
        pass

    sample_chunks = []
    widths = []
    chunk_timings = []
    chunk_diagnostics = []
    raw_diagnostics = {}
    runners = {}
    resident_lane_width = (
        chunk_size
        if selected.num_channels > chunk_size
        else selected.num_channels
    )
    total_started = time.perf_counter()
    for chunk_index, local_start in enumerate(
        range(0, selected.num_channels, chunk_size)
    ):
        local_end = min(local_start + chunk_size, selected.num_channels)
        width = local_end - local_start
        chunk = _chunk_inputs(selected, local_start, local_end)
        # Match fit_jwst.get_samples_chunked: fold the stage key by the
        # production chunk index, not by the starting channel number.
        global_chunk_index = (args.start + local_start) // chunk_size
        key = jax.random.fold_in(base_key, global_chunk_index)
        mark = compilation.mark()
        started = time.perf_counter()
        if args.backend == "joint_nuts":
            samples, diagnostics = _run_joint_chunk(
                selected,
                chunk,
                key,
                runners,
                width,
                nuts_kwargs,
                mcmc_kwargs,
            )
        else:
            samples, diagnostics = _run_independent_chunk(
                args.backend,
                selected,
                chunk,
                key,
                runners,
                resident_lane_width,
                nuts_kwargs,
                mcmc_kwargs,
                backend_overrides,
            )
        samples = jax.device_get(samples)
        elapsed = time.perf_counter() - started
        compile_stats = compilation.since(mark)
        arrays = _diagnostic_arrays(diagnostics)
        sample_chunks.append(samples)
        widths.append(width)
        for name, array in arrays.items():
            raw_diagnostics[f"chunk_{chunk_index:03d}_{name}"] = array
        diagnostic_row = {
            "chunk_index": chunk_index,
            "channels": [args.start + local_start, args.start + local_end],
            **_diagnostic_summary(arrays),
        }
        chunk_diagnostics.append(diagnostic_row)
        chunk_timings.append(
            {
                "chunk_index": chunk_index,
                "channels": [args.start + local_start, args.start + local_end],
                "wall_seconds": elapsed,
                "compile_inclusive": compile_stats[
                    "total_recorded_compile_seconds"
                ] > 0.0,
                **compile_stats,
            }
        )
        print(
            f"chunk {args.start + local_start}:{args.start + local_end} "
            f"completed in {elapsed:.3f} s "
            f"(recorded compile {compile_stats['total_recorded_compile_seconds']:.3f} s)",
            flush=True,
        )
    total_wall = time.perf_counter() - total_started

    samples = _concatenate_samples(sample_chunks, widths)
    pickle_path = Path(f"{output_prefix}.pkl")
    with open(pickle_path, "wb") as stream:
        pickle.dump(samples, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(f"{output_prefix}.npz", **samples)
    np.savez_compressed(f"{output_prefix}.diagnostics.npz", **raw_diagnostics)

    compile_inclusive_wall = sum(
        row["wall_seconds"]
        for row in chunk_timings
        if row["compile_inclusive"]
    )
    steady_rows = [
        row for row in chunk_timings if not row["compile_inclusive"]
    ]
    timing = {
        "dump": str(Path(args.dump).resolve()),
        "backend": args.backend,
        "platform": args.platform,
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "channels": [args.start, end],
        "warmup": mcmc_kwargs["num_warmup"],
        "effective_warmup": (
            int(nuts_kwargs.get("laplace_warmup", 150))
            if nuts_kwargs.get("mass_matrix") == "laplace"
            else mcmc_kwargs["num_warmup"]
        ),
        "samples": mcmc_kwargs["num_samples"],
        "chunk_size": chunk_size,
        "seed_index": args.seed,
        "rng_source": rng_source,
        "pipeline_master_seed": (
            None
            if args.seed is None
            else _PIPELINE_MASTER_SEED + int(args.seed)
        ),
        "potential_atol": potential_atol,
        "nuts_kwargs": nuts_kwargs,
        "builder_overrides": builder_overrides,
        "backend_overrides": backend_overrides,
        "resident_lane_width": resident_lane_width,
        "total_wall_seconds": total_wall,
        "compile_inclusive_wall_seconds": compile_inclusive_wall,
        "recorded_compile_seconds": sum(
            row["total_recorded_compile_seconds"] for row in chunk_timings
        ),
        "steady_wall_seconds": sum(row["wall_seconds"] for row in steady_rows),
        "steady_chunk_wall_median_seconds": (
            float(np.median([row["wall_seconds"] for row in steady_rows]))
            if steady_rows
            else None
        ),
        "timing_note": (
            "A chunk is compile-inclusive when JAX reported compilation "
            "events during its wall interval. Steady timing is reported only "
            "for chunks with no such event."
        ),
        "chunks": chunk_timings,
    }
    _write_json(f"{output_prefix}.timing.json", timing)
    _write_json(
        f"{output_prefix}.diagnostics.json",
        {
            "backend": args.backend,
            "chunks": chunk_diagnostics,
            "num_divergences_total": sum(
                row.get("num_divergences", 0) for row in chunk_diagnostics
            ),
            "raw_arrays": f"{output_prefix}.diagnostics.npz",
        },
    )
    arviz_result = _arviz_diagnostics(samples, selected.num_channels)
    _write_json(f"{output_prefix}.arviz.json", arviz_result)

    if args.compare:
        comparison = compare_samples(
            samples,
            _load_sample_mapping(args.compare),
            args.start,
            end,
        )
        _write_json(f"{output_prefix}.comparison.json", comparison)
        _print_comparison(comparison)

    print(f"Saved samples: {pickle_path} and {output_prefix}.npz")
    print(
        f"Saved timing/diagnostics: {output_prefix}.timing.json, "
        f"{output_prefix}.diagnostics.json, {output_prefix}.arviz.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

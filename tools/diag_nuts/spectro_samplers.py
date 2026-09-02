#!/usr/bin/env python3
"""Matched CPU sampler accounting for the production-shaped spectro model."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS
from numpyro.infer.hmc import hmc
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.independent_nuts import get_samples_independent
from tools.diag_nuts.common import (
    block,
    effective_sample_sizes,
    json_dump,
    jsonable,
    latent_labels,
    make_spectro_problem,
    posterior_agreement,
    sample_moments,
    site_base,
    slice_lane,
    summarize_steps,
)


CONFIGS = (
    "reference",
    "joint_diag",
    "joint_dense",
    "independent_dense",
    "laplace_d10_w300",
    "laplace_d10_w150",
    "laplace_d10_w150_ta90",
    "laplace_d10_w150_ta95",
    "laplace_d5_w150",
    "laplace_d6_w150",
    "laplace_map_d10_w50",
    "non_gaussian",
    "jitter_ablation",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=CONFIGS, required=True)
    p.add_argument("--channels", type=int, default=40)
    p.add_argument("--cadences", type=int, default=230)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--reference-channels", type=int, default=None)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument(
        "--geometry",
        default="acceleration_reports/diag_nuts/spectro_geometry.npz",
    )
    p.add_argument(
        "--reference",
        default="acceleration_reports/diag_nuts/reference_draws.npz",
    )
    p.add_argument("--negligible-jitter", action="store_true")
    p.add_argument(
        "--output-dir", default="acceleration_reports/diag_nuts"
    )
    return p


def save_draws(path: Path, draws: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{name: np.asarray(jax.device_get(value)) for name, value in draws.items()}
    )


def load_draws(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def trace_latents(problem) -> set[str]:
    err, obs, kwargs, init = slice_lane(problem, 0)
    info = initialize_model(
        jax.random.PRNGKey(999),
        problem.model,
        init_strategy=init_to_value(values=init),
        model_args=(problem.t, err),
        model_kwargs={"y": obs, **kwargs},
    )
    return set(info.param_info.z)


def summarize_run(
    *,
    name: str,
    draws: Mapping[str, Any],
    steps: Any,
    accept: Any,
    diverging: Any,
    step_size: Any,
    wall_seconds: float,
    latent_names: set[str],
    phase_seconds: Mapping[str, float] | None = None,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    draws_np = {k: np.asarray(jax.device_get(v)) for k, v in draws.items()}
    steps_np = np.asarray(jax.device_get(steps))
    ess = effective_sample_sizes(draws_np, latent_names)
    ess_summary = ess["summary"]
    num_draws = int(next(iter(draws_np.values())).shape[0])
    result = {
        "config": name,
        "wall_seconds": float(wall_seconds),
        "phase_seconds": dict(phase_seconds or {}),
        "num_draws": num_draws,
        "steps": summarize_steps(steps_np),
        "mean_accept_prob": float(np.mean(np.asarray(jax.device_get(accept)))),
        "divergences": int(np.sum(np.asarray(jax.device_get(diverging)))),
        "step_size": jsonable(np.asarray(jax.device_get(step_size))),
        "ess": ess,
        "ess_per_draw_mean": float(ess_summary["mean"] / num_draws),
        "ess_per_draw_median": float(ess_summary["median"] / num_draws),
        "ess_per_second_sum": float(
            sum(
                np.sum(np.asarray(value))
                for key, value in ess.items()
                if key != "summary"
            )
            / wall_seconds
        ),
    }
    if reference is not None:
        result["posterior_agreement"] = posterior_agreement(
            draws_np, reference, latent_names
        )
    return result


def run_joint(problem, *, dense: bool, warmup: int, samples: int, seed: int):
    kernel = NUTS(
        problem.model,
        init_strategy=init_to_value(values=problem.init_params),
        dense_mass=dense,
        regularize_mass_matrix=True,
        target_accept_prob=0.8,
        max_tree_depth=10,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        progress_bar=False,
        jit_model_args=True,
    )
    start = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(seed),
        problem.t,
        problem.yerr,
        y=problem.y,
        extra_fields=("num_steps", "accept_prob", "diverging"),
        **problem.model_kwargs,
    )
    draws = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    block((draws, extra))
    elapsed = time.perf_counter() - start
    return draws, extra, mcmc.last_state.adapt_state.step_size, elapsed


def run_independent(problem, *, warmup: int, samples: int, seed: int):
    start = time.perf_counter()
    draws, diagnostics = get_samples_independent(
        problem.model,
        jax.random.PRNGKey(seed),
        problem.t,
        problem.yerr,
        problem.y,
        problem.init_params,
        num_warmup=warmup,
        num_samples=samples,
        lane_width=int(problem.yerr.shape[0]),
        channel_varying_kwargs=problem.channel_varying,
        dense_mass=True,
        regularize_mass_matrix=True,
        target_accept_prob=0.8,
        max_tree_depth=10,
        return_diagnostics=True,
        **problem.model_kwargs,
    )
    block(
        (
            draws,
            diagnostics.num_steps,
            diagnostics.accept_prob,
            diagnostics.diverging,
            diagnostics.step_size,
        )
    )
    elapsed = time.perf_counter() - start
    return draws, diagnostics, elapsed


def _partition_kwargs(problem):
    varying = {}
    shared = {}
    varying_names = frozenset(problem.channel_varying)
    channels = int(problem.yerr.shape[0])
    for name, value in problem.model_kwargs.items():
        array = jnp.asarray(value)
        if name in varying_names:
            if array.shape[0] != channels:
                raise ValueError(f"Bad channel axis for {name}: {array.shape}")
            varying[name] = array[:, None, ...]
        else:
            shared[name] = array
    return varying, shared


def _batched_initial_z(problem, info, unravel, maps, *, start_at_map: bool):
    channels = int(problem.yerr.shape[0])
    if start_at_map:
        trees = [unravel(jnp.asarray(maps[i])) for i in range(channels)]
    else:
        trees = []
        for channel in range(channels):
            _, _, _, init = slice_lane(problem, channel)
            z = {}
            for name in sorted(info.param_info.z):
                site = info.model_trace[name]
                value = init.get(name, site["value"])
                z[name] = biject_to(site["fn"].support).inv(jnp.asarray(value))
            trees.append(z)
    return jax.tree.map(lambda *values: jnp.stack(values), *trees)


def run_laplace(
    problem,
    geometry_path: str,
    *,
    warmup: int,
    samples: int,
    max_tree_depth: int,
    start_at_map: bool,
    target_accept: float,
    seed: int,
):
    with np.load(geometry_path, allow_pickle=False) as geometry:
        maps = np.asarray(geometry["map"])
        covariance = np.asarray(geometry["covariance"])
    channels = int(problem.yerr.shape[0])
    if maps.shape[0] != channels:
        raise ValueError(
            f"Geometry has {maps.shape[0]} lanes but problem has {channels}."
        )
    err0, obs0, kwargs0, init0 = slice_lane(problem, 0)
    info = initialize_model(
        jax.random.PRNGKey(seed + 1),
        problem.model,
        init_strategy=init_to_value(values=init0),
        dynamic_args=True,
        model_args=(problem.t, err0),
        model_kwargs={"y": obs0, **kwargs0},
    )
    flat, unravel = ravel_pytree(info.param_info.z)
    if covariance.shape[1:] != (flat.size, flat.size):
        raise ValueError("Laplace matrix dimension does not match model state.")
    z0 = _batched_initial_z(
        problem, info, unravel, maps, start_at_map=start_at_map
    )
    varying, shared = _partition_kwargs(problem)
    errors = jnp.asarray(problem.yerr)[:, None, :]
    observations = jnp.asarray(problem.y)[:, None, :]
    inverse_mass = jnp.asarray(covariance)
    init_kernel, sample_kernel = hmc(
        potential_fn_gen=info.potential_fn, algo="NUTS"
    )
    varying_axes = {name: 0 for name in varying}

    def init_one(z, inverse_mm, err, obs, lane_kwargs, key):
        return init_kernel(
            z,
            num_warmup=warmup,
            step_size=1.0,
            inverse_mass_matrix=inverse_mm,
            adapt_step_size=True,
            adapt_mass_matrix=False,
            dense_mass=True,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
            find_heuristic_step_size=False,
            regularize_mass_matrix=False,
            model_args=(problem.t, err),
            model_kwargs={"y": obs, **shared, **lane_kwargs},
            rng_key=key,
        )

    init_program = jax.jit(
        jax.vmap(init_one, in_axes=(0, 0, 0, 0, varying_axes, 0))
    )

    def advance(states):
        def one(state, err, obs, lane_kwargs):
            return sample_kernel(
                state,
                model_args=(problem.t, err),
                model_kwargs={"y": obs, **shared, **lane_kwargs},
            )

        return jax.vmap(one, in_axes=(0, 0, 0, varying_axes))(
            states, errors, observations, varying
        )

    warmup_program = jax.jit(
        lambda states: jax.lax.fori_loop(0, warmup, lambda _, state: advance(state), states)
    )

    def collect(states):
        def step(state, _):
            state = advance(state)
            return state, (state.z, state.num_steps, state.accept_prob, state.diverging)

        return jax.lax.scan(step, states, xs=None, length=samples)

    sample_program = jax.jit(collect)

    def post_lane(z, err, obs, lane_kwargs):
        return info.postprocess_fn(
            problem.t, err, y=obs, **shared, **lane_kwargs
        )(z)

    post_lanes = jax.vmap(post_lane, in_axes=(0, 0, 0, varying_axes))
    post_program = jax.jit(jax.vmap(post_lanes, in_axes=(0, None, None, None)))

    phases = {}
    total_start = time.perf_counter()
    start = time.perf_counter()
    states = init_program(
        z0,
        inverse_mass,
        errors,
        observations,
        varying,
        jax.random.split(jax.random.PRNGKey(seed), channels),
    )
    block(states)
    phases["initialize_and_compile"] = time.perf_counter() - start

    start = time.perf_counter()
    states = warmup_program(states)
    block(states)
    phases["warmup_and_compile"] = time.perf_counter() - start

    start = time.perf_counter()
    states, (z_draws, steps, accept, diverging) = sample_program(states)
    block((states, z_draws, steps, accept, diverging))
    phases["sampling_and_compile"] = time.perf_counter() - start

    start = time.perf_counter()
    draws = post_program(z_draws, errors, observations, varying)
    block(draws)
    phases["postprocess_and_compile"] = time.perf_counter() - start
    phases["total"] = time.perf_counter() - total_start
    return (
        draws,
        steps,
        accept,
        diverging,
        states.adapt_state.step_size,
        phases,
    )


BOUNDS = {
    "c": (0.9, 1.1),
    "c1": (0.0, 1.0),
    "c2": (0.001, 1.0),
    "log_jitter": (np.log(1.0e-6), np.log(1.0)),
    "rors": (np.sqrt(1.0e-5), np.sqrt(0.5)),
    "v": (-0.1, 0.1),
}


def constrain(label: str, z: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    low, high = BOUNDS[site_base(label)]
    return low + (high - low) * expit(z)


def run_non_gaussian(geometry_path: str, reference_path: str, seed: int):
    with np.load(geometry_path, allow_pickle=False) as geometry:
        maps = np.asarray(geometry["map"])
        covariance = np.asarray(geometry["covariance"])
        labels = [str(x) for x in geometry["labels"]]
    reference = load_draws(reference_path)
    rng = np.random.default_rng(seed)
    probabilities = np.asarray([0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99])
    rows = []
    for channel in range(maps.shape[0]):
        z_laplace = rng.multivariate_normal(maps[channel], covariance[channel], size=20000)
        for index, label in enumerate(labels):
            name = site_base(label)
            ref = np.asarray(reference[name])
            # All current spectro latents are scalar per channel, except rors's
            # trailing one-planet axis.
            ref_site = ref[:, channel]
            if ref_site.ndim > 1:
                element = int(label.split("[", 1)[1].rstrip("]")) if "[" in label else 0
                ref_site = ref_site[:, element]
            lap = constrain(label, z_laplace[:, index])
            ref_m = sample_moments(ref_site)
            lap_m = sample_moments(lap)
            ref_q = np.quantile(ref_site, probabilities)
            lap_q = np.quantile(lap, probabilities)
            scale = max(ref_m["sd"], np.finfo(np.float64).tiny)
            rows.append(
                {
                    "channel": channel,
                    "site": label,
                    "nuts": ref_m,
                    "laplace": lap_m,
                    "quantile_probabilities": probabilities.tolist(),
                    "nuts_quantiles": ref_q.tolist(),
                    "laplace_quantiles": lap_q.tolist(),
                    "qq_max_abs_sigma": float(np.max(np.abs(ref_q - lap_q)) / scale),
                    "sigma_over_prior_width": float(ref_m["sd"] / (BOUNDS[name][1] - BOUNDS[name][0])),
                }
            )
    return {
        "laplace_draws_per_channel": 20000,
        "rows": rows,
        "summary": {
            "max_abs_nuts_skewness": float(
                max(abs(row["nuts"]["skewness"]) for row in rows)
            ),
            "max_abs_nuts_excess_kurtosis": float(
                max(abs(row["nuts"]["excess_kurtosis"]) for row in rows)
            ),
            "median_qq_max_abs_sigma": float(
                np.median([row["qq_max_abs_sigma"] for row in rows])
            ),
            "max_qq_max_abs_sigma": float(
                max(row["qq_max_abs_sigma"] for row in rows)
            ),
        },
    }


def physical_map_from_geometry(path: str | Path, channel: int = 0) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as geometry:
        flat = np.asarray(geometry["map"])[channel]
        labels = [str(x) for x in geometry["labels"]]
    values = {}
    for index, label in enumerate(labels):
        name = site_base(label)
        value = constrain(label, flat[index])
        if name == "rors":
            values[name] = jnp.asarray([[value]])
        else:
            values[name] = jnp.asarray([value])
    return values


def run_jitter_ablation(problem, geometry_path: str, seed: int):
    from numpyro.handlers import condition

    if int(problem.yerr.shape[0]) != 1:
        raise ValueError("jitter_ablation requires --channels 1.")
    map_values = physical_map_from_geometry(geometry_path)
    results = {}
    for fixed in (False, True):
        model = (
            condition(problem.model, data={"log_jitter": map_values["log_jitter"]})
            if fixed
            else problem.model
        )
        init = dict(map_values)
        if fixed:
            init.pop("log_jitter", None)
        kernel = NUTS(
            model,
            init_strategy=init_to_value(values=init),
            dense_mass=True,
            regularize_mass_matrix=True,
            target_accept_prob=0.8,
            max_tree_depth=10,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=500,
            num_samples=500,
            progress_bar=False,
            jit_model_args=True,
        )
        start = time.perf_counter()
        mcmc.run(
            jax.random.PRNGKey(seed + int(fixed)),
            problem.t,
            problem.yerr,
            y=problem.y,
            extra_fields=("num_steps", "accept_prob", "diverging"),
            **problem.model_kwargs,
        )
        draws = mcmc.get_samples()
        extra = mcmc.get_extra_fields()
        block((draws, extra))
        wall = time.perf_counter() - start
        results["jitter_fixed" if fixed else "jitter_free"] = {
            "wall_seconds": wall,
            "steps": summarize_steps(np.asarray(extra["num_steps"])),
            "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
            "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        }
    free = results["jitter_free"]
    held = results["jitter_fixed"]
    results["comparison"] = {
        "mean_step_reduction_factor": free["steps"]["mean"] / held["steps"]["mean"],
        "wall_speedup": free["wall_seconds"] / held["wall_seconds"],
    }
    return results


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = make_spectro_problem(
        args.channels,
        args.cadences,
        args.seed,
        negligible_jitter=args.negligible_jitter,
        reference_channels=args.reference_channels,
    )
    latent_names = trace_latents(problem)
    reference = (
        load_draws(args.reference)
        if Path(args.reference).is_file() and args.config != "reference"
        else None
    )

    if args.config == "non_gaussian":
        payload = run_non_gaussian(args.geometry, args.reference, args.seed + 700)
        json_dump(output_dir / "non_gaussian.json", jsonable(payload))
        print("wrote non_gaussian.json")
        return
    if args.config == "jitter_ablation":
        payload = run_jitter_ablation(problem, args.geometry, args.seed + 800)
        json_dump(output_dir / "jitter_ablation.json", jsonable(payload))
        print("wrote jitter_ablation.json")
        return

    if args.config == "reference":
        warmup, samples = max(args.warmup, 500), max(args.samples, 1000)
        draws, diagnostics, wall = run_independent(
            problem, warmup=warmup, samples=samples, seed=args.seed + 200
        )
        result = summarize_run(
            name=args.config,
            draws=draws,
            steps=diagnostics.num_steps,
            accept=diagnostics.accept_prob,
            diverging=diagnostics.diverging,
            step_size=diagnostics.step_size,
            wall_seconds=wall,
            latent_names=latent_names,
        )
        draw_path = Path(args.reference)
    elif args.config.startswith("joint_"):
        draws, extra, step_size, wall = run_joint(
            problem,
            dense=args.config == "joint_dense",
            warmup=args.warmup,
            samples=args.samples,
            seed=args.seed + (301 if args.config == "joint_diag" else 302),
        )
        result = summarize_run(
            name=args.config,
            draws=draws,
            steps=extra["num_steps"],
            accept=extra["accept_prob"],
            diverging=extra["diverging"],
            step_size=step_size,
            wall_seconds=wall,
            latent_names=latent_names,
            reference=reference,
        )
        draw_path = output_dir / f"{args.config}_draws.npz"
    elif args.config == "independent_dense":
        draws, diagnostics, wall = run_independent(
            problem,
            warmup=args.warmup,
            samples=args.samples,
            seed=args.seed + 400,
        )
        result = summarize_run(
            name=args.config,
            draws=draws,
            steps=diagnostics.num_steps,
            accept=diagnostics.accept_prob,
            diverging=diagnostics.diverging,
            step_size=diagnostics.step_size,
            wall_seconds=wall,
            latent_names=latent_names,
            reference=reference,
        )
        draw_path = output_dir / "independent_dense_draws.npz"
    else:
        settings = {
            "laplace_d10_w300": (300, 10, False, 0.8),
            "laplace_d10_w150": (150, 10, False, 0.8),
            "laplace_d10_w150_ta90": (150, 10, False, 0.9),
            "laplace_d10_w150_ta95": (150, 10, False, 0.95),
            "laplace_d5_w150": (150, 5, False, 0.8),
            "laplace_d6_w150": (150, 6, False, 0.8),
            "laplace_map_d10_w50": (50, 10, True, 0.8),
        }
        warmup, depth, at_map, target_accept = settings[args.config]
        draws, steps, accept, diverging, step_size, phases = run_laplace(
            problem,
            args.geometry,
            warmup=warmup,
            samples=args.samples,
            max_tree_depth=depth,
            start_at_map=at_map,
            target_accept=target_accept,
            seed=args.seed + 500 + depth + warmup,
        )
        result = summarize_run(
            name=args.config,
            draws=draws,
            steps=steps,
            accept=accept,
            diverging=diverging,
            step_size=step_size,
            wall_seconds=phases["total"],
            phase_seconds=phases,
            latent_names=latent_names,
            reference=reference,
        )
        result["laplace_settings"] = {
            "adapt_mass_matrix": False,
            "adapt_step_size": True,
            "warmup": warmup,
            "max_tree_depth": depth,
            "start_at_map": at_map,
            "target_accept": target_accept,
        }
        draw_path = output_dir / f"{args.config}_draws.npz"

    save_draws(draw_path, draws)
    result.update({"problem": problem.metadata, "draws": str(draw_path)})
    json_dump(output_dir / f"{args.config}.json", jsonable(result))
    print(
        f"wrote {args.config}: wall={result['wall_seconds']:.3f}s "
        f"mean_steps={result['steps']['mean']:.2f} "
        f"ESS/draw={result['ess_per_draw_mean']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

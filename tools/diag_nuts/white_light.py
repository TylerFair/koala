#!/usr/bin/env python3
"""Synthetic 2000-cadence white-light geometry and matched NUTS comparison."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diag_nuts.common import (
    block,
    effective_sample_sizes,
    json_dump,
    jsonable,
    latent_labels,
    make_whitelight_problem,
    posterior_agreement,
    sample_moments,
    summarize_steps,
)


MODES = ("geometry", "production", "laplace", "non_gaussian")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--cadences", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument(
        "--output-dir", default="acceleration_reports/diag_nuts"
    )
    return p


def bounds(problem):
    return {
        "_b_0": (-2.0, 2.0),
        "c": (0.9, 1.1),
        "c1": (0.0, 1.0),
        "c2": (0.001, 1.0),
        "logD_0": (math.log(0.0007), math.log(1.0)),
        "log_jitter": (math.log(1.0e-5), math.log(1.0e-2)),
        "rors_0": (math.sqrt(1.0e-6), math.sqrt(0.5)),
        "t0_0": (float(np.min(problem.t)), float(np.max(problem.t))),
        "v": (-0.1, 0.1),
    }


def constrain_label(problem, label, value):
    low, high = bounds(problem)[label]
    return low + (high - low) * expit(value)


def geometry(problem, output_dir: Path, seed: int):
    info = initialize_model(
        jax.random.PRNGKey(seed + 10),
        problem.model,
        init_strategy=init_to_value(values=problem.init_params),
        model_args=(problem.t, problem.yerr),
        model_kwargs={"y": problem.y, **problem.model_kwargs},
    )
    flat0, unravel = ravel_pytree(info.param_info.z)
    labels = latent_labels(info.param_info.z)
    potential = lambda flat: info.potential_fn(unravel(flat))
    value_grad = jax.jit(jax.value_and_grad(potential))
    hessian_fn = jax.jit(jax.hessian(potential))
    block(value_grad(flat0))
    block(hessian_fn(flat0))

    def scipy_vg(x):
        value, grad = value_grad(jnp.asarray(x))
        return float(value), np.asarray(grad)

    start = time.perf_counter()
    result = minimize(
        scipy_vg,
        np.asarray(flat0),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1.0e-13, "gtol": 1.0e-8},
    )
    newton_iterations = 0
    x_refined = np.asarray(result.x, dtype=np.float64)
    value_refined, grad_refined = scipy_vg(x_refined)
    for _ in range(50):
        if np.max(np.abs(grad_refined)) <= 1.0e-4:
            break
        hess_refined = np.asarray(hessian_fn(jnp.asarray(x_refined)))
        hess_refined = 0.5 * (hess_refined + hess_refined.T)
        min_eig_refined = float(np.min(np.linalg.eigvalsh(hess_refined)))
        if min_eig_refined <= 1.0e-8:
            hess_refined = hess_refined + (
                1.0e-6 - min_eig_refined
            ) * np.eye(hess_refined.shape[0])
        step = -np.linalg.solve(hess_refined, grad_refined)
        norm = np.linalg.norm(step)
        if norm > 0.5:
            step *= 0.5 / norm
        accepted = False
        for power in range(30):
            candidate_x = x_refined + (0.5**power) * step
            candidate_value, candidate_grad = scipy_vg(candidate_x)
            if np.isfinite(candidate_value) and candidate_value < value_refined:
                x_refined = candidate_x
                value_refined = candidate_value
                grad_refined = candidate_grad
                newton_iterations += 1
                accepted = True
                break
        if not accepted:
            break
    if value_refined <= result.fun:
        result.x = x_refined
        result.fun = value_refined
        result.jac = grad_refined
    elapsed = time.perf_counter() - start
    map_flat = np.asarray(result.x)
    hessian = np.asarray(hessian_fn(jnp.asarray(map_flat)))
    hessian = 0.5 * (hessian + hessian.T)
    eigvals, eigvecs = np.linalg.eigh(hessian)
    floor = max(float(np.max(eigvals)) * 1.0e-10, 1.0e-10)
    covariance = (eigvecs * (1.0 / np.maximum(eigvals, floor))) @ eigvecs.T
    covariance = 0.5 * (covariance + covariance.T)
    sd = np.sqrt(np.diag(covariance))
    corr = covariance / np.outer(sd, sd)
    post = info.postprocess_fn(unravel(jnp.asarray(map_flat)))
    physical_map = {
        name: jsonable(np.asarray(value))
        for name, value in post.items()
        if name in set(info.param_info.z) | {"b_0", "duration_0"}
    }
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append(
                {"pair": f"{labels[i]}--{labels[j]}", "correlation": float(corr[i, j])}
            )
    pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)
    np.savez_compressed(
        output_dir / "white_geometry.npz",
        map=map_flat,
        hessian=hessian,
        covariance=covariance,
        labels=np.asarray(labels),
    )
    payload = {
        "problem": problem.metadata,
        "current_pipeline_initialization": (
            "fit_jwst.py lines 5705-5734 run geometry, transit/LD, then all-site "
            "optimx stages; lines 5790-5819 pass soln through init_to_value"
        ),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evals": int(result.nfev),
            "seconds_after_compilation": elapsed,
            "gradient_max_abs": float(np.max(np.abs(result.jac))),
            "newton_refinement_iterations": newton_iterations,
        },
        "labels": labels,
        "physical_map": physical_map,
        "hessian": {
            "eigenvalues": eigvals.tolist(),
            "positive_definite": bool(np.all(eigvals > 0)),
            "condition_number": (
                float(np.max(eigvals) / np.min(eigvals)) if np.min(eigvals) > 0 else None
            ),
            "regularization_floor": floor,
            "correlation_matrix": corr.tolist(),
            "top_correlations": pairs,
            "strong_correlations_abs_ge_0p5": [
                pair for pair in pairs if abs(pair["correlation"]) >= 0.5
            ],
        },
    }
    json_dump(output_dir / "white_geometry.json", jsonable(payload))
    print(
        f"white geometry: success={result.success} cond={payload['hessian']['condition_number']} "
        f"grad={payload['optimizer']['gradient_max_abs']:.3e}",
        flush=True,
    )


def load_map(problem, output_dir: Path):
    with np.load(output_dir / "white_geometry.npz", allow_pickle=False) as archive:
        map_flat = np.asarray(archive["map"])
        covariance = np.asarray(archive["covariance"])
        labels = [str(x) for x in archive["labels"]]
    physical = {
        label: jnp.asarray(constrain_label(problem, label, map_flat[i]))
        for i, label in enumerate(labels)
    }
    return map_flat, covariance, labels, physical


def run_sampler(problem, output_dir: Path, *, mode: str, warmup: int, samples: int, seed: int):
    _, covariance, _, physical_map = load_map(problem, output_dir)
    if mode == "production":
        kernel = NUTS(
            problem.model,
            init_strategy=init_to_value(values=physical_map),
            dense_mass=False,
            regularize_mass_matrix=True,
            target_accept_prob=0.8,
            max_tree_depth=10,
        )
    else:
        kernel = NUTS(
            problem.model,
            init_strategy=init_to_value(values=physical_map),
            inverse_mass_matrix=jnp.asarray(covariance),
            adapt_step_size=True,
            adapt_mass_matrix=False,
            dense_mass=True,
            regularize_mass_matrix=False,
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
    wall = time.perf_counter() - start
    latents = set(physical_map)
    ess = effective_sample_sizes(draws, latents)
    payload = {
        "mode": mode,
        "problem": problem.metadata,
        "warmup": warmup,
        "samples": samples,
        "wall_seconds": wall,
        "steps": summarize_steps(np.asarray(extra["num_steps"])),
        "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "step_size": jsonable(np.asarray(mcmc.last_state.adapt_state.step_size)),
        "ess": ess,
        "ess_per_draw_mean": float(ess["summary"]["mean"] / samples),
        "ess_per_second_sum": float(
            sum(np.sum(np.asarray(value)) for key, value in ess.items() if key != "summary")
            / wall
        ),
    }
    ref_path = output_dir / "white_production_draws.npz"
    if mode == "laplace" and ref_path.is_file():
        with np.load(ref_path, allow_pickle=False) as archive:
            reference = {key: archive[key] for key in archive.files}
        payload["posterior_agreement"] = posterior_agreement(draws, reference, latents)
        with (output_dir / "white_production.json").open("r", encoding="utf-8") as stream:
            import json

            production = json.load(stream)
        payload["equal_ess_comparison"] = {
            "summed_ess_per_second_speedup": (
                payload["ess_per_second_sum"] / production["ess_per_second_sum"]
            ),
            "mean_ess_per_draw_ratio": (
                payload["ess_per_draw_mean"] / production["ess_per_draw_mean"]
            ),
            "mean_steps_reduction_factor": (
                production["steps"]["mean"] / payload["steps"]["mean"]
            ),
            "wall_speedup_at_equal_draw_count": production["wall_seconds"] / wall,
        }
    np.savez_compressed(
        output_dir / f"white_{mode}_draws.npz",
        **{name: np.asarray(value) for name, value in draws.items()},
    )
    json_dump(output_dir / f"white_{mode}.json", jsonable(payload))
    print(
        f"white {mode}: wall={wall:.3f}s mean_steps={payload['steps']['mean']:.2f} "
        f"ESS/draw={payload['ess_per_draw_mean']:.3f}",
        flush=True,
    )


def non_gaussian(problem, output_dir: Path, seed: int):
    map_flat, covariance, labels, _ = load_map(problem, output_dir)
    with np.load(output_dir / "white_production_draws.npz", allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(map_flat, covariance, size=50000)
    probabilities = np.asarray([0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99])
    rows = []
    for index, label in enumerate(labels):
        lap = constrain_label(problem, label, z[:, index])
        ref_name = label
        ref = np.asarray(reference[ref_name]).reshape(-1)
        ref_m = sample_moments(ref)
        lap_m = sample_moments(lap)
        ref_q = np.quantile(ref, probabilities)
        lap_q = np.quantile(lap, probabilities)
        rows.append(
            {
                "site": label,
                "nuts": ref_m,
                "laplace": lap_m,
                "qq_max_abs_sigma": float(
                    np.max(np.abs(ref_q - lap_q)) / max(ref_m["sd"], 1.0e-300)
                ),
            }
        )
    # b is deterministic abs(_b); explicitly quantify it because the sign
    # symmetry is the expected non-Gaussian white-light feature.
    b_lap = np.abs(constrain_label(problem, "_b_0", z[:, labels.index("_b_0")]))
    b_ref = np.asarray(reference["b_0"]).reshape(-1)
    b_ref_m = sample_moments(b_ref)
    b_lap_m = sample_moments(b_lap)
    rows.append(
        {
            "site": "b_0 (deterministic abs)",
            "nuts": b_ref_m,
            "laplace": b_lap_m,
            "qq_max_abs_sigma": float(
                np.max(
                    np.abs(
                        np.quantile(b_ref, probabilities)
                        - np.quantile(b_lap, probabilities)
                    )
                )
                / b_ref_m["sd"]
            ),
        }
    )
    payload = {"laplace_draws": 50000, "rows": rows}
    json_dump(output_dir / "white_non_gaussian.json", jsonable(payload))
    print("wrote white_non_gaussian.json")


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = make_whitelight_problem(args.cadences, args.seed)
    if args.mode == "geometry":
        geometry(problem, output_dir, args.seed)
    elif args.mode == "non_gaussian":
        non_gaussian(problem, output_dir, args.seed + 400)
    else:
        default_warmup = 1000 if args.mode == "production" else 100
        run_sampler(
            problem,
            output_dir,
            mode=args.mode,
            warmup=default_warmup if args.warmup is None else args.warmup,
            samples=args.samples,
            seed=args.seed + (200 if args.mode == "production" else 300),
        )


if __name__ == "__main__":
    main()

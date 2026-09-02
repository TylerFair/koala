#!/usr/bin/env python3
"""MAP/Hessian geometry for each lane of the synthetic spectroscopic problem."""

from __future__ import annotations

import argparse
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
from numpyro.distributions.transforms import biject_to
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diag_nuts.common import (
    PRIOR_WIDTHS,
    block,
    json_dump,
    jsonable,
    latent_labels,
    make_spectro_problem,
    load_real_spectro_problem,
    site_base,
    slice_lane,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=40)
    p.add_argument("--cadences", type=int, default=230)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--reference-channels", type=int, default=None)
    p.add_argument("--negligible-jitter", action="store_true")
    p.add_argument("--stage-input", default=None)
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument(
        "--output",
        default="acceleration_reports/diag_nuts/spectro_geometry.json",
    )
    p.add_argument(
        "--npz",
        default="acceleration_reports/diag_nuts/spectro_geometry.npz",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    problem = (
        load_real_spectro_problem(args.stage_input, args.channels)
        if args.stage_input
        else make_spectro_problem(
            args.channels,
            args.cadences,
            args.seed,
            negligible_jitter=args.negligible_jitter,
            reference_channels=args.reference_channels,
        )
    )
    args.channels = int(problem.yerr.shape[0])
    err0, obs0, kwargs0, init0 = slice_lane(problem, 0)
    info = initialize_model(
        jax.random.PRNGKey(args.seed + 100),
        problem.model,
        init_strategy=init_to_value(values=init0),
        dynamic_args=True,
        model_args=(problem.t, err0),
        model_kwargs={"y": obs0, **kwargs0},
    )
    flat0, unravel = ravel_pytree(info.param_info.z)
    labels = latent_labels(info.param_info.z)
    latent_names = {
        name
        for name, site in info.model_trace.items()
        if site.get("type") == "sample" and not site.get("is_observed", False)
    }
    if len(labels) != flat0.size:
        raise RuntimeError("Latent labels and flattened state disagree.")

    potential_gen = info.potential_fn

    def potential_flat(flat, err, obs, kwargs):
        return potential_gen(problem.t, err, y=obs, **kwargs)(unravel(flat))

    potential_only = jax.jit(potential_flat)
    value_grad = jax.jit(jax.value_and_grad(potential_flat))
    hessian_fn = jax.jit(jax.hessian(potential_flat))

    ordered_latents = sorted(latent_names)

    def constrained_flat(flat, err, obs, kwargs):
        post = info.postprocess_fn(problem.t, err, y=obs, **kwargs)(unravel(flat))
        pieces = [jnp.ravel(post[name]) for name in ordered_latents]
        return jnp.concatenate(pieces)

    physical_jacobian = jax.jit(jax.jacrev(constrained_flat))
    physical_labels = []
    for name in ordered_latents:
        size = int(np.size(np.asarray(info.model_trace[name]["value"])))
        if size == 1:
            physical_labels.append(name)
        else:
            physical_labels.extend(f"{name}[{i}]" for i in range(size))

    def initial_z(init):
        z = {}
        for name in sorted(info.param_info.z):
            site = info.model_trace[name]
            value = init.get(name, site["value"])
            z[name] = biject_to(site["fn"].support).inv(jnp.asarray(value))
        return np.asarray(ravel_pytree(z)[0], dtype=np.float64)

    # Compile the shared-shape differentiated functions before starting the
    # per-channel optimizer accounting.
    block(value_grad(flat0, err0, obs0, kwargs0))
    block(hessian_fn(flat0, err0, obs0, kwargs0))
    block(physical_jacobian(flat0, err0, obs0, kwargs0))

    maps = []
    hessians = []
    covariances = []
    rows = []
    profiles = []
    total_start = time.perf_counter()
    for channel in range(args.channels):
        err, obs, kwargs, init = slice_lane(problem, channel)
        x0 = initial_z(init)

        def scipy_value_grad(x):
            value, grad = value_grad(jnp.asarray(x), err, obs, kwargs)
            return float(value), np.asarray(grad, dtype=np.float64)

        start = time.perf_counter()
        result = minimize(
            scipy_value_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": args.maxiter, "ftol": 1.0e-13, "gtol": 1.0e-8},
        )
        used_trust_exact = False
        trust_iterations = 0
        if np.max(np.abs(result.jac)) > 1.0e-4:
            # L-BFGS line searches can terminate at a piecewise-smooth transit
            # contact boundary.  A local trust-region solve uses the exact
            # Hessian requested by this diagnostic and is much less sensitive
            # to that line-search failure.  Keep it only when it improves U.
            def scipy_hessian(x):
                return np.asarray(
                    hessian_fn(jnp.asarray(x), err, obs, kwargs),
                    dtype=np.float64,
                )

            trust_result = minimize(
                scipy_value_grad,
                np.asarray(result.x, dtype=np.float64),
                method="trust-exact",
                jac=True,
                hess=scipy_hessian,
                options={"maxiter": min(args.maxiter, 200), "gtol": 1.0e-7},
            )
            trust_iterations = int(trust_result.nit)
            if np.isfinite(trust_result.fun) and trust_result.fun <= result.fun:
                result = trust_result
                used_trust_exact = True
        used_newton_refinement = False
        newton_iterations = 0
        if np.max(np.abs(result.jac)) > 1.0e-4:
            # Contact branches can make scipy's quasi-Newton line search stop.
            # Refine locally with damped exact-Hessian Newton steps, accepting
            # only finite downhill points.  This stays in the smooth local
            # branch and cannot replace a valid mode with a worse point.
            x_refined = np.asarray(result.x, dtype=np.float64)
            value_refined, grad_refined = scipy_value_grad(x_refined)
            for _ in range(30):
                if np.max(np.abs(grad_refined)) <= 1.0e-5:
                    break
                hess_refined = np.asarray(
                    hessian_fn(jnp.asarray(x_refined), err, obs, kwargs),
                    dtype=np.float64,
                )
                min_eig_refined = float(
                    np.min(np.linalg.eigvalsh(0.5 * (hess_refined + hess_refined.T)))
                )
                if min_eig_refined <= 1.0e-8:
                    hess_refined = hess_refined + (
                        1.0e-6 - min_eig_refined
                    ) * np.eye(hess_refined.shape[0])
                step = -np.linalg.solve(hess_refined, grad_refined)
                norm = np.linalg.norm(step)
                if norm > 1.0:
                    step = step / norm
                accepted = False
                for power in range(25):
                    alpha = 0.5**power
                    candidate_x = x_refined + alpha * step
                    candidate_value, candidate_grad = scipy_value_grad(candidate_x)
                    if np.isfinite(candidate_value) and candidate_value < value_refined:
                        x_refined = candidate_x
                        value_refined = candidate_value
                        grad_refined = candidate_grad
                        accepted = True
                        used_newton_refinement = True
                        newton_iterations += 1
                        break
                if not accepted:
                    break
            if value_refined <= result.fun:
                result.x = x_refined
                result.fun = value_refined
                result.jac = grad_refined
        elapsed = time.perf_counter() - start
        map_flat = np.asarray(result.x, dtype=np.float64)
        hessian = np.asarray(hessian_fn(jnp.asarray(map_flat), err, obs, kwargs))
        hessian = 0.5 * (hessian + hessian.T)
        eigvals, eigvecs = np.linalg.eigh(hessian)
        max_eig = float(np.max(eigvals))
        floor = max(max_eig * 1.0e-10, 1.0e-10)
        regularized = np.maximum(eigvals, floor)
        covariance = (eigvecs * (1.0 / regularized)) @ eigvecs.T
        covariance = 0.5 * (covariance + covariance.T)
        sd = np.sqrt(np.diag(covariance))
        corr = covariance / np.outer(sd, sd)

        jac = np.asarray(
            physical_jacobian(jnp.asarray(map_flat), err, obs, kwargs)
        )
        physical_cov = jac @ covariance @ jac.T
        physical_sd = np.sqrt(np.maximum(np.diag(physical_cov), 0.0))
        sigma_prior_ratios = {}
        for label, sigma in zip(physical_labels, physical_sd, strict=True):
            width = PRIOR_WIDTHS.get(site_base(label))
            if width is not None:
                sigma_prior_ratios[label] = float(sigma / width)

        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append(
                    {
                        "pair": f"{labels[i]}--{labels[j]}",
                        "correlation": float(corr[i, j]),
                    }
                )
        pairs.sort(key=lambda row: abs(row["correlation"]), reverse=True)

        profile = None
        if "log_jitter" in labels:
            jitter_index = labels.index("log_jitter")
            grid = np.linspace(-12.0, 12.0, 241)
            values = []
            for offset in grid:
                point = map_flat.copy()
                point[jitter_index] += offset
                values.append(float(potential_only(jnp.asarray(point), err, obs, kwargs)))
            delta = np.asarray(values) - np.min(values)
            profile = {
                "channel": channel,
                "offset_grid": grid.tolist(),
                "delta_potential": delta.tolist(),
                "width_delta_u_le_0p5": float(
                    np.ptp(grid[delta <= 0.5]) if np.any(delta <= 0.5) else 0.0
                ),
                "width_delta_u_le_2": float(
                    np.ptp(grid[delta <= 2.0]) if np.any(delta <= 2.0) else 0.0
                ),
                "map_unconstrained": float(map_flat[jitter_index]),
                "local_sigma_unconstrained": float(sd[jitter_index]),
            }
            profiles.append(profile)

        row = {
            "channel": channel,
            "optimizer_success": bool(
                result.success or np.max(np.abs(result.jac)) <= 1.0e-4
            ),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_iterations": int(result.nit),
            "optimizer_function_evals": int(result.nfev),
            "optimizer_used_newton_refinement": used_newton_refinement,
            "optimizer_newton_iterations": newton_iterations,
            "optimizer_used_trust_exact": used_trust_exact,
            "optimizer_trust_iterations": trust_iterations,
            "optimizer_seconds": elapsed,
            "map_potential": float(result.fun),
            "gradient_max_abs": float(np.max(np.abs(result.jac))),
            "hessian_eigenvalue_min": float(np.min(eigvals)),
            "hessian_eigenvalue_max": max_eig,
            "hessian_positive_definite": bool(np.all(eigvals > 0.0)),
            "hessian_condition_number": (
                float(max_eig / np.min(eigvals)) if np.min(eigvals) > 0 else None
            ),
            "hessian_regularization_floor": floor,
            "strong_correlations_abs_ge_0p5": [
                pair for pair in pairs if abs(pair["correlation"]) >= 0.5
            ],
            "top_correlations": pairs[:8],
            "posterior_sigma_over_prior_width": sigma_prior_ratios,
            "jitter_profile": profile,
        }
        print(
            f"channel {channel:02d}: success={result.success} nit={result.nit} "
            f"cond={row['hessian_condition_number']} "
            f"min_eig={row['hessian_eigenvalue_min']:.3e}",
            flush=True,
        )
        maps.append(map_flat)
        hessians.append(hessian)
        covariances.append(covariance)
        rows.append(row)

    map_array = np.stack(maps)
    hessian_array = np.stack(hessians)
    covariance_array = np.stack(covariances)
    npz_path = Path(args.npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        map=map_array,
        hessian=hessian_array,
        covariance=covariance_array,
        labels=np.asarray(labels),
        physical_labels=np.asarray(physical_labels),
    )
    conditions = np.asarray(
        [row["hessian_condition_number"] for row in rows if row["hessian_condition_number"]]
    )
    payload = {
        "problem": problem.metadata,
        "coordinate_system": "NumPyro unconstrained coordinates",
        "labels": labels,
        "physical_labels": physical_labels,
        "channels": rows,
        "summary": {
            "total_seconds_after_compilation": time.perf_counter() - total_start,
            "optimizer_successes": int(sum(row["optimizer_success"] for row in rows)),
            "positive_definite_hessians": int(
                sum(row["hessian_positive_definite"] for row in rows)
            ),
            "condition_number_min": float(np.min(conditions)),
            "condition_number_median": float(np.median(conditions)),
            "condition_number_max": float(np.max(conditions)),
            "npz": str(npz_path),
        },
    }
    json_dump(args.output, jsonable(payload))
    print(f"wrote {args.output} and {args.npz}")


if __name__ == "__main__":
    main()

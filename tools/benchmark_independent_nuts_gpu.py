#!/usr/bin/env python3
"""Compare joint NumPyro NUTS with independently adapted GPU-batched NUTS.

Both backends sample the same synthetic, production-shaped jaxoplanet
power-2 + linear model, including the conservative static transit window.
Timings include compilation, warmup, sampling, and deterministic
postprocessing because that is the wall time relevant to ``fit_jwst.py``.

Example
-------
python tools/benchmark_independent_nuts_gpu.py \
    --platform gpu --channels 40 --cadences 513 --warmup 300 --samples 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import jax

jax.config.update("jax_enable_x64", True)
if __name__ == "__main__":
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == "--platform" and index + 1 < len(sys.argv):
            jax.config.update("jax_platform_name", sys.argv[index + 1])
            break
        if token.startswith("--platform="):
            jax.config.update("jax_platform_name", token.split("=", 1)[1])
            break

import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.diagnostics import effective_sample_size

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.independent_nuts import get_samples_independent
from models.jaxoplanet import build_transit_window_indices, create_vectorized_model


@dataclass(frozen=True)
class BackendResult:
    wall_seconds: float
    mean_steps: float
    median_steps: float
    max_steps: int
    lane_max_mean_steps: float
    divergences: int
    divergence_fraction: float
    mean_accept_prob: float
    rors_ess_mean: float
    rors_ess_min: float
    rors_ess_per_second: float


def _ready(tree):
    return jax.block_until_ready(tree)


def _rors_ess(draws, wall_seconds):
    rors = np.asarray(draws["rors"])[..., 0]
    ess = np.asarray(
        [
            effective_sample_size(jnp.asarray(rors[:, channel])[None, :])
            for channel in range(rors.shape[1])
        ],
        dtype=np.float64,
    )
    return float(ess.mean()), float(ess.min()), float(ess.sum() / wall_seconds)


def make_problem(
    channels: int,
    cadences: int,
    seed_value: int,
    *,
    reference_channels: int | None = None,
):
    """Build a synthetic problem, optionally nested in a wider realization.

    When ``reference_channels`` is supplied, wavelength-dependent truth values
    and noise are generated once at that width and the first ``channels`` are
    retained.  Channels follow a base-2 van der Corput order, so even a short
    prefix spans the modeled wavelength phase instead of sampling only one
    edge.  A resident-width sweep can therefore compare representative exact
    prefixes of one realization instead of changing the dataset at every width.
    """
    channels = int(channels)
    cadences = int(cadences)
    reference_channels = (
        channels if reference_channels is None else int(reference_channels)
    )
    if channels < 1 or cadences < 1:
        raise ValueError("channels and cadences must be positive.")
    if reference_channels < channels:
        raise ValueError("reference_channels must be at least channels.")
    duration = jnp.asarray([0.11693087083333333], dtype=jnp.float64)
    period = jnp.asarray([4.05528043], dtype=jnp.float64)
    t0 = jnp.asarray([0.0], dtype=jnp.float64)
    impact = jnp.asarray([0.4498], dtype=jnp.float64)
    t = jnp.linspace(
        -2.5 * duration[0],
        2.5 * duration[0],
        cadences,
        dtype=jnp.float64,
    )
    window_indices = build_transit_window_indices(
        np.asarray(t),
        np.asarray(period),
        np.asarray(t0),
        np.asarray(duration),
    )
    model = create_vectorized_model(
        detrend_type="linear",
        ld_mode="informed",
        trend_mode="free",
        n_planets=1,
        ld_profile="power2",
        param_method="duration",
        transit_window="auto",
        transit_window_indices=window_indices,
    )

    def van_der_corput(index: int) -> float:
        value = 0.0
        denominator = 1.0
        while index:
            index, remainder = divmod(index, 2)
            denominator *= 2.0
            value += remainder / denominator
        return value

    reference_phase = jnp.asarray(
        [
            2.0 * van_der_corput(index + 1) - 1.0
            for index in range(reference_channels)
        ],
        dtype=jnp.float64,
    )
    wavelength_phase = reference_phase[:channels]
    true_rors = (0.145 + 0.002 * wavelength_phase)[:, None]
    true_c1 = 0.52 + 0.05 * wavelength_phase
    true_c2 = 0.42 - 0.04 * wavelength_phase
    true_c = 1.0 + 2.0e-4 * wavelength_phase
    true_v = 1.5e-3 * wavelength_phase
    true_log_jitter = jnp.log(2.0e-4 + 0.3e-4 * (wavelength_phase + 1.0))
    yerr_per_lc = 5.0e-4 + 0.5e-4 * (wavelength_phase + 1.0)
    yerr = jnp.broadcast_to(yerr_per_lc[:, None], (channels, cadences))
    mu_u_ld = jnp.stack((true_c1, true_c2), axis=1)
    sigma_u_ld = jnp.full((channels, 2), 0.05, dtype=jnp.float64)

    model_kwargs = {
        "mu_duration": duration,
        "mu_t0": t0,
        "mu_b": impact,
        "mu_depths": true_rors**2,
        "PERIOD": period,
        "mu_u_ld": mu_u_ld,
        "sigma_u_ld": sigma_u_ld,
        "precomputed_yerr_per_lc": yerr_per_lc,
    }
    truth_values = {
        "rors": true_rors,
        "c1": true_c1,
        "c2": true_c2,
        "c": true_c,
        "v": true_v,
        "log_jitter": true_log_jitter,
    }
    conditioned = substitute(model, data=truth_values)
    truth_trace = trace(seed(conditioned, jax.random.PRNGKey(seed_value))).get_trace(
        t,
        yerr,
        y=None,
        **model_kwargs,
    )
    mean_flux = truth_trace["obs"]["fn"].loc
    total_error = truth_trace["obs"]["fn"].scale
    noise_key = jax.random.PRNGKey(seed_value + 1)
    # Generate the full reference-width noise array before slicing.  JAX random
    # draws are shape-dependent, so drawing directly at ``channels`` would not
    # make the smaller realization an exact prefix of the larger one.
    reference_noise = jax.random.normal(
        noise_key, (reference_channels, cadences), dtype=mean_flux.dtype
    )
    y = mean_flux + total_error * reference_noise[:channels]

    init_params = {
        "rors": jnp.full_like(true_rors, 0.145),
        "c1": mu_u_ld[:, 0],
        "c2": mu_u_ld[:, 1],
        "c": jnp.ones(channels, dtype=jnp.float64),
        "v": jnp.zeros(channels, dtype=jnp.float64),
        "log_jitter": jnp.log(yerr_per_lc),
    }
    channel_varying = (
        "mu_depths",
        "mu_u_ld",
        "sigma_u_ld",
        "precomputed_yerr_per_lc",
    )
    metadata = {
        "window_cadences": int(len(window_indices)),
        "window_fraction": float(len(window_indices) / cadences),
        "synthetic_reference_channels": reference_channels,
        "synthetic_channel_subset": f"prefix[0:{channels}]",
        "synthetic_nested_realization": True,
        "synthetic_channel_order": "base2_van_der_corput_prefix",
        "synthetic_channel_phase": np.asarray(wavelength_phase).tolist(),
    }
    return model, t, yerr, y, init_params, model_kwargs, channel_varying, metadata


def run_joint(
    model,
    key,
    t,
    yerr,
    y,
    init_params,
    model_kwargs,
    warmup,
    samples,
    max_tree_depth,
):
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_params),
        dense_mass=False,
        regularize_mass_matrix=True,
        target_accept_prob=0.8,
        max_tree_depth=max_tree_depth,
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
        key,
        t,
        yerr,
        y=y,
        extra_fields=("num_steps", "diverging", "accept_prob"),
        **model_kwargs,
    )
    draws = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    _ready((draws, extra))
    elapsed = time.perf_counter() - start

    steps = np.asarray(extra["num_steps"])
    diverging = np.asarray(extra["diverging"])
    accept = np.asarray(extra["accept_prob"])
    ess_mean, ess_min, ess_per_second = _rors_ess(draws, elapsed)
    metrics = BackendResult(
        wall_seconds=float(elapsed),
        mean_steps=float(steps.mean()),
        median_steps=float(np.median(steps)),
        max_steps=int(steps.max()),
        lane_max_mean_steps=float(steps.mean()),
        divergences=int(diverging.sum()),
        divergence_fraction=float(diverging.mean()),
        mean_accept_prob=float(accept.mean()),
        rors_ess_mean=ess_mean,
        rors_ess_min=ess_min,
        rors_ess_per_second=ess_per_second,
    )
    return draws, metrics


def run_independent(
    model,
    key,
    t,
    yerr,
    y,
    init_params,
    model_kwargs,
    channel_varying,
    warmup,
    samples,
    lane_width,
    max_tree_depth,
):
    start = time.perf_counter()
    draws, diagnostics = get_samples_independent(
        model,
        key,
        t,
        yerr,
        y,
        init_params,
        num_warmup=warmup,
        num_samples=samples,
        lane_width=lane_width,
        channel_varying_kwargs=channel_varying,
        dense_mass=True,
        regularize_mass_matrix=True,
        target_accept_prob=0.8,
        max_tree_depth=max_tree_depth,
        return_diagnostics=True,
        **model_kwargs,
    )
    _ready(draws)
    _ready(
        (
            diagnostics.num_steps,
            diagnostics.accept_prob,
            diagnostics.diverging,
            diagnostics.step_size,
        )
    )
    elapsed = time.perf_counter() - start

    steps = np.asarray(diagnostics.num_steps)
    diverging = np.asarray(diagnostics.diverging)
    accept = np.asarray(diagnostics.accept_prob)
    ess_mean, ess_min, ess_per_second = _rors_ess(draws, elapsed)
    metrics = BackendResult(
        wall_seconds=float(elapsed),
        mean_steps=float(steps.mean()),
        median_steps=float(np.median(steps)),
        max_steps=int(steps.max()),
        lane_max_mean_steps=float(steps.max(axis=1).mean()),
        divergences=int(diverging.sum()),
        divergence_fraction=float(diverging.mean()),
        mean_accept_prob=float(accept.mean()),
        rors_ess_mean=ess_mean,
        rors_ess_min=ess_min,
        rors_ess_per_second=ess_per_second,
    )
    return draws, metrics


def posterior_agreement(joint_draws, independent_draws):
    joint_rors = np.asarray(joint_draws["rors"])[..., 0]
    independent_rors = np.asarray(independent_draws["rors"])[..., 0]
    joint_median = np.median(joint_rors, axis=0)
    independent_median = np.median(independent_rors, axis=0)
    pooled_scale = np.sqrt(
        0.5
        * (
            np.var(joint_rors, axis=0, ddof=1)
            + np.var(independent_rors, axis=0, ddof=1)
        )
    )
    delta = independent_median - joint_median
    return {
        "rors_median_max_abs": float(np.max(np.abs(delta))),
        "rors_median_rms": float(np.sqrt(np.mean(delta**2))),
        "rors_median_max_pooled_sd": float(
            np.max(np.abs(delta) / np.maximum(pooled_scale, 1.0e-12))
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--cadences", type=int, default=513)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--lane-width",
        type=int,
        default=None,
        help="Fixed independent GPU width; defaults to --channels.",
    )
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--json", dest="json_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.channels, args.cadences, args.samples) < 1 or args.warmup < 0:
        raise SystemExit("channels/cadences/samples must be positive; warmup >= 0")
    lane_width = args.channels if args.lane_width is None else args.lane_width
    if lane_width < args.channels:
        raise SystemExit("lane-width must be at least channels")

    (
        model,
        t,
        yerr,
        y,
        init_params,
        model_kwargs,
        channel_varying,
        metadata,
    ) = make_problem(args.channels, args.cadences, args.seed)
    key_joint, key_independent = jax.random.split(jax.random.PRNGKey(args.seed + 2))
    joint_draws, joint_metrics = run_joint(
        model,
        key_joint,
        t,
        yerr,
        y,
        init_params,
        model_kwargs,
        args.warmup,
        args.samples,
        args.max_tree_depth,
    )
    independent_draws, independent_metrics = run_independent(
        model,
        key_independent,
        t,
        yerr,
        y,
        init_params,
        model_kwargs,
        channel_varying,
        args.warmup,
        args.samples,
        lane_width,
        args.max_tree_depth,
    )
    agreement = posterior_agreement(joint_draws, independent_draws)
    result = {
        "device": str(jax.devices()[0]),
        "channels": args.channels,
        "cadences": args.cadences,
        "warmup": args.warmup,
        "samples": args.samples,
        "lane_width": lane_width,
        **metadata,
        "joint": asdict(joint_metrics),
        "independent": asdict(independent_metrics),
        "wall_speedup": float(
            joint_metrics.wall_seconds / independent_metrics.wall_seconds
        ),
        "rors_ess_per_second_speedup": float(
            independent_metrics.rors_ess_per_second
            / max(joint_metrics.rors_ess_per_second, 1.0e-300)
        ),
        "posterior_agreement": agreement,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()

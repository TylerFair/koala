#!/usr/bin/env python3
"""Plan and run the six standard limb-darkening variants for one config."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _variant_stage(stage, variant):
    """Convert one structural stage dump to the invariant data-form schema."""
    import jax.numpy as jnp
    from dataclasses import replace

    kwargs = dict(stage.model_kwargs)
    init = dict(stage.init_params)
    n = stage.num_channels
    zeros = jnp.zeros((n, 2), dtype=jnp.float64)
    ones = jnp.ones((n, 2), dtype=jnp.float64)
    if variant.name.startswith("fixed_"):
        if "ld_fixed" in kwargs:
            center = jnp.asarray(kwargs.pop("ld_fixed"))
        elif variant.law == "quadratic" and "ld_uplus_uminus" in init:
            sumdiff = jnp.asarray(init.pop("ld_uplus_uminus"))
            center = jnp.stack(
                ((sumdiff[:, 0] + sumdiff[:, 1]) / 2.0,
                 (sumdiff[:, 0] - sumdiff[:, 1]) / 2.0), axis=-1,
            )
        else:
            center = jnp.stack((init.pop("c1"), init.pop("c2")), axis=-1)
        scale, low, high, code = zeros, zeros, ones, 0
        latent = 0.5 * ones
    elif variant.name == "uniform_power2":
        kwargs.pop("mu_u_ld", None)
        kwargs.pop("sigma_u_ld", None)
        center, scale, low, high, code = zeros, -ones, zeros, ones, 1
        if "ld_decorrelated" in init:
            from models.ld_parameterization import Power2MaxtedTransform
            latent = Power2MaxtedTransform().inv(
                jnp.asarray(init.pop("ld_decorrelated"))
            )
        else:
            coeff = jnp.stack((init.pop("c1"), init.pop("c2")), axis=-1)
            latent = coeff
    elif variant.name == "uniform_quadratic":
        center, scale = zeros, -ones
        low = jnp.broadcast_to(jnp.array([-1.0, -2.0]), (n, 2))
        high = jnp.broadcast_to(jnp.array([2.0, 2.0]), (n, 2))
        code = 2
        latent = jnp.asarray(init.pop("ld_uplus_uminus"))
    elif variant.name == "sing_quadratic":
        center = jnp.asarray(kwargs.pop("mu_u_ld"))
        scale = jnp.asarray(kwargs.pop("sigma_u_ld"))
        low = jnp.broadcast_to(jnp.array([0.0, -1.0]), (n, 2))
        high = ones
        code = 3
        initial_l = jnp.asarray(init.pop("limb_l"))
        initial_delta = jnp.asarray(init.pop("limb_delta"))
        initial_fraction = 0.5 + 2.0 * initial_delta / (1.0 - initial_l)
        latent = jnp.stack((initial_l, initial_fraction), axis=-1)
    else:
        center = jnp.asarray(kwargs.pop("mu_u_ld"))
        scale = jnp.asarray(kwargs.pop("sigma_u_ld"))
        low = jnp.broadcast_to(
            jnp.array([0.0, 0.001] if variant.law == "power2" else [0.0, 0.0]),
            (n, 2),
        )
        high, code = ones, 0
        if variant.law == "power2":
            latent = jnp.stack((init.pop("c1"), init.pop("c2")), axis=-1)
        else:
            latent = jnp.asarray(init.pop("u"))
    init.pop("u", None)
    if variant.name == "sing_quadratic":
        latent_low, latent_high = zeros, ones
    else:
        latent_low, latent_high = low, high
    fraction = jnp.clip(
        (latent - latent_low) / (latent_high - latent_low), 1e-8, 1.0 - 1e-8
    )
    init["ld_variant_latent"] = jnp.log(fraction) - jnp.log1p(-fraction)
    kwargs.update(
        ld_center=center, ld_scale=scale, ld_low=low, ld_high=high,
        ld_latent_low=latent_low, ld_latent_high=latent_high,
        ld_map_code=jnp.asarray(code, dtype=jnp.int32),
    )
    varying = tuple(dict.fromkeys(
        (*stage.channel_varying_kwargs, "ld_center", "ld_scale", "ld_low", "ld_high",
         "ld_latent_low", "ld_latent_high")
    ))
    return replace(stage, init_params=init, model_kwargs=kwargs,
                   channel_varying_kwargs=varying)


def run_spectroscopic_loop(stage_paths, output_dir, *, samples=1000):
    """Run same-shaped stage dumps while retaining one runner per LD law."""
    import jax
    import numpy as np
    import pandas as pd
    from tools.spectro_stage_inputs import load_stage_inputs
    import fit_jwst
    from tools.run_sampler_on_stage_inputs import _CompilationEvents, _arviz_diagnostics

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = {item.name: item for item in LOOP_VARIANTS}
    runners = {"power2": {}, "quadratic": {}}
    common_models = {}
    common_sampler_options = {}
    runner_caches = {"power2": {}, "quadratic": {}}
    joint_runner_caches = {"power2": {}, "quadratic": {}}
    compilation = _CompilationEvents()
    try:
        jax.monitoring.register_event_duration_secs_listener(compilation.listener)
    except ValueError:
        pass
    summary = []
    for name, path in stage_paths:
        variant = variants[name]
        raw = load_stage_inputs(
            path, validate_potential=False,
            builder_overrides={"ld_variant_as_data": True,
                                     "ld_mode": "fixed",
                                     "ld_profile": variant.law},
        )
        if variant.law not in common_models:
            common_models[variant.law] = raw.model
        else:
            from dataclasses import replace
            raw = replace(raw, model=common_models[variant.law])
        stage = _variant_stage(raw, variant)
        mcmc_kwargs = {**stage.mcmc_kwargs, "num_samples": int(samples),
                       "progress_bar": False}
        nuts_kwargs = dict(stage.nuts_kwargs)
        nuts_kwargs.pop("init_strategy", None)
        if variant.law not in common_sampler_options:
            common_sampler_options[variant.law] = (nuts_kwargs, mcmc_kwargs)
        else:
            nuts_kwargs, mcmc_kwargs = common_sampler_options[variant.law]
        stage_name = str(stage.meta.get("stage_kind", "stage"))
        variant_dir = output / name / stage_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        independent_builds_before = sum(
            runner.program_build_count
            for runner in runner_caches[variant.law].values()
            if hasattr(runner, "program_build_count")
        )
        joint_runners_before = len(joint_runner_caches[variant.law])
        mark, started = compilation.mark(), time.perf_counter()
        posterior = fit_jwst.get_samples_chunked(
            stage.model, stage.rng_key, stage.t, stage.yerr, stage.y,
            stage.init_params, stage.chunk_size,
            nuts_kwargs=nuts_kwargs, mcmc_kwargs=mcmc_kwargs,
            output_dir=str(variant_dir), checkpoint_prefix="loop",
            sampler_backend="independent_nuts",
            channel_varying_kwargs=stage.channel_varying_kwargs,
            checkpoint_signature={"stage": stage_name, "variant": name,
                                  "ld_variant_as_data": True},
            spectro_min_depth_ess=400.0, spectro_max_divergences=0,
            adaptive_fallback_model=stage.model,
            adaptive_fallback_init_params=stage.init_params,
            # The observed production swap chain needs at most four adaptive
            # lanes per chunk.  A four-lane joint fallback amortizes one
            # low/high compile without paying for forty nuisance lanes.
            adaptive_fallback_resident_width=4,
            _joint_mcmc_runner_cache=joint_runner_caches[variant.law],
            _independent_mcmc_runner_cache=runner_caches[variant.law],
            **stage.model_kwargs,
        )
        draws = dict(posterior)
        sampler_used = list(getattr(posterior, "sampler_used", None)
                            or ["independent_nuts"] * stage.num_channels)
        elapsed = time.perf_counter() - started
        compile_stats = compilation.since(mark)
        np.savez_compressed(variant_dir / "posterior_samples.npz", **draws)
        depth = np.asarray(draws["depths"])[..., 0]
        wavelength = np.asarray(stage.meta["wavelength"])
        wavelength_err = np.asarray(stage.meta.get(
            "wavelength_err", np.full_like(wavelength, np.nan)
        ))
        depth_median = np.median(depth, axis=0)
        depth_error = np.std(depth, axis=0, ddof=1)
        sample_diagnostics = _arviz_diagnostics(draws, stage.num_channels)
        independent_builds_after = sum(
            runner.program_build_count
            for runner in runner_caches[variant.law].values()
            if hasattr(runner, "program_build_count")
        )
        joint_runners_after = len(joint_runner_caches[variant.law])
        table = pd.DataFrame({
            "wavelength": wavelength,
            "wavelength_err": wavelength_err,
            "depth00": depth_median,
            "depth_err00": depth_error,
            "depth_ppm00": 1e6 * depth_median,
            "depth_err_ppm00": 1e6 * depth_error,
            "sampler_used": sampler_used,
        })
        table.to_csv(variant_dir / "spectrum.csv", index=False)
        row = {"variant": name, "stage": stage.meta.get("stage_kind"),
               "wall_seconds": elapsed, **compile_stats,
               "program_build_count": sum(
                   runner.program_build_count
                   for runner in runner_caches[variant.law].values()
                   if hasattr(runner, "program_build_count")
               ),
               "new_independent_program_builds": (
                   independent_builds_after - independent_builds_before
               ),
               "joint_runner_count": joint_runners_after,
               "new_joint_runner_builds": (
                   joint_runners_after - joint_runners_before
               ),
               "sample_diagnostics": sample_diagnostics,
               "sampler_used_counts": {
                   sampler: sampler_used.count(sampler)
                   for sampler in sorted(set(sampler_used))
               }}
        (variant_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        summary.append(row)
    (output / "loop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@dataclass(frozen=True)
class LoopVariant:
    name: str
    law: str
    prior: str
    offset_source: str | None = None


# Uniform quadratic precedes Sing because its low-resolution posterior is the
# calibration sample for the fitted gray offset.
LOOP_VARIANTS = (
    LoopVariant("fixed_power2", "power2", "fixed"),
    LoopVariant("informed_power2", "power2", "informed"),
    LoopVariant("uniform_power2", "power2", "uniform"),
    LoopVariant("fixed_quadratic", "quadratic", "fixed"),
    LoopVariant("uniform_quadratic", "quadratic", "uniform"),
    LoopVariant("sing_quadratic", "quadratic", "sing", "uniform_quadratic"),
)

SPECTRUM_COLUMNS = (
    "wavelength", "wavelength_err", "depth00", "depth_err00",
    "depth_ppm00", "depth_err_ppm00", "sampler_used",
)


def run_whitelight_variants(model, variant_inputs, *, t, yerr, y,
                            output_dir, rng_key, nuts_kwargs=None,
                            mcmc_kwargs=None):
    """Run one white-light callable repeatedly with dynamic LD arguments.

    ``variant_inputs`` contains ``(name, model_kwargs, init_values)`` tuples.
    The returned posterior-median geometry is also written in the handoff JSON
    consumed by stage-dump preparation.
    """
    import numpy as np
    import numpyro
    from models.jaxoplanet.builder import derive_geometry

    nuts_kwargs = dict(nuts_kwargs or {})
    mcmc_options = {"num_warmup": 1000, "num_samples": 1000,
                    "progress_bar": False, "jit_model_args": True,
                    **dict(mcmc_kwargs or {})}
    sampler = numpyro.infer.NUTS(model, **nuts_kwargs)
    mcmc = numpyro.infer.MCMC(sampler, **mcmc_options)
    results = {}
    for index, (name, model_kwargs, init_values) in enumerate(variant_inputs):
        mcmc.sampler._init_strategy = numpyro.infer.init_to_value(
            values=init_values
        )
        mcmc.run(jax.random.fold_in(rng_key, index), t, yerr, y=y,
                 **model_kwargs)
        samples = mcmc.get_samples(group_by_chain=False)
        period = model_kwargs["prior_params"]["period"]
        geometry = derive_geometry(samples, period)
        medians = {
            key: np.median(np.asarray(jax.device_get(value)), axis=0).tolist()
            for key, value in geometry.items()
        }
        target = Path(output_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        handoff = {"estimator": "posterior_median", "geometry": medians}
        (target / "whitelight_geometry_handoff.json").write_text(
            json.dumps(handoff, indent=2) + "\n", encoding="utf-8"
        )
        results[name] = {"samples": samples, "geometry": medians}
    return results, mcmc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", action="append", default=[],
                        help="In-process stage as VARIANT=/absolute/dump.pkl")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--cache-dir",
        default="/scratch/midway3/tfairnington/jax_cache/ld_loop",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

    if args.stage:
        pairs = []
        for item in args.stage:
            if "=" not in item:
                parser.error("--stage requires VARIANT=PATH")
            pairs.append(tuple(item.split("=", 1)))
        run_spectroscopic_loop(pairs, args.output_dir, samples=args.samples)
        return 0
    if not args.config:
        parser.error("either --stage or --config is required")

    with open(args.config, encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generated = root / "loop_configs"
    generated.mkdir(exist_ok=True)
    plan = []
    for variant in LOOP_VARIANTS:
        cfg = copy.deepcopy(base)
        flags = cfg.setdefault("flags", {})
        flags["ld_profile"] = variant.law
        flags["ld_prior"] = variant.prior
        flags["ld_variant_as_data"] = True
        flags["compile_box"] = True
        flags["jax_compilation_cache_dir"] = str(cache)
        if variant.offset_source:
            flags["ld_loop_offset_source"] = str(root / variant.offset_source)
        cfg["output_dir"] = str(root / variant.name)
        config_path = generated / f"{variant.name}.yaml"
        if config_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite generated loop config: {config_path}"
            )
        with open(config_path, "x", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
        plan.append({"variant": variant.name, "config": str(config_path)})

    with open(root / "loop_plan.json", "x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
    if args.plan_only:
        return 0
    raise RuntimeError(
        "Execution is disabled until fit_jwst exposes an in-process staged API; "
        "running six subprocesses would not satisfy loop-mode compilation reuse."
    )
    # Retained below as the intended handoff once the staged API is available.
    for row in plan:
        subprocess.run(
            [sys.executable, str(Path(__file__).parents[1] / "fit_jwst.py"),
             "-c", row["config"]],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

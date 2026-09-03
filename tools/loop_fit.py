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
    from models.ld_parameterization import Power2MaxtedTransform

    kwargs = dict(stage.model_kwargs)
    init = dict(stage.init_params)
    n = stage.num_channels
    zeros = jnp.zeros((n, 2), dtype=jnp.float64)
    ones = jnp.ones((n, 2), dtype=jnp.float64)
    if variant.name.startswith("fixed_"):
        center = jnp.asarray(kwargs.pop("ld_fixed"))
        scale, low, high, code = zeros, zeros, ones, 0
        latent = zeros
    elif variant.name == "uniform_power2":
        center, scale, low, high, code = zeros, -ones, zeros, ones, 1
        if "ld_decorrelated" in init:
            latent = jnp.asarray(init.pop("ld_decorrelated"))
        else:
            coeff = jnp.stack((init.pop("c1"), init.pop("c2")), axis=-1)
            latent = Power2MaxtedTransform()(coeff)
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
        latent = jnp.stack(
            (init.pop("limb_l"), init.pop("limb_delta")), axis=-1
        )
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
    init["ld_variant_latent"] = latent
    kwargs.update(
        ld_center=center, ld_scale=scale, ld_low=low, ld_high=high,
        ld_map_code=jnp.asarray(code, dtype=jnp.int32),
    )
    varying = tuple(dict.fromkeys(
        (*stage.channel_varying_kwargs, "ld_center", "ld_scale", "ld_low", "ld_high")
    ))
    return replace(stage, init_params=init, model_kwargs=kwargs,
                   channel_varying_kwargs=varying)


def run_spectroscopic_loop(stage_paths, output_dir, *, samples=1000):
    """Run same-shaped stage dumps while retaining one runner per LD law."""
    import jax
    import numpy as np
    import pandas as pd
    from tools.spectro_stage_inputs import load_stage_inputs
    from tools.run_sampler_on_stage_inputs import (
        _CompilationEvents, _chunk_inputs, _run_independent_chunk,
        _concatenate_samples, _diagnostic_arrays, _diagnostic_summary,
        _arviz_diagnostics,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = {item.name: item for item in LOOP_VARIANTS}
    runners = {"power2": {}, "quadratic": {}}
    common_models = {}
    common_sampler_options = {}
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
        chunks, widths, diagnostics_rows = [], [], []
        mark, started = compilation.mark(), time.perf_counter()
        for index, start in enumerate(range(0, stage.num_channels, stage.chunk_size)):
            end = min(start + stage.chunk_size, stage.num_channels)
            chunk = _chunk_inputs(stage, start, end)
            values, diagnostics = _run_independent_chunk(
                "independent_nuts", stage, chunk,
                jax.random.fold_in(stage.rng_key, index), runners[variant.law],
                stage.chunk_size, nuts_kwargs, mcmc_kwargs, {},
            )
            chunks.append(jax.device_get(values)); widths.append(end - start)
            diagnostics_rows.append(_diagnostic_summary(
                _diagnostic_arrays(diagnostics)
            ))
        draws = _concatenate_samples(chunks, widths)
        elapsed = time.perf_counter() - started
        compile_stats = compilation.since(mark)
        stage_name = str(stage.meta.get("stage_kind", "stage"))
        variant_dir = output / name / stage_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        depth = np.asarray(draws["depths"])[..., 0]
        wavelength = np.asarray(stage.meta["wavelength"])
        wavelength_err = np.asarray(stage.meta.get(
            "wavelength_err", np.full_like(wavelength, np.nan)
        ))
        depth_median = np.median(depth, axis=0)
        depth_error = np.std(depth, axis=0, ddof=1)
        sample_diagnostics = _arviz_diagnostics(draws, stage.num_channels)
        table = pd.DataFrame({
            "wavelength": wavelength,
            "wavelength_err": wavelength_err,
            "depth00": depth_median,
            "depth_err00": depth_error,
            "depth_ppm00": 1e6 * depth_median,
            "depth_err_ppm00": 1e6 * depth_error,
            "sampler_used": "independent_nuts",
        })
        table.to_csv(variant_dir / "spectrum.csv", index=False)
        row = {"variant": name, "stage": stage.meta.get("stage_kind"),
               "wall_seconds": elapsed, **compile_stats,
               "program_build_count": sum(r.program_build_count for r in runners[variant.law].values()),
               "sample_diagnostics": sample_diagnostics,
               "chunks": diagnostics_rows}
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

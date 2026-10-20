"""Materialize isolated fits and run them in one Python/JAX process."""
from __future__ import annotations

import copy
import itertools
import json
import os
from pathlib import Path
import time

import yaml


def _component(value):
    value = str(value)
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe grid name: {value!r}")
    return value


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def prepare_grid(base_config, workspace, *, trends=("linear", "quadratic"),
                 laws=("quadratic", "power2"), treatments=("fixed", "uniform"),
                 matrix=None):
    """Create eight candidates by default, or accept an explicit variant matrix.

    Input paths retain their meaning relative to the preparation working
    directory, as in the single-fit CLI. Fits, exports and compile caches are
    isolated from the base configuration's output directory. Model-dependent
    clipping is disabled: use common, preselected exclusions in the base YAML.
    """
    base_path = Path(base_config).resolve()
    base = yaml.safe_load(base_path.read_text())
    spec = yaml.safe_load(Path(matrix).read_text()) if matrix else {}
    variants = spec.get("variants")
    if variants is None:
        if not trends or not laws or not treatments:
            raise ValueError("Grid axes must be nonempty.")
        if set(laws) - {"quadratic", "power2"}:
            raise ValueError("Grid laws must be quadratic or power2.")
        if set(treatments) - {"fixed", "uniform", "stellarprior", "gaussian", "sing"}:
            raise ValueError("Unsupported limb-darkening treatment.")
        variants = [dict(name=f"{prior}_{law}_{trend}", overrides={"flags": {
            "detrending_type": trend, "ld_profile": law, "ld_prior": prior,
        }}) for trend, law, prior in itertools.product(trends, laws, treatments)]
    names = [_component(v["name"]) for v in variants]
    if not names or len(set(names)) != len(names):
        raise ValueError("Grid requires nonempty, unique variant names.")
    # Validate everything before creating a workspace or any partial configs.
    configs = []
    root = Path(workspace).resolve()
    for variant, name in zip(variants, names):
        cfg = copy.deepcopy(base)
        _merge(cfg, variant.get("overrides", {}))
        flags = cfg.setdefault("flags", {})
        law, prior = flags.get('ld_profile', 'quadratic'), flags.get('ld_prior', 'gaussian')
        if law not in {'quadratic', 'power2'}:
            raise ValueError('Grid LD laws must be quadratic or power2.')
        if (prior not in {'fixed', 'uniform', 'stellarprior', 'gaussian', 'sing'}
                or (prior == 'stellarprior' and law != 'power2')
                or (prior == 'sing' and law != 'quadratic')):
            raise ValueError(f'Unsupported prior {prior!r} for {law!r}; use a law-specific matrix.')
        if flags.get("light_curve_model", "transit") != "transit":
            raise ValueError("This initial grid supports transmission spectra only.")
        if flags.get("transit_engine", "jaxoplanet") != "jaxoplanet":
            raise ValueError("This initial grid requires the jaxoplanet engine.")
        if flags.get("trend_inference", "free") == "gaussian_marginalized":
            raise ValueError("Pointwise scoring requires sampled trend coefficients.")
        flags.update(spectro_cadence_reduction="off", plots=flags.get("plots", "minimal"))
        if (flags.get("ld_profile", "quadratic") != "power2"
                or flags.get("ld_prior") not in {"uniform", "gaussian"}):
            flags["ld_parameterization"] = "coefficients"
        flags["analysis_stage"] = "highres"
        flags["need_lowres"] = False
        for binning in ("resolution", "pixels"):
            if isinstance(cfg.get(binning), dict):
                cfg[binning].pop("low", None)
                cfg[binning].pop("reference_grid_lr", None)
        flags["jax_compilation_cache_dir"] = str(root / "compile_cache")
        cfg.setdefault("outlier_clip", {}).update(whitelight_sigma=float("inf"),
                                                  spectroscopic_sigma=float("inf"))
        cfg["path"] = str(Path(cfg.get("path", ".")).resolve())
        cfg["output_dir"] = str(root / "fits" / name)
        configs.append(cfg)
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    rows = []
    for name, cfg in zip(names, configs):
        config = root / "configs" / f"{name}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        rows.append(dict(name=name, config=str(config.relative_to(root)),
                         fit_dir=f"fits/{name}", stage_inputs=f"stage_inputs/{name}"))
    manifest = root / "analysis.yaml"
    manifest.write_text(yaml.safe_dump(dict(schema_version=1,
        dataset=spec.get("dataset", base.get("planet", {}).get("name", "grid")),
        variants=rows, common_clipping="disabled; base exclusions retained"), sort_keys=False))
    return manifest


def resolve_path(manifest, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(manifest).resolve().parent / path).resolve()


def run_grid(manifest, *, runner=None):
    """Run all fits with existing artifact/checkpoint gates and a shared JAX cache.

    Reinvocation lets the pipeline validate and resume its own artifacts; a
    timing record is never treated as proof that a fit can be skipped. Each
    candidate still receives independent warmup and geometry inference.
    """
    production = runner is None
    if production:
        from koala.pipeline import run as runner
    manifest = Path(manifest).resolve()
    spec = yaml.safe_load(manifest.read_text())
    previous = os.environ.get("FIT_JWST_DUMP_SAMPLER_INPUTS")
    timings = []
    if production:
        from .fitting import SweepContext
        sweep = SweepContext(manifest.parent)
    try:
        for variant in spec["variants"]:
            config = resolve_path(manifest, variant["config"])
            stage = resolve_path(manifest, variant["stage_inputs"])
            stage.mkdir(parents=True, exist_ok=True)
            os.environ["FIT_JWST_DUMP_SAMPLER_INPUTS"] = str(stage)
            print(f"Grid fit: {variant['name']}", flush=True)
            start = time.perf_counter()
            options = dict(execution_context=sweep) if production else {}
            cfg = yaml.safe_load(config.read_text())
            runner(cfg, config_path=str(config), **options)
            timings.append(dict(name=variant["name"], seconds=time.perf_counter() - start))
            temporary = manifest.parent / "fit_timings.json.tmp"
            temporary.write_text(json.dumps(timings, indent=2))
            temporary.replace(manifest.parent / "fit_timings.json")
            if production:
                sweep.finish_target(cfg['flags'].get('grid_cache_memory_mb'))
                (manifest.parent / "sweep_cache_stats.json").write_text(json.dumps(sweep.stats, indent=2))
    finally:
        if previous is None:
            os.environ.pop("FIT_JWST_DUMP_SAMPLER_INPUTS", None)
        else:
            os.environ["FIT_JWST_DUMP_SAMPLER_INPUTS"] = previous
    return timings

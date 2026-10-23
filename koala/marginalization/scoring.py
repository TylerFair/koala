"""Resume-safe PSIS scores and model-marginalized transmission products."""
from __future__ import annotations
from koala.output_paths import OutputPaths

import hashlib
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .grid import resolve_path
from .likelihood import (load_stage, checkpoint_files, observation_identity,
                         require_same_observations, valid_mask, channel_stage,
                         channel_samples, CompiledLikelihood, score_loglik,
                         CheckpointPosterior, slice_stage)
from .weights import model_weights, mix_depths


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    """Invalidate score caches when model code or numerical libraries change."""
    import numpyro
    import scipy
    import jaxlib
    from importlib.metadata import version
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.rglob("*.py")) + sorted((root.parent / "models").rglob("*.py"))
    return dict(files={str(p.relative_to(root.parent)): _file_digest(p) for p in files},
                jax=jax.__version__, numpy=np.__version__, numpyro=numpyro.__version__,
                scipy=scipy.__version__, jaxlib=jaxlib.__version__,
                jaxoplanet=version("jaxoplanet"), x64=bool(jax.config.x64_enabled))


def score_fingerprint(stage_file, chunks, code):
    payload = dict(schema=1, code=code, stage=_file_digest(stage_file),
                   checkpoints=[_file_digest(path) for path in chunks])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _save_npz(path, **arrays):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _device_loglik_buffer(draws,times):
    """Use a bounded GPU slab when it fits comfortably in available memory."""
    device = jax.devices()[0]
    if device.platform != 'gpu':
        return None
    stats = device.memory_stats() or {}
    limit = int(stats.get('bytes_limit',0))
    used = int(stats.get('bytes_in_use',0))
    budget = min(2*1024**3,.10*limit,.5*max(0,limit-used-512*1024**2))
    if draws*times*8 > budget:
        return None
    print(f'GPU likelihood scoring: {draws*times*8/1024**2:.1f} MiB resident for one channel.',flush=True)
    return jnp.empty((draws,1,times),dtype=jnp.float64)


@partial(jax.jit,donate_argnums=(0,))
def _append_loglik(buffer,batch,start):
    return jax.lax.dynamic_update_slice(buffer,batch,(start,0,0))


def _score_device_channel(buffer,evaluator,samples,stage,point_batch_size):
    for start,stop,batch in evaluator.batches(samples,stage,device=True):
        buffer = _append_loglik(buffer,batch,start)
    elpd,khat = score_loglik(buffer,point_batch_size=point_batch_size)
    return buffer,elpd,khat


def score_variant(stage, samples, *, draw_batch_size=32, point_batch_size=256, scratch=None):
    """Evaluate one channel at a time with a reusable dynamic-data executable.

    Only [draw, one channel, valid time] is retained in a temporary memmap;
    likelihood tensors for the full spectrum are never written or allocated.
    PSIS batches points, preserving all posterior draws for each observation.
    """
    if isinstance(samples, CheckpointPosterior):
        return _score_checkpoint_variant(stage, samples, draw_batch_size=draw_batch_size,
                                         point_batch_size=point_batch_size, scratch=scratch)
    channels, times = stage.num_channels, int(valid_mask(stage).sum())
    draws = next(iter(samples.values())).shape[0]
    elpd, khat = np.empty((channels, times)), np.empty((channels, times))
    evaluator = CompiledLikelihood(stage, draw_batch_size)
    buffer = _device_loglik_buffer(draws,times)
    if buffer is not None:
        for channel in range(channels):
            buffer,score,diagnostic = _score_device_channel(buffer,evaluator,
                channel_samples(samples,stage,channel),channel_stage(stage,channel),point_batch_size)
            elpd[channel],khat[channel] = score[0],diagnostic[0]
        return elpd,khat
    with tempfile.TemporaryDirectory(dir=scratch, prefix="koala_loglik_") as directory:
        values = np.lib.format.open_memmap(Path(directory) / "channel.npy", mode="w+",
                                           dtype=np.float64, shape=(draws, 1, times))
        try:
            for channel in range(channels):
                one_stage = channel_stage(stage, channel)
                one_samples = channel_samples(samples, stage, channel)
                for start, stop, batch in evaluator.batches(one_samples, one_stage):
                    values[start:stop] = batch
                score, diagnostic = score_loglik(values, point_batch_size=point_batch_size)
                elpd[channel], khat[channel] = score[0], diagnostic[0]
        finally:
            del values
    return elpd, khat


def _score_checkpoint_variant(stage, posterior, *, draw_batch_size, point_batch_size, scratch):
    """Score one disk chunk at a time, reusing one channel likelihood program."""
    channels, times = stage.num_channels, int(valid_mask(stage).sum())
    elpd, khat = np.empty((channels, times)), np.empty((channels, times))
    evaluator = CompiledLikelihood(stage, draw_batch_size)
    buffer = _device_loglik_buffer(posterior.draws,times)
    if buffer is not None:
        for begin,end,samples in posterior.blocks():
            chunk_stage = slice_stage(stage,begin,end)
            for local in range(end-begin):
                buffer,score,diagnostic = _score_device_channel(buffer,evaluator,
                    channel_samples(samples,chunk_stage,local),channel_stage(chunk_stage,local),point_batch_size)
                elpd[begin+local],khat[begin+local] = score[0],diagnostic[0]
        return elpd,khat
    with tempfile.TemporaryDirectory(dir=scratch, prefix='koala_loglik_') as directory:
        values = np.lib.format.open_memmap(Path(directory)/'channel.npy', mode='w+',
            dtype=np.float64, shape=(posterior.draws, 1, times))
        try:
            for begin, end, samples in posterior.blocks():
                chunk_stage = slice_stage(stage, begin, end)
                for local in range(end-begin):
                    one_stage = channel_stage(chunk_stage, local)
                    one_samples = channel_samples(samples, chunk_stage, local)
                    for start, stop, batch in evaluator.batches(one_samples, one_stage):
                        values[start:stop] = batch
                    score, diagnostic = score_loglik(values, point_batch_size=point_batch_size)
                    elpd[begin+local], khat[begin+local] = score[0], diagnostic[0]
        finally:
            del values
    return elpd, khat


def score_grid(manifest, output=None, *, stage_kind="high_resolution", method="stacking",
               pooling="channel", draw_batch_size=32, point_batch_size=256,
               n_out=10000, seed=0, allow_unreliable=False, **weight_options):
    """Score completed fits, gate PSIS reliability, and write an absolute mixture.

    Each cache contains only scores, diagnostics, and depth posteriors. Cache
    identity covers exact exported inputs, posterior files, code and libraries.
    The headline retains between-model absolute depth differences.
    """
    from models.stacking import write_stacked_spectrum
    if not jax.config.x64_enabled:
        raise ValueError("Scoring requires JAX_ENABLE_X64=1.")
    manifest = Path(manifest).resolve()
    spec = yaml.safe_load(manifest.read_text())
    variants = spec["variants"]
    names = [v["name"] for v in variants]
    if not names or len(set(names)) != len(names):
        raise ValueError("Scoring requires nonempty unique model names.")
    output = Path(output).resolve() if output else manifest.parent / "marginalized"
    output.mkdir(parents=True, exist_ok=True)
    paths = OutputPaths(output)
    cache_dir = Path(paths.state_dir) / "scores"
    cache_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    code = code_identity()
    scores, diagnostics, depths, fingerprints, draw_counts, timings = [], [], [], [], [], []
    reference = None
    for variant in variants:
        tick = time.perf_counter()
        files = sorted(resolve_path(manifest, variant["stage_inputs"]).glob(f"*_{stage_kind}_inputs.pkl"))
        if len(files) != 1:
            raise ValueError(f"Expected one {stage_kind} export for {variant['name']}; found {len(files)}.")
        stage = load_stage(files[0])
        if reference is None:
            reference = observation_identity(stage)
        else:
            require_same_observations(reference, stage)
        chunks = checkpoint_files(resolve_path(manifest, variant["fit_dir"]), stage)
        fingerprint = score_fingerprint(files[0], chunks, code)
        cache = cache_dir / f"{fingerprint}.npz"
        reused = cache.exists()
        if reused:
            with np.load(cache, allow_pickle=False) as saved:
                elpd, khat, depth = saved["elpd"], saved["khat"], saved["depth"]
                draws = int(saved["draws"])
        else:
            print(f"Scoring: {variant['name']}", flush=True)
            samples = CheckpointPosterior(chunks, stage)
            draws = samples.draws
            elpd, khat = score_variant(stage, samples, draw_batch_size=draw_batch_size,
                                      point_batch_size=point_batch_size, scratch=cache_dir)
            depth = samples.fill_depth()
            _save_npz(cache, elpd=elpd, khat=khat, depth=depth, draws=draws)
        expected = (stage.num_channels, int(valid_mask(stage).sum()))
        if (elpd.shape != expected or khat.shape != expected or depth.shape != (draws, stage.num_channels)
                or not np.all(np.isfinite(elpd)) or not np.all(np.isfinite(depth))):
            raise ValueError("Invalid reduced score cache.")
        scores.append(elpd)
        diagnostics.append(khat)
        depths.append(depth)
        fingerprints.append(fingerprint)
        draw_counts.append(draws)
        timings.append(dict(name=variant["name"], seconds=time.perf_counter()-tick, cache_reused=reused))
    # Sample-count dependent PSIS threshold (Vehtari et al.), never above 0.7.
    thresholds = np.array([min(.7, 1-1/np.log10(n)) for n in draw_counts])
    khat = np.asarray(diagnostics)
    bad = (~np.isfinite(khat)) | (khat > thresholds[:, None, None])
    report = dict(models=names, method=method, pooling=pooling, fingerprints=fingerprints,
                  psis_thresholds=thresholds.tolist(), psis_bad_points_per_model=bad.sum(axis=(1, 2)).tolist(),
                  reliable=bool(not bad.any()), timings=timings,
                  spectrum="absolute; no achromatic offsets removed",
                  sample_semantics="joint spectrum" if pooling == "global" else "channel marginal samples",
                  geometry="conditional on each fitted white-light handoff unless joint geometry was enabled",
                  validation="pointwise PSIS assumes conditional independence of observed cadences")
    Path(paths.state("scoring_diagnostics.json")).write_text(json.dumps(report, indent=2))
    if bad.any() and not allow_unreliable:
        raise ValueError(f"PSIS reliability gate failed: {report['psis_bad_points_per_model']}; "
                         "diagnostics saved. Refit/validate those observations or explicitly allow exploratory output.")
    weights, weight_draws, weight_info = model_weights(np.asarray(scores), method=method,
        pooling=pooling, seed=seed, **weight_options)
    mixture, summary = mix_depths(depths, weights, n_out=n_out, seed=seed)
    wavelength = reference[-1]
    arrays = dict(wavelength=wavelength, model_names=np.asarray(names), weights=weights,
                  elpd=np.asarray(scores), khat=khat, mixture=mixture,
                  model_depth_median=np.asarray([np.median(x, axis=0) for x in depths]))
    if weight_draws is not None:
        arrays["weight_draws"] = weight_draws
    # Prepare both files before replacing a previous product.
    product_paths, grid = OutputPaths.from_stem(output / stage.meta["stage_label"].removesuffix("_"+stage_kind))
    # A batched grid's stage label is a model name, which must not name the combined spectrum.
    spectrum_path = Path(product_paths.table(grid, "spectrum") if grid else paths.table("marginalized", "spectrum"))
    temporary_csv = spectrum_path.with_name(f".{spectrum_path.name}.{os.getpid()}.tmp")
    try:
        write_stacked_spectrum(temporary_csv, wavelength, summary, names, depths, weights)
        _save_npz(Path(paths.state("posterior.npz")), **arrays)
        temporary_csv.replace(spectrum_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
    report.update(seed=seed, n_out=n_out, weight_diagnostics=weight_info,
                  total_seconds=time.perf_counter()-started,
                  exploratory=bool(bad.any()))
    Path(paths.state("diagnostics.json")).write_text(json.dumps(report, indent=2))
    return spectrum_path

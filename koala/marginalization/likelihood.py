"""Bounded-memory, fixed-shape JIT replay of the fitted pointwise likelihood."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import pickle
import re
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from jax.scipy.special import logsumexp

from models.psis import psis_smooth_log_weights


@dataclass(frozen=True)
class Stage:
    model: object
    t: object
    yerr: object
    y: object
    model_kwargs: dict
    meta: dict

    @property
    def num_channels(self):
        return np.shape(self.y)[0]

    @property
    def num_cadences(self):
        return np.shape(self.y)[1]


def load_stage(path):
    """Replay exact Normal observations, including window/reduction exports.

    Window/reduction factors aggregate Normal likelihoods; turn those algebraic
    shortcuts off to recover individual observations. Preserve LD evaluation,
    geometry, priors and transit-grid/table settings from the fitted model.
    """
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported stacking input schema.")
    spec = payload["model_builder"]
    module, name = spec["identity"].rsplit(".", 1)
    builder = getattr(import_module(module), name)
    kwargs = dict(spec.get("kwargs", {}))
    if kwargs.get("trend_mode") == "gaussian_marginalized":
        raise ValueError("Integrated trends need conditional LOO, not Normal pointwise replay.")
    if kwargs.get("ld_profile") == "quadratic" and kwargs.get("ld_mode") == "uniform":
        kwargs.setdefault("ld_uniform_basis", "coefficients")
    if spec["identity"] == "models.jaxoplanet.builder.create_vectorized_model":
        if kwargs.get("ld_profile") == "power2":
            kwargs.setdefault("power2_evaluation", "polynomial")
        kwargs.update(cadence_reduction="off", transit_window="off", transit_window_indices=None)
    run_kwargs = payload.get('model_kwargs', {})
    if (spec['identity'] == 'models.jaxoplanet.builder.create_vectorized_model'
            and kwargs.get('param_method', 'duration') == 'duration'
            and 'mu_a_rs' not in run_kwargs and 'rors_reference' not in run_kwargs):
        raise ValueError('Historical fixed-T14 export has no reference orbit. Refit/export '
                         'with the wavelength-independent geometry model before replay.')
    model = builder if spec.get("callable_is_model") else builder(**kwargs)
    return Stage(model, payload["t"], payload["yerr"], payload["indiv_y"],
                 dict(payload.get("model_kwargs", {})), payload["meta"])


def checkpoint_files(fit_dir, stage):
    """Select exactly one complete stage-specific checkpoint family."""
    kind = str(stage.meta["stage_kind"]).replace("highres", "high_resolution").replace("lowres", "low_resolution")
    label = stage.meta["stage_label"]
    suffix = "_" + kind
    if not label.endswith(suffix):
        raise ValueError("Stage label and kind do not match.")
    wanted = label[:-len(suffix)]
    families = {}
    for path in (Path(fit_dir) / "chunks").glob("*chunk_*_*.*"):
        match = re.fullmatch(r"(.+)_chunk_(\d+)_(\d+)\.(?:pkl|npz)", path.name)
        if match and (match[1] == wanted or match[1].startswith(wanted + "_")):
            families.setdefault(match[1], []).append((int(match[2]), int(match[3]), path))
    complete = []
    for prefix, entries in families.items():
        manifest = Path(fit_dir) / "chunks" / (prefix + ".manifest.json")
        if manifest.exists():
            signature_kind = json.loads(manifest.read_text()).get("checkpoint_signature", {}).get("stage", "")
            signature_kind = signature_kind.replace("highres", "high_resolution").replace("lowres", "low_resolution")
            if signature_kind and signature_kind != kind:
                continue
        entries.sort()
        if (entries[0][0] != 0 or entries[-1][1] != stage.num_channels
                or any(a[1] != b[0] for a, b in zip(entries, entries[1:]))):
            raise ValueError(f"Incomplete checkpoint family: {prefix}")
        complete.append([item[2] for item in entries])
    if len(complete) != 1:
        raise ValueError(f"Expected one complete checkpoint family, found {len(complete)}.")
    return complete[0]


def load_checkpoint(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved)
    with path.open('rb') as stream:
        value = pickle.load(stream)
    return value.get('samples', value)


def load_samples(paths):
    payloads = []
    for path in paths:
        payloads.append(load_checkpoint(path))
    if len(payloads) == 1:
        return {key: np.asarray(value) for key, value in payloads[0].items()}
    if any(set(p) != set(payloads[0]) for p in payloads):
        raise ValueError("Checkpoint sample sites differ.")
    if {"t0", "b", "duration", "a_rs"}.intersection(payloads[0]):
        raise ValueError("Separate joint-geometry checkpoint chunks cannot form one shared geometry posterior.")
    return {key: np.concatenate([np.asarray(p[key]) for p in payloads], axis=1)
            for key in payloads[0]}


class CheckpointPosterior:
    """Keep one checkpoint resident; retain only the full depth posterior."""
    def __init__(self, paths, stage):
        self.paths, self.stage = list(paths), stage
        if not self.paths:
            raise ValueError('No posterior checkpoints found.')
        self.first = load_checkpoint(self.paths[0])
        if len(self.paths) > 1 and {'t0', 'b', 'duration', 'a_rs'}.intersection(self.first):
            raise ValueError('Separate joint-geometry checkpoint chunks cannot form one shared geometry posterior.')
        radius = np.asarray(self.first['rors'])
        if radius.ndim != 3 or radius.shape[2] != 1:
            raise ValueError('Initial grid scoring requires one planet and [draw, channel, 1] radii.')
        self.draws = radius.shape[0]
        if self.draws < 6:
            raise ValueError('PSIS requires at least six posterior draws.')
        self.depth = np.empty((self.draws, stage.num_channels))
        self.depth_complete = False

    def blocks(self):
        offset = 0
        for index, path in enumerate(self.paths):
            values = self.first if index == 0 else load_checkpoint(path)
            arrays = {key:np.asarray(value) for key,value in values.items()}
            if set(arrays) != set(self.first) or any(v.ndim < 1 or v.shape[0] != self.draws for v in arrays.values()):
                raise ValueError('Checkpoint sample sites or posterior draw counts differ.')
            radius = arrays['rors']
            if radius.ndim != 3 or radius.shape[2] != 1:
                raise ValueError('Initial grid scoring requires one planet and [draw, channel, 1] radii.')
            end = offset+radius.shape[1]
            match = re.fullmatch(r'.+_chunk_(\d+)_(\d+)\.(?:pkl|npz)', Path(path).name)
            if match and (int(match[1]), int(match[2])) != (offset, end):
                raise ValueError('Checkpoint filename range does not match its posterior channels.')
            shared = {'t0', 'b', 'duration', 'a_rs', 'period', 'ecc', 'omega', 'orbit_a_rs'}
            if any(v.ndim > 1 and v.shape[1] != radius.shape[1]
                   for k,v in arrays.items() if k not in shared):
                raise ValueError('Checkpoint posterior channel axes differ.')
            if (end > self.stage.num_channels or not np.isfinite(radius).all()
                    or np.any(radius < 0) or np.any(radius > np.sqrt(np.finfo(float).max))):
                raise ValueError('Invalid posterior radius values or channel count.')
            self.depth[:, offset:end] = radius[..., 0]**2
            yield offset, end, arrays
            offset = end
        if offset != self.stage.num_channels:
            raise ValueError('Posterior channel count does not match the stage.')
        self.depth_complete = True

    def fill_depth(self):
        if not self.depth_complete:
            for _ in self.blocks():
                pass
        return self.depth


def valid_mask(stage):
    mask = np.asarray(stage.model_kwargs.get("likelihood_mask", np.ones(stage.num_cadences)), dtype=bool)
    if mask.shape != (stage.num_cadences,) or not mask.any():
        raise ValueError("Scoring needs a nonempty common one-dimensional cadence mask.")
    return mask


def observation_identity(stage):
    mask = valid_mask(stage)
    return (np.asarray(stage.t)[mask], np.asarray(stage.y)[:, mask],
            np.asarray(stage.yerr)[:, mask], np.asarray(stage.meta["wavelength"]))


def require_same_observations(reference, stage):
    current = observation_identity(stage)
    for name, first, second in zip(("time", "flux", "uncertainty", "wavelength"), reference, current):
        if not np.array_equal(first, second):
            raise ValueError(f"Variant {name} observations differ; use a common mask/extraction before fitting.")


class CompiledLikelihood:
    """One compiled evaluator per fitted stage; batches always have the same shape."""

    def __init__(self, stage, batch_size=32):
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not jax.config.x64_enabled:
            raise ValueError("Likelihood scoring requires JAX_ENABLE_X64=1.")
        self.stage, self.batch_size = stage, int(batch_size)
        valid = jnp.asarray(np.flatnonzero(valid_mask(stage)))
        # Importing all model modules happens before tracing, via load_stage.
        def one(sample, t, yerr, y, kwargs):
            trace = handlers.trace(handlers.seed(handlers.substitute(stage.model, data=sample), 0)).get_trace(
                t, yerr, y=y, **kwargs)
            observed = [s for name, s in trace.items() if s.get("type") == "sample"
                        and s.get("is_observed")
                        and name not in {"ld_variant_prior", "ld_decorrelated_support",
                                         "ld_maxted_uniform_prior", "ld_maxted_uniform_support"}]
            latent = {name for name, site in trace.items()
                      if site.get("type") == "sample" and not site.get("is_observed")}
            missing = latent.difference(sample)
            if missing:
                raise ValueError(f"Posterior is missing sampled sites: {sorted(missing)}")
            if len(observed) != 1:
                raise ValueError("Scoring requires one unaggregated observed likelihood site.")
            site = observed[0]
            values = site["fn"].log_prob(site["value"])
            if site.get("mask") is not None:
                values = jnp.where(site["mask"], values, 0.0)
            if values.shape != y.shape:
                raise ValueError("Observed likelihood is not pointwise [channel, time].")
            return values[:, valid]
        self.evaluate = jax.jit(jax.vmap(one, in_axes=(0, None, None, None, None)))

    def batches(self, samples, stage=None, *, device=False):
        stage = self.stage if stage is None else stage
        arrays = {key: np.asarray(value) for key, value in samples.items()}
        sizes = {value.shape[0] for value in arrays.values() if value.ndim}
        if not arrays or len(sizes) != 1 or any(not value.ndim for value in arrays.values()):
            raise ValueError("All posterior sites need the same nonempty draw axis.")
        size = sizes.pop()
        if size < 1:
            raise ValueError("Posterior draw axis is empty.")
        for start in range(0, size, self.batch_size):
            stop = min(size, start + self.batch_size)
            batch = {key: np.pad(value[start:stop], [(0, self.batch_size - (stop-start))]
                                  + [(0, 0)] * (value.ndim - 1), mode="edge")
                     for key, value in arrays.items()}
            values = self.evaluate(batch, stage.t, stage.yerr, stage.y,stage.model_kwargs)[:stop-start]
            if device:
                finite = bool(jnp.all(jnp.isfinite(values)))
            else:
                values = np.asarray(values,dtype=np.float64)
                finite = np.all(np.isfinite(values))
            if not finite:
                raise ValueError("Nonfinite pointwise likelihood; refusing to score.")
            yield start, stop, values


def channel_stage(stage, channel):
    return slice_stage(stage, channel, channel+1)


def slice_stage(stage, begin, end):
    """Slice only the production registry of channel-varying arguments."""
    from koala.constants import JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
    kwargs = dict(stage.model_kwargs)
    varying = getattr(stage.model, '_koala_channel_varying_kwargs',
                      JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS)
    for key in varying:
        if key in kwargs and kwargs[key] is not None:
            array = np.asarray(kwargs[key])
            if array.ndim and array.shape[0] == stage.num_channels:
                kwargs[key] = array[begin:end]
    return Stage(stage.model, stage.t, np.asarray(stage.yerr)[begin:end],
                 np.asarray(stage.y)[begin:end], kwargs, stage.meta)


def channel_samples(samples, stage, channel):
    # Geometry is shared by joint models, and is never a wavelength axis.
    shared = {"t0", "b", "duration", "a_rs", "period", "ecc", "omega", "orbit_a_rs"}
    return {key: value[:, channel:channel+1] if key not in shared and value.ndim > 1
            and value.shape[1] == stage.num_channels else value
            for key, value in samples.items()}


@jax.jit
def _psis_points(values):
    """Input [point, draw]; JIT once per shape, including the PSIS sort."""
    smoothed, khat = psis_smooth_log_weights(-values)
    return logsumexp(values + smoothed, axis=-1), khat


def score_loglik(values, *, point_batch_size=256):
    """PSIS with bounded working memory and a fixed padded point batch."""
    device = isinstance(values,jax.Array)
    if not device:
        values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] < 6 or min(values.shape[1:]) < 1:
        raise ValueError("PSIS needs [draw>=6, channel, time] likelihoods.")
    if point_batch_size < 1:
        raise ValueError("point_batch_size must be positive.")
    channels, times = values.shape[1:]
    # Flatten channel/time only; do not copy the full draw tensor to float64.
    elpd, khat = np.empty(channels * times), np.empty(channels * times)
    for start in range(0, channels * times, point_batch_size):
        stop = min(channels * times, start + point_batch_size)
        indices = np.arange(start, stop)
        if device:
            block = values[:,jnp.asarray(indices // times),jnp.asarray(indices % times)].T
            finite = bool(jnp.all(jnp.isfinite(block)))
            block = jnp.pad(block,[(0,point_batch_size-len(block)),(0,0)],mode='edge')
        else:
            block = np.asarray(values[:,indices // times,indices % times].T,dtype=np.float64)
            finite = np.all(np.isfinite(block))
            block = np.pad(block,[(0,point_batch_size-len(block)),(0,0)],mode='edge')
        if not finite:
            raise ValueError("Nonfinite log likelihood.")
        score, diagnostic = _psis_points(block)
        elpd[start:stop], khat[start:stop] = np.asarray(score)[:stop-start], np.asarray(diagnostic)[:stop-start]
    return elpd.reshape(channels, times), khat.reshape(channels, times)

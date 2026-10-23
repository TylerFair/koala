"""Sampling helpers."""

from koala.console import debug, info, quiet_sampling, sampling_options, channel_batches, single_batch_progress, sampling_history_notes

import os
import dataclasses
import json
import pickle
import hashlib
import uuid
import re
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from models.common import _to_f64, _tree_to_f64
from .constants import (
    CHUNK_CHECKPOINT_SCHEMA_VERSION,
    CHUNK_CHECKPOINT_TARGET_REVISION,
    SAMPLING_WORKLOAD_SCHEMA_VERSION,
)
from .constants import _JAXOPLANET_STATIC_EVAL_KEYS, _SURFACE_STATIC_EVAL_KEYS
from .data import _pad_spectro_cadences_exact
from .artifacts import _update_checkpoint_hash

from .outputs import _save_mcmc_diagnostics
from models.mcmc_diagnostics import PosteriorSamples, pool_chains, rank_diagnostics

def _build_numpyro_mcmc(model, init_params, nuts_kwargs, mcmc_kwargs):
    """Build one shape-specialized MCMC runner.

    NumPyro's ``jit_model_args=True`` path makes model values dynamic while
    array shapes remain static.  The same runner can therefore be reused for
    equal-width spectroscopic chunks: every call still performs an independent
    warmup/adaptation, but the expensive NUTS transition program is compiled
    only once for that resident GPU shape.
    """
    nuts_defaults = {
        "regularize_mass_matrix": False,
        "init_strategy": numpyro.infer.init_to_value(values=init_params),
        "target_accept_prob": 0.9,
    }
    nuts_defaults.update(nuts_kwargs)

    mcmc_defaults = {
        "num_warmup": 1000,
        "num_samples": 1000,
        "progress_bar": True,
        "jit_model_args": True,
    }
    mcmc_defaults.update(sampling_options(mcmc_kwargs))
    return (
        numpyro.infer.MCMC(numpyro.infer.NUTS(model, **nuts_defaults), **mcmc_defaults),
        nuts_defaults,
        mcmc_defaults,
    )


def get_samples(model, key, t, yerr, indiv_y, init_params, nuts_kwargs=None,
                mcmc_kwargs=None, diagnostics_path=None, _mcmc_runner=None,
                **model_kwargs):
    t = _to_f64(t)
    yerr = _to_f64(yerr)
    indiv_y = _to_f64(indiv_y)
    init_params = _tree_to_f64(init_params)
    model_kwargs = _tree_to_f64(model_kwargs)
    nuts_kwargs = dict(nuts_kwargs or {})
    mcmc_kwargs = dict(mcmc_kwargs or {})

    if _mcmc_runner is None:
        mcmc, nuts_defaults, _ = _build_numpyro_mcmc(
            model, init_params, nuts_kwargs, mcmc_kwargs
        )
    else:
        mcmc = _mcmc_runner
        # The compiled transition program is reusable, but each wavelength
        # block must still start from its own physical initial values. NumPyro
        # stores this strategy on the kernel rather than accepting constrained
        # values in MCMC.run().
        # Match the fresh-runner path exactly: an explicitly supplied
        # strategy takes precedence, otherwise use this block's physical
        # initial values.  ``NUTS.init`` calls ``initialize_model`` on every
        # ``MCMC.run``, so replacing the strategy here does not reuse the
        # previous block's initial state or adaptation.
        mcmc.sampler._init_strategy = nuts_kwargs.get(
            'init_strategy', numpyro.infer.init_to_value(values=init_params),
        )
        nuts_defaults = {'regularize_mass_matrix': False, 'target_accept_prob': 0.9}
        nuts_defaults.update(nuts_kwargs)
    mcmc.run(
        key, t, yerr, y=indiv_y,
        extra_fields=('diverging', 'accept_prob', 'potential_energy', 'num_steps'),
        **model_kwargs,
    )
    _save_mcmc_diagnostics(
        mcmc, max_tree_depth=nuts_defaults.get('max_tree_depth'), output_path=diagnostics_path,
    )
    chains = int(mcmc.num_chains)
    values = pool_chains(mcmc.get_samples(group_by_chain=True))
    names = tuple(mcmc.last_state.z)
    return PosteriorSamples(values, num_chains=chains, sampled_sites=names)


def _slice_by_channel(value, sl, num_lcs):
    if value is None:
        return None
    if hasattr(value, "baseline") and (
        hasattr(value, "differences") or hasattr(value, "uniform")
    ):
        if np.ndim(value.baseline) <= 1:
            return value
        return type(value)(*(None if field is None else field[sl] for field in value))
    if isinstance(value, (np.ndarray, jnp.ndarray)) and value.ndim > 0 and value.shape[0] == num_lcs:
        return value[sl]
    return value


def _take_by_channel(value, channel_indices, num_lcs):
    """Take arbitrary wavelength channels from a channel-varying value."""
    if value is None:
        return None
    if hasattr(value, "baseline") and (
        hasattr(value, "differences") or hasattr(value, "uniform")
    ):
        if np.ndim(value.baseline) <= 1:
            return value
        indices = jnp.asarray(channel_indices, dtype=jnp.int32)
        return type(value)(*(None if field is None else field[indices] for field in value))
    if (
        isinstance(value, (np.ndarray, jnp.ndarray))
        and value.ndim > 0
        and value.shape[0] == num_lcs
    ):
        return value[jnp.asarray(channel_indices, dtype=jnp.int32)]
    return value


def _callable_identity(value):
    return (
        f"{getattr(value, '__module__', type(value).__module__)}."
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


def _tree_to_numpy_for_pickle(value):
    """Copy JAX/NumPy pytrees to host values without changing structure."""
    if isinstance(value, dict):
        return {
            key: _tree_to_numpy_for_pickle(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_tree_to_numpy_for_pickle(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_numpy_for_pickle(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if callable(value):
        return value
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError):
        return value
    if array.ndim == 0:
        return array.item()
    return array


def _build_spectroscopic_model(builder, **builder_kwargs):
    """Retain builder arguments so checkpoint identity includes every prior."""
    model = builder(**builder_kwargs)
    model.__sampler_input_builder__ = {
        "identity": _callable_identity(builder),
        "kwargs": _tree_to_numpy_for_pickle(builder_kwargs),
        "callable_is_model": False,
    }
    return model


def _sampler_model_builder_spec(model):
    spec = getattr(model, "__sampler_input_builder__", None)
    if spec is not None:
        return _tree_to_numpy_for_pickle(spec)
    # This fallback supports importable model callables in focused tests and
    # external users. Production spectroscopic models always take the builder
    # path above, so their construction kwargs are never inferred.
    return {'identity': _callable_identity(model), 'kwargs': {}, 'callable_is_model': True}


def _save_stacking_inputs(model, t, yerr, y, model_kwargs, metadata):
    """Export the likelihood inputs needed by the user-facing stacking tools."""
    directory = os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
    if not directory or metadata is None:
        return
    os.makedirs(directory, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", metadata["stage_label"])
    path = os.path.join(directory, f"{label}_inputs.pkl")
    payload = {
        "schema_version": 1,
        "model_builder": _sampler_model_builder_spec(model),
        "t": t, "yerr": yerr, "indiv_y": y,
        "model_kwargs": model_kwargs,
        "meta": {**metadata, "num_channels": int(y.shape[0])},
    }
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "wb") as stream:
        host_payload = jax.tree_util.tree_map(
            lambda value: np.asarray(value) if isinstance(value, jax.Array) else value,
            payload,
        )
        pickle.dump(host_payload, stream)
    os.replace(temporary, path)


def _chunk_checkpoint_fingerprint(
    model, rng_key, t, yerr, indiv_y, init_params, *, chunk_size, sampler_backend, nuts_kwargs,
    mcmc_kwargs, channel_varying_kwargs, checkpoint_signature, model_kwargs,
    spectro_min_depth_ess=0.0, spectro_max_divergences=0,
    adaptive_fallback_resident_width=None,
):
    """Fingerprint the posterior target, inputs, and sampling configuration."""
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": CHUNK_CHECKPOINT_SCHEMA_VERSION,
            "target_revision": CHUNK_CHECKPOINT_TARGET_REVISION,
            "model": (
                f"{getattr(model, '__module__', type(model).__module__)}."
                f"{getattr(model, '__qualname__', type(model).__qualname__)}"
            ),
            # Builder keyword arguments (prior choices such as jitter_prior)
            # must participate so checkpoints from a different model
            # configuration are never reused.
            "model_builder": _sampler_model_builder_spec(model),
            "checkpoint_signature": checkpoint_signature,
            "rng_key": rng_key,
            "chunk_size": int(chunk_size),
            "sampler_backend": str(sampler_backend),
            "nuts_kwargs": dict(nuts_kwargs or {}),
            "mcmc_kwargs": dict(mcmc_kwargs or {}),
            "channel_varying_kwargs": tuple(channel_varying_kwargs),
            "t": t,
            "yerr": yerr,
            "y": indiv_y,
            "init_params": init_params,
            "model_kwargs": model_kwargs,
            "spectro_min_depth_ess": float(spectro_min_depth_ess),
            "spectro_max_divergences": int(spectro_max_divergences),
            "adaptive_fallback_resident_width": (
                None if adaptive_fallback_resident_width is None else
                int(adaptive_fallback_resident_width)
            ),
        },
    )
    return hasher.hexdigest()


def _sampling_workload_fingerprint(
    model, t, yerr, indiv_y, init_params, *, sampler_backend, nuts_kwargs,
    channel_varying_kwargs, checkpoint_signature, model_kwargs,
):
    """Fingerprint the science target independently of draws and batching.

    A short pilot and a production run may intentionally use different random
    keys, warmup/sample counts, or resident widths.  They may not silently use
    different data, priors, model code, or transition-kernel controls.
    """

    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    relative_sources = (
        "koala/pipeline.py",
        "koala/white_light.py",
        "koala/spectroscopy.py",
        "koala/sampling.py",
        "models/channel_batching.py",
        "models/mcmc_diagnostics.py",
        "models/independent_nuts.py",
        "models/independent_hmc.py",
        "models/trend_marginal.py",
        "models/linear_marginalization.py",
        "models/trends.py",
        "models/jaxoplanet/builder.py",
        "models/jaxoplanet/core.py",
        "models/jaxoplanet/limb_dark_streamed.py",
    )
    source_sha256 = {}
    for relative_path in relative_sources:
        path = os.path.join(repository_root, relative_path)
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                source_sha256[relative_path] = hashlib.sha256(stream.read()).hexdigest()

    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema_version": SAMPLING_WORKLOAD_SCHEMA_VERSION,
            "checkpoint_target_revision": CHUNK_CHECKPOINT_TARGET_REVISION,
            "model": (
                f"{getattr(model, '__module__', type(model).__module__)}."
                f"{getattr(model, '__qualname__', type(model).__qualname__)}"
            ),
            "sampler_backend": str(sampler_backend),
            "nuts_kwargs": dict(nuts_kwargs or {}),
            "channel_varying_kwargs": tuple(channel_varying_kwargs),
            "checkpoint_signature": checkpoint_signature,
            "t": t,
            "yerr": yerr,
            "y": indiv_y,
            "init_params": init_params,
            "model_kwargs": model_kwargs,
            "source_sha256": source_sha256,
        },
    )
    return hasher.hexdigest()


def _annotate_sampler_diagnostics(
    path, *, sampling_workload_fingerprint, sampler_backend,
):
    """Atomically bind a sampler diagnostic to its exact science workload."""

    if path is None or sampling_workload_fingerprint is None:
        return
    if not os.path.isfile(path):
        raise RuntimeError(f'Sampler did not write its requested diagnostics file: {path}')
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload.update(
        {
            "diagnostic_provenance_schema_version": 1,
            "sampling_workload_fingerprint_sha256": str(sampling_workload_fingerprint),
            "sampler_backend": str(sampler_backend),
        }
    )
    def _strict_json_value(value):
        """Represent optional non-finite diagnostics as JSON null."""
        if isinstance(value, dict):
            return {key: _strict_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_strict_json_value(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    payload = _strict_json_value(payload)
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _write_or_validate_checkpoint_manifest(
    checkpoint_dir, logical_prefix, effective_prefix, fingerprint, checkpoint_signature,
):
    manifest_path = os.path.join(checkpoint_dir, f'{effective_prefix}.manifest.json')
    payload = {
        "schema_version": CHUNK_CHECKPOINT_SCHEMA_VERSION,
        "logical_prefix": logical_prefix,
        "effective_prefix": effective_prefix,
        "fingerprint_sha256": fingerprint,
        "checkpoint_signature": checkpoint_signature,
    }
    # Signatures intentionally include the complete surface-model contract.
    # Parsed geometry and emission values can be NumPy/JAX arrays, while the
    # manifest is portable JSON rather than a pickle.
    def _json_compatible(value):
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return _json_compatible(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {
                str(key): _json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_json_compatible(item) for item in value]
        if isinstance(value, np.ndarray):
            return _json_compatible(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        try:
            array = np.asarray(jax.device_get(value))
        except Exception:
            return value
        if array.ndim:
            return _json_compatible(array.tolist())
        return array.item()

    payload = _json_compatible(payload)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != payload:
            raise RuntimeError(
                f'Chunk checkpoint manifest does not match the current fit: {manifest_path}',
            )
        return manifest_path

    temporary = f"{manifest_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return manifest_path


def _chunk_ranges_for(num_lcs, chunk_size):
    return [
        (start, min(start + chunk_size, num_lcs))
        for start in range(0, num_lcs, chunk_size)
    ]


def _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end):
    return os.path.join(checkpoint_dir, f"{checkpoint_prefix}_chunk_{start}_{end}.pkl")


_SamplerSamples = PosteriorSamples


def _load_chunk_samples(chunk_file):
    with open(chunk_file, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "samples" in payload and "sampler_used" in payload:
        return _SamplerSamples(payload["samples"], sampler_used=payload["sampler_used"],
                               num_chains=payload.get("num_chains", 1), sampled_sites=payload.get("sampled_sites"))
    return payload


def _concatenate_chunk_samples(samples_chunks):
    samples = {}
    for key in samples_chunks[0].keys():
        arrays = [chunk[key] for chunk in samples_chunks]
        samples[key] = np.concatenate(arrays, axis=1)
    sampler_used = []
    for chunk in samples_chunks:
        provenance = getattr(chunk, "sampler_used", None)
        if provenance is None:
            sampler_used = []
            break
        sampler_used.extend(provenance)
    chains = {getattr(chunk, "num_chains", 1) for chunk in samples_chunks}
    if len(chains) != 1:
        raise ValueError("Posterior chunks have inconsistent chain counts.")
    site_sets = [getattr(chunk, "sampled_sites", None) for chunk in samples_chunks]
    sampled_sites = None
    if all(sites is not None for sites in site_sets):
        sampled_sites = tuple(sorted(set.intersection(*(set(sites) for sites in site_sets))))
    return _SamplerSamples(samples, sampler_used=sampler_used or None,
                           num_chains=chains.pop(), sampled_sites=sampled_sites)


def _spectro_failed_lanes(samples, diagnostics_path, min_depth_ess, max_divergences,
                          surface_science_sites=None):
    """Return failed lanes and auditable gate metrics for an exact-MCMC run.

    Only the science sites are gated: the transit depth (``rors**2``), or the
    free surface sites (eclipse depth, phase amplitudes, spot contrast) when the
    depth is fixed. In spectroscopy the geometry is fixed or shared, so no
    per-channel geometry site exists. A lane fails when a gated site has bulk
    or tail ESS below ``min_depth_ess``, rank-normalised split R-hat >= 1.01
    (multi-chain runs), a single chain's bulk ESS below
    ``min_depth_ess / num_chains``, or more than ``max_divergences``
    divergences.

    Limb-darkening, trend and jitter coordinates are nuisance parameters:
    their ESS and R-hat are reported under ``nuisance_*`` for auditing but do
    not trigger continuation or a sampler swap. A nuisance site at a prior
    boundary (for example Kipping ``q`` near 0 or 1) mixes slowly without
    affecting the depth marginal, which is what the gate certifies.
    """
    depth_values = samples.get("depths")
    if depth_values is None and "rors" in samples:
        depth_values = jnp.asarray(samples["rors"]) ** 2
    if depth_values is None:
        return np.asarray([], dtype=int), {"depth_ess_per_channel": None,
                                           "num_divergences_per_channel": None}
    depth_values = np.asarray(jax.device_get(depth_values))
    geometry_fixed = "_geometry_fixed" in samples
    science_values = {} if geometry_fixed else {"depths": depth_values}
    for name, value in samples.items():
        if re.fullmatch(
            r"_(?:eclipse_depth|phase_c1|phase_d1|stellar_spot_contrast)_\d+",
            name,
        ) and (surface_science_sites is None or name in surface_science_sites):
            science_values[name.removeprefix("_")] = np.asarray(jax.device_get(value))
    if surface_science_sites is not None:
        for name in surface_science_sites:
            if name not in samples:
                science_values[name.removeprefix('_')] = np.full(depth_values.shape[:2], np.nan)
            elif name.removeprefix('_') not in science_values:
                science_values[name.removeprefix('_')] = np.asarray(jax.device_get(samples[name]))
    chains = getattr(samples, 'num_chains', 1)
    num_lanes = depth_values.shape[1]

    def lane_metrics(values_by_name):
        """Per-lane bulk/tail ESS and R-hat (multi-chain) or ESS (one chain)."""
        bulk_by_name, tail_by_name, rhat_by_name = {}, {}, {}
        if chains > 1:
            grouped = {name: value.reshape((-1, chains) + value.shape[1:]).swapaxes(0, 1)
                       for name, value in values_by_name.items()}
            modern = rank_diagnostics(grouped) if grouped else None
            for name in values_by_name:
                def reduce_lanes(metric, maximum=False):
                    value = np.asarray(modern[metric][name]).reshape(num_lanes, -1)
                    return (np.max(value, axis=1) if maximum else np.min(value, axis=1))
                bulk_by_name[name] = reduce_lanes('bulk_ess')
                tail_by_name[name] = reduce_lanes('tail_ess')
                rhat_by_name[name] = reduce_lanes('rhat', True)
            return bulk_by_name, tail_by_name, rhat_by_name
        for name, values in values_by_name.items():
            per_lane = []
            for lane in range(num_lanes):
                value = jnp.asarray(values[:, lane])
                if bool(jnp.all(jnp.isfinite(value))) and bool(jnp.all(value == value[0])):
                    per_lane.append(0.0)
                    continue
                ess = numpyro.diagnostics.effective_sample_size(value[None, ...])
                ess_values = np.asarray(jax.device_get(ess), dtype=float)
                per_lane.append(
                    float(np.min(ess_values)) if np.all(np.isfinite(ess_values)) else np.nan,
                )
            bulk_by_name[name] = np.asarray(per_lane, dtype=float)
        return bulk_by_name, tail_by_name, rhat_by_name

    science_bulk, science_tail_arrays, science_rhat_arrays = lane_metrics(science_values)
    lane_ess = np.full(num_lanes, np.inf, dtype=float)
    lane_tail = np.full(num_lanes, np.inf, dtype=float)
    lane_rhat = np.ones(num_lanes, dtype=float)
    for name in science_values:
        lane_ess = np.minimum(lane_ess, science_bulk[name])
        if name in science_tail_arrays:
            lane_tail = np.minimum(lane_tail, science_tail_arrays[name])
            lane_rhat = np.maximum(lane_rhat, science_rhat_arrays[name])
    # Every chain must contribute: per-chain bulk ESS of each gated site of at
    # least min_depth_ess / num_chains (100 for the 400 / 4 default).
    lane_chain_ess = np.full(num_lanes, np.inf, dtype=float)
    science_chain_ess = {}
    if chains > 1 and science_values:
        for name, value in science_values.items():
            grouped = value.reshape((-1, chains) + value.shape[1:]).swapaxes(0, 1)
            per_chain = np.stack([
                np.asarray(rank_diagnostics({name: grouped[chain:chain + 1]})['bulk_ess'][name])
                .reshape(num_lanes, -1).min(axis=1)
                for chain in range(chains)
            ])
            science_chain_ess[name] = per_chain.min(axis=0)
            lane_chain_ess = np.minimum(lane_chain_ess, science_chain_ess[name])

    # Nuisance diagnostics: reported, never gated.
    nuisance_values = {}
    sampled_sites = getattr(samples, 'sampled_sites', None)
    if chains > 1 and sampled_sites is not None:
        for name in sorted((set(sampled_sites) - {'rors'}) & samples.keys()):
            nuisance_values[name] = np.asarray(jax.device_get(samples[name]))
    nuisance_bulk, nuisance_tail, nuisance_rhat = lane_metrics(nuisance_values)

    divergences = np.zeros(num_lanes, dtype=int)
    if diagnostics_path is not None and os.path.isfile(diagnostics_path):
        with open(diagnostics_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        per_lane = payload.get("num_divergences_per_channel")
        if per_lane is not None and len(per_lane) == num_lanes:
            divergences = np.asarray(per_lane, dtype=int)
        elif int(payload.get("num_divergences", 0)) > int(max_divergences):
            # A joint diagnostic cannot localize the event, so conservatively
            # reject every lane in that selective attempt.
            divergences[:] = int(payload["num_divergences"])
    ess_failed = (
        (~np.isfinite(lane_ess)) | (lane_ess < float(min_depth_ess))
        if science_values else np.zeros_like(lane_ess, dtype=bool)
    )
    rank_failed = ((~np.isfinite(lane_tail)) | (lane_tail < float(min_depth_ess))
                   | (~np.isfinite(lane_rhat)) | (lane_rhat >= 1.01)
                   | (~np.isfinite(lane_chain_ess))
                   | (lane_chain_ess < float(min_depth_ess) / chains)) if chains > 1 else False
    failed = np.flatnonzero(ess_failed | rank_failed | (divergences > int(max_divergences)))
    as_lists = lambda metrics: {name: np.asarray(value).tolist() for name, value in metrics.items()}
    return failed, {
        "depth_ess_per_channel": as_lists(science_bulk).get("depths"),
        "science_ess_per_channel": as_lists(science_bulk),
        "science_tail_ess_per_channel": as_lists(science_tail_arrays),
        "science_rhat_per_channel": as_lists(science_rhat_arrays),
        "science_min_chain_ess_per_channel": as_lists(science_chain_ess),
        "nuisance_ess_per_channel": as_lists(nuisance_bulk),
        "nuisance_tail_ess_per_channel": as_lists(nuisance_tail),
        "nuisance_rhat_per_channel": as_lists(nuisance_rhat),
        "num_chains": chains,
        "num_divergences_per_channel": divergences.tolist(),
    }


def _spectro_sampler_swap_order(primary):
    if primary == "independent_nuts":
        return ("independent_hmc", "joint_nuts")
    if primary == "independent_hmc":
        return ("independent_nuts", "joint_nuts")
    return ()


def _resolve_parallel_chunk_job(parallel_job_count=None, parallel_job_index=None):
    if parallel_job_count is None:
        env_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if env_count is not None:
            parallel_job_count = int(env_count)

    if parallel_job_index is None:
        slurm_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if slurm_idx is not None:
            slurm_min = int(os.getenv("SLURM_ARRAY_TASK_MIN", slurm_idx))
            parallel_job_index = int(slurm_idx) - slurm_min

    return parallel_job_count, parallel_job_index


def _sample_independent_chunk(
    backend, cache, model, key, t, errors, observations, initial, *, nuts_kwargs, mcmc_kwargs,
    width, varying, diagnostics_path, progress_label=None, **model_kwargs,
):
    """Reuse a shape-specialized independent runner for a chunk or retry."""
    if backend == "independent_hmc":
        from models.independent_hmc import (
            build_independent_hmc_runner as build_runner,
            get_samples_independent_hmc as sample,
        )
    else:
        from models.independent_nuts import (
            build_independent_nuts_runner as build_runner,
            get_samples_independent as sample,
        )
    mcmc_kwargs = sampling_options(mcmc_kwargs)
    runner_key = (backend, int(width), varying)
    runner = cache.get(runner_key)
    if runner is None:
        runner = build_runner(
            model, nuts_kwargs=nuts_kwargs, mcmc_kwargs=mcmc_kwargs,
            lane_width=width, channel_varying_kwargs=varying,
        )
        cache[runner_key] = runner
    if progress_label is not None:
        try:
            runner.progress_label = progress_label
        except AttributeError:
            pass
    return sample(
        model, key, t, errors, observations, initial, nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs, diagnostics_path=diagnostics_path, lane_width=width,
        channel_varying_kwargs=varying, _runner=runner, **model_kwargs,
    )


def _align_power2_fallback_coordinates(primary, fallback, model):
    """Express fallback physical coefficients in the primary posterior basis.

    An adaptive safety fit may sample physical c/alpha rather than the smooth
    LD coordinates. Keeping the original failed fit's latent values in those
    lanes would serialize mutually inconsistent LD representations.
    """
    from models.ld_parameterization import Power2MaxtedTransform, power2_smooth_transform
    mode = getattr(model, '_koala_ld_mode', None)
    if mode not in {'uniform', 'gaussian', 'stellarprior'} or not all(
            name in fallback for name in ('c1', 'c2')):
        return fallback
    aligned = dict(fallback)
    coefficients = jnp.stack((fallback['c1'], fallback['c2']), axis=-1)
    if 'ld_smooth' in primary and 'ld_smooth' not in aligned:
        aligned['ld_smooth'] = power2_smooth_transform(
            'uniform' if mode == 'uniform' else 'gaussian').inv(coefficients)
    if 'ld_decorrelated' in primary and 'ld_decorrelated' not in aligned:
        aligned['ld_decorrelated'] = Power2MaxtedTransform()(coefficients)
    if 'ld_coefficients' in primary and 'ld_coefficients' not in aligned:
        aligned['ld_coefficients'] = coefficients
    return PosteriorSamples(aligned, num_chains=getattr(fallback, 'num_chains', 1),
                            sampled_sites=getattr(fallback, 'sampled_sites', None))


def _continue_independent_to_gate(backend, runner, samples, diagnostics_path,
                                  min_ess, max_divergences, science_sites, label,
                                  *, max_extra_blocks=3, initial_failed=None):
    """Extend warmed chains and gate the evenly spaced retained posterior.

    Every channel keeps the same requested draw count. Selecting uniformly
    across each chain's longer history avoids duplicating samples to align
    chunk shapes; diagnostics are recomputed on the actual retained draws.
    """
    failed = (np.asarray(initial_failed) if initial_failed is not None else
              _spectro_failed_lanes(samples, diagnostics_path, min_ess,
                  max_divergences, surface_science_sites=science_sites)[0])
    attempts = []
    if (getattr(samples, 'num_chains', 1) < 2 or min_ess <= 0 or not failed.size
            or not callable(getattr(runner, 'continue_raw', None))
            or not hasattr(runner, '_last_pooled_raw')):
        return samples, failed, attempts
    history = samples
    raw_history = dict(runner._last_pooled_raw)
    if bool(jnp.any(raw_history['diverging'])):
        return samples, failed, attempts
    chains = samples.num_chains
    retained = next(iter(samples.values())).shape[0] // chains
    if backend == 'independent_hmc':
        from models.independent_hmc import IndependentHMCDiagnostics as Diagnostics
        from models.independent_hmc import _save_hmc_diagnostics as save_diagnostics
    else:
        from models.independent_nuts import IndependentNUTSDiagnostics as Diagnostics
        from models.independent_nuts import _save_diagnostics as save_diagnostics
    for block in range(1, max_extra_blocks + 1):
        debug(f'{failed.size} wavelength channel{"s" if failed.size != 1 else ""} '
             f'need{"" if failed.size != 1 else "s"} extra draws (extension {block}).')
        debug(f'  {label} - continuing warmed {backend} chains (block {block}); '
              f'failed local lanes {failed.tolist()}.')
        extra, raw = runner.continue_raw()
        if bool(jnp.any(raw['diverging'])):
            debug(f'  {label} - continuation diverged; changing sampler.')
            break
        history = PosteriorSamples(
            {name:jnp.concatenate((history[name], extra[name]), axis=0) for name in history},
            num_chains=chains, sampled_sites=samples.sampled_sites)
        merged = dict(raw)
        for name in ('num_steps', 'accept_prob', 'diverging'):
            merged[name] = jnp.concatenate((raw_history[name], raw[name]), axis=0)
        raw_history = merged
        total = next(iter(history.values())).shape[0] // chains
        indices = np.floor((np.arange(retained) + .5) * total / retained).astype(int)
        selected = (indices[:, None] * chains + np.arange(chains)).reshape(-1)
        samples = PosteriorSamples({name:value[selected] for name,value in history.items()},
                                  num_chains=chains, sampled_sites=history.sampled_sites)
        selected_raw = dict(raw_history)
        for name in ('num_steps', 'accept_prob', 'diverging'):
            selected_raw[name] = raw_history[name][selected]
        selected_raw['mean_accept_prob'] = selected_raw['accept_prob'].mean(axis=0)
        if 'mean_accept_prob_by_chain' in selected_raw:
            selected_raw['mean_accept_prob_by_chain'] = selected_raw['accept_prob'].reshape(
                retained, chains, -1).mean(axis=0)
        save_diagnostics(diagnostics_path, Diagnostics(**selected_raw))
        failed, gate = _spectro_failed_lanes(samples, diagnostics_path, min_ess,
            max_divergences, surface_science_sites=science_sites)
        attempts.append(dict(sampler=backend, sampling_extension_blocks=block,
                             raw_draws_per_chain=total, retained_draws_per_chain=retained, **gate))
        if not failed.size:
            break
    return samples, failed, attempts


@quiet_sampling()
def get_samples_chunked(
    model, key, t, yerr, indiv_y, init_params, chunk_size, nuts_kwargs=None, mcmc_kwargs=None,
    chunk_mode='serial', parallel_job_count=None, parallel_job_index=None, output_dir=None,
    checkpoint_prefix=None, sampler_backend='joint_nuts', channel_varying_kwargs=(),
    checkpoint_signature=None, sampling_workload_fingerprint=None, spectro_min_depth_ess=0.0,
    spectro_max_divergences=0, adaptive_fallback_model=None,
    adaptive_fallback_init_params=None, adaptive_fallback_resident_width=None,
    _joint_mcmc_runner_cache=None, _independent_mcmc_runner_cache=None, **model_kwargs,
):
    """
    Run MCMC in chunks with checkpoint support for wall-time resilience.

    Parameters:
    -----------
    output_dir : str, optional
        Directory to save checkpoint files. If None, no checkpointing is used.
    checkpoint_prefix : str, optional
        Prefix for checkpoint filenames (e.g., 'WASP-39_PRISM_native').
        Checkpoints saved as: {output_dir}/chunks/{checkpoint_prefix}_chunk_{start}_{end}.pkl
    """
    num_lcs = indiv_y.shape[0]
    chunk_size = int(chunk_size)
    if chunk_size > num_lcs:
        # A single chunk never reuses its compiled program, so padding it to
        # the configured width only samples dead lanes.  Shrink to the real
        # channel count; multi-chunk stages keep the fixed width so the last
        # partial chunk reuses the compiled runner instead of recompiling.
        debug(
            f"Chunk width {chunk_size} exceeds the {num_lcs} channels in this "
            f"stage; using a resident width of {num_lcs} (no padded lanes)."
        )
        chunk_size = num_lcs
    chunk_mode = str(chunk_mode).lower()
    if sampler_backend != "joint_nuts" and checkpoint_prefix is not None:
        # Never silently reuse samples made by a different transition kernel
        # when comparing or switching sampler backends.
        checkpoint_prefix = f"{checkpoint_prefix}_{sampler_backend}"
    if chunk_mode not in {"serial", "parallel", "combine"}:
        raise ValueError(
            "chunk_mode must be one of {'serial', 'parallel', 'combine'}; "
            f"received {chunk_mode!r}."
        )
    debug(
        f"Running chunked MCMC: {num_lcs} channels in blocks of {chunk_size} "
        f"(mode={chunk_mode})"
    )

    # Set up checkpointing
    use_checkpoints = (output_dir is not None and checkpoint_prefix is not None)
    checkpoint_fingerprint = None
    if use_checkpoints:
        from .output_paths import OutputPaths
        checkpoint_dir = str(OutputPaths(output_dir).chunks_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logical_checkpoint_prefix = checkpoint_prefix
        checkpoint_fingerprint = _chunk_checkpoint_fingerprint(
            model, jax.random.key_data(key), t, yerr, indiv_y, init_params,
            chunk_size=chunk_size, sampler_backend=sampler_backend, nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs, channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=checkpoint_signature, model_kwargs=model_kwargs,
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
            adaptive_fallback_resident_width=adaptive_fallback_resident_width,
        )
        checkpoint_prefix = (
            f"{logical_checkpoint_prefix}_cfg{checkpoint_fingerprint[:12]}"
        )
        manifest_path = _write_or_validate_checkpoint_manifest(
            checkpoint_dir, logical_checkpoint_prefix, checkpoint_prefix,
            checkpoint_fingerprint, checkpoint_signature,
        )
        debug(f"Checkpoint directory: {checkpoint_dir}")
        debug(f'Checkpoint fingerprint: {checkpoint_fingerprint[:12]} ({manifest_path})')
    elif chunk_mode != "serial":
        raise ValueError("chunk_mode='parallel' or 'combine' requires output_dir and checkpoint_prefix.")

    all_chunk_ranges = _chunk_ranges_for(num_lcs, chunk_size)

    if chunk_mode == "parallel":
        parallel_job_count, parallel_job_index = _resolve_parallel_chunk_job(
            parallel_job_count=parallel_job_count, parallel_job_index=parallel_job_index,
        )
        if parallel_job_count is None or parallel_job_index is None:
            raise ValueError(
                "chunk_mode='parallel' requires chunk_parallel_job_count/chunk_parallel_job_index "
                "or SLURM array task environment variables."
            )
        if parallel_job_count < 1:
            raise ValueError("chunk_parallel_job_count must be >= 1.")
        if not (0 <= parallel_job_index < parallel_job_count):
            raise ValueError(
                "chunk_parallel_job_index must satisfy "
                f"0 <= index < count; received index={parallel_job_index}, "
                f"count={parallel_job_count}."
            )
        target_chunk_indices = [
            idx for idx in range(len(all_chunk_ranges))
            if idx % parallel_job_count == parallel_job_index
        ]
        debug(
            f"Parallel chunk job {parallel_job_index + 1}/{parallel_job_count} "
            f"assigned chunk indices {target_chunk_indices}"
        )
        if not target_chunk_indices:
            debug("No chunks assigned to this job; exiting without computation.")
            return None
    else:
        target_chunk_indices = list(range(len(all_chunk_ranges)))

    if chunk_mode == "combine":
        missing = []
        corrupt = []
        samples_chunks = []
        for batch_index in channel_batches(list(range(len(all_chunk_ranges))), all_chunk_ranges):
            start, end = all_chunk_ranges[batch_index]
            chunk_file = _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end)
            if not os.path.exists(chunk_file):
                missing.append((start, end))
                continue
            try:
                samples_chunks.append(_load_chunk_samples(chunk_file))
            except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
                corrupt.append((start, end, error))
        if missing:
            raise FileNotFoundError(
                "Cannot combine chunk checkpoints because the following chunk files are missing: "
                + ", ".join(f"{start}:{end}" for start, end in missing)
            )
        if corrupt:
            raise RuntimeError(
                "Cannot combine corrupt chunk checkpoint(s): "
                + ", ".join(
                    f"{start}:{end} ({type(error).__name__})"
                    for start, end, error in corrupt
                )
            )
        debug(f"\nConcatenating {len(samples_chunks)} chunks...")
        samples = _concatenate_chunk_samples(samples_chunks)
        debug(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
        return samples

    samples_chunks = []
    # One runner per exact resident/padded width. Every numerical chunk value
    # remains a dynamic argument; only the shape-specialized transition
    # programs are reused.
    joint_mcmc_runners = (
        {}
        if _joint_mcmc_runner_cache is None
        else _joint_mcmc_runner_cache
    )
    independent_mcmc_runners = (
        {}
        if _independent_mcmc_runner_cache is None
        else _independent_mcmc_runner_cache
    )

    # Derive every chunk's stream from its global index.  This makes serial,
    # resumed, and array-parallel execution assign exactly the same distinct
    # key to a given chunk, independent of which earlier checkpoints exist.
    base_key = key
    console_failures = []
    restored_channels = 0
    extra_draws_channels = 0
    refitted_channels = 0
    for chunk_idx in channel_batches(target_chunk_indices, all_chunk_ranges):
        start, end = all_chunk_ranges[chunk_idx]
        sl = slice(start, end)

        init_chunk = {
            k: _slice_by_channel(v, sl, num_lcs)
            for k, v in init_params.items()
        }
        channel_varying_set = frozenset(channel_varying_kwargs)
        kwargs_chunk = {
            k: (
                _slice_by_channel(v, sl, num_lcs)
                if k in channel_varying_set else v
            )
            for k, v in model_kwargs.items()
        }
        yerr_chunk = yerr[sl]
        y_chunk = indiv_y[sl]
        chunk_file = None
        if use_checkpoints:
            chunk_file = _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end)

        # Check if this chunk already exists
        if use_checkpoints:
            if os.path.exists(chunk_file):
                debug(f"  chunk {start}:{end} - LOADING from checkpoint")
                try:
                    samples_chunk = _load_chunk_samples(chunk_file)
                except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
                    info(f'Previous samples for channels {start + 1} to {end} could not be read; refitting.')
                    debug(
                        f"  chunk {start}:{end} - checkpoint unreadable "
                        f"({type(error).__name__}); recomputing atomically"
                    )
                else:
                    loaded_failed, _ = _spectro_failed_lanes(samples_chunk,
                        os.path.splitext(chunk_file)[0] + '.diagnostics.json',
                        spectro_min_depth_ess, spectro_max_divergences,
                        surface_science_sites=getattr(model, 'surface_science_sites', None))
                    console_failures.extend(start + int(i) for i in loaded_failed)
                    try:
                        with open(os.path.splitext(chunk_file)[0] + '.diagnostics.json') as stream:
                            history = json.load(stream).get('sampler_gate_attempts', [])
                    except (OSError, ValueError):
                        history = []
                    extra_count, refit_count = sampling_history_notes(
                        history, end - start, spectro_min_depth_ess, spectro_max_divergences)
                    extra_draws_channels += extra_count
                    refitted_channels += refit_count
                    restored_channels += end - start
                    samples_chunks.append(samples_chunk)
                    continue  # Skip computation for this chunk

        # Compute this chunk
        key_chunk = jax.random.fold_in(base_key, chunk_idx)
        debug(f"  chunk {start}:{end} - COMPUTING ({end - start} channels)")
        diagnostics_path = (
            os.path.splitext(chunk_file)[0] + ".diagnostics.json"
            if use_checkpoints else None
        )
        if sampler_backend in {"independent_nuts", "independent_hmc"}:
            varying_for_chunk = tuple(
                name for name in channel_varying_kwargs if name in kwargs_chunk
            )
            samples_chunk = _sample_independent_chunk(
                sampler_backend, independent_mcmc_runners, model, key_chunk, t, yerr_chunk,
                y_chunk, init_chunk, nuts_kwargs=nuts_kwargs, mcmc_kwargs=mcmc_kwargs,
                width=int(chunk_size), varying=varying_for_chunk,
                diagnostics_path=diagnostics_path, progress_label=f'chunk {start}:{end}',
                **kwargs_chunk,
            )
        elif sampler_backend == "joint_nuts":
            runner = None
            if bool(dict(mcmc_kwargs or {}).get("jit_model_args", True)):
                resident_width = end - start
                runner = joint_mcmc_runners.get(resident_width)
                if runner is None:
                    runner, _, _ = _build_numpyro_mcmc(
                        model, _tree_to_f64(init_chunk), dict(nuts_kwargs or {}),
                        dict(mcmc_kwargs or {}),
                    )
                    joint_mcmc_runners[resident_width] = runner
                else:
                    debug(
                        f"  chunk {start}:{end} - reusing compiled "
                        f"joint-NUTS runner (width {resident_width})"
                    )
            samples_chunk = get_samples(
                model, key_chunk, t, yerr_chunk, y_chunk, init_chunk, nuts_kwargs=nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs, diagnostics_path=diagnostics_path,
                _mcmc_runner=runner, **kwargs_chunk,
            )
        else:
            raise ValueError(
                "sampler_backend must be one of "
                "{'joint_nuts', 'independent_nuts', 'independent_hmc'}; "
                f"received {sampler_backend!r}."
            )
        sampler_used = [sampler_backend] * (end - start)
        gate_attempts = []
        if sampler_backend in {"independent_nuts", "independent_hmc"}:
            failed, gate = _spectro_failed_lanes(
                samples_chunk, diagnostics_path, spectro_min_depth_ess,
                spectro_max_divergences,
                surface_science_sites=getattr(model, 'surface_science_sites', None),
            )
            gate_attempts.append({"sampler": sampler_backend, **gate})
            if failed.size:
                primary_runner = independent_mcmc_runners.get(
                    (sampler_backend, int(chunk_size), varying_for_chunk))
                samples_chunk, failed, extensions = _continue_independent_to_gate(
                    sampler_backend, primary_runner, samples_chunk, diagnostics_path,
                    spectro_min_depth_ess, spectro_max_divergences,
                    getattr(model, 'surface_science_sites', None), f'chunk {start}:{end}',
                    initial_failed=failed)
                gate_attempts.extend(extensions)
            for attempt_index, fallback_backend in enumerate(
                _spectro_sampler_swap_order(sampler_backend), start=1
            ):
                if not failed.size:
                    break
                selected = failed.copy()
                debug(f'{len(selected)} wavelength channels need a fallback sampler.')
                debug(
                    f"  chunk {start}:{end} - gate failed in local lanes "
                    f"{selected.tolist()}; re-running only those lanes with "
                    f"{fallback_backend}."
                )

                def _select_lanes(value):
                    if hasattr(value, "baseline") and (
                        hasattr(value, "differences") or hasattr(value, "uniform")
                    ):
                        return _take_by_channel(value, selected, end - start)
                    try:
                        array = jnp.asarray(value)
                    except (TypeError, ValueError):
                        return value
                    if array.ndim and array.shape[0] == end - start:
                        return array[selected]
                    return value

                fallback_init = {name: _select_lanes(value)
                                 for name, value in init_chunk.items()}
                fallback_kwargs = {
                    name: (_select_lanes(value)
                           if name in channel_varying_set else value)
                    for name, value in kwargs_chunk.items()
                }
                fallback_path = (
                    None if diagnostics_path is None else
                    diagnostics_path.replace(
                        ".json", f".swap{attempt_index}_{fallback_backend}.json"
                    )
                )
                fallback_nuts = dict(nuts_kwargs or {})
                if fallback_backend in {"independent_hmc", "independent_nuts"}:
                    if fallback_backend == "independent_hmc":
                        fallback_nuts.pop("max_tree_depth", None)
                        fallback_nuts.pop("laplace_max_tree_depth", None)
                        fallback_nuts.update(
                            mass_matrix='laplace', laplace_warmup=150,
                            laplace_target_accept=0.85, laplace_start_at_map=True, num_steps=8,
                            trajectory_jitter=0.25,
                        )
                    else:
                        fallback_nuts.pop("num_steps", None)
                        fallback_nuts.pop("trajectory_jitter", None)
                        fallback_nuts.update(
                            mass_matrix="laplace", laplace_warmup=150,
                            laplace_target_accept=(
                                0.99 if float((nuts_kwargs or {}).get(
                                    "laplace_target_accept", 0.95)) >= 0.99 else 0.95
                            ),
                            laplace_start_at_map=True,
                        )
                    fallback = _sample_independent_chunk(
                        fallback_backend, independent_mcmc_runners, model,
                        jax.random.fold_in(key_chunk, 99173 + attempt_index),
                        t, yerr_chunk[selected], y_chunk[selected], fallback_init,
                        nuts_kwargs=fallback_nuts, mcmc_kwargs=mcmc_kwargs,
                        diagnostics_path=fallback_path, width=chunk_size,
                        varying=varying_for_chunk, **fallback_kwargs,
                    )
                else:
                    adaptive_model = (
                        model if adaptive_fallback_model is None
                        else adaptive_fallback_model
                    )
                    if adaptive_fallback_init_params is not None:
                        fallback_init = {}
                        for name, value in adaptive_fallback_init_params.items():
                            try:
                                array = jnp.asarray(value)
                            except (TypeError, ValueError):
                                fallback_init[name] = value
                                continue
                            if array.ndim and array.shape[0] == num_lcs:
                                fallback_init[name] = array[start:end][selected]
                            else:
                                fallback_init[name] = _select_lanes(value)
                    adaptive_nuts = {
                        key: value for key, value in fallback_nuts.items()
                        if key in {"dense_mass", "regularize_mass_matrix",
                                   "target_accept_prob", "max_tree_depth"}
                    }
                    # This is the legacy adaptive joint-NUTS safety net, not
                    # the shallow-tree production Laplace kernel. Restore its
                    # historical tree-depth allowance for difficult lanes.
                    adaptive_nuts.update(target_accept_prob=0.95, max_tree_depth=10)
                    adaptive_width = (
                        int(selected.size)
                        if adaptive_fallback_resident_width is None else
                        int(adaptive_fallback_resident_width)
                    )
                    if adaptive_width < 1:
                        raise ValueError('adaptive_fallback_resident_width must be positive')
                    adaptive_runner_key = ('adaptive_fallback', adaptive_width)
                    adaptive_runner = joint_mcmc_runners.get(adaptive_runner_key)
                    adaptive_batches = []
                    adaptive_divergences = []
                    for batch_index, batch_start in enumerate(
                        range(0, int(selected.size), adaptive_width)
                    ):
                        batch_stop = min(batch_start + adaptive_width, int(selected.size))
                        batch_positions = jnp.arange(batch_start, batch_stop)
                        batch_size = batch_stop - batch_start

                        def _batch_adaptive(value):
                            if hasattr(value, "baseline") and (
                                hasattr(value, "differences") or hasattr(value, "uniform")
                            ):
                                batched = _take_by_channel(
                                    value, batch_positions, int(selected.size)
                                )
                                if (
                                    np.ndim(batched.baseline) > 1
                                    and adaptive_width > batch_size
                                ):
                                    pad = adaptive_width - batch_size
                                    batched = type(batched)(*(
                                        None if field is None else jnp.concatenate(
                                            (field, jnp.repeat(field[-1:], pad, axis=0)),
                                        )
                                        for field in batched
                                    ))
                                return batched
                            try:
                                array = jnp.asarray(value)
                            except (TypeError, ValueError):
                                return value
                            if (array.ndim and
                                    array.shape[0] == int(selected.size)):
                                array = array[batch_positions]
                                if adaptive_width > batch_size:
                                    array = jnp.concatenate((
                                        array,
                                        jnp.repeat(
                                            array[-1:], adaptive_width - batch_size, axis=0,
                                        ),
                                    ), axis=0)
                            return array

                        adaptive_yerr = _batch_adaptive(yerr_chunk[selected])
                        adaptive_y = _batch_adaptive(y_chunk[selected])
                        adaptive_init = {
                            name: _batch_adaptive(value)
                            for name, value in fallback_init.items()
                        }
                        adaptive_kwargs = {
                            name: (
                                _batch_adaptive(value)
                                if name in channel_varying_set else value
                            )
                            for name, value in fallback_kwargs.items()
                        }
                        if adaptive_runner is None:
                            adaptive_runner, _, _ = _build_numpyro_mcmc(
                                adaptive_model, _tree_to_f64(adaptive_init),
                                adaptive_nuts, dict(mcmc_kwargs or {}),
                            )
                            joint_mcmc_runners[adaptive_runner_key] = (
                                adaptive_runner
                            )
                        else:
                            debug(
                                f"  chunk {start}:{end} - reusing compiled "
                                f"adaptive joint-NUTS runner "
                                f"(resident width {adaptive_width}, batch "
                                f"{batch_index + 1})"
                            )
                        batch_path = fallback_path
                        if fallback_path is not None and int(selected.size) > adaptive_width:
                            batch_path = fallback_path.replace(
                                ".json", f".batch{batch_index + 1}.json"
                            )
                        batch_samples = get_samples(
                            adaptive_model,
                            jax.random.fold_in(
                                key_chunk, 99173 + attempt_index + 1009 * batch_index,
                            ),
                            t, adaptive_yerr, adaptive_y, adaptive_init,
                            nuts_kwargs=adaptive_nuts,
                            mcmc_kwargs=mcmc_kwargs,
                            diagnostics_path=batch_path,
                            _mcmc_runner=adaptive_runner,
                            **adaptive_kwargs,
                        )
                        adaptive_batches.append(jax.tree.map(
                            lambda value: value[:, :batch_size], batch_samples
                        ))
                        batch_divergence = 0
                        if batch_path is not None and os.path.isfile(batch_path):
                            with open(batch_path, "r", encoding="utf-8") as stream:
                                batch_divergence = int(
                                    json.load(stream).get("num_divergences", 0)
                                )
                        adaptive_divergences.extend([batch_divergence] * batch_size)
                    fallback = jax.tree.map(
                        lambda *values: jnp.concatenate(values, axis=1), *adaptive_batches,
                    )
                    if (fallback_path is not None and
                            int(selected.size) > adaptive_width):
                        with open(fallback_path, "x", encoding="utf-8") as stream:
                            json.dump({
                                "num_divergences": int(sum(adaptive_divergences)),
                                "num_divergences_per_channel": (
                                    adaptive_divergences
                                ),
                                "resident_width": adaptive_width,
                                "num_batches": len(adaptive_batches),
                            }, stream, indent=2)
                            stream.write("\n")
                fallback_failed, fallback_gate = _spectro_failed_lanes(
                    fallback, fallback_path, spectro_min_depth_ess, spectro_max_divergences,
                    surface_science_sites=getattr(model, 'surface_science_sites', None),
                )
                gate_attempts.append({
                    "sampler": fallback_backend,
                    "input_local_lanes": selected.tolist(),
                    **fallback_gate,
                })
                if fallback_failed.size and fallback_backend in {
                        'independent_nuts', 'independent_hmc'}:
                    fallback_runner = independent_mcmc_runners.get(
                        (fallback_backend, int(chunk_size), varying_for_chunk))
                    fallback, fallback_failed, extensions = _continue_independent_to_gate(
                        fallback_backend, fallback_runner, fallback, fallback_path,
                        spectro_min_depth_ess, spectro_max_divergences,
                        getattr(model, 'surface_science_sites', None),
                        f'chunk {start}:{end} fallback {fallback_backend}',
                        initial_failed=fallback_failed)
                    gate_attempts.extend({**attempt, 'input_local_lanes':selected.tolist()}
                                         for attempt in extensions)
                fallback = _align_power2_fallback_coordinates(samples_chunk, fallback, model)
                # A sampler swap with the same coordinates must preserve the
                # free-site metadata used by subsequent diagnostics. Adaptive
                # fallbacks may use another coordinate system; their own gate
                # has already checked those free sites before this merge.
                sampled_sites = getattr(samples_chunk, 'sampled_sites', None)
                if sampled_sites != getattr(fallback, 'sampled_sites', None):
                    sampled_sites = None
                samples_chunk = PosteriorSamples({
                    name: (
                        jnp.asarray(values).at[:, selected].set(fallback[name])
                        if name in fallback else values
                    )
                    for name, values in samples_chunk.items()
                }, num_chains=int(dict(mcmc_kwargs or {}).get('num_chains', 1)),
                   sampled_sites=sampled_sites)
                for lane in selected:
                    sampler_used[int(lane)] = fallback_backend
                failed = selected[fallback_failed]
        extra_count, refit_count = sampling_history_notes(
            gate_attempts, end - start, spectro_min_depth_ess, spectro_max_divergences)
        extra_draws_channels += extra_count
        refitted_channels += refit_count
        samples_chunk = _SamplerSamples(samples_chunk, sampler_used=sampler_used,
            num_chains=int(dict(mcmc_kwargs or {}).get("num_chains", 1)),
            sampled_sites=getattr(samples_chunk, "sampled_sites", None))
        _annotate_sampler_diagnostics(
            diagnostics_path, sampling_workload_fingerprint=sampling_workload_fingerprint,
            sampler_backend=sampler_backend,
        )
        if diagnostics_path is not None and os.path.isfile(diagnostics_path):
            with open(diagnostics_path, "r", encoding="utf-8") as stream:
                diagnostic_payload = json.load(stream)
            diagnostic_payload["sampler_used"] = sampler_used
            diagnostic_payload["sampler_gate_attempts"] = gate_attempts
            temporary = f"{diagnostics_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(diagnostic_payload, stream, indent=2, sort_keys=True)
            os.replace(temporary, diagnostics_path)
        if sampler_backend in {'independent_nuts', 'independent_hmc'} and failed.size:
            info(f'{len(failed)} wavelength channels failed the convergence checks. See koala.log and the diagnostics files.')
            # Retain the complete audit even when the gate prevents saving a
            # posterior checkpoint or starting downstream stages.
            raise RuntimeError(
                f"chunk {start}:{end} failed the spectroscopic ESS/divergence "
                f"gate after adaptive NUTS in local lanes {failed.tolist()}"
            )
        if sampler_backend == 'joint_nuts':
            observed_failed, _ = _spectro_failed_lanes(samples_chunk, diagnostics_path,
                spectro_min_depth_ess, spectro_max_divergences,
                surface_science_sites=getattr(model, 'surface_science_sites', None))
            console_failures.extend(start + int(i) for i in observed_failed)
        samples_chunk = jax.device_get(samples_chunk)
        samples_chunks.append(samples_chunk)

        # Save checkpoint immediately after computation
        if use_checkpoints:
            chunk_file = _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end)
            temporary_chunk = (
                f"{chunk_file}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            with open(temporary_chunk, 'wb') as f:
                pickle.dump(
                    {'samples': dict(samples_chunk), 'sampler_used': samples_chunk.sampler_used,
                     'num_chains': samples_chunk.num_chains,
                     'sampled_sites': samples_chunk.sampled_sites}, f
                )
            os.replace(temporary_chunk, chunk_file)
            debug(f"  chunk {start}:{end} - SAVED checkpoint")

    if chunk_mode == "parallel":
        existing = [
            os.path.exists(_chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end))
            for start, end in all_chunk_ranges
        ]
        done = sum(existing)
        debug(
            f"Parallel chunk job complete: {done}/{len(all_chunk_ranges)} checkpoint files present. "
            "Run the compatibility combine step after all chunks are finished."
        )
        return None

    # Concatenate all chunks
    debug(f"\nConcatenating {len(samples_chunks)} chunks...")
    samples = _concatenate_chunk_samples(samples_chunks)
    samples.console_failed_indices = console_failures
    samples.console_extra_draws = extra_draws_channels
    samples.console_refitted = refitted_channels
    if restored_channels:
        info(f'Reusing previous spectroscopy samples for {restored_channels} wavelength channels.')
    debug(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
    return samples


@quiet_sampling()
@single_batch_progress
def _run_sampling_stage(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    use_chunked=False,
    chunk_size=None,
    chunk_mode="serial",
    parallel_job_count=None,
    parallel_job_index=None,
    output_dir=None,
    checkpoint_prefix=None,
    sampler_backend="joint_nuts",
    channel_varying_kwargs=(),
    checkpoint_signature=None,
    spectro_min_depth_ess=0.0,
    spectro_max_divergences=0,
    stacking_metadata=None,
    adaptive_fallback_model=None,
    adaptive_fallback_init_params=None,
    execution_context=None,
    **model_kwargs,
):
    # Cadence padding is the production compile-shape policy.
    original_cadences = int(jnp.asarray(t).shape[0])
    t, indiv_y, yerr, likelihood_mask = _pad_spectro_cadences_exact(
        t, indiv_y, yerr, multiple=256
    )
    padded_kwargs = dict(model_kwargs)
    for name, value in model_kwargs.items():
        if hasattr(value, "baseline") and (
            hasattr(value, "differences") or hasattr(value, "uniform")
        ):
            pad = int(t.shape[0]) - original_cadences
            padded_kwargs[name] = type(value)(*(
                None if field is None else jnp.pad(
                    field, [(0, 0)] * (field.ndim - 1) + [(0, pad)], constant_values=0.0,
                )
                for field in value
            ))
            continue
        try:
            array = jnp.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.ndim and array.shape[-1] == original_cadences:
            pad = int(t.shape[0]) - original_cadences
            padded_kwargs[name] = jnp.pad(
                array, [(0, 0)] * (array.ndim - 1) + [(0, pad)], constant_values=0.0,
            )
    signature = inspect.signature(model)
    accepts_mask = (
        "likelihood_mask" in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )
    if accepts_mask:
        padded_kwargs["likelihood_mask"] = likelihood_mask
        model_kwargs = padded_kwargs
    else:
        t = t[:original_cadences]
        indiv_y = indiv_y[..., :original_cadences]
        yerr = yerr[..., :original_cadences]

    _save_stacking_inputs(model, t, yerr, indiv_y, model_kwargs, stacking_metadata)
    sampling_workload_fingerprint = _sampling_workload_fingerprint(
        model, t, yerr, indiv_y, init_params, sampler_backend=sampler_backend,
        nuts_kwargs=nuts_kwargs, channel_varying_kwargs=channel_varying_kwargs,
        checkpoint_signature=checkpoint_signature, model_kwargs=model_kwargs,
    )
    if use_chunked:
        effective_chunk_size = 1 if chunk_size is None else int(chunk_size)
        if effective_chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 when chunked sampling is enabled.")
        runner_kwargs = {}
        if execution_context is not None:
            joint_cache, independent_cache = execution_context.runner_caches(model, nuts_kwargs, mcmc_kwargs)
            runner_kwargs = dict(_joint_mcmc_runner_cache=joint_cache,
                                 _independent_mcmc_runner_cache=independent_cache)
        return get_samples_chunked(
            model, key, t, yerr, indiv_y, init_params, effective_chunk_size,
            nuts_kwargs=nuts_kwargs, mcmc_kwargs=mcmc_kwargs, chunk_mode=chunk_mode,
            parallel_job_count=parallel_job_count, parallel_job_index=parallel_job_index,
            output_dir=output_dir, checkpoint_prefix=checkpoint_prefix,
            sampler_backend=sampler_backend, channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=checkpoint_signature,
            sampling_workload_fingerprint=sampling_workload_fingerprint,
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
            adaptive_fallback_model=adaptive_fallback_model,
            adaptive_fallback_init_params=adaptive_fallback_init_params, **runner_kwargs, **model_kwargs,
        )
    if sampler_backend in {"independent_nuts", "independent_hmc"}:
        if sampler_backend == "independent_hmc":
            from models.independent_hmc import get_samples_independent_hmc
            independent_sampler = get_samples_independent_hmc
        else:
            from models.independent_nuts import get_samples_independent
            independent_sampler = get_samples_independent
        return independent_sampler(
            model,
            key,
            t,
            yerr,
            indiv_y,
            init_params,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=sampling_options(mcmc_kwargs),
            lane_width=int(indiv_y.shape[0]),
            channel_varying_kwargs=tuple(
                (name for name in channel_varying_kwargs if name in model_kwargs),
            ),
            **model_kwargs,
        )
    if sampler_backend != "joint_nuts":
        raise ValueError(
            "sampler_backend must be one of "
            "{'joint_nuts', 'independent_nuts', 'independent_hmc'}; "
            f"received {sampler_backend!r}."
        )
    return get_samples(
        model, key, t, yerr, indiv_y, init_params, nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs, **model_kwargs,
    )


def evaluate_channels_sequentially(
    map_params, final_in_axes, time_values, num_lcs, detrend_type_multiwave, selected_kernel,
    transit_engine, surface_config, jaxoplanet_kernel, ld_profile, gp_trend, spot_trend,
    spot_trend2, jump_trend, exp_trend, attach_surface_eval_metadata,
    has_single_spot_spectroscopic,
):
    """Evaluate channels in the original bounded-memory sequential order."""
    @jax.jit
    def eval_channel(channel_params, t_val, *extra_args):
        if transit_engine == 'jaxoplanet':
            channel_params = attach_surface_eval_metadata(channel_params, surface_config)
            channel_params = {
                **channel_params,
                "_jaxoplanet_kernel": jaxoplanet_kernel,
                "_ld_profile": ld_profile,
            }
        return selected_kernel(channel_params, t_val, *extra_args)

    model_all_list = []
    for i in range(num_lcs):
        channel_params = {}
        for k, v in map_params.items():
            if k in _JAXOPLANET_STATIC_EVAL_KEYS or k in _SURFACE_STATIC_EVAL_KEYS:
                continue
            if final_in_axes[k] == 0:
                channel_params[k] = v[i]
            else:
                channel_params[k] = v

        if 'gp_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel(channel_params, time_values, gp_trend)
        elif '2spot_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel(channel_params, time_values, spot_trend, spot_trend2)
        elif has_single_spot_spectroscopic(detrend_type_multiwave):
            if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
                ch_model = eval_channel(channel_params, time_values, spot_trend, jump_trend)
            else:
                ch_model = eval_channel(channel_params, time_values, spot_trend)
        elif 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel(channel_params, time_values, jump_trend)
        elif 'explinear_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel(channel_params, time_values, exp_trend)
        else:
            ch_model = eval_channel(channel_params, time_values)

        model_all_list.append(ch_model)

    return jnp.stack(model_all_list, axis=0)

"""Sampling helpers."""

import os
import sys
import glob
import csv
import json
import pickle
import hashlib
import uuid
import re
import time
import tempfile
import difflib
import warnings
import logging
import inspect
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting import (
    plot_harmonica_limb_spectra,
    plot_harmonica_transmission_posterior,
    plot_harmonica_transmission_strings,
    plot_map_fits,
    plot_map_residuals,
    plot_noise_binning_from_csv,
    plot_transmission_spectrum,
    plot_wavelength_offset_summary,
    plot_whitelight_curve,
    plot_whitelight_residuals,
    plot_whitelight_summary,
)
import argparse
import yaml
import jaxopt
import arviz as az
from createdatacube import SpectroData, process_spectroscopy_data
from matplotlib.widgets import Slider, Button, TextBox
from jaxoplanet.experimental import calc_poly_coeffs
import tinygp
from models.common import _to_f64, _tree_to_f64, get_I_power2, compute_transit_model_auto
from models.ld_parameterization import Power2MaxtedTransform
from models.sing_ld import (
    SING_TABULATED_OFFSET,
    SING_TABULATED_SCATTER,
    estimate_gray_offset,
    quadratic_to_sing,
    write_offset_artifact,
)
from models.harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_cos_i_from_geometry,
    harmonica_impact_param_from_cos_i,
    harmonica_duration_from_geometry,
    harmonica_duration_from_cos_i,
    harmonica_odd_coeff_specs,
    HARMONICA_SPECTRO_RORS_MIN,
    HARMONICA_SPECTRO_RORS_MAX,
)
from models.trends import (
    spot_crossing, compute_lc_linear, compute_lc_quadratic, compute_lc_cubic,
    compute_lc_quartic, compute_lc_linear_discontinuity, compute_lc_explinear,
    compute_lc_spot, compute_lc_2spot, compute_lc_none
)
from models.gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_cubic_gp_mean, compute_lc_quartic_gp_mean, compute_lc_explinear_gp_mean
)
from models.detrend import resolve_detrend_kernel
from models.trend_marginal import (
    marginalized_trend_coefficient_names,
    materialize_marginalized_trend_samples,
)
from surface_outputs import save_surface_results
from models.limb_darkening_config import (
    LD_PRIORS,
    resolve_ld_prior,
    validate_ld_profile,
)
from .constants import *
from .constants import (
    _JUMP_WIDTH_DAYS, _VALID_HARMONICA_SPECTRO_PARAMETERIZATIONS,
    _JAXOPLANET_STATIC_EVAL_KEYS, _SURFACE_STATIC_EVAL_KEYS,
    _JAXOPLANET_DYNAMIC_EVAL_KEYS,
)
from .config import (
    UnknownFlagWarning, _validate_flag_keys, _build_gaussian_trend_prior,
    _has_stellar_ld_uncertainties, _resolve_ld_prior_mode,
    _resolve_ld_parameterization, _spectro_detrend_type,
    _has_single_spot_spectroscopic, load_config, _resolve_stage_vmap_width,
    _resolve_stage_mcmc_kwargs, _resolve_whitelight_laplace_options,
    _resolve_whitelight_mass_matrix, _resolve_whitelight_trend_parameterization,
    _resolve_compile_cache_options, _resolve_ld_prior_cache_options,
    _resolve_harmonica_stage_nuts_kwargs,
    _resolve_jaxoplanet_spectro_nuts_kwargs,
)
from .data import _pad_spectro_cadences_exact, jax_bin_lightcurve
from .artifacts import (
    _update_checkpoint_hash, _science_artifact_fingerprint,
    _science_artifact_manifest_matches, _write_science_artifact_manifest,
    _file_content_identity, _optional_file_content_identity,
    _directory_metadata_identity, _atomic_save_npy, _atomic_savez,
    _atomic_savez_compressed, _atomic_dataframe_csv,
)
from models.harmonica.core import (
    _ALL_ODD_COEFF_SPECS,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_area_radius_and_q,
)

from .constants import *
from .artifacts import _update_checkpoint_hash
from .config import _resolve_stage_vmap_width
from .data import _pad_spectro_cadences_exact
from .outputs import _save_mcmc_diagnostics

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
    mcmc_defaults.update(mcmc_kwargs)
    return (
        numpyro.infer.MCMC(
            numpyro.infer.NUTS(model, **nuts_defaults),
            **mcmc_defaults,
        ),
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
            "init_strategy",
            numpyro.infer.init_to_value(values=init_params),
        )
        nuts_defaults = {
            "regularize_mass_matrix": False,
            "target_accept_prob": 0.9,
        }
        nuts_defaults.update(nuts_kwargs)
    mcmc.run(
        key,
        t,
        yerr,
        y=indiv_y,
        extra_fields=("diverging", "accept_prob", "potential_energy", "num_steps"),
        **model_kwargs,
    )
    _save_mcmc_diagnostics(
        mcmc,
        max_tree_depth=nuts_defaults.get("max_tree_depth"),
        output_path=diagnostics_path,
    )
    return mcmc.get_samples()


def _slice_by_channel(value, sl, num_lcs):
    if value is None:
        return None
    if hasattr(value, "baseline") and (
        hasattr(value, "differences") or hasattr(value, "uniform")
    ):
        if np.ndim(value.baseline) <= 1:
            return value
        return type(value)(*(
            None if field is None else field[sl] for field in value
        ))
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
        return type(value)(*(
            None if field is None else field[indices] for field in value
        ))
    if (
        isinstance(value, (np.ndarray, jnp.ndarray))
        and value.ndim > 0
        and value.shape[0] == num_lcs
    ):
        return value[jnp.asarray(channel_indices, dtype=jnp.int32)]
    return value


class SamplerInputsDumpExit(RuntimeError):
    """Internal clean-stop signal after the requested first high-res dump."""


_DUMPED_FIRST_HIGHRES_STAGE = False


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
        # NumPyro initialization strategies are top-level callables/partials
        # and remain pickleable.  Keeping them is necessary for an exact
        # offline replay when a caller explicitly overrides init_strategy.
        return value
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError):
        return value
    if array.ndim == 0:
        return array.item()
    return array


def _build_spectroscopic_model(builder, **builder_kwargs):
    """Build a model and retain the exact importable builder invocation."""
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
    return {
        "identity": _callable_identity(model),
        "kwargs": {},
        "callable_is_model": True,
    }


def _sampler_initial_potential(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    nuts_kwargs,
    model_kwargs,
):
    """Evaluate the exact NumPyro potential used to initialize joint NUTS."""
    init_strategy = dict(nuts_kwargs or {}).get(
        "init_strategy",
        numpyro.infer.init_to_value(values=init_params),
    )
    model_info = numpyro.infer.util.initialize_model(
        key,
        model,
        init_strategy=init_strategy,
        dynamic_args=False,
        model_args=(t, yerr),
        model_kwargs={"y": indiv_y, **model_kwargs},
        validate_grad=False,
    )
    potential = model_info.potential_fn(model_info.param_info.z)
    return (
        float(np.asarray(jax.device_get(potential))),
        _tree_to_numpy_for_pickle(model_info.param_info.z),
    )


def _safe_stage_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return label or "spectroscopic_stage"


def _maybe_dump_sampler_inputs(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    *,
    nuts_kwargs,
    mcmc_kwargs,
    chunk_size,
    use_chunked,
    sampler_backend,
    channel_varying_kwargs,
    checkpoint_signature,
    dump_metadata,
    model_kwargs,
):
    """Atomically dump one replayable spectroscopic stage when requested."""
    dump_directory = os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
    if not dump_directory:
        return None

    metadata = dict(dump_metadata or {})
    stage_kind = str(
        metadata.get(
            "stage_kind",
            (checkpoint_signature or {}).get("stage", "spectroscopic"),
        )
    )
    stage_label = _safe_stage_label(
        metadata.get("stage_label", stage_kind)
    )
    num_channels = int(np.shape(indiv_y)[0])
    num_cadences = int(np.shape(indiv_y)[1])
    active_cadences = metadata.get("active_window_cadences", num_cadences)
    metadata.update(
        {
            "stage_label": stage_label,
            "stage_kind": stage_kind,
            "num_channels": num_channels,
            "num_cadences": num_cadences,
            "active_window_cadences": int(active_cadences),
        }
    )

    potential, unconstrained_init = _sampler_initial_potential(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        nuts_kwargs,
        model_kwargs,
    )
    effective_chunk_size = (
        int(chunk_size)
        if chunk_size is not None
        else num_channels
    )
    payload = {
        "schema_version": SAMPLER_STAGE_INPUT_SCHEMA_VERSION,
        "model_builder": _sampler_model_builder_spec(model),
        "rng_key": _tree_to_numpy_for_pickle(key),
        "t": _tree_to_numpy_for_pickle(t),
        "yerr": _tree_to_numpy_for_pickle(yerr),
        "indiv_y": _tree_to_numpy_for_pickle(indiv_y),
        "init_params": _tree_to_numpy_for_pickle(init_params),
        "nuts_kwargs": _tree_to_numpy_for_pickle(dict(nuts_kwargs or {})),
        "mcmc_kwargs": _tree_to_numpy_for_pickle(dict(mcmc_kwargs or {})),
        "chunk_size": effective_chunk_size,
        "configured_chunk_size": (
            None if chunk_size is None else int(chunk_size)
        ),
        "use_chunked": bool(use_chunked),
        "sampler_backend": str(sampler_backend),
        "channel_varying_kwargs": tuple(channel_varying_kwargs),
        "model_kwargs": _tree_to_numpy_for_pickle(model_kwargs),
        "checkpoint_signature": _tree_to_numpy_for_pickle(
            checkpoint_signature
        ),
        "initial_potential": potential,
        "unconstrained_init": unconstrained_init,
        "meta": _tree_to_numpy_for_pickle(metadata),
    }

    dump_directory = os.path.abspath(os.path.expanduser(dump_directory))
    os.makedirs(dump_directory, exist_ok=True)
    output_path = os.path.join(
        dump_directory, f"{stage_label}_inputs.pkl"
    )
    temporary_path = f"{output_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary_path, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_path)
    print(
        "Dumped replayable spectroscopic sampler inputs to "
        f"{output_path} (initial potential={potential:.17g})",
        flush=True,
    )

    global _DUMPED_FIRST_HIGHRES_STAGE
    is_highres = stage_kind.lower() in {
        "high_resolution", "highres", "high-resolution"
    }
    if (
        is_highres
        and not _DUMPED_FIRST_HIGHRES_STAGE
        and os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT", "0") == "1"
    ):
        _DUMPED_FIRST_HIGHRES_STAGE = True
        raise SamplerInputsDumpExit(output_path)
    if is_highres:
        _DUMPED_FIRST_HIGHRES_STAGE = True
    return output_path


def _chunk_checkpoint_fingerprint(
    model,
    rng_key,
    t,
    yerr,
    indiv_y,
    init_params,
    *,
    chunk_size,
    sampler_backend,
    nuts_kwargs,
    mcmc_kwargs,
    channel_varying_kwargs,
    checkpoint_signature,
    model_kwargs,
    spectro_min_depth_ess=0.0,
    spectro_max_divergences=0,
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
    model,
    t,
    yerr,
    indiv_y,
    init_params,
    *,
    sampler_backend,
    nuts_kwargs,
    channel_varying_kwargs,
    checkpoint_signature,
    model_kwargs,
):
    """Fingerprint the science target independently of draws and batching.

    A short pilot and a production run may intentionally use different random
    keys, warmup/sample counts, or resident widths.  They may not silently use
    different data, priors, model code, or transition-kernel controls.
    """

    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    relative_sources = (
        "fit_jwst.py",
        "models/channel_batching.py",
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
        if relative_path == "fit_jwst.py":
            # Wave-2 is a mechanical split of this exact source revision. Keep
            # its pre-split workload identity so package relocation alone does
            # not change checkpoint names or random streams.
            source_sha256[relative_path] = (
                "aff1af24745ae42959f5b2a32aba03b3dbbc5eaf0288003b03e1f4a324ba8090"
            )
            continue
        path = os.path.join(repository_root, relative_path)
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                source_sha256[relative_path] = hashlib.sha256(
                    stream.read()
                ).hexdigest()

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
    path,
    *,
    sampling_workload_fingerprint,
    sampler_backend,
):
    """Atomically bind a sampler diagnostic to its exact science workload."""

    if path is None or sampling_workload_fingerprint is None:
        return
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Sampler did not write its requested diagnostics file: {path}"
        )
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload.update(
        {
            "diagnostic_provenance_schema_version": 1,
            "sampling_workload_fingerprint_sha256": str(
                sampling_workload_fingerprint
            ),
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
    checkpoint_dir,
    logical_prefix,
    effective_prefix,
    fingerprint,
    checkpoint_signature,
):
    manifest_path = os.path.join(
        checkpoint_dir, f"{effective_prefix}.manifest.json"
    )
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
                "Chunk checkpoint manifest does not match the current fit: "
                f"{manifest_path}"
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


class _SamplerSamples(dict):
    """Sample mapping carrying non-array per-channel sampler provenance."""

    def __init__(self, *args, sampler_used=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_used = None if sampler_used is None else list(sampler_used)


def _load_chunk_samples(chunk_file):
    with open(chunk_file, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "samples" in payload and "sampler_used" in payload:
        return _SamplerSamples(payload["samples"], sampler_used=payload["sampler_used"])
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
    return _SamplerSamples(samples, sampler_used=sampler_used or None)


def _spectro_failed_lanes(samples, diagnostics_path, min_depth_ess, max_divergences):
    """Return failed lanes and auditable gate metrics for an exact-MCMC run."""
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
            r"_(?:eclipse_depth|dayside_flux|nightside_flux|hotspot_offset|stellar_spot_contrast)_\d+",
            name,
        ):
            science_values[name.removeprefix("_")] = np.asarray(
                jax.device_get(value)
            )
    science_ess = {}
    lane_ess = np.full(depth_values.shape[1], np.inf, dtype=float)
    for name, values in science_values.items():
        per_lane = []
        for lane in range(depth_values.shape[1]):
            value = jnp.asarray(values[:, lane])
            ess = numpyro.diagnostics.effective_sample_size(value[None, ...])
            ess_values = np.asarray(jax.device_get(ess), dtype=float)
            per_lane.append(
                float(np.min(ess_values))
                if np.all(np.isfinite(ess_values)) else np.nan
            )
        science_ess[name] = per_lane
        lane_ess = np.minimum(lane_ess, np.asarray(per_lane))
    divergences = np.zeros(depth_values.shape[1], dtype=int)
    if diagnostics_path is not None and os.path.isfile(diagnostics_path):
        with open(diagnostics_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        per_lane = payload.get("num_divergences_per_channel")
        if per_lane is not None and len(per_lane) == depth_values.shape[1]:
            divergences = np.asarray(per_lane, dtype=int)
        elif int(payload.get("num_divergences", 0)) > int(max_divergences):
            # A joint diagnostic cannot localize the event, so conservatively
            # reject every lane in that selective attempt.
            divergences[:] = int(payload["num_divergences"])
    failed = np.flatnonzero(
        (~np.isfinite(lane_ess))
        | (lane_ess < float(min_depth_ess))
        | (divergences > int(max_divergences))
    )
    return failed, {
        "depth_ess_per_channel": science_ess.get("depths"),
        "science_ess_per_channel": science_ess,
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
        env_count = os.getenv("JWSTJAXFIT_CHUNK_TASK_COUNT")
        if env_count is None:
            env_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if env_count is not None:
            parallel_job_count = int(env_count)

    if parallel_job_index is None:
        env_idx = os.getenv("JWSTJAXFIT_CHUNK_TASK_ID")
        if env_idx is not None:
            parallel_job_index = int(env_idx)
        else:
            slurm_idx = os.getenv("SLURM_ARRAY_TASK_ID")
            if slurm_idx is not None:
                slurm_min = int(os.getenv("SLURM_ARRAY_TASK_MIN", slurm_idx))
                parallel_job_index = int(slurm_idx) - slurm_min

    return parallel_job_count, parallel_job_index


def get_samples_chunked(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    chunk_size,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    chunk_mode="serial",
    parallel_job_count=None,
    parallel_job_index=None,
    output_dir=None,
    checkpoint_prefix=None,
    sampler_backend="joint_nuts",
    channel_varying_kwargs=(),
    checkpoint_signature=None,
    sampling_workload_fingerprint=None,
    spectro_min_depth_ess=0.0,
    spectro_max_divergences=0,
    adaptive_fallback_model=None,
    adaptive_fallback_init_params=None,
    adaptive_fallback_resident_width=None,
    _joint_mcmc_runner_cache=None,
    _independent_mcmc_runner_cache=None,
    **model_kwargs,
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
    print(
        f"Running chunked MCMC: {num_lcs} channels in blocks of {chunk_size} "
        f"(mode={chunk_mode})"
    )

    # Set up checkpointing
    use_checkpoints = (output_dir is not None and checkpoint_prefix is not None)
    checkpoint_fingerprint = None
    if use_checkpoints:
        checkpoint_dir = os.path.join(output_dir, 'chunks')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logical_checkpoint_prefix = checkpoint_prefix
        checkpoint_fingerprint = _chunk_checkpoint_fingerprint(
            model,
            jax.random.key_data(key),
            t,
            yerr,
            indiv_y,
            init_params,
            chunk_size=chunk_size,
            sampler_backend=sampler_backend,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=checkpoint_signature,
            model_kwargs=model_kwargs,
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
            adaptive_fallback_resident_width=(
                adaptive_fallback_resident_width
            ),
        )
        checkpoint_prefix = (
            f"{logical_checkpoint_prefix}_cfg{checkpoint_fingerprint[:12]}"
        )
        manifest_path = _write_or_validate_checkpoint_manifest(
            checkpoint_dir,
            logical_checkpoint_prefix,
            checkpoint_prefix,
            checkpoint_fingerprint,
            checkpoint_signature,
        )
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(
            "Checkpoint fingerprint: "
            f"{checkpoint_fingerprint[:12]} ({manifest_path})"
        )
    elif chunk_mode != "serial":
        raise ValueError("chunk_mode='parallel' or 'combine' requires output_dir and checkpoint_prefix.")

    all_chunk_ranges = _chunk_ranges_for(num_lcs, chunk_size)

    if chunk_mode == "parallel":
        parallel_job_count, parallel_job_index = _resolve_parallel_chunk_job(
            parallel_job_count=parallel_job_count,
            parallel_job_index=parallel_job_index,
        )
        if parallel_job_count is None or parallel_job_index is None:
            raise ValueError(
                "chunk_mode='parallel' requires chunk_parallel_job_count/chunk_parallel_job_index "
                "or SLURM/JWSTJAXFIT chunk task environment variables."
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
        print(
            f"Parallel chunk job {parallel_job_index + 1}/{parallel_job_count} "
            f"assigned chunk indices {target_chunk_indices}"
        )
        if not target_chunk_indices:
            print("No chunks assigned to this job; exiting without computation.")
            return None
    else:
        target_chunk_indices = list(range(len(all_chunk_ranges)))

    if chunk_mode == "combine":
        missing = []
        corrupt = []
        samples_chunks = []
        for start, end in all_chunk_ranges:
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
        print(f"\nConcatenating {len(samples_chunks)} chunks...")
        samples = _concatenate_chunk_samples(samples_chunks)
        print(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
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
    for chunk_idx in target_chunk_indices:
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
            chunk_file = _chunk_checkpoint_path(
                checkpoint_dir, checkpoint_prefix, start, end
            )

        # Check if this chunk already exists
        if use_checkpoints:
            if os.path.exists(chunk_file):
                print(f"  chunk {start}:{end} - LOADING from checkpoint")
                try:
                    samples_chunk = _load_chunk_samples(chunk_file)
                except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
                    print(
                        f"  chunk {start}:{end} - checkpoint unreadable "
                        f"({type(error).__name__}); recomputing atomically"
                    )
                else:
                    samples_chunks.append(samples_chunk)
                    continue  # Skip computation for this chunk

        # Compute this chunk
        key_chunk = jax.random.fold_in(base_key, chunk_idx)
        print(f"  chunk {start}:{end} - COMPUTING ({end - start} channels)")
        diagnostics_path = (
            os.path.splitext(chunk_file)[0] + ".diagnostics.json"
            if use_checkpoints else None
        )
        if sampler_backend in {"independent_nuts", "independent_hmc"}:
            from models.independent_nuts import (
                build_independent_nuts_runner,
                get_samples_independent,
            )

            varying_for_chunk = tuple(
                name for name in channel_varying_kwargs
                if name in kwargs_chunk
            )
            if sampler_backend == "independent_hmc":
                from models.independent_hmc import (
                    build_independent_hmc_runner,
                    get_samples_independent_hmc,
                )
                independent_sampler = get_samples_independent_hmc
                independent_builder = build_independent_hmc_runner
                sampler_label = "fixed-step HMC"
                backend_options = {}
            else:
                independent_sampler = get_samples_independent
                independent_builder = build_independent_nuts_runner
                sampler_label = "NUTS"
                backend_options = {}
            padded_width = int(chunk_size)
            runner_key = (
                sampler_backend,
                padded_width,
                varying_for_chunk,
            )
            runner = independent_mcmc_runners.get(runner_key)
            if runner is None:
                runner = independent_builder(
                    model,
                    nuts_kwargs=nuts_kwargs,
                    mcmc_kwargs=mcmc_kwargs,
                    lane_width=padded_width,
                    channel_varying_kwargs=varying_for_chunk,
                    **backend_options,
                )
                independent_mcmc_runners[runner_key] = runner
            else:
                print(
                    f"  chunk {start}:{end} - reusing compiled independent "
                    f"{sampler_label} runner (padded width {padded_width})"
                )
            print(
                f"  chunk {start}:{end} - independent GPU {sampler_label} "
                f"({end - start} active lanes, padded width {padded_width})"
            )
            samples_chunk = independent_sampler(
                model, key_chunk, t, yerr_chunk, y_chunk, init_chunk,
                nuts_kwargs=nuts_kwargs, mcmc_kwargs=mcmc_kwargs,
                diagnostics_path=diagnostics_path,
                lane_width=padded_width,
                channel_varying_kwargs=varying_for_chunk,
                _runner=runner, **backend_options, **kwargs_chunk,
            )
        elif sampler_backend == "joint_nuts":
            runner = None
            if bool(dict(mcmc_kwargs or {}).get("jit_model_args", True)):
                resident_width = end - start
                runner = joint_mcmc_runners.get(resident_width)
                if runner is None:
                    runner, _, _ = _build_numpyro_mcmc(
                        model,
                        _tree_to_f64(init_chunk),
                        dict(nuts_kwargs or {}),
                        dict(mcmc_kwargs or {}),
                    )
                    joint_mcmc_runners[resident_width] = runner
                else:
                    print(
                        f"  chunk {start}:{end} - reusing compiled "
                        f"joint-NUTS runner (width {resident_width})"
                    )
            samples_chunk = get_samples(
                model, key_chunk, t, yerr_chunk, y_chunk, init_chunk,
                nuts_kwargs=nuts_kwargs, mcmc_kwargs=mcmc_kwargs,
                diagnostics_path=diagnostics_path,
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
            )
            gate_attempts.append({"sampler": sampler_backend, **gate})
            for attempt_index, fallback_backend in enumerate(
                _spectro_sampler_swap_order(sampler_backend), start=1
            ):
                if not failed.size:
                    break
                selected = failed.copy()
                print(
                    f"  chunk {start}:{end} - gate failed in local lanes "
                    f"{selected.tolist()}; re-running only those lanes with "
                    f"{fallback_backend}."
                )

                def _select_lanes(value):
                    if hasattr(value, "baseline") and (
                        hasattr(value, "differences") or hasattr(value, "uniform")
                    ):
                        return _take_by_channel(
                            value, selected, end - start
                        )
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
                if fallback_backend == "independent_hmc":
                    from models.independent_hmc import (
                        build_independent_hmc_runner,
                        get_samples_independent_hmc,
                    )
                    fallback_nuts.pop("max_tree_depth", None)
                    fallback_nuts.pop("laplace_max_tree_depth", None)
                    fallback_nuts.update(
                        mass_matrix="laplace", laplace_warmup=150,
                        laplace_target_accept=0.85,
                        laplace_start_at_map=True, num_steps=8,
                        trajectory_jitter=0.25,
                    )
                    fallback_runner_key = (
                        fallback_backend, int(chunk_size), varying_for_chunk,
                    )
                    fallback_runner = independent_mcmc_runners.get(
                        fallback_runner_key
                    )
                    if fallback_runner is None:
                        fallback_runner = build_independent_hmc_runner(
                            model, nuts_kwargs=fallback_nuts,
                            mcmc_kwargs=mcmc_kwargs, lane_width=chunk_size,
                            channel_varying_kwargs=varying_for_chunk,
                        )
                        independent_mcmc_runners[fallback_runner_key] = (
                            fallback_runner
                        )
                    else:
                        print(
                            f"  chunk {start}:{end} - reusing compiled "
                            f"fallback HMC-8 runner (padded width {chunk_size})"
                        )
                    fallback = get_samples_independent_hmc(
                        model, jax.random.fold_in(key_chunk, 99173 + attempt_index),
                        t, yerr_chunk[selected], y_chunk[selected], fallback_init,
                        nuts_kwargs=fallback_nuts, mcmc_kwargs=mcmc_kwargs,
                        diagnostics_path=fallback_path, lane_width=chunk_size,
                        channel_varying_kwargs=varying_for_chunk,
                        _runner=fallback_runner, **fallback_kwargs,
                    )
                elif fallback_backend == "independent_nuts":
                    from models.independent_nuts import (
                        build_independent_nuts_runner,
                        get_samples_independent,
                    )
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
                    fallback_runner_key = (
                        fallback_backend, int(chunk_size), varying_for_chunk,
                    )
                    fallback_runner = independent_mcmc_runners.get(
                        fallback_runner_key
                    )
                    if fallback_runner is None:
                        fallback_runner = build_independent_nuts_runner(
                            model, nuts_kwargs=fallback_nuts,
                            mcmc_kwargs=mcmc_kwargs, lane_width=chunk_size,
                            channel_varying_kwargs=varying_for_chunk,
                        )
                        independent_mcmc_runners[fallback_runner_key] = (
                            fallback_runner
                        )
                    else:
                        print(
                            f"  chunk {start}:{end} - reusing compiled "
                            f"fallback Laplace-NUTS runner "
                            f"(padded width {chunk_size})"
                        )
                    fallback = get_samples_independent(
                        model, jax.random.fold_in(key_chunk, 99173 + attempt_index),
                        t, yerr_chunk[selected], y_chunk[selected], fallback_init,
                        nuts_kwargs=fallback_nuts, mcmc_kwargs=mcmc_kwargs,
                        diagnostics_path=fallback_path, lane_width=chunk_size,
                        channel_varying_kwargs=varying_for_chunk,
                        _runner=fallback_runner, **fallback_kwargs,
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
                    adaptive_nuts.update(
                        target_accept_prob=0.95, max_tree_depth=10
                    )
                    adaptive_width = (
                        int(selected.size)
                        if adaptive_fallback_resident_width is None else
                        int(adaptive_fallback_resident_width)
                    )
                    if adaptive_width < 1:
                        raise ValueError(
                            "adaptive_fallback_resident_width must be positive"
                        )
                    adaptive_runner_key = (
                        "adaptive_fallback", adaptive_width,
                    )
                    adaptive_runner = joint_mcmc_runners.get(
                        adaptive_runner_key
                    )
                    adaptive_batches = []
                    adaptive_divergences = []
                    for batch_index, batch_start in enumerate(
                        range(0, int(selected.size), adaptive_width)
                    ):
                        batch_stop = min(
                            batch_start + adaptive_width, int(selected.size)
                        )
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
                                        None if field is None else jnp.concatenate((
                                            field,
                                            jnp.repeat(field[-1:], pad, axis=0),
                                        ))
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
                                            array[-1:],
                                            adaptive_width - batch_size,
                                            axis=0,
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
                            print(
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
                                key_chunk,
                                99173 + attempt_index + 1009 * batch_index,
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
                        adaptive_divergences.extend(
                            [batch_divergence] * batch_size
                        )
                    fallback = jax.tree.map(
                        lambda *values: jnp.concatenate(values, axis=1),
                        *adaptive_batches,
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
                    fallback, fallback_path, spectro_min_depth_ess,
                    spectro_max_divergences,
                )
                gate_attempts.append({
                    "sampler": fallback_backend,
                    "input_local_lanes": selected.tolist(),
                    **fallback_gate,
                })
                samples_chunk = {
                    name: (
                        jnp.asarray(values).at[:, selected].set(fallback[name])
                        if name in fallback else values
                    )
                    for name, values in samples_chunk.items()
                }
                for lane in selected:
                    sampler_used[int(lane)] = fallback_backend
                failed = selected[fallback_failed]
            if failed.size:
                raise RuntimeError(
                    f"chunk {start}:{end} failed the spectroscopic ESS/divergence "
                    f"gate after adaptive NUTS in local lanes {failed.tolist()}"
                )
        samples_chunk = _SamplerSamples(samples_chunk, sampler_used=sampler_used)
        _annotate_sampler_diagnostics(
            diagnostics_path,
            sampling_workload_fingerprint=sampling_workload_fingerprint,
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
                    {"samples": dict(samples_chunk),
                     "sampler_used": samples_chunk.sampler_used}, f
                )
            os.replace(temporary_chunk, chunk_file)
            print(f"  chunk {start}:{end} - SAVED checkpoint")

    if chunk_mode == "parallel":
        existing = [
            os.path.exists(_chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end))
            for start, end in all_chunk_ranges
        ]
        done = sum(existing)
        print(
            f"Parallel chunk job complete: {done}/{len(all_chunk_ranges)} checkpoint files present. "
            "Run the compatibility combine step after all chunks are finished."
        )
        return None

    # Concatenate all chunks
    print(f"\nConcatenating {len(samples_chunks)} chunks...")
    samples = _concatenate_chunk_samples(samples_chunks)
    print(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
    return samples


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
    dump_metadata=None,
    adaptive_fallback_model=None,
    adaptive_fallback_init_params=None,
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
                    field,
                    [(0, 0)] * (field.ndim - 1) + [(0, pad)],
                    constant_values=0.0,
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
                array,
                [(0, 0)] * (array.ndim - 1) + [(0, pad)],
                constant_values=0.0,
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

    _maybe_dump_sampler_inputs(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        chunk_size=chunk_size,
        use_chunked=use_chunked,
        sampler_backend=sampler_backend,
        channel_varying_kwargs=channel_varying_kwargs,
        checkpoint_signature=checkpoint_signature,
        dump_metadata=dump_metadata,
        model_kwargs=model_kwargs,
    )
    sampling_workload_fingerprint = _sampling_workload_fingerprint(
        model,
        t,
        yerr,
        indiv_y,
        init_params,
        sampler_backend=sampler_backend,
        nuts_kwargs=nuts_kwargs,
        channel_varying_kwargs=channel_varying_kwargs,
        checkpoint_signature=checkpoint_signature,
        model_kwargs=model_kwargs,
    )
    if use_chunked:
        effective_chunk_size = 1 if chunk_size is None else int(chunk_size)
        if effective_chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 when chunked sampling is enabled.")
        return get_samples_chunked(
            model,
            key,
            t,
            yerr,
            indiv_y,
            init_params,
            effective_chunk_size,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            chunk_mode=chunk_mode,
            parallel_job_count=parallel_job_count,
            parallel_job_index=parallel_job_index,
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
            sampler_backend=sampler_backend,
            channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=checkpoint_signature,
            sampling_workload_fingerprint=sampling_workload_fingerprint,
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
            adaptive_fallback_model=adaptive_fallback_model,
            adaptive_fallback_init_params=adaptive_fallback_init_params,
            **model_kwargs,
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
            mcmc_kwargs=mcmc_kwargs,
            lane_width=int(indiv_y.shape[0]),
            channel_varying_kwargs=tuple(
                name for name in channel_varying_kwargs
                if name in model_kwargs
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
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        **model_kwargs,
    )

"""Geometry helpers."""

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

def _geometry_chain_quality(grouped_samples):
    """Return exact-MCMC quality metrics for white-light science sites."""
    aliases = {
        "t0": ("t0_0", "t0"),
        "b": ("b_0", "b"),
        "duration": ("logD_0", "duration_0", "duration", "logD"),
        "rors": ("rors_0", "rors"),
    }
    if "_geometry_fixed" in grouped_samples:
        aliases = {}
    for name in grouped_samples:
        if re.fullmatch(
            r"_(?:eclipse_depth|dayside_flux|nightside_flux|hotspot_offset|stellar_spot_contrast)_\d+",
            name,
        ):
            aliases[name.removeprefix("_")] = (name,)
    ess = {}
    rhat = {}
    for label, names in aliases.items():
        name = next((candidate for candidate in names if candidate in grouped_samples), None)
        if name is None:
            continue
        values = jnp.asarray(grouped_samples[name], dtype=jnp.float64)
        if values.ndim < 2:
            values = values.reshape((1, values.shape[0]))
        site_ess = numpyro.diagnostics.effective_sample_size(values)
        ess_values = np.asarray(jax.device_get(site_ess), dtype=float)
        ess[label] = (
            float(np.min(ess_values))
            if np.all(np.isfinite(ess_values)) else np.nan
        )
        if values.shape[0] > 1:
            site_rhat = numpyro.diagnostics.split_gelman_rubin(values)
            rhat_values = np.asarray(jax.device_get(site_rhat), dtype=float)
            rhat[label] = (
                float(np.max(rhat_values))
                if np.all(np.isfinite(rhat_values)) else np.nan
            )
    return ess, rhat


def _continue_mcmc_until_geometry_gate(
    mcmc,
    rng_key,
    run_args,
    run_kwargs,
    *,
    min_ess=400.0,
    max_divergences=0,
    max_extra_blocks=3,
    failfast_ess=50.0,
):
    """Append post-warmup draws until the white-light geometry gate passes."""
    sample_blocks = [jax.device_get(mcmc.get_samples(group_by_chain=True))]
    extra_blocks = [jax.device_get(mcmc.get_extra_fields(group_by_chain=True))]
    extra_count = 0
    while True:
        grouped = {
            name: np.concatenate([np.asarray(block[name]) for block in sample_blocks], axis=1)
            for name in sample_blocks[0]
        }
        grouped_extra = {
            name: np.concatenate([np.asarray(block[name]) for block in extra_blocks], axis=1)
            for name in extra_blocks[0]
        }
        ess, rhat = _geometry_chain_quality(grouped)
        divergences = int(np.sum(np.asarray(grouped_extra.get("diverging", []))))
        finite_ess = bool(ess) and bool(np.all(np.isfinite(list(ess.values()))))
        passes = finite_ess and min(ess.values()) >= float(min_ess)
        passes = passes and divergences <= int(max_divergences)
        print(
            "White-light quality gate: "
            f"ESS={ess}, rhat={rhat or 'n/a'}, divergences={divergences}, "
            f"pass={passes}, extra_blocks={extra_count}."
        )
        failfast = bool(
            extra_count == 0
            and ess
            and min(ess.values()) < float(failfast_ess)
        )
        if failfast:
            print(
                "WHITE-LIGHT QUALITY GATE FAIL-FAST: first-block minimum "
                f"science ESS is below {float(failfast_ess):g}; skipping "
                "extension blocks."
            )
        if passes or failfast or extra_count >= int(max_extra_blocks):
            flat = {
                name: values.reshape((-1,) + values.shape[2:])
                for name, values in grouped.items()
            }
            flat_extra = {
                name: values.reshape((-1,) + values.shape[2:])
                for name, values in grouped_extra.items()
            }
            return flat, grouped, flat_extra, {
                "whitelight_geometry_bulk_ess": ess,
                "whitelight_geometry_rhat": rhat,
                "whitelight_quality_gate_passed": passes,
                "whitelight_failfast_triggered": failfast,
                "whitelight_extra_blocks": extra_count,
                "whitelight_total_retained_draws": int(
                    next(iter(flat.values())).shape[0]
                ),
            }
        extra_count += 1
        print(
            "WHITE-LIGHT QUALITY GATE NOT MET; continuing the same warmed "
            f"chain with retained block {extra_count}/{max_extra_blocks}."
        )
        mcmc.post_warmup_state = mcmc.last_state
        rng_key, block_key = jax.random.split(rng_key)
        continuation_kwargs = dict(run_kwargs)
        continuation_kwargs.pop("init_params", None)
        mcmc.run(block_key, *run_args, **continuation_kwargs)
        sample_blocks.append(jax.device_get(mcmc.get_samples(group_by_chain=True)))
        extra_blocks.append(jax.device_get(mcmc.get_extra_fields(group_by_chain=True)))


def _posterior_num_draws(posterior_samples):
    """Validate a flattened posterior sample mapping and return its draw count."""
    if not posterior_samples:
        raise ValueError("Posterior samples are empty.")
    draw_counts = {}
    for name, values in posterior_samples.items():
        array = np.asarray(jax.device_get(values))
        if array.ndim == 0:
            raise ValueError(
                f"Posterior site '{name}' is missing a leading draw dimension."
            )
        draw_counts[name] = int(array.shape[0])
    unique_counts = set(draw_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            "Posterior sites have inconsistent draw counts: "
            f"{draw_counts}."
        )
    num_draws = unique_counts.pop()
    if num_draws < 1:
        raise ValueError("Posterior samples contain no retained draws.")
    return num_draws


def _geometry_from_white_light_medians(bestfit_params, period):
    """Build the compatibility handoff from the reported posterior medians."""
    periods = np.atleast_1d(np.asarray(period, dtype=np.float64))
    geometry = {
        "period": periods.tolist(),
        "duration": np.atleast_1d(np.asarray(bestfit_params["duration"], dtype=float)).tolist(),
        "t0": np.atleast_1d(np.asarray(bestfit_params["t0"], dtype=float)).tolist(),
        "b": np.atleast_1d(np.asarray(bestfit_params["b"], dtype=float)).tolist(),
        "a_rs": np.atleast_1d(np.asarray(bestfit_params["a_rs"], dtype=float)).tolist(),
        "cos_i": np.atleast_1d(np.asarray(bestfit_params["cos_i"], dtype=float)).tolist(),
        "rors": np.atleast_1d(np.asarray(bestfit_params["rors"], dtype=float)).tolist(),
    }
    geometry["inclination"] = np.arccos(
        np.clip(np.asarray(geometry["cos_i"], dtype=float), 0.0, 1.0)
    ).tolist()
    lengths = {name: len(values) for name, values in geometry.items()}
    if set(lengths.values()) != {len(periods)}:
        raise RuntimeError(
            "White-light median geometry has inconsistent planet dimensions: "
            f"{lengths}."
        )
    if not all(np.all(np.isfinite(values)) for values in geometry.values()):
        raise RuntimeError("White-light median geometry contains non-finite values.")
    return geometry


def _selected_geometry_primitives(samples, num_planets):
    """Serialize the primitive coordinates that identify a selected draw."""
    names = []
    for planet_index in range(int(num_planets)):
        names.extend(
            f"{base}_{planet_index}"
            for base in (
                "t0", "rors", "_b", "logD", "log_a_rs", "duration",
                "b", "a_rs", "cos_i", "inc",
            )
        )
    primitives = {}
    for name in names:
        if name not in samples:
            continue
        value = np.asarray(jax.device_get(samples[name]))
        primitives[name] = (
            float(value.reshape(-1)[0])
            if value.size == 1
            else value.reshape(-1).astype(float).tolist()
        )
    return primitives


def _whitelight_geometry_handoff_fingerprint(payload):
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION,
            "target_revision": WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION,
            "payload": payload,
        },
    )
    return hasher.hexdigest()


def _write_whitelight_geometry_handoff(path, payload):
    """Atomically persist a self-validating white-light geometry handoff."""
    payload = dict(payload)
    payload["schema_version"] = WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION
    payload["target_revision"] = WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint_sha256"
    }
    payload["artifact_fingerprint_sha256"] = (
        _whitelight_geometry_handoff_fingerprint(fingerprint_payload)
    )
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return payload


def _load_whitelight_geometry_handoff(
    path,
    *,
    expected_posterior_fingerprint,
    expected_estimator,
):
    """Load a handoff only if its schema, fingerprint, and target all match."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version")
        != WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION
        or payload.get("target_revision")
        != WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION
        or payload.get("posterior_fingerprint_sha256")
        != expected_posterior_fingerprint
        or payload.get("estimator") != expected_estimator
    ):
        return None
    expected_artifact_fingerprint = payload.get("artifact_fingerprint_sha256")
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint_sha256"
    }
    if expected_artifact_fingerprint != _whitelight_geometry_handoff_fingerprint(
        fingerprint_payload
    ):
        return None
    geometry = payload.get("geometry")
    required = {
        "period", "duration", "t0", "b", "a_rs", "cos_i",
        "inclination", "rors",
    }
    if not isinstance(geometry, dict) or not required.issubset(geometry):
        return None
    try:
        lengths = {
            len(np.atleast_1d(np.asarray(geometry[name], dtype=float)))
            for name in required
        }
        finite = all(
            np.all(np.isfinite(np.asarray(geometry[name], dtype=float)))
            for name in required
        )
    except (TypeError, ValueError):
        return None
    if len(lengths) != 1 or next(iter(lengths), 0) < 1 or not finite:
        return None
    if any(
        payload.get(name) is not None
        for name in (
            "selected_flat_draw_index", "selected_chain_index",
            "selected_draw_index", "summed_data_log_likelihood",
        )
    ):
        return None
    return payload

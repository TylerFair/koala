"""Harmonica Products helpers."""

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
from .artifacts import _atomic_dataframe_csv, _atomic_savez_compressed
from .outputs import get_asym_errors

def _coerce_wavelength_axis(wavelengths, wavelength_err=None):
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wavelength_err is None:
        wl_err = np.zeros_like(wl)
    else:
        wl_err = np.atleast_1d(np.asarray(wavelength_err, dtype=float))
        if wl_err.size == 1 and wl.size > 1:
            wl_err = np.repeat(wl_err, wl.size)
        if wl_err.size != wl.size:
            raise ValueError(
                f"wavelength_err must be scalar or length {wl.size}, got {wl_err.shape}."
            )
    return wl, wl_err


_HARMONICA_LIMB_LOGGER = logging.getLogger(__name__)


_LEGACY_LIMB_SCHEMA_WARNING_EMITTED = False


def _harmonica_frac_site(name):
    return f"{name}_frac"


def _harmonica_coeff_to_frac(coeff, radius):
    radius_arr = np.asarray(radius, dtype=float)
    scale = max(float(np.nanmin(np.abs(radius_arr))), 1e-6)
    return float(coeff) / scale


def _harmonica_coeff_array_to_frac(coeff, radius):
    coeff_arr = np.asarray(coeff, dtype=float)
    radius_arr = np.asarray(radius, dtype=float)
    return coeff_arr / np.maximum(np.abs(radius_arr), 1e-6)


def _harmonica_coeff_array_to_delta_r(coeff):
    return 2.0 * np.asarray(coeff, dtype=float)


def _harmonica_coefficients_to_half_area_init(a0, a1):
    """Convert WL coefficients to an interior, sampler-supported init."""
    area_radius, q = harmonica_half_area_area_radius_and_q(a0, a1)
    area_radius = np.asarray(area_radius, dtype=float)
    q = np.asarray(q, dtype=float)
    # Stay materially inside the truncated distribution support.  Merely
    # applying ``nextafter`` to the endpoint can still round through the
    # inverse transform to +/-inf during NumPyro initialization.
    interior_limit = float(HARMONICA_HALF_AREA_CONVEX_Q_LIMIT) * (1.0 - 1e-10)
    finite_radius = np.sqrt(
        np.maximum(
            0.0,
            np.asarray(a0, dtype=float) ** 2
            + 0.5 * np.asarray(a1, dtype=float) ** 2,
        )
    )
    radius_low = HARMONICA_SPECTRO_RORS_MIN * (1.0 + 1e-10)
    radius_high = HARMONICA_SPECTRO_RORS_MAX * (1.0 - 1e-10)
    area_radius = np.where(np.isfinite(area_radius), area_radius, finite_radius)
    area_radius = np.clip(
        np.nan_to_num(
            area_radius,
            nan=radius_low,
            posinf=radius_high,
            neginf=radius_low,
        ),
        radius_low,
        radius_high,
    )
    q = np.nan_to_num(q, nan=0.0, posinf=interior_limit, neginf=-interior_limit)
    return area_radius, np.clip(q, -interior_limit, interior_limit)


def _validate_harmonica_spectro_parameterization(value, max_harmonic_order):
    """Normalize and validate the configured Harmonica spectroscopic shape."""
    value = str(value).lower()
    if value not in _VALID_HARMONICA_SPECTRO_PARAMETERIZATIONS:
        choices = "{'fractional', 'delta_r', 'half_area'}"
        raise ValueError(
            "flags.harmonica_spectro_parameterization must be one of "
            f"{choices}. Received '{value}'."
        )
    if value == 'half_area' and int(max_harmonic_order) != 1:
        raise ValueError(
            "flags.harmonica_spectro_parameterization='half_area' requires "
            "flags.harmonica_max_order: 1."
        )
    return value


def _harmonica_model_radius_samples(samples, parameterization):
    """Select the radius consumed by the Harmonica forward model.

    ``half_area`` deliberately stores the total-area-equivalent radius in the
    usual ``rors`` site, while its deterministic ``a0`` is the mean polar
    radius required by the Harmonica transmission string.  Legacy
    parameterizations continue to use ``rors`` directly.
    """
    if parameterization == 'half_area':
        if 'a0' not in samples:
            raise KeyError(
                "half_area posterior samples must contain deterministic 'a0'."
            )
        return samples['a0']
    return samples['rors']


def _harmonica_checkpoint_prefix(base_prefix, transit_engine, parameterization):
    """Keep opt-in half-area chunks separate from legacy checkpoint families."""
    if transit_engine == 'harmonica' and parameterization == 'half_area':
        return f"{base_prefix}_half_area"
    return base_prefix


def _harmonica_spectro_artifact_stem(base_stem, transit_engine, parameterization):
    """Isolate half-area LR reuse artifacts from legacy parameterizations."""
    return _harmonica_checkpoint_prefix(
        base_stem, transit_engine, parameterization
    )


def _coerce_harmonica_sample_matrix(samples, planet_index=0):
    arr = np.asarray(samples)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 3:
        return arr[:, :, planet_index]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported harmonica sample shape: {arr.shape}")


def _extract_harmonica_limb_samples(rors_samples, harmonic_samples, planet_index=0):
    a0 = np.asarray(rors_samples)

    if a0.ndim == 1:
        a0 = a0[:, np.newaxis]
    elif a0.ndim == 3:
        a0 = a0[:, :, planet_index]
    elif a0.ndim != 2:
        raise ValueError(f"Unsupported rors sample shape: {a0.shape}")

    coeff_samples = {}
    for name in HARMONICA_ODD_HARMONICS:
        if name not in harmonic_samples:
            continue
        coeff = _coerce_harmonica_sample_matrix(
            harmonic_samples[name], planet_index=planet_index
        )
        if coeff.shape[1] == 1 and a0.shape[1] > 1:
            coeff = np.repeat(coeff, a0.shape[1], axis=1)
        if a0.shape != coeff.shape:
            raise ValueError(
                f"rors and {name} sample shapes must match after broadcasting, got {a0.shape} and {coeff.shape}."
            )
        coeff_samples[name] = coeff

    return a0, coeff_samples


def _sum_harmonica_odd_samples(coeff_samples):
    if not coeff_samples:
        return 0.0
    total = None
    for coeff in coeff_samples.values():
        total = coeff if total is None else total + coeff
    return total


def _harmonica_limb_product_samples(a0_samples, coeff_samples):
    """Return neutral-index half-area radii, depths, and endpoint diagnostics.

    Harmonica's odd-cosine transmission string is

    ``r(theta) = a0 + sum_n a_n cos(n theta)``.

    Index one is the interval centred on ``theta=pi`` and index two is the
    interval centred on ``theta=0``.  These indices deliberately make no
    morning/evening or leading/trailing claim; that assignment requires an
    external orbital-geometry convention.  Each representative depth is
    twice that half's area divided by pi, so it is directly comparable to the
    depth of a circular semicircle.  For N_c=1 this gives

    ``D_two = a0**2 + a1**2/2 + 4*a0*a1/pi`` and
    ``D_one = a0**2 + a1**2/2 - 4*a0*a1/pi``.

    The implementation also handles the supported higher odd cosine terms
    exactly.  Orthogonality makes their squared contributions add in
    quadrature, while their signed half-area coefficient is
    ``sum_n a_n sin(n*pi/2)/n``.
    """
    a0 = np.asarray(a0_samples, dtype=float)
    odd_sum = np.zeros_like(a0)
    odd_square_sum = np.zeros_like(a0)
    area_asymmetry_coefficient = np.zeros_like(a0)

    for name, coeff_samples_for_order in coeff_samples.items():
        order = int(name[1:])
        if order % 2 != 1:
            raise ValueError(
                f"Harmonica limb products require odd cosine terms; got {name}."
            )
        coeff = np.asarray(coeff_samples_for_order, dtype=float)
        if coeff.shape != a0.shape:
            raise ValueError(
                f"a0 and {name} sample shapes must match, got {a0.shape} and {coeff.shape}."
            )
        odd_sum = odd_sum + coeff
        odd_square_sum = odd_square_sum + coeff**2
        half_area_sign = (-1.0) ** ((order - 1) // 2)
        area_asymmetry_coefficient = (
            area_asymmetry_coefficient + coeff * half_area_sign / order
        )

    total_area_depth = a0**2 + 0.5 * odd_square_sum
    half_area_contrast = (4.0 / np.pi) * a0 * area_asymmetry_coefficient
    two_endpoint_radius = a0 + odd_sum
    one_endpoint_radius = a0 - odd_sum
    depth_one = total_area_depth - half_area_contrast
    depth_two = total_area_depth + half_area_contrast

    return {
        "rp_one": np.sqrt(np.maximum(depth_one, 0.0)),
        "rp_two": np.sqrt(np.maximum(depth_two, 0.0)),
        "depth_one": depth_one,
        "depth_two": depth_two,
        "depth_total_area": total_area_depth,
        "rp_one_endpoint": one_endpoint_radius,
        "rp_two_endpoint": two_endpoint_radius,
        "depth_one_endpoint": one_endpoint_radius**2,
        "depth_two_endpoint": two_endpoint_radius**2,
        "asymmetry_coefficient": area_asymmetry_coefficient,
        "endpoint_delta_r": two_endpoint_radius - one_endpoint_radius,
    }


def _harmonica_percentile_summary(samples):
    """Return posterior median and asymmetric 16th/84th percentile errors."""
    median = np.nanpercentile(samples, 50, axis=0)
    err_lo = median - np.nanpercentile(samples, 16, axis=0)
    err_hi = np.nanpercentile(samples, 84, axis=0) - median
    return median, err_lo, err_hi


def _harmonica_r_vector_from_values(a0, coeff_values):
    active_orders = [
        int(name[1:])
        for name in HARMONICA_ODD_HARMONICS
        if name in coeff_values
    ]
    highest_order = max(active_orders, default=1)
    r = np.zeros(2 * highest_order + 1, dtype=float)
    r[0] = a0
    for name, value in coeff_values.items():
        order = int(name[1:])
        r[2 * order - 1] = value
    return r


def _harmonica_sample_payload(samples):
    return {name: samples[name] for name in HARMONICA_ODD_HARMONICS if name in samples}


def _has_harmonica_odd_samples(samples):
    return any(name in samples for name in HARMONICA_ODD_HARMONICS)


def _augment_harmonica_params(params, a_rs, ecc, omega, a1=None, a3=None, a5=None,
                              c_ld=None, alpha_ld=None, u1_ld=None, u2_ld=None):
    out = dict(params)
    out["a_rs"] = jnp.asarray(a_rs)
    out["ecc"] = jnp.asarray(ecc)
    out["omega"] = jnp.asarray(omega)
    if a1 is not None:
        out["a1"] = jnp.asarray(a1)
    if a3 is not None:
        out["a3"] = jnp.asarray(a3)
    if a5 is not None:
        out["a5"] = jnp.asarray(a5)
    if c_ld is not None:
        out["c_ld"] = jnp.asarray(c_ld)
    if alpha_ld is not None:
        out["alpha_ld"] = jnp.asarray(alpha_ld)
    if u1_ld is not None:
        out["u1_ld"] = jnp.asarray(u1_ld)
    if u2_ld is not None:
        out["u2_ld"] = jnp.asarray(u2_ld)
    return out


def _harmonica_cosi_from_b(b, a_rs, ecc, omega):
    return jnp.clip(
        harmonica_cos_i_from_geometry(
            jnp.asarray(b, dtype=jnp.float64),
            jnp.asarray(a_rs, dtype=jnp.float64),
            ecc=jnp.asarray(ecc, dtype=jnp.float64),
            omega=jnp.asarray(omega, dtype=jnp.float64),
        ),
        0.0,
        1.0,
    )


def _add_rp_summary_from_depth(frame, depth_name, rp_name):
    """Derive dimensionless radius summaries from depth summaries in ppm."""
    depth_median = np.asarray(frame[f"{depth_name}_median"], dtype=float) / 1e6
    depth_low = np.asarray(frame[f"{depth_name}_err_lo"], dtype=float) / 1e6
    depth_high = np.asarray(frame[f"{depth_name}_err_hi"], dtype=float) / 1e6
    rp_median = np.sqrt(np.maximum(depth_median, 0.0))
    frame[f"{rp_name}_median"] = rp_median
    frame[f"{rp_name}_err_lo"] = rp_median - np.sqrt(
        np.maximum(depth_median - depth_low, 0.0)
    )
    frame[f"{rp_name}_err_hi"] = np.sqrt(
        np.maximum(depth_median + depth_high, 0.0)
    ) - rp_median


def load_harmonica_limb_dataframe(source):
    """Load a schema-v3 limb CSV, or normalize a schema-v2 CSV in memory.

    In schema v3, ``one`` and ``two`` are arbitrary angular indices. Mapping
    them to morning/evening or leading/trailing requires external knowledge of
    the system's orbital geometry. For legacy v2, morning maps to index one and
    evening maps to index two, exactly as documented; the source file is never
    modified.
    """
    global _LEGACY_LIMB_SCHEMA_WARNING_EMITTED
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = pd.read_csv(source)
    if "limb_product_schema_version" not in frame:
        raise ValueError("Limb product is missing limb_product_schema_version.")
    versions = np.unique(
        np.asarray(frame["limb_product_schema_version"], dtype=int)
    )
    if versions.size != 1:
        raise ValueError(f"Limb product mixes schema versions: {versions.tolist()}.")
    version = int(versions[0])
    if version == 2:
        if not _LEGACY_LIMB_SCHEMA_WARNING_EMITTED:
            _HARMONICA_LIMB_LOGGER.warning(
                "Reading legacy Harmonica limb-product schema v2: mapping "
                "depth_morning_* to depth_one_* and depth_evening_* to "
                "depth_two_* in memory. The source file is unchanged."
            )
            _LEGACY_LIMB_SCHEMA_WARNING_EMITTED = True
        legacy_mappings = {
            "depth_morning": "depth_one",
            "depth_evening": "depth_two",
            "depth_trailing_endpoint": "depth_one_endpoint",
            "depth_leading_endpoint": "depth_two_endpoint",
        }
        for old_name, new_name in legacy_mappings.items():
            for suffix in ("median", "err_lo", "err_hi"):
                old_column = f"{old_name}_{suffix}"
                if old_column in frame and f"{new_name}_{suffix}" not in frame:
                    frame[f"{new_name}_{suffix}"] = frame[old_column]
        for depth_name, rp_name in (
            ("depth_one", "rp_one"),
            ("depth_two", "rp_two"),
            ("depth_one_endpoint", "rp_one_endpoint"),
            ("depth_two_endpoint", "rp_two_endpoint"),
        ):
            if (
                f"{depth_name}_median" in frame
                and f"{rp_name}_median" not in frame
            ):
                _add_rp_summary_from_depth(frame, depth_name, rp_name)
        return frame
    if version != HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Harmonica limb-product schema v{version}; "
            f"this reader supports v2 and v{HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION}."
        )
    required = {
        f"{name}_{suffix}"
        for name in ("rp_one", "rp_two", "depth_one", "depth_two", "depth_total_area")
        for suffix in ("median", "err_lo", "err_hi")
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Schema-v3 limb product is missing columns: {missing}.")
    return frame


def build_harmonica_limb_dataframe(wavelengths, wavelength_err, rors_samples, harmonic_samples,
                                   bandpass_min=None, bandpass_max=None, planet_index=0):
    wl_arr, wl_err_arr = _coerce_wavelength_axis(wavelengths, wavelength_err)
    a0_samp, coeff_samp = _extract_harmonica_limb_samples(
        rors_samples, harmonic_samples, planet_index=planet_index
    )
    if wl_arr.size != a0_samp.shape[1]:
        raise ValueError(
            f"Number of wavelengths ({wl_arr.size}) does not match sample axis ({a0_samp.shape[1]})."
        )

    limb_products = _harmonica_limb_product_samples(a0_samp, coeff_samp)
    depth_products_ppm = {
        name: values * 1e6
        for name, values in limb_products.items()
        if name.startswith("depth_")
    }
    a0_depth_samp = (a0_samp**2) * 1e6

    product_summaries = {
        name: _harmonica_percentile_summary(values)
        for name, values in depth_products_ppm.items()
    }
    one_med, one_lo, one_hi = product_summaries["depth_one"]
    two_med, two_lo, two_hi = product_summaries["depth_two"]
    total_med, total_lo, total_hi = product_summaries["depth_total_area"]
    one_endpoint_med, one_endpoint_lo, one_endpoint_hi = (
        product_summaries["depth_one_endpoint"]
    )
    two_endpoint_med, two_endpoint_lo, two_endpoint_hi = (
        product_summaries["depth_two_endpoint"]
    )
    rp_summaries = {
        name: _harmonica_percentile_summary(limb_products[name])
        for name in ("rp_one", "rp_two", "rp_one_endpoint", "rp_two_endpoint")
    }
    rp_one_med, rp_one_lo, rp_one_hi = rp_summaries["rp_one"]
    rp_two_med, rp_two_lo, rp_two_hi = rp_summaries["rp_two"]
    rp_one_endpoint_med, rp_one_endpoint_lo, rp_one_endpoint_hi = rp_summaries[
        "rp_one_endpoint"
    ]
    rp_two_endpoint_med, rp_two_endpoint_lo, rp_two_endpoint_hi = rp_summaries[
        "rp_two_endpoint"
    ]
    a0_depth_med, a0_depth_lo, a0_depth_hi = _harmonica_percentile_summary(
        a0_depth_samp
    )

    asymmetry_med, asymmetry_lo, asymmetry_hi = _harmonica_percentile_summary(
        limb_products["asymmetry_coefficient"]
    )
    delta_r_med, delta_r_lo, delta_r_hi = _harmonica_percentile_summary(
        limb_products["endpoint_delta_r"]
    )

    # Raw Rp/Rs statistics (needed for transmission string plots)
    a0_raw_med, a0_raw_lo, a0_raw_hi = _harmonica_percentile_summary(a0_samp)

    limb_df = pd.DataFrame({
        "wavelength": wl_arr,
        "wavelength_err": wl_err_arr,
        "planet_index": int(planet_index),
        "limb_product_schema_version": HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION,
        "limb_product_convention": HARMONICA_LIMB_PRODUCT_CONVENTION,
        # Neutral indices: assigning physical hemisphere names requires external geometry.
        "rp_one_median": rp_one_med,
        "rp_one_err_lo": rp_one_lo,
        "rp_one_err_hi": rp_one_hi,
        "rp_two_median": rp_two_med,
        "rp_two_err_lo": rp_two_lo,
        "rp_two_err_hi": rp_two_hi,
        "depth_one_median": one_med,
        "depth_one_err_lo": one_lo,
        "depth_one_err_hi": one_hi,
        "depth_two_median": two_med,
        "depth_two_err_lo": two_lo,
        "depth_two_err_hi": two_hi,
        "depth_total_area_median": total_med,
        "depth_total_area_err_lo": total_lo,
        "depth_total_area_err_hi": total_hi,
        # Index-one endpoint is theta=pi; index-two endpoint is theta=0.
        "rp_one_endpoint_median": rp_one_endpoint_med,
        "rp_one_endpoint_err_lo": rp_one_endpoint_lo,
        "rp_one_endpoint_err_hi": rp_one_endpoint_hi,
        "rp_two_endpoint_median": rp_two_endpoint_med,
        "rp_two_endpoint_err_lo": rp_two_endpoint_lo,
        "rp_two_endpoint_err_hi": rp_two_endpoint_hi,
        "depth_one_endpoint_median": one_endpoint_med,
        "depth_one_endpoint_err_lo": one_endpoint_lo,
        "depth_one_endpoint_err_hi": one_endpoint_hi,
        "depth_two_endpoint_median": two_endpoint_med,
        "depth_two_endpoint_err_lo": two_endpoint_lo,
        "depth_two_endpoint_err_hi": two_endpoint_hi,
        "asymmetry_coefficient_median": asymmetry_med,
        "asymmetry_coefficient_err_lo": asymmetry_lo,
        "asymmetry_coefficient_err_hi": asymmetry_hi,
        "endpoint_delta_r_median": delta_r_med,
        "endpoint_delta_r_err_lo": delta_r_lo,
        "endpoint_delta_r_err_hi": delta_r_hi,
        # a0**2 is retained for compatibility but is not the total silhouette area.
        "depth_a0_median": a0_depth_med,
        "depth_a0_err_lo": a0_depth_lo,
        "depth_a0_err_hi": a0_depth_hi,
        "a0_median": a0_raw_med,
        "a0_err_lo": a0_raw_lo,
        "a0_err_hi": a0_raw_hi,
    })
    for name, coeff in coeff_samp.items():
        coeff_med, coeff_lo, coeff_hi = _harmonica_percentile_summary(coeff)
        limb_df[f"{name}_median"] = coeff_med
        limb_df[f"{name}_err_lo"] = coeff_lo
        limb_df[f"{name}_err_hi"] = coeff_hi
    if bandpass_min is not None:
        limb_df["bandpass_min"] = bandpass_min
    if bandpass_max is not None:
        limb_df["bandpass_max"] = bandpass_max
    return limb_df, a0_samp, coeff_samp


def _save_harmonica_limb_posterior_samples(
    path,
    wavelengths,
    wavelength_err,
    a0_samples,
    coefficient_samples,
    planet_index=0,
):
    """Persist the joint limb posterior without marginalizing its covariance."""
    wl_arr, wl_err_arr = _coerce_wavelength_axis(wavelengths, wavelength_err)
    a0_samp = np.asarray(a0_samples, dtype=np.float64)
    coeff_samp = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in coefficient_samples.items()
    }
    if a0_samp.ndim != 2 or a0_samp.shape[1] != wl_arr.size:
        raise ValueError(
            "a0_samples must have shape (draw, wavelength) matching wavelengths."
        )
    for name, values in coeff_samp.items():
        if values.shape != a0_samp.shape:
            raise ValueError(
                f"{name} samples have shape {values.shape}; expected {a0_samp.shape}."
            )

    products = _harmonica_limb_product_samples(a0_samp, coeff_samp)
    payload = {
        "limb_product_schema_version": np.asarray(
            HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION, dtype=np.int64
        ),
        "limb_product_convention": np.asarray(HARMONICA_LIMB_PRODUCT_CONVENTION),
        "sample_axes": np.asarray("draw,wavelength"),
        "depth_units": np.asarray("fractional_stellar_area"),
        "wavelength_units": np.asarray("micron"),
        "planet_index": np.asarray(planet_index, dtype=np.int64),
        "wavelength": wl_arr,
        "wavelength_err": wl_err_arr,
        "a0": a0_samp,
    }
    payload.update(coeff_samp)
    payload.update(products)
    _atomic_savez_compressed(path, **payload)
    print(f"Saved joint limb posterior samples to {path}")


def save_harmonica_limb_products(
    wavelengths,
    wavelength_err,
    rors_samples,
    harmonic_samples,
    csv_path,
    limb_spectrum_path,
    transmission_strings_path,
    title_prefix,
    bandpass_min=None,
    bandpass_max=None,
    posterior_strings_path=None,
    posterior_samples_path=None,
    planet_index=0,
):
    from harmonica import HarmonicaTransit

    limb_df, a0_samp, coeff_samp = build_harmonica_limb_dataframe(
        wavelengths,
        wavelength_err,
        rors_samples,
        harmonic_samples,
        bandpass_min=bandpass_min,
        bandpass_max=bandpass_max,
        planet_index=planet_index,
    )
    _atomic_dataframe_csv(limb_df, csv_path, index=False)
    print(f"Saved limb spectra to {csv_path}")

    if posterior_samples_path is not None:
        _save_harmonica_limb_posterior_samples(
            posterior_samples_path,
            wavelengths,
            wavelength_err,
            a0_samp,
            coeff_samp,
            planet_index=planet_index,
        )

    wl_arr = limb_df["wavelength"].values
    plot_harmonica_limb_spectra(
        limb_df,
        limb_spectrum_path,
        instrument_label=title_prefix,
    )
    print(f"Saved limb spectra plot to {limb_spectrum_path}")

    theta = np.linspace(-np.pi, np.pi, 1000)
    ht = HarmonicaTransit()

    a0_med = limb_df["a0_median"].values
    coeff_med = {
        name: limb_df[f"{name}_median"].values
        for name in HARMONICA_ODD_HARMONICS
        if f"{name}_median" in limb_df
    }

    if wl_arr.size > 1:
        radius_curves = []
        for i in range(len(wl_arr)):
            r_i = _harmonica_r_vector_from_values(
                a0_med[i],
                {name: values[i] for name, values in coeff_med.items()},
            )
            ht.set_planet_transmission_string(r_i.copy())
            radius_curves.append(ht.get_planet_transmission_string(theta))
        plot_harmonica_transmission_strings(
            theta,
            radius_curves,
            a0_med,
            transmission_strings_path,
            instrument_label=title_prefix,
        )
        print(f"Saved transmission string plot to {transmission_strings_path}")
    else:
        posterior_strings_path = transmission_strings_path

    if posterior_strings_path is not None:
        i_show = len(wl_arr) // 2
        n_draw = min(200, a0_samp.shape[0])
        draw_idx = np.random.choice(a0_samp.shape[0], n_draw, replace=False)
        posterior_radius_curves = []
        for j in draw_idx:
            r_sample = _harmonica_r_vector_from_values(
                a0_samp[j, i_show],
                {
                    name: coeff[j, i_show]
                    for name, coeff in coeff_samp.items()
                },
            )
            ht.set_planet_transmission_string(r_sample.copy())
            posterior_radius_curves.append(
                ht.get_planet_transmission_string(theta)
            )
        r_med = _harmonica_r_vector_from_values(
            a0_med[i_show],
            {name: values[i_show] for name, values in coeff_med.items()},
        )
        ht.set_planet_transmission_string(r_med.copy())
        rp_med = ht.get_planet_transmission_string(theta)
        plot_harmonica_transmission_posterior(
            theta,
            posterior_radius_curves,
            rp_med,
            a0_med[i_show],
            posterior_strings_path,
            instrument_label=title_prefix,
        )
        print(f"Saved transmission string posterior plot to {posterior_strings_path}")

    return limb_df

"""Surface helpers."""

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
from .limb_darkening import _posterior_quadratic_ld_median

def _attach_jaxoplanet_eval_metadata(
    params,
    t,
    *,
    jaxoplanet_kernel,
    ld_profile,
    transit_window_optimization="off",
):
    """Attach routing and time-axis-specific acceleration data to diagnostics.

    Phase offsets and window indices are rebuilt for *t*, so a post-fit model
    cannot accidentally reuse indices from a pre-clipping cadence axis.
    """
    from models.jaxoplanet.core import (
        build_transit_phase_offsets,
        build_transit_window_indices,
    )

    out = dict(params)
    for name in (*_JAXOPLANET_STATIC_EVAL_KEYS, *_JAXOPLANET_DYNAMIC_EVAL_KEYS):
        out.pop(name, None)
    out["_jaxoplanet_kernel"] = str(jaxoplanet_kernel)
    out["_ld_profile"] = str(ld_profile)

    duration_geometry = "duration" in out and "a_rs" not in out
    if not duration_geometry:
        return out

    phase_offsets, phase_mask = build_transit_phase_offsets(
        t, out["period"], out["t0"], out["duration"]
    )
    out["_transit_phase_offsets"] = phase_offsets
    out["_transit_phase_mask"] = phase_mask
    if transit_window_optimization == "auto":
        indices = build_transit_window_indices(
            np.asarray(t),
            np.asarray(out["period"]),
            np.asarray(out["t0"]),
            np.asarray(out["duration"]),
        )
        out["_transit_window_indices"] = jnp.asarray(indices, dtype=jnp.int32)
    elif transit_window_optimization != "off":
        raise ValueError(
            "transit_window_optimization must be one of {'auto', 'off'}."
        )
    return out


def _attach_surface_eval_metadata(params, surface_config):
    """Attach static map geometry and fixed/default surface values for plotting."""
    out = dict(params)
    if not surface_config or (
        surface_config.get("model", "transit") == "transit"
        and not surface_config.get("spots")
    ):
        return out
    out["_surface_model"] = surface_config["model"]
    out["_stellar_spots"] = surface_config.get("spots", ())
    for name in (
        "eclipse_depth", "dayside_flux", "nightside_flux", "hotspot_offset"
    ):
        indexed = sorted(
            (
                (int(key.rsplit("_", 1)[1]), value)
                for key, value in out.items()
                if re.fullmatch(rf"_{name}_\d+", key)
            ),
            key=lambda item: item[0],
        )
        if name not in out and indexed:
            out[name] = jnp.stack([value for _, value in indexed])
        if name in surface_config and name not in out:
            out[name] = jnp.asarray(surface_config[name], dtype=jnp.float64)
    if surface_config.get("spots"):
        out["stellar_rotation_period"] = jnp.asarray(
            surface_config["stellar_rotation_period"], dtype=jnp.float64
        )
        if "stellar_spot_contrast" not in out:
            indexed = sorted(
                (
                    (int(key.rsplit("_", 1)[1]), value)
                    for key, value in out.items()
                    if re.fullmatch(r"_stellar_spot_contrast_\d+", key)
                ),
                key=lambda item: item[0],
            )
            out["stellar_spot_contrast"] = (
                jnp.stack([value for _, value in indexed])
                if indexed else jnp.asarray(
                    [spot["contrast"] for spot in surface_config["spots"]],
                    dtype=jnp.float64,
                )
            )
    return out


def _seed_surface_spectroscopic_init(init_params, surface_config, num_lcs):
    """Seed every free surface latent at its physical user prior center."""
    if not surface_config:
        return init_params
    phase_flux_names = (
        {"dayside_flux", "nightside_flux"}
        if surface_config.get("model") == "phase_curve" else set()
    )
    for name in (
        "eclipse_depth", "dayside_flux", "nightside_flux", "hotspot_offset"
    ):
        if name in phase_flux_names:
            continue
        width_name = f"{name}_prior_width"
        if name not in surface_config or width_name not in surface_config:
            continue
        for planet, (center, width) in enumerate(zip(
            np.atleast_1d(surface_config[name]),
            np.atleast_1d(surface_config[width_name]),
        )):
            if float(width) > 0.0:
                init_params[f"_{name}_{planet}"] = jnp.full(
                    int(num_lcs), float(center), dtype=jnp.float64
                )
    if phase_flux_names:
        init_params.update(
            _phase_curve_flux_init_sites(
                surface_config, shape=(int(num_lcs),)
            )
        )
    for spot, spec in enumerate(surface_config.get("spots", ())):
        if float(spec.get("contrast_prior_width", 0.0)) > 0.0:
            init_params[f"_stellar_spot_contrast_{spot}"] = jnp.full(
                int(num_lcs), float(spec["contrast"]), dtype=jnp.float64
            )
    return init_params


def _phase_curve_flux_init_sites(surface_config, shape=()):
    """Return prior-center values for the smooth physical phase-flux sites."""
    if not surface_config or surface_config.get("model") != "phase_curve":
        return {}
    from models.jaxoplanet.builder import phase_flux_to_conditional_quantile

    day_centers = np.atleast_1d(surface_config["dayside_flux"])
    day_widths = np.atleast_1d(surface_config["dayside_flux_prior_width"])
    night_centers = np.atleast_1d(surface_config["nightside_flux"])
    night_widths = np.atleast_1d(surface_config["nightside_flux_prior_width"])
    result = {}

    def store(name, planet, value):
        value = float(np.asarray(jax.device_get(value)))
        result[f"_{name}_{planet}"] = (
            jnp.full(shape, value, dtype=jnp.float64)
            if shape else jnp.asarray(value, dtype=jnp.float64)
        )

    for planet, (day, day_width, night, night_width) in enumerate(zip(
        day_centers, day_widths, night_centers, night_widths
    )):
        day_free = float(day_width) > 0.0
        night_free = float(night_width) > 0.0
        if day_free and night_free:
            store("dayside_flux", planet, day)
            quantile = phase_flux_to_conditional_quantile(
                night, day, night, night_width
            )
            store("nightside_flux_quantile", planet, quantile)
        elif day_free:
            quantile = phase_flux_to_conditional_quantile(
                day, night, day, day_width
            )
            store("dayside_flux_quantile", planet, quantile)
        elif night_free:
            quantile = phase_flux_to_conditional_quantile(
                night, day, night, night_width
            )
            store("nightside_flux_quantile", planet, quantile)
    return result


def _prepare_fixed_surface_basis(
    surface_config, t, u, *, period, t0, a_rs, b, rors, ecc, omega
):
    """Precompute an exact native linear basis for a fixed surface model."""
    if surface_config.get("fit_geometry", True):
        return None
    from models.jaxoplanet.surface_basis import (
        EmissionLightCurveBasis,
        SpotLightCurveBasis,
        prepare_emission_light_curve_basis,
        prepare_spot_light_curve_basis,
    )

    model = surface_config.get("model", "transit")
    spots = surface_config.get("spots", ())
    if not ((model == "transit" and spots) or (model in {"eclipse", "phase_curve"} and not spots)):
        return None
    base = {
        "_surface_model": model,
        "period": period,
        "t0": t0,
        "a_rs": a_rs,
        "b": b,
        "rors": rors,
        "ecc": ecc,
        "omega": omega,
    }
    if spots:
        base["stellar_rotation_period"] = surface_config["stellar_rotation_period"]

    def prepare_one(coefficients):
        fixed = {**base, "u": coefficients}
        if spots:
            return prepare_spot_light_curve_basis(
                fixed, t, spots=spots
            )
        return prepare_emission_light_curve_basis(
            fixed, t, model=model
        )

    def stack_bases(bases):
        return type(bases[0])(*(
            None if field is None else jnp.stack([
                basis[index] for basis in bases
            ])
            for index, field in enumerate(bases[0])
        ))

    u_grid = np.asarray(u, dtype=float)
    if u_grid.ndim == 1:
        return prepare_one(u_grid)
    if u_grid.ndim != 2:
        raise ValueError("Fixed surface-basis limb darkening must have one row per channel.")
    # A shared explicit LD law is the common fast path.  For a wavelength
    # dependent fixed grid, prepare each exact native basis once and stack it;
    # the sampler still sees only the small linear contrast model.
    if np.allclose(u_grid, u_grid[0], rtol=0.0, atol=0.0):
        shared = prepare_one(u_grid[0])
        return type(shared)(*(
            None if field is None else jnp.broadcast_to(
                field, (u_grid.shape[0],) + field.shape
            )
            for field in shared
        ))
    return stack_bases([prepare_one(row) for row in u_grid])


def _surface_basis_on_time_mask(basis, mask):
    """Select cadence leaves while preserving the exact surface-basis type."""
    if basis is None:
        return None
    mask = jnp.asarray(mask, dtype=bool)
    return type(basis)(*(
        None if field is None else field[..., mask]
        for field in basis
    ))


def _select_transit_eval_params(
    params,
    transit_engine,
    param_method,
    *,
    t=None,
    jaxoplanet_kernel=None,
    ld_profile=None,
    transit_window_optimization="off",
    surface_config=None,
):
    out = dict(params)
    if transit_engine != 'jaxoplanet':
        return out
    if param_method == 'a_rs':
        out.pop('duration', None)
    else:
        out.pop('a_rs', None)
        out.pop('ecc', None)
        out.pop('omega', None)
        out.pop('cos_i', None)
        out.pop('inc', None)
    if t is not None and jaxoplanet_kernel is not None and ld_profile is not None:
        out = _attach_jaxoplanet_eval_metadata(
            out,
            t,
            jaxoplanet_kernel=jaxoplanet_kernel,
            ld_profile=ld_profile,
            transit_window_optimization=transit_window_optimization,
        )
    out = _attach_surface_eval_metadata(out, surface_config)
    return out

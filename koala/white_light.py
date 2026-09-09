"""Literal white-light target, cache, inference, and output stage."""

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
    plot_whitelight_corner,
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
    GP_HYPERPARAMETER_BOUNDS,
    GP_MODEL_REVISION,
    compute_gp_training_prediction,
    resolve_gp_mean_function,
    resolve_gp_solver,
    validate_gp_times,
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
    ArtifactSet,
)
from .outputs import (
    _param_at, _poly_trend_np, _soft_step_np, _trend_from_params_np,
    plot_noise_binning_robust, _noise_binning_stats, save_noise_binning_data,
    save_whitelight_timeseries, _save_mcmc_diagnostics, compute_aic,
    get_asym_errors, compute_u_from_c, fit_polynomial, plot_poly_fit,
    save_results, save_detailed_fit_results, get_robust_sigma,
    calculate_beta_metrics, run_beta_monte_carlo,
)
from .geometry import (
    _geometry_chain_quality, _continue_mcmc_until_geometry_gate,
    _posterior_num_draws, _geometry_from_white_light_medians,
    _selected_geometry_primitives, _whitelight_geometry_handoff_fingerprint,
    _write_whitelight_geometry_handoff, _load_whitelight_geometry_handoff,
)
from .sampling import (
    _DUMPED_FIRST_HIGHRES_STAGE, SamplerInputsDumpExit, _build_numpyro_mcmc,
    get_samples, _slice_by_channel, _take_by_channel, _callable_identity,
    _tree_to_numpy_for_pickle, _build_spectroscopic_model,
    _sampler_model_builder_spec, _sampler_initial_potential, _safe_stage_label,
    _maybe_dump_sampler_inputs, _chunk_checkpoint_fingerprint,
    _sampling_workload_fingerprint, _annotate_sampler_diagnostics,
    _write_or_validate_checkpoint_manifest, _chunk_ranges_for,
    _chunk_checkpoint_path, _SamplerSamples, _load_chunk_samples,
    _concatenate_chunk_samples, _spectro_failed_lanes,
    _spectro_sampler_swap_order, _resolve_parallel_chunk_job,
    get_samples_chunked, _run_sampling_stage,
)
from .limb_darkening import (
    _power2_ld_initial_sites, _power2_ld_optimization_sites,
    _clip_ld_initial_values, _quadratic_uniform_initial_sites,
    _validate_whitelight_optimized_start, _get_ld_mode_bounds, _clip_ld_range,
    _gaussian_pdf, _build_axis, _combine_weighted_means_and_sigmas,
    _compute_ld_coeffs_with_retry, get_limb_darkening, build_sing_ld_prior,
    _assess_sing_gray_fit, get_or_build_power2_ld_prior,
    _posterior_quadratic_ld_median,
)
from .harmonica_products import (
    _HARMONICA_LIMB_LOGGER, _LEGACY_LIMB_SCHEMA_WARNING_EMITTED,
    _coerce_wavelength_axis, _harmonica_frac_site, _harmonica_coeff_to_frac,
    _harmonica_coeff_array_to_frac, _harmonica_coeff_array_to_delta_r,
    _harmonica_coefficients_to_half_area_init,
    _validate_harmonica_spectro_parameterization,
    _harmonica_model_radius_samples, _harmonica_checkpoint_prefix,
    _harmonica_spectro_artifact_stem, _coerce_harmonica_sample_matrix,
    _extract_harmonica_limb_samples, _sum_harmonica_odd_samples,
    _harmonica_limb_product_samples, _harmonica_percentile_summary,
    _harmonica_r_vector_from_values, _harmonica_sample_payload,
    _has_harmonica_odd_samples, _augment_harmonica_params,
    _harmonica_cosi_from_b, _add_rp_summary_from_depth,
    load_harmonica_limb_dataframe, build_harmonica_limb_dataframe,
    _save_harmonica_limb_posterior_samples, save_harmonica_limb_products,
)
from .surface import (
    _attach_jaxoplanet_eval_metadata, _attach_surface_eval_metadata,
    _seed_surface_spectroscopic_init, _phase_curve_flux_init_sites,
    _prepare_fixed_surface_basis, _surface_basis_on_time_mask,
    _select_transit_eval_params,
)
from .spectroscopy import run_low_resolution_stage, run_high_resolution_stage
from models.harmonica.core import (
    _ALL_ODD_COEFF_SPECS,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_area_radius_and_q,
)


def _compute_whitelight_gp_products(
    params, t, error, y, *, detrend_type, gp_solver
):
    """Return aligned training-point GP products for white-light output."""
    mu, var = compute_gp_training_prediction(
        params,
        t,
        error,
        y,
        detrend_type=detrend_type,
        gp_solver=gp_solver,
        assume_sorted=True,
        include_mean=True,
    )
    t_ref = jnp.min(t)
    parametric_mean = resolve_gp_mean_function(detrend_type)(
        params, t, t_ref=t_ref
    )
    planet_model = compute_transit_model_auto(params, t)
    return {
        "mu": mu,
        "var": var,
        "planet_model_only": planet_model,
        "trend_flux_total": mu - planet_model - 1.0,
        "parametric_mean": parametric_mean,
        "gp_stochastic_component": mu - parametric_mean,
    }



def _repair_gp_support_edges(soln, *, margin_fraction=0.01, edge_tolerance=1.0e-8):
    """Move GP hyperparameter starts strictly inside their Uniform supports.

    A constrained optimizer can return ``GP_log_sigma`` or ``GP_log_rho`` on
    (or, through roundoff, marginally outside) a prior edge; there the
    unconstrained coordinate is infinite and NumPyro cannot initialize.  The
    repaired value sits ``margin_fraction`` of the support width inside the
    offending edge so the Laplace MAP refinement can still move it.
    """
    repaired = dict(soln)
    for site, (lower, upper) in GP_HYPERPARAMETER_BOUNDS.items():
        if site not in repaired:
            continue
        width = upper - lower
        value = float(np.asarray(repaired[site]))
        tolerance = edge_tolerance * max(1.0, width)
        if not np.isfinite(value):
            reset = 0.5 * (lower + upper)
        elif value <= lower + tolerance:
            reset = lower + margin_fraction * width
        elif value >= upper - tolerance:
            reset = upper - margin_fraction * width
        else:
            continue
        print(
            f"WARNING: white-light optimizer placed {site} on or outside its "
            f"support edge ({value:.6g}); resetting it to {reset:.6g} before "
            "Laplace preparation."
        )
        repaired[site] = jnp.asarray(reset, dtype=jnp.asarray(repaired[site]).dtype)
    return repaired

def _white_light_fingerprint_config(cfg):
    """Exclude execution-only GP solver selection from scientific identity."""
    return {key: value for key, value in cfg.items() if key != "gp_solver"}


def _save_whitelight_gp_database(path, wl_flux, products):
    """Write the legacy GP handoff schema consumed by spectroscopy."""
    frame = pd.DataFrame({
        'wl_flux': wl_flux,
        'gp_flux': products["mu"],
        'gp_err': jnp.sqrt(products["var"]),
        'gp_trend': products["gp_stochastic_component"],
    })
    _atomic_dataframe_csv(frame, path, index=True)

from . import artifacts, config, constants, data, geometry, harmonica_products
from . import limb_darkening, outputs, sampling, surface
for _module in (artifacts, config, constants, data, geometry, harmonica_products,
                limb_darkening, outputs, sampling, surface):
    globals().update({name: value for name, value in vars(_module).items()
                      if not name.startswith("__")})

def run_white_light_stage(
    HARMONICA_A_RS,
    HARMONICA_A_RS_PRIOR_MAX,
    HARMONICA_A_RS_PRIOR_MIN,
    HARMONICA_ECC,
    HARMONICA_INC_PRIOR_MAX,
    HARMONICA_INC_PRIOR_MIN,
    HARMONICA_ODD_HARMONICS,
    HARMONICA_OMEGA,
    NUTS_KWARGS,
    PERIOD_FIXED,
    PRIOR_B,
    PRIOR_DUR,
    PRIOR_RPRS,
    PRIOR_T0,
    T0_PRIOR_WIDTH,
    _engine_wl_kw,
    _explicit_ld_grid,
    cfg,
    create_whitelight_model,
    data,
    derive_geometry,
    detrending_type,
    explicit_ld,
    harmonica_wl_nuts_kwargs,
    instrument,
    instrument_full_str,
    jaxoplanet_kernel,
    jump_guess,
    key_master,
    ld_prior_mode,
    ld_profile,
    ld_uniform_basis,
    n_planets,
    order,
    output_dir,
    param_method,
    planet_str,
    prepare_laplace_metric,
    quadratic_uniform_physical_bounds,
    sld,
    spot_amp,
    spot_amp2,
    spot_mu,
    spot_mu2,
    spot_sigma,
    spot_sigma2,
    stellar_cfg,
    step_width_days,
    step_width_mode,
    surface_config,
    t_jump_guess,
    transit_engine,
    uses_surface_model,
    whitelight_geometry_estimator,
    whitelight_laplace_options,
    whitelight_ld_parameterization,
    whitelight_mass_matrix,
    whitelight_mcmc_kwargs,
    whitelight_sigma,
    whitelight_trend_parameterization,
    whitelight_two_spot_ordering,
):
    is_gp_detrending = 'gp' in detrending_type
    gp_solver = None
    if is_gp_detrending:
        validate_gp_times(data.wl_time)
        gp_solver = resolve_gp_solver(cfg.get('gp_solver'))
        print(f"White-light GP solver: {gp_solver}")
    _engine_wl_kw = dict(_engine_wl_kw)
    _engine_wl_kw.update(
        gp_solver=gp_solver,
        gp_assume_sorted=True,
    )

    if explicit_ld is not None:
        U_mu_wl, U_sigma_wl = _explicit_ld_grid(), None
    elif ld_prior_mode == 'stellarprior':
        U_mu_wl, U_sigma_wl = get_or_build_power2_ld_prior(
            stellar_cfg, data.wavelengths_unbinned, 0.0, instrument,
            order=order if instrument == 'NIRISS/SOSS' else None,
            output_dir=output_dir,
            cache_label='whitelight'
        )
    else:
        if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
            U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
        elif instrument == 'NIRISS/SOSS':
            U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
        U_sigma_wl = None

    wl_mask_path = f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy'
    wl_params_path = f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv'
    wl_gp_path = f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv'
    wl_limb_csv_path = f"{output_dir}/{instrument_full_str}_whitelight_limb_spectra.csv"
    wl_limb_samples_path = f"{output_dir}/{instrument_full_str}_whitelight_limb_posterior_samples.npz"
    wl_manifest_path = f'{output_dir}/{instrument_full_str}_whitelight_artifacts.manifest.json'
    wl_geometry_handoff_path = (
        f'{output_dir}/{instrument_full_str}_whitelight_geometry_handoff.json'
    )
    wl_artifact_fingerprint = _science_artifact_fingerprint(
        "whitelight",
        {
            "config": _white_light_fingerprint_config(cfg),
            "time": data.wl_time,
            "flux": data.wl_flux,
            "flux_err": data.wl_flux_err,
            "wavelengths_unbinned": data.wavelengths_unbinned,
            "transit_engine": transit_engine,
            "ld_profile": ld_profile,
            "ld_prior_mode": ld_prior_mode,
            "detrending_type": detrending_type,
            "trend_parameterization": whitelight_trend_parameterization,
            "two_spot_ordering": whitelight_two_spot_ordering,
            "whitelight_mass_matrix": whitelight_mass_matrix,
            "param_method": param_method,
            "geometry_estimator": whitelight_geometry_estimator,
            "whitelight_sigma": whitelight_sigma,
            "ld_coefficients": U_mu_wl,
            "ld_uncertainties": U_sigma_wl,
            "step_width_mode": step_width_mode,
            "step_width_days": step_width_days,
            **(
                {"gp_model_revision": GP_MODEL_REVISION}
                if is_gp_detrending else {}
            ),
        },
    )
    wl_geometry_handoff = _load_whitelight_geometry_handoff(
        wl_geometry_handoff_path,
        expected_posterior_fingerprint=wl_artifact_fingerprint,
        expected_estimator=whitelight_geometry_estimator,
    )
    required_wl_limb_products_exist = bool(
        transit_engine != 'harmonica'
        or not HARMONICA_ODD_HARMONICS
        or (
            os.path.exists(wl_limb_csv_path)
            and os.path.exists(wl_limb_samples_path)
        )
    )
    wl_artifact_set = ArtifactSet(
        stage="whitelight",
        manifest_path=wl_manifest_path,
        fingerprint=wl_artifact_fingerprint,
        required_paths=(
            wl_mask_path,
            wl_params_path,
            *((wl_gp_path,) if 'gp' in detrending_type else ()),
        ),
    )
    stringcheck = wl_artifact_set.is_reusable(
        extra_condition=(
            required_wl_limb_products_exist
            and wl_geometry_handoff is not None
        )
    )
    if not stringcheck and (
        os.path.exists(wl_mask_path) or os.path.exists(wl_params_path)
    ):
        print(
            "Existing white-light products or geometry handoff are unverified "
            "or do not match the current data/configuration; recomputing them."
        )

    if not stringcheck or ('gp' in detrending_type):
        if not stringcheck or not os.path.exists(wl_gp_path):
            plt.scatter(data.wl_time, data.wl_flux, c='k', s=6, alpha=0.5)
            plt.savefig(f'{output_dir}/stuff.png')
            plt.close()
            print('Fitting whitelight for outliers and bestfit parameters')
            hyper_params_wl = {
                "duration": PRIOR_DUR,
                "t0": PRIOR_T0,
                'period': PERIOD_FIXED,
                'b': PRIOR_B,
                'rprs': PRIOR_RPRS,
                'u': U_mu_wl,
                'a_rs': HARMONICA_A_RS,
                'a_rs_prior_min': HARMONICA_A_RS_PRIOR_MIN,
                'a_rs_prior_max': HARMONICA_A_RS_PRIOR_MAX,
                'inc_prior_min': HARMONICA_INC_PRIOR_MIN,
                'inc_prior_max': HARMONICA_INC_PRIOR_MAX,
                'ecc': HARMONICA_ECC,
                'omega': HARMONICA_OMEGA,
                't0_prior_width': T0_PRIOR_WIDTH,
            }
            if '2spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu
                hyper_params_wl['spot_guess2'] = spot_mu2
            elif 'spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu
            if 'linear_discontinuity' in detrending_type:
                hyper_params_wl['t_jump_guess'] = t_jump_guess if t_jump_guess is not None else 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
                hyper_params_wl['jump_guess'] = jump_guess
                hyper_params_wl['step_width_days'] = step_width_days

            hyper_params_wl['u'] = U_mu_wl
            if ld_profile == 'power2' and ld_prior_mode == 'stellarprior' and U_sigma_wl is not None:
                hyper_params_wl['u_sigma'] = U_sigma_wl

            init_params_wl = {
                'c': 1.0,
                'v': 0.0,
                'log_jitter': jnp.log(1e-4),
                'b': PRIOR_B,
                'rors': PRIOR_RPRS
            }
            init_params_wl['u'] = U_mu_wl
            if ld_profile == 'quadratic' and ld_prior_mode == 'uniform':
                init_params_wl.pop('u', None)
                init_params_wl.update(_quadratic_uniform_initial_sites(
                    U_mu_wl, ld_uniform_basis
                ))
            # Power-2 models sample the native (c, alpha) coefficients at the
            # latent sites ``c1`` and ``c2``.  Supplying only ``u`` silently
            # falls back to NumPyro's default initialization and defeats the
            # stellar-informed starting point for both transit engines.
            if ld_profile == 'power2' and ld_prior_mode != 'fixed':
                init_params_wl.update(_power2_ld_initial_sites(
                    U_mu_wl, whitelight_ld_parameterization
                ))
            if transit_engine == 'harmonica':
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    init_params_wl[_harmonica_frac_site(harmonic_name)] = _harmonica_coeff_to_frac(
                        HARMONICA_INIT_ODD_COEFF,
                        PRIOR_RPRS,
                    )

            for i in range(n_planets):
                if param_method == 'a_rs':
                    init_params_wl[f'log_a_rs_{i}'] = jnp.log(HARMONICA_A_RS[i])
                    init_params_wl[f'_b_{i}'] = PRIOR_B[i]
                else:
                    init_params_wl[f'logD_{i}'] = jnp.log(PRIOR_DUR[i])
                    init_params_wl[f'_b_{i}'] = PRIOR_B[i]
                init_params_wl[f't0_{i}'] = PRIOR_T0[i]
                init_params_wl[f'rors_{i}'] = PRIOR_RPRS[i]

            for surface_name in (
                'eclipse_depth', 'dayside_flux', 'nightside_flux',
                'hotspot_offset',
            ):
                if (
                    surface_config.get('model') == 'phase_curve'
                    and surface_name in {'dayside_flux', 'nightside_flux'}
                ):
                    continue
                width_name = f'{surface_name}_prior_width'
                if width_name in surface_config:
                    for i, (center, width) in enumerate(zip(
                        np.atleast_1d(surface_config[surface_name]),
                        np.atleast_1d(surface_config[width_name]),
                    )):
                        if width > 0.0:
                            init_params_wl[f'_{surface_name}_{i}'] = center
            init_params_wl.update(
                _phase_curve_flux_init_sites(surface_config)
            )
            for i, spot in enumerate(surface_config.get('spots', ())):
                if spot.get('contrast_prior_width', 0.0) > 0.0:
                    init_params_wl[f'_stellar_spot_contrast_{i}'] = spot['contrast']
            if uses_surface_model and not surface_config.get('fit_geometry', True):
                for i in range(n_planets):
                    init_params_wl.pop(f'log_a_rs_{i}', None)
                    init_params_wl.pop(f'_b_{i}', None)
                    init_params_wl.pop(f't0_{i}', None)
                    init_params_wl.pop(f'rors_{i}', None)
                    init_params_wl[f't0_{i}'] = PRIOR_T0[i]
                    init_params_wl[f'b_{i}'] = PRIOR_B[i]
                    init_params_wl[f'rors_{i}'] = PRIOR_RPRS[i]
                    init_params_wl[f'a_rs_{i}'] = HARMONICA_A_RS[i]
                    init_params_wl[f'duration_{i}'] = PRIOR_DUR[i]

            if 'quadratic' in detrending_type:
                init_params_wl['v2'] = 0.0
            if 'cubic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0
            if 'quartic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0; init_params_wl['v4'] = 0.0
            if 'explinear' in detrending_type:
                init_params_wl['A'] = 0.001
                # ``tau`` is deterministic; ``log_tau`` is the actual latent
                # site and its prior spans 1e-3--1e-1 days.
                init_params_wl['log_tau'] = jnp.log(1e-2)
            if uses_surface_model:
                soln = dict(init_params_wl)
                print("Using the physical surface prior center for initialization.")
            elif 'gp' in detrending_type:
                init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(0.1)
            if 'linear_discontinuity' in detrending_type:
                if t_jump_guess is not None:
                    init_params_wl['t_jump'] = t_jump_guess
                else:
                    init_params_wl['t_jump'] = 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
                init_params_wl['jump'] = jump_guess
                if step_width_mode == 'free':
                    cadence = float(np.median(np.diff(np.sort(np.asarray(data.wl_time)))))
                    init_params_wl['log_width'] = np.log(
                        np.sqrt(0.5 * cadence * (30.0 / (24.0 * 60.0)))
                    )
            if 'spot' in detrending_type:
                init_params_wl['spot_amp'] = spot_amp
                init_params_wl['spot_mu'] = spot_mu
                init_params_wl['spot_sigma'] = spot_sigma
            if '2spot' in detrending_type:
                init_params_wl['spot_amp2'] = spot_amp2
                init_params_wl['spot_mu2'] = spot_mu2
                init_params_wl['spot_sigma2'] = spot_sigma2
            # Step 1 of the Sing recipe measures LD freely.  The resulting
            # artifact is deliberately consumed only by the spectroscopic fit.
            wl_ld_mode = 'uniform' if ld_prior_mode == 'sing' else ld_prior_mode
            if wl_ld_mode == 'fixed':
                spot_basis_wl = _prepare_fixed_surface_basis(
                    surface_config,
                    data.wl_time,
                    U_mu_wl,
                    period=PERIOD_FIXED,
                    t0=PRIOR_T0,
                    a_rs=HARMONICA_A_RS,
                    b=PRIOR_B,
                    rors=PRIOR_RPRS,
                    ecc=HARMONICA_ECC,
                    omega=HARMONICA_OMEGA,
                )
                if spot_basis_wl is not None:
                    _engine_wl_kw['surface_basis'] = spot_basis_wl
                    print(
                        "Prepared exact fixed-geometry surface basis "
                        "for white-light inference."
                    )
            whitelight_model_for_run = create_whitelight_model(
                detrend_type=detrending_type,
                n_planets=n_planets,
                ld_mode=wl_ld_mode,
                step_width_mode=step_width_mode,
                **_engine_wl_kw,
            )

            def _get_preopt_sanity_model(params, t_vals):
                if 'gp' in detrending_type:
                    return compute_lc_linear(params, t_vals)
                try:
                    return resolve_detrend_kernel(detrending_type)(params, t_vals)
                except KeyError:
                    return compute_lc_linear(params, t_vals)

            def _build_preopt_physical_params(init_params, n_planets_eval):
                params_eval = hyper_params_wl.copy()
                params_eval.update(init_params)

                def _ensure_len(value, default):
                    arr = jnp.atleast_1d(jnp.asarray(params_eval.get(value, default)))
                    if arr.size == 1 and n_planets_eval > 1:
                        return jnp.repeat(arr, n_planets_eval)
                    return arr

                params_eval["period"] = _ensure_len("period", PERIOD_FIXED)
                params_eval["t0"] = _ensure_len("t0", PRIOR_T0)
                params_eval["b"] = _ensure_len("b", PRIOR_B)
                params_eval["rors"] = _ensure_len("rors", PRIOR_RPRS)

                if all(f"t0_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["t0"] = jnp.array([init_params[f"t0_{i}"] for i in range(n_planets_eval)])
                if all(f"_b_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["b"] = jnp.array([jnp.abs(init_params[f"_b_{i}"]) for i in range(n_planets_eval)])
                if all(f"rors_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["rors"] = jnp.array([init_params[f"rors_{i}"] for i in range(n_planets_eval)])
                elif all(f"depths_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["rors"] = jnp.array([jnp.sqrt(init_params[f"depths_{i}"]) for i in range(n_planets_eval)])

                if transit_engine == 'harmonica':
                    params_eval["a_rs"] = _ensure_len("a_rs", HARMONICA_A_RS)
                    if all(f"log_a_rs_{i}" in init_params for i in range(n_planets_eval)):
                        params_eval["a_rs"] = jnp.array([jnp.exp(init_params[f"log_a_rs_{i}"]) for i in range(n_planets_eval)])
                    elif all(f"logD_{i}" in init_params for i in range(n_planets_eval)):
                        params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets_eval)])
                        params_eval["a_rs"] = harmonica_a_rs_from_duration(
                            params_eval["period"],
                            params_eval["duration"],
                            params_eval["b"],
                            params_eval["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    # Compatibility fallback for older direct-inc initializations.
                    if all(f"inc_{i}" in init_params for i in range(n_planets_eval)):
                        incs = jnp.array([init_params[f"inc_{i}"] for i in range(n_planets_eval)])
                        params_eval["inc"] = incs
                        params_eval["cos_i"] = jnp.cos(incs)
                        params_eval["b"] = harmonica_impact_param_from_cos_i(
                            params_eval["cos_i"], params_eval["a_rs"],
                            ecc=hyper_params_wl['ecc'], omega=hyper_params_wl['omega'],
                        )
                        params_eval["duration"] = harmonica_duration_from_cos_i(
                            params_eval["period"],
                            params_eval["a_rs"],
                            params_eval["cos_i"],
                            params_eval["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    else:
                        params_eval["duration"] = harmonica_duration_from_geometry(
                            params_eval["period"],
                            params_eval["a_rs"],
                            params_eval["b"],
                            params_eval["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                        params_eval["cos_i"] = _harmonica_cosi_from_b(
                            params_eval["b"], params_eval["a_rs"],
                            hyper_params_wl['ecc'], hyper_params_wl['omega'],
                        )
                    if ld_profile == "power2":
                        if "c1" in init_params:
                            params_eval["c_ld"] = init_params["c1"]
                        if "c2" in init_params:
                            params_eval["alpha_ld"] = init_params["c2"]
                    else:
                        params_eval["u1_ld"] = U_mu_wl[0]
                        params_eval["u2_ld"] = U_mu_wl[1]
                else:
                    params_eval["duration"] = _ensure_len("duration", PRIOR_DUR)
                    if all(f"logD_{i}" in init_params for i in range(n_planets_eval)):
                        params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets_eval)])

                return _attach_surface_eval_metadata(params_eval, surface_config)

            try:
                params_preopt = _build_preopt_physical_params(init_params_wl, n_planets)
                flux_preopt = _get_preopt_sanity_model(params_preopt, data.wl_time)
                finite_preopt = bool(jnp.all(jnp.isfinite(flux_preopt)))
                print(
                    "Pre-opt init model check:",
                    {
                        "finite": finite_preopt,
                        "flux_min": float(jnp.nanmin(flux_preopt)),
                        "flux_max": float(jnp.nanmax(flux_preopt)),
                    },
                )

                fig, axes = plt.subplots(
                    2, 1, figsize=(10, 7), sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                axes[0].scatter(data.wl_time, data.wl_flux, c='k', s=10, alpha=0.3, label='Data')
                axes[0].plot(data.wl_time, flux_preopt, color='tab:red', lw=2.0, label='Init Model')
                axes[0].set_ylabel('Flux')
                axes[0].legend()
                axes[0].set_title(f'Pre-Optimization Check: {detrending_type}')

                preopt_resid = data.wl_flux - flux_preopt
                axes[1].axhline(0.0, color='0.5', lw=1.0, ls='--')
                axes[1].scatter(data.wl_time, preopt_resid, c='tab:blue', s=8, alpha=0.4)
                axes[1].set_xlabel('Time (BJD)')
                axes[1].set_ylabel('Resid.')

                plt.tight_layout()
                preopt_plot_path = f'{output_dir}/00_{instrument_full_str}_preopt_init_check.png'
                plt.savefig(preopt_plot_path)
                plt.close(fig)
                print(f"Saved pre-optimization sanity check plot to: {preopt_plot_path}")
            except Exception as e:
                print(f"Could not create pre-optimization sanity plot: {e}")
            
            if 'gp' in detrending_type:
                print("--- Running Pre-Fit with Linear Detrending to stabilize GP ---")
                whitelight_model_prefit = create_whitelight_model(
                    detrend_type='linear',
                    n_planets=n_planets,
                    ld_mode=wl_ld_mode,
                    step_width_mode=step_width_mode,
                    **_engine_wl_kw,
                )
                init_params_prefit = init_params_wl.copy()
                init_params_prefit.pop('GP_log_sigma', None)
                init_params_prefit.pop('GP_log_rho', None)
                soln = optimx.optimize(whitelight_model_prefit, start=init_params_prefit)(
                    key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
                )
            else:
                soln = optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            
            
            print("Plotting Initial vs. Optimized Model sanity check...")
            
            def _get_sanity_model(params, t_vals):
                if 'gp' in detrending_type:
                    return compute_lc_linear(params, t_vals)
                try:
                    return resolve_detrend_kernel(detrending_type)(params, t_vals)
                except KeyError:
                    return compute_lc_linear(params, t_vals)
            def _soln_to_physical_params(soln, base_params, n_planets=1):
                p = dict(base_params)
                for key, value in soln.items():
                    if key != "obs":
                        p[key] = value

                def have_all(prefix):
                    return all(f"{prefix}_{i}" in p for i in range(n_planets))

                have_log_a_rs = have_all("log_a_rs")
                have_logD = have_all("logD")

                if have_all("t0"):
                    p["t0"] = jnp.array([p[f"t0_{i}"] for i in range(n_planets)])
                if have_all("b"):
                    p["b"] = jnp.array([p[f"b_{i}"] for i in range(n_planets)])
                elif have_all("_b"):
                    p["b"] = jnp.array([jnp.abs(p[f"_b_{i}"]) for i in range(n_planets)])
                if have_all("rors"):
                    p["rors"] = jnp.array([p[f"rors_{i}"] for i in range(n_planets)])
                elif have_all("depths"):
                    p["rors"] = jnp.array([jnp.sqrt(p[f"depths_{i}"]) for i in range(n_planets)])

                if transit_engine == 'harmonica':
                    if have_all("duration"):
                        p["duration"] = jnp.array([p[f"duration_{i}"] for i in range(n_planets)])
                    elif have_logD:
                        p["duration"] = jnp.array([jnp.exp(p[f"logD_{i}"]) for i in range(n_planets)])
                    if have_all("a_rs"):
                        p["a_rs"] = jnp.array([p[f"a_rs_{i}"] for i in range(n_planets)])
                    elif have_log_a_rs:
                        p["a_rs"] = jnp.array([jnp.exp(p[f"log_a_rs_{i}"]) for i in range(n_planets)])
                else:
                    if have_all("duration"):
                        p["duration"] = jnp.array([p[f"duration_{i}"] for i in range(n_planets)])
                    elif have_logD:
                        p["duration"] = jnp.array([jnp.exp(p[f"logD_{i}"]) for i in range(n_planets)])
                    if have_all("a_rs"):
                        p["a_rs"] = jnp.array([p[f"a_rs_{i}"] for i in range(n_planets)])
                    elif have_log_a_rs:
                        p["a_rs"] = jnp.array([jnp.exp(p[f"log_a_rs_{i}"]) for i in range(n_planets)])
                    if "a_rs" not in p and "duration" in p and "b" in p:
                        p["a_rs"] = harmonica_a_rs_from_duration(
                            p["period"],
                            p["duration"],
                            p["b"],
                            p["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    if "duration" not in p and "a_rs" in p and "b" in p:
                        p["duration"] = harmonica_duration_from_geometry(
                            p["period"],
                            p["a_rs"],
                            p["b"],
                            p["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    if "a_rs" in p and "b" in p:
                        p["cos_i"] = _harmonica_cosi_from_b(
                            p["b"], p["a_rs"], hyper_params_wl['ecc'], hyper_params_wl['omega']
                        )
                        p["inc"] = jnp.arccos(jnp.clip(p["cos_i"], 0.0, 1.0))
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    frac_key = _harmonica_frac_site(harmonic_name)
                    if harmonic_name not in p and frac_key in soln and "rors" in p:
                        p[harmonic_name] = soln[frac_key] * jnp.maximum(jnp.min(jnp.asarray(p["rors"])), 1e-6)
                if transit_engine == 'harmonica' and "rors" in p:
                    have_inc = have_all("inc")
                    if have_inc and "a_rs" in p and "b" not in p:
                        incs = jnp.array([p[f"inc_{i}"] for i in range(n_planets)])
                        p["inc"] = incs
                        p["cos_i"] = jnp.cos(incs)
                        p["b"] = harmonica_impact_param_from_cos_i(
                            p["cos_i"], p["a_rs"],
                            ecc=hyper_params_wl['ecc'], omega=hyper_params_wl['omega'],
                        )
                        p["duration"] = harmonica_duration_from_cos_i(
                            p["period"],
                            p["a_rs"],
                            p["cos_i"],
                            p["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    else:
                        if "duration" in p and "b" in p and (have_logD or "a_rs" not in p):
                            p["a_rs"] = harmonica_a_rs_from_duration(
                                p["period"],
                                p["duration"],
                                p["b"],
                                p["rors"],
                                ecc=hyper_params_wl['ecc'],
                                omega=hyper_params_wl['omega'],
                            )
                        if "a_rs" in p and "b" in p and (have_log_a_rs or "duration" not in p):
                            p["duration"] = harmonica_duration_from_geometry(
                                p["period"],
                                p["a_rs"],
                                p["b"],
                                p["rors"],
                                ecc=hyper_params_wl['ecc'],
                                omega=hyper_params_wl['omega'],
                            )
                        if "a_rs" in p and "b" in p:
                            p["cos_i"] = _harmonica_cosi_from_b(
                                p["b"], p["a_rs"], hyper_params_wl['ecc'], hyper_params_wl['omega']
                            )
                        if "duration" not in p and "cos_i" in p:
                            p["duration"] = harmonica_duration_from_cos_i(
                                p["period"],
                                p["a_rs"],
                                p["cos_i"],
                                p["rors"],
                                ecc=hyper_params_wl['ecc'],
                                omega=hyper_params_wl['omega'],
                            )
                    if ld_profile == "power2" and "c1" in p and "c2" in p:
                        p["c_ld"] = p["c1"]
                        p["alpha_ld"] = p["c2"]
                    elif ld_profile == "quadratic":
                        p["u1_ld"] = U_mu_wl[0]
                        p["u2_ld"] = U_mu_wl[1]

                return _attach_surface_eval_metadata(p, surface_config)

            try:

                params_complete = hyper_params_wl.copy()
                params_complete.update(init_params_wl)

                n_planets_sanity = len(jnp.atleast_1d(PERIOD_FIXED))

                def _ensure_len(value, n):
                    arr = jnp.atleast_1d(value)
                    if arr.size == 1 and n > 1:
                        return jnp.repeat(arr, n)
                    return arr

                params_complete["period"] = _ensure_len(params_complete.get("period", PERIOD_FIXED), n_planets_sanity)
                params_complete["duration"] = _ensure_len(params_complete.get("duration", PRIOR_DUR), n_planets_sanity)
                params_complete["t0"] = _ensure_len(params_complete.get("t0", PRIOR_T0), n_planets_sanity)
                params_complete["b"] = _ensure_len(params_complete.get("b", PRIOR_B), n_planets_sanity)
                params_complete["rors"] = _ensure_len(params_complete.get("rors", PRIOR_RPRS), n_planets_sanity)
                if transit_engine == 'harmonica':
                    if all(f'inc_{i}' in init_params_wl for i in range(n_planets_sanity)):
                        # Compatibility fallback for older direct-inc initializations.
                        params_complete["a_rs"] = jnp.array([jnp.exp(init_params_wl[f'log_a_rs_{i}']) for i in range(n_planets_sanity)])
                        incs = jnp.array([init_params_wl[f'inc_{i}'] for i in range(n_planets_sanity)])
                        params_complete["inc"] = incs
                        params_complete["cos_i"] = jnp.cos(incs)
                        params_complete["b"] = harmonica_impact_param_from_cos_i(
                            params_complete["cos_i"], params_complete["a_rs"],
                            ecc=hyper_params_wl['ecc'], omega=hyper_params_wl['omega'],
                        )
                        params_complete["duration"] = harmonica_duration_from_cos_i(
                            params_complete["period"],
                            params_complete["a_rs"],
                            params_complete["cos_i"],
                            params_complete["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    elif all(f'log_a_rs_{i}' in init_params_wl for i in range(n_planets_sanity)):
                        params_complete["a_rs"] = jnp.array([jnp.exp(init_params_wl[f'log_a_rs_{i}']) for i in range(n_planets_sanity)])
                        params_complete["duration"] = harmonica_duration_from_geometry(
                            params_complete["period"],
                            params_complete["a_rs"],
                            params_complete["b"],
                            params_complete["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                        params_complete["cos_i"] = _harmonica_cosi_from_b(
                            params_complete["b"],
                            params_complete["a_rs"],
                            hyper_params_wl['ecc'],
                            hyper_params_wl['omega'],
                        )
                    else:
                        params_complete["a_rs"] = harmonica_a_rs_from_duration(
                            params_complete["period"],
                            params_complete["duration"],
                            params_complete["b"],
                            params_complete["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                        params_complete["cos_i"] = _harmonica_cosi_from_b(
                            params_complete["b"],
                            params_complete["a_rs"],
                            hyper_params_wl['ecc'],
                            hyper_params_wl['omega'],
                        )
                    if ld_profile == "power2":
                        if "c_ld" not in params_complete and "c1" in params_complete:
                            params_complete["c_ld"] = params_complete["c1"]
                        if "alpha_ld" not in params_complete and "c2" in params_complete:
                            params_complete["alpha_ld"] = params_complete["c2"]
                    else:
                        params_complete["u1_ld"] = U_mu_wl[0]
                        params_complete["u2_ld"] = U_mu_wl[1]
                elif all(f'log_a_rs_{i}' in init_params_wl for i in range(n_planets_sanity)):
                    params_complete["a_rs"] = jnp.array([jnp.exp(init_params_wl[f'log_a_rs_{i}']) for i in range(n_planets_sanity)])
                    params_complete["duration"] = harmonica_duration_from_geometry(
                        params_complete["period"],
                        params_complete["a_rs"],
                        params_complete["b"],
                        params_complete["rors"],
                        ecc=hyper_params_wl['ecc'],
                        omega=hyper_params_wl['omega'],
                    )
                    params_complete["cos_i"] = _harmonica_cosi_from_b(
                        params_complete["b"],
                        params_complete["a_rs"],
                        hyper_params_wl['ecc'],
                        hyper_params_wl['omega'],
                    )

                key_sanity = jax.random.PRNGKey(0) if key_master is None else key_master
                keys = jax.random.split(key_sanity, num=3)

                if n_planets_sanity == 1:
                    if param_method == 'a_rs':
                        _opt_sites = ["log_a_rs_0", "t0_0", "_b_0"]
                    else:
                        _opt_sites = ["logD_0", "t0_0", "_b_0"]
                else:
                    _opt_sites = None
                if uses_surface_model:
                    soln = dict(init_params_wl)
                else:
                    stage1 = optimx.optimize(
                        whitelight_model_for_run,
                        sites=_opt_sites,
                        start=init_params_wl,
                    )
                    soln = stage1(keys[0], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                stage2_sites = ["rors_0"]
                if wl_ld_mode != "fixed":
                    if transit_engine == 'harmonica':
                        stage2_sites.extend(["c1", "c2"])
                    elif ld_profile == "quadratic":
                        stage2_sites.append(
                            ('ld_uplus_uminus'
                             if (wl_ld_mode == 'uniform'
                                 and ld_uniform_basis == 'uplus_uminus')
                             else 'u')
                        )
                    elif ld_profile == "power2":
                        stage2_sites.extend(_power2_ld_optimization_sites(
                            whitelight_ld_parameterization
                        ))
                if n_planets_sanity != 1:
                    stage2_sites = None

                if not uses_surface_model:
                    stage2 = optimx.optimize(
                        whitelight_model_for_run,
                        sites=stage2_sites,
                        start=soln,
                    )
                    soln = stage2(keys[1], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                    stage3 = optimx.optimize(
                        whitelight_model_for_run,
                        start=soln,
                    )
                    soln = stage3(keys[2], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                # Constrained optimizers may return a Uniform latent exactly
                # on its support edge.  That point has an infinite
                # unconstrained coordinate and cannot initialize NumPyro.
                if (
                    'linear_discontinuity' in detrending_type
                    and step_width_mode == 'free'
                ):
                    cadence = float(
                        np.median(np.diff(np.sort(np.asarray(data.wl_time))))
                    )
                    log_width_lo = np.log(0.5 * cadence)
                    log_width_hi = np.log(30.0 / (24.0 * 60.0))
                    optimized_log_width = float(
                        np.asarray(soln.get('log_width', np.nan))
                    )
                    edge_tolerance = 1.0e-8 * max(
                        1.0, abs(log_width_hi - log_width_lo)
                    )
                    if (
                        not np.isfinite(optimized_log_width)
                        or optimized_log_width <= log_width_lo + edge_tolerance
                        or optimized_log_width >= log_width_hi - edge_tolerance
                    ):
                        reset_log_width = 0.5 * (
                            log_width_lo + log_width_hi
                        )
                        print(
                            "WARNING: white-light optimizer placed log_width "
                            "on or outside its support edge; resetting it to "
                            "the interior prior midpoint before Laplace "
                            "preparation."
                        )
                        soln = dict(soln)
                        soln['log_width'] = jnp.asarray(reset_log_width)
                if is_gp_detrending:
                    soln = _repair_gp_support_edges(soln)

                optimized_start_valid, optimized_start_reasons = (
                    _validate_whitelight_optimized_start(
                        soln,
                        ld_profile=ld_profile,
                        ld_parameterization=whitelight_ld_parameterization,
                        ld_prior_mode=ld_prior_mode,
                        quadratic_uniform_bounds=quadratic_uniform_physical_bounds,
                        n_planets=n_planets_sanity,
                    )
                )
                if not optimized_start_valid:
                    print(
                        "WARNING: discarding nonphysical white-light optimizer "
                        "solution and falling back to the prior physical start: "
                        + "; ".join(optimized_start_reasons),
                        flush=True,
                    )
                    soln = dict(init_params_wl)
                    fallback_valid, fallback_reasons = (
                        _validate_whitelight_optimized_start(
                            soln,
                            ld_profile=ld_profile,
                            ld_parameterization=whitelight_ld_parameterization,
                            ld_prior_mode=ld_prior_mode,
                            quadratic_uniform_bounds=quadratic_uniform_physical_bounds,
                            n_planets=n_planets_sanity,
                        )
                    )
                    if not fallback_valid:
                        raise RuntimeError(
                            "White-light prior fallback start is invalid: "
                            + "; ".join(fallback_reasons)
                        )

                params_opt = _soln_to_physical_params(soln, params_complete, n_planets=n_planets_sanity)
                if "rors" not in params_opt and "depths" in params_opt:
                    params_opt["rors"] = jnp.sqrt(params_opt["depths"])

                flux_init = _get_sanity_model(params_complete, data.wl_time)
                flux_opt  = _get_sanity_model(params_opt,      data.wl_time)
                plt.figure(figsize=(10, 6))
                plt.scatter(data.wl_time, data.wl_flux, c='k', s=10, alpha=0.3, label='Data')
                
                plt.plot(data.wl_time, flux_init, color='red', linestyle='--', lw=2, alpha=0.7, 
                         label='Init Guess')
                plt.plot(data.wl_time, flux_opt, color='blue', lw=2.5, 
                         label='Optimized (Gradient)')
                
                plt.title(f'Sanity Check: {detrending_type} (GP Pre-fit: {"gp" in detrending_type})')
                plt.xlabel('Time (BJD)')
                plt.ylabel('Flux')
                plt.legend()
                plt.tight_layout()
                
                sanity_plot_path = f'{output_dir}/00_{instrument_full_str}_init_vs_opt_check.png'
                plt.savefig(sanity_plot_path)
                plt.close()
                print(f"Saved sanity check plot to: {sanity_plot_path}")
                
            except Exception as e:
                print(f"Could not create sanity plot: {e}")

            wl_nuts_kwargs = {
                "init_strategy": numpyro.infer.init_to_value(values=soln),
                **(harmonica_wl_nuts_kwargs if transit_engine == 'harmonica' else NUTS_KWARGS),
            }
            laplace_preparation = None
            laplace_diagnostics = {}
            wl_run_init_params = None
            adaptive_wl_nuts_kwargs = dict(wl_nuts_kwargs)
            adaptive_wl_mcmc_kwargs = dict(whitelight_mcmc_kwargs)
            if (
                transit_engine == 'jaxoplanet'
                and whitelight_laplace_options["mass_matrix"] == "laplace"
            ):
                from models.independent_nuts import prepare_laplace_metric

                preparation_start = time.perf_counter()
                laplace_preparation = prepare_laplace_metric(
                    whitelight_model_for_run,
                    key_master,
                    soln,
                    data.wl_time,
                    data.wl_flux_err,
                    model_kwargs={
                        "y": data.wl_flux,
                        "prior_params": hyper_params_wl,
                    },
                    hessian_method=whitelight_laplace_options["hessian_method"],
                    trust_radius=whitelight_laplace_options["trust_radius"],
                    max_iterations=200,
                    # White-light time/flux coordinates naturally span a
                    # wider curvature range than per-channel spectroscopy;
                    # the 1e-8 lane floor would truncate real geometry modes.
                    eigenvalue_floor=1.0e-12,
                    # Starry surface objectives are substantially larger than
                    # transit-only objectives.  Evaluating finite-difference
                    # and line-search points sequentially keeps peak compiler
                    # memory bounded without changing the target or metric.
                    sequential_evaluations=uses_surface_model,
                )
                laplace_preparation.inverse_mass_matrix.block_until_ready()
                preparation_wall = time.perf_counter() - preparation_start
                wl_nuts_kwargs.update(
                    dense_mass=True,
                    inverse_mass_matrix=laplace_preparation.inverse_mass_matrix,
                    adapt_mass_matrix=False,
                    target_accept_prob=whitelight_laplace_options["target_accept"],
                    max_tree_depth=whitelight_laplace_options["max_tree_depth"],
                )
                whitelight_mcmc_kwargs = {
                    **whitelight_mcmc_kwargs,
                    "num_warmup": whitelight_laplace_options["warmup"],
                }
                wl_run_init_params = laplace_preparation.unconstrained_map
                laplace_diagnostics = {
                    "mass_matrix": "laplace",
                    "laplace_preparation_wall_seconds": preparation_wall,
                    "map_gradient_norm": float(laplace_preparation.gradient_norm),
                    "map_newton_decrement": float(laplace_preparation.newton_decrement),
                    "map_iterations": int(laplace_preparation.iterations),
                    "map_condition_number": float(laplace_preparation.condition_number),
                    "map_hessian_min_eigenvalue": float(
                        laplace_preparation.hessian_min_eigenvalue
                    ),
                    "laplace_hessian_method": whitelight_laplace_options[
                        "hessian_method"
                    ],
                }
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(whitelight_model_for_run, **wl_nuts_kwargs),
                progress_bar=True,
                jit_model_args=True,
                **whitelight_mcmc_kwargs,
            )
            sampling_start = time.perf_counter()
            wl_run_args = (data.wl_time, data.wl_flux_err)
            wl_run_kwargs = {
                "y": data.wl_flux,
                "prior_params": hyper_params_wl,
                "init_params": wl_run_init_params,
                "extra_fields": (
                    "diverging", "accept_prob", "potential_energy", "num_steps"
                ),
            }
            mcmc.run(key_master, *wl_run_args, **wl_run_kwargs)
            jax.block_until_ready(next(iter(mcmc.get_samples().values())))
            laplace_diagnostics["sampling_wall_seconds"] = (
                time.perf_counter() - sampling_start
            )
            (
                wl_samples,
                wl_samples_grouped,
                wl_extra_fields,
                wl_quality_diagnostics,
            ) = _continue_mcmc_until_geometry_gate(
                mcmc,
                key_master,
                wl_run_args,
                wl_run_kwargs,
                min_ess=400.0,
                max_divergences=0,
                max_extra_blocks=3,
                failfast_ess=50.0,
            )
            laplace_diagnostics.update(wl_quality_diagnostics)
            laplace_diagnostics.setdefault("mass_matrix", "adaptive")
            if (
                not wl_quality_diagnostics["whitelight_quality_gate_passed"]
                and laplace_diagnostics["mass_matrix"] == "laplace"
            ):
                print(
                    "WARNING: LAPLACE WHITE-LIGHT CHAIN FAILED THE QUALITY "
                    "GATE; AUTOMATICALLY FALLING BACK TO ADAPTIVE NUTS."
                )
                _, adaptive_key = jax.random.split(key_master)
                mcmc = numpyro.infer.MCMC(
                    numpyro.infer.NUTS(
                        whitelight_model_for_run, **adaptive_wl_nuts_kwargs
                    ),
                    progress_bar=True,
                    jit_model_args=True,
                    **adaptive_wl_mcmc_kwargs,
                )
                adaptive_run_kwargs = dict(wl_run_kwargs)
                adaptive_run_kwargs["init_params"] = None
                mcmc.run(adaptive_key, *wl_run_args, **adaptive_run_kwargs)
                (
                    wl_samples,
                    wl_samples_grouped,
                    wl_extra_fields,
                    wl_quality_diagnostics,
                ) = _continue_mcmc_until_geometry_gate(
                    mcmc,
                    adaptive_key,
                    wl_run_args,
                    adaptive_run_kwargs,
                    min_ess=400.0,
                    max_divergences=0,
                    max_extra_blocks=3,
                    failfast_ess=0.0,
                )
                laplace_diagnostics.update(wl_quality_diagnostics)
                laplace_diagnostics["mass_matrix"] = "adaptive_fallback"
                laplace_diagnostics["whitelight_laplace_fell_back"] = True
            if not wl_quality_diagnostics["whitelight_quality_gate_passed"]:
                print(
                    "WARNING: WHITE-LIGHT MCMC FAILED ITS ESS/DIVERGENCE GATE "
                    "AFTER ALL EXTENSION BLOCKS."
                )
            _save_mcmc_diagnostics(
                mcmc,
                max_tree_depth=wl_nuts_kwargs.get("max_tree_depth", 10),
                output_path=os.path.join(
                    output_dir, "whitelight_mcmc_diagnostics.json"
                ),
                extra_diagnostics=laplace_diagnostics,
                samples_override=wl_samples,
                extra_fields_override=wl_extra_fields,
            )
            inf_data = az.from_dict(posterior=wl_samples_grouped)
            print(az.summary(inf_data, var_names=None, round_to=7))

            bestfit_params_wl = {
                'period': PERIOD_FIXED,
            }
            def set_param_stats(name, data_samples, axis=0):
                med, low, high = get_asym_errors(data_samples, axis=axis)
                bestfit_params_wl[name] = med
                bestfit_params_wl[f'{name}_err_low'] = low
                bestfit_params_wl[f'{name}_err_high'] = high
            def set_fixed_param_stats(name, values):
                values = jnp.asarray(values)
                zeros = jnp.zeros_like(values)
                bestfit_params_wl[name] = values
                bestfit_params_wl[f'{name}_err'] = zeros
                bestfit_params_wl[f'{name}_err_low'] = zeros
                bestfit_params_wl[f'{name}_err_high'] = zeros

            if ld_profile == 'power2':
                if 'c1' in wl_samples and 'c2' in wl_samples:
                    set_param_stats('c1', wl_samples['c1'], axis=0)
                    set_param_stats('c2', wl_samples['c2'], axis=0)
                else:
                    set_fixed_param_stats('c1', U_mu_wl[0])
                    set_fixed_param_stats('c2', U_mu_wl[1])
                POLY_DEGREE = 12
                MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
                power2_profile = get_I_power2(bestfit_params_wl['c1'], bestfit_params_wl['c2'], MUS)
                u_poly = calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
                bestfit_params_wl['u'] = u_poly
                if transit_engine == 'harmonica':
                    bestfit_params_wl['c_ld'] = bestfit_params_wl['c1']
                    bestfit_params_wl['alpha_ld'] = bestfit_params_wl['c2']
            else:
                if 'u' in wl_samples:
                    set_param_stats('u', wl_samples['u'], axis=0)
                else:
                    set_fixed_param_stats('u', U_mu_wl)
                if transit_engine == 'harmonica':
                    if 'u1' in wl_samples and 'u2' in wl_samples:
                        set_param_stats('u1', wl_samples['u1'], axis=0)
                        set_param_stats('u2', wl_samples['u2'], axis=0)
                    else:
                        set_fixed_param_stats('u1', U_mu_wl[0])
                        set_fixed_param_stats('u2', U_mu_wl[1])
                    bestfit_params_wl['u1_ld'] = bestfit_params_wl['u1']
                    bestfit_params_wl['u2_ld'] = bestfit_params_wl['u2']
            set_fixed_param_stats('ecc', HARMONICA_ECC)
            set_fixed_param_stats('omega', HARMONICA_OMEGA)
            for surface_name in (
                'eclipse_depth', 'dayside_flux', 'nightside_flux',
                'hotspot_offset', 'stellar_spot_contrast',
            ):
                if surface_name in wl_samples:
                    set_param_stats(surface_name, wl_samples[surface_name])
            if transit_engine == 'harmonica':
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    if harmonic_name in wl_samples:
                        set_param_stats(harmonic_name, wl_samples[harmonic_name])

            # Post-hoc geometry derivation (engine-agnostic).
            derived = derive_geometry(wl_samples, PERIOD_FIXED, ecc=HARMONICA_ECC, omega=HARMONICA_OMEGA)

            durations_fit, t0s_fit, bs_fit, rors_fit = [], [], [], []
            cos_is_fit, a_rss_fit = [], []
            durations_err, t0s_err, bs_err, rors_err, depths_err = [], [], [], [], []
            cos_is_err, a_rss_err = [], []
            durations_err_low, t0s_err_low, bs_err_low, rors_err_low, depths_err_low = [], [], [], [], []
            cos_is_err_low, a_rss_err_low = [], []
            durations_err_high, t0s_err_high, bs_err_high, rors_err_high, depths_err_high = [], [], [], [], []
            cos_is_err_high, a_rss_err_high = [], []

            for i in range(n_planets):
                dur_s = derived[f'duration_{i}']
                b_s = derived[f'b_{i}']
                a_rs_s = derived[f'a_rs_{i}']
                cos_i_s = derived[f'cos_i_{i}']

                med_d, low_d, high_d = get_asym_errors(dur_s)
                med_t0, low_t0, high_t0 = get_asym_errors(wl_samples[f't0_{i}'])
                med_b, low_b, high_b = get_asym_errors(b_s)
                med_r, low_r, high_r = get_asym_errors(wl_samples[f'rors_{i}'])
                med_depth, low_depth, high_depth = get_asym_errors(wl_samples[f'rors_{i}']**2)
                med_cos_i, low_cos_i, high_cos_i = get_asym_errors(cos_i_s)
                med_a_rs, low_a_rs, high_a_rs = get_asym_errors(a_rs_s)

                durations_fit.append(med_d); t0s_fit.append(med_t0)
                bs_fit.append(med_b); rors_fit.append(med_r)
                cos_is_fit.append(med_cos_i); a_rss_fit.append(med_a_rs)

                durations_err.append(jnp.std(dur_s))
                t0s_err.append(jnp.std(wl_samples[f't0_{i}']))
                bs_err.append(jnp.std(b_s))
                rors_err.append(jnp.std(wl_samples[f'rors_{i}']))
                depths_err.append(jnp.std(wl_samples[f'rors_{i}']**2))
                cos_is_err.append(jnp.std(cos_i_s))
                a_rss_err.append(jnp.std(a_rs_s))

                durations_err_low.append(low_d); durations_err_high.append(high_d)
                t0s_err_low.append(low_t0); t0s_err_high.append(high_t0)
                bs_err_low.append(low_b); bs_err_high.append(high_b)
                rors_err_low.append(low_r); rors_err_high.append(high_r)
                depths_err_low.append(low_depth); depths_err_high.append(high_depth)
                cos_is_err_low.append(low_cos_i); cos_is_err_high.append(high_cos_i)
                a_rss_err_low.append(low_a_rs); a_rss_err_high.append(high_a_rs)

            bestfit_params_wl['duration'] = jnp.array(durations_fit)
            bestfit_params_wl['t0'] = jnp.array(t0s_fit)
            bestfit_params_wl['b'] = jnp.array(bs_fit)
            bestfit_params_wl['rors'] = jnp.array(rors_fit)
            bestfit_params_wl['depths'] = jnp.array(rors_fit)**2
            bestfit_params_wl['cos_i'] = jnp.array(cos_is_fit)
            bestfit_params_wl['a_rs'] = jnp.array(a_rss_fit)

            bestfit_params_wl['duration_err'] = jnp.array(durations_err)
            bestfit_params_wl['t0_err'] = jnp.array(t0s_err)
            bestfit_params_wl['b_err'] = jnp.array(bs_err)
            bestfit_params_wl['rors_err'] = jnp.array(rors_err)
            bestfit_params_wl['depths_err'] = jnp.array(depths_err)
            bestfit_params_wl['cos_i_err'] = jnp.array(cos_is_err)
            bestfit_params_wl['a_rs_err'] = jnp.array(a_rss_err)

            bestfit_params_wl['duration_err_low'] = jnp.array(durations_err_low)
            bestfit_params_wl['duration_err_high'] = jnp.array(durations_err_high)
            bestfit_params_wl['t0_err_low'] = jnp.array(t0s_err_low)
            bestfit_params_wl['t0_err_high'] = jnp.array(t0s_err_high)
            bestfit_params_wl['b_err_low'] = jnp.array(bs_err_low)
            bestfit_params_wl['b_err_high'] = jnp.array(bs_err_high)
            bestfit_params_wl['rors_err_low'] = jnp.array(rors_err_low)
            bestfit_params_wl['rors_err_high'] = jnp.array(rors_err_high)
            bestfit_params_wl['depths_err_low'] = jnp.array(depths_err_low)
            bestfit_params_wl['depths_err_high'] = jnp.array(depths_err_high)
            bestfit_params_wl['cos_i_err_low'] = jnp.array(cos_is_err_low)
            bestfit_params_wl['cos_i_err_high'] = jnp.array(cos_is_err_high)
            bestfit_params_wl['a_rs_err_low'] = jnp.array(a_rss_err_low)
            bestfit_params_wl['a_rs_err_high'] = jnp.array(a_rss_err_high)

            set_param_stats('error', wl_samples['error'])
            
            if detrending_type != 'none':
                set_param_stats('c', wl_samples['c'])
                if detrending_type != 'gp': 
                    set_param_stats('v', wl_samples['v'])
                if 'v' in wl_samples and 'v' not in bestfit_params_wl:
                     set_param_stats('v', wl_samples['v'])

            if 'v2' in wl_samples: set_param_stats('v2', wl_samples['v2'])
            if 'v3' in wl_samples: set_param_stats('v3', wl_samples['v3'])
            if 'v4' in wl_samples: set_param_stats('v4', wl_samples['v4'])
            if 'explinear' in detrending_type:
                set_param_stats('A', wl_samples['A'])
                set_param_stats('tau', wl_samples['tau'])
            if 'spot' in detrending_type:
                set_param_stats('spot_amp', wl_samples['spot_amp'])
                set_param_stats('spot_mu', wl_samples['spot_mu'])
                set_param_stats('spot_sigma', wl_samples['spot_sigma'])
            if '2spot' in detrending_type:
                set_param_stats('spot_amp2', wl_samples['spot_amp2'])
                set_param_stats('spot_mu2', wl_samples['spot_mu2'])
                set_param_stats('spot_sigma2', wl_samples['spot_sigma2'])
            if 'linear_discontinuity' in detrending_type:
                set_param_stats('t_jump', wl_samples['t_jump'])
                set_param_stats('jump', wl_samples['jump'])
                set_param_stats('width', wl_samples['width'])
                set_param_stats('width_minutes', wl_samples['width_minutes'])
                if 'log_width' in wl_samples:
                    set_param_stats('log_width', wl_samples['log_width'])
    
            if 'gp' in detrending_type:
                set_param_stats('GP_log_sigma', wl_samples['GP_log_sigma'])
                set_param_stats('GP_log_rho', wl_samples['GP_log_rho'])

            wl_handoff_geometry = _geometry_from_white_light_medians(
                bestfit_params_wl, PERIOD_FIXED
            )
            wl_handoff_payload = {
                "estimator": "posterior_median",
                "posterior_fingerprint_sha256": wl_artifact_fingerprint,
                "transit_engine": transit_engine,
                "param_method": param_method,
                "num_retained_draws": _posterior_num_draws(wl_samples),
                "selected_flat_draw_index": None,
                "selected_chain_index": None,
                "selected_draw_index": None,
                "summed_data_log_likelihood": None,
                "selected_primitive_parameters": {},
                "geometry": wl_handoff_geometry,
            }
            spot_trend, spot_trend2, jump_trend = None, None, None
            if 'spot' in detrending_type and '2spot' not in detrending_type:
                spot_trend = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp"],
                    bestfit_params_wl["spot_mu"],
                    jnp.abs(bestfit_params_wl["spot_sigma"])
                )
            if '2spot' in detrending_type:
                spot_trend = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp"],
                    bestfit_params_wl["spot_mu"],
                    jnp.abs(bestfit_params_wl["spot_sigma"])
                )
                spot_trend2 = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp2"],
                    bestfit_params_wl["spot_mu2"],
                    jnp.abs(bestfit_params_wl["spot_sigma2"])
                )
            if 'linear_discontinuity' in detrending_type:
                jump_trend = bestfit_params_wl["jump"] * _soft_step_np(
                    np.array(data.wl_time), bestfit_params_wl["t_jump"],
                    bestfit_params_wl["width"],
                )

            model_eval_params_wl = _select_transit_eval_params(
                bestfit_params_wl,
                transit_engine=transit_engine,
                param_method=param_method,
                t=data.wl_time,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
                surface_config=surface_config,
            )

            if 'gp' in detrending_type:
                gp_products = _compute_whitelight_gp_products(
                    model_eval_params_wl,
                    data.wl_time,
                    bestfit_params_wl['error'],
                    data.wl_flux,
                    detrend_type=detrending_type,
                    gp_solver=gp_solver,
                )
                mu = gp_products["mu"]
                var = gp_products["var"]
                wl_transit_model = mu
                planet_model_only = gp_products["planet_model_only"]
                trend_flux_total = gp_products["trend_flux_total"]
                parametric_mean_val = gp_products["parametric_mean"]
                gp_stochastic_component = gp_products["gp_stochastic_component"]

            else:
                try:
                    wl_transit_model = resolve_detrend_kernel(detrending_type)(model_eval_params_wl, data.wl_time)
                except KeyError:
                    print('Error with model, not defined!')
                    exit()

            wl_residual = data.wl_flux - wl_transit_model
            wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
            wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma
            wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))
            plot_whitelight_curve(
                data.wl_time,
                data.wl_flux,
                data.wl_flux_err,
                wl_transit_model,
                f"{output_dir}/11_{instrument_full_str}_whitelightmodel.png",
                instrument_label=instrument_full_str,
                t0_reference=bestfit_params_wl['t0'][0],
                outlier_mask=wl_mad_mask,
            )
            plot_whitelight_residuals(
                data.wl_time,
                wl_residual,
                data.wl_flux_err,
                f"{output_dir}/12_{instrument_full_str}_whitelightresidual.png",
                instrument_label=instrument_full_str,
                t0_reference=bestfit_params_wl['t0'][0],
                outlier_mask=wl_mad_mask,
            )
            try:
                corner_path = plot_whitelight_corner(
                    wl_samples,
                    f"{output_dir}/13_{instrument_full_str}_whitelight_corner.png",
                    instrument_label=instrument_full_str,
                )
            except Exception as error:  # A diagnostic must never abort the fit.
                print(f"White-light corner plot skipped: {error}")
            else:
                if corner_path:
                    print(f"Saved white-light corner plot to: {corner_path}")

            t_masked = data.wl_time[~wl_mad_mask]
            f_masked = data.wl_flux[~wl_mad_mask]
            # The diagnostic evaluator carries time-axis-specific phase
            # offsets. Rebuild them after outlier clipping instead of reusing
            # the full-cadence arrays with ``t_masked``.
            masked_model_eval_params_wl = _select_transit_eval_params(
                bestfit_params_wl,
                transit_engine=transit_engine,
                param_method=param_method,
                t=t_masked,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
                surface_config=surface_config,
            )

            if 'gp' in detrending_type:
                planet_model_masked = compute_transit_model_auto(masked_model_eval_params_wl, t_masked)
                mu_masked = mu[~wl_mad_mask] 
                total_trend_at_points = mu_masked - planet_model_masked
                detrended_flux = f_masked - (total_trend_at_points - 1.0)
            else:
                trend = _trend_from_params_np(
                    detrending_type,
                    np.array(t_masked),
                    bestfit_params_wl
                )
                detrended_flux = f_masked - trend + 1.0

            transit_only_model = compute_transit_model_auto(masked_model_eval_params_wl, t_masked) + 1.0
            wl_flux_err_full = np.broadcast_to(
                np.asarray(data.wl_flux_err, dtype=float),
                np.asarray(data.wl_time).shape,
            )
            plot_whitelight_curve(
                t_masked,
                detrended_flux,
                wl_flux_err_full[~np.asarray(wl_mad_mask)],
                transit_only_model,
                f'{output_dir}/14_{instrument_full_str}_whitelightdetrended.png',
                instrument_label=instrument_full_str,
                t0_reference=bestfit_params_wl['t0'][0],
                flux_label="Detrended Flux",
            )

            dt = np.median(np.diff(data.wl_time)) * 86400 
            residuals_arr = np.array(wl_residual[~wl_mad_mask])
            beta, _, _, _ = calculate_beta_metrics(residuals_arr, dt)
            mc_betas, _, _, _, _ = run_beta_monte_carlo(
                residuals_arr, dt, n_sims=500
            )

            mu_sim = float(np.mean(mc_betas))
            std_sim = float(np.std(mc_betas))
            z_score = (beta - mu_sim) / std_sim

            print(
                "White-light beta summary: "
                f"measured={beta:.3f}, MC={mu_sim:.3f}+/-{std_sim:.3f}, "
                f"significance={z_score:.2f} sigma"
            )

            plot_whitelight_summary(
                bestfit_params_wl,
                f'{output_dir}/15_{instrument_full_str}_whitelight_summary.png',
                instrument_label=instrument_full_str,
            )

            transit_only_full = np.asarray(
                compute_transit_model_auto(model_eval_params_wl, data.wl_time) + 1.0,
                dtype=float,
            )
            bestfit_model_full = np.asarray(wl_transit_model, dtype=float)
            trend_model_full = bestfit_model_full - transit_only_full
            gp_flux_full = np.asarray(mu, dtype=float) if 'gp' in detrending_type else None
            gp_err_full = np.asarray(jnp.sqrt(var), dtype=float) if 'gp' in detrending_type else None
            gp_trend_full = np.asarray(gp_stochastic_component, dtype=float) if 'gp' in detrending_type else None
            save_whitelight_timeseries(
                time=data.wl_time,
                flux=data.wl_flux,
                flux_err=data.wl_flux_err,
                bestfit_model=bestfit_model_full,
                output_csv=f'{output_dir}/{instrument_full_str}_whitelight_timeseries.csv',
                t0_reference=float(np.asarray(bestfit_params_wl['t0']).ravel()[0]) if np.size(bestfit_params_wl['t0']) else None,
                transit_model=transit_only_full,
                trend_model=trend_model_full,
                residual=np.asarray(wl_residual, dtype=float),
                outlier_mask=np.asarray(wl_mad_mask, dtype=bool),
                gp_flux=gp_flux_full,
                gp_err=gp_err_full,
                gp_trend=gp_trend_full,
            )

            
            _atomic_save_npy(wl_mask_path, wl_mad_mask)
            
            if 'gp' in detrending_type:
                _save_whitelight_gp_database(
                    wl_gp_path, data.wl_flux, gp_products
                )
            
            rows = []
            for i in range(n_planets):
                row = {
                    'planet_id': i,
                    'period': bestfit_params_wl['period'][i],
                    'duration': bestfit_params_wl['duration'][i],
                    't0': bestfit_params_wl['t0'][i],
                    'b': bestfit_params_wl['b'][i],
                    'rors': bestfit_params_wl['rors'][i],
                    'depths': bestfit_params_wl['depths'][i],
                    'duration_err': bestfit_params_wl['duration_err'][i],
                    't0_err': bestfit_params_wl['t0_err'][i],
                    'b_err': bestfit_params_wl['b_err'][i],
                    'rors_err': bestfit_params_wl['rors_err'][i],
                    'depths_err': bestfit_params_wl['depths_err'][i],
                    'duration_err_low': bestfit_params_wl['duration_err_low'][i],
                    'duration_err_high': bestfit_params_wl['duration_err_high'][i],
                    't0_err_low': bestfit_params_wl['t0_err_low'][i],
                    't0_err_high': bestfit_params_wl['t0_err_high'][i],
                    'b_err_low': bestfit_params_wl['b_err_low'][i],
                    'b_err_high': bestfit_params_wl['b_err_high'][i],
                    'rors_err_low': bestfit_params_wl['rors_err_low'][i],
                    'rors_err_high': bestfit_params_wl['rors_err_high'][i],
                    'depths_err_low': bestfit_params_wl['depths_err_low'][i],
                    'depths_err_high': bestfit_params_wl['depths_err_high'][i],
                }
                if 'a_rs' in bestfit_params_wl:
                    row['a_rs'] = bestfit_params_wl['a_rs'][i]
                    row['a_rs_err'] = bestfit_params_wl['a_rs_err'][i]
                    row['a_rs_err_low'] = bestfit_params_wl['a_rs_err_low'][i]
                    row['a_rs_err_high'] = bestfit_params_wl['a_rs_err_high'][i]
                if 'cos_i' in bestfit_params_wl:
                    row['cos_i'] = bestfit_params_wl['cos_i'][i]
                    row['cos_i_err'] = bestfit_params_wl['cos_i_err'][i]
                    row['cos_i_err_low'] = bestfit_params_wl['cos_i_err_low'][i]
                    row['cos_i_err_high'] = bestfit_params_wl['cos_i_err_high'][i]
                if 'ecc' in bestfit_params_wl:
                    row['ecc'] = bestfit_params_wl['ecc'][i]
                    row['ecc_err'] = bestfit_params_wl['ecc_err'][i]
                    row['ecc_err_low'] = bestfit_params_wl['ecc_err_low'][i]
                    row['ecc_err_high'] = bestfit_params_wl['ecc_err_high'][i]
                if 'omega' in bestfit_params_wl:
                    row['omega'] = bestfit_params_wl['omega'][i]
                    row['omega_err'] = bestfit_params_wl['omega_err'][i]
                    row['omega_err_low'] = bestfit_params_wl['omega_err_low'][i]
                    row['omega_err_high'] = bestfit_params_wl['omega_err_high'][i]
                
                def add_scalar_param(name):
                    if name in bestfit_params_wl:
                        row[name] = bestfit_params_wl[name]
                        if f'{name}_err_low' in bestfit_params_wl:
                            row[f'{name}_err_low'] = bestfit_params_wl[f'{name}_err_low']
                            row[f'{name}_err_high'] = bestfit_params_wl[f'{name}_err_high']

                if detrending_type != 'none':
                    add_scalar_param('c')
                    add_scalar_param('v')
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    add_scalar_param(harmonic_name)
                
                if 'c1' in bestfit_params_wl:
                    add_scalar_param('c1')
                    add_scalar_param('c2')
                    row['u1'] = bestfit_params_wl['c1']
                    row['u2'] = bestfit_params_wl['c2']
                    row['u1_err_low'] = bestfit_params_wl['c1_err_low']
                    row['u1_err_high'] = bestfit_params_wl['c1_err_high']
                    row['u2_err_low'] = bestfit_params_wl['c2_err_low']
                    row['u2_err_high'] = bestfit_params_wl['c2_err_high']
                else:
                    row['u1'] = bestfit_params_wl['u'][0]
                    row['u2'] = bestfit_params_wl['u'][1]
                    if 'u_err_low' in bestfit_params_wl:
                        row['u1_err_low'] = bestfit_params_wl['u_err_low'][0]
                        row['u1_err_high'] = bestfit_params_wl['u_err_high'][0]
                        row['u2_err_low'] = bestfit_params_wl['u_err_low'][1]
                        row['u2_err_high'] = bestfit_params_wl['u_err_high'][1]
                
                add_scalar_param('v2')
                add_scalar_param('v3')
                add_scalar_param('v4')
                add_scalar_param('A')
                add_scalar_param('tau')
                add_scalar_param('spot_amp')
                add_scalar_param('spot_mu')
                add_scalar_param('spot_sigma')
                add_scalar_param('spot_amp2')
                add_scalar_param('spot_mu2')
                add_scalar_param('spot_sigma2')
                add_scalar_param('t_jump')
                add_scalar_param('jump')
                add_scalar_param('width')
                add_scalar_param('width_minutes')
                add_scalar_param('log_width')
                add_scalar_param('GP_log_sigma')
                add_scalar_param('GP_log_rho')
                for name in (
                    'eclipse_depth', 'dayside_flux', 'nightside_flux',
                    'hotspot_offset',
                ):
                    if name in bestfit_params_wl:
                        values = np.atleast_1d(bestfit_params_wl[name])
                        row[name] = values[i]
                        row[f'{name}_err_low'] = np.atleast_1d(
                            bestfit_params_wl[f'{name}_err_low']
                        )[i]
                        row[f'{name}_err_high'] = np.atleast_1d(
                            bestfit_params_wl[f'{name}_err_high']
                        )[i]
                if 'stellar_spot_contrast' in bestfit_params_wl:
                    for spot_index, value in enumerate(np.atleast_1d(
                        bestfit_params_wl['stellar_spot_contrast']
                    )):
                        row[f'stellar_spot_contrast_{spot_index}'] = value
                        row[f'stellar_spot_contrast_{spot_index}_err_low'] = np.atleast_1d(
                            bestfit_params_wl['stellar_spot_contrast_err_low']
                        )[spot_index]
                        row[f'stellar_spot_contrast_{spot_index}_err_high'] = np.atleast_1d(
                            bestfit_params_wl['stellar_spot_contrast_err_high']
                        )[spot_index]
                
                rows.append(row)
        
            df = pd.DataFrame(rows)
            _atomic_dataframe_csv(df, wl_params_path, index=False)
            if uses_surface_model:
                white_surface_samples = {
                    name: np.asarray(wl_samples[name])[:, None, ...]
                    for name in (
                        *SURFACE_PARAMS,
                        'eclipse_depth_ppm', 'dayside_flux_ppm',
                        'nightside_flux_ppm', 'hotspot_offset_deg',
                    )
                    if name in wl_samples
                }
                band = np.asarray(data.wavelengths_unbinned, dtype=float)
                save_surface_results(
                    np.asarray([np.nanmean(band)]),
                    np.asarray([0.5 * (np.nanmax(band) - np.nanmin(band))]),
                    white_surface_samples,
                    wl_params_path,
                )
            bestfit_params_wl_df = pd.read_csv(wl_params_path)
            if transit_engine == 'harmonica' and _has_harmonica_odd_samples(wl_samples):
                band_wl_raw = np.asarray(data.wavelengths_hr)
                band_wl_err_raw = np.asarray(data.wavelengths_err_hr)
                if band_wl_raw.size == 0:
                    band_wl_raw = np.asarray(data.wavelengths_lr)
                    band_wl_err_raw = np.asarray(data.wavelengths_err_lr)
                band_wl_arr, band_wl_err_arr = _coerce_wavelength_axis(
                    band_wl_raw,
                    band_wl_err_raw if band_wl_raw.size > 0 else None,
                )
                bandpass_min = float(np.nanmin(band_wl_arr - band_wl_err_arr))
                bandpass_max = float(np.nanmax(band_wl_arr + band_wl_err_arr))
                save_harmonica_limb_products(
                    wavelengths=0.5 * (bandpass_min + bandpass_max),
                    wavelength_err=0.5 * (bandpass_max - bandpass_min),
                    rors_samples=wl_samples['rors_0'],
                    harmonic_samples=_harmonica_sample_payload(wl_samples),
                    csv_path=wl_limb_csv_path,
                    limb_spectrum_path=f"{output_dir}/16_{instrument_full_str}_whitelight_limb_spectra.png",
                    transmission_strings_path=f"{output_dir}/17_{instrument_full_str}_whitelight_transmission_string.png",
                    posterior_samples_path=wl_limb_samples_path,
                    title_prefix=f"{planet_str} - Whitelight",
                    bandpass_min=bandpass_min,
                    bandpass_max=bandpass_max,
                )

            wl_geometry_handoff = _write_whitelight_geometry_handoff(
                wl_geometry_handoff_path, wl_handoff_payload
            )
            print(
                "Saved white-light fixed-geometry handoff to "
                f"{wl_geometry_handoff_path}."
            )
            wl_artifact_set.write_manifest()
        else:
            print(f'GP trends already exist...')
            wl_mad_mask = np.load(wl_mask_path)
            bestfit_params_wl_df = pd.read_csv(wl_params_path)
    else:
        print(f'Whitelight outliers and bestfit parameters already exist...')
        wl_mad_mask = np.load(wl_mask_path)
        bestfit_params_wl_df = pd.read_csv(wl_params_path)
    return (wl_mad_mask, bestfit_params_wl_df, wl_geometry_handoff,)

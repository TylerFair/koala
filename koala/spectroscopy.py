"""Literal low- and high-resolution spectroscopic stages."""

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
from models.cadence_reduction import (
    build_linear_oot_statistics,
    build_linear_spectro_trend_design,
    linear_spectro_trend_coefficient_names,
)
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
    _resolve_spectro_joint_geometry,
    _resolve_spectro_joint_geometry_sampler,
    _resolve_spectro_joint_geometry_chunk_size,
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
    _joint_geometry_sites, _whitelight_geometry_prior,
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
    get_samples_chunked, _run_sampling_stage, evaluate_channels_sequentially,
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
from models.harmonica.core import (
    _ALL_ODD_COEFF_SPECS,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_area_radius_and_q,
)

from . import artifacts, config, constants, data, geometry, harmonica_products
from . import limb_darkening, outputs, sampling, surface
for _module in (artifacts, config, constants, data, geometry, harmonica_products,
                limb_darkening, outputs, sampling, surface):
    globals().update({name: value for name, value in vars(_module).items()
                      if not name.startswith("__")})


def _prepare_joint_geometry(
    flags, *, stage_name, transit_engine, spectro_sampler, nuts_kwargs,
    bestfit_params_wl_df, wl_geometry_handoff, param_method,
):
    """Resolve ``flags.spectro_joint_geometry`` for one spectroscopic stage.

    Returns ``(joint, sampler, nuts_kwargs)`` where ``joint`` is ``None`` when
    the flag is off and otherwise a mapping with the prior (``prior``), the
    inflation factor, the model keyword arguments (``model_kwargs``), the
    initial values (``init_params``) and the summary used by the writers.
    """
    enabled, inflation = _resolve_spectro_joint_geometry(flags)
    if not enabled:
        return None, spectro_sampler, nuts_kwargs
    if transit_engine != 'jaxoplanet':
        raise ValueError(
            "flags.spectro_joint_geometry=true is only supported with "
            "flags.transit_engine='jaxoplanet'; the harmonica engine keeps "
            "the fixed white-light geometry."
        )
    spectro_sampler, nuts_kwargs = _resolve_spectro_joint_geometry_sampler(
        spectro_sampler, True, nuts_kwargs, stage_name=stage_name
    )
    prior = _whitelight_geometry_prior(
        bestfit_params_wl_df, wl_geometry_handoff,
        param_method=param_method, prior_inflation=inflation,
    )
    sites = _joint_geometry_sites(param_method)
    joint = {
        'prior': prior,
        'prior_inflation': inflation,
        'sites': sites,
        'model_kwargs': {
            f'sigma_{name}': jnp.asarray(prior[f'sigma_{name}'], dtype=jnp.float64)
            for name in sites
        },
        'init_params': {
            name: jnp.asarray(prior[name], dtype=jnp.float64) for name in sites
        },
        'summary': {
            name: {
                'prior_center': np.asarray(prior[name], dtype=float),
                'prior_sigma': np.asarray(prior[f'sigma_{name}'], dtype=float),
                'whitelight_std': (
                    np.asarray(prior[f'sigma_{name}'], dtype=float) / inflation
                ),
            }
            for name in sites
        },
    }
    print(
        f"Joint geometry ({stage_name}): sampling shared "
        f"{', '.join(sites)} with Gaussian priors of "
        f"{inflation:g}x the white-light posterior std; "
        f"sampler={spectro_sampler}."
    )
    for name in sites:
        print(
            f"  {name}: centre={np.asarray(prior[name], dtype=float)}, "
            f"prior sigma={np.asarray(prior[f'sigma_{name}'], dtype=float)}"
        )
    return joint, spectro_sampler, nuts_kwargs


def _joint_geometry_window_duration(duration_base, joint, param_method):
    """Widen the static transit window so it covers the shared-geometry prior."""
    duration = np.atleast_1d(np.asarray(duration_base, dtype=float))
    if joint is None:
        return duration
    prior = joint['prior']
    margin = 5.0 * np.atleast_1d(np.asarray(prior['sigma_t0'], dtype=float))
    if param_method == 'duration':
        margin = margin + 2.5 * np.atleast_1d(
            np.asarray(prior['sigma_duration'], dtype=float)
        )
    return duration + 2.0 * margin


def _joint_geometry_posterior_medians(samples, joint):
    """Return the posterior medians of the shared geometry sites."""
    if joint is None:
        return {}
    return {
        name: jnp.nanmedian(jnp.asarray(samples[name]), axis=0)
        for name in joint['sites']
    }


def _run_low_resolution_stage_hook(
    A_RS_BASE,
    B_BASE,
    COSI_BASE,
    DEPTH_BASE,
    DURATION_BASE,
    HARMONICA_ECC,
    HARMONICA_ODD_HARMONICS,
    HARMONICA_OMEGA,
    PERIOD_FIXED,
    RORS_BASE,
    T0_BASE,
    _align_trend_to_time,
    _engine_spectro_kw,
    _explicit_ld_grid,
    bestfit_params_wl_df,
    build_transit_window_indices,
    cfg,
    config_path,
    chunk_mode,
    chunk_parallel_job_count,
    chunk_parallel_job_index,
    create_vectorized_model,
    data,
    detrending_type,
    exp_trend,
    explicit_ld,
    fixed_tau_spectro,
    flags,
    harmonica_lr_nuts_kwargs,
    harmonica_spectro_fit_jitter,
    harmonica_spectro_odd_frac_sigma,
    harmonica_spectro_parameterization,
    instrument,
    instrument_full_str,
    jaxoplanet_kernel,
    jaxoplanet_lr_nuts_kwargs,
    jump_trend,
    key_map_lr,
    key_mcmc_lr,
    ld_prior_mode,
    ld_profile,
    ld_uniform_basis,
    lowres_mcmc_kwargs,
    lr_artifact_stem,
    lr_bin_str,
    max_harmonic_order,
    n_planets,
    need_lowres_analysis,
    order,
    output_dir,
    param_method,
    planet_str,
    plots_mode,
    sld,
    spec_good_mask,
    spectro_fixed_timescale_trends,
    spectro_ld_parameterization,
    spectro_max_divergences,
    spectro_min_depth_ess,
    spectro_sampler,
    spectroscopic_sigma,
    spot_trend,
    spot_trend2,
    stellar_cfg,
    surface_config,
    transit_engine,
    transit_window_optimization,
    trend_inference,
    uses_surface_model,
    valid,
    vmap_chunk_size_lr,
    whitelight_geometry_estimator,
    wl_geometry_handoff,
    wl_mad_mask,
    wl_time_good,
):
    if need_lowres_analysis:
        # Compute the exact LD inputs before deciding whether a previous LR fit
        # is reusable. This binds cache validity to the coefficients actually
        # consumed by the likelihood, including changes to external LD grids.
        if explicit_ld is not None:
            U_mu_lr, U_sigma_lr = _explicit_ld_grid(data.wavelengths_lr), None
        elif ld_prior_mode == 'stellarprior':
            U_mu_lr, U_sigma_lr = get_or_build_power2_ld_prior(
                stellar_cfg,
                data.wavelengths_lr,
                data.wavelengths_err_lr,
                instrument,
                order=order if instrument == 'NIRISS/SOSS' else None,
                output_dir=output_dir,
                cache_label=lr_bin_str,
            )
        else:
            if instrument in [
                'NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM',
                'NIRSPEC/G140H', 'NIRSPEC/G235H', 'MIRI/LRS',
            ]:
                U_mu_lr = get_limb_darkening(
                    sld,
                    data.wavelengths_lr,
                    data.wavelengths_err_lr,
                    instrument,
                    ld_profile=ld_profile,
                    ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None),
                )
            elif instrument == 'NIRISS/SOSS':
                U_mu_lr = get_limb_darkening(
                    sld,
                    data.wavelengths_lr,
                    data.wavelengths_err_lr,
                    instrument,
                    order=order,
                    ld_profile=ld_profile,
                    ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None),
                )
            U_sigma_lr = None
            sing_model_c_lr = None
            if ld_prior_mode == 'sing':
                sing_model_c_lr = jnp.asarray(U_mu_lr)
                U_mu_lr, U_sigma_lr = build_sing_ld_prior(U_mu_lr, flags, stellar_cfg)

        lr_mask_path = (
            f"{output_dir}/{lr_artifact_stem}_spectroscopic_outlier_mask.npy"
        )
        lr_params_path = f"{output_dir}/{lr_artifact_stem}_bestfit_params.csv"
        lr_limb_csv_path = f"{output_dir}/{lr_artifact_stem}_limb_spectra.csv"
        lr_limb_samples_path = (
            f"{output_dir}/{lr_artifact_stem}_limb_posterior_samples.npz"
        )
        lr_manifest_path = (
            f"{output_dir}/{lr_artifact_stem}_artifacts.manifest.json"
        )
        lr_artifact_fingerprint = _science_artifact_fingerprint(
            "low_resolution",
            {
                "config": cfg,
                "time": data.time,
                "flux": data.flux_lr,
                "flux_err": data.flux_err_lr,
                "wavelength": data.wavelengths_lr,
                "wavelength_err": data.wavelengths_err_lr,
                "white_light_mask": wl_mad_mask,
                "white_light_fit": bestfit_params_wl_df.to_dict(orient="list"),
                "white_light_geometry_handoff_fingerprint": (
                    wl_geometry_handoff["artifact_fingerprint_sha256"]
                ),
                "transit_engine": transit_engine,
                "ld_profile": ld_profile,
                "ld_prior_mode": ld_prior_mode,
                "ld_coefficients": U_mu_lr,
                "ld_uncertainties": U_sigma_lr,
                "detrending_type": detrending_type,
                "harmonica_spectro_parameterization": (
                    harmonica_spectro_parameterization
                ),
                "spectroscopic_sigma": spectroscopic_sigma,
            },
        )
        required_lr_limb_products_exist = bool(
            transit_engine != 'harmonica'
            or not HARMONICA_ODD_HARMONICS
            or (
                os.path.exists(lr_limb_csv_path)
                and os.path.exists(lr_limb_samples_path)
            )
        )
        lr_artifact_set = ArtifactSet(
            stage="low_resolution",
            manifest_path=lr_manifest_path,
            fingerprint=lr_artifact_fingerprint,
            required_paths=(lr_mask_path, lr_params_path),
        )
        lr_cache_valid = lr_artifact_set.is_reusable(
            extra_condition=required_lr_limb_products_exist
        )
        if lr_cache_valid:
            print(
                f"Reusing low-res results from {lr_bin_str}; "
                "skipping low-res fit."
            )
            time_mask = np.load(lr_mask_path)
            valid = ~time_mask
        elif os.path.exists(lr_mask_path) or os.path.exists(lr_params_path):
            print(
                "Existing low-resolution products are incomplete, unverified, "
                "or do not match the current data/LD target; recomputing them."
            )

    if need_lowres_analysis and valid is None:
        if uses_surface_model:
            jax.clear_caches()
            print("Released white-light JAX executables before surface spectroscopy.")
        print(f"\n--- Running Low-Resolution Analysis (Binned to {lr_bin_str}) ---")
        joint_geometry_lr, spectro_sampler, jaxoplanet_lr_nuts_kwargs = (
            _prepare_joint_geometry(
                flags,
                stage_name='lowres',
                transit_engine=transit_engine,
                spectro_sampler=spectro_sampler,
                nuts_kwargs=jaxoplanet_lr_nuts_kwargs,
                bestfit_params_wl_df=bestfit_params_wl_df,
                wl_geometry_handoff=wl_geometry_handoff,
                param_method=param_method,
            )
        )
        time_lr = jnp.array(data.time[spec_good_mask])
        flux_lr = jnp.array(data.flux_lr[:, spec_good_mask])
        flux_err_lr = jnp.array(data.flux_err_lr[:, spec_good_mask])
        num_lcs_lr = int(data.flux_err_lr.shape[0])

        if 'gp' in detrending_type:
            gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            gp_trend_raw = gp_df['gp_trend'].values
            if len(gp_trend_raw) == len(wl_mad_mask):
                gp_trend_raw = gp_trend_raw[~wl_mad_mask]
            gp_trend = jnp.array(_align_trend_to_time(gp_trend_raw, wl_time_good, np.array(time_lr)))
        else:
            gp_trend = None
        detrend_type_multiwave = _spectro_detrend_type(
            detrending_type, spectro_fixed_timescale_trends
        )

        print(f"Low-res: {num_lcs_lr} light curves.")
        DEPTHS_BASE_LR = jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))

        init_params_lr = {
            "u": _clip_ld_initial_values(U_mu_lr, ld_profile, 'low-res'),
            "rors": jnp.tile(RORS_BASE, (num_lcs_lr, 1))
        }
        _seed_surface_spectroscopic_init(
            init_params_lr, surface_config, num_lcs_lr
        )
        if ld_profile == 'quadratic' and ld_prior_mode == 'uniform':
            init_params_lr.pop('u', None)
            init_params_lr.update(_quadratic_uniform_initial_sites(
                U_mu_lr, ld_uniform_basis
            ))
        if ld_prior_mode == 'sing':
            init_params_lr.pop('u', None)
            init_params_lr['limb_l'] = jnp.asarray(U_mu_lr)[:, 0]
            init_params_lr['limb_delta'] = jnp.asarray(U_mu_lr)[:, 1]
        if ld_profile == 'power2' and ld_prior_mode not in {'fixed', 'interpolated'}:
            init_params_lr.pop('u', None)
            init_params_lr.update(
                _power2_ld_initial_sites(U_mu_lr, spectro_ld_parameterization)
            )
        if transit_engine == 'harmonica':
            if harmonica_spectro_parameterization == 'delta_r':
                init_val = (
                    bestfit_params_wl_df['a1'].values[0]
                    if 'a1' in bestfit_params_wl_df.columns
                    else HARMONICA_INIT_ODD_COEFF
                )
                init_delta_r = jnp.atleast_1d(
                    jnp.asarray(_harmonica_coeff_array_to_delta_r(init_val))
                )
                init_params_lr['delta_r'] = jnp.tile(
                    init_delta_r[None, :],
                    (num_lcs_lr, 1),
                )
            elif harmonica_spectro_parameterization == 'half_area':
                init_val = (
                    bestfit_params_wl_df['a1'].values
                    if 'a1' in bestfit_params_wl_df.columns
                    else np.full_like(RORS_BASE, HARMONICA_INIT_ODD_COEFF)
                )
                init_area_radius, init_q = (
                    _harmonica_coefficients_to_half_area_init(
                        RORS_BASE, init_val
                    )
                )
                init_params_lr['rors'] = jnp.tile(
                    jnp.atleast_1d(jnp.asarray(init_area_radius))[None, :],
                    (num_lcs_lr, 1),
                )
                init_params_lr['q'] = jnp.tile(
                    jnp.atleast_1d(jnp.asarray(init_q))[None, :],
                    (num_lcs_lr, 1),
                )
            else:
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    init_val = bestfit_params_wl_df[harmonic_name].values[0] if harmonic_name in bestfit_params_wl_df.columns else HARMONICA_INIT_ODD_COEFF
                    init_frac = _harmonica_coeff_array_to_frac(init_val, RORS_BASE)
                    init_params_lr[_harmonica_frac_site(harmonic_name)] = jnp.tile(
                        jnp.asarray(init_frac)[None, :],
                        (num_lcs_lr, 1),
                    )
        if detrend_type_multiwave != 'none':
            init_params_lr['c'] = jnp.clip(
                jnp.full(num_lcs_lr, bestfit_params_wl_df['c'].values[0]),
                0.900001, 1.099999,
            )
            if 'v' in bestfit_params_wl_df.columns:
                 init_params_lr['v'] = jnp.clip(
                     jnp.full(num_lcs_lr, bestfit_params_wl_df['v'].values[0]),
                     -0.099999, 0.099999,
                 )

        if 'explinear' in detrend_type_multiwave:
            init_params_lr['A'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['A'].values[0])
            if 'explinear_spectroscopic' not in detrend_type_multiwave:
                init_params_lr['log_tau'] = jnp.full(
                    num_lcs_lr,
                    jnp.log(bestfit_params_wl_df['tau'].values[0]),
                )
        
        lr_trend_mode = (
            'gaussian_marginalized'
            if trend_inference == 'gaussian_marginalized'
            else 'free'
        )
        lr_ld_mode = ld_prior_mode
        
        lr_transit_window_indices = None
        if (
            transit_engine == 'jaxoplanet'
            and transit_window_optimization == 'auto'
            and param_method == 'duration'
        ):
            lr_transit_window_indices = build_transit_window_indices(
                np.asarray(time_lr),
                np.asarray(PERIOD_FIXED),
                np.asarray(T0_BASE),
                _joint_geometry_window_duration(
                    DURATION_BASE, joint_geometry_lr, param_method
                ),
            )
            print(
                "Jaxoplanet transit window (low-res): "
                f"{len(lr_transit_window_indices)}/{time_lr.size} active cadences."
            )

        lr_model_builder_kwargs = {
            'detrend_type': detrend_type_multiwave,
            'ld_mode': lr_ld_mode,
            'trend_mode': lr_trend_mode,
            'n_planets': n_planets,
            **(
                {
                    'transit_window_indices': lr_transit_window_indices,
                    'transit_grid_non_grazing': bool(
                        np.max(np.abs(np.asarray(B_BASE)))
                        < 1.0 - np.sqrt(0.5)
                    ),
                    'transit_grid_outer_contact_safe': bool(
                        np.max(np.abs(np.asarray(B_BASE)))
                        < 1.0 + np.sqrt(1.0e-5) - 1.0e-5
                    ),
                    'joint_geometry': joint_geometry_lr is not None,
                }
                if transit_engine == 'jaxoplanet' else {}
            ),
            **_engine_spectro_kw,
        }
        spot_basis_lr = None
        if lr_ld_mode == 'fixed':
            spot_basis_lr = _prepare_fixed_surface_basis(
                surface_config,
                time_lr,
                U_mu_lr,
                period=PERIOD_FIXED,
                t0=T0_BASE,
                a_rs=A_RS_BASE,
                b=B_BASE,
                rors=RORS_BASE,
                ecc=HARMONICA_ECC,
                omega=HARMONICA_OMEGA,
                ld_profile=ld_profile,
            )
            if spot_basis_lr is not None:
                print(
                    "Prepared exact fixed-geometry surface basis for "
                    "low-resolution inference."
                )
        lr_model_for_run = _build_spectroscopic_model(
            create_vectorized_model,
            **lr_model_builder_kwargs,
        )
        lr_adaptive_fallback_model = None
        lr_adaptive_fallback_init = None
        if (
            transit_engine == 'jaxoplanet'
            and ld_profile == 'power2'
            and lr_ld_mode not in {'fixed', 'interpolated'}
            and spectro_ld_parameterization != 'coefficients'
        ):
            coefficient_builder_kwargs = dict(lr_model_builder_kwargs)
            coefficient_builder_kwargs['ld_parameterization'] = 'coefficients'
            lr_adaptive_fallback_model = _build_spectroscopic_model(
                create_vectorized_model, **coefficient_builder_kwargs
            )
            lr_adaptive_fallback_init = dict(init_params_lr)
            lr_adaptive_fallback_init.pop('ld_decorrelated', None)
            lr_adaptive_fallback_init.update(
                _power2_ld_initial_sites(U_mu_lr, 'coefficients')
            )

        model_run_args_lr = {
            'mu_duration': DURATION_BASE,
            'mu_t0': T0_BASE,
            'mu_b': B_BASE,
            'mu_depths': DEPTHS_BASE_LR,
            'PERIOD': PERIOD_FIXED,
        }
        if transit_engine == 'jaxoplanet':
            model_run_args_lr['precomputed_yerr_per_lc'] = jnp.nanmedian(
                flux_err_lr, axis=1
            )
            if spot_basis_lr is not None:
                model_run_args_lr['surface_basis_data'] = spot_basis_lr
        if transit_engine == 'harmonica':
            model_run_args_lr['mu_cos_i'] = COSI_BASE
            model_run_args_lr['harmonica_a_rs'] = A_RS_BASE
            model_run_args_lr['harmonica_ecc'] = HARMONICA_ECC
            model_run_args_lr['harmonica_omega'] = HARMONICA_OMEGA
        elif param_method == 'a_rs':
            model_run_args_lr['mu_a_rs'] = A_RS_BASE
            model_run_args_lr['mu_ecc'] = HARMONICA_ECC
            model_run_args_lr['mu_omega'] = HARMONICA_OMEGA
        if joint_geometry_lr is not None:
            model_run_args_lr.update(joint_geometry_lr['model_kwargs'])
            init_params_lr.update(joint_geometry_lr['init_params'])

        if lr_ld_mode == 'fixed':
            model_run_args_lr['ld_fixed'] = U_mu_lr
        elif lr_ld_mode in {'gaussian', 'stellarprior', 'sing'}:
            model_run_args_lr['mu_u_ld'] = U_mu_lr
            if ((ld_profile == 'power2' and lr_ld_mode == 'stellarprior') or lr_ld_mode == 'sing') and U_sigma_lr is not None:
                model_run_args_lr['sigma_u_ld'] = U_sigma_lr
        elif lr_ld_mode == 'uniform':
            pass
        else:
            raise ValueError(f"Unknown ld_prior mode: {lr_ld_mode}")
        
        if 'gp_spectroscopic' in detrend_type_multiwave:
            model_run_args_lr['gp_trend'] = gp_trend
            init_params_lr['A_gp'] = jnp.ones(num_lcs_lr)
        if '2spot_spectroscopic' in detrend_type_multiwave:
            if spot_trend is None or spot_trend2 is None:
                raise ValueError("2spot_spectroscopic requires WL spot and spot2 trends.")
            if len(spot_trend) != len(time_lr):
                spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_lr))
            if len(spot_trend2) != len(time_lr):
                spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_lr))
            model_run_args_lr['spot_trend'] = spot_trend
            model_run_args_lr['spot_trend2'] = spot_trend2
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
            init_params_lr['A_spot2'] = jnp.ones(num_lcs_lr)
        elif _has_single_spot_spectroscopic(detrend_type_multiwave):
            if spot_trend is None:
                raise ValueError(f"{detrend_type_multiwave} requires WL spot to build spot_trend.")
            if spot_trend is not None and len(spot_trend) != len(time_lr):
                spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_lr))
            model_run_args_lr['spot_trend'] = spot_trend
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
        if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            if jump_trend is None:
                raise ValueError("linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.")
            if jump_trend is not None and len(jump_trend) != len(time_lr):
                jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_lr))
            model_run_args_lr['jump_trend'] = jump_trend
            init_params_lr['A_jump'] = jnp.ones(num_lcs_lr)
        if 'explinear_spectroscopic' in detrend_type_multiwave:
            exp_trend_lr = _align_trend_to_time(
                exp_trend, wl_time_good, np.asarray(time_lr)
            )
            model_run_args_lr['exp_trend'] = exp_trend_lr
            model_run_args_lr['fixed_tau'] = fixed_tau_spectro

        if (
            transit_engine == 'jaxoplanet'
            and _engine_spectro_kw.get('cadence_reduction') == 'auto'
            and int(time_lr.size) > 5000
            and lr_trend_mode == 'free'
            and lr_transit_window_indices is not None
            and linear_spectro_trend_coefficient_names(
                detrend_type_multiwave
            ) is not None
        ):
            trend_names_lr, trend_design_lr = (
                build_linear_spectro_trend_design(
                    detrend_type_multiwave,
                    time_lr,
                    exp_trend=model_run_args_lr.get('exp_trend'),
                    spot_trend=model_run_args_lr.get('spot_trend'),
                    spot_trend2=model_run_args_lr.get('spot_trend2'),
                    jump_trend=model_run_args_lr.get('jump_trend'),
                )
            )
            reference_beta_lr = np.column_stack([
                np.asarray(init_params_lr.get(
                    name, np.zeros(num_lcs_lr, dtype=np.float64)
                ))
                for name in trend_names_lr
            ])
            model_run_args_lr['trend_design'] = trend_design_lr
            model_run_args_lr.update(build_linear_oot_statistics(
                time_lr,
                flux_lr,
                flux_err_lr,
                lr_transit_window_indices,
                trend_design_lr,
                reference_beta_lr,
            ))

        trend_names_lr = None
        if lr_trend_mode == 'gaussian_marginalized':
            (
                model_run_args_lr['trend_prior_mean'],
                model_run_args_lr['trend_prior_scale'],
                trend_names_lr,
            ) = _build_gaussian_trend_prior(
                detrend_type_multiwave,
                num_lcs_lr,
                init_params_lr,
                flags,
            )
            # These coefficients are integrated out exactly.  Keep only
            # genuinely nonlinear trend sites (for example log_tau).
            for trend_name in trend_names_lr:
                init_params_lr.pop(trend_name, None)

        # Sing et al. step 1: calibrate a single gray stellar offset from a
        # separate free-quadratic fit to the existing coarse channel set.
        if ld_prior_mode == 'sing':
            gray_fingerprint_inputs = {
                'kind': 'sing_gray_offset_v3_uplus_uminus_ess_weighted',
                'time': np.asarray(time_lr),
                'flux': np.asarray(flux_lr),
                'flux_err': np.asarray(flux_err_lr),
                'wavelength': np.asarray(data.wavelengths_lr),
                'wavelength_err': np.asarray(data.wavelengths_err_lr),
                'model_c': np.asarray(sing_model_c_lr),
                'mu_min': float(stellar_cfg.get('ld_mu_min', 0.2)),
                'sampler': 'independent_nuts_laplace_metric_ta099_depth10',
                'geometry_handoff': wl_geometry_handoff['artifact_fingerprint_sha256'],
            }
            gray_digest = _science_artifact_fingerprint(
                'sing_gray_offset', gray_fingerprint_inputs
            )
            gray_artifact_path = os.path.join(
                output_dir,
                f"{instrument_full_str}_{lr_bin_str}_sing_gray_offset_{gray_digest[:12]}.json",
            )
            fitted_offsets = None
            if os.path.exists(gray_artifact_path):
                try:
                    with open(gray_artifact_path) as handle:
                        cached_gray = json.load(handle)
                    if cached_gray.get('fingerprint_sha256') == hashlib.sha256(
                        json.dumps(gray_fingerprint_inputs, sort_keys=True, default=str).encode()
                    ).hexdigest():
                        fitted_offsets = cached_gray
                        print(f"[Sing LD] reusing fitted gray offsets from {gray_artifact_path}", flush=True)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    fitted_offsets = None
            if fitted_offsets is None:
                print("[Sing LD] running broad independent (u_plus, u_minus) coarse calibration pass", flush=True)
                gray_builder_kwargs = dict(lr_model_builder_kwargs)
                gray_builder_kwargs['ld_mode'] = 'uniform'
                # The gray calibration pass is a fixed-geometry lane fit.
                gray_builder_kwargs.pop('joint_geometry', None)
                gray_model = _build_spectroscopic_model(
                    create_vectorized_model, **gray_builder_kwargs
                )
                gray_init = dict(init_params_lr)
                gray_init.pop('limb_l', None)
                gray_init.pop('limb_delta', None)
                gray_init.pop('limb_u_plus', None)
                gray_init.pop('limb_u_minus', None)
                gray_init.pop('u', None)
                for shared_name in (
                    joint_geometry_lr['sites'] if joint_geometry_lr else ()
                ):
                    gray_init.pop(shared_name, None)
                gray_l_init, gray_delta_init = quadratic_to_sing(
                    np.asarray(sing_model_c_lr)[:, 0],
                    np.asarray(sing_model_c_lr)[:, 1],
                )
                gray_u_plus_init = 1.0 - np.asarray(gray_l_init)
                gray_u_minus_init = gray_u_plus_init - 8.0 * np.asarray(gray_delta_init)
                gray_init['ld_uplus_uminus'] = jnp.column_stack((
                    jnp.asarray(gray_u_plus_init),
                    jnp.asarray(gray_u_minus_init),
                ))
                gray_args = dict(model_run_args_lr)
                gray_args.pop('mu_u_ld', None)
                gray_args.pop('sigma_u_ld', None)
                for shared_kwarg in (
                    joint_geometry_lr['model_kwargs'] if joint_geometry_lr else ()
                ):
                    gray_args.pop(shared_kwarg, None)
                gray_prefix = _harmonica_checkpoint_prefix(
                    f"{instrument_full_str}_{lr_bin_str}_sing_gray_free_{gray_digest[:12]}",
                    transit_engine,
                    harmonica_spectro_parameterization,
                )
                gray_mcmc = {
                    'num_warmup': 1000,
                    'num_samples': 1000,
                }
                gray_nuts_kwargs = dict(jaxoplanet_lr_nuts_kwargs)
                gray_nuts_kwargs.update(
                    mass_matrix='laplace',
                    laplace_hessian_method='finite_difference',
                    laplace_warmup=1000,
                    laplace_target_accept=0.99,
                    laplace_max_tree_depth=10,
                    target_accept_prob=0.99,
                    max_tree_depth=10,
                )
                try:
                    gray_samples = _run_sampling_stage(
                        gray_model,
                        jax.random.fold_in(key_mcmc_lr, 91027),
                        time_lr, flux_err_lr, flux_lr, gray_init,
                        nuts_kwargs=gray_nuts_kwargs,
                        mcmc_kwargs=gray_mcmc,
                        use_chunked=True,
                        chunk_size=(
                            min(40, int(num_lcs_lr))
                            if vmap_chunk_size_lr == 'auto'
                            else (vmap_chunk_size_lr or num_lcs_lr)
                        ),
                        chunk_mode='serial',
                        output_dir=output_dir,
                        checkpoint_prefix=gray_prefix,
                        sampler_backend='independent_nuts',
                        channel_varying_kwargs=(JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS if transit_engine == 'jaxoplanet' else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS),
                        checkpoint_signature={
                            'stage': 'sing_gray_offset_calibration',
                            'ld_profile': 'quadratic',
                            'ld_mode': 'uniform',
                            'fingerprint': gray_digest,
                        },
                        **gray_args,
                    )
                    gray_diag_paths = sorted(glob.glob(os.path.join(
                        output_dir, 'chunks', f"{gray_prefix}*_chunk_*.diagnostics.json"
                    )))
                    gray_ok, gray_diagnostics = _assess_sing_gray_fit(
                        gray_samples,
                        gray_diag_paths,
                        min_ess=100.0,
                    )
                    if not gray_ok:
                        raise RuntimeError(f"gray calibration diagnostics failed: {gray_diagnostics}")
                    if gray_diagnostics['channels_below_ess_warning']:
                        print(
                            "[Sing LD] WARNING: calibration LD bulk ESS below "
                            f"{gray_diagnostics['required_min_bulk_ess']:.0f} in channels "
                            f"{gray_diagnostics['channels_below_ess_warning']}; "
                            "retaining the fit with ESS-inflated pooling weights.",
                            flush=True,
                        )
                    fitted_c = np.column_stack((
                        np.nanmedian(np.asarray(gray_samples['c1'], dtype=float), axis=0),
                        np.nanmedian(np.asarray(gray_samples['c2'], dtype=float), axis=0),
                    ))
                    fitted_c_sigma = np.column_stack((
                        np.nanstd(np.asarray(gray_samples['c1'], dtype=float), axis=0, ddof=1),
                        np.nanstd(np.asarray(gray_samples['c2'], dtype=float), axis=0, ddof=1),
                    ))
                    gray_ess = np.column_stack((
                        gray_diagnostics['bulk_ess_l_per_channel'],
                        gray_diagnostics['bulk_ess_delta_per_channel'],
                    ))
                    fitted_offsets = estimate_gray_offset(
                        fitted_c, np.asarray(sing_model_c_lr), fitted_c_sigma,
                        ess=gray_ess,
                        n_draws=np.asarray(gray_samples['c1']).shape[0],
                    )
                    fitted_offsets['diagnostics'] = gray_diagnostics
                    fitted_offsets['tabulated_l'] = SING_TABULATED_OFFSET['l']
                    fitted_offsets['tabulated_delta'] = SING_TABULATED_OFFSET['delta']
                    fitted_offsets['tabulated_l_sigma'] = SING_TABULATED_SCATTER['l']
                    fitted_offsets['tabulated_delta_sigma'] = SING_TABULATED_SCATTER['delta']
                    write_offset_artifact(
                        gray_artifact_path, fitted_offsets, gray_fingerprint_inputs
                    )
                except Exception as error:
                    print(
                        f"[Sing LD] WARNING: free gray calibration unavailable or failed ({error}); "
                        "falling back to tabulated Stagger offsets.",
                        flush=True,
                    )
                    fitted_offsets = None
            if fitted_offsets is not None:
                print(
                    "[Sing LD] gray offsets fitted vs tabulated: "
                    f"delta_l={float(fitted_offsets['l']):+.6f} +/- {float(fitted_offsets['l_sigma']):.6f} "
                    f"vs {SING_TABULATED_OFFSET['l']:+.6f}; "
                    f"delta_delta={float(fitted_offsets['delta']):+.6f} +/- {float(fitted_offsets['delta_sigma']):.6f} "
                    f"vs {SING_TABULATED_OFFSET['delta']:+.6f}",
                    flush=True,
                )
                U_mu_lr, U_sigma_lr = build_sing_ld_prior(
                    sing_model_c_lr, flags, stellar_cfg,
                    offsets_override=fitted_offsets,
                )
                model_run_args_lr['mu_u_ld'] = U_mu_lr
                model_run_args_lr['sigma_u_ld'] = U_sigma_lr
                init_params_lr['limb_l'] = jnp.asarray(U_mu_lr)[:, 0]
                init_params_lr['limb_delta'] = jnp.asarray(U_mu_lr)[:, 1]

        lr_stage_sampler_kwargs = (
            harmonica_lr_nuts_kwargs
            if transit_engine == 'harmonica'
            else jaxoplanet_lr_nuts_kwargs
        )
        vmap_chunk_size_lr = _resolve_spectro_joint_geometry_chunk_size(
            vmap_chunk_size_lr, num_lcs_lr, joint_geometry_lr is not None
        )
        vmap_chunk_size_lr = _resolve_stage_vmap_width(
            flags,
            'lowres',
            vmap_chunk_size_lr,
            sampler_backend=spectro_sampler,
            trend_inference=(
                'gaussian_marginalized'
                if lr_trend_mode == 'gaussian_marginalized'
                else 'sampled_uniform'
            ),
            num_cadences=int(time_lr.size),
            active_transit_cadences=(
                int(len(lr_transit_window_indices))
                if lr_transit_window_indices is not None
                else int(time_lr.size)
            ),
            mcmc_kwargs=lowres_mcmc_kwargs,
            sampler_kwargs=lr_stage_sampler_kwargs,
            transit_engine=transit_engine,
            ld_profile=ld_profile,
            ld_mode=lr_ld_mode,
            detrend_type=detrend_type_multiwave,
            param_method=param_method,
            n_planets=n_planets,
            transit_window=transit_window_optimization,
            transit_grid_non_grazing=lr_model_builder_kwargs.get(
                'transit_grid_non_grazing', False
            ),
            transit_grid_outer_contact_safe=lr_model_builder_kwargs.get(
                'transit_grid_outer_contact_safe', False
            ),
        )

        samples_lr = _run_sampling_stage(
            lr_model_for_run,
            key_mcmc_lr,
            time_lr,
            flux_err_lr,
            flux_lr,
            init_params_lr,
            nuts_kwargs=lr_stage_sampler_kwargs,
            mcmc_kwargs=lowres_mcmc_kwargs,
            use_chunked=(
                vmap_chunk_size_lr is not None
                or spectro_sampler in {'independent_nuts', 'independent_hmc'}
            ),
            chunk_size=vmap_chunk_size_lr,
            chunk_mode=chunk_mode,
            parallel_job_count=chunk_parallel_job_count,
            parallel_job_index=chunk_parallel_job_index,
            output_dir=output_dir,
            checkpoint_prefix=_harmonica_checkpoint_prefix(
                f"{instrument_full_str}_{lr_bin_str}",
                transit_engine,
                harmonica_spectro_parameterization,
            ),
            checkpoint_signature={
                'stage': 'low_resolution',
                'spectro_joint_geometry': joint_geometry_lr is not None,
                'spectro_joint_geometry_prior_inflation': (
                    None if joint_geometry_lr is None
                    else joint_geometry_lr['prior_inflation']
                ),
                'transit_engine': transit_engine,
                'ld_profile': ld_profile,
                'ld_mode': lr_ld_mode,
                'trend_inference': trend_inference,
                'detrend_type': detrend_type_multiwave,
                'param_method': param_method,
                'whitelight_geometry_estimator': whitelight_geometry_estimator,
                'whitelight_geometry_handoff_fingerprint': (
                    wl_geometry_handoff['artifact_fingerprint_sha256']
                ),
                'jaxoplanet_kernel': jaxoplanet_kernel,
                'transit_window_optimization': transit_window_optimization,
                'spectro_cadence_reduction': _engine_spectro_kw.get(
                    'cadence_reduction', 'off'
                ),
                'spectro_transit_grid': _engine_spectro_kw.get(
                    'transit_grid', 'off'
                ),
                'spectro_transit_grid_nodes': _engine_spectro_kw.get(
                    'transit_grid_nodes'
                ),
                'surface_config': surface_config,
                'harmonica_max_order': max_harmonic_order,
                'harmonica_spectro_parameterization': harmonica_spectro_parameterization,
                'harmonica_spectro_fit_jitter': harmonica_spectro_fit_jitter,
                'harmonica_spectro_odd_frac_sigma': harmonica_spectro_odd_frac_sigma,
            },
            sampler_backend=spectro_sampler,
            channel_varying_kwargs=(
                JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
                if transit_engine == 'jaxoplanet'
                else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS
            ),
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
            dump_metadata={
                'config_path': os.path.abspath(config_path),
                'stage_kind': 'low_resolution',
                'stage_label': (
                    f"{instrument_full_str}_{lr_bin_str}_low_resolution"
                ),
                'instrument': instrument,
                'instrument_label': instrument_full_str,
                'output_dir': os.path.abspath(output_dir),
                'wavelength': np.asarray(data.wavelengths_lr),
                'wavelength_err': np.asarray(data.wavelengths_err_lr),
                'active_window_cadences': (
                    int(len(lr_transit_window_indices))
                    if lr_transit_window_indices is not None
                    else int(time_lr.size)
                ),
                'transit_engine': transit_engine,
            },
            adaptive_fallback_model=lr_adaptive_fallback_model,
            adaptive_fallback_init_params=lr_adaptive_fallback_init,
            **model_run_args_lr,
        )
        if samples_lr is None:
            print("\nLow-resolution chunk job complete. Exiting before final aggregation.")
            return
        if trend_names_lr is not None:
            samples_lr = materialize_marginalized_trend_samples(
                samples_lr,
                trend_names_lr,
                key_map_lr,
            )

        if ld_profile == 'quadratic' and 'u1' in samples_lr and 'u2' in samples_lr:
            ld_u_lr = np.stack(
                (np.array(samples_lr['u1']), np.array(samples_lr['u2'])),
                axis=-1,
            )
        elif 'u' in samples_lr:
            ld_u_lr = np.array(samples_lr["u"])
        elif ld_profile == 'power2' and 'c1' in samples_lr:
            c1_med_lr = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med_lr = jnp.nanmedian(samples_lr['c2'], axis=0)
            ld_u_lr = jax.vmap(compute_u_from_c)(c1_med_lr, c2_med_lr)
            ld_u_lr = np.array(ld_u_lr)

        if detrend_type_multiwave != 'none':
            trend_c_lr = np.array(samples_lr["c"])
            if 'v' in samples_lr: trend_v_lr = np.array(samples_lr["v"])
        if 'explinear' in detrend_type_multiwave:
            trend_A_lr = np.array(samples_lr["A"])
            trend_tau_lr = np.array(samples_lr["tau"])

        map_params_lr = {
            "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
            "rors": jnp.nanmedian(
                _harmonica_model_radius_samples(
                    samples_lr, harmonica_spectro_parameterization
                )
                if transit_engine == 'harmonica'
                else samples_lr["rors"],
                axis=0,
            ),
            "period": PERIOD_FIXED,
        }
        
        if ld_profile == 'power2':
            c1_med = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med = jnp.nanmedian(samples_lr['c2'], axis=0)
            map_params_lr['c1'] = c1_med
            map_params_lr['c2'] = c2_med
            map_params_lr['u'] = jax.vmap(compute_u_from_c)(c1_med, c2_med)
            if transit_engine == 'harmonica':
                map_params_lr['c_ld'] = c1_med
                map_params_lr['alpha_ld'] = c2_med
        else:
            quadratic_u_med = _posterior_quadratic_ld_median(
                samples_lr, U_mu_lr
            )
            map_params_lr['u'] = quadratic_u_med
            if transit_engine == 'harmonica':
                map_params_lr['u1_ld'] = quadratic_u_med[..., 0]
                map_params_lr['u2_ld'] = quadratic_u_med[..., 1]

        if transit_engine == 'harmonica':
            map_params_lr = _augment_harmonica_params(
                map_params_lr,
                a_rs=A_RS_BASE,
                ecc=HARMONICA_ECC,
                omega=HARMONICA_OMEGA,
                a1=jnp.nanmedian(samples_lr["a1"], axis=0) if "a1" in samples_lr else None,
                a3=jnp.nanmedian(samples_lr["a3"], axis=0) if "a3" in samples_lr else None,
                a5=jnp.nanmedian(samples_lr["a5"], axis=0) if "a5" in samples_lr else None,
            )
        elif param_method == 'a_rs':
            map_params_lr.update({
                "a_rs": A_RS_BASE,
                "ecc": HARMONICA_ECC,
                "omega": HARMONICA_OMEGA,
            })
        map_params_lr.update(
            _joint_geometry_posterior_medians(samples_lr, joint_geometry_lr)
        )

        map_params_lr.update({k: jnp.nanmedian(samples_lr[k], axis=0) for k in TREND_PARAMS if k in samples_lr})
        map_params_lr.update({
            k: jnp.nanmedian(samples_lr[k], axis=0)
            for k in SURFACE_PARAMS if k in samples_lr
        })
        if transit_engine == 'jaxoplanet':
            map_params_lr = _attach_surface_eval_metadata(
                map_params_lr, surface_config
            )
            map_params_lr = _attach_jaxoplanet_eval_metadata(
                map_params_lr,
                time_lr,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
                transit_window_optimization=transit_window_optimization,
            )

        selected_kernel = resolve_detrend_kernel(detrend_type_multiwave)
        in_axes_map = {'rors': 0}
        in_axes_map.update({
            name: 0 for name in ('u', 'c1', 'c2') if name in map_params_lr
        })
        if transit_engine == 'harmonica':
            if ld_profile == 'power2':
                in_axes_map.update({'c_ld': 0, 'alpha_ld': 0})
            else:
                in_axes_map.update({'u1_ld': 0, 'u2_ld': 0})
            in_axes_map.update({name: 0 for name in HARMONICA_ODD_HARMONICS if name in map_params_lr})
        in_axes_map.update({k: 0 for k in TREND_PARAMS if k in map_params_lr})
        in_axes_map.update({k: 0 for k in SURFACE_PARAMS if k in map_params_lr})
        
        final_in_axes = {k: in_axes_map.get(k, None) for k in map_params_lr.keys()}

        num_lcs_lr = flux_lr.shape[0]
        model_all = evaluate_channels_sequentially(
            map_params_lr,
            final_in_axes,
            time_lr,
            num_lcs_lr,
            detrend_type_multiwave,
            selected_kernel,
            transit_engine,
            surface_config,
            jaxoplanet_kernel,
            ld_profile,
            gp_trend,
            spot_trend,
            spot_trend2,
            jump_trend,
            locals().get('exp_trend_lr'),
            _attach_surface_eval_metadata,
            _has_single_spot_spectroscopic,
        )

        residuals = flux_lr - model_all
        _lr_dt_sec = float(np.nanmedian(np.diff(np.array(time_lr)))) * 86400.0
        if plots_mode == 'full':
            plot_noise_binning_robust(
                residuals, _lr_dt_sec,
                f"{output_dir}/25_{lr_artifact_stem}_noisebin.png",
                title=f"Low-Res Noise Binning ({lr_bin_str})",
            )
        save_noise_binning_data(
            residuals,
            f"{output_dir}/25_{lr_artifact_stem}_noisebin.csv",
        )

        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)
        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas
        time_mask = np.any(point_mask, axis=0)
        valid = ~time_mask
        _atomic_save_npy(lr_mask_path, time_mask)
        gp_trend_lr = gp_trend
        spot_trend_lr = spot_trend
        spot_trend2_lr = spot_trend2
        jump_trend_lr = jump_trend
        exp_trend_lr_save = (
            exp_trend_lr if 'explinear_spectroscopic' in detrend_type_multiwave else None
        )
        time_lr = time_lr[valid]
        flux_lr = flux_lr[:, valid]
        flux_err_lr = flux_err_lr[:, valid]
        if gp_trend_lr is not None: gp_trend_lr = gp_trend_lr[valid]
        if spot_trend_lr is not None: spot_trend_lr = spot_trend_lr[valid]
        if spot_trend2_lr is not None: spot_trend2_lr = spot_trend2_lr[valid]
        if jump_trend_lr is not None: jump_trend_lr = jump_trend_lr[valid]
        if exp_trend_lr_save is not None: exp_trend_lr_save = exp_trend_lr_save[valid]
        if transit_engine == 'jaxoplanet':
            map_params_lr = _attach_jaxoplanet_eval_metadata(
                map_params_lr,
                time_lr,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
                transit_window_optimization=transit_window_optimization,
            )
        plot_params_lr = dict(map_params_lr)
        if spot_basis_lr is not None:
            plot_params_lr['_surface_basis'] = _surface_basis_on_time_mask(
                spot_basis_lr, valid
            )
            plot_params_lr['_surface_basis_time'] = jnp.asarray(time_lr)
        
        median_total_error_lr = np.nanmedian(samples_lr['total_error'], axis=0)
        if plots_mode == 'full':
            plot_wavelength_offset_summary(time_lr, flux_lr, median_total_error_lr, data.wavelengths_lr,
                                         plot_params_lr, {
                                             "period": PERIOD_FIXED,
                                             "transit_engine": transit_engine,
                                             "param_method": param_method,
                                         },
                                         f"{output_dir}/22_{lr_artifact_stem}_summary.png",
                                         detrend_type=detrend_type_multiwave, gp_trend=gp_trend_lr, spot_trend=spot_trend_lr, spot_trend2=spot_trend2_lr, jump_trend=jump_trend_lr, exp_trend=exp_trend_lr_save)

        wl_lr = np.array(data.wavelengths_lr)

        if surface_config['model'] != 'eclipse':
            plot_transmission_spectrum(
                wl_lr, samples_lr["rors"],
                f"{output_dir}/24_{lr_artifact_stem}_spectrum",
            )
            save_results(
                wl_lr, data.wavelengths_err_lr, samples_lr,
                f"{output_dir}/{lr_artifact_stem}.csv",
            )
        save_surface_results(
            wl_lr, data.wavelengths_err_lr, samples_lr,
            f"{output_dir}/{lr_artifact_stem}.csv",
        )
        save_detailed_fit_results(time_lr, flux_lr, flux_err_lr, data.wavelengths_lr, data.wavelengths_err_lr, samples_lr, map_params_lr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{lr_artifact_stem}", median_total_error_lr, gp_trend=gp_trend_lr, spot_trend=spot_trend_lr, jump_trend=jump_trend_lr, joint_geometry=(None if joint_geometry_lr is None else joint_geometry_lr['summary']))
        if transit_engine == 'harmonica' and _has_harmonica_odd_samples(samples_lr):
            save_harmonica_limb_products(
                wavelengths=wl_lr,
                wavelength_err=data.wavelengths_err_lr,
                rors_samples=_harmonica_model_radius_samples(
                    samples_lr, harmonica_spectro_parameterization
                ),
                harmonic_samples=_harmonica_sample_payload(samples_lr),
                csv_path=lr_limb_csv_path,
                limb_spectrum_path=f"{output_dir}/26_{lr_artifact_stem}_limb_spectra.png",
                transmission_strings_path=f"{output_dir}/27_{lr_artifact_stem}_transmission_strings.png",
                posterior_strings_path=f"{output_dir}/28_{lr_artifact_stem}_transmission_string_posterior.png",
                posterior_samples_path=lr_limb_samples_path,
                title_prefix=f"{planet_str} - {lr_bin_str}",
            )
        lr_artifact_set.write_manifest()
    return (valid, spot_trend, spot_trend2, jump_trend,)


def _run_high_resolution_stage_hook(
    A_RS_BASE,
    B_BASE,
    COSI_BASE,
    DEPTH_BASE,
    DURATION_BASE,
    HARMONICA_ECC,
    HARMONICA_ODD_HARMONICS,
    HARMONICA_OMEGA,
    PERIOD_FIXED,
    RORS_BASE,
    T0_BASE,
    _align_trend_to_time,
    _engine_spectro_kw,
    _explicit_ld_grid,
    bestfit_params_wl_df,
    build_transit_window_indices,
    config_path,
    chunk_mode,
    chunk_parallel_job_count,
    chunk_parallel_job_index,
    create_vectorized_model,
    data,
    detrending_type,
    exp_trend,
    explicit_ld,
    fixed_tau_spectro,
    flags,
    harmonica_hr_nuts_kwargs,
    harmonica_spectro_fit_jitter,
    harmonica_spectro_odd_frac_sigma,
    harmonica_spectro_parameterization,
    highres_mcmc_kwargs,
    hr_artifact_stem,
    hr_bin_str,
    instrument,
    instrument_full_str,
    jaxoplanet_hr_nuts_kwargs,
    jaxoplanet_kernel,
    jump_trend,
    key_map_hr,
    key_mcmc_hr,
    ld_prior_mode,
    ld_profile,
    ld_uniform_basis,
    max_harmonic_order,
    n_planets,
    order,
    output_dir,
    param_method,
    planet_str,
    plots_mode,
    sld,
    spec_good_mask,
    spectro_fixed_timescale_trends,
    spectro_ld_parameterization,
    spectro_max_divergences,
    spectro_min_depth_ess,
    spectro_sampler,
    spot_trend,
    spot_trend2,
    stellar_cfg,
    surface_config,
    transit_engine,
    transit_window_optimization,
    trend_inference,
    uses_surface_model,
    valid,
    vmap_chunk_size_hr,
    whitelight_geometry_estimator,
    wl_geometry_handoff,
    wl_mad_mask,
    wl_time_good,
):
    if uses_surface_model:
        jax.clear_caches()
        print("Released low-resolution JAX executables before high-resolution surface inference.")
    print(f"\n--- Running High-Resolution Analysis (Binned to {hr_bin_str}) ---")
    joint_geometry_hr, spectro_sampler, jaxoplanet_hr_nuts_kwargs = (
        _prepare_joint_geometry(
            flags,
            stage_name='highres',
            transit_engine=transit_engine,
            spectro_sampler=spectro_sampler,
            nuts_kwargs=jaxoplanet_hr_nuts_kwargs,
            bestfit_params_wl_df=bestfit_params_wl_df,
            wl_geometry_handoff=wl_geometry_handoff,
            param_method=param_method,
        )
    )
    time_hr = jnp.array(data.time[spec_good_mask])
    flux_hr = jnp.array(data.flux_hr[:, spec_good_mask])
    flux_err_hr = jnp.array(data.flux_err_hr[:, spec_good_mask])

    if 'gp' in detrending_type:
        gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
        gp_trend_raw = gp_df['gp_trend'].values
        if len(gp_trend_raw) == len(wl_mad_mask):
            gp_trend_raw = gp_trend_raw[~wl_mad_mask]
        gp_trend = jnp.array(_align_trend_to_time(gp_trend_raw, wl_time_good, np.array(time_hr)))
    else:
        gp_trend = None
    detrend_type_multiwave = _spectro_detrend_type(
        detrending_type, spectro_fixed_timescale_trends
    )

    if valid is not None:
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        if spot_trend2 is not None and len(spot_trend2) != len(time_hr):
            spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_hr))
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_hr))
        if gp_trend is not None and len(gp_trend) != len(time_hr):
            gp_trend = _align_trend_to_time(gp_trend, wl_time_good, np.array(time_hr))

        time_hr = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr = flux_err_hr[:, valid]
        if gp_trend is not None: gp_trend = gp_trend[valid]
        if spot_trend is not None: spot_trend = spot_trend[valid]
        if spot_trend2 is not None: spot_trend2 = spot_trend2[valid]
        if jump_trend is not None: jump_trend = jump_trend[valid]

    num_lcs_hr = flux_err_hr.shape[0]
    DEPTHS_BASE_HR = jnp.tile(DEPTH_BASE, (num_lcs_hr, 1))
    
    hr_ld_mode = ld_prior_mode
    if trend_inference == 'gaussian_marginalized':
        hr_trend_mode = 'gaussian_marginalized'
    else:
        hr_trend_mode = 'free'

    model_run_args_hr = {}
    wl_hr = np.array(data.wavelengths_hr)

    if hr_ld_mode in {'fixed', 'gaussian', 'stellarprior', 'sing', 'uniform'}:
        if explicit_ld is not None:
            U_mu_hr_init = _explicit_ld_grid(wl_hr)
            U_sigma_hr_init = None
        elif ld_prior_mode == 'stellarprior':
            U_mu_hr_init, U_sigma_hr_init = get_or_build_power2_ld_prior(
                stellar_cfg, wl_hr, data.wavelengths_err_hr, instrument,
                order=order if instrument == 'NIRISS/SOSS' else None,
                output_dir=output_dir,
                cache_label=hr_bin_str,
            )
        else:
            if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H', 'MIRI/LRS']:
                U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
            elif instrument == 'NIRISS/SOSS':
                U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
            U_sigma_hr_init = None
            if ld_prior_mode == 'sing':
                U_mu_hr_init, U_sigma_hr_init = build_sing_ld_prior(U_mu_hr_init, flags, stellar_cfg)
        if hr_ld_mode == 'fixed':
            model_run_args_hr['ld_fixed'] = U_mu_hr_init
        elif hr_ld_mode in {'gaussian', 'stellarprior', 'sing'}:
            model_run_args_hr['mu_u_ld'] = U_mu_hr_init
            if ((ld_profile == 'power2' and hr_ld_mode == 'stellarprior') or hr_ld_mode == 'sing') and U_sigma_hr_init is not None:
                model_run_args_hr['sigma_u_ld'] = U_sigma_hr_init
        elif hr_ld_mode == 'uniform':
            pass
        else:
            raise ValueError(f"Unknown ld_prior mode: {hr_ld_mode}")

    model_run_args_hr['mu_duration'] = DURATION_BASE
    model_run_args_hr['mu_t0'] = T0_BASE
    model_run_args_hr['mu_b'] = B_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = PERIOD_FIXED
    if transit_engine == 'jaxoplanet':
        model_run_args_hr['precomputed_yerr_per_lc'] = jnp.nanmedian(
            flux_err_hr, axis=1
        )
    if transit_engine == 'harmonica':
        model_run_args_hr['mu_cos_i'] = COSI_BASE
        model_run_args_hr['harmonica_a_rs'] = A_RS_BASE
        model_run_args_hr['harmonica_ecc'] = HARMONICA_ECC
        model_run_args_hr['harmonica_omega'] = HARMONICA_OMEGA
    elif param_method == 'a_rs':
        model_run_args_hr['mu_a_rs'] = A_RS_BASE
        model_run_args_hr['mu_ecc'] = HARMONICA_ECC
        model_run_args_hr['mu_omega'] = HARMONICA_OMEGA

    init_params_hr = {
        "rors": jnp.tile(RORS_BASE, (num_lcs_hr, 1)),
        "u": _clip_ld_initial_values(
            U_mu_hr_init,
            ld_profile, 'high-res',
        ),
    }
    if joint_geometry_hr is not None:
        model_run_args_hr.update(joint_geometry_hr['model_kwargs'])
        init_params_hr.update(joint_geometry_hr['init_params'])
    _seed_surface_spectroscopic_init(
        init_params_hr, surface_config, num_lcs_hr
    )
    if ld_profile == 'quadratic' and hr_ld_mode == 'uniform':
        init_params_hr.pop('u', None)
        init_params_hr.update(_quadratic_uniform_initial_sites(
            U_mu_hr_init, ld_uniform_basis
        ))
    if hr_ld_mode == 'sing':
        init_params_hr.pop('u', None)
        init_params_hr['limb_l'] = jnp.asarray(U_mu_hr_init)[:, 0]
        init_params_hr['limb_delta'] = jnp.asarray(U_mu_hr_init)[:, 1]
    if ld_profile == 'power2' and hr_ld_mode not in {'fixed', 'interpolated'}:
        init_params_hr.pop('u', None)
        init_params_hr.update(
            _power2_ld_initial_sites(
                U_mu_hr_init, spectro_ld_parameterization
            )
        )
    if transit_engine == 'harmonica':
        if harmonica_spectro_parameterization == 'delta_r':
            init_val = (
                bestfit_params_wl_df['a1'].values[0]
                if 'a1' in bestfit_params_wl_df.columns
                else HARMONICA_INIT_ODD_COEFF
            )
            init_delta_r = jnp.atleast_1d(
                jnp.asarray(_harmonica_coeff_array_to_delta_r(init_val))
            )
            init_params_hr['delta_r'] = jnp.tile(
                init_delta_r[None, :],
                (num_lcs_hr, 1),
            )
        elif harmonica_spectro_parameterization == 'half_area':
            init_val = (
                bestfit_params_wl_df['a1'].values
                if 'a1' in bestfit_params_wl_df.columns
                else np.full_like(RORS_BASE, HARMONICA_INIT_ODD_COEFF)
            )
            init_area_radius, init_q = (
                _harmonica_coefficients_to_half_area_init(
                    RORS_BASE, init_val
                )
            )
            init_params_hr['rors'] = jnp.tile(
                jnp.atleast_1d(jnp.asarray(init_area_radius))[None, :],
                (num_lcs_hr, 1),
            )
            init_params_hr['q'] = jnp.tile(
                jnp.atleast_1d(jnp.asarray(init_q))[None, :],
                (num_lcs_hr, 1),
            )
        else:
            for harmonic_name in HARMONICA_ODD_HARMONICS:
                init_val = bestfit_params_wl_df[harmonic_name].values[0] if harmonic_name in bestfit_params_wl_df.columns else HARMONICA_INIT_ODD_COEFF
                init_frac = _harmonica_coeff_array_to_frac(init_val, RORS_BASE)
                init_params_hr[_harmonica_frac_site(harmonic_name)] = jnp.tile(
                    jnp.asarray(init_frac)[None, :],
                    (num_lcs_hr, 1),
                )
    if hr_trend_mode in {'free', 'gaussian_marginalized'}:
        if detrend_type_multiwave != 'none':
            init_params_hr["c"] = jnp.clip(
                jnp.full(num_lcs_hr, bestfit_params_wl_df['c'].values[0]),
                0.900001, 1.099999,
            )
            if 'v' in bestfit_params_wl_df.columns:
                init_params_hr["v"] = jnp.clip(
                    jnp.full(num_lcs_hr, bestfit_params_wl_df['v'].values[0]),
                    -0.099999, 0.099999,
                )
        if 'explinear' in detrend_type_multiwave:
            init_params_hr['A'] = jnp.full(
                num_lcs_hr, bestfit_params_wl_df['A'].values[0]
            )
            if 'explinear_spectroscopic' not in detrend_type_multiwave:
                init_params_hr['log_tau'] = jnp.full(
                    num_lcs_hr,
                    jnp.log(bestfit_params_wl_df['tau'].values[0]),
                )

    hr_transit_window_indices = None
    if (
        transit_engine == 'jaxoplanet'
        and transit_window_optimization == 'auto'
        and param_method == 'duration'
    ):
        hr_transit_window_indices = build_transit_window_indices(
            np.asarray(time_hr),
            np.asarray(PERIOD_FIXED),
            np.asarray(T0_BASE),
            _joint_geometry_window_duration(
                DURATION_BASE, joint_geometry_hr, param_method
            ),
        )
        print(
            "Jaxoplanet transit window (high-res): "
            f"{len(hr_transit_window_indices)}/{time_hr.size} active cadences."
        )

    hr_model_builder_kwargs = {
        'detrend_type': detrend_type_multiwave,
        'ld_mode': hr_ld_mode,
        'trend_mode': hr_trend_mode,
        'n_planets': n_planets,
        **(
            {
                'transit_window_indices': hr_transit_window_indices,
                'transit_grid_non_grazing': bool(
                    np.max(np.abs(np.asarray(B_BASE)))
                    < 1.0 - np.sqrt(0.5)
                ),
                'transit_grid_outer_contact_safe': bool(
                    np.max(np.abs(np.asarray(B_BASE)))
                    < 1.0 + np.sqrt(1.0e-5) - 1.0e-5
                ),
                'joint_geometry': joint_geometry_hr is not None,
            }
            if transit_engine == 'jaxoplanet' else {}
        ),
        **_engine_spectro_kw,
    }
    spot_basis_hr = None
    if hr_ld_mode == 'fixed':
        spot_basis_hr = _prepare_fixed_surface_basis(
            surface_config,
            time_hr,
            U_mu_hr_init,
            period=PERIOD_FIXED,
            t0=T0_BASE,
            a_rs=A_RS_BASE,
            b=B_BASE,
            rors=RORS_BASE,
            ecc=HARMONICA_ECC,
            omega=HARMONICA_OMEGA,
            ld_profile=ld_profile,
        )
        if spot_basis_hr is not None:
            model_run_args_hr['surface_basis_data'] = spot_basis_hr
            print(
                "Prepared exact fixed-geometry surface basis for "
                "high-resolution inference."
            )
    hr_model_for_run = _build_spectroscopic_model(
        create_vectorized_model,
        **hr_model_builder_kwargs,
    )
    hr_adaptive_fallback_model = None
    hr_adaptive_fallback_init = None
    if (
        transit_engine == 'jaxoplanet'
        and ld_profile == 'power2'
        and hr_ld_mode not in {'fixed', 'interpolated'}
        and spectro_ld_parameterization != 'coefficients'
    ):
        coefficient_builder_kwargs = dict(hr_model_builder_kwargs)
        coefficient_builder_kwargs['ld_parameterization'] = 'coefficients'
        hr_adaptive_fallback_model = _build_spectroscopic_model(
            create_vectorized_model, **coefficient_builder_kwargs
        )
        hr_adaptive_fallback_init = dict(init_params_hr)
        hr_adaptive_fallback_init.pop('ld_decorrelated', None)
        hr_adaptive_fallback_init.update(
            _power2_ld_initial_sites(U_mu_hr_init, 'coefficients')
        )
    
    if 'gp_spectroscopic' in detrend_type_multiwave:
        model_run_args_hr['gp_trend'] = gp_trend
        init_params_hr['A_gp'] = jnp.ones(num_lcs_hr)
    if '2spot_spectroscopic' in detrend_type_multiwave:
        if spot_trend is None or spot_trend2 is None:
            raise ValueError("2spot_spectroscopic requires WL spot and spot2 trends.")
        if len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        if len(spot_trend2) != len(time_hr):
            spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_hr))
        model_run_args_hr['spot_trend'] = spot_trend
        model_run_args_hr['spot_trend2'] = spot_trend2
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
        init_params_hr['A_spot2'] = jnp.ones(num_lcs_hr)
    elif _has_single_spot_spectroscopic(detrend_type_multiwave):
        if spot_trend is None:
            raise ValueError(f"{detrend_type_multiwave} requires WL spot to build spot_trend.")
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        model_run_args_hr['spot_trend'] = spot_trend
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
    if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
        if jump_trend is None:
            raise ValueError("linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.")
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_hr))
        model_run_args_hr['jump_trend'] = jump_trend
        init_params_hr['A_jump'] = jnp.ones(num_lcs_hr)
    if 'explinear_spectroscopic' in detrend_type_multiwave:
        exp_trend_hr = _align_trend_to_time(
            exp_trend, wl_time_good, np.asarray(time_hr)
        )
        model_run_args_hr['exp_trend'] = exp_trend_hr
        model_run_args_hr['fixed_tau'] = fixed_tau_spectro

    if (
        transit_engine == 'jaxoplanet'
        and _engine_spectro_kw.get('cadence_reduction') == 'auto'
        and int(time_hr.size) > 5000
        and hr_trend_mode == 'free'
        and hr_transit_window_indices is not None
        and linear_spectro_trend_coefficient_names(
            detrend_type_multiwave
        ) is not None
    ):
        trend_names_hr, trend_design_hr = build_linear_spectro_trend_design(
            detrend_type_multiwave,
            time_hr,
            exp_trend=model_run_args_hr.get('exp_trend'),
            spot_trend=model_run_args_hr.get('spot_trend'),
            spot_trend2=model_run_args_hr.get('spot_trend2'),
            jump_trend=model_run_args_hr.get('jump_trend'),
        )
        reference_beta_hr = np.column_stack([
            np.asarray(init_params_hr.get(
                name, np.zeros(num_lcs_hr, dtype=np.float64)
            ))
            for name in trend_names_hr
        ])
        model_run_args_hr['trend_design'] = trend_design_hr
        model_run_args_hr.update(build_linear_oot_statistics(
            time_hr,
            flux_hr,
            flux_err_hr,
            hr_transit_window_indices,
            trend_design_hr,
            reference_beta_hr,
        ))

    trend_names_hr = None
    if hr_trend_mode == 'gaussian_marginalized':
        (
            model_run_args_hr['trend_prior_mean'],
            model_run_args_hr['trend_prior_scale'],
            trend_names_hr,
        ) = _build_gaussian_trend_prior(
            detrend_type_multiwave,
            int(num_lcs_hr),
            init_params_hr,
            flags,
        )
        for trend_name in trend_names_hr:
            init_params_hr.pop(trend_name, None)

    hr_stage_sampler_kwargs = (
        harmonica_hr_nuts_kwargs
        if transit_engine == 'harmonica'
        else jaxoplanet_hr_nuts_kwargs
    )
    vmap_chunk_size_hr = _resolve_spectro_joint_geometry_chunk_size(
        vmap_chunk_size_hr, num_lcs_hr, joint_geometry_hr is not None
    )
    vmap_chunk_size_hr = _resolve_stage_vmap_width(
        flags,
        'highres',
        vmap_chunk_size_hr,
        sampler_backend=spectro_sampler,
        trend_inference=(
            'fixed'
            if hr_trend_mode == 'fixed'
            else (
                'gaussian_marginalized'
                if hr_trend_mode == 'gaussian_marginalized'
                else 'sampled_uniform'
            )
        ),
        num_cadences=int(time_hr.size),
        active_transit_cadences=(
            int(len(hr_transit_window_indices))
            if hr_transit_window_indices is not None
            else int(time_hr.size)
        ),
        mcmc_kwargs=highres_mcmc_kwargs,
        sampler_kwargs=hr_stage_sampler_kwargs,
        transit_engine=transit_engine,
        ld_profile=ld_profile,
        ld_mode=hr_ld_mode,
        detrend_type=detrend_type_multiwave,
        param_method=param_method,
        n_planets=n_planets,
        transit_window=transit_window_optimization,
        transit_grid_non_grazing=hr_model_builder_kwargs.get(
            'transit_grid_non_grazing', False
        ),
        transit_grid_outer_contact_safe=hr_model_builder_kwargs.get(
            'transit_grid_outer_contact_safe', False
        ),
    )

    if (
        chunk_mode != 'serial'
        and vmap_chunk_size_hr is None
    ):
        raise ValueError(
            "flags.chunk_mode requires flags.spectro_chunk_size or flags.vmap_chunk."
        )

    samples_hr = _run_sampling_stage(
        hr_model_for_run,
        key_mcmc_hr,
        time_hr,
        flux_err_hr,
        flux_hr,
        init_params_hr,
        nuts_kwargs=hr_stage_sampler_kwargs,
        mcmc_kwargs=highres_mcmc_kwargs,
        use_chunked=(
            vmap_chunk_size_hr is not None
            or spectro_sampler in {'independent_nuts', 'independent_hmc'}
        ),
        chunk_size=vmap_chunk_size_hr,
        chunk_mode=chunk_mode,
        parallel_job_count=chunk_parallel_job_count,
        parallel_job_index=chunk_parallel_job_index,
        output_dir=output_dir,
        checkpoint_prefix=_harmonica_checkpoint_prefix(
            f"{instrument_full_str}_{hr_bin_str}",
            transit_engine,
            harmonica_spectro_parameterization,
        ),
        checkpoint_signature={
            'stage': 'high_resolution',
            'spectro_joint_geometry': joint_geometry_hr is not None,
            'spectro_joint_geometry_prior_inflation': (
                None if joint_geometry_hr is None
                else joint_geometry_hr['prior_inflation']
            ),
            'transit_engine': transit_engine,
            'ld_profile': ld_profile,
            'ld_mode': hr_ld_mode,
            'trend_inference': trend_inference,
            'detrend_type': detrend_type_multiwave,
            'param_method': param_method,
            'whitelight_geometry_estimator': whitelight_geometry_estimator,
            'whitelight_geometry_handoff_fingerprint': (
                wl_geometry_handoff['artifact_fingerprint_sha256']
            ),
            'jaxoplanet_kernel': jaxoplanet_kernel,
            'transit_window_optimization': transit_window_optimization,
            'spectro_cadence_reduction': _engine_spectro_kw.get(
                'cadence_reduction', 'off'
            ),
            'spectro_transit_grid': _engine_spectro_kw.get(
                'transit_grid', 'off'
            ),
            'spectro_transit_grid_nodes': _engine_spectro_kw.get(
                'transit_grid_nodes'
            ),
            'surface_config': surface_config,
            'harmonica_max_order': max_harmonic_order,
            'harmonica_spectro_parameterization': harmonica_spectro_parameterization,
            'harmonica_spectro_fit_jitter': harmonica_spectro_fit_jitter,
            'harmonica_spectro_odd_frac_sigma': harmonica_spectro_odd_frac_sigma,
        },
        sampler_backend=spectro_sampler,
        channel_varying_kwargs=(
            JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
            if transit_engine == 'jaxoplanet'
            else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS
        ),
        spectro_min_depth_ess=spectro_min_depth_ess,
        spectro_max_divergences=spectro_max_divergences,
        dump_metadata={
            'config_path': os.path.abspath(config_path),
            'stage_kind': 'high_resolution',
            'stage_label': (
                f"{instrument_full_str}_{hr_bin_str}_high_resolution"
            ),
            'instrument': instrument,
            'instrument_label': instrument_full_str,
            'output_dir': os.path.abspath(output_dir),
            'wavelength': np.asarray(data.wavelengths_hr),
            'wavelength_err': np.asarray(data.wavelengths_err_hr),
            'active_window_cadences': (
                int(len(hr_transit_window_indices))
                if hr_transit_window_indices is not None
                else int(time_hr.size)
            ),
            'transit_engine': transit_engine,
        },
        adaptive_fallback_model=hr_adaptive_fallback_model,
        adaptive_fallback_init_params=hr_adaptive_fallback_init,
        **model_run_args_hr
    )
    if samples_hr is None:
        print("\nHigh-resolution chunk job complete. Exiting before final aggregation.")
        return
    if trend_names_hr is not None:
        samples_hr = materialize_marginalized_trend_samples(
            samples_hr,
            trend_names_hr,
            key_map_hr,
        )

    map_params_hr = {
        "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
        "rors": jnp.nanmedian(
            _harmonica_model_radius_samples(
                samples_hr, harmonica_spectro_parameterization
            )
            if transit_engine == 'harmonica'
            else samples_hr["rors"],
            axis=0,
        ),
        "period": PERIOD_FIXED
    }
    if ld_profile == 'quadratic':
        quadratic_hr_coeffs = U_mu_hr_init
        quadratic_u_med_hr = _posterior_quadratic_ld_median(
            samples_hr, quadratic_hr_coeffs
        )
        map_params_hr["u"] = quadratic_u_med_hr
        if transit_engine == 'harmonica':
            map_params_hr['u1_ld'] = quadratic_u_med_hr[..., 0]
            map_params_hr['u2_ld'] = quadratic_u_med_hr[..., 1]
    elif ld_profile == 'power2':
        if 'c1' in samples_hr:
            c1_med_hr = jnp.nanmedian(samples_hr['c1'], axis=0)
            c2_med_hr = jnp.nanmedian(samples_hr['c2'], axis=0)
            map_params_hr['c1'] = c1_med_hr
            map_params_hr['c2'] = c2_med_hr
            map_params_hr['u'] = jax.vmap(compute_u_from_c)(c1_med_hr, c2_med_hr)
            if transit_engine == 'harmonica':
                map_params_hr['c_ld'] = c1_med_hr
                map_params_hr['alpha_ld'] = c2_med_hr
        else:
             if 'u' in samples_hr:
                 map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    if transit_engine == 'harmonica':
        map_params_hr = _augment_harmonica_params(
            map_params_hr,
            a_rs=A_RS_BASE,
            ecc=HARMONICA_ECC,
            omega=HARMONICA_OMEGA,
            a1=jnp.nanmedian(samples_hr["a1"], axis=0) if "a1" in samples_hr else None,
            a3=jnp.nanmedian(samples_hr["a3"], axis=0) if "a3" in samples_hr else None,
            a5=jnp.nanmedian(samples_hr["a5"], axis=0) if "a5" in samples_hr else None,
        )
    elif param_method == 'a_rs':
        map_params_hr.update({
            "a_rs": A_RS_BASE,
            "ecc": HARMONICA_ECC,
            "omega": HARMONICA_OMEGA,
        })
    map_params_hr.update(
        _joint_geometry_posterior_medians(samples_hr, joint_geometry_hr)
    )
    map_params_hr.update({k: jnp.nanmedian(samples_hr[k], axis=0) for k in TREND_PARAMS if k in samples_hr})
    map_params_hr.update({
        k: jnp.nanmedian(samples_hr[k], axis=0)
        for k in SURFACE_PARAMS if k in samples_hr
    })
    if transit_engine == 'jaxoplanet':
        map_params_hr = _attach_surface_eval_metadata(
            map_params_hr, surface_config
        )
        map_params_hr = _attach_jaxoplanet_eval_metadata(
            map_params_hr,
            time_hr,
            jaxoplanet_kernel=jaxoplanet_kernel,
            ld_profile=ld_profile,
            transit_window_optimization=transit_window_optimization,
        )

    in_axes_map_hr = {"rors": 0}
    in_axes_map_hr.update({
        name: 0 for name in ('u', 'c1', 'c2') if name in map_params_hr
    })
    if transit_engine == 'harmonica':
        if ld_profile == 'power2':
            in_axes_map_hr.update({'c_ld': 0, 'alpha_ld': 0})
        else:
            in_axes_map_hr.update({'u1_ld': 0, 'u2_ld': 0})
        in_axes_map_hr.update({name: 0 for name in HARMONICA_ODD_HARMONICS if name in map_params_hr})
    in_axes_map_hr.update({k: 0 for k in TREND_PARAMS if k in map_params_hr})
    in_axes_map_hr.update({k: 0 for k in SURFACE_PARAMS if k in map_params_hr})
    
    final_in_axes_hr = {k: in_axes_map_hr.get(k, None) for k in map_params_hr.keys()}
    selected_kernel_hr = resolve_detrend_kernel(detrend_type_multiwave)
    
    num_lcs_hr = flux_err_hr.shape[0]
    model_all_hr = evaluate_channels_sequentially(
        map_params_hr,
        final_in_axes_hr,
        time_hr,
        num_lcs_hr,
        detrend_type_multiwave,
        selected_kernel_hr,
        transit_engine,
        surface_config,
        jaxoplanet_kernel,
        ld_profile,
        gp_trend,
        spot_trend,
        spot_trend2,
        jump_trend,
        locals().get('exp_trend_hr'),
        _attach_surface_eval_metadata,
        _has_single_spot_spectroscopic,
    )

    residuals_hr = np.array(flux_hr - model_all_hr)
    _hr_dt_sec = float(np.nanmedian(np.diff(np.array(time_hr)))) * 86400.0
    if plots_mode == 'full':
        plot_noise_binning_robust(
            residuals_hr, _hr_dt_sec,
            f"{output_dir}/36_{hr_artifact_stem}_noisebin.png",
            title=f"High-Res Noise Binning ({hr_bin_str})",
        )
    save_noise_binning_data(
        residuals_hr,
        f"{output_dir}/36_{hr_artifact_stem}_noisebin.csv",
    )

    median_total_error_hr = np.nanmedian(samples_hr['total_error'], axis=0)
    plot_params_hr = dict(map_params_hr)
    if spot_basis_hr is not None:
        plot_params_hr['_surface_basis'] = spot_basis_hr
        plot_params_hr['_surface_basis_time'] = jnp.asarray(time_hr)
    if plots_mode == 'full':
        plot_wavelength_offset_summary(time_hr, flux_hr, median_total_error_hr, data.wavelengths_hr,
                                        plot_params_hr, {
                                            "period": PERIOD_FIXED,
                                            "transit_engine": transit_engine,
                                            "param_method": param_method,
                                        },
                                        f"{output_dir}/34_{hr_artifact_stem}_summary.png",
                                      detrend_type=detrend_type_multiwave, gp_trend=gp_trend, spot_trend=spot_trend, spot_trend2=spot_trend2, jump_trend=jump_trend, exp_trend=(exp_trend_hr if 'explinear_spectroscopic' in detrend_type_multiwave else None))

    if surface_config['model'] != 'eclipse':
        plot_transmission_spectrum(
            wl_hr, samples_hr["rors"],
            f"{output_dir}/31_{hr_artifact_stem}_spectrum",
        )
        save_results(
            wl_hr, data.wavelengths_err_hr, samples_hr,
            f"{output_dir}/{hr_artifact_stem}.csv",
        )
    save_surface_results(
        wl_hr, data.wavelengths_err_hr, samples_hr,
        f"{output_dir}/{hr_artifact_stem}.csv",
    )
    save_detailed_fit_results(time_hr, flux_hr, flux_err_hr, data.wavelengths_hr, data.wavelengths_err_hr, samples_hr, map_params_hr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{hr_artifact_stem}", median_total_error_hr, gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend, joint_geometry=(None if joint_geometry_hr is None else joint_geometry_hr['summary']))

    if transit_engine == 'harmonica' and _has_harmonica_odd_samples(samples_hr):
        save_harmonica_limb_products(
            wavelengths=wl_hr,
            wavelength_err=data.wavelengths_err_hr,
            rors_samples=_harmonica_model_radius_samples(
                samples_hr, harmonica_spectro_parameterization
            ),
            harmonic_samples=_harmonica_sample_payload(samples_hr),
            csv_path=f"{output_dir}/{hr_artifact_stem}_limb_spectra.csv",
            limb_spectrum_path=f"{output_dir}/35_{hr_artifact_stem}_limb_spectra.png",
            transmission_strings_path=f"{output_dir}/36_{hr_artifact_stem}_transmission_strings.png",
            posterior_strings_path=f"{output_dir}/37_{hr_artifact_stem}_transmission_string_posterior.png",
            posterior_samples_path=f"{output_dir}/{hr_artifact_stem}_limb_posterior_samples.npz",
            title_prefix=f"{planet_str} - {hr_bin_str}",
        )
    return True


_SPECTROSCOPIC_STAGE_HOOKS = {
    "lowres": _run_low_resolution_stage_hook,
    "highres": _run_high_resolution_stage_hook,
}


def run_spectroscopic_stage(stage, *args, **kwargs):
    """Run one literal spectroscopic stage through the shared dispatch seam."""
    try:
        stage_hook = _SPECTROSCOPIC_STAGE_HOOKS[stage]
    except KeyError as error:
        raise ValueError(
            f"Unknown spectroscopic stage {stage!r}; expected 'lowres' or 'highres'."
        ) from error
    return stage_hook(*args, **kwargs)


def run_low_resolution_stage(*args, **kwargs):
    """Compatibility wrapper for the low-resolution stage."""
    return run_spectroscopic_stage("lowres", *args, **kwargs)


def run_high_resolution_stage(*args, **kwargs):
    """Compatibility wrapper for the high-resolution stage."""
    return run_spectroscopic_stage("highres", *args, **kwargs)

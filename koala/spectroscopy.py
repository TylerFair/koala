"""Low- and high-resolution spectroscopic fitting."""

from .output_paths import OutputPaths, state_path

from koala.console import debug, info, spectroscopy_verdict, grid_description, joint_display

import os
from types import SimpleNamespace
import glob
import json
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from models.sing_ld import (
    SING_TABULATED_OFFSET,
    SING_TABULATED_SCATTER,
    estimate_gray_offset,
    quadratic_to_sing,
    write_offset_artifact,
)
from models.detrend import resolve_detrend_kernel
from models.cadence_reduction import (
    build_linear_oot_statistics,
    build_linear_spectro_trend_design,
    linear_spectro_trend_coefficient_names,
)
from models.trend_marginal import materialize_marginalized_trend_samples
from .surface_outputs import save_surface_results
from .constants import (
    HARMONICA_CHANNEL_VARYING_MODEL_KWARGS,
    HARMONICA_INIT_ODD_COEFF,
    JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS,
    SURFACE_PARAMS,
    TREND_PARAMS,
)
from .config import (
    _build_gaussian_trend_prior,
    _spectro_detrend_type,
    _has_single_spot_spectroscopic,
    _resolve_stage_vmap_width,
    _resolve_spectro_joint_geometry,
    _resolve_spectro_joint_geometry_sampler,
    _resolve_spectro_joint_geometry_chunk_size,
)
from .artifacts import _science_artifact_fingerprint, _atomic_save_npy, ArtifactSet
from .outputs import (
    save_noise_binning_data,
    compute_u_from_c,
    save_results,
    save_detailed_fit_results,
)
from .geometry import _joint_geometry_sites, _whitelight_geometry_prior
from .sampling import _run_sampling_stage, evaluate_channels_sequentially
from .limb_darkening import (
    _power2_ld_initial_sites,
    _clip_ld_initial_values,
    _quadratic_uniform_initial_sites,
    get_limb_darkening,
    build_sing_ld_prior,
    _assess_sing_gray_fit,
    get_or_build_power2_ld_prior, get_or_build_quadratic_ld_prior,
    _posterior_quadratic_ld_median,
)
from .harmonica_products import (
    _harmonica_frac_site,
    _harmonica_coeff_array_to_frac,
    _harmonica_coeff_array_to_delta_r,
    _harmonica_coefficients_to_half_area_init,
    _harmonica_model_radius_samples,
    _harmonica_checkpoint_prefix,
    _harmonica_sample_payload,
    _has_harmonica_odd_samples,
    _augment_harmonica_params,
    save_harmonica_limb_products,
)
from .surface import (
    _attach_jaxoplanet_eval_metadata,
    _attach_surface_eval_metadata,
    _seed_surface_spectroscopic_init,
    _prepare_fixed_surface_basis,
    _surface_basis_on_time_mask,
)


from .sampling import _build_spectroscopic_model

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
    debug(
        f"Joint geometry ({stage_name}): sampling shared "
        f"{', '.join(sites)} with Gaussian priors of "
        f"{inflation:g}x the white-light posterior std; "
        f"sampler={spectro_sampler}."
    )
    for name in sites:
        debug(
            f"  {name}: centre={np.asarray(prior[name], dtype=float)}, "
            f"prior sigma={np.asarray(prior[f'sigma_{name}'], dtype=float)}"
        )
    return joint, spectro_sampler, nuts_kwargs


def _joint_geometry_posterior_medians(samples, joint):
    """Return the posterior medians of the shared geometry sites."""
    if joint is None:
        return {}
    return {
        name: jnp.nanmedian(jnp.asarray(samples[name]), axis=0)
        for name in joint['sites']
    }


def _spectroscopic_posterior_model(
    *, A_RS_BASE, B_BASE, DURATION_BASE, HARMONICA_ECC, HARMONICA_ODD_HARMONICS,
    HARMONICA_OMEGA, PERIOD_FIXED, T0_BASE, detrend_type_multiwave,
    harmonica_spectro_parameterization, jaxoplanet_kernel, joint_geometry, ld_coefficients,
    ld_profile, param_method, samples, surface_config, time, transit_engine,
    transit_window_optimization,
):
    """Build the median channel model and its channel-varying axes."""
    map_params = {
        "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
        "rors": jnp.nanmedian(
            _harmonica_model_radius_samples(samples, harmonica_spectro_parameterization)
            if transit_engine == 'harmonica'
            else samples["rors"],
            axis=0,
        ),
        "period": PERIOD_FIXED,
    }

    if ld_profile == 'power2' and 'c1' in samples:
        c1_med = jnp.nanmedian(samples['c1'], axis=0)
        c2_med = jnp.nanmedian(samples['c2'], axis=0)
        map_params['c1'] = c1_med
        map_params['c2'] = c2_med
        map_params['u'] = jax.vmap(compute_u_from_c)(c1_med, c2_med)
        if transit_engine == 'harmonica':
            map_params['c_ld'] = c1_med
            map_params['alpha_ld'] = c2_med
    elif ld_profile == 'power2':
        if 'u' in samples:
            map_params['u'] = jnp.nanmedian(np.array(samples['u']), axis=0)
    else:
        quadratic_u_med = _posterior_quadratic_ld_median(samples, ld_coefficients)
        map_params['u'] = quadratic_u_med
        if transit_engine == 'harmonica':
            map_params['u1_ld'] = quadratic_u_med[..., 0]
            map_params['u2_ld'] = quadratic_u_med[..., 1]

    if transit_engine == 'harmonica':
        map_params = _augment_harmonica_params(
            map_params, a_rs=A_RS_BASE, ecc=HARMONICA_ECC, omega=HARMONICA_OMEGA,
            a1=jnp.nanmedian(samples['a1'], axis=0) if 'a1' in samples else None,
            a3=jnp.nanmedian(samples['a3'], axis=0) if 'a3' in samples else None,
            a5=jnp.nanmedian(samples['a5'], axis=0) if 'a5' in samples else None,
        )
    else:
        map_params.update({'a_rs': A_RS_BASE, 'ecc': HARMONICA_ECC, 'omega': HARMONICA_OMEGA})
    if 'dt0' in samples:
        map_params['t0'] = jnp.asarray(T0_BASE) + jnp.nanmedian(samples['dt0'], axis=0)
    shared = _joint_geometry_posterior_medians(samples, joint_geometry)
    map_params.update(shared)
    if 'duration' in shared:
        # The duration posterior's derived shared orbit is recorded by builder.
        map_params['a_rs'] = jnp.nanmedian(samples['orbit_a_rs'], axis=0)

    map_params.update({k: jnp.nanmedian(samples[k], axis=0) for k in TREND_PARAMS if k in samples})
    map_params.update(
        {k: jnp.nanmedian(samples[k], axis=0) for k in SURFACE_PARAMS if k in samples},
    )
    if transit_engine == 'jaxoplanet':
        map_params = _attach_surface_eval_metadata(map_params, surface_config)
        map_params = _attach_jaxoplanet_eval_metadata(
            map_params, time, jaxoplanet_kernel=jaxoplanet_kernel, ld_profile=ld_profile,
            transit_window_optimization=transit_window_optimization,
        )

    selected_kernel = resolve_detrend_kernel(detrend_type_multiwave)
    in_axes_map = {'rors': 0}
    if 'dt0' in samples:
        in_axes_map['t0'] = 0
    in_axes_map.update({name: 0 for name in ('u', 'c1', 'c2') if name in map_params})
    if transit_engine == 'harmonica':
        if ld_profile == 'power2':
            in_axes_map.update({'c_ld': 0, 'alpha_ld': 0})
        else:
            in_axes_map.update({'u1_ld': 0, 'u2_ld': 0})
        in_axes_map.update({name: 0 for name in HARMONICA_ODD_HARMONICS if name in map_params})
    in_axes_map.update({k: 0 for k in TREND_PARAMS if k in map_params})
    in_axes_map.update({k: 0 for k in SURFACE_PARAMS if k in map_params})

    final_in_axes = {k: in_axes_map.get(k, None) for k in map_params.keys()}
    return map_params, final_in_axes, selected_kernel


def _calibrate_sing_offsets(
    *, U_mu_lr, create_vectorized_model, data, flags, flux_err_lr, flux_lr,
    harmonica_spectro_parameterization, init_params_lr, instrument_full_str,
    jaxoplanet_lr_nuts_kwargs, joint_geometry_lr, key_mcmc_lr, lr_bin_str,
    lr_model_builder_kwargs, model_run_args_lr, num_lcs_lr, output_dir, sing_model_c_lr,
    stellar_cfg, time_lr, transit_engine, vmap_chunk_size_lr, wl_geometry_handoff,
):
    """Calibrate the Sing gray offset before the low-resolution fit."""
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
    gray_digest = _science_artifact_fingerprint('sing_gray_offset', gray_fingerprint_inputs)
    gray_artifact_path = state_path(
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
                debug(f"[Sing LD] reusing fitted gray offsets from {gray_artifact_path}", flush=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            fitted_offsets = None
    if fitted_offsets is None:
        debug("[Sing LD] running broad independent (u_plus, u_minus) coarse calibration pass", flush=True)
        gray_builder_kwargs = dict(lr_model_builder_kwargs)
        gray_builder_kwargs['ld_mode'] = 'uniform'
        # The gray calibration pass is a fixed-geometry lane fit.
        gray_builder_kwargs.pop('joint_geometry', None)
        gray_model = _build_spectroscopic_model(create_vectorized_model, **gray_builder_kwargs)
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
            np.asarray(sing_model_c_lr)[:, 0], np.asarray(sing_model_c_lr)[:, 1],
        )
        gray_u_plus_init = 1.0 - np.asarray(gray_l_init)
        gray_u_minus_init = gray_u_plus_init - 8.0 * np.asarray(gray_delta_init)
        gray_init['ld_uplus_uminus'] = jnp.column_stack(
            (jnp.asarray(gray_u_plus_init), jnp.asarray(gray_u_minus_init)),
        )
        gray_args = dict(model_run_args_lr)
        gray_args.pop('mu_u_ld', None)
        gray_args.pop('sigma_u_ld', None)
        for shared_kwarg in (
            joint_geometry_lr['model_kwargs'] if joint_geometry_lr else ()
        ):
            gray_args.pop(shared_kwarg, None)
        gray_prefix = _harmonica_checkpoint_prefix(
            f'{instrument_full_str}_{lr_bin_str}_sing_gray_free_{gray_digest[:12]}',
            transit_engine, harmonica_spectro_parameterization,
        )
        gray_mcmc = {'num_warmup': 1000, 'num_samples': 1000}
        gray_nuts_kwargs = dict(jaxoplanet_lr_nuts_kwargs)
        gray_nuts_kwargs.update(
            mass_matrix='laplace', laplace_hessian_method='finite_difference',
            laplace_warmup=1000, laplace_target_accept=0.99, laplace_max_tree_depth=10,
            target_accept_prob=0.99, max_tree_depth=10,
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
                OutputPaths(output_dir).state_dir, 'chunks', f"{gray_prefix}*_chunk_*.diagnostics.json"
            )))
            gray_ok, gray_diagnostics = _assess_sing_gray_fit(
                gray_samples, gray_diag_paths, min_ess=100.0,
            )
            if not gray_ok:
                raise RuntimeError(f"gray calibration diagnostics failed: {gray_diagnostics}")
            if gray_diagnostics['channels_below_ess_warning']:
                info('Some limb-darkening calibration estimates have low effective sample size; their uncertainties are increased.')
                debug(
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
                fitted_c, np.asarray(sing_model_c_lr), fitted_c_sigma, ess=gray_ess,
                n_draws=np.asarray(gray_samples['c1']).shape[0],
            )
            fitted_offsets['diagnostics'] = gray_diagnostics
            fitted_offsets['tabulated_l'] = SING_TABULATED_OFFSET['l']
            fitted_offsets['tabulated_delta'] = SING_TABULATED_OFFSET['delta']
            fitted_offsets['tabulated_l_sigma'] = SING_TABULATED_SCATTER['l']
            fitted_offsets['tabulated_delta_sigma'] = SING_TABULATED_SCATTER['delta']
            write_offset_artifact(gray_artifact_path, fitted_offsets, gray_fingerprint_inputs)
        except Exception as error:
            info('The limb-darkening calibration failed; using the tabulated stagger offsets.')
            debug(
                f"[Sing LD] WARNING: free gray calibration unavailable or failed ({error}); "
                "falling back to tabulated Stagger offsets.",
                flush=True,
            )
            fitted_offsets = None
    if fitted_offsets is not None:
        debug(
            "[Sing LD] gray offsets fitted vs tabulated: "
            f"delta_l={float(fitted_offsets['l']):+.6f} +/- {float(fitted_offsets['l_sigma']):.6f} "
            f"vs {SING_TABULATED_OFFSET['l']:+.6f}; "
            f"delta_delta={float(fitted_offsets['delta']):+.6f} +/- {float(fitted_offsets['delta_sigma']):.6f} "
            f"vs {SING_TABULATED_OFFSET['delta']:+.6f}",
            flush=True,
        )
        U_mu_lr, U_sigma_lr = build_sing_ld_prior(
            sing_model_c_lr, flags, stellar_cfg, offsets_override=fitted_offsets,
        )
        model_run_args_lr['mu_u_ld'] = U_mu_lr
        model_run_args_lr['sigma_u_ld'] = U_sigma_lr
        init_params_lr['limb_l'] = jnp.asarray(U_mu_lr)[:, 0]
        init_params_lr['limb_delta'] = jnp.asarray(U_mu_lr)[:, 1]
    return U_mu_lr


def _initialize_spectroscopic_start(
    *, HARMONICA_ODD_HARMONICS, RORS_BASE, U_mu, bestfit_params_wl_df, detrend_type_multiwave,
    harmonica_spectro_parameterization, init_params, ld_prior_mode, ld_profile,
    ld_uniform_basis, num_lcs, spectro_ld_parameterization, surface_config, transit_engine,
):
    """Initialize channel limb darkening, harmonics, and systematics from white light."""
    _seed_surface_spectroscopic_init(init_params, surface_config, num_lcs)
    if ld_profile == 'quadratic' and (ld_prior_mode == 'uniform' or (
            ld_uniform_basis == 'kipping' and ld_prior_mode in {'gaussian', 'stellarprior'})):
        init_params.pop('u', None)
        init_params.update(_quadratic_uniform_initial_sites(U_mu, ld_uniform_basis))
    if ld_prior_mode == 'fixed_u1':
        init_params.pop('u', None)
        init_params['u2'] = jnp.asarray(U_mu)[..., 1]
    if ld_prior_mode == 'sing':
        init_params.pop('u', None)
        init_params['limb_l'] = jnp.asarray(U_mu)[:, 0]
        init_params['limb_delta'] = jnp.asarray(U_mu)[:, 1]
    if ld_profile == 'power2' and ld_prior_mode not in {'fixed', 'interpolated'}:
        init_params.pop('u', None)
        init_params.update(_power2_ld_initial_sites(U_mu, spectro_ld_parameterization, ld_prior_mode))
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
            init_params['delta_r'] = jnp.tile(init_delta_r[None, :], (num_lcs, 1))
        elif harmonica_spectro_parameterization == 'half_area':
            init_val = (
                bestfit_params_wl_df['a1'].values
                if 'a1' in bestfit_params_wl_df.columns
                else np.full_like(RORS_BASE, HARMONICA_INIT_ODD_COEFF)
            )
            init_area_radius, init_q = (
                _harmonica_coefficients_to_half_area_init(RORS_BASE, init_val)
            )
            init_params['rors'] = jnp.tile(
                jnp.atleast_1d(jnp.asarray(init_area_radius))[None, :], (num_lcs, 1),
            )
            init_params['q'] = jnp.tile(
                jnp.atleast_1d(jnp.asarray(init_q))[None, :], (num_lcs, 1),
            )
        else:
            for harmonic_name in HARMONICA_ODD_HARMONICS:
                init_val = bestfit_params_wl_df[harmonic_name].values[0] if harmonic_name in bestfit_params_wl_df.columns else HARMONICA_INIT_ODD_COEFF
                init_frac = _harmonica_coeff_array_to_frac(init_val, RORS_BASE)
                init_params[_harmonica_frac_site(harmonic_name)] = jnp.tile(
                    jnp.asarray(init_frac)[None, :], (num_lcs, 1),
                )
    if detrend_type_multiwave != 'none':
        init_params['c'] = jnp.clip(
            jnp.full(num_lcs, bestfit_params_wl_df['c'].values[0]), 0.900001, 1.099999,
        )
        if 'v' in bestfit_params_wl_df.columns:
             init_params['v'] = jnp.clip(
                 jnp.full(num_lcs, bestfit_params_wl_df['v'].values[0]), -0.099999, 0.099999,
             )

    if 'explinear' in detrend_type_multiwave:
        init_params['A'] = jnp.full(num_lcs, bestfit_params_wl_df['A'].values[0])
        if 'explinear_spectroscopic' not in detrend_type_multiwave:
            init_params['log_tau'] = jnp.full(
                num_lcs, jnp.log(bestfit_params_wl_df['tau'].values[0]),
            )

def _spectroscopic_ld(model, inputs, wavelengths, widths, bin_str):
    if joint_display.get() is not None:
        from .console import observation_label
        prefix = getattr(inputs, 'instrument_full_str', None)
        label = observation_label(prefix) if prefix else inputs.instrument
        info('')
        info(f'{label}: spectroscopy {grid_description(bin_str)}, {len(wavelengths)} channels, '
             'orbit fixed to the joint white-light posterior')
    sweep = getattr(inputs, 'execution_context', None)
    compute = lambda: _spectroscopic_ld_uncached(model, inputs, wavelengths, widths, bin_str)
    if sweep is None:
        return compute()
    prior_kind = model.ld_prior_mode if model.ld_prior_mode in {'sing', 'stellarprior'} else 'point'
    return sweep.coefficients((id(inputs.sld), model.ld_profile, prior_kind, wavelengths,
        widths, inputs.instrument, inputs.order, inputs.stellar_cfg, inputs.explicit_ld,
        inputs.flags if prior_kind == 'sing' else None), compute)


def _spectroscopic_ld_uncached(model, inputs, wavelengths, widths, bin_str):
    """Resolve exactly the LD grid consumed by either spectroscopic stage."""
    sigma, sing_coefficients = None, None
    if inputs.explicit_ld is not None:
        coefficients = inputs._explicit_ld_grid(wavelengths)
    elif model.ld_prior_mode == 'stellarprior':
        stellar_builder = (get_or_build_quadratic_ld_prior if model.ld_profile == 'quadratic'
                           else get_or_build_power2_ld_prior)
        coefficients, sigma = stellar_builder(
            inputs.stellar_cfg, wavelengths, widths, inputs.instrument,
            order=inputs.order, output_dir=inputs.output_dir, cache_label=bin_str,
        )
    else:
        coefficients = get_limb_darkening(
            inputs.sld, wavelengths, widths, inputs.instrument, order=inputs.order,
            ld_profile=model.ld_profile,
            ld_mu_min=inputs.stellar_cfg.get('ld_mu_min', 0.2) if model.ld_prior_mode == 'sing' else None,
        )
        if model.ld_prior_mode == 'sing':
            sing_coefficients = jnp.asarray(coefficients)
            coefficients, sigma = build_sing_ld_prior(coefficients, inputs.flags, inputs.stellar_cfg)
    return coefficients, sigma, sing_coefficients


def _spectroscopic_window_duration(geometry):
    """Conservative contact envelope over the ENTIRE channel radius prior.

    Eccentric orbits use a full-period window until an exact contact envelope
    is proven. No Gaussian shared geometry has a finite support envelope.
    """
    from models.harmonica.core import harmonica_duration_from_geometry
    if np.any(np.asarray(geometry.HARMONICA_ECC) != 0):
        return np.asarray(geometry.PERIOD_FIXED)
    return np.asarray(harmonica_duration_from_geometry(
        geometry.PERIOD_FIXED, geometry.A_RS_BASE, geometry.B_BASE,
        np.sqrt(.5), ecc=0., omega=geometry.HARMONICA_OMEGA))


def _spectroscopic_model(geometry, model, inputs, time, ld_coefficients,
                         init_params, joint_geometry, ld_mode, trend_mode, detrend_type, label):
    """Build a stage model and its coefficient-space fallback."""
    timing_flags = getattr(inputs, 'flags', {})
    free_t0 = timing_flags.get('spectro_free_t0', False)
    if not isinstance(free_t0, (bool, np.bool_)):
        raise ValueError('flags.spectro_free_t0 must be a boolean.')
    half_width = float(timing_flags.get('spectro_t0_half_width_s', 1800.0)) / 86400.0
    if free_t0:
        if model.transit_engine != 'jaxoplanet' or joint_geometry is not None:
            raise ValueError('spectro_free_t0 requires jaxoplanet and fixed white-light orbit geometry.')
        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError('spectro_t0_half_width_s must be finite and positive.')
        init_params['dt0'] = jnp.zeros_like(init_params['rors'])
    if ld_mode == 'fixed_u1' or (ld_mode == 'uniform' and getattr(model, 'ld_uniform_basis', 'uplus_uminus') == 'coefficients'):
        low, high = model._engine_spectro_kw.get('ld_uniform_coefficient_bounds', (0., 1.))
        site = 'u2' if ld_mode == 'fixed_u1' else 'u'
        if site in init_params:
            init_params[site] = jnp.clip(init_params[site], low + 1e-4*(high-low), high - 1e-4*(high-low))
    transit_window_indices = None
    if (
        model.transit_engine == 'jaxoplanet'
        and model.transit_window_optimization == 'auto'
        and joint_geometry is None
    ):
        transit_window_indices = inputs.build_transit_window_indices(
            np.asarray(time),
            np.asarray(geometry.PERIOD_FIXED),
            np.asarray(geometry.T0_BASE),
            _spectroscopic_window_duration(geometry) + (2 * half_width if free_t0 else 0),
        )
        debug(
            f"Jaxoplanet transit window ({label}): "
            f"{len(transit_window_indices)}/{time.size} active cadences."
        )

    if (model._engine_spectro_kw or {}).get('trend_basis') == 'batman_normalized':
        from models.spectro_trends import normalized_trend_initial_values
        normalized = normalized_trend_initial_values(np.asarray(time), init_params)
        init_params.update({name: value for name, value in normalized.items()
                            if name.startswith('trend_')})
    builder_kwargs = {
        'detrend_type': detrend_type,
        'ld_mode': ld_mode,
        'trend_mode': trend_mode,
        'n_planets': geometry.n_planets,
        **(
            {
                'transit_window_indices': transit_window_indices,
                'transit_grid_non_grazing': bool(
                    np.max(np.abs(np.asarray(geometry.B_BASE))) < 1.0 - np.sqrt(0.5),
                ),
                'transit_grid_outer_contact_safe': bool(
                    np.max(np.abs(np.asarray(geometry.B_BASE))) < 1.0 + np.sqrt(1e-05) - 1e-05,
                ),
                'joint_geometry': joint_geometry is not None,
            }
            if model.transit_engine == 'jaxoplanet' else {}
        ),
        **model._engine_spectro_kw,
    }
    if model.transit_engine == 'jaxoplanet':
        builder_kwargs['transit_grid'] = 'off'
    if free_t0:
        builder_kwargs['surface_config'] = {
            **model.surface_config, 'spectro_free_t0': True,
            'spectro_t0_half_width_days': half_width,
        }
    transit_separation = None
    if (model.transit_engine == 'jaxoplanet' and joint_geometry is None and not free_t0
            and model.surface_config.get('model') == 'transit'
            and not model.surface_config.get('spots')):
        from models.jaxoplanet.core import fixed_orbit_separation
        transit_separation = jax.vmap(lambda p, t0, a, b, e, w:
            fixed_orbit_separation(time, p, t0, a, b, e, jnp.rad2deg(w)))(
                jnp.atleast_1d(geometry.PERIOD_FIXED), jnp.atleast_1d(geometry.T0_BASE),
                jnp.atleast_1d(geometry.A_RS_BASE), jnp.atleast_1d(geometry.B_BASE),
                jnp.atleast_1d(geometry.HARMONICA_ECC), jnp.atleast_1d(geometry.HARMONICA_OMEGA))
        debug(f'Prepared one wavelength-independent Keplerian orbit for {label}.')
    surface_basis = None
    if ld_mode == 'fixed' and not free_t0:
        surface_basis = _prepare_fixed_surface_basis(
            model.surface_config, time, ld_coefficients, period=geometry.PERIOD_FIXED,
            t0=geometry.T0_BASE, a_rs=geometry.A_RS_BASE, b=geometry.B_BASE,
            rors=geometry.RORS_BASE, ecc=geometry.HARMONICA_ECC,
            omega=geometry.HARMONICA_OMEGA, ld_profile=model.ld_profile,
        )
        if surface_basis is not None:
            debug(f'Prepared exact fixed-geometry surface basis for {label} inference.')
    fit_model = _build_spectroscopic_model(model.create_vectorized_model, **builder_kwargs)
    # The post-fit time-shift response reads each channel's model light curve from the
    # 'obs' site, which the exact cadence-reduced likelihood replaces with factors.
    response_model = fit_model
    if free_t0 and builder_kwargs.get('cadence_reduction') == 'auto':
        response_model = _build_spectroscopic_model(
            model.create_vectorized_model, **{**builder_kwargs, 'cadence_reduction': 'off'})
    fallback_model = None
    fallback_init = None
    if (
        model.transit_engine == 'jaxoplanet'
        and model.ld_profile == 'power2'
        and ld_mode not in {'fixed', 'interpolated'}
        and model.spectro_ld_parameterization != 'coefficients'
    ):
        coefficient_builder_kwargs = dict(builder_kwargs)
        coefficient_builder_kwargs['ld_parameterization'] = 'coefficients'
        fallback_model = _build_spectroscopic_model(
        model.create_vectorized_model, **coefficient_builder_kwargs
        )
        fallback_init = dict(init_params)
        fallback_init.pop('ld_decorrelated', None)
        fallback_init.pop('ld_smooth', None)
        fallback_init.update(_power2_ld_initial_sites(ld_coefficients, 'coefficients'))

    return SimpleNamespace(
        model=fit_model, response_model=response_model, builder_kwargs=builder_kwargs,
        surface_basis=surface_basis, time=time,
        init_params=init_params, joint_geometry=joint_geometry, ld_mode=ld_mode,
        trend_mode=trend_mode, detrend_type=detrend_type, fallback_model=fallback_model,
        fallback_init=fallback_init, transit_window_indices=transit_window_indices,
        transit_separation=transit_separation, free_t0=bool(free_t0),
    )


def _sample_spectroscopic_stage(geometry, model, inputs, stage, fit, spectro_sampler,
                                flux, flux_error, model_kwargs, trend_names):
    """Run either channel stage with the same fallback and checkpoint policy."""
    if getattr(fit, 'free_t0', False):
        # Keep the original likelihood/LD/trend inputs for the post-fit response.
        fit.time_shift_model_kwargs = dict(model_kwargs)
        fit.time_shift_model_kwargs['likelihood_mask'] = (
            np.isfinite(np.asarray(flux)) & np.isfinite(np.asarray(flux_error))
            & (np.asarray(flux_error) > 0)
        )
    chunk_size = stage.chunk_size
    num_lcs = flux.shape[0]
    chunk_size = _resolve_spectro_joint_geometry_chunk_size(
        chunk_size, num_lcs, fit.joint_geometry is not None
    )
    chunk_size = _resolve_stage_vmap_width(
        inputs.flags,
        stage.name,
        chunk_size,
        sampler_backend=spectro_sampler,
        trend_inference=(
            'gaussian_marginalized'
            if fit.trend_mode == 'gaussian_marginalized'
            else 'sampled_uniform'
        ),
        num_cadences=int(fit.time.size),
        num_channels=int(num_lcs),
        active_transit_cadences=(
            int(len(fit.transit_window_indices))
            if fit.transit_window_indices is not None
            else int(fit.time.size)
        ),
        mcmc_kwargs=stage.mcmc_kwargs,
        sampler_kwargs=stage.nuts_kwargs,
        transit_engine=model.transit_engine,
        ld_profile=model.ld_profile,
        ld_mode=fit.ld_mode,
        detrend_type=fit.detrend_type,
        param_method=geometry.param_method,
        n_planets=geometry.n_planets,
        transit_window=model.transit_window_optimization,
        transit_grid_non_grazing=fit.builder_kwargs.get('transit_grid_non_grazing', False),
        transit_grid_outer_contact_safe=fit.builder_kwargs.get(
            'transit_grid_outer_contact_safe', False
        ),
    )

    orbit_description = ('constrained by' if fit.joint_geometry is not None else 'fixed to')
    if joint_display.get() is None:
        info(f'Spectroscopy {grid_description(stage.bin_str)}: {len(stage.wavelengths)} channels, '
             f'orbit {orbit_description} the white-light posterior')
    samples = _run_sampling_stage(
        fit.model,
        stage.key_mcmc,
        fit.time,
        flux_error,
        flux,
        fit.init_params,
        execution_context=getattr(inputs, 'execution_context', None),
        nuts_kwargs=stage.nuts_kwargs,
        mcmc_kwargs=stage.mcmc_kwargs,
        use_chunked=(
            chunk_size is not None
            or spectro_sampler in {'independent_nuts', 'independent_hmc'}
        ),
        chunk_size=chunk_size,
        chunk_mode=inputs.chunk_mode,
        parallel_job_count=inputs.chunk_parallel_job_count,
        parallel_job_index=inputs.chunk_parallel_job_index,
        output_dir=inputs.output_dir,
        checkpoint_prefix=_harmonica_checkpoint_prefix(
            f'{inputs.instrument_full_str}_{stage.bin_str}', model.transit_engine,
            model.harmonica_spectro_parameterization,
        ),
        checkpoint_signature={
            **({'sweep_model_identity': inputs.execution_context.science_identity()}
               if getattr(inputs, 'execution_context', None) is not None else {}),
            **({'spectro_free_t0': True,
                'spectro_t0_half_width_days': fit.builder_kwargs['surface_config']['spectro_t0_half_width_days']}
               if getattr(fit, 'free_t0', False) else {}),
            'stage': stage.resolution,
            'spectro_joint_geometry': fit.joint_geometry is not None,
            'spectro_joint_geometry_prior_inflation': (
                None if fit.joint_geometry is None
                else fit.joint_geometry['prior_inflation']
            ),
            'transit_engine': model.transit_engine,
            'ld_profile': model.ld_profile,
            'ld_mode': fit.ld_mode,
            'trend_inference': model.trend_inference,
            'detrend_type': fit.detrend_type,
            'param_method': geometry.param_method,
            'whitelight_geometry_estimator': geometry.whitelight_geometry_estimator,
            'whitelight_geometry_handoff_fingerprint': (
                geometry.wl_geometry_handoff['artifact_fingerprint_sha256']
            ),
            'jaxoplanet_kernel': model.jaxoplanet_kernel,
            'transit_window_optimization': model.transit_window_optimization,
            'spectro_cadence_reduction': model._engine_spectro_kw.get(
                'cadence_reduction', 'off'
            ),
            'spectro_transit_grid': model._engine_spectro_kw.get('transit_grid', 'off'),
            'spectro_transit_grid_nodes': model._engine_spectro_kw.get('transit_grid_nodes'),
            'surface_config': model.surface_config,
            'harmonica_max_order': inputs.max_harmonic_order,
            'harmonica_spectro_parameterization': model.harmonica_spectro_parameterization,
            'harmonica_spectro_fit_jitter': model.harmonica_spectro_fit_jitter,
            'harmonica_spectro_odd_frac_sigma': model.harmonica_spectro_odd_frac_sigma,
        },
        sampler_backend=spectro_sampler,
        channel_varying_kwargs=(
            JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
            if model.transit_engine == 'jaxoplanet'
            else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS
        ),
        spectro_min_depth_ess=model.spectro_min_depth_ess,
        spectro_max_divergences=model.spectro_max_divergences,
        stacking_metadata={
            'stage_kind': stage.resolution,
            'stage_label': f"{inputs.instrument_full_str}_{stage.bin_str}_{stage.resolution}",
            'wavelength': np.asarray(stage.wavelengths),
            'wavelength_err': np.asarray(stage.wavelength_err),
        },
        adaptive_fallback_model=fit.fallback_model,
        adaptive_fallback_init_params=fit.fallback_init,
        **model_kwargs,
    )
    if samples is None:
        debug(f"{stage.resolution.replace('_', '-').capitalize()} chunk job complete.")
        return
    from .sampling import _spectro_failed_lanes
    failed, _ = _spectro_failed_lanes(samples, None, model.spectro_min_depth_ess,
        model.spectro_max_divergences, surface_science_sites=getattr(fit.model, 'surface_science_sites', None))
    failed = getattr(samples, 'console_failed_indices', failed)
    spectroscopy_verdict(stage.wavelengths, failed,
        extra_draws=getattr(samples, 'console_extra_draws', 0),
        refitted=getattr(samples, 'console_refitted', 0))
    sweep = getattr(inputs, 'execution_context', None)
    if sweep is not None:
        sweep.remember(stage.name, samples, law=model.ld_profile)
    if trend_names is not None:
        samples = materialize_marginalized_trend_samples(samples, trend_names, stage.key_map)

    return samples


def _spectroscopic_trends(inputs, detrend_type, time, num_lcs, model_kwargs, init_params,
                           gp_trend, spot_trend, spot_trend2, jump_trend, fixed_tau):
    """Align white-light templates and initialize their channel amplitudes."""
    templates = {'spot_trend': spot_trend, 'spot_trend2': spot_trend2, 'jump_trend': jump_trend}
    names = []
    if 'gp_spectroscopic' in detrend_type:
        model_kwargs['gp_trend'] = gp_trend
        init_params['A_gp'] = jnp.ones(num_lcs)
    if '2spot_spectroscopic' in detrend_type:
        if spot_trend is None or spot_trend2 is None:
            raise ValueError('2spot_spectroscopic requires WL spot and spot2 trends.')
        names += [('spot_trend', 'A_spot'), ('spot_trend2', 'A_spot2')]
    elif _has_single_spot_spectroscopic(detrend_type):
        if spot_trend is None:
            raise ValueError(f'{detrend_type} requires WL spot to build spot_trend.')
        names.append(('spot_trend', 'A_spot'))
    if 'linear_discontinuity_spectroscopic' in detrend_type:
        if jump_trend is None:
            raise ValueError('linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.')
        names.append(('jump_trend', 'A_jump'))
    for name, amplitude in names:
        values = templates[name]
        if len(values) != len(time):
            values = inputs._align_trend_to_time(values, inputs.wl_time_good, np.array(time))
        templates[name] = model_kwargs[name] = values
        init_params[amplitude] = jnp.ones(num_lcs)
    exp_trend = None
    if 'explinear_spectroscopic' in detrend_type:
        exp_trend = inputs._align_trend_to_time(inputs.exp_trend, inputs.wl_time_good, np.asarray(time))
        model_kwargs.update(exp_trend=exp_trend, fixed_tau=fixed_tau)
    return templates['spot_trend'], templates['spot_trend2'], templates['jump_trend'], exp_trend


def _prepare_trend_likelihood(model, inputs, fit, flux, flux_error, model_kwargs):
    """Prepare the optional exact reductions of the linear trend likelihood."""
    num_lcs = flux.shape[0]
    if getattr(fit, 'transit_separation', None) is not None:
        model_kwargs['transit_separation'] = fit.transit_separation
    # Short light curves use the reduction only for the fused quadratic score, and only
    # with the CUDA kernel; with the JAX score alone it is slower there.
    fused_quadratic = (
        fit.trend_mode == 'free' and model.ld_profile == 'quadratic'
        and bool(getattr(inputs, 'flags', {}).get('spectro_free_t0', False))
    )
    if fused_quadratic and int(fit.time.size) <= 5000:
        from models.jaxoplanet.quadratic_cuda import accelerated
        fused_quadratic = accelerated()
    if (
        model.transit_engine == 'jaxoplanet'
        and model._engine_spectro_kw.get('cadence_reduction') == 'auto'
        and (int(fit.time.size) > 5000 or fused_quadratic)
        and (fit.trend_mode == 'free' or (
            fit.trend_mode == 'gaussian_marginalized'
            and fit.detrend_type in {'constant', 'linear', 'quadratic'}))
        and fit.transit_window_indices is not None
        and linear_spectro_trend_coefficient_names(fit.detrend_type) is not None
    ):
        trend_names, trend_design = (
            build_linear_spectro_trend_design(
                fit.detrend_type, fit.time, exp_trend=model_kwargs.get('exp_trend'),
                spot_trend=model_kwargs.get('spot_trend'),
                spot_trend2=model_kwargs.get('spot_trend2'),
                jump_trend=model_kwargs.get('jump_trend'),
            )
        )
        reference_beta = np.column_stack([
            np.asarray(fit.init_params.get(name, np.zeros(num_lcs, dtype=np.float64)))
            for name in trend_names
        ])
        model_kwargs['trend_design'] = trend_design
        model_kwargs.update(build_linear_oot_statistics(
            fit.time, flux, flux_error, fit.transit_window_indices, trend_design,
            reference_beta,
        ))

    trend_names = None
    if fit.trend_mode == 'gaussian_marginalized':
        (model_kwargs['trend_prior_mean'], model_kwargs['trend_prior_scale'], trend_names) = _build_gaussian_trend_prior(
            fit.detrend_type, num_lcs, fit.init_params, inputs.flags,
        )
        # These coefficients are integrated out exactly.  Keep only
        # genuinely nonlinear trend sites (for example log_tau).
        for trend_name in trend_names:
            fit.init_params.pop(trend_name, None)

    return trend_names


def _evaluate_spectroscopic_posterior(geometry, model, fit, coefficients, samples,
                                     flux, gp_trend, spot_trend, spot_trend2, jump_trend, exp_trend):
    """Reconstruct the median model for channel residuals and saved products."""
    params, final_in_axes, selected_kernel = _spectroscopic_posterior_model(
        A_RS_BASE=geometry.A_RS_BASE, B_BASE=geometry.B_BASE, DURATION_BASE=geometry.DURATION_BASE,
        HARMONICA_ECC=geometry.HARMONICA_ECC, HARMONICA_ODD_HARMONICS=geometry.HARMONICA_ODD_HARMONICS,
        HARMONICA_OMEGA=geometry.HARMONICA_OMEGA, PERIOD_FIXED=geometry.PERIOD_FIXED, T0_BASE=geometry.T0_BASE,
        detrend_type_multiwave=fit.detrend_type,
        harmonica_spectro_parameterization=model.harmonica_spectro_parameterization,
        jaxoplanet_kernel=model.jaxoplanet_kernel, joint_geometry=fit.joint_geometry,
        ld_coefficients=coefficients, ld_profile=model.ld_profile, param_method=geometry.param_method,
        samples=samples, surface_config=model.surface_config, time=fit.time,
        transit_engine=model.transit_engine,
        transit_window_optimization=model.transit_window_optimization,
    )

    num_lcs = flux.shape[0]
    model_all = evaluate_channels_sequentially(
        params, final_in_axes, fit.time, num_lcs, fit.detrend_type, selected_kernel,
        model.transit_engine, model.surface_config, model.jaxoplanet_kernel, model.ld_profile,
        gp_trend, spot_trend, spot_trend2, jump_trend, exp_trend,
        _attach_surface_eval_metadata, _has_single_spot_spectroscopic,
    )

    return params, model_all


def _save_spectroscopic_products(geometry, model, inputs, stage, fit, samples,
                                  time, flux, flux_error, params, plot_params, templates, model_all, initial_ld):
    """Write either resolution's science tables and diagnostic figures."""
    from . import diagnostic_figures as figures
    from .outputs import _trend_from_params_np
    paths = OutputPaths(inputs.output_dir, inputs.instrument_full_str)
    grid = stage.bin_str
    prefix = paths.stem(grid)
    gp_trend, spot_trend, spot_trend2, jump_trend, exp_trend = templates
    total_error = np.nanmedian(samples['total_error'], axis=0)
    orbit = dict(period=geometry.PERIOD_FIXED)
    wavelengths = np.asarray(stage.wavelengths)
    label = inputs.instrument_full_str + ' | ' + figures.grid_title(grid)
    if model.surface_config['model'] != 'eclipse':
        figures.spectrum(wavelengths, samples['rors'], paths.figure(grid, 'spectrum'), label=label,
                         wavelength_err=stage.wavelength_err)
        save_results(wavelengths, stage.wavelength_err, samples, paths.table(grid, 'spectrum'))
    save_surface_results(wavelengths, stage.wavelength_err, samples, paths.table(grid, 'spectrum'),
                         plots=inputs.plots_mode)
    if getattr(fit, 'free_t0', False):
        from .time_shift import save_time_shift_products
        from .time_shift_response import spectroscopic_responses
        responses, response_seconds = spectroscopic_responses(
            fit, samples, flux_error, ld_profile=model.ld_profile,
        )
        save_time_shift_products(
            wavelengths, stage.wavelength_err, samples, prefix, geometry,
            responses=responses, response_seconds=response_seconds,
        )
    params_frame, _ = save_detailed_fit_results(
        time, flux, flux_error, stage.wavelengths, stage.wavelength_err, samples,
        params, orbit, fit.detrend_type, prefix, total_error,
        gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend,
        joint_geometry=None if fit.joint_geometry is None else fit.joint_geometry['summary'],
    )
    if inputs.plots_mode == 'full':
        trends = np.asarray([_trend_from_params_np(
            fit.detrend_type, np.asarray(time), plot_params, idx=index,
            gp_trend=gp_trend, spot_trend=spot_trend, spot_trend2=spot_trend2,
            jump_trend=jump_trend, exp_trend=exp_trend) for index in range(len(wavelengths))])
        figures.channel_diagnostics(time, flux, model_all, wavelengths, trends,
                                    paths.figure(grid, 'lightcurves'), label=label, event=model.surface_config['model'],
                                    t0=figures.event_epoch(geometry.T0_BASE, geometry.PERIOD_FIXED, model.surface_config["model"]))
        figures.residual_map(time, wavelengths, np.asarray(flux)-model_all,
                             paths.figure(grid, 'residual_map'), label=label, event=model.surface_config['model'], wavelength_err=stage.wavelength_err,
                             t0=figures.event_epoch(geometry.T0_BASE, geometry.PERIOD_FIXED, model.surface_config["model"]))
        figures.limb_darkening(wavelengths, params_frame, initial_ld,
                               paths.figure(grid, 'limb_darkening'), label=label, law=model.ld_profile)
        white_path = paths.table('whitelight', 'lightcurve')
        white = pd.read_csv(white_path) if os.path.exists(white_path) else None
        white_residual = (None if white is None else
                          white.loc[white.is_outlier == 0, 'residual'].to_numpy())
        figures.noise(np.asarray(flux)-model_all, paths.figure(grid, 'noise'),
                      label=label, white=white_residual, wavelengths=wavelengths)
    if model.transit_engine == 'harmonica' and _has_harmonica_odd_samples(samples):
        save_harmonica_limb_products(
            wavelengths=wavelengths, wavelength_err=stage.wavelength_err,
            rors_samples=_harmonica_model_radius_samples(samples, model.harmonica_spectro_parameterization),
            harmonic_samples=_harmonica_sample_payload(samples), csv_path=paths.table(grid, 'limbs'),
            limb_spectrum_path=paths.figure(grid, 'limbs') if inputs.plots_mode == 'full' else None,
            transmission_strings_path=paths.figure(grid, 'limb_profiles') if inputs.plots_mode == 'full' else None,
            posterior_strings_path=paths.figure(grid, 'limb_outline') if inputs.plots_mode == 'full' else None,
            posterior_samples_path=paths.table(grid, 'limb_samples', 'npz'),
            title_prefix=inputs.instrument_full_str + ' | ' + figures.grid_title(stage.bin_str),
        )


def run_low_resolution_stage(
    geometry, model, inputs, *, cfg, harmonica_lr_nuts_kwargs, jaxoplanet_lr_nuts_kwargs,
    jump_trend, key_map_lr, key_mcmc_lr, lowres_mcmc_kwargs, lr_artifact_stem, lr_bin_str,
    need_lowres_analysis, spectro_sampler, spectroscopic_sigma, spot_trend, spot_trend2, valid,
    vmap_chunk_size_lr,
):
    if need_lowres_analysis:
        # Compute the exact LD inputs before deciding whether a previous LR fit
        # is reusable. This binds cache validity to the coefficients actually
        # consumed by the likelihood, including changes to external LD grids.
        U_mu_lr, U_sigma_lr, sing_model_c_lr = _spectroscopic_ld(
            model, inputs, inputs.dataset.wavelengths_lr, inputs.dataset.wavelengths_err_lr, lr_bin_str,
        )

        lr_mask_path = (
            state_path(inputs.output_dir, f"{lr_artifact_stem}_spectroscopic_outlier_mask.npy")
        )
        lr_params_path = OutputPaths(inputs.output_dir, inputs.instrument_full_str).table(lr_bin_str, "params", "csv")
        lr_limb_csv_path = OutputPaths(inputs.output_dir, inputs.instrument_full_str).table(lr_bin_str, "limbs", "csv")
        lr_limb_samples_path = (
            OutputPaths(inputs.output_dir, inputs.instrument_full_str).table(lr_bin_str, "limb_samples", "npz")
        )
        lr_manifest_path = (
            state_path(inputs.output_dir, f"{lr_artifact_stem}_artifacts.manifest.json")
        )
        lr_artifact_fingerprint = _science_artifact_fingerprint(
            "low_resolution",
            {
                "config": cfg,
                "time": inputs.dataset.time,
                "flux": inputs.dataset.flux_lr,
                "flux_err": inputs.dataset.flux_err_lr,
                "wavelength": inputs.dataset.wavelengths_lr,
                "wavelength_err": inputs.dataset.wavelengths_err_lr,
                "white_light_mask": inputs.wl_mad_mask,
                "white_light_fit": inputs.bestfit_params_wl_df.to_dict(orient="list"),
                "white_light_geometry_handoff_fingerprint": (
                    geometry.wl_geometry_handoff["artifact_fingerprint_sha256"]
                ),
                "transit_engine": model.transit_engine,
                "ld_profile": model.ld_profile,
                "ld_prior_mode": model.ld_prior_mode,
                "ld_coefficients": U_mu_lr,
                "ld_uncertainties": U_sigma_lr,
                "detrending_type": model.detrending_type,
                "harmonica_spectro_parameterization": (
                    model.harmonica_spectro_parameterization
                ),
                "spectroscopic_sigma": spectroscopic_sigma,
            },
        )
        required_lr_limb_products_exist = bool(
            model.transit_engine != 'harmonica'
            or not geometry.HARMONICA_ODD_HARMONICS
            or (
                os.path.exists(lr_limb_csv_path)
                and os.path.exists(lr_limb_samples_path)
            )
        )
        lr_artifact_set = ArtifactSet(
            stage='low_resolution', manifest_path=lr_manifest_path,
            fingerprint=lr_artifact_fingerprint, required_paths=(lr_mask_path, lr_params_path),
        )
        lr_cache_valid = lr_artifact_set.is_reusable(
            extra_condition=required_lr_limb_products_exist
        )
        if lr_cache_valid:
            info(f'Reusing the previous spectroscopy fit {grid_description(lr_bin_str)}.')
            time_mask = np.load(lr_mask_path)
            valid = ~time_mask
        elif os.path.exists(lr_mask_path) or os.path.exists(lr_params_path):
            info('The previous low-resolution fit does not match this run; refitting it.')
            debug(
                "Existing low-resolution products are incomplete, unverified, "
                "or do not match the current data/LD target; recomputing them."
            )

    if need_lowres_analysis and valid is None:
        if model.uses_surface_model:
            jax.clear_caches()
            debug("Released white-light JAX executables before surface spectroscopy.")
        debug(f"\n--- Running Low-Resolution Analysis (Binned to {lr_bin_str}) ---")
        joint_geometry_lr, spectro_sampler, jaxoplanet_lr_nuts_kwargs = (
            _prepare_joint_geometry(
                inputs.flags, stage_name='lowres', transit_engine=model.transit_engine,
                spectro_sampler=spectro_sampler, nuts_kwargs=jaxoplanet_lr_nuts_kwargs,
                bestfit_params_wl_df=inputs.bestfit_params_wl_df,
                wl_geometry_handoff=geometry.wl_geometry_handoff,
                param_method=geometry.param_method,
            )
        )
        time_lr = jnp.array(inputs.dataset.time[inputs.spec_good_mask])
        flux_lr = jnp.array(inputs.dataset.flux_lr[:, inputs.spec_good_mask])
        flux_err_lr = jnp.array(inputs.dataset.flux_err_lr[:, inputs.spec_good_mask])
        num_lcs_lr = int(inputs.dataset.flux_err_lr.shape[0])

        if 'gp' in model.detrending_type:
            gp_df = pd.read_csv(state_path(inputs.output_dir, f'{inputs.instrument_full_str}_whitelight_GP_database.csv'))
            gp_trend_raw = gp_df['gp_trend'].values
            if len(gp_trend_raw) == len(inputs.wl_mad_mask):
                gp_trend_raw = gp_trend_raw[~inputs.wl_mad_mask]
            gp_trend = jnp.array(inputs._align_trend_to_time(gp_trend_raw, inputs.wl_time_good, np.array(time_lr)))
        else:
            gp_trend = None
        detrend_type_multiwave = _spectro_detrend_type(
            model.detrending_type, model.spectro_fixed_timescale_trends
        )

        debug(f"Low-res: {num_lcs_lr} light curves.")
        DEPTHS_BASE_LR = jnp.tile(geometry.DEPTH_BASE, (num_lcs_lr, 1))

        init_params_lr = {
            "u": _clip_ld_initial_values(U_mu_lr, model.ld_profile, 'low-res'),
            "rors": jnp.tile(geometry.RORS_BASE, (num_lcs_lr, 1))
        }
        if model.transit_engine == 'jaxoplanet':
            init_params_lr['log_jitter'] = jnp.log(
                model._engine_spectro_kw.get('jitter_prior_center', .5)
                * jnp.nanmedian(flux_err_lr, axis=1))
        _initialize_spectroscopic_start(
            HARMONICA_ODD_HARMONICS=geometry.HARMONICA_ODD_HARMONICS, RORS_BASE=geometry.RORS_BASE, U_mu=U_mu_lr,
            bestfit_params_wl_df=inputs.bestfit_params_wl_df,
            detrend_type_multiwave=detrend_type_multiwave,
            harmonica_spectro_parameterization=model.harmonica_spectro_parameterization,
            init_params=init_params_lr, ld_prior_mode=model.ld_prior_mode, ld_profile=model.ld_profile,
            ld_uniform_basis=model.ld_uniform_basis, num_lcs=num_lcs_lr,
            spectro_ld_parameterization=model.spectro_ld_parameterization,
            surface_config=model.surface_config, transit_engine=model.transit_engine,
        )

        lr_trend_mode = (
            'gaussian_marginalized'
            if model.trend_inference == 'gaussian_marginalized'
            else 'free'
        )
        lr_ld_mode = model.ld_prior_mode

        fit_lr = _spectroscopic_model(
            geometry, model, inputs, time_lr, U_mu_lr, init_params_lr, joint_geometry_lr,
            lr_ld_mode, lr_trend_mode, detrend_type_multiwave, 'low-resolution',
        )
        lr_model_builder_kwargs = fit_lr.builder_kwargs
        spot_basis_lr = fit_lr.surface_basis
        lr_adaptive_fallback_model, lr_adaptive_fallback_init = fit_lr.fallback_model, fit_lr.fallback_init

        model_run_args_lr = {
            'mu_duration': geometry.DURATION_BASE,
            'mu_t0': geometry.T0_BASE,
            'mu_b': geometry.B_BASE,
            'mu_depths': DEPTHS_BASE_LR,
            'PERIOD': geometry.PERIOD_FIXED,
        }
        if model.transit_engine == 'jaxoplanet':
            model_run_args_lr['precomputed_yerr_per_lc'] = jnp.nanmedian(flux_err_lr, axis=1)
            if spot_basis_lr is not None:
                model_run_args_lr['surface_basis_data'] = spot_basis_lr
        if model.transit_engine == 'harmonica':
            model_run_args_lr['mu_cos_i'] = geometry.COSI_BASE
            model_run_args_lr['harmonica_a_rs'] = geometry.A_RS_BASE
            model_run_args_lr['harmonica_ecc'] = geometry.HARMONICA_ECC
            model_run_args_lr['harmonica_omega'] = geometry.HARMONICA_OMEGA
        else:
            model_run_args_lr['rors_reference'] = geometry.RORS_BASE
            model_run_args_lr['mu_a_rs'] = geometry.A_RS_BASE
            model_run_args_lr['mu_ecc'] = geometry.HARMONICA_ECC
            model_run_args_lr['mu_omega'] = geometry.HARMONICA_OMEGA
        if joint_geometry_lr is not None:
            model_run_args_lr.update(joint_geometry_lr['model_kwargs'])
            init_params_lr.update(joint_geometry_lr['init_params'])

        if lr_ld_mode in {'fixed', 'fixed_u1'}:
            model_run_args_lr['ld_fixed'] = U_mu_lr
        elif lr_ld_mode in {'gaussian', 'stellarprior', 'sing'}:
            model_run_args_lr['mu_u_ld'] = U_mu_lr
            if ((lr_ld_mode == 'stellarprior') or lr_ld_mode == 'sing') and U_sigma_lr is not None:
                model_run_args_lr['sigma_u_ld'] = U_sigma_lr
        elif lr_ld_mode == 'uniform':
            pass
        else:
            raise ValueError(f"Unknown ld_prior mode: {lr_ld_mode}")

        spot_trend, spot_trend2, jump_trend, exp_trend_lr = _spectroscopic_trends(
            inputs, detrend_type_multiwave, time_lr, num_lcs_lr, model_run_args_lr, init_params_lr,
            gp_trend, spot_trend, spot_trend2, jump_trend, model.fixed_tau_spectro,
        )
        trend_names_lr = _prepare_trend_likelihood(
            model, inputs, fit_lr, flux_lr, flux_err_lr, model_run_args_lr,
        )

        # Sing et al. step 1: calibrate a single gray stellar offset from a
        # separate free-quadratic fit to the existing coarse channel set.
        if model.ld_prior_mode == 'sing':
            U_mu_lr = _calibrate_sing_offsets(
                U_mu_lr=U_mu_lr, create_vectorized_model=model.create_vectorized_model, data=inputs.data,
                flags=inputs.flags, flux_err_lr=flux_err_lr, flux_lr=flux_lr,
                harmonica_spectro_parameterization=model.harmonica_spectro_parameterization,
                init_params_lr=init_params_lr, instrument_full_str=inputs.instrument_full_str,
                jaxoplanet_lr_nuts_kwargs=jaxoplanet_lr_nuts_kwargs,
                joint_geometry_lr=joint_geometry_lr, key_mcmc_lr=key_mcmc_lr,
                lr_bin_str=lr_bin_str, lr_model_builder_kwargs=lr_model_builder_kwargs,
                model_run_args_lr=model_run_args_lr, num_lcs_lr=num_lcs_lr,
                output_dir=inputs.output_dir, sing_model_c_lr=sing_model_c_lr,
                stellar_cfg=inputs.stellar_cfg, time_lr=time_lr, transit_engine=model.transit_engine,
                vmap_chunk_size_lr=vmap_chunk_size_lr, wl_geometry_handoff=geometry.wl_geometry_handoff,
            )

        stage = SimpleNamespace(
            name='lowres', resolution='low_resolution', bin_str=lr_bin_str, artifact_stem=lr_artifact_stem,
            key_mcmc=key_mcmc_lr, key_map=key_map_lr, chunk_size=vmap_chunk_size_lr,
            nuts_kwargs=harmonica_lr_nuts_kwargs if model.transit_engine == 'harmonica' else jaxoplanet_lr_nuts_kwargs,
            mcmc_kwargs=lowres_mcmc_kwargs, wavelengths=inputs.dataset.wavelengths_lr,
            wavelength_err=inputs.dataset.wavelengths_err_lr,
        )
        samples_lr = _sample_spectroscopic_stage(
            geometry, model, inputs, stage, fit_lr, spectro_sampler,
            flux_lr, flux_err_lr, model_run_args_lr, trend_names_lr,
        )
        if samples_lr is None:
            return

        map_params_lr, model_all = _evaluate_spectroscopic_posterior(
            geometry, model, fit_lr, U_mu_lr, samples_lr, flux_lr,
            gp_trend, spot_trend, spot_trend2, jump_trend, exp_trend_lr,
        )

        residuals = flux_lr - model_all
        _lr_dt_sec = float(np.nanmedian(np.diff(np.array(time_lr)))) * 86400.0
        save_noise_binning_data(
            residuals, state_path(inputs.output_dir, f"{lr_artifact_stem}_noise.csv"),
        )

        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)
        # A threshold of 0 (the default) clips nothing.
        point_mask = (np.abs(residuals - medians) > spectroscopic_sigma * sigmas if spectroscopic_sigma
                      else np.zeros(residuals.shape, dtype=bool))
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
        if model.transit_engine == 'jaxoplanet':
            map_params_lr = _attach_jaxoplanet_eval_metadata(
                map_params_lr, time_lr, jaxoplanet_kernel=model.jaxoplanet_kernel,
                ld_profile=model.ld_profile,
                transit_window_optimization=model.transit_window_optimization,
            )
        plot_params_lr = dict(map_params_lr)
        if spot_basis_lr is not None:
            plot_params_lr['_surface_basis'] = _surface_basis_on_time_mask(spot_basis_lr, valid)
            plot_params_lr['_surface_basis_time'] = jnp.asarray(time_lr)

        _save_spectroscopic_products(
            geometry, model, inputs, stage, fit_lr, samples_lr, time_lr, flux_lr,
            flux_err_lr, map_params_lr, plot_params_lr,
            (gp_trend_lr, spot_trend_lr, spot_trend2_lr, jump_trend_lr, exp_trend_lr_save),
            np.asarray(model_all)[:, valid], U_mu_lr,
        )
        lr_artifact_set.write_manifest()
    return (valid, spot_trend, spot_trend2, jump_trend,)


def run_high_resolution_stage(
    geometry, model, inputs, *, harmonica_hr_nuts_kwargs, highres_mcmc_kwargs,
    hr_artifact_stem, hr_bin_str, jaxoplanet_hr_nuts_kwargs, jump_trend, key_map_hr,
    key_mcmc_hr, spectro_sampler, spot_trend, spot_trend2, valid, vmap_chunk_size_hr,
):
    if model.uses_surface_model:
        jax.clear_caches()
        debug("Released low-resolution JAX executables before high-resolution surface inference.")
    debug(f"\n--- Running High-Resolution Analysis (Binned to {hr_bin_str}) ---")
    joint_geometry_hr, spectro_sampler, jaxoplanet_hr_nuts_kwargs = (
        _prepare_joint_geometry(
            inputs.flags, stage_name='highres', transit_engine=model.transit_engine,
            spectro_sampler=spectro_sampler, nuts_kwargs=jaxoplanet_hr_nuts_kwargs,
            bestfit_params_wl_df=inputs.bestfit_params_wl_df,
            wl_geometry_handoff=geometry.wl_geometry_handoff,
            param_method=geometry.param_method,
        )
    )
    time_hr = jnp.array(inputs.dataset.time[inputs.spec_good_mask])
    flux_hr = jnp.array(inputs.dataset.flux_hr[:, inputs.spec_good_mask])
    flux_err_hr = jnp.array(inputs.dataset.flux_err_hr[:, inputs.spec_good_mask])

    if 'gp' in model.detrending_type:
        gp_df = pd.read_csv(state_path(inputs.output_dir, f'{inputs.instrument_full_str}_whitelight_GP_database.csv'))
        gp_trend_raw = gp_df['gp_trend'].values
        if len(gp_trend_raw) == len(inputs.wl_mad_mask):
            gp_trend_raw = gp_trend_raw[~inputs.wl_mad_mask]
        gp_trend = jnp.array(inputs._align_trend_to_time(gp_trend_raw, inputs.wl_time_good, np.array(time_hr)))
    else:
        gp_trend = None
    detrend_type_multiwave = _spectro_detrend_type(
        model.detrending_type, model.spectro_fixed_timescale_trends
    )

    if valid is not None:
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = inputs._align_trend_to_time(spot_trend, inputs.wl_time_good, np.array(time_hr))
        if spot_trend2 is not None and len(spot_trend2) != len(time_hr):
            spot_trend2 = inputs._align_trend_to_time(spot_trend2, inputs.wl_time_good, np.array(time_hr))
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = inputs._align_trend_to_time(jump_trend, inputs.wl_time_good, np.array(time_hr))
        if gp_trend is not None and len(gp_trend) != len(time_hr):
            gp_trend = inputs._align_trend_to_time(gp_trend, inputs.wl_time_good, np.array(time_hr))

        time_hr = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr = flux_err_hr[:, valid]
        if gp_trend is not None: gp_trend = gp_trend[valid]
        if spot_trend is not None: spot_trend = spot_trend[valid]
        if spot_trend2 is not None: spot_trend2 = spot_trend2[valid]
        if jump_trend is not None: jump_trend = jump_trend[valid]

    num_lcs_hr = flux_err_hr.shape[0]
    DEPTHS_BASE_HR = jnp.tile(geometry.DEPTH_BASE, (num_lcs_hr, 1))

    hr_ld_mode = model.ld_prior_mode
    if model.trend_inference == 'gaussian_marginalized':
        hr_trend_mode = 'gaussian_marginalized'
    else:
        hr_trend_mode = 'free'

    model_run_args_hr = {}
    wl_hr = np.array(inputs.dataset.wavelengths_hr)

    if hr_ld_mode in {'fixed', 'fixed_u1', 'gaussian', 'stellarprior', 'sing', 'uniform'}:
        U_mu_hr_init, U_sigma_hr_init, _ = _spectroscopic_ld(
            model, inputs, wl_hr, inputs.dataset.wavelengths_err_hr, hr_bin_str,
        )
        if hr_ld_mode in {'fixed', 'fixed_u1'}:
            model_run_args_hr['ld_fixed'] = U_mu_hr_init
        elif hr_ld_mode in {'gaussian', 'stellarprior', 'sing'}:
            model_run_args_hr['mu_u_ld'] = U_mu_hr_init
            if ((hr_ld_mode == 'stellarprior') or hr_ld_mode == 'sing') and U_sigma_hr_init is not None:
                model_run_args_hr['sigma_u_ld'] = U_sigma_hr_init
        elif hr_ld_mode == 'uniform':
            pass
        else:
            raise ValueError(f"Unknown ld_prior mode: {hr_ld_mode}")

    model_run_args_hr['mu_duration'] = geometry.DURATION_BASE
    model_run_args_hr['mu_t0'] = geometry.T0_BASE
    model_run_args_hr['mu_b'] = geometry.B_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = geometry.PERIOD_FIXED
    if model.transit_engine == 'jaxoplanet':
        model_run_args_hr['precomputed_yerr_per_lc'] = jnp.nanmedian(flux_err_hr, axis=1)
    if model.transit_engine == 'harmonica':
        model_run_args_hr['mu_cos_i'] = geometry.COSI_BASE
        model_run_args_hr['harmonica_a_rs'] = geometry.A_RS_BASE
        model_run_args_hr['harmonica_ecc'] = geometry.HARMONICA_ECC
        model_run_args_hr['harmonica_omega'] = geometry.HARMONICA_OMEGA
    else:
        model_run_args_hr['rors_reference'] = geometry.RORS_BASE
        model_run_args_hr['mu_a_rs'] = geometry.A_RS_BASE
        model_run_args_hr['mu_ecc'] = geometry.HARMONICA_ECC
        model_run_args_hr['mu_omega'] = geometry.HARMONICA_OMEGA

    init_params_hr = {
        "rors": jnp.tile(geometry.RORS_BASE, (num_lcs_hr, 1)),
        "u": _clip_ld_initial_values(U_mu_hr_init, model.ld_profile, 'high-res'),
    }
    if model.transit_engine == 'jaxoplanet':
        init_params_hr['log_jitter'] = jnp.log(
            model._engine_spectro_kw.get('jitter_prior_center', .5)
            * jnp.nanmedian(flux_err_hr, axis=1))
    if joint_geometry_hr is not None:
        model_run_args_hr.update(joint_geometry_hr['model_kwargs'])
        init_params_hr.update(joint_geometry_hr['init_params'])
    _initialize_spectroscopic_start(
        HARMONICA_ODD_HARMONICS=geometry.HARMONICA_ODD_HARMONICS, RORS_BASE=geometry.RORS_BASE, U_mu=U_mu_hr_init,
        bestfit_params_wl_df=inputs.bestfit_params_wl_df,
        detrend_type_multiwave=detrend_type_multiwave,
        harmonica_spectro_parameterization=model.harmonica_spectro_parameterization,
        init_params=init_params_hr, ld_prior_mode=model.ld_prior_mode, ld_profile=model.ld_profile,
        ld_uniform_basis=model.ld_uniform_basis, num_lcs=num_lcs_hr,
        spectro_ld_parameterization=model.spectro_ld_parameterization,
        surface_config=model.surface_config, transit_engine=model.transit_engine,
    )

    sweep = getattr(inputs, 'execution_context', None)
    if sweep is not None:
        if 'quadratic' in detrend_type_multiwave:
            init_params_hr.setdefault('v2', jnp.zeros(num_lcs_hr))
        init_params_hr.update(sweep.warm_start('highres', init_params_hr, law=model.ld_profile,
            site_bounds=sweep.ld_site_bounds(model.ld_profile, hr_ld_mode, model.ld_uniform_basis,
                model._engine_spectro_kw.get('ld_uniform_coefficient_bounds', (0., 1.)))))
    fit_hr = _spectroscopic_model(
        geometry, model, inputs, time_hr, U_mu_hr_init, init_params_hr, joint_geometry_hr,
        hr_ld_mode, hr_trend_mode, detrend_type_multiwave, 'high-resolution',
    )
    spot_basis_hr = fit_hr.surface_basis
    hr_adaptive_fallback_model, hr_adaptive_fallback_init = fit_hr.fallback_model, fit_hr.fallback_init
    if spot_basis_hr is not None:
        model_run_args_hr['surface_basis_data'] = spot_basis_hr

    spot_trend, spot_trend2, jump_trend, exp_trend_hr = _spectroscopic_trends(
        inputs, detrend_type_multiwave, time_hr, num_lcs_hr, model_run_args_hr, init_params_hr,
        gp_trend, spot_trend, spot_trend2, jump_trend, model.fixed_tau_spectro,
    )
    trend_names_hr = _prepare_trend_likelihood(
        model, inputs, fit_hr, flux_hr, flux_err_hr, model_run_args_hr,
    )

    stage = SimpleNamespace(
        name='highres', resolution='high_resolution', bin_str=hr_bin_str, artifact_stem=hr_artifact_stem,
        key_mcmc=key_mcmc_hr, key_map=key_map_hr, chunk_size=vmap_chunk_size_hr,
        nuts_kwargs=harmonica_hr_nuts_kwargs if model.transit_engine == 'harmonica' else jaxoplanet_hr_nuts_kwargs,
        mcmc_kwargs=highres_mcmc_kwargs, wavelengths=inputs.dataset.wavelengths_hr,
        wavelength_err=inputs.dataset.wavelengths_err_hr,
    )
    samples_hr = _sample_spectroscopic_stage(
        geometry, model, inputs, stage, fit_hr, spectro_sampler,
        flux_hr, flux_err_hr, model_run_args_hr, trend_names_hr,
    )
    if samples_hr is None:
        return

    map_params_hr, model_all_hr = _evaluate_spectroscopic_posterior(
        geometry, model, fit_hr, U_mu_hr_init, samples_hr, flux_hr,
        gp_trend, spot_trend, spot_trend2, jump_trend, exp_trend_hr,
    )

    residuals_hr = np.array(flux_hr - model_all_hr)
    _hr_dt_sec = float(np.nanmedian(np.diff(np.array(time_hr)))) * 86400.0
    save_noise_binning_data(
        residuals_hr, state_path(inputs.output_dir, f"{hr_artifact_stem}_noise.csv"),
    )

    plot_params_hr = dict(map_params_hr)
    if spot_basis_hr is not None:
        plot_params_hr['_surface_basis'] = spot_basis_hr
        plot_params_hr['_surface_basis_time'] = jnp.asarray(time_hr)
    _save_spectroscopic_products(
        geometry, model, inputs, stage, fit_hr, samples_hr, time_hr, flux_hr,
        flux_err_hr, map_params_hr, plot_params_hr,
        (gp_trend, spot_trend, spot_trend2, jump_trend, exp_trend_hr),
        model_all_hr, U_mu_hr_init,
    )
    return True

"""White-light fitting, diagnostics, and geometry handoff."""

from .output_paths import OutputPaths, state_path
from .diagnostic_figures import white_light, white_light_start, event_epoch

from koala.console import debug, info, white_light_verdict, parameter_summary

import os
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro_ext.optim as optimx
import pandas as pd
from .plotting import plot_whitelight_corner
from jaxoplanet.experimental import calc_poly_coeffs
from models.common import get_I_power2, compute_transit_model_auto
from models.harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_impact_param_from_cos_i,
    harmonica_duration_from_geometry,
    harmonica_duration_from_cos_i,
)
from models.trends import spot_crossing, compute_lc_linear
from models.gp import (
    GP_HYPERPARAMETER_BOUNDS,
    GP_MODEL_REVISION,
    compute_gp_training_prediction,
    resolve_gp_mean_function,
    resolve_gp_solver,
    validate_gp_times,
)
from models.detrend import resolve_detrend_kernel
from .surface_outputs import save_surface_results
from .constants import HARMONICA_INIT_ODD_COEFF, SURFACE_PARAMS
from .config import trend_prior_tau_range
from .artifacts import (
    _science_artifact_fingerprint,
    _atomic_save_npy,
    _atomic_dataframe_csv,
    ArtifactSet,
)
from .outputs import (
    _soft_step_np,
    _trend_from_params_np,
    save_whitelight_timeseries,
    _save_mcmc_diagnostics,
    get_asym_errors,
    calculate_beta_metrics,
    run_beta_monte_carlo,
)
from .geometry import (
    _continue_mcmc_until_geometry_gate,
    _posterior_num_draws,
    _geometry_from_white_light_medians,
    _write_whitelight_geometry_handoff,
    _load_whitelight_geometry_handoff,
)
from .readers import TIME_OFFSET, TIME_SYSTEM
from .limb_darkening import (
    _repair_gaussian_ld_start,
    _power2_ld_initial_sites,
    _power2_ld_optimization_sites,
    _quadratic_uniform_initial_sites,
    _validate_whitelight_optimized_start,
    get_limb_darkening,
    get_or_build_power2_ld_prior, get_or_build_quadratic_ld_prior,
)
from .harmonica_products import (
    _coerce_wavelength_axis,
    _harmonica_frac_site,
    _harmonica_coeff_to_frac,
    _harmonica_sample_payload,
    _has_harmonica_odd_samples,
    _harmonica_cosi_from_b,
    save_harmonica_limb_products,
)
from .surface import (
    _attach_surface_eval_metadata,
    _prepare_fixed_surface_basis,
    _select_transit_eval_params,
)


def _compute_whitelight_gp_products(
    params, t, error, y, *, detrend_type, gp_solver
):
    """Return aligned training-point GP products for white-light output."""
    mu, var = compute_gp_training_prediction(
        params, t, error, y, detrend_type=detrend_type, gp_solver=gp_solver,
        assume_sorted=True, include_mean=True,
    )
    t_ref = jnp.min(t)
    parametric_mean = resolve_gp_mean_function(detrend_type)(params, t, t_ref=t_ref)
    planet_model = compute_transit_model_auto(params, t)
    return {
        "mu": mu,
        "var": var,
        "planet_model_only": planet_model,
        # Total systematics factor (parametric trend plus GP), given
        # F = (1 + planet) * systematics.
        "trend_flux_total": mu / (1.0 + planet_model),
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
        info('A white-light trend start reached its prior boundary; moving the start inside it.')
        debug(
            f"WARNING: white-light optimizer placed {site} on or outside its "
            f"support edge ({value:.6g}); resetting it to {reset:.6g} before "
            "Laplace preparation."
        )
        repaired[site] = jnp.asarray(reset, dtype=jnp.asarray(repaired[site]).dtype)
    return repaired

def _white_light_fingerprint_config(cfg):
    """Keep scientific settings when identifying a reusable white-light fit.

    Spectroscopic batch size, stage selection, cache location and output paths
    cannot change this posterior. Data arrays, LD, trends, geometry, seed and
    sampling requirements remain part of the surrounding artifact identity.
    """
    result = {key: value for key, value in cfg.items()
              if key not in {"gp_solver", "output_dir"}}
    if "flags" in result:
        execution_flags = {"analysis_stage", "vmap_chunk", "spectro_chunk_size",
                           "jax_compilation_cache_dir", "cpu_host_devices"}
        flags = {key: value for key, value in result["flags"].items()
                 if key not in execution_flags}
        if flags:
            result["flags"] = flags
        else:
            result.pop("flags")
    return result


def _whitelight_geometry_sites(specs, n_planets, param_method):
    """Map each latent white-light geometry site to its starting value.

    Every free specification contributes one latent site (``t0_i``,
    ``rors_i``, ``b_i``, ``period_i`` and ``duration_i`` or ``a_rs_i``, or
    the ``log`` form for a log-uniform prior); fixed ones contribute none.
    """
    from models.priors import latent_init_site

    sites = {}
    names = ['period', 't0', 'rprs', 'b', 'duration' if param_method == 'duration' else 'a_rs']
    for i in range(n_planets):
        for name in names:
            site = f"{'rors' if name == 'rprs' else name}_{i}"
            init = latent_init_site(specs[name][i], site)
            if init is not None:
                sites[init[0]] = init[1]
    return sites


def _apply_trend_prior_starts(init, trend_priors):
    """Replace built-in trend starts by the configured ``trend_priors`` starts."""
    from models.priors import latent_init_site

    default_sites = {'c': 'c', 'v': 'v', **{f'v{k}':f'v{k}' for k in (2,3,4)},
                     'A': 'A', 'tau': 'log_tau', 'jitter': 'log_jitter'}
    for name, spec in (trend_priors or {}).items():
        if default_sites[name] not in init:
            continue
        init.pop(default_sites[name])
        site = latent_init_site(spec, name)
        if site is not None:
            init[site[0]] = site[1]


def _start_at_scanned_eclipse(init, surface_config, specs, data, detrending_type,
                              tau_range=(1e-3, 1e-1), **geometry):
    """Move a free eclipse time (and depth) start onto the best-fitting epoch."""
    from models.priors import latent_init_site
    from .surface import scan_eclipse_time

    t0_spec = specs['t0'][0]
    depth_spec = (surface_config.get('eclipse_depth_spec') or (None,))[0]
    if depth_spec is None:
        return
    geometry = {k: float(np.atleast_1d(np.asarray(v))[0]) for k, v in geometry.items()}
    found = scan_eclipse_time(
        surface_config, data.wl_time, data.wl_flux, data.wl_flux_err,
        t0_spec, depth_spec, detrending_type, tau_range=tau_range, **geometry,
    )
    if found is None:
        return
    t0, depth, delta_chi2 = found
    debug(f"Eclipse-time scan: best eclipse at {t0 + 0.5 * geometry['period']:.5f} "
          f"({(t0 - t0_spec.value) * 1440:+.1f} min from the configured start), "
          f"depth {depth * 1e6:.0f} ppm, delta chi2 {delta_chi2:.1f} over no eclipse.")
    if delta_chi2 < 25.0:
        info('The white-light eclipse is detected at less than 5 sigma; its timing is poorly constrained.')
        debug("WARNING: the white-light eclipse is detected at less than 5 sigma. "
              "Its timing is poorly constrained; for a circular orbit consider "
              "fixing planet.eclipse_time at the ephemeris prediction.")
    if delta_chi2 < 9.0:
        return
    init[latent_init_site(t0_spec, 't0_0')[0]] = jnp.asarray(t0)
    if surface_config.get('model') in {'eclipse', 'phase_curve'} and depth_spec.free:
        lo = depth_spec.low if depth_spec.low is not None else 0.0
        hi = depth_spec.high if depth_spec.high is not None else np.inf
        margin = 1e-3 * (hi - lo) if np.isfinite(hi) else 1e-7
        depth = float(np.clip(depth, lo + margin, hi - margin))
        name, _ = latent_init_site(depth_spec, '_eclipse_depth_0')
        init[name] = jnp.log(depth) if name.startswith('log') else jnp.asarray(depth)


def _planet_site_array(mapping, stem, n_planets, transform=None):
    """Stack ``{stem}_{i}`` for every planet, or return None if any is missing."""
    names = [f"{stem}_{i}" for i in range(n_planets)]
    if not all(name in mapping for name in names):
        return None
    return jnp.array([mapping[name] if transform is None else transform(mapping[name])
                      for name in names])


def _update_planet_arrays(target, values, n_planets, names):
    """Collect direct or latent per-planet sites into physical arrays."""
    aliases = {'b': ('_b', jnp.abs), 'rors': ('depths', jnp.sqrt),
               'duration': ('logD', jnp.exp), 'a_rs': ('log_a_rs', jnp.exp)}
    for name in names:
        array = _planet_site_array(values, name, n_planets)
        if array is None and name in aliases:
            stem, transform = aliases[name]
            array = _planet_site_array(values, stem, n_planets, transform)
        if array is not None:
            target[name] = array


def _save_whitelight_gp_database(path, wl_flux, products):
    """Write the legacy GP handoff schema consumed by spectroscopy."""
    frame = pd.DataFrame({
        'wl_flux': wl_flux,
        'gp_flux': products["mu"],
        'gp_err': jnp.sqrt(products["var"]),
        'gp_trend': products["gp_stochastic_component"],
    })
    _atomic_dataframe_csv(frame, path, index=True)


def _evaluate_initial_model(params, time, detrending_type):
    """Evaluate the deterministic mean for initialization diagnostics."""
    if 'gp' in detrending_type:
        return compute_lc_linear(params, time)
    try:
        return resolve_detrend_kernel(detrending_type)(params, time)
    except KeyError:
        return compute_lc_linear(params, time)


def _build_preopt_physical_params(
    init_params, n_planets_eval, *, context, U_mu_wl, hyper_params_wl,
):
    params_eval = hyper_params_wl.copy()
    params_eval.update(init_params)
    for name in ('c', 'v', 'A', 'tau'):
        # Log-uniform trend sites (the default ``log_tau``) start in log space.
        if f"log_{name}" in init_params:
            params_eval[name] = jnp.exp(init_params[f"log_{name}"])

    def _ensure_len(value, default):
        arr = jnp.atleast_1d(jnp.asarray(params_eval.get(value, default)))
        if arr.size == 1 and n_planets_eval > 1:
            return jnp.repeat(arr, n_planets_eval)
        return arr

    params_eval["period"] = _ensure_len("period", context.PERIOD_FIXED)
    params_eval["t0"] = _ensure_len("t0", context.PRIOR_T0)
    params_eval["b"] = _ensure_len("b", context.PRIOR_B)
    params_eval["rors"] = _ensure_len("rors", context.PRIOR_RPRS)

    _update_planet_arrays(params_eval, init_params, n_planets_eval, ('period', 't0', 'b', 'rors'))

    if context.param_method == 'duration':
        params_eval['duration'] = _ensure_len('duration', context.PRIOR_DUR)
        _update_planet_arrays(params_eval, init_params, n_planets_eval, ('duration',))
        params_eval['a_rs'] = harmonica_a_rs_from_duration(
            params_eval['period'],params_eval['duration'],params_eval['b'],params_eval['rors'],
            ecc=hyper_params_wl['ecc'],omega=hyper_params_wl['omega'])
    if context.transit_engine == 'harmonica':
        params_eval["a_rs"] = _ensure_len("a_rs", context.HARMONICA_A_RS)
        if all(f"log_a_rs_{i}" in init_params for i in range(n_planets_eval)):
            params_eval["a_rs"] = jnp.array([jnp.exp(init_params[f"log_a_rs_{i}"]) for i in range(n_planets_eval)])
        elif all(f"logD_{i}" in init_params for i in range(n_planets_eval)):
            params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets_eval)])
            params_eval["a_rs"] = harmonica_a_rs_from_duration(
                params_eval['period'], params_eval['duration'], params_eval['b'],
                params_eval['rors'], ecc=hyper_params_wl['ecc'],
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
                params_eval['period'], params_eval['a_rs'], params_eval['cos_i'],
                params_eval['rors'], ecc=hyper_params_wl['ecc'],
                omega=hyper_params_wl['omega'],
            )
        else:
            params_eval["duration"] = harmonica_duration_from_geometry(
                params_eval['period'], params_eval['a_rs'], params_eval['b'],
                params_eval['rors'], ecc=hyper_params_wl['ecc'],
                omega=hyper_params_wl['omega'],
            )
            params_eval["cos_i"] = _harmonica_cosi_from_b(
                params_eval["b"], params_eval["a_rs"],
                hyper_params_wl['ecc'], hyper_params_wl['omega'],
            )
        if context.ld_profile == "power2":
            if "c1" in init_params:
                params_eval["c_ld"] = init_params["c1"]
            if "c2" in init_params:
                params_eval["alpha_ld"] = init_params["c2"]
        else:
            params_eval["u1_ld"] = U_mu_wl[0]
            params_eval["u2_ld"] = U_mu_wl[1]
    else:
        params_eval["duration"] = _ensure_len("duration", context.PRIOR_DUR)
        _update_planet_arrays(params_eval, init_params, n_planets_eval, ('duration',))
        if context.param_method == 'a_rs':
            _update_planet_arrays(params_eval, init_params, n_planets_eval, ('a_rs',))
            if "a_rs" in params_eval:
                params_eval["duration"] = harmonica_duration_from_geometry(
                    params_eval['period'], params_eval['a_rs'], params_eval['b'],
                    params_eval['rors'], ecc=hyper_params_wl['ecc'],
                    omega=hyper_params_wl['omega'],
                )

    return _attach_surface_eval_metadata(params_eval, context.surface_config)


def _soln_to_physical_params(soln, base_params, n_planets=1, *, context, U_mu_wl, hyper_params_wl):
    p = dict(base_params)
    for key, value in soln.items():
        if key != "obs":
            p[key] = value
    for name in ('c', 'v', 'A', 'tau'):
        if f"log_{name}" in p:
            p[name] = jnp.exp(p[f"log_{name}"])

    def have_all(prefix):
        return all(f"{prefix}_{i}" in p for i in range(n_planets))

    have_log_a_rs = have_all("log_a_rs")
    have_logD = have_all("logD")

    _update_planet_arrays(p, p, n_planets, ('period', 't0', 'b', 'rors', 'duration', 'a_rs'))
    if context.transit_engine != 'harmonica':
        if ("a_rs" not in p or context.param_method == "duration") and "duration" in p and "b" in p:
            p["a_rs"] = harmonica_a_rs_from_duration(
                p['period'], p['duration'], p['b'], p['rors'], ecc=hyper_params_wl['ecc'],
                omega=hyper_params_wl['omega'],
            )
        if "duration" not in p and "a_rs" in p and "b" in p:
            p["duration"] = harmonica_duration_from_geometry(
                p['period'], p['a_rs'], p['b'], p['rors'], ecc=hyper_params_wl['ecc'],
                omega=hyper_params_wl['omega'],
            )
        if "a_rs" in p and "b" in p:
            p["cos_i"] = _harmonica_cosi_from_b(
                p["b"], p["a_rs"], hyper_params_wl['ecc'], hyper_params_wl['omega']
            )
            p["inc"] = jnp.arccos(jnp.clip(p["cos_i"], 0.0, 1.0))
    for harmonic_name in context.HARMONICA_ODD_HARMONICS:
        frac_key = _harmonica_frac_site(harmonic_name)
        if harmonic_name not in p and frac_key in soln and "rors" in p:
            p[harmonic_name] = soln[frac_key] * jnp.maximum(jnp.min(jnp.asarray(p["rors"])), 1e-6)
    if context.transit_engine == 'harmonica' and "rors" in p:
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
                p['period'], p['a_rs'], p['cos_i'], p['rors'], ecc=hyper_params_wl['ecc'],
                omega=hyper_params_wl['omega'],
            )
        else:
            if "duration" in p and "b" in p and (context.param_method == "duration" or have_logD or "a_rs" not in p):
                p["a_rs"] = harmonica_a_rs_from_duration(
                    p['period'], p['duration'], p['b'], p['rors'], ecc=hyper_params_wl['ecc'],
                    omega=hyper_params_wl['omega'],
                )
            if "a_rs" in p and "b" in p and (have_log_a_rs or "duration" not in p):
                p["duration"] = harmonica_duration_from_geometry(
                    p['period'], p['a_rs'], p['b'], p['rors'], ecc=hyper_params_wl['ecc'],
                    omega=hyper_params_wl['omega'],
                )
            if "a_rs" in p and "b" in p:
                p["cos_i"] = _harmonica_cosi_from_b(
                    p["b"], p["a_rs"], hyper_params_wl['ecc'], hyper_params_wl['omega']
                )
            if "duration" not in p and "cos_i" in p:
                p["duration"] = harmonica_duration_from_cos_i(
                    p['period'], p['a_rs'], p['cos_i'], p['rors'], ecc=hyper_params_wl['ecc'],
                    omega=hyper_params_wl['omega'],
                )
        if context.ld_profile == "power2" and "c1" in p and "c2" in p:
            p["c_ld"] = p["c1"]
            p["alpha_ld"] = p["c2"]
        elif context.ld_profile == "quadratic":
            p["u1_ld"] = U_mu_wl[0]
            p["u2_ld"] = U_mu_wl[1]

    return _attach_surface_eval_metadata(p, context.surface_config)


def _save_white_light_parameter_table(*, context, bestfit_params_wl, wl_params_path):
    """Write the white-light parameter table with its established column order."""
    rows = []
    for i in range(context.n_planets):
        row = {'planet_id': i}
        for name in ('period', 'period_err', 'period_err_low', 'period_err_high', 'duration', 't0'):
            row[name] = bestfit_params_wl[name][i]
        row['t0_bjd_tdb'] = float(bestfit_params_wl['t0'][i]) + TIME_OFFSET
        for name in (
            'b', 'rors', 'depths', 'duration_err', 't0_err', 'b_err', 'rors_err',
            'depths_err', 'duration_err_low', 'duration_err_high', 't0_err_low', 't0_err_high', 'b_err_low', 'b_err_high', 'rors_err_low', 'rors_err_high', 'depths_err_low', 'depths_err_high',
        ):
            row[name] = bestfit_params_wl[name][i]
        for name in ('a_rs', 'cos_i', 'ecc', 'omega'):
            if name in bestfit_params_wl:
                for suffix in ('', '_err', '_err_low', '_err_high'):
                    row[name + suffix] = bestfit_params_wl[name + suffix][i]


        if context.detrending_type != 'none':
            _add_scalar_parameter('c', bestfit_params_wl=bestfit_params_wl, row=row)
            _add_scalar_parameter('v', bestfit_params_wl=bestfit_params_wl, row=row)
        for harmonic_name in context.HARMONICA_ODD_HARMONICS:
            _add_scalar_parameter(harmonic_name, bestfit_params_wl=bestfit_params_wl, row=row)

        if 'c1' in bestfit_params_wl:
            _add_scalar_parameter('c1', bestfit_params_wl=bestfit_params_wl, row=row)
            _add_scalar_parameter('c2', bestfit_params_wl=bestfit_params_wl, row=row)
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
            for name in ('q1', 'q2'):
                _add_scalar_parameter(name, bestfit_params_wl=bestfit_params_wl, row=row)

        for name in (
            'v2', 'v3', 'v4', 'A', 'tau', 'spot_amp', 'spot_mu', 'spot_sigma', 'spot_amp2',
            'spot_mu2', 'spot_sigma2', 't_jump', 'jump', 'width', 'width_minutes', 'log_width', 'GP_log_sigma', 'GP_log_rho',
        ):
            _add_scalar_parameter(name, bestfit_params_wl=bestfit_params_wl, row=row)
        for name in (
            'eclipse_depth', 'phase_c1', 'phase_d1',
            'nightside_flux', 'phase_amplitude', 'phase_offset_deg',
        ):
            if name in bestfit_params_wl:
                values = np.atleast_1d(bestfit_params_wl[name])
                row[name] = values[i]
                row[f'{name}_err_low'] = np.atleast_1d(bestfit_params_wl[f'{name}_err_low'])[i]
                row[f'{name}_err_high'] = np.atleast_1d(bestfit_params_wl[f'{name}_err_high'])[i]
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


def _repair_white_jitter_start(solution, initial_values, jitter_spec=None):
    """Keep the default bounded jitter's initial coordinate strictly interior.

    GPU optimization can round log_jitter to the exact Uniform endpoint;
    the inverse interval transform is then infinite. Reset only the start,
    never the prior bound or any Laplace/sampler control. Explicit jitter
    ParameterSpecs retain their own support and initialization semantics.
    """
    if jitter_spec is not None or 'log_jitter' not in solution:
        return solution
    low, high = np.log(1e-5), np.log(1e-2)
    margin = 1e-8 * (high-low)
    value = float(np.asarray(solution['log_jitter']))
    if np.isfinite(value) and low+margin < value < high-margin:
        return solution
    reset = float(np.asarray(initial_values.get('log_jitter', np.log(1e-4))))
    if not np.isfinite(reset) or not low+margin < reset < high-margin:
        reset = .5*(low+high)
    repaired = dict(solution)
    repaired['log_jitter'] = jnp.asarray(reset)
    debug('The white-light jitter start reached its prior boundary; moving the start inside it.')
    debug('White-light optimizer log_jitter reached its prior support edge; '
          'resetting the initial value inside the unchanged prior.', flush=True)
    return repaired


def _optimize_white_light_start(
    *, context, U_mu_wl, geometry_sites, hyper_params_wl, init_params_wl, is_gp_detrending,
    soln, whitelight_model_for_run, wl_ld_mode,
):
    """Refine the physical start and save the initialization comparison."""
    try:

        n_planets_sanity = len(jnp.atleast_1d(context.PERIOD_FIXED))
        params_complete = _build_preopt_physical_params(
            init_params_wl, n_planets_sanity, context=context,
            U_mu_wl=U_mu_wl, hyper_params_wl=hyper_params_wl,
        )

        key_sanity = jax.random.PRNGKey(0) if context.key_master is None else context.key_master
        keys = jax.random.split(key_sanity, num=3)

        if n_planets_sanity == 1:
            _opt_sites = [
                name for name in (
                    "period_0", "log_a_rs_0", "a_rs_0", "logD_0",
                    "duration_0", "t0_0", "_b_0", "b_0",
                )
                if name in geometry_sites
            ] or None
        else:
            _opt_sites = None
        # Spotted stars still evaluate native starry, too slow to optimize
        # here; unspotted eclipse/phase-curve models use the closed-form path.
        skip_optimizer = bool(context.surface_config.get('spots'))
        if skip_optimizer:
            soln = dict(init_params_wl)
        else:
            stage1 = optimx.optimize(
                whitelight_model_for_run, sites=_opt_sites, start=init_params_wl,
            )
            soln = stage1(keys[0], context.data.wl_time, context.data.wl_flux_err, y=context.data.wl_flux, prior_params=hyper_params_wl)
            soln = _repair_white_jitter_start(soln, init_params_wl,
                (getattr(context, 'trend_priors', None) or {}).get('jitter'))

        stage2_sites = ["rors_0"] if "rors_0" in geometry_sites else []
        if wl_ld_mode != "fixed":
            if wl_ld_mode == 'fixed_u1':
                stage2_sites.append('u2')
            elif context.transit_engine == 'harmonica':
                stage2_sites.extend(['u'] if context.ld_profile == 'quadratic' and context.ld_uniform_basis == 'coefficients'
                                    else ['ld_q'] if context.ld_profile == 'quadratic' and context.ld_uniform_basis == 'kipping'
                                    else ['ld_uplus_uminus'] if context.ld_profile == 'quadratic'
                                    else ['c1', 'c2'])
            elif context.ld_profile == "quadratic":
                stage2_sites.append(
                    ('ld_q' if context.ld_uniform_basis == 'kipping'
                     else 'ld_uplus_uminus'
                     if (wl_ld_mode == 'uniform'
                         and context.ld_uniform_basis == 'uplus_uminus')
                     else 'u')
                )
            elif context.ld_profile == "power2":
                stage2_sites.extend(_power2_ld_optimization_sites(
                    context.whitelight_ld_parameterization
                ))
        if n_planets_sanity != 1 or not stage2_sites:
            stage2_sites = None

        if not skip_optimizer:
            stage2 = optimx.optimize(whitelight_model_for_run, sites=stage2_sites, start=soln)
            soln = stage2(keys[1], context.data.wl_time, context.data.wl_flux_err, y=context.data.wl_flux, prior_params=hyper_params_wl)
            soln = _repair_white_jitter_start(soln, init_params_wl,
                (getattr(context, 'trend_priors', None) or {}).get('jitter'))

            stage3 = optimx.optimize(whitelight_model_for_run, start=soln)
            soln = stage3(keys[2], context.data.wl_time, context.data.wl_flux_err, y=context.data.wl_flux, prior_params=hyper_params_wl)
            soln = _repair_white_jitter_start(soln, init_params_wl,
                (getattr(context, 'trend_priors', None) or {}).get('jitter'))

        # Constrained optimizers may return a Uniform latent exactly
        # on its support edge.  That point has an infinite
        # unconstrained coordinate and cannot initialize NumPyro.
        if (
            'linear_discontinuity' in context.detrending_type
            and context.step_width_mode == 'free'
        ):
            cadence = float(np.median(np.diff(np.sort(np.asarray(context.data.wl_time)))))
            log_width_lo = np.log(0.5 * cadence)
            log_width_hi = np.log(30.0 / (24.0 * 60.0))
            optimized_log_width = float(np.asarray(soln.get('log_width', np.nan)))
            edge_tolerance = 1.0e-8 * max(1.0, abs(log_width_hi - log_width_lo))
            if (
                not np.isfinite(optimized_log_width)
                or optimized_log_width <= log_width_lo + edge_tolerance
                or optimized_log_width >= log_width_hi - edge_tolerance
            ):
                reset_log_width = 0.5 * (
                    log_width_lo + log_width_hi
                )
                info('The white-light trend width reached its prior boundary; moving the start inside it.')
                debug(
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
                soln, ld_profile=context.ld_profile,
                ld_parameterization=context.whitelight_ld_parameterization,
                ld_prior_mode=context.ld_prior_mode,
                quadratic_uniform_bounds=context.quadratic_uniform_physical_bounds,
                n_planets=n_planets_sanity,
            )
        )
        if not optimized_start_valid:
            info('The optimized white-light start was not physical; using the prior start.')
            debug(
                "WARNING: discarding nonphysical white-light optimizer "
                "solution and falling back to the prior physical start: "
                + "; ".join(optimized_start_reasons),
                flush=True,
            )
            soln = dict(init_params_wl)
            fallback_valid, fallback_reasons = (
                _validate_whitelight_optimized_start(
                    soln, ld_profile=context.ld_profile,
                    ld_parameterization=context.whitelight_ld_parameterization,
                    ld_prior_mode=context.ld_prior_mode,
                    quadratic_uniform_bounds=context.quadratic_uniform_physical_bounds,
                    n_planets=n_planets_sanity,
                )
            )
            if not fallback_valid:
                raise RuntimeError(
                    "White-light prior fallback start is invalid: "
                    + "; ".join(fallback_reasons)
                )

        params_opt = _soln_to_physical_params(
            soln, params_complete, context=context, n_planets=n_planets_sanity, U_mu_wl=U_mu_wl,
            hyper_params_wl=hyper_params_wl,
        )
        if "rors" not in params_opt and "depths" in params_opt:
            params_opt["rors"] = jnp.sqrt(params_opt["depths"])

        flux_init = _evaluate_initial_model(params_complete, context.data.wl_time, context.detrending_type)
        flux_opt  = _evaluate_initial_model(params_opt, context.data.wl_time, context.detrending_type)
        if getattr(context, 'plots_mode', context.cfg.get('flags', {}).get('plots', 'full')) != 'minimal':
            white_light_start(context.data.wl_time, context.data.wl_flux, context.data.wl_flux_err,
                              flux_init, flux_opt,
                              OutputPaths(context.output_dir).figure('whitelight', 'start'),
                              label=context.instrument_full_str, event=context.surface_config.get("model", "transit"),
                              t0=event_epoch(context.PRIOR_T0, context.PERIOD_FIXED, context.surface_config.get("model", "transit")))

    except Exception as e:
        info('The white-light sanity figure could not be written; see koala.log.')
        debug(f"Could not create sanity plot: {e}")
    return _repair_white_jitter_start(soln, init_params_wl,
        (getattr(context, 'trend_priors', None) or {}).get('jitter'))


def _plot_white_light_start(*, context, U_mu_wl, hyper_params_wl, init_params_wl):
    """Save the initial model and residual diagnostic."""
    try:
        params_preopt = _build_preopt_physical_params(
            init_params_wl, context.n_planets, context=context, U_mu_wl=U_mu_wl,
            hyper_params_wl=hyper_params_wl,
        )
        flux_preopt = _evaluate_initial_model(params_preopt, context.data.wl_time, context.detrending_type)

        # Retain the evaluated start for diagnostics; the optimized comparison
        # writes the single start figure after optimization.
        context._plot_initial_model = flux_preopt

    except Exception as e:
        info('The initial white-light figure could not be written; see koala.log.')
        debug(f"Could not create pre-optimization sanity plot: {e}")


def _initialize_white_light(*, context, U_mu_wl, U_sigma_wl, _engine_wl_kw, geometry_sites):
    """Build the physical prior start and its white-light model."""
    hyper_params_wl = {
        "duration": context.PRIOR_DUR,
        "t0": context.PRIOR_T0,
        'period': context.PERIOD_FIXED,
        'b': context.PRIOR_B,
        'rprs': context.PRIOR_RPRS,
        'u': U_mu_wl,
        'a_rs': context.HARMONICA_A_RS,
        'ecc': context.HARMONICA_ECC,
        'omega': context.HARMONICA_OMEGA,
    }
    if '2spot' in context.detrending_type:
        hyper_params_wl['spot_guess'] = context.spot_mu
        hyper_params_wl['spot_guess2'] = context.spot_mu2
    elif 'spot' in context.detrending_type:
        hyper_params_wl['spot_guess'] = context.spot_mu
    if 'linear_discontinuity' in context.detrending_type:
        hyper_params_wl['t_jump_guess'] = context.t_jump_guess if context.t_jump_guess is not None else 0.5 * (jnp.min(context.data.wl_time) + jnp.max(context.data.wl_time))
        hyper_params_wl['jump_guess'] = context.jump_guess
        hyper_params_wl['step_width_days'] = context.step_width_days

    hyper_params_wl['u'] = U_mu_wl
    if context.ld_prior_mode == 'stellarprior' and U_sigma_wl is not None:
        hyper_params_wl['u_sigma'] = U_sigma_wl

    init_params_wl = {
        'c': 1.0,
        'v': 0.0,
        'log_jitter': jnp.log(1e-4),
        'b': context.PRIOR_B,
        'rors': context.PRIOR_RPRS
    }
    init_params_wl['u'] = U_mu_wl
    if context.ld_prior_mode == 'fixed_u1':
        init_params_wl.pop('u', None)
        low, high = context._engine_wl_kw.get('ld_uniform_coefficient_bounds', (0., 1.))
        init_params_wl['u2'] = jnp.clip(jnp.asarray(U_mu_wl)[..., 1],
            low + 1e-4*(high-low), high - 1e-4*(high-low))
    if context.ld_profile == 'quadratic' and (context.ld_prior_mode == 'uniform' or (
            context.ld_uniform_basis == 'kipping'
            and context.ld_prior_mode in {'gaussian', 'stellarprior'})):
        init_params_wl.pop('u', None)
        init_params_wl.update(_quadratic_uniform_initial_sites(
            U_mu_wl, context.ld_uniform_basis
        ))
    if context.ld_profile == 'quadratic' and context.ld_prior_mode == 'uniform' and context.ld_uniform_basis == 'coefficients':
        low, high = context._engine_wl_kw.get('ld_uniform_coefficient_bounds', (0., 1.))
        init_params_wl['u'] = jnp.clip(init_params_wl['u'], low + 1e-4*(high-low), high - 1e-4*(high-low))
    # Power-2 models sample the native (c, alpha) coefficients at the
    # latent sites ``c1`` and ``c2``.  Supplying only ``u`` silently
    # falls back to NumPyro's default initialization and defeats the
    # stellar-informed starting point for both transit engines.
    if context.ld_profile == 'power2' and context.ld_prior_mode != 'fixed':
        init_params_wl.update(_power2_ld_initial_sites(
            U_mu_wl, context.whitelight_ld_parameterization, context.ld_prior_mode
        ))
    if context.transit_engine == 'harmonica':
        for harmonic_name in context.HARMONICA_ODD_HARMONICS:
            init_params_wl[_harmonica_frac_site(harmonic_name)] = _harmonica_coeff_to_frac(
                HARMONICA_INIT_ODD_COEFF, context.PRIOR_RPRS,
            )

    init_params_wl.update(geometry_sites)

    for surface_name in ('eclipse_depth', 'phase_c1', 'phase_d1'):
        spec_name = f'{surface_name}_spec'
        if spec_name in context.surface_config:
            from models.priors import latent_init_site
            for i, spec in enumerate(context.surface_config[spec_name]):
                init = latent_init_site(spec, f'_{surface_name}_{i}')
                if init is not None:
                    init_params_wl[init[0]] = init[1]
    _start_at_scanned_eclipse(
        init_params_wl, context.surface_config, context.planet_parameter_specs, context.data,
        context.detrending_type, period=context.PERIOD_FIXED, a_rs=context.HARMONICA_A_RS,
        b=context.PRIOR_B, rors=context.PRIOR_RPRS, ecc=context.HARMONICA_ECC,
        omega=context.HARMONICA_OMEGA,
        tau_range=trend_prior_tau_range(getattr(context, 'trend_priors', None)),
    )
    for i, spot in enumerate(context.surface_config.get('spots', ())):
        if spot.get('contrast_prior_width', 0.0) > 0.0:
            init_params_wl[f'_stellar_spot_contrast_{i}'] = spot['contrast']
    polynomial_degree = max((degree for name, degree in
                             (('quadratic', 2), ('cubic', 3), ('quartic', 4))
                             if name in context.detrending_type), default=1)
    init_params_wl.update({f'v{degree}': 0.0 for degree in range(2, polynomial_degree + 1)})
    if 'explinear' in context.detrending_type:
        init_params_wl['A'] = 0.001
        # ``tau`` is deterministic; ``log_tau`` is the actual latent
        # site and its prior spans 1e-3--1e-1 days.
        init_params_wl['log_tau'] = jnp.log(1e-2)
    _apply_trend_prior_starts(init_params_wl, getattr(context, 'trend_priors', None))
    if not context.uses_surface_model and 'gp' in context.detrending_type:
        init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(context.data.wl_flux_err))
        init_params_wl['GP_log_rho'] = jnp.log(0.1)
    if 'linear_discontinuity' in context.detrending_type:
        if context.t_jump_guess is not None:
            init_params_wl['t_jump'] = context.t_jump_guess
        else:
            init_params_wl['t_jump'] = 0.5 * (jnp.min(context.data.wl_time) + jnp.max(context.data.wl_time))
        init_params_wl['jump'] = context.jump_guess
        if context.step_width_mode == 'free':
            cadence = float(np.median(np.diff(np.sort(np.asarray(context.data.wl_time)))))
            init_params_wl['log_width'] = np.log(
                np.sqrt(0.5 * cadence * (30.0 / (24.0 * 60.0)))
            )
    if 'spot' in context.detrending_type:
        init_params_wl['spot_amp'] = context.spot_amp
        init_params_wl['spot_mu'] = context.spot_mu
        init_params_wl['spot_sigma'] = context.spot_sigma
    if '2spot' in context.detrending_type:
        init_params_wl['spot_amp2'] = context.spot_amp2
        init_params_wl['spot_mu2'] = context.spot_mu2
        init_params_wl['spot_sigma2'] = context.spot_sigma2
    if ('spot' in context.detrending_type
            and context.whitelight_trend_parameterization == 'cadence'):
        # init_to_value ignores deterministic physical sites. Supply the
        # sampled cadence sites too, so staged optimization starts at the
        # configured crossing instead of a randomly initialized broad bump.
        cadence = jnp.median(jnp.diff(jnp.sort(context.data.wl_time)))
        for suffix in (('', '2') if '2spot' in context.detrending_type else ('',)):
            init_params_wl[f'spot_mu{suffix}_offset_cadences'] = jnp.asarray(0.)
            init_params_wl[f'spot_sigma{suffix}_cadences'] = (
                init_params_wl[f'spot_sigma{suffix}'] / cadence
            )
    # Step 1 of the Sing recipe measures LD freely.  The resulting
    # artifact is deliberately consumed only by the spectroscopic fit.
    wl_ld_mode = 'uniform' if context.ld_prior_mode == 'sing' else context.ld_prior_mode
    if wl_ld_mode == 'fixed':
        spot_basis_wl = _prepare_fixed_surface_basis(
            context.surface_config, context.data.wl_time, U_mu_wl, period=context.PERIOD_FIXED,
            t0=context.PRIOR_T0, a_rs=context.HARMONICA_A_RS, b=context.PRIOR_B,
            rors=context.PRIOR_RPRS, ecc=context.HARMONICA_ECC, omega=context.HARMONICA_OMEGA,
            ld_profile=context.ld_profile,
        )
        if spot_basis_wl is not None:
            _engine_wl_kw['surface_basis'] = spot_basis_wl
            debug('Prepared exact fixed-geometry surface basis for white-light inference.')
    whitelight_model_for_run = context.create_whitelight_model(
        detrend_type=context.detrending_type, n_planets=context.n_planets, ld_mode=wl_ld_mode,
        step_width_mode=context.step_width_mode, **_engine_wl_kw,
    )
    sweep = getattr(context, 'execution_context', None)
    if sweep is not None:
        init_params_wl = sweep.warm_start('white', init_params_wl, law=context.ld_profile,
            parameter_specs=context.planet_parameter_specs,
            site_bounds=sweep.ld_site_bounds(context.ld_profile, wl_ld_mode, context.ld_uniform_basis,
                context._engine_wl_kw.get('ld_uniform_coefficient_bounds', (0., 1.))))
    init_params_wl = _repair_gaussian_ld_start(
        init_params_wl, ld_profile=context.ld_profile,
        ld_prior_mode=context.ld_prior_mode, ld_uniform_basis=context.ld_uniform_basis,
    )
    return hyper_params_wl, init_params_wl, wl_ld_mode, whitelight_model_for_run


def _summarize_white_light(*, context, U_mu_wl, wl_samples):
    """Summarize limb darkening, orbital geometry, and systematics draws."""
    bestfit_params_wl = {}

    if context.ld_profile == 'power2':
        if 'c1' in wl_samples and 'c2' in wl_samples:
            _set_parameter_stats('c1', wl_samples['c1'], axis=0, bestfit_params_wl=bestfit_params_wl)
            _set_parameter_stats('c2', wl_samples['c2'], axis=0, bestfit_params_wl=bestfit_params_wl)
        else:
            _set_fixed_parameter_stats('c1', U_mu_wl[0], bestfit_params_wl=bestfit_params_wl)
            _set_fixed_parameter_stats('c2', U_mu_wl[1], bestfit_params_wl=bestfit_params_wl)
        POLY_DEGREE = 12
        MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
        power2_profile = get_I_power2(bestfit_params_wl['c1'], bestfit_params_wl['c2'], MUS)
        u_poly = calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
        bestfit_params_wl['u'] = u_poly
        if context.transit_engine == 'harmonica':
            bestfit_params_wl['c_ld'] = bestfit_params_wl['c1']
            bestfit_params_wl['alpha_ld'] = bestfit_params_wl['c2']
    else:
        if 'u' in wl_samples:
            _set_parameter_stats('u', wl_samples['u'], axis=0, bestfit_params_wl=bestfit_params_wl)
        else:
            _set_fixed_parameter_stats('u', U_mu_wl, bestfit_params_wl=bestfit_params_wl)
        if 'q1' in wl_samples and 'q2' in wl_samples:
            _set_parameter_stats('q1', wl_samples['q1'], axis=0, bestfit_params_wl=bestfit_params_wl)
            _set_parameter_stats('q2', wl_samples['q2'], axis=0, bestfit_params_wl=bestfit_params_wl)
        elif context.ld_uniform_basis == 'kipping':
            from models.model_building import kipping_q_from_u
            q_fixed = kipping_q_from_u(jnp.asarray(U_mu_wl))
            _set_fixed_parameter_stats('q1', q_fixed[0], bestfit_params_wl=bestfit_params_wl)
            _set_fixed_parameter_stats('q2', q_fixed[1], bestfit_params_wl=bestfit_params_wl)
        if context.transit_engine == 'harmonica':
            if 'u1' in wl_samples and 'u2' in wl_samples:
                _set_parameter_stats('u1', wl_samples['u1'], axis=0, bestfit_params_wl=bestfit_params_wl)
                _set_parameter_stats('u2', wl_samples['u2'], axis=0, bestfit_params_wl=bestfit_params_wl)
            else:
                _set_fixed_parameter_stats('u1', U_mu_wl[0], bestfit_params_wl=bestfit_params_wl)
                _set_fixed_parameter_stats('u2', U_mu_wl[1], bestfit_params_wl=bestfit_params_wl)
            bestfit_params_wl['u1_ld'] = bestfit_params_wl['u1']
            bestfit_params_wl['u2_ld'] = bestfit_params_wl['u2']
    _set_fixed_parameter_stats('ecc', context.HARMONICA_ECC, bestfit_params_wl=bestfit_params_wl)
    _set_fixed_parameter_stats('omega', context.HARMONICA_OMEGA, bestfit_params_wl=bestfit_params_wl)
    period_draws = _planet_site_array(wl_samples, 'period', context.n_planets)
    if period_draws is not None:
        # (n_planets, draws) -> (draws, n_planets)
        period_draws = period_draws.T
        _set_parameter_stats('period', period_draws, axis=0, bestfit_params_wl=bestfit_params_wl)
        bestfit_params_wl['period_err'] = jnp.std(period_draws, axis=0)
        debug(
            "White-light period posterior (median, -err, +err): "
            f"{np.asarray(bestfit_params_wl['period']).tolist()}, "
            f"{np.asarray(bestfit_params_wl['period_err_low']).tolist()}, "
            f"{np.asarray(bestfit_params_wl['period_err_high']).tolist()}"
        )
    else:
        _set_fixed_parameter_stats('period', context.PERIOD_FIXED, bestfit_params_wl=bestfit_params_wl)
    for surface_name in (
        'eclipse_depth', 'phase_c1', 'phase_d1', 'stellar_spot_contrast',
    ):
        if surface_name in wl_samples:
            _set_parameter_stats(surface_name, wl_samples[surface_name], bestfit_params_wl=bestfit_params_wl)
    if 'phase_c1' in wl_samples:
        from models.jaxoplanet.surface import phase_curve_derived_quantities
        derived = phase_curve_derived_quantities(
            wl_samples['eclipse_depth'], wl_samples['phase_c1'], wl_samples['phase_d1'],
        )
        for derived_name, derived_samples in derived.items():
            _set_parameter_stats(derived_name, np.asarray(derived_samples), bestfit_params_wl=bestfit_params_wl)
    if context.transit_engine == 'harmonica':
        for harmonic_name in context.HARMONICA_ODD_HARMONICS:
            if harmonic_name in wl_samples:
                _set_parameter_stats(harmonic_name, wl_samples[harmonic_name], bestfit_params_wl=bestfit_params_wl)

    # Post-hoc geometry derivation (engine-agnostic).
    derived = context.derive_geometry(wl_samples, context.PERIOD_FIXED, ecc=context.HARMONICA_ECC, omega=context.HARMONICA_OMEGA)

    series = {
        'duration': [derived[f'duration_{i}'] for i in range(context.n_planets)],
        't0': [wl_samples[f't0_{i}'] for i in range(context.n_planets)],
        'b': [derived[f'b_{i}'] for i in range(context.n_planets)],
        'rors': [wl_samples[f'rors_{i}'] for i in range(context.n_planets)],
        'depths': [wl_samples[f'rors_{i}']**2 for i in range(context.n_planets)],
        'cos_i': [derived[f'cos_i_{i}'] for i in range(context.n_planets)],
        'a_rs': [derived[f'a_rs_{i}'] for i in range(context.n_planets)],
    }
    for name, draws in series.items():
        summaries = [get_asym_errors(values) for values in draws]
        for index, suffix in enumerate(('', '_err_low', '_err_high')):
            bestfit_params_wl[name + suffix] = jnp.array([
                summary[index] for summary in summaries
            ])
        bestfit_params_wl[name + '_err'] = jnp.array([jnp.std(values) for values in draws])
    # The reported depth is the squared median radius, not median depth.
    bestfit_params_wl['depths'] = bestfit_params_wl['rors']**2

    names = ['c'] if context.detrending_type != 'none' else []
    names += [name for name in ('v', 'v2', 'v3', 'v4') if name in wl_samples]
    for trend, parameters in (
        ('explinear', ('A', 'tau')),
        ('spot', ('spot_amp', 'spot_mu', 'spot_sigma')),
        ('2spot', ('spot_amp2', 'spot_mu2', 'spot_sigma2')),
        ('linear_discontinuity', ('t_jump', 'jump', 'width', 'width_minutes')),
        ('gp', ('GP_log_sigma', 'GP_log_rho')),
    ):
        if trend in context.detrending_type:
            names.extend(parameters)
            if trend == 'linear_discontinuity' and 'log_width' in wl_samples:
                names.append('log_width')
    for name in names:
        _set_parameter_stats(name, wl_samples[name], bestfit_params_wl=bestfit_params_wl)
    for index in range(context.n_planets):
        if getattr(context, 'joint_member_products', False):
            break
        if context.n_planets > 1:
            info(f'  Planet {index + 1}')
        for label, stem, unit in (('rprs', 'rors', ''), ('b', 'b', ''),
                                  ('t0', 't0', TIME_SYSTEM), ('duration', 'duration', 'd'),
                                  ('period', 'period', 'd'), ('a_rs', 'a_rs', '')):
            specs = context.planet_parameter_specs.get(label)
            if specs is None or not specs[index].free:
                continue
            value = float(bestfit_params_wl[stem][index])
            sd = float(bestfit_params_wl[stem + '_err'][index])
            info(parameter_summary(label, value, sd, unit))
        for label in ('eclipse_depth', 'phase_c1', 'phase_d1', 'stellar_spot_contrast'):
            if label in wl_samples:
                values = np.asarray(wl_samples[label])
                scale, unit = (1e6, 'ppm') if label == 'eclipse_depth' else (1., '')
                info(parameter_summary(label.replace('_', ' '),
                    float(np.median(values)) * scale, float(np.std(values)) * scale, unit))
    return bestfit_params_wl


def _set_parameter_stats(
    name, data_samples, axis=0, *, bestfit_params_wl,
):
    med, low, high = get_asym_errors(data_samples, axis=axis)
    bestfit_params_wl[name] = med
    bestfit_params_wl[f'{name}_err_low'] = low
    bestfit_params_wl[f'{name}_err_high'] = high


def _set_fixed_parameter_stats(
    name, values, *, bestfit_params_wl,
):
    values = jnp.asarray(values)
    zeros = jnp.zeros_like(values)
    bestfit_params_wl[name] = values
    bestfit_params_wl[f'{name}_err'] = zeros
    bestfit_params_wl[f'{name}_err_low'] = zeros
    bestfit_params_wl[f'{name}_err_high'] = zeros


def _add_scalar_parameter(
    name, *, bestfit_params_wl, row,
):
    if name in bestfit_params_wl:
        row[name] = bestfit_params_wl[name]
        if f'{name}_err_low' in bestfit_params_wl:
            row[f'{name}_err_low'] = bestfit_params_wl[f'{name}_err_low']
            row[f'{name}_err_high'] = bestfit_params_wl[f'{name}_err_high']


def _white_light_jitter_metric(preparation, sites=('log_jitter',)):
    """Allow for a broad low-jitter tail in the otherwise local metric.

    For the default bounded log jitter, the likelihood can flatten towards
    its lower bound. The Hessian at the mode then understates its variance.
    Integrate a conditional slice of the original unconstrained posterior;
    widen just this covariance coordinate, retaining its correlations. This
    changes the proposal metric only, including for dispersed chain starts.
    A joint white light passes one site per power-2 member; each is sliced
    with the others held at the mode.
    """
    from dataclasses import replace
    from jax.flatten_util import ravel_pytree

    sites = [site for site in sites if site in preparation.unconstrained_map]
    if not sites:
        return preparation, {}
    flat, unravel = ravel_pytree(preparation.unconstrained_map)
    covariance = preparation.inverse_mass_matrix
    scale = jnp.ones_like(flat)
    offsets = jnp.linspace(-16., 16., 257)
    diagnostics = {}
    for site in sites:
        mask, _ = ravel_pytree({
            name: jnp.full_like(value, int(name == site))
            for name, value in preparation.unconstrained_map.items()
        })
        index = int(jnp.argmax(mask))
        values = jax.jit(lambda offsets: jax.lax.map(
            lambda offset: preparation.potential_fn(unravel(flat.at[index].add(offset))),
            offsets,
        ))(offsets)
        weights = jax.nn.softmax(-values)
        mean = jnp.sum(weights * offsets)
        variance = jnp.sum(weights * (offsets - mean)**2)
        local_variance = covariance[index, index]
        if not bool(jnp.isfinite(variance)):
            continue
        scale = scale.at[index].set(jnp.sqrt(jnp.maximum(1., variance / local_variance)))
        suffix = '' if site == 'log_jitter' else '[' + site.rsplit('/', 1)[0] + ']'
        diagnostics[f'laplace_jitter_local_variance{suffix}'] = float(local_variance)
        diagnostics[f'laplace_jitter_slice_variance{suffix}'] = float(variance)
        diagnostics[f'laplace_jitter_metric_variance{suffix}'] = float(local_variance * scale[index]**2)
    if not diagnostics:
        return preparation, {}
    covariance = covariance * scale[:, None] * scale[None, :]
    return replace(preparation, inverse_mass_matrix=covariance), diagnostics


def _white_light_geometry_coordinates(preparation):
    """Follow the even-in-b geometry ridge when its curvature is resolved.

    A local covariance regresses other coordinates linearly on signed b.
    Near a central transit the same relation is quadratic: the two signs
    have identical light curves. Replace the linear regression by a fixed
    quadratic shear of the unconstrained coordinate. Its Jacobian determinant
    is one, so the original potential needs no density correction.
    """
    from dataclasses import replace
    from jax.flatten_util import ravel_pytree

    sites = [name for name in preparation.unconstrained_map
             if name.startswith('_b_') and name[3:].isdigit()]
    if len(sites) != 1 or getattr(preparation, 'postprocess_fn', None) is None:
        return preparation, False, {}
    flat, unravel = ravel_pytree(preparation.unconstrained_map)
    mask, _ = ravel_pytree({
        name: jnp.full_like(value, int(name == sites[0]))
        for name, value in preparation.unconstrained_map.items()
    })
    index = int(jnp.argmax(mask))
    covariance = preparation.inverse_mass_matrix
    center = float(flat[index])
    if abs(center) < 1e-6:
        return preparation, False, {}
    relative_width = float(jnp.sqrt(covariance[index, index]) / abs(center))
    diagnostics = {'laplace_b_relative_width': relative_width}
    # The quadratic correction exceeds 10% of the linear displacement over
    # one local standard deviation. Narrow, already healthy fits retain
    # their original coordinates, metric, starts and random keys.
    if relative_width <= 0.2:
        return preparation, False, diagnostics
    coefficient = (covariance[:, index] / (2 * center * covariance[index, index])).at[index].set(0.)

    def original_coordinates(position):
        values, _ = ravel_pytree(position)
        return unravel(values + coefficient * (values[index]**2 - center**2))

    inverse_jacobian = jnp.eye(flat.size) - jnp.outer(
        2 * center * coefficient, jnp.eye(flat.size)[index],
    )
    transformed_covariance = inverse_jacobian @ covariance @ inverse_jacobian.T
    diagnostics['laplace_b_quadratic_shear'] = True
    diagnostics['laplace_b_reflection_site'] = sites[0]
    return replace(
        preparation,
        inverse_mass_matrix=transformed_covariance,
        potential_fn=lambda position: preparation.potential_fn(original_coordinates(position)),
        postprocess_fn=lambda position: preparation.postprocess_fn(original_coordinates(position)),
    ), True, diagnostics


class _FoldedImpactNUTS(numpyro.infer.NUTS):
    """Mix a Metropolis sign reflection with NUTS in the curved coordinates.

    The folded impact parameter has two equivalent signs. Local HMC can
    mix the physical geometry well while remaining in one sign for many
    draws. A symmetric reflection proposal visits both copies. Evaluate
    its actual potential and gradient: the Metropolis correction preserves
    the target even with floating-point asymmetry or a non-even potential.
    """

    def __init__(self, model=None, *, folded_site='_b_0', **kwargs):
        super().__init__(model, **kwargs)
        self._folded_site = folded_site

    def sample(self, state, model_args, model_kwargs):
        state = super().sample(state, model_args, model_kwargs)
        batched = state.z[self._folded_site].ndim > 0
        if batched:
            keys = jax.vmap(lambda key: jax.random.split(key, 2))(state.rng_key)
            next_key, move_key = keys[:, 0], keys[:, 1]
            uniform = jax.vmap(jax.random.uniform)(move_key)
        else:
            next_key, move_key = jax.random.split(state.rng_key)
            uniform = jax.random.uniform(move_key)
        proposal = {**state.z, self._folded_site: -state.z[self._folded_site]}
        evaluate = jax.value_and_grad(self._potential_fn)
        potential, gradient = jax.vmap(evaluate)(proposal) if batched else evaluate(proposal)
        take = uniform < .5 * jnp.exp(jnp.minimum(0., state.potential_energy - potential))

        def choose(new, old):
            shape = take.shape + (1,) * (new.ndim - take.ndim)
            return jnp.where(take.reshape(shape), new, old)

        delta = jnp.where(take, potential - state.potential_energy, 0.)
        return state._replace(
            z=jax.tree.map(choose, proposal, state.z),
            z_grad=jax.tree.map(choose, gradient, state.z_grad),
            potential_energy=jnp.where(take, potential, state.potential_energy),
            energy=state.energy + delta, rng_key=next_key,
        )


def _sample_white_light(
    *, context, hyper_params_wl, science_sites, soln, whitelight_mcmc_kwargs,
    whitelight_model_for_run,
):
    """Sample white light with the production metric, quality gate, and fallback."""
    reuse_path = (context.cfg.get('flags') or {}).get('whitelight_reuse_samples')
    if reuse_path:
        # Opt-in: reuse a previous run's saved ``whitelight_samples_by_chain.npz``
        # as the white-light posterior. No sampling and no quality gate; the
        # user takes responsibility for that posterior's convergence.
        from models.mcmc_diagnostics import PosteriorSamples, pool_chains
        with np.load(os.path.expanduser(str(reuse_path))) as saved:
            grouped = {name: jnp.asarray(saved[name]) for name in saved.files}
        chains, draws = next(iter(grouped.values())).shape[:2]
        info('Reusing a supplied white-light posterior without rechecking convergence.')
        debug(
            "WARNING: flags.whitelight_reuse_samples is set; using the saved "
            f"white-light posterior {reuse_path} ({chains} chains x {draws} draws) "
            "without resampling or re-checking its quality gate."
        )
        np.savez_compressed(state_path(context.output_dir, 'whitelight_samples_by_chain.npz'),
                            **{name: np.asarray(value) for name, value in grouped.items()})
        return PosteriorSamples(pool_chains(grouped), num_chains=chains)
    soln = _repair_gaussian_ld_start(
        soln, ld_profile=context.ld_profile,
        ld_prior_mode=getattr(context, 'ld_prior_mode', None),
        ld_uniform_basis=getattr(context, 'ld_uniform_basis', 'uplus_uminus'),
    )
    wl_nuts_kwargs = {
        "init_strategy": numpyro.infer.init_to_value(values=soln),
        **(context.harmonica_wl_nuts_kwargs if context.transit_engine == 'harmonica' else context.NUTS_KWARGS),
    }
    laplace_preparation = None
    laplace_diagnostics = {}
    wl_run_init_params = None
    adaptive_wl_nuts_kwargs = dict(wl_nuts_kwargs)
    adaptive_wl_mcmc_kwargs = dict(whitelight_mcmc_kwargs)
    wl_sampler_model = whitelight_model_for_run
    wl_nuts_type = numpyro.infer.NUTS
    wl_postprocess_kwargs = {}
    if (
        context.transit_engine in {'jaxoplanet', 'harmonica'}
        and context.whitelight_laplace_options["mass_matrix"] == "laplace"
    ):
        from models.independent_nuts import prepare_laplace_metric

        preparation_start = time.perf_counter()
        laplace_preparation = prepare_laplace_metric(
            whitelight_model_for_run,
            context.key_master,
            soln,
            context.data.wl_time,
            context.data.wl_flux_err,
            model_kwargs={'y': context.data.wl_flux, 'prior_params': hyper_params_wl},
            hessian_method=context.whitelight_laplace_options["hessian_method"],
            # Reuse compiled derivative/candidate programs across MAP steps;
            # keep the same target, Hessian, floor and line-search candidates.
            map_execution="split",
            trust_radius=context.whitelight_laplace_options["trust_radius"],
            max_iterations=200,
            # White-light time/flux coordinates naturally span a
            # wider curvature range than per-channel spectroscopy;
            # the 1e-8 lane floor would truncate real geometry modes.
            eigenvalue_floor=1.0e-12,
            # Surface models and long direct power-2 series have large
            # quadrature intermediates. Sequential finite-difference
            # and line-search points bound memory without changing
            # the objective or candidate metric calculations.
            sequential_evaluations=(context.uses_surface_model or (
                context.ld_profile == "power2"
                and int(context.data.wl_time.size) > 5000
            )),
        )
        jitter_diagnostics = {}
        jitter_sites = getattr(context, 'jitter_metric_sites', None)
        if jitter_sites is None:
            jitter_sites = ['log_jitter'] if (
                context.ld_profile == 'power2'
                and getattr(context, 'ld_prior_mode', None) == 'stellarprior'
                and not (getattr(context, 'trend_priors', None) or {}).get('jitter')) else []
        if jitter_sites:
            laplace_preparation, jitter_diagnostics = _white_light_jitter_metric(
                laplace_preparation, jitter_sites)
        laplace_preparation, geometry_coordinates, geometry_diagnostics = _white_light_geometry_coordinates(
            laplace_preparation,
        )
        if geometry_coordinates:
            wl_sampler_model = None
            wl_nuts_type = _FoldedImpactNUTS
            wl_nuts_kwargs['potential_fn'] = laplace_preparation.potential_fn
            adaptive_wl_nuts_kwargs['potential_fn'] = laplace_preparation.potential_fn
            wl_nuts_kwargs['folded_site'] = geometry_diagnostics['laplace_b_reflection_site']
            adaptive_wl_nuts_kwargs['folded_site'] = geometry_diagnostics['laplace_b_reflection_site']
            wl_postprocess_kwargs['postprocess_fn'] = laplace_preparation.postprocess_fn
        laplace_preparation.inverse_mass_matrix.block_until_ready()
        preparation_wall = time.perf_counter() - preparation_start
        wl_nuts_kwargs.update(
            dense_mass=True, inverse_mass_matrix=laplace_preparation.inverse_mass_matrix,
            adapt_mass_matrix=False,
            target_accept_prob=context.whitelight_laplace_options['target_accept'],
            max_tree_depth=context.whitelight_laplace_options['max_tree_depth'],
        )
        whitelight_mcmc_kwargs = {
            **whitelight_mcmc_kwargs,
            "num_warmup": context.whitelight_laplace_options["warmup"],
        }
        wl_run_init_params = laplace_preparation.unconstrained_map
        laplace_diagnostics = {
            "mass_matrix": "laplace",
            "laplace_preparation_wall_seconds": preparation_wall,
            "map_gradient_norm": float(laplace_preparation.gradient_norm),
            "map_newton_decrement": float(laplace_preparation.newton_decrement),
            "map_iterations": int(laplace_preparation.iterations),
            "map_condition_number": float(laplace_preparation.condition_number),
            "map_hessian_min_eigenvalue": float(laplace_preparation.hessian_min_eigenvalue),
            "laplace_hessian_method": context.whitelight_laplace_options[
                "hessian_method"
            ],
            **jitter_diagnostics,
            **geometry_diagnostics,
        }
    chains = int(whitelight_mcmc_kwargs.get('num_chains', 1))
    if wl_run_init_params is not None and chains > 1:
        from models.independent_nuts import dispersed_laplace_starts
        wl_run_init_params, start_diagnostics = dispersed_laplace_starts(
            laplace_preparation, jax.random.fold_in(context.key_master, 81721), chains)
        start_diagnostics = {name: np.asarray(value).tolist()
                             for name, value in start_diagnostics.items()}
        laplace_diagnostics.update(start_diagnostics)
        debug(f'White-light independent starts: {start_diagnostics}', flush=True)
    mcmc = numpyro.infer.MCMC(
        wl_nuts_type(wl_sampler_model, **wl_nuts_kwargs), progress_bar=True,
        jit_model_args=True, **wl_postprocess_kwargs, **whitelight_mcmc_kwargs,
    )
    sampling_start = time.perf_counter()
    wl_run_args = (context.data.wl_time, context.data.wl_flux_err)
    wl_run_kwargs = {
        "y": context.data.wl_flux,
        "prior_params": hyper_params_wl,
        "init_params": wl_run_init_params,
        "extra_fields": ('diverging', 'accept_prob', 'potential_energy', 'num_steps'),
    }
    heading = getattr(context, 'console_sampling_heading', None)
    info(heading or f'White light: {getattr(context, "surface_config", {}).get("model", "transit")} with a {getattr(context, "detrending_type", "linear")} trend, '
         f'{whitelight_mcmc_kwargs.get("num_chains", 1)} chains of {whitelight_mcmc_kwargs.get("num_samples", 1000)} draws')
    mcmc.run(context.key_master, *wl_run_args, **wl_run_kwargs)
    jax.block_until_ready(next(iter(mcmc.get_samples().values())))
    laplace_diagnostics["initial_block_sampling_wall_seconds"] = (
        time.perf_counter() - sampling_start
    )
    if chains > 1:
        science_sites = dict(science_sites)
        # The HMC state contains latent sites only, excluding fixed and
        # deterministic values without a model-specific whitelist.
        nuisance = set(mcmc.last_state.z) - set(science_sites.values())
        science_sites.update({name: name for name in nuisance})
    (wl_samples, wl_samples_grouped, wl_extra_fields, wl_quality_diagnostics) = _continue_mcmc_until_geometry_gate(
        mcmc, context.key_master, wl_run_args, wl_run_kwargs, min_ess=400.0, max_divergences=0,
        max_extra_blocks=3, failfast_ess=50.0, science_sites=science_sites,
    )
    sampling_attempts = [dict(wl_quality_diagnostics)]
    laplace_diagnostics.update(wl_quality_diagnostics)
    laplace_diagnostics.setdefault("mass_matrix", "adaptive")
    if _retry_laplace_with_smaller_steps(
        wl_quality_diagnostics, laplace_diagnostics["mass_matrix"],
        wl_nuts_kwargs.get("target_accept_prob", 0.8),
    ):
        debug(
            f"White-light Laplace chain had {wl_quality_diagnostics['whitelight_divergences']} "
            "divergent transition(s); rerunning it once with target acceptance "
            f"{WHITELIGHT_RETRY_TARGET_ACCEPT} (smaller steps) before any fallback."
        )
        info('White-light sampling had divergences; retrying with smaller steps.')
        retry_key = jax.random.fold_in(context.key_master, 90211)
        retry_nuts_kwargs = {**wl_nuts_kwargs, "target_accept_prob": WHITELIGHT_RETRY_TARGET_ACCEPT}
        mcmc = numpyro.infer.MCMC(
            wl_nuts_type(wl_sampler_model, **retry_nuts_kwargs), progress_bar=True,
            jit_model_args=True, **wl_postprocess_kwargs, **whitelight_mcmc_kwargs,
        )
        mcmc.run(retry_key, *wl_run_args, **wl_run_kwargs)
        (wl_samples, wl_samples_grouped, wl_extra_fields, wl_quality_diagnostics) = _continue_mcmc_until_geometry_gate(
            mcmc, retry_key, wl_run_args, wl_run_kwargs, min_ess=400.0, max_divergences=0,
            max_extra_blocks=3, failfast_ess=50.0, science_sites=science_sites,
        )
        sampling_attempts.append(dict(wl_quality_diagnostics))
        laplace_diagnostics.update(wl_quality_diagnostics)
        laplace_diagnostics["whitelight_laplace_retry_target_accept"] = WHITELIGHT_RETRY_TARGET_ACCEPT
    if (
        not wl_quality_diagnostics["whitelight_quality_gate_passed"]
        and laplace_diagnostics["mass_matrix"] == "laplace"
    ):
        debug(
            "WARNING: LAPLACE WHITE-LIGHT CHAIN FAILED THE QUALITY "
            "GATE; AUTOMATICALLY FALLING BACK TO ADAPTIVE NUTS."
        )
        info('White-light convergence checks failed; retrying with adaptive NUTS.')
        _, adaptive_key = jax.random.split(context.key_master)
        # Retain the validated MAP and initial covariance, then let each
        # chain learn its own dense metric. Restarting from an identity
        # diagonal metric discards the useful preparation on correlated
        # geometry/limb-darkening posteriors.
        adaptive_wl_nuts_kwargs.update(
            dense_mass=True,
            inverse_mass_matrix=laplace_preparation.inverse_mass_matrix,
            adapt_mass_matrix=True,
            target_accept_prob=0.9,
        )
        mcmc = numpyro.infer.MCMC(
            wl_nuts_type(wl_sampler_model, **adaptive_wl_nuts_kwargs),
            progress_bar=True, jit_model_args=True, **wl_postprocess_kwargs, **adaptive_wl_mcmc_kwargs,
        )
        adaptive_run_kwargs = dict(wl_run_kwargs)
        adaptive_run_kwargs["init_params"] = wl_run_init_params
        mcmc.run(adaptive_key, *wl_run_args, **adaptive_run_kwargs)
        (wl_samples, wl_samples_grouped, wl_extra_fields, wl_quality_diagnostics) = _continue_mcmc_until_geometry_gate(
            mcmc, adaptive_key, wl_run_args, adaptive_run_kwargs, min_ess=400.0,
            max_divergences=0, max_extra_blocks=3, failfast_ess=0.0,
            science_sites=science_sites,
        )
        sampling_attempts.append(dict(wl_quality_diagnostics))
        laplace_diagnostics.update(wl_quality_diagnostics)
        laplace_diagnostics["mass_matrix"] = "adaptive_fallback"
        laplace_diagnostics["whitelight_laplace_fell_back"] = True
    if not wl_quality_diagnostics["whitelight_quality_gate_passed"]:
        debug(
            "WARNING: WHITE-LIGHT MCMC FAILED ITS ESS/DIVERGENCE GATE "
            "AFTER ALL EXTENSION BLOCKS."
        )
    laplace_diagnostics['whitelight_sampling_attempts'] = sampling_attempts
    laplace_diagnostics['sampling_wall_seconds'] = time.perf_counter() - sampling_start
    laplace_diagnostics['sampling_wall_scope'] = 'initial_warmup_draws_extensions_and_fallbacks'
    if chains > 1:
        np.savez_compressed(state_path(context.output_dir, 'whitelight_samples_by_chain.npz'),
                            **{name:np.asarray(value) for name,value in wl_samples_grouped.items()})
    _save_mcmc_diagnostics(
        mcmc, max_tree_depth=wl_nuts_kwargs.get('max_tree_depth', 10),
        output_path=state_path(context.output_dir, 'whitelight_mcmc_diagnostics.json'),
        extra_diagnostics=laplace_diagnostics, samples_override=wl_samples,
        extra_fields_override=wl_extra_fields,
    )
    white_light_verdict(wl_quality_diagnostics)
    if chains > 1 and not wl_quality_diagnostics['whitelight_quality_gate_passed']:
        raise RuntimeError(
            'White-light chains failed the bulk/tail ESS, R-hat, or divergence '
            'gate after all retries; diagnostics were saved. '
            'The spectroscopic fit cannot use this geometry.'
        )
    return wl_samples


WHITELIGHT_RETRY_TARGET_ACCEPT = 0.99


def _retry_laplace_with_smaller_steps(diagnostics, mass_matrix, target_accept):
    """Whether a failed Laplace white-light chain should be rerun once with smaller steps.

    Only divergences are a step-size problem: a chain that failed on ESS or
    R-hat alone goes straight to the existing fallback, and a chain already at
    the retry acceptance target has nothing smaller to try.
    """
    return (
        mass_matrix == "laplace"
        and not diagnostics["whitelight_quality_gate_passed"]
        and int(diagnostics.get("whitelight_divergences", 0)) > 0
        and float(target_accept) < WHITELIGHT_RETRY_TARGET_ACCEPT
    )


def _save_white_light_diagnostics(
    *, context, bestfit_params_wl, gp_solver, wl_gp_path, wl_mask_path, wl_samples,
):
    """Write fitted curves and residual diagnostics, and return the outlier mask."""
    model_eval_params_wl = _select_transit_eval_params(
        bestfit_params_wl, transit_engine=context.transit_engine,
        param_method=context.param_method, t=context.data.wl_time,
        jaxoplanet_kernel=context.jaxoplanet_kernel, ld_profile=context.ld_profile,
        surface_config=context.surface_config,
    )

    if 'gp' in context.detrending_type:
        gp_products = _compute_whitelight_gp_products(
            model_eval_params_wl, context.data.wl_time, bestfit_params_wl['error'],
            context.data.wl_flux, detrend_type=context.detrending_type, gp_solver=gp_solver,
        )
        mu = gp_products["mu"]
        var = gp_products["var"]
        wl_transit_model = mu
        gp_stochastic_component = gp_products["gp_stochastic_component"]

    else:
        try:
            wl_transit_model = resolve_detrend_kernel(context.detrending_type)(model_eval_params_wl, context.data.wl_time)
        except KeyError:
            info('The configured white-light model is not defined.')
            exit()

    wl_residual = context.data.wl_flux - wl_transit_model
    wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
    wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > context.whitelight_sigma * wl_sigma
    if not context.whitelight_sigma:  # 0 turns the clip off
        wl_mad_mask = jnp.zeros_like(wl_mad_mask)
    from .console import residual_summary
    residual_summary(float(jnp.std(wl_residual[~wl_mad_mask])) * 1e6,
                     int(jnp.sum(wl_mad_mask)), context.whitelight_sigma,
                     f'order {context.order}' if context.order is not None else context.instrument_full_str)
    if getattr(context, 'plots_mode', context.cfg.get('flags', {}).get('plots', 'full')) != 'minimal':
        try:
            plot_whitelight_corner(wl_samples, OutputPaths(context.output_dir).figure('whitelight', 'corner'),
                                  instrument_label=context.instrument_full_str)
        except Exception as error:
            info('The white-light corner figure could not be written; see koala.log.')
            debug(f'White-light corner plot skipped: {error}')

    t_masked = context.data.wl_time[~wl_mad_mask]
    f_masked = context.data.wl_flux[~wl_mad_mask]
    # The diagnostic evaluator carries time-axis-specific phase
    # offsets. Rebuild them after outlier clipping instead of reusing
    # the full-cadence arrays with ``t_masked``.
    masked_model_eval_params_wl = _select_transit_eval_params(
        bestfit_params_wl, transit_engine=context.transit_engine,
        param_method=context.param_method, t=t_masked,
        jaxoplanet_kernel=context.jaxoplanet_kernel, ld_profile=context.ld_profile,
        surface_config=context.surface_config,
    )

    # F = (1 + transit) * systematics, so detrending divides by the
    # systematics factor.
    if 'gp' in context.detrending_type:
        planet_model_masked = compute_transit_model_auto(masked_model_eval_params_wl, t_masked)
        mu_masked = mu[~wl_mad_mask]
        systematics_at_points = mu_masked / (1.0 + planet_model_masked)
        detrended_flux = f_masked / systematics_at_points
    else:
        trend = _trend_from_params_np(
            context.detrending_type, np.array(t_masked), bestfit_params_wl,
        )
        detrended_flux = f_masked / trend

    transit_only_model = compute_transit_model_auto(masked_model_eval_params_wl, t_masked) + 1.0
    wl_flux_err_full = np.broadcast_to(
        np.asarray(context.data.wl_flux_err, dtype=float),
        np.asarray(context.data.wl_time).shape,
    )
    dt = np.median(np.diff(context.data.wl_time)) * 86400
    residuals_arr = np.array(wl_residual[~wl_mad_mask])
    beta, _, _, _ = calculate_beta_metrics(residuals_arr, dt)
    mc_betas, _, _, _, _ = run_beta_monte_carlo(residuals_arr, dt, n_sims=500)

    mu_sim = float(np.mean(mc_betas))
    std_sim = float(np.std(mc_betas))
    z_score = (beta - mu_sim) / std_sim

    debug(
        "White-light beta summary: "
        f"measured={beta:.3f}, MC={mu_sim:.3f}+/-{std_sim:.3f}, "
        f"significance={z_score:.2f} sigma"
    )

    transit_only_full = np.asarray(
        compute_transit_model_auto(model_eval_params_wl, context.data.wl_time) + 1.0,
        dtype=float,
    )
    bestfit_model_full = np.asarray(wl_transit_model, dtype=float)
    # Systematics factor: full model divided by the transit-only model.
    trend_model_full = bestfit_model_full / transit_only_full
    white_light(context.data.wl_time, context.data.wl_flux, context.data.wl_flux_err,
                bestfit_model_full, transit_only_full,
                OutputPaths(context.output_dir).figure('whitelight'),
                label=context.instrument_full_str, event=context.surface_config.get("model", "transit"),
                t0=event_epoch(np.asarray(bestfit_params_wl['t0']).ravel()[0], context.PERIOD_FIXED,
                               context.surface_config.get("model", "transit")), outlier=wl_mad_mask)

    gp_flux_full = np.asarray(mu, dtype=float) if 'gp' in context.detrending_type else None
    gp_err_full = np.asarray(jnp.sqrt(var), dtype=float) if 'gp' in context.detrending_type else None
    gp_trend_full = np.asarray(gp_stochastic_component, dtype=float) if 'gp' in context.detrending_type else None
    save_whitelight_timeseries(
        time=context.data.wl_time,
        flux=context.data.wl_flux,
        flux_err=context.data.wl_flux_err,
        bestfit_model=bestfit_model_full,
        output_csv=OutputPaths(context.output_dir, context.instrument_full_str).table("whitelight", "lightcurve", "csv"),
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

    if 'gp' in context.detrending_type:
        _save_whitelight_gp_database(wl_gp_path, context.data.wl_flux, gp_products)
    return wl_mad_mask

def prepare_white_light_ld(context):
    """Build the ordinary member-specific white-light LD prior."""
    if context.explicit_ld is not None:
        U_mu_wl, U_sigma_wl = context._explicit_ld_grid(), None
    elif context.ld_prior_mode == 'stellarprior':
        stellar_builder = (get_or_build_quadratic_ld_prior if context.ld_profile == 'quadratic'
                           else get_or_build_power2_ld_prior)
        U_mu_wl, U_sigma_wl = stellar_builder(
            context.stellar_cfg, context.data.wavelengths_unbinned, 0.0, context.instrument,
            order=context.order, output_dir=context.output_dir, cache_label='whitelight',
        )
    else:
        compute_ld = lambda: get_limb_darkening(context.sld, context.data.wavelengths_unbinned, 0.0, context.instrument, order=context.order, ld_profile=context.ld_profile, ld_mu_min=(context.stellar_cfg.get('ld_mu_min', 0.2) if context.ld_prior_mode == 'sing' else None))
        sweep = getattr(context, 'execution_context', None)
        U_mu_wl = (compute_ld() if sweep is None else sweep.coefficients(
            (id(context.sld), context.data.wavelengths_unbinned, context.instrument,
             context.order, context.ld_profile,
             context.stellar_cfg.get('ld_mu_min', .2) if context.ld_prior_mode == 'sing' else None),
            compute_ld))
        U_sigma_wl = None

    return U_mu_wl, U_sigma_wl


def run_white_light_stage(context, *, posterior=None):
    _engine_wl_kw = context._engine_wl_kw
    is_gp_detrending = 'gp' in context.detrending_type
    gp_solver = None
    if is_gp_detrending:
        validate_gp_times(context.data.wl_time)
        gp_solver = resolve_gp_solver(context.cfg.get('gp_solver'))
        debug(f"White-light GP solver: {gp_solver}")
    # Latent white-light geometry sites and their starting values, resolved
    # from the parameter specifications.
    geometry_sites = _whitelight_geometry_sites(
        context.planet_parameter_specs, context.n_planets, context.param_method,
    )
    from .geometry import _whitelight_science_sites
    science_sites = _whitelight_science_sites(
        context.planet_parameter_specs, context.surface_config, context.param_method,
    )
    _engine_wl_kw = dict(_engine_wl_kw)
    _engine_wl_kw.update(
        gp_solver=gp_solver, gp_assume_sorted=True,
        parameter_priors=context.planet_parameter_specs,
    )

    U_mu_wl, U_sigma_wl = prepare_white_light_ld(context)

    wl_mask_path = state_path(context.output_dir, f"{context.instrument_full_str}_whitelight_outlier_mask.npy")
    wl_params_path = OutputPaths(context.output_dir, context.instrument_full_str).table("whitelight", "params", "csv")
    wl_gp_path = state_path(context.output_dir, f"{context.instrument_full_str}_whitelight_GP_database.csv")
    wl_limb_csv_path = OutputPaths(context.output_dir, context.instrument_full_str).table("whitelight", "limbs", "csv")
    wl_limb_samples_path = OutputPaths(context.output_dir, context.instrument_full_str).table("whitelight", "limb_samples", "npz")
    wl_manifest_path = state_path(context.output_dir, f"{context.instrument_full_str}_whitelight_artifacts.manifest.json")
    wl_geometry_handoff_path = (
        state_path(context.output_dir, f"{context.instrument_full_str}_whitelight_geometry_handoff.json")
    )
    wl_artifact_fingerprint = _science_artifact_fingerprint(
        "whitelight",
        {
            "config": _white_light_fingerprint_config(context.cfg),
            "time": context.data.wl_time,
            "flux": context.data.wl_flux,
            "flux_err": context.data.wl_flux_err,
            "wavelengths_unbinned": context.data.wavelengths_unbinned,
            "transit_engine": context.transit_engine,
            "ld_profile": context.ld_profile,
            "ld_prior_mode": context.ld_prior_mode,
            "detrending_type": context.detrending_type,
            "trend_parameterization": context.whitelight_trend_parameterization,
            "two_spot_ordering": context.whitelight_two_spot_ordering,
            "whitelight_mass_matrix": context.whitelight_mass_matrix,
            "param_method": context.param_method,
            "geometry_estimator": context.whitelight_geometry_estimator,
            "whitelight_sigma": context.whitelight_sigma,
            "ld_coefficients": U_mu_wl,
            "ld_uncertainties": U_sigma_wl,
            "step_width_mode": context.step_width_mode,
            "step_width_days": context.step_width_days,
            **({'sweep_model_identity': context.execution_context.science_identity()}
               if getattr(context, 'execution_context', None) is not None else {}),
            **(
                {"gp_model_revision": GP_MODEL_REVISION}
                if is_gp_detrending else {}
            ),
        },
    )
    if posterior is not None:
        wl_artifact_fingerprint = posterior["fingerprint"]
    wl_geometry_handoff = _load_whitelight_geometry_handoff(
        wl_geometry_handoff_path, expected_posterior_fingerprint=wl_artifact_fingerprint,
        expected_estimator=context.whitelight_geometry_estimator,
    )
    required_wl_limb_products_exist = bool(
        context.transit_engine != 'harmonica'
        or not context.HARMONICA_ODD_HARMONICS
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
            *((wl_gp_path,) if 'gp' in context.detrending_type else ()),
        ),
    )
    stringcheck = wl_artifact_set.is_reusable(
        extra_condition=required_wl_limb_products_exist and wl_geometry_handoff is not None,
    )
    if not stringcheck and (
        os.path.exists(wl_mask_path) or os.path.exists(wl_params_path)
    ):
        info('The previous white-light fit does not match this run; refitting it.')
        debug(
            "Existing white-light products or geometry handoff are unverified "
            "or do not match the current data/configuration; recomputing them."
        )

    if posterior is not None:
        stringcheck = False

    if not stringcheck:
        if posterior is None:
            debug('Fitting whitelight for outliers and bestfit parameters')
            hyper_params_wl, init_params_wl, wl_ld_mode, whitelight_model_for_run = _initialize_white_light(
                context=context, U_mu_wl=U_mu_wl, U_sigma_wl=U_sigma_wl,
                _engine_wl_kw=_engine_wl_kw, geometry_sites=geometry_sites,
            )

            _plot_white_light_start(
                context=context, U_mu_wl=U_mu_wl, hyper_params_wl=hyper_params_wl,
                init_params_wl=init_params_wl,
            )

            if 'gp' in context.detrending_type:
                debug("--- Running Pre-Fit with Linear Detrending to stabilize GP ---")
                whitelight_model_prefit = context.create_whitelight_model(
                    detrend_type='linear', n_planets=context.n_planets, ld_mode=wl_ld_mode,
                    step_width_mode=context.step_width_mode, **_engine_wl_kw,
                )
                init_params_prefit = init_params_wl.copy()
                init_params_prefit.pop('GP_log_sigma', None)
                init_params_prefit.pop('GP_log_rho', None)
                soln = optimx.optimize(whitelight_model_prefit, start=init_params_prefit)(
                    context.key_master, context.data.wl_time, context.data.wl_flux_err, y=context.data.wl_flux, prior_params=hyper_params_wl
                )
            elif getattr(context, 'execution_context', None) is not None and not context.uses_surface_model:
                # The following staged optimizer restarts from init_params_wl;
                # its first step discards this preliminary solve completely.
                soln = dict(init_params_wl)
            else:
                soln = optimx.optimize(whitelight_model_for_run, start=init_params_wl)(context.key_master, context.data.wl_time, context.data.wl_flux_err, y=context.data.wl_flux, prior_params=hyper_params_wl)


            soln = _optimize_white_light_start(
                context=context, U_mu_wl=U_mu_wl, geometry_sites=geometry_sites,
                hyper_params_wl=hyper_params_wl, init_params_wl=init_params_wl,
                is_gp_detrending=is_gp_detrending, soln=soln,
                whitelight_model_for_run=whitelight_model_for_run, wl_ld_mode=wl_ld_mode,
            )

            wl_samples = _sample_white_light(
                context=context, hyper_params_wl=hyper_params_wl, science_sites=science_sites,
                soln=soln, whitelight_mcmc_kwargs=context.whitelight_mcmc_kwargs,
                whitelight_model_for_run=whitelight_model_for_run,
            )
        else:
            debug("Writing white-light products from the joint posterior.")
            wl_samples = posterior["samples"]
            np.savez_compressed(
                state_path(context.output_dir, "whitelight_samples_by_chain.npz"),
                **posterior["grouped"],
            )
            import shutil
            shutil.copyfile(posterior["diagnostics_path"],
                            state_path(context.output_dir, "whitelight_mcmc_diagnostics.json"))

        sweep = getattr(context, 'execution_context', None)
        if sweep is not None:
            sweep.remember('white', wl_samples, law=context.ld_profile)

        bestfit_params_wl = _summarize_white_light(
            context=context, U_mu_wl=U_mu_wl, wl_samples=wl_samples,
        )

        wl_handoff_geometry = _geometry_from_white_light_medians(
            bestfit_params_wl, bestfit_params_wl['period']
        )
        wl_handoff_payload = {
            "estimator": "posterior_median",
            "posterior_fingerprint_sha256": wl_artifact_fingerprint,
            "transit_engine": context.transit_engine,
            "param_method": context.param_method,
            "num_retained_draws": _posterior_num_draws(wl_samples),
            "selected_flat_draw_index": None,
            "selected_chain_index": None,
            "selected_draw_index": None,
            "summed_data_log_likelihood": None,
            "selected_primitive_parameters": {},
            "geometry": wl_handoff_geometry,
        }
        wl_mad_mask = _save_white_light_diagnostics(
            context=context, bestfit_params_wl=bestfit_params_wl, gp_solver=gp_solver,
            wl_gp_path=wl_gp_path, wl_mask_path=wl_mask_path, wl_samples=wl_samples,
        )

        _save_white_light_parameter_table(
            context=context, bestfit_params_wl=bestfit_params_wl,
            wl_params_path=wl_params_path,
        )

        if context.uses_surface_model:
            white_surface_samples = {
                name: np.asarray(wl_samples[name])[:, None, ...]
                for name in (
                    *SURFACE_PARAMS,
                    'eclipse_depth_ppm', 'phase_c1_ppm', 'phase_d1_ppm',
                    'nightside_flux_ppm', 'phase_amplitude_ppm', 'phase_offset_deg',
                )
                if name in wl_samples
            }
            band = np.asarray(context.data.wavelengths_unbinned, dtype=float)
            save_surface_results(
                np.asarray([np.nanmean(band)]),
                np.asarray([0.5 * (np.nanmax(band) - np.nanmin(band))]), white_surface_samples,
                OutputPaths(context.output_dir, context.instrument_full_str).table("whitelight", "spectrum"),
                plots=getattr(context, "plots_mode", context.cfg.get("flags", {}).get("plots", "full")),
            )
        bestfit_params_wl_df = pd.read_csv(wl_params_path)
        if context.transit_engine == 'harmonica' and _has_harmonica_odd_samples(wl_samples):
            band_wl_raw = np.asarray(context.data.wavelengths_hr)
            band_wl_err_raw = np.asarray(context.data.wavelengths_err_hr)
            if band_wl_raw.size == 0:
                band_wl_raw = np.asarray(context.data.wavelengths_lr)
                band_wl_err_raw = np.asarray(context.data.wavelengths_err_lr)
            band_wl_arr, band_wl_err_arr = _coerce_wavelength_axis(
                band_wl_raw, band_wl_err_raw if band_wl_raw.size > 0 else None,
            )
            bandpass_min = float(np.nanmin(band_wl_arr - band_wl_err_arr))
            bandpass_max = float(np.nanmax(band_wl_arr + band_wl_err_arr))
            save_harmonica_limb_products(
                wavelengths=0.5 * (bandpass_min + bandpass_max),
                wavelength_err=0.5 * (bandpass_max - bandpass_min),
                rors_samples=wl_samples['rors_0'],
                harmonic_samples=_harmonica_sample_payload(wl_samples),
                csv_path=wl_limb_csv_path,
                limb_spectrum_path=(OutputPaths(context.output_dir).figure("whitelight", "limbs")
                                    if getattr(context, "plots_mode", context.cfg.get("flags", {}).get("plots", "full")) == "full" else None),
                transmission_strings_path=(OutputPaths(context.output_dir).figure("whitelight", "limb_outline")
                                           if getattr(context, "plots_mode", context.cfg.get("flags", {}).get("plots", "full")) == "full" else None),
                posterior_samples_path=wl_limb_samples_path,
                title_prefix=context.instrument_full_str + ' | White light',
                bandpass_min=bandpass_min,
                bandpass_max=bandpass_max,
            )

        wl_geometry_handoff = _write_whitelight_geometry_handoff(
            wl_geometry_handoff_path, wl_handoff_payload
        )
        debug(f'Saved white-light fixed-geometry handoff to {wl_geometry_handoff_path}.')
        wl_artifact_set.write_manifest()
    else:
        info('Reusing the white-light fit from a previous run')
        diagnostic_path = state_path(context.output_dir, 'whitelight_mcmc_diagnostics.json')
        try:
            with open(diagnostic_path) as stream:
                saved_diagnostics = json.load(stream)
            if 'whitelight_quality_gate_passed' in saved_diagnostics:
                white_light_verdict(saved_diagnostics)
        except (OSError, ValueError, TypeError):
            debug('White-light convergence summary is unavailable in the saved diagnostics.')
        wl_mad_mask = np.load(wl_mask_path)
        bestfit_params_wl_df = pd.read_csv(wl_params_path)
        sweep = getattr(context, 'execution_context', None)
        posterior_path = state_path(context.output_dir, 'whitelight_samples_by_chain.npz')
        if sweep is not None and os.path.exists(posterior_path):
            with np.load(posterior_path, allow_pickle=False) as grouped:
                restored = {name: value.reshape((-1,) + value.shape[2:])
                            for name, value in grouped.items()}
            sweep.remember('white', restored, law=context.ld_profile)
    return (wl_mad_mask, bestfit_params_wl_df, wl_geometry_handoff,)

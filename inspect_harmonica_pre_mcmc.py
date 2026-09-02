import argparse
import json
import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
HARMONICA_REPO = REPO_DIR.parent / "harmonica_bell-main"
MPL_CACHE = REPO_DIR / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if HARMONICA_REPO.is_dir() and str(HARMONICA_REPO) not in sys.path:
    sys.path.insert(0, str(HARMONICA_REPO))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro_ext.optim as optimx
import pandas as pd
from exotic_ld import StellarLimbDarkening
from jaxoplanet.experimental import calc_poly_coeffs

from createdatacube import SpectroData, process_spectroscopy_data
from fit_jwst import (
    HARMONICA_INIT_ODD_COEFF,
    _bin_time_series_numpy,
    _harmonica_cosi_from_b,
    _trend_from_params_np,
    bin_spectrodata_in_time,
    get_asym_errors,
    get_limb_darkening,
    get_samples,
    load_config,
    save_harmonica_limb_products,
)
from models.harmonica import create_whitelight_model
from models.common import compute_transit_model_auto
from models.harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_duration_from_geometry,
    harmonica_duration_from_cos_i,
    harmonica_geometry_is_valid,
    harmonica_odd_coeff_specs,
)
from models.trends import (
    compute_lc_2spot,
    compute_lc_cubic,
    compute_lc_explinear,
    compute_lc_linear,
    compute_lc_linear_discontinuity,
    compute_lc_none,
    compute_lc_quadratic,
    compute_lc_quartic,
    compute_lc_spot,
)


def _harmonica_frac_site(name):
    return f"{name}_frac"


def _harmonica_coeff_to_frac(coeff, radius):
    radius_arr = np.asarray(radius, dtype=float)
    scale = max(float(np.nanmin(np.abs(radius_arr))), 1e-6)
    return float(coeff) / scale


def _resolve_path(value, base_dir):
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _serialise(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _get_sanity_model(detrending_type, params, t_vals):
    if "gp" in detrending_type:
        return compute_lc_linear(params, t_vals)
    if detrending_type == "linear":
        return compute_lc_linear(params, t_vals)
    if detrending_type == "quadratic":
        return compute_lc_quadratic(params, t_vals)
    if detrending_type == "cubic":
        return compute_lc_cubic(params, t_vals)
    if detrending_type == "quartic":
        return compute_lc_quartic(params, t_vals)
    if detrending_type == "explinear":
        return compute_lc_explinear(params, t_vals)
    if detrending_type == "linear_discontinuity":
        return compute_lc_linear_discontinuity(params, t_vals)
    if detrending_type == "spot":
        return compute_lc_spot(params, t_vals)
    if detrending_type == "2spot":
        return compute_lc_2spot(params, t_vals)
    if detrending_type == "none":
        return compute_lc_none(params, t_vals)
    return compute_lc_linear(params, t_vals)


def _ensure_len(value, n_items):
    arr = jnp.atleast_1d(jnp.asarray(value, dtype=jnp.float64))
    if arr.size == 1 and n_items > 1:
        arr = jnp.repeat(arr, n_items)
    return arr


def _power2_to_poly_u(c1, c2, degree=12, n_mu=300):
    mus = jnp.linspace(0.0, 1.0, n_mu, endpoint=True)
    profile = 1.0 - c1 * (1.0 - mus ** c2)
    return calc_poly_coeffs(mus, profile, poly_degree=degree)


def _as_float_stats(samples):
    med, low, high = get_asym_errors(samples)
    return {
        "median": float(np.asarray(med)),
        "sd": float(np.asarray(np.std(samples))),
        "err_low": float(np.asarray(low)),
        "err_high": float(np.asarray(high)),
    }


def _band_mcmc_summary_row(label, wl_bundle, samples, transit_engine, period, harmonica_ecc, harmonica_omega):
    row = {
        "label": label,
        "wl_min": float(wl_bundle["wl_min"]),
        "wl_max": float(wl_bundle["wl_max"]),
        "wl_center": 0.5 * (float(wl_bundle["wl_min"]) + float(wl_bundle["wl_max"])),
        "wl_half_width": 0.5 * (float(wl_bundle["wl_max"]) - float(wl_bundle["wl_min"])),
        "n_wave": int(wl_bundle["n_wave"]),
        "n_time": int(np.asarray(wl_bundle["time"]).size),
    }

    rors_samples = np.asarray(samples["rors_0"])
    depth_samples = rors_samples ** 2
    row.update({
        "rors": _as_float_stats(rors_samples)["median"],
        "rors_sd": _as_float_stats(rors_samples)["sd"],
        "rors_err_low": _as_float_stats(rors_samples)["err_low"],
        "rors_err_high": _as_float_stats(rors_samples)["err_high"],
        "depth": _as_float_stats(depth_samples)["median"],
        "depth_sd": _as_float_stats(depth_samples)["sd"],
        "depth_err_low": _as_float_stats(depth_samples)["err_low"],
        "depth_err_high": _as_float_stats(depth_samples)["err_high"],
        "t0": _as_float_stats(np.asarray(samples["t0_0"]))["median"],
        "t0_sd": _as_float_stats(np.asarray(samples["t0_0"]))["sd"],
        "t0_err_low": _as_float_stats(np.asarray(samples["t0_0"]))["err_low"],
        "t0_err_high": _as_float_stats(np.asarray(samples["t0_0"]))["err_high"],
        "b": _as_float_stats(np.asarray(samples["b_0"]))["median"],
        "b_sd": _as_float_stats(np.asarray(samples["b_0"]))["sd"],
        "b_err_low": _as_float_stats(np.asarray(samples["b_0"]))["err_low"],
        "b_err_high": _as_float_stats(np.asarray(samples["b_0"]))["err_high"],
    })

    if "duration_0" in samples:
        duration_samples = np.asarray(samples["duration_0"])
    elif "logD_0" in samples:
        duration_samples = np.exp(np.asarray(samples["logD_0"]))
    else:
        duration_samples = None
    if duration_samples is not None:
        duration_stats = _as_float_stats(duration_samples)
        row.update({
            "duration": duration_stats["median"],
            "duration_sd": duration_stats["sd"],
            "duration_err_low": duration_stats["err_low"],
            "duration_err_high": duration_stats["err_high"],
        })

    if transit_engine == "harmonica":
        if "a_rs_0" in samples:
            a_rs_samples = np.asarray(samples["a_rs_0"])
        elif "log_a_rs_0" in samples:
            a_rs_samples = np.exp(np.asarray(samples["log_a_rs_0"]))
        elif duration_samples is not None:
            a_rs_samples = np.asarray(
                harmonica_a_rs_from_duration(
                    period,
                    duration_samples,
                    np.asarray(samples["b_0"]),
                    rors_samples,
                    ecc=harmonica_ecc,
                    omega=harmonica_omega,
                )
            )
        else:
            a_rs_samples = None
        if a_rs_samples is not None:
            a_rs_stats = _as_float_stats(a_rs_samples)
            row.update({
                "a_rs": a_rs_stats["median"],
                "a_rs_sd": a_rs_stats["sd"],
                "a_rs_err_low": a_rs_stats["err_low"],
                "a_rs_err_high": a_rs_stats["err_high"],
            })

        if "cos_i_0" in samples:
            cos_i_samples = np.asarray(samples["cos_i_0"])
        elif a_rs_samples is not None:
            cos_i_samples = np.asarray(
                _harmonica_cosi_from_b(
                    np.asarray(samples["b_0"]),
                    a_rs_samples,
                    harmonica_ecc,
                    harmonica_omega,
                )
            )
        else:
            cos_i_samples = None
        if cos_i_samples is not None:
            cos_i_stats = _as_float_stats(cos_i_samples)
            row.update({
                "cos_i": cos_i_stats["median"],
                "cos_i_sd": cos_i_stats["sd"],
                "cos_i_err_low": cos_i_stats["err_low"],
                "cos_i_err_high": cos_i_stats["err_high"],
            })

    for key in ("a1", "a1_frac", "a3", "a3_frac", "a5", "a5_frac", "c1", "c2", "c", "v", "error", "log_jitter"):
        if key in samples:
            stats = _as_float_stats(np.asarray(samples[key]))
            row[key] = stats["median"]
            row[f"{key}_sd"] = stats["sd"]
            row[f"{key}_err_low"] = stats["err_low"]
            row[f"{key}_err_high"] = stats["err_high"]
    return row


def _save_harmonica_map_limb_products(result, output_dir, instrument_full_str, suffix, title_prefix):
    if result.get("engine") != "harmonica":
        return None

    params_opt = result["params_opt"]
    wl_bundle = result["wl_bundle"]
    wl_center = 0.5 * (float(wl_bundle["wl_min"]) + float(wl_bundle["wl_max"]))
    wl_half_width = 0.5 * (float(wl_bundle["wl_max"]) - float(wl_bundle["wl_min"]))

    harmonic_samples = {}
    for harmonic_name in result["odd_harmonics"]:
        if harmonic_name in params_opt:
            harmonic_samples[harmonic_name] = np.asarray(
                [float(np.asarray(params_opt[harmonic_name]).ravel()[0])],
                dtype=float,
            )

    csv_path = output_dir / f"{instrument_full_str}_{suffix}_limb_spectra.csv"
    limb_fig_path = output_dir / f"{instrument_full_str}_{suffix}_limb_spectra.png"
    string_fig_path = output_dir / f"{instrument_full_str}_{suffix}_transmission_string.png"

    save_harmonica_limb_products(
        wavelengths=np.asarray([wl_center], dtype=float),
        wavelength_err=np.asarray([wl_half_width], dtype=float),
        rors_samples=np.asarray([float(np.asarray(params_opt["rors"]).ravel()[0])], dtype=float),
        harmonic_samples=harmonic_samples,
        csv_path=str(csv_path),
        limb_spectrum_path=str(limb_fig_path),
        transmission_strings_path=str(string_fig_path),
        title_prefix=title_prefix,
        bandpass_min=float(wl_bundle["wl_min"]),
        bandpass_max=float(wl_bundle["wl_max"]),
    )

    return {
        "csv": csv_path,
        "limb_figure": limb_fig_path,
        "string_figure": string_fig_path,
    }


def _build_preopt_physical_params(
    init_params,
    hyper_params_wl,
    period_fixed,
    prior_t0,
    prior_b,
    prior_rprs,
    prior_dur,
    harmonica_a_rs,
    harmonica_ecc,
    harmonica_omega,
    transit_engine,
    n_planets,
):
    params_eval = dict(hyper_params_wl)
    params_eval.update(init_params)

    params_eval["period"] = _ensure_len(params_eval.get("period", period_fixed), n_planets)
    params_eval["t0"] = _ensure_len(params_eval.get("t0", prior_t0), n_planets)
    params_eval["b"] = _ensure_len(params_eval.get("b", prior_b), n_planets)
    params_eval["rors"] = _ensure_len(params_eval.get("rors", prior_rprs), n_planets)

    if all(f"t0_{i}" in init_params for i in range(n_planets)):
        params_eval["t0"] = jnp.array([init_params[f"t0_{i}"] for i in range(n_planets)])
    if all(f"_b_{i}" in init_params for i in range(n_planets)):
        params_eval["b"] = jnp.array([jnp.abs(init_params[f"_b_{i}"]) for i in range(n_planets)])
    if all(f"rors_{i}" in init_params for i in range(n_planets)):
        params_eval["rors"] = jnp.array([init_params[f"rors_{i}"] for i in range(n_planets)])
    elif all(f"depths_{i}" in init_params for i in range(n_planets)):
        params_eval["rors"] = jnp.array([jnp.sqrt(init_params[f"depths_{i}"]) for i in range(n_planets)])

    if transit_engine == "harmonica":
        params_eval["a_rs"] = _ensure_len(params_eval.get("a_rs", harmonica_a_rs), n_planets)
        if all(f"log_a_rs_{i}" in init_params for i in range(n_planets)):
            params_eval["a_rs"] = jnp.array([jnp.exp(init_params[f"log_a_rs_{i}"]) for i in range(n_planets)])
        elif all(f"logD_{i}" in init_params for i in range(n_planets)):
            params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets)])
            params_eval["a_rs"] = harmonica_a_rs_from_duration(
                params_eval["period"],
                params_eval["duration"],
                params_eval["b"],
                params_eval["rors"],
                ecc=harmonica_ecc,
                omega=harmonica_omega,
            )
        params_eval["duration"] = harmonica_duration_from_geometry(
            params_eval["period"],
            params_eval["a_rs"],
            params_eval["b"],
            params_eval["rors"],
            ecc=harmonica_ecc,
            omega=harmonica_omega,
        )
        params_eval["cos_i"] = _harmonica_cosi_from_b(
            params_eval["b"], params_eval["a_rs"], harmonica_ecc, harmonica_omega
        )
        if "c1" in init_params:
            params_eval["c_ld"] = init_params["c1"]
        if "c2" in init_params:
            params_eval["alpha_ld"] = init_params["c2"]
    else:
        params_eval["duration"] = _ensure_len(params_eval.get("duration", prior_dur), n_planets)
        if all(f"logD_{i}" in init_params for i in range(n_planets)):
            params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets)])

    return params_eval


def _ensure_sanity_ld_params(params, transit_engine, ld_profile, u_mu_wl):
    params = dict(params)
    if transit_engine == "harmonica":
        if "c_ld" not in params and "c1" in params:
            params["c_ld"] = params["c1"]
        if "alpha_ld" not in params and "c2" in params:
            params["alpha_ld"] = params["c2"]
        return params

    if ld_profile == "power2":
        c1 = params.get("c1", u_mu_wl[0])
        c2 = params.get("c2", u_mu_wl[1])
        params["u"] = _power2_to_poly_u(jnp.asarray(c1), jnp.asarray(c2))
    elif "u" not in params:
        params["u"] = jnp.asarray(u_mu_wl, dtype=jnp.float64)
    return params


def _soln_to_physical_params(soln, base_params, hyper_params_wl, n_planets, transit_engine):
    params = dict(base_params)
    for key, value in soln.items():
        if key != "obs":
            params[key] = value

    def have_all(prefix):
        return all(f"{prefix}_{i}" in params for i in range(n_planets))

    have_log_a_rs = have_all("log_a_rs")
    have_logD = have_all("logD")

    if have_all("t0"):
        params["t0"] = jnp.array([params[f"t0_{i}"] for i in range(n_planets)])
    if have_all("b"):
        params["b"] = jnp.array([params[f"b_{i}"] for i in range(n_planets)])
    elif have_all("_b"):
        params["b"] = jnp.array([jnp.abs(params[f"_b_{i}"]) for i in range(n_planets)])
    if have_all("rors"):
        params["rors"] = jnp.array([params[f"rors_{i}"] for i in range(n_planets)])
    elif have_all("depths"):
        params["rors"] = jnp.array([jnp.sqrt(params[f"depths_{i}"]) for i in range(n_planets)])

    if transit_engine == "harmonica":
        if have_all("duration"):
            params["duration"] = jnp.array([params[f"duration_{i}"] for i in range(n_planets)])
        elif have_logD:
            params["duration"] = jnp.array([jnp.exp(params[f"logD_{i}"]) for i in range(n_planets)])
        if have_all("a_rs"):
            params["a_rs"] = jnp.array([params[f"a_rs_{i}"] for i in range(n_planets)])
        elif have_log_a_rs:
            params["a_rs"] = jnp.array([jnp.exp(params[f"log_a_rs_{i}"]) for i in range(n_planets)])
    else:
        if have_all("duration"):
            params["duration"] = jnp.array([params[f"duration_{i}"] for i in range(n_planets)])
        elif have_logD:
            params["duration"] = jnp.array([jnp.exp(params[f"logD_{i}"]) for i in range(n_planets)])

    for harmonic_name in ("a1", "a3", "a5"):
        frac_key = _harmonica_frac_site(harmonic_name)
        if harmonic_name not in params and frac_key in soln and "rors" in params:
            params[harmonic_name] = soln[frac_key] * jnp.maximum(jnp.min(jnp.asarray(params["rors"])), 1e-6)

    if transit_engine == "harmonica" and "rors" in params:
        if "duration" in params and "b" in params and (have_logD or "a_rs" not in params):
            params["a_rs"] = harmonica_a_rs_from_duration(
                params["period"],
                params["duration"],
                params["b"],
                params["rors"],
                ecc=hyper_params_wl["ecc"],
                omega=hyper_params_wl["omega"],
            )
        if "a_rs" in params and "b" in params and (have_log_a_rs or "duration" not in params):
            params["duration"] = harmonica_duration_from_geometry(
                params["period"],
                params["a_rs"],
                params["b"],
                params["rors"],
                ecc=hyper_params_wl["ecc"],
                omega=hyper_params_wl["omega"],
            )
        if "a_rs" in params and "b" in params:
            params["cos_i"] = _harmonica_cosi_from_b(
                params["b"], params["a_rs"], hyper_params_wl["ecc"], hyper_params_wl["omega"]
            )
        if "c1" in params and "c2" in params:
            params["c_ld"] = params["c1"]
            params["alpha_ld"] = params["c2"]
        elif "duration" not in params and "cos_i" in params:
            params["duration"] = harmonica_duration_from_cos_i(
                params["period"],
                params["a_rs"],
                params["cos_i"],
                params["rors"],
                ecc=hyper_params_wl["ecc"],
                omega=hyper_params_wl["omega"],
            )

    return params


def _count_fitted_params(detrending_type, transit_engine, n_planets, ld_profile, wl_ld_mode, odd_harmonics):
    detrend_components = set(detrending_type.split("+"))
    n_params = 4 * n_planets + 1  # t0, geometry term, _b, depths per planet + log_jitter

    if wl_ld_mode == "free":
        if ld_profile == "quadratic":
            n_params += 2
        elif ld_profile == "power2":
            n_params += 2

    has_offset_term = not detrend_components.isdisjoint(
        {"linear", "quadratic", "cubic", "quartic", "linear_discontinuity", "explinear", "spot", "2spot", "gp"}
    )
    if has_offset_term:
        n_params += 1

    has_linear = not detrend_components.isdisjoint(
        {"linear", "quadratic", "cubic", "quartic", "linear_discontinuity", "explinear", "spot", "2spot"}
    )
    if has_linear:
        n_params += 1
    if not detrend_components.isdisjoint({"quadratic", "cubic", "quartic"}):
        n_params += 1
    if not detrend_components.isdisjoint({"cubic", "quartic"}):
        n_params += 1
    if "quartic" in detrend_components:
        n_params += 1
    if "linear_discontinuity" in detrend_components:
        n_params += 2
    if "explinear" in detrend_components:
        n_params += 2
    if "spot" in detrend_components:
        n_params += 3
    if "2spot" in detrend_components:
        n_params += 6
    if "gp" in detrend_components:
        n_params += 2
    if transit_engine == "harmonica":
        n_params += len(odd_harmonics)
    return n_params


def _build_metrics(time, wl_flux, wl_flux_err, flux_init, flux_opt, transit_params_opt, detrending_type, n_fit_params):
    resid_init = np.asarray(wl_flux - flux_init)
    resid_opt = np.asarray(wl_flux - flux_opt)
    transit_signal_opt = np.asarray(compute_transit_model_auto(transit_params_opt, time))
    trend_opt = np.asarray(_trend_from_params_np(detrending_type, np.asarray(time), transit_params_opt))
    detrended_flux_opt = np.asarray(wl_flux) - trend_opt + 1.0
    sigma_opt = np.asarray(transit_params_opt.get("error", wl_flux_err))
    sigma_opt = np.where(np.isfinite(sigma_opt) & (sigma_opt > 0.0), sigma_opt, np.asarray(wl_flux_err))
    sigma_data = np.asarray(wl_flux_err)
    sigma_data = np.where(np.isfinite(sigma_data) & (sigma_data > 0.0), sigma_data, np.nanmedian(sigma_opt))
    chi2_init = float(np.nansum((resid_init / sigma_opt) ** 2))
    chi2_opt = float(np.nansum((resid_opt / sigma_opt) ** 2))
    chi2_data_opt = float(np.nansum((resid_opt / sigma_data) ** 2))
    n_time = int(np.asarray(time).size)
    dof = max(1, n_time - int(n_fit_params))
    loglike_opt = float(
        -0.5 * np.nansum((resid_opt / sigma_opt) ** 2 + np.log(2.0 * np.pi * sigma_opt ** 2))
    )
    return {
        "rms_init_ppm": float(1e6 * np.nanstd(resid_init)),
        "rms_opt_ppm": float(1e6 * np.nanstd(resid_opt)),
        "mad_init_ppm": float(1e6 * np.nanmedian(np.abs(resid_init - np.nanmedian(resid_init)))),
        "mad_opt_ppm": float(1e6 * np.nanmedian(np.abs(resid_opt - np.nanmedian(resid_opt)))),
        "chi2_init": chi2_init,
        "chi2_opt": chi2_opt,
        "chi2_red_opt": float(chi2_opt / dof),
        "chi2_dataerr_opt": chi2_data_opt,
        "chi2_red_dataerr_opt": float(chi2_data_opt / dof),
        "n_fit_params": int(n_fit_params),
        "dof": int(dof),
        "loglike_opt": loglike_opt,
        "aic_opt": float(2 * n_fit_params - 2 * loglike_opt),
        "bic_opt": float(n_fit_params * np.log(n_time) - 2 * loglike_opt),
        "model_depth_init_ppm": float(1e6 * (1.0 - np.nanmin(np.asarray(flux_init)))),
        "model_depth_opt_ppm": float(1e6 * (1.0 - np.nanmin(np.asarray(flux_opt)))),
        "transit_only_depth_opt_ppm": float(1e6 * (-np.nanmin(transit_signal_opt))),
        "rors2_depth_opt_ppm": float(1e6 * np.asarray(transit_params_opt["rors"])[0] ** 2),
        "rough_data_depth_ppm": float(1e6 * (1.0 - np.nanmin(np.asarray(wl_flux)))),
        "trend_span_opt_ppm": float(1e6 * (np.nanmax(trend_opt) - np.nanmin(trend_opt))),
        "detrended_depth_opt_ppm": float(1e6 * (1.0 - np.nanmin(detrended_flux_opt))),
    }


def _count_site_params(site_names):
    n_params = 0
    for site_name in site_names:
        n_params += 2 if site_name == "u" else 1
    return n_params


def _slugify(text):
    slug = str(text).strip().replace(" ", "_").replace("/", "_")
    for old, new in [(".", "p"), ("-", "_"), (":", "_")]:
        slug = slug.replace(old, new)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_").lower()


def _build_oot_mask(time, prior_t0, prior_dur):
    time = np.asarray(time, dtype=float)
    prior_t0 = np.atleast_1d(np.asarray(prior_t0, dtype=float))
    prior_dur = np.atleast_1d(np.asarray(prior_dur, dtype=float))
    in_transit = np.zeros_like(time, dtype=bool)
    for t0_i, dur_i in zip(prior_t0, prior_dur):
        in_transit |= (time >= (t0_i - 0.6 * dur_i)) & (time <= (t0_i + 0.6 * dur_i))
    return ~in_transit


def _extract_band_lightcurve(
    data,
    wl_min,
    wl_max,
    label,
    instrument,
    order,
    sld,
    ld_profile,
    prior_t0,
    prior_dur,
    timebin_settings=None,
):
    wavelengths = np.asarray(data.wavelengths_unbinned, dtype=float)
    flux_unbinned = np.asarray(data.flux_unbinned, dtype=float)
    band_mask = (wavelengths >= float(wl_min)) & (wavelengths <= float(wl_max))
    if not np.any(band_mask):
        raise ValueError(f"No unbinned wavelengths found in band {wl_min:.3f}-{wl_max:.3f} um.")

    time = np.asarray(data.wl_time, dtype=float)
    band_flux_raw = np.nansum(flux_unbinned[:, band_mask], axis=1)
    oot_mask = _build_oot_mask(time, prior_t0, prior_dur)
    if not np.any(oot_mask):
        raise ValueError(f"No out-of-transit points available to normalize band {label}.")

    band_norm = np.nanmedian(band_flux_raw[oot_mask])
    if not np.isfinite(band_norm) or band_norm == 0.0:
        raise ValueError(f"Could not normalize band {label}: invalid OOT median {band_norm}.")

    band_flux = band_flux_raw / band_norm
    band_flux_err_scalar = float(np.nanmedian(np.abs(np.diff(band_flux))))
    if not np.isfinite(band_flux_err_scalar) or band_flux_err_scalar <= 0.0:
        band_flux_err_scalar = float(np.nanstd(band_flux[oot_mask]))
    band_flux_err_scalar = max(band_flux_err_scalar, 1e-7)
    band_flux_err = np.full_like(band_flux, band_flux_err_scalar, dtype=float)

    if timebin_settings and timebin_settings.get("enabled", False):
        time, band_flux, band_flux_err, _ = _bin_time_series_numpy(
            time,
            band_flux,
            band_flux_err,
            dt_seconds=float(timebin_settings["dt_seconds"]),
            method=str(timebin_settings["method"]),
        )
        if band_flux_err is None:
            band_flux_err = np.full_like(band_flux, band_flux_err_scalar, dtype=float)

    if instrument == "NIRISS/SOSS":
        u_mu_band = get_limb_darkening(
            sld,
            wavelengths[band_mask],
            0.0,
            instrument,
            order=order,
            ld_profile=ld_profile,
        )
    else:
        u_mu_band = get_limb_darkening(
            sld,
            wavelengths[band_mask],
            0.0,
            instrument,
            ld_profile=ld_profile,
        )

    return {
        "label": label,
        "slug": _slugify(label),
        "wl_min": float(wl_min),
        "wl_max": float(wl_max),
        "n_wave": int(np.sum(band_mask)),
        "time": jnp.asarray(time, dtype=jnp.float64),
        "flux": jnp.asarray(band_flux, dtype=jnp.float64),
        "flux_err": jnp.asarray(band_flux_err, dtype=jnp.float64),
        "u": jnp.asarray(u_mu_band, dtype=jnp.float64),
    }


def _fixed_geometry_fit_sites(detrending_type, engine_name, wl_ld_mode, ld_profile, odd_harmonics, n_planets):
    detrend_components = set(detrending_type.split("+"))
    sites = [f"rors_{i}" for i in range(n_planets)]
    sites.append("log_jitter")

    has_offset_term = not detrend_components.isdisjoint(
        {"linear", "quadratic", "cubic", "quartic", "linear_discontinuity", "explinear", "spot", "2spot", "gp"}
    )
    if has_offset_term:
        sites.append("c")

    has_linear = not detrend_components.isdisjoint(
        {"linear", "quadratic", "cubic", "quartic", "linear_discontinuity", "explinear", "spot", "2spot"}
    )
    if has_linear:
        sites.append("v")
    if not detrend_components.isdisjoint({"quadratic", "cubic", "quartic"}):
        sites.append("v2")
    if not detrend_components.isdisjoint({"cubic", "quartic"}):
        sites.append("v3")
    if "quartic" in detrend_components:
        sites.append("v4")
    if "linear_discontinuity" in detrend_components:
        sites.extend(["t_jump", "jump"])
    if "explinear" in detrend_components:
        sites.extend(["A", "log_tau"])
    if "spot" in detrend_components:
        sites.extend(["spot_amp", "spot_mu", "spot_sigma"])
    if "2spot" in detrend_components:
        sites.extend(["spot_amp", "spot_mu", "spot_sigma", "spot_amp2", "spot_mu2", "spot_sigma2"])
    if "gp" in detrend_components:
        sites.extend(["GP_log_sigma", "GP_log_rho"])

    if wl_ld_mode == "free":
        if engine_name == "harmonica":
            sites.extend(["c1", "c2"])
        elif ld_profile == "quadratic":
            sites.append("u")
        elif ld_profile == "power2":
            sites.extend(["c1", "c2"])

    if engine_name == "harmonica":
        sites.extend([_harmonica_frac_site(name) for name in odd_harmonics])

    return sites


def _plot_diagnostic(
    output_path,
    time,
    wl_flux,
    flux_init,
    flux_opt,
    detrending_type,
    params_opt,
    metrics,
    transit_engine,
):
    time = np.asarray(time)
    wl_flux = np.asarray(wl_flux)
    flux_init = np.asarray(flux_init)
    flux_opt = np.asarray(flux_opt)

    trend_opt = np.asarray(_trend_from_params_np(detrending_type, time, params_opt))
    detrended_flux_opt = wl_flux - trend_opt + 1.0
    transit_only_flux_opt = np.asarray(compute_transit_model_auto(params_opt, time)) + 1.0
    resid_init = wl_flux - flux_init
    resid_opt = wl_flux - flux_opt

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.2, 1.5]},
    )

    axes[0].scatter(time, wl_flux, c="k", s=8, alpha=0.35, label="Data")
    axes[0].plot(time, flux_init, color="tab:red", lw=1.8, ls="--", label="Init")
    axes[0].plot(time, flux_opt, color="tab:blue", lw=2.2, label="MAP")
    axes[0].set_ylabel("Flux")
    title_bits = [f"{detrending_type}", transit_engine]
    if "a1" in params_opt:
        title_bits.append(f"a1={float(np.asarray(params_opt['a1']).ravel()[0]):+.5f}")
    title_bits.append(f"depth={metrics['transit_only_depth_opt_ppm']:.0f} ppm")
    axes[0].set_title("Pre-MCMC Full-Model Check: " + " | ".join(title_bits))
    axes[0].legend(loc="best")

    axes[1].scatter(time, detrended_flux_opt, c="0.25", s=8, alpha=0.35, label="Data - trend + 1")
    axes[1].plot(time, transit_only_flux_opt, color="tab:green", lw=2.1, label="Transit only (MAP)")
    axes[1].set_ylabel("Detrended Flux")
    axes[1].legend(loc="best")

    axes[2].axhline(0.0, color="0.5", lw=1.0, ls="--")
    axes[2].scatter(time, 1e6 * resid_init, c="tab:red", s=7, alpha=0.2, label="Init resid")
    axes[2].scatter(time, 1e6 * resid_opt, c="tab:blue", s=7, alpha=0.35, label="MAP resid")
    axes[2].set_xlabel("Time (BJD)")
    axes[2].set_ylabel("Resid. [ppm]")
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_engine_comparison(output_path, time, wl_flux, engine_results, title=None):
    time = np.asarray(time)
    wl_flux = np.asarray(wl_flux)
    colors = {"harmonica": "tab:blue", "jaxoplanet": "tab:orange"}

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 2.2, 1.5]},
    )

    axes[0].scatter(time, wl_flux, c="k", s=8, alpha=0.3, label="Data")
    for engine_name, result in engine_results.items():
        label = (
            f"{engine_name} MAP "
            f"(chi2_red={result['metrics']['chi2_red_opt']:.3f}, "
            f"RMS={result['metrics']['rms_opt_ppm']:.1f} ppm)"
        )
        axes[0].plot(time, np.asarray(result["flux_opt"]), color=colors.get(engine_name), lw=2.0, label=label)
    axes[0].set_ylabel("Flux")
    axes[0].set_title(title or "Pre-MCMC Engine Comparison: spherical vs first-order harmonica")
    axes[0].legend(loc="best")

    for engine_name, result in engine_results.items():
        params_opt = result["params_opt"]
        trend_opt = np.asarray(_trend_from_params_np(result["detrending_type"], time, params_opt))
        detrended_flux = wl_flux - trend_opt + 1.0
        transit_only_flux = np.asarray(compute_transit_model_auto(params_opt, time)) + 1.0
        axes[1].scatter(
            time,
            detrended_flux,
            s=7,
            alpha=0.15,
            color=colors.get(engine_name),
            label=f"{engine_name} detrended data",
        )
        axes[1].plot(
            time,
            transit_only_flux,
            color=colors.get(engine_name),
            lw=2.0,
            label=f"{engine_name} transit only",
        )
    axes[1].set_ylabel("Detrended Flux")
    axes[1].legend(loc="best")

    axes[2].axhline(0.0, color="0.5", lw=1.0, ls="--")
    for engine_name, result in engine_results.items():
        resid = np.asarray(wl_flux - result["flux_opt"])
        axes[2].scatter(
            time,
            1e6 * resid,
            s=7,
            alpha=0.35,
            color=colors.get(engine_name),
            label=f"{engine_name} residuals",
        )
    axes[2].set_xlabel("Time (BJD)")
    axes[2].set_ylabel("Resid. [ppm]")
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _planet_rows(params_opt, metrics, odd_harmonics):
    n_planets = len(np.atleast_1d(np.asarray(params_opt["period"])))
    rows = []
    for i in range(n_planets):
        row = {
            "planet_index": i,
            "period": float(np.asarray(params_opt["period"])[i]),
            "t0": float(np.asarray(params_opt["t0"])[i]),
            "duration": float(np.asarray(params_opt["duration"])[i]),
            "b": float(np.asarray(params_opt["b"])[i]),
            "rors": float(np.asarray(params_opt["rors"])[i]),
            "depth": float(np.asarray(params_opt["rors"])[i] ** 2),
        }
        if "a_rs" in params_opt:
            row["a_rs"] = float(np.asarray(params_opt["a_rs"])[i])
        if "cos_i" in params_opt:
            row["cos_i"] = float(np.asarray(params_opt["cos_i"])[i])
        if "c1" in params_opt:
            row["c1"] = float(np.asarray(params_opt["c1"]))
        if "c2" in params_opt:
            row["c2"] = float(np.asarray(params_opt["c2"]))
        if "c" in params_opt:
            row["c"] = float(np.asarray(params_opt["c"]))
        if "v" in params_opt:
            row["v"] = float(np.asarray(params_opt["v"]))
        for harmonic_name in odd_harmonics:
            if harmonic_name in params_opt:
                harmonic_value = np.asarray(params_opt[harmonic_name]).ravel()[0]
                row[harmonic_name] = float(harmonic_value)
        row.update(metrics)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run harmonica white-light MAP diagnostics, with optional two-band white-light MCMC products."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--base-path", help="Override config 'path'.")
    parser.add_argument("--input-dir", help="Override config 'input_dir'.")
    parser.add_argument("--fits-file", help="Override config 'fits_file'.")
    parser.add_argument("--output-dir", help="Override config 'output_dir'.")
    parser.add_argument("--force-reprocess", action="store_true", help="Rebuild spectroscopy cache even if it exists.")
    parser.add_argument(
        "--run-band-mcmc",
        action="store_true",
        help="Run white-light MCMC on the configured paper_bandpasses and save harmonica limb products.",
    )
    parser.add_argument("--band-mcmc-warmup", type=int, help="Override the band white-light MCMC warmup draws.")
    parser.add_argument("--band-mcmc-samples", type=int, help="Override the band white-light MCMC posterior draws.")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = load_config(str(cfg_path))

    instrument = cfg["instrument"]
    if instrument in ["NIRSPEC/G395H", "NIRSPEC/G395M", "NIRSPEC/PRISM", "NIRSPEC/G140H", "NIRSPEC/G235H"]:
        nrs = cfg["nrs"]
    elif instrument == "NIRISS/SOSS":
        order = cfg["order"]
    else:
        raise ValueError(f"Unsupported instrument: {instrument}")

    planet_cfg = cfg["planet"]
    stellar_cfg = cfg["stellar"]
    flags = cfg.get("flags", {})
    resolution = cfg.get("resolution", None)
    pixels = cfg.get("pixels", None)
    if resolution is None:
        if pixels is None:
            raise ValueError("Must specify either 'resolution' or 'pixels'.")
        bins = pixels
        high_resolution_bins = bins.get("high", None)
        low_resolution_bins = bins.get("low", None)
    elif pixels is None:
        bins = resolution
        high_resolution_bins = bins.get("high", None)
        low_resolution_bins = bins.get("low", None)
    else:
        raise ValueError("Specify only one of 'resolution' or 'pixels'.")

    outlier_clip = cfg.get("outlier_clip", {})
    planet_str = planet_cfg["name"]
    mask_integrations_start = outlier_clip.get("mask_integrations_start", None)
    mask_integrations_end = outlier_clip.get("mask_integrations_end", None)

    base_path_value = args.base_path if args.base_path is not None else cfg.get("path", ".")
    base_path = _resolve_path(base_path_value, cfg_dir)
    input_dir = _resolve_path(args.input_dir if args.input_dir is not None else cfg.get("input_dir", planet_str + "_NIRSPEC"), base_path)
    output_dir = _resolve_path(args.output_dir if args.output_dir is not None else cfg.get("output_dir", planet_str + "_RESULTS"), base_path)
    fits_file = _resolve_path(args.fits_file if args.fits_file is not None else cfg.get("fits_file"), input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detrending_type = flags.get("detrending_type", "linear")
    mask_start = flags.get("mask_start", False)
    mask_end = flags.get("mask_end", False)
    ld_profile = flags.get("ld_profile", "quadratic")
    transit_engine = flags.get("transit_engine", "jaxoplanet")
    max_harmonic_order = int(flags.get("harmonica_max_order", 1))
    harmonica_wl_parameterization = flags.get("param_method", flags.get("harmonica_wl_parameterization", "duration"))
    if harmonica_wl_parameterization not in {"a_rs", "duration"}:
        raise ValueError(
            "flags.param_method must be 'a_rs' or 'duration'. "
            f"Received '{harmonica_wl_parameterization}'."
        )
    harmonica_wl_dense_mass = bool(flags.get("harmonica_wl_dense_mass", True))
    harmonica_wl_regularize_mass_matrix = bool(
        flags.get("harmonica_wl_regularize_mass_matrix", True)
    )
    harmonica_wl_max_tree_depth = int(flags.get("harmonica_wl_max_tree_depth", 8))
    harmonica_wl_target_accept = float(flags.get("harmonica_wl_target_accept", 0.8))
    fix_ld = flags.get("fix_ld", False)
    spot_amp = flags.get("spot_amp", 0.0)
    spot_mu = flags.get("spot_center", 0.0)
    spot_sigma = flags.get("spot_width", 0.0)
    spot_amp2 = flags.get("spot_amp2", flags.get("spot_amp_2", 0.0))
    spot_mu2 = flags.get("spot_center2", flags.get("spot_center_2", 0.0))
    spot_sigma2 = flags.get("spot_width2", flags.get("spot_width_2", 0.0))
    t_jump_guess = flags.get("t_jump_guess", None)
    jump_guess = flags.get("jump_guess", 0.0)
    odd_harmonics = tuple(name for name, _ in harmonica_odd_coeff_specs(max_harmonic_order))

    host_device = cfg.get("host_device", "gpu").lower()
    if transit_engine == "harmonica" and host_device != "cpu":
        host_device = "cpu"
    jax.config.update("jax_platform_name", host_device)
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    periods = jnp.atleast_1d(planet_cfg["period"])
    n_planets = len(periods)
    prior_dur = jnp.atleast_1d(planet_cfg["duration"])
    prior_t0 = jnp.atleast_1d(planet_cfg["t0"])
    prior_b = jnp.atleast_1d(planet_cfg["b"])
    prior_rprs = jnp.atleast_1d(planet_cfg["rprs"])
    prior_depth = prior_rprs ** 2

    def _planet_cfg_array(key, default):
        arr = np.atleast_1d(np.asarray(planet_cfg.get(key, default), dtype=float))
        if arr.size == 1 and n_planets > 1:
            arr = np.repeat(arr, n_planets)
        elif arr.size != n_planets:
            raise ValueError(f"`planet.{key}` must be scalar or length {n_planets}, got shape {arr.shape}.")
        return jnp.asarray(arr, dtype=jnp.float64)

    if "a_rs" not in planet_cfg:
        raise KeyError("'a_rs' is required in planet config for harmonica engine")
    harmonica_a_rs = _planet_cfg_array("a_rs", None)
    harmonica_ecc = _planet_cfg_array("ecc", 0.0)
    harmonica_omega = _planet_cfg_array("omega", 0.0)
    harmonica_a_rs_prior_min = _planet_cfg_array("a_rs_prior_min", np.maximum(2.0, 0.5 * np.asarray(harmonica_a_rs)))
    harmonica_a_rs_prior_max = _planet_cfg_array("a_rs_prior_max", np.maximum(10.0, 2.0 * np.asarray(harmonica_a_rs)))

    cfg_duration_from_a_rs = harmonica_duration_from_geometry(
        periods,
        harmonica_a_rs,
        prior_b,
        prior_rprs,
        ecc=harmonica_ecc,
        omega=harmonica_omega,
    )
    cfg_a_rs_from_duration = harmonica_a_rs_from_duration(
        periods,
        prior_dur,
        prior_b,
        prior_rprs,
        ecc=harmonica_ecc,
        omega=harmonica_omega,
    )
    cfg_duration_delta = jnp.abs(cfg_duration_from_a_rs - prior_dur)
    cfg_a_rs_delta = jnp.abs(cfg_a_rs_from_duration - harmonica_a_rs)
    if transit_engine == "harmonica" and (
        bool(jnp.any(cfg_duration_delta > 1e-6)) or bool(jnp.any(cfg_a_rs_delta > 1e-6))
    ):
        print("Warning: harmonica config geometry is inconsistent between `duration` and `a_rs`.")
        for i in range(n_planets):
            print(
                f"  planet {i}: cfg duration={float(prior_dur[i]):.9f} d, "
                f"duration(a_rs)={float(cfg_duration_from_a_rs[i]):.9f} d, "
                f"cfg a_rs={float(harmonica_a_rs[i]):.6f}, "
                f"a_rs(duration)={float(cfg_a_rs_from_duration[i]):.6f}"
            )

    ld_data_path = _resolve_path(stellar_cfg.get("ld_data_path", "../exotic_ld_data"), cfg_dir)
    sld = StellarLimbDarkening(
        M_H=stellar_cfg["feh"],
        Teff=stellar_cfg["teff"],
        logg=stellar_cfg["logg"],
        ld_model=stellar_cfg.get("ld_model", "stagger"),
        ld_data_path=str(ld_data_path),
    )

    if instrument in ["NIRSPEC/G395H", "NIRSPEC/G395M", "NIRSPEC/PRISM", "NIRSPEC/G140H", "NIRSPEC/G235H"]:
        mini_instrument = f"nrs{nrs}"
    elif instrument == "NIRISS/SOSS":
        mini_instrument = f"order{order}"
    else:
        mini_instrument = ""

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    if bins == resolution:
        spectro_data_file = output_dir / f"{instrument_full_str}_spectroscopy_data_{low_resolution_bins}LR_{high_resolution_bins}HR.pkl"
    else:
        spectro_data_file = output_dir / f"{instrument_full_str}_spectroscopy_data_{low_resolution_bins}pix_{high_resolution_bins}pix.pkl"

    if args.force_reprocess or not spectro_data_file.exists() or mask_start is not False:
        data = process_spectroscopy_data(
            instrument,
            str(input_dir),
            str(output_dir),
            planet_str,
            cfg,
            str(fits_file),
            mask_start,
            mask_end,
            mask_integrations_start,
            mask_integrations_end,
        )
        data.save(str(spectro_data_file))
    else:
        data = SpectroData.load(str(spectro_data_file))

    timebin_cfg = cfg.get("time_binning", {})
    do_timebin = timebin_cfg.get("enabled", False) or flags.get("bin_time", False)
    dt_seconds = (
        timebin_cfg.get("dt_seconds", None)
        or flags.get("bin_dt_seconds", None)
        or 120.0
    )
    method = (
        timebin_cfg.get("method", None)
        or flags.get("bin_method", None)
        or "weighted"
    )
    if do_timebin:
        bin_whitelight = timebin_cfg.get("whitelight", True) if "whitelight" in timebin_cfg else flags.get("bin_whitelight", True)
        bin_spectroscopic = timebin_cfg.get("spectroscopic", True) if "spectroscopic" in timebin_cfg else flags.get("bin_spectroscopic", True)
        data = bin_spectrodata_in_time(
            data,
            dt_seconds=float(dt_seconds),
            method=str(method),
            bin_whitelight=bool(bin_whitelight),
            bin_spectroscopic=bool(bin_spectroscopic),
        )
    timebin_settings = {
        "enabled": bool(do_timebin),
        "dt_seconds": float(dt_seconds),
        "method": str(method),
    }

    if instrument in ["NIRSPEC/G395H", "NIRSPEC/G395M", "NIRSPEC/PRISM", "MIRI/LRS", "NIRSPEC/G140H", "NIRSPEC/G235H"]:
        u_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, ld_profile=ld_profile)
    elif instrument == "NIRISS/SOSS":
        u_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order, ld_profile=ld_profile)
    else:
        raise ValueError(f"Unsupported instrument for LD: {instrument}")

    hyper_params_wl = {
        "duration": prior_dur,
        "t0": prior_t0,
        "period": periods,
        "u": u_mu_wl,
        "a_rs": harmonica_a_rs,
        "a_rs_prior_min": harmonica_a_rs_prior_min,
        "a_rs_prior_max": harmonica_a_rs_prior_max,
        "ecc": harmonica_ecc,
        "omega": harmonica_omega,
    }
    if "2spot" in detrending_type:
        hyper_params_wl["spot_guess"] = spot_mu
        hyper_params_wl["spot_guess2"] = spot_mu2
    elif "spot" in detrending_type:
        hyper_params_wl["spot_guess"] = spot_mu
    if "linear_discontinuity" in detrending_type:
        hyper_params_wl["t_jump_guess"] = (
            t_jump_guess if t_jump_guess is not None else 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
        )
        hyper_params_wl["jump_guess"] = jump_guess

    full_wl_bundle = {
        "label": "full_order_whitelight",
        "slug": "full_order_whitelight",
        "wl_min": float(np.nanmin(np.asarray(data.wavelengths_unbinned))),
        "wl_max": float(np.nanmax(np.asarray(data.wavelengths_unbinned))),
        "n_wave": int(np.asarray(data.wavelengths_unbinned).size),
        "time": jnp.asarray(data.wl_time, dtype=jnp.float64),
        "flux": jnp.asarray(data.wl_flux, dtype=jnp.float64),
        "flux_err": jnp.asarray(data.wl_flux_err, dtype=jnp.float64),
        "u": jnp.asarray(u_mu_wl, dtype=jnp.float64),
    }

    wl_ld_mode = "fixed" if fix_ld else "free"
    paper_wl_ld_mode = str(flags.get("paper_wl_ld_mode", "free")).lower()
    if paper_wl_ld_mode not in {"free", "fixed"}:
        raise ValueError(f"Unsupported paper_wl_ld_mode: {paper_wl_ld_mode}")
    band_mcmc_cfg = cfg.get("inspect_band_mcmc", {})
    run_band_mcmc = bool(args.run_band_mcmc or band_mcmc_cfg.get("enabled", False))
    band_mcmc_num_warmup = int(
        args.band_mcmc_warmup
        if args.band_mcmc_warmup is not None
        else band_mcmc_cfg.get("num_warmup", 1000)
    )
    band_mcmc_num_samples = int(
        args.band_mcmc_samples
        if args.band_mcmc_samples is not None
        else band_mcmc_cfg.get("num_samples", 1000)
    )
    if transit_engine == "harmonica" and ld_profile != "power2":
        raise ValueError(
            "The harmonica engine requires flags.ld_profile: 'power2'. "
            f"Received '{ld_profile}'."
        )

    def _make_hyper_params(current_u, current_time, prior_t0_fit, prior_dur_fit, harmonica_a_rs_fit):
        hyper_params_fit = {
            "duration": prior_dur_fit,
            "t0": prior_t0_fit,
            "period": periods,
            "u": current_u,
            "a_rs": harmonica_a_rs_fit,
            "a_rs_prior_min": harmonica_a_rs_prior_min,
            "a_rs_prior_max": harmonica_a_rs_prior_max,
            "ecc": harmonica_ecc,
            "omega": harmonica_omega,
        }
        if "2spot" in detrending_type:
            hyper_params_fit["spot_guess"] = spot_mu
            hyper_params_fit["spot_guess2"] = spot_mu2
        elif "spot" in detrending_type:
            hyper_params_fit["spot_guess"] = spot_mu
        if "linear_discontinuity" in detrending_type:
            hyper_params_fit["t_jump_guess"] = (
                t_jump_guess if t_jump_guess is not None else 0.5 * (jnp.min(current_time) + jnp.max(current_time))
            )
            hyper_params_fit["jump_guess"] = jump_guess
        return hyper_params_fit

    def _build_init_params_wl(
        engine_name,
        wl_bundle,
        prior_t0_fit,
        prior_b_fit,
        prior_rprs_fit,
        prior_depth_fit,
        prior_dur_fit,
        harmonica_a_rs_fit,
        fixed_geometry_params=None,
    ):
        current_u = wl_bundle["u"]
        current_flux_err = wl_bundle["flux_err"]
        current_time = wl_bundle["time"]
        init_u = current_u if (engine_name == "harmonica" or ld_profile != "power2") else _power2_to_poly_u(current_u[0], current_u[1])
        init_params_wl = {
            "c": 1.0,
            "v": 0.0,
            "log_jitter": jnp.log(1e-4),
            "b": prior_b_fit,
            "rors": prior_rprs_fit,
            "u": init_u,
        }
        if engine_name == "harmonica":
            init_params_wl["c1"] = current_u[0]
            init_params_wl["c2"] = current_u[1]
            for harmonic_name in odd_harmonics:
                init_params_wl[_harmonica_frac_site(harmonic_name)] = (
                    _harmonica_coeff_to_frac(
                        float(np.asarray(fixed_geometry_params[harmonic_name]).ravel()[0]),
                        prior_rprs_fit,
                    )
                    if fixed_geometry_params is not None and harmonic_name in fixed_geometry_params
                    else _harmonica_coeff_to_frac(HARMONICA_INIT_ODD_COEFF, prior_rprs_fit)
                )

        for i in range(n_planets):
            if fixed_geometry_params is not None:
                if engine_name == "harmonica":
                    if harmonica_wl_parameterization == "a_rs":
                        init_params_wl[f"log_a_rs_{i}"] = jnp.log(np.asarray(fixed_geometry_params["a_rs"])[i])
                    else:
                        init_params_wl[f"logD_{i}"] = jnp.log(np.asarray(fixed_geometry_params["duration"])[i])
                else:
                    init_params_wl[f"logD_{i}"] = jnp.log(np.asarray(fixed_geometry_params["duration"])[i])
                init_params_wl[f"t0_{i}"] = np.asarray(fixed_geometry_params["t0"])[i]
                init_params_wl[f"_b_{i}"] = np.asarray(fixed_geometry_params["b"])[i]
            else:
                if engine_name == "harmonica":
                    if harmonica_wl_parameterization == "a_rs":
                        init_params_wl[f"log_a_rs_{i}"] = jnp.log(harmonica_a_rs_fit[i])
                    else:
                        init_params_wl[f"logD_{i}"] = jnp.log(prior_dur_fit[i])
                else:
                    init_params_wl[f"logD_{i}"] = jnp.log(prior_dur_fit[i])
                init_params_wl[f"t0_{i}"] = prior_t0_fit[i]
                init_params_wl[f"_b_{i}"] = prior_b_fit[i]
            init_params_wl[f"rors_{i}"] = prior_rprs_fit[i]

        if "quadratic" in detrending_type:
            init_params_wl["v2"] = 0.0
        if "cubic" in detrending_type:
            init_params_wl["v2"] = 0.0
            init_params_wl["v3"] = 0.0
        if "quartic" in detrending_type:
            init_params_wl["v2"] = 0.0
            init_params_wl["v3"] = 0.0
            init_params_wl["v4"] = 0.0
        if "explinear" in detrending_type:
            init_params_wl["A"] = 0.001
            init_params_wl["tau"] = 0.5
        if "gp" in detrending_type:
            init_params_wl["GP_log_sigma"] = jnp.log(jnp.nanmedian(current_flux_err))
            init_params_wl["GP_log_rho"] = jnp.log(0.1)
        if "linear_discontinuity" in detrending_type:
            init_params_wl["t_jump"] = (
                t_jump_guess if t_jump_guess is not None else 0.5 * (jnp.min(current_time) + jnp.max(current_time))
            )
            init_params_wl["jump"] = jump_guess
        if "spot" in detrending_type:
            init_params_wl["spot_amp"] = spot_amp
            init_params_wl["spot_mu"] = spot_mu
            init_params_wl["spot_sigma"] = spot_sigma
        if "2spot" in detrending_type:
            init_params_wl["spot_amp2"] = spot_amp2
            init_params_wl["spot_mu2"] = spot_mu2
            init_params_wl["spot_sigma2"] = spot_sigma2
        return init_params_wl

    def _run_map_fit(
        engine_name,
        wl_bundle,
        key_seed,
        prior_t0_fit,
        prior_b_fit,
        prior_rprs_fit,
        prior_depth_fit,
        prior_dur_fit,
        harmonica_a_rs_fit,
        fit_sites=None,
        fixed_geometry_params=None,
        wl_ld_mode_override=None,
    ):
        engine_odd_harmonics = odd_harmonics if engine_name == "harmonica" else tuple()
        engine_max_order = max_harmonic_order if engine_name == "harmonica" else 1
        active_wl_ld_mode = wl_ld_mode if wl_ld_mode_override is None else wl_ld_mode_override
        hyper_params_fit = _make_hyper_params(
            wl_bundle["u"], wl_bundle["time"], prior_t0_fit, prior_dur_fit, harmonica_a_rs_fit
        )
        init_params_wl = _build_init_params_wl(
            engine_name,
            wl_bundle,
            prior_t0_fit,
            prior_b_fit,
            prior_rprs_fit,
            prior_depth_fit,
            prior_dur_fit,
            harmonica_a_rs_fit,
            fixed_geometry_params=fixed_geometry_params,
        )

        whitelight_model_for_run = create_whitelight_model(
            detrend_type=detrending_type,
            n_planets=n_planets,
            ld_mode=active_wl_ld_mode,
            max_harmonic_order=engine_max_order,
            param_method=harmonica_wl_parameterization,
        )

        params_preopt = _build_preopt_physical_params(
            init_params_wl,
            hyper_params_fit,
            periods,
            prior_t0_fit,
            prior_b_fit,
            prior_rprs_fit,
            prior_dur_fit,
            harmonica_a_rs_fit,
            harmonica_ecc,
            harmonica_omega,
            engine_name,
            n_planets,
        )
        params_preopt = _ensure_sanity_ld_params(params_preopt, engine_name, ld_profile, wl_bundle["u"])
        flux_preopt = _get_sanity_model(detrending_type, params_preopt, wl_bundle["time"])

        keys = jax.random.split(key_seed, num=3)
        if fit_sites is None:
            stage1 = optimx.optimize(
                whitelight_model_for_run,
                sites=(
                    ["log_a_rs_0", "t0_0", "_b_0"]
                    if (
                        engine_name == "harmonica"
                        and harmonica_wl_parameterization == "a_rs"
                        and n_planets == 1
                    )
                    else (["logD_0", "t0_0", "_b_0"] if n_planets == 1 else None)
                ),
                start=init_params_wl,
            )
            soln = stage1(
                keys[0],
                wl_bundle["time"],
                wl_bundle["flux_err"],
                y=wl_bundle["flux"],
                prior_params=hyper_params_fit,
            )

            stage2_sites = ["rors_0"]
            if active_wl_ld_mode == "free":
                if engine_name == "harmonica":
                    stage2_sites.extend(["c1", "c2"])
                elif ld_profile == "quadratic":
                    stage2_sites.append("u")
                elif ld_profile == "power2":
                    stage2_sites.extend(["c1", "c2"])
            if n_planets != 1:
                stage2_sites = None
            stage2 = optimx.optimize(whitelight_model_for_run, sites=stage2_sites, start=soln)
            soln = stage2(
                keys[1],
                wl_bundle["time"],
                wl_bundle["flux_err"],
                y=wl_bundle["flux"],
                prior_params=hyper_params_fit,
            )

            stage3 = optimx.optimize(whitelight_model_for_run, start=soln)
            soln = stage3(
                keys[2],
                wl_bundle["time"],
                wl_bundle["flux_err"],
                y=wl_bundle["flux"],
                prior_params=hyper_params_fit,
            )
            n_fit_params = _count_fitted_params(
                detrending_type, engine_name, n_planets, ld_profile, active_wl_ld_mode, engine_odd_harmonics
            )
        else:
            stage = optimx.optimize(whitelight_model_for_run, sites=fit_sites, start=init_params_wl)
            soln = stage(
                keys[0],
                wl_bundle["time"],
                wl_bundle["flux_err"],
                y=wl_bundle["flux"],
                prior_params=hyper_params_fit,
            )
            n_fit_params = _count_site_params(fit_sites)

        params_opt = _soln_to_physical_params(soln, params_preopt, hyper_params_fit, n_planets, engine_name)
        if "rors" not in params_opt and "depths" in params_opt:
            params_opt["rors"] = jnp.sqrt(params_opt["depths"])
        params_opt = _ensure_sanity_ld_params(params_opt, engine_name, ld_profile, wl_bundle["u"])

        flux_init = flux_preopt
        flux_opt = _get_sanity_model(detrending_type, params_opt, wl_bundle["time"])

        metrics = _build_metrics(
            wl_bundle["time"],
            wl_bundle["flux"],
            wl_bundle["flux_err"],
            flux_init,
            flux_opt,
            params_opt,
            detrending_type,
            n_fit_params,
        )

        geometry_ok = None
        if engine_name == "harmonica":
            geometry_ok = np.asarray(
                harmonica_geometry_is_valid(
                    params_opt["b"],
                    params_opt["a_rs"],
                    ecc=harmonica_ecc,
                    omega=harmonica_omega,
                )
            ).tolist()

        return {
            "engine": engine_name,
            "odd_harmonics": engine_odd_harmonics,
            "params_preopt": params_preopt,
            "params_opt": params_opt,
            "soln": soln,
            "init_params": init_params_wl,
            "hyper_params_fit": hyper_params_fit,
            "flux_init": flux_init,
            "flux_opt": flux_opt,
            "metrics": metrics,
            "geometry_ok": geometry_ok,
            "detrending_type": detrending_type,
            "wl_ld_mode": active_wl_ld_mode,
            "wl_bundle": wl_bundle,
            "fit_sites": fit_sites,
        }

    def _run_band_mcmc(map_result, key_seed):
        engine_name = map_result["engine"]
        active_wl_ld_mode = map_result["wl_ld_mode"]
        engine_max_order = max_harmonic_order if engine_name == "harmonica" else 1
        whitelight_model_for_run = create_whitelight_model(
            detrend_type=detrending_type,
            n_planets=n_planets,
            ld_mode=active_wl_ld_mode,
            max_harmonic_order=engine_max_order,
            param_method=harmonica_wl_parameterization,
        )
        init_params_mcmc = dict(map_result["init_params"])
        init_params_mcmc.update(map_result["soln"])

        nuts_kwargs = {}
        if engine_name == "harmonica":
            nuts_kwargs.update(
                dense_mass=harmonica_wl_dense_mass,
                regularize_mass_matrix=harmonica_wl_regularize_mass_matrix,
                max_tree_depth=harmonica_wl_max_tree_depth,
                target_accept_prob=harmonica_wl_target_accept,
            )
            print(
                "Using harmonica band white-light NUTS settings: "
                f"dense_mass={harmonica_wl_dense_mass}, "
                f"regularize_mass_matrix={harmonica_wl_regularize_mass_matrix}, "
                f"max_tree_depth={harmonica_wl_max_tree_depth}, "
                f"target_accept_prob={harmonica_wl_target_accept}"
            )

        return get_samples(
            whitelight_model_for_run,
            key_seed,
            map_result["wl_bundle"]["time"],
            map_result["wl_bundle"]["flux_err"],
            map_result["wl_bundle"]["flux"],
            init_params_mcmc,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs={
                "num_warmup": band_mcmc_num_warmup,
                "num_samples": band_mcmc_num_samples,
                "progress_bar": True,
                "jit_model_args": True,
            },
            prior_params=map_result["hyper_params_fit"],
        )

    compare_engine = None
    if transit_engine == "harmonica":
        compare_engine = "jaxoplanet"
    elif transit_engine == "jaxoplanet":
        compare_engine = "harmonica"

    engine_sequence = [transit_engine]
    if compare_engine is not None:
        engine_sequence.append(compare_engine)

    engine_keys = jax.random.split(key_master, num=len(engine_sequence))
    engine_results = {
        engine_name: _run_map_fit(
            engine_name,
            full_wl_bundle,
            engine_key,
            prior_t0,
            prior_b,
            prior_rprs,
            prior_depth,
            prior_dur,
            harmonica_a_rs,
        )
        for engine_name, engine_key in zip(engine_sequence, engine_keys)
    }

    primary_result = engine_results[transit_engine]
    params_preopt = primary_result["params_preopt"]
    params_opt = primary_result["params_opt"]
    metrics = primary_result["metrics"]

    summary = {
        "config_path": cfg_path,
        "resolved_paths": {
            "base_path": base_path,
            "input_dir": input_dir,
            "fits_file": fits_file,
            "output_dir": output_dir,
            "ld_data_path": ld_data_path,
            "spectro_cache": spectro_data_file,
        },
        "instrument": instrument,
        "instrument_full_str": instrument_full_str,
        "detrending_type": detrending_type,
        "transit_engine": transit_engine,
        "ld_profile": ld_profile,
        "harmonica_max_order": max_harmonic_order,
        "fix_ld": fix_ld,
        "time_points": int(np.asarray(data.wl_time).size),
        "harmonica_geometry_valid": primary_result["geometry_ok"],
        "metrics": metrics,
        "init_physical_params": params_preopt,
        "map_physical_params": params_opt,
        "engine_comparison": {
            engine_name: {
                "metrics": result["metrics"],
                "harmonica_geometry_valid": result["geometry_ok"],
                "map_physical_params": result["params_opt"],
            }
            for engine_name, result in engine_results.items()
        },
    }

    json_path = output_dir / f"{instrument_full_str}_pre_mcmc_map_summary.json"
    with open(json_path, "w") as fh:
        json.dump(_serialise(summary), fh, indent=2, sort_keys=True)

    csv_path = output_dir / f"{instrument_full_str}_pre_mcmc_map_params.csv"
    pd.DataFrame(_planet_rows(params_opt, metrics, primary_result["odd_harmonics"])).to_csv(csv_path, index=False)

    fig_path = output_dir / f"00_{instrument_full_str}_pre_mcmc_full_model_check.png"
    _plot_diagnostic(
        fig_path,
        data.wl_time,
        data.wl_flux,
        primary_result["flux_init"],
        primary_result["flux_opt"],
        detrending_type,
        params_opt,
        metrics,
        transit_engine,
    )

    comparison_rows = []
    for engine_name, result in engine_results.items():
        row = {
            "engine": engine_name,
            "t0": float(np.asarray(result["params_opt"]["t0"])[0]),
            "duration": float(np.asarray(result["params_opt"]["duration"])[0]),
            "b": float(np.asarray(result["params_opt"]["b"])[0]),
            "rors": float(np.asarray(result["params_opt"]["rors"])[0]),
            "depth": float(np.asarray(result["params_opt"]["rors"])[0] ** 2),
        }
        if "a_rs" in result["params_opt"]:
            row["a_rs"] = float(np.asarray(result["params_opt"]["a_rs"])[0])
        if "cos_i" in result["params_opt"]:
            row["cos_i"] = float(np.asarray(result["params_opt"]["cos_i"])[0])
        for harmonic_name in result["odd_harmonics"]:
            if harmonic_name in result["params_opt"]:
                row[harmonic_name] = float(np.asarray(result["params_opt"][harmonic_name]).ravel()[0])
        row.update(result["metrics"])
        comparison_rows.append(row)

    comparison_csv_path = output_dir / f"{instrument_full_str}_pre_mcmc_engine_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)

    comparison_json_path = output_dir / f"{instrument_full_str}_pre_mcmc_engine_comparison.json"
    with open(comparison_json_path, "w") as fh:
        json.dump(
            _serialise(
                {
                    "config_path": cfg_path,
                    "engines": {
                        engine_name: {
                            "metrics": result["metrics"],
                            "harmonica_geometry_valid": result["geometry_ok"],
                            "map_physical_params": result["params_opt"],
                        }
                        for engine_name, result in engine_results.items()
                    },
                }
            ),
            fh,
            indent=2,
            sort_keys=True,
        )

    comparison_fig_path = output_dir / f"00_{instrument_full_str}_pre_mcmc_engine_compare.png"
    _plot_engine_comparison(
        comparison_fig_path,
        full_wl_bundle["time"],
        full_wl_bundle["flux"],
        engine_results,
    )

    primary_harmonica_limb_products = None
    if "harmonica" in engine_results:
        primary_harmonica_limb_products = _save_harmonica_map_limb_products(
            engine_results["harmonica"],
            output_dir,
            instrument_full_str,
            "pre_mcmc_harmonica_map",
            f"{planet_str} - Full-order white-light optimx MAP",
        )

    paper_json_path = None
    paper_csv_path = None
    paper_band1_fig_path = None
    paper_band2_fig_path = None
    paper_band1_harmonica_limb_products = None
    paper_band2_harmonica_limb_products = None
    paper_protocol_summary = None
    paper_band2_results = None
    band_mcmc_summary_json_path = None
    band_mcmc_summary_csv_path = None
    band_mcmc_limb_csv_path = None
    band_mcmc_limb_fig_path = None
    band_mcmc_strings_fig_path = None
    band_mcmc_band1_string_path = None
    band_mcmc_band2_string_path = None
    band_mcmc_summary = None
    if n_planets == 1:
        paper_bandpasses = cfg.get(
            "paper_bandpasses",
            [
                {"label": "band1_0.90_1.20um", "wl_min": 0.90, "wl_max": 1.20},
                {"label": "band2_1.35_1.50um", "wl_min": 1.35, "wl_max": 1.50},
            ],
        )
        if len(paper_bandpasses) >= 2:
            band1_def, band2_def = paper_bandpasses[0], paper_bandpasses[1]
            band1_bundle = _extract_band_lightcurve(
                data,
                band1_def["wl_min"],
                band1_def["wl_max"],
                band1_def.get("label", "band1"),
                instrument,
                order if instrument == "NIRISS/SOSS" else None,
                sld,
                ld_profile,
                prior_t0,
                prior_dur,
                timebin_settings=timebin_settings,
            )
            band2_bundle = _extract_band_lightcurve(
                data,
                band2_def["wl_min"],
                band2_def["wl_max"],
                band2_def.get("label", "band2"),
                instrument,
                order if instrument == "NIRISS/SOSS" else None,
                sld,
                ld_profile,
                prior_t0,
                prior_dur,
                timebin_settings=timebin_settings,
            )

            paper_band1_results = {}
            paper_band2_results = {}
            paper_keys = jax.random.split(jax.random.fold_in(key_master, 2026), num=2 * len(engine_sequence))
            for i, engine_name in enumerate(engine_sequence):
                band1_result = _run_map_fit(
                    engine_name,
                    band1_bundle,
                    paper_keys[2 * i],
                    prior_t0,
                    prior_b,
                    prior_rprs,
                    prior_depth,
                    prior_dur,
                    harmonica_a_rs,
                    wl_ld_mode_override=paper_wl_ld_mode,
                )
                fixed_geometry_sites = _fixed_geometry_fit_sites(
                    detrending_type,
                    engine_name,
                    paper_wl_ld_mode,
                    ld_profile,
                    odd_harmonics,
                    n_planets,
                )
                band1_params = band1_result["params_opt"]
                band2_result = _run_map_fit(
                    engine_name,
                    band2_bundle,
                    paper_keys[2 * i + 1],
                    jnp.atleast_1d(band1_params["t0"]),
                    jnp.atleast_1d(band1_params["b"]),
                    prior_rprs,
                    prior_depth,
                    jnp.atleast_1d(band1_params["duration"]),
                    _ensure_len(band1_params.get("a_rs", harmonica_a_rs), n_planets),
                    fit_sites=fixed_geometry_sites,
                    fixed_geometry_params=band1_params,
                    wl_ld_mode_override=paper_wl_ld_mode,
                )
                paper_band1_results[engine_name] = band1_result
                paper_band2_results[engine_name] = band2_result

            paper_band1_fig_path = output_dir / f"00_{instrument_full_str}_{band1_bundle['slug']}_engine_compare.png"
            _plot_engine_comparison(
                paper_band1_fig_path,
                band1_bundle["time"],
                band1_bundle["flux"],
                paper_band1_results,
                title=(
                    "Paper-Style Band 1 Geometry Fit: "
                    f"{band1_bundle['wl_min']:.2f}-{band1_bundle['wl_max']:.2f} um"
                ),
            )

            paper_band2_fig_path = output_dir / f"00_{instrument_full_str}_{band2_bundle['slug']}_fixed_geometry_engine_compare.png"
            _plot_engine_comparison(
                paper_band2_fig_path,
                band2_bundle["time"],
                band2_bundle["flux"],
                paper_band2_results,
                title=(
                    "Paper-Style Band 2 Fixed-Geometry Comparison: "
                    f"{band2_bundle['wl_min']:.2f}-{band2_bundle['wl_max']:.2f} um"
                ),
            )

            paper_protocol_summary = {
                "band1": {
                    "label": band1_bundle["label"],
                    "wl_min": band1_bundle["wl_min"],
                    "wl_max": band1_bundle["wl_max"],
                    "n_wave": band1_bundle["n_wave"],
                    "time_points": int(np.asarray(band1_bundle["time"]).size),
                },
                "band2": {
                    "label": band2_bundle["label"],
                    "wl_min": band2_bundle["wl_min"],
                    "wl_max": band2_bundle["wl_max"],
                    "n_wave": band2_bundle["n_wave"],
                    "time_points": int(np.asarray(band2_bundle["time"]).size),
                },
                "band1_geometry_results": {
                    engine_name: {
                        "metrics": result["metrics"],
                        "harmonica_geometry_valid": result["geometry_ok"],
                        "map_physical_params": result["params_opt"],
                    }
                    for engine_name, result in paper_band1_results.items()
                },
                "band2_fixed_geometry_results": {
                    engine_name: {
                        "metrics": result["metrics"],
                        "harmonica_geometry_valid": result["geometry_ok"],
                        "map_physical_params": result["params_opt"],
                        "fit_sites": result["fit_sites"],
                    }
                    for engine_name, result in paper_band2_results.items()
                },
            }

            paper_json_path = output_dir / f"{instrument_full_str}_paper_style_two_band_comparison.json"
            with open(paper_json_path, "w") as fh:
                json.dump(
                    _serialise(
                        {
                            "config_path": cfg_path,
                            "paper_protocol": paper_protocol_summary,
                        }
                    ),
                    fh,
                    indent=2,
                    sort_keys=True,
                )

            paper_rows = []
            for stage_name, stage_results in [
                ("band1_geometry", paper_band1_results),
                ("band2_fixed_geometry", paper_band2_results),
            ]:
                for engine_name, result in stage_results.items():
                    row = {
                        "stage": stage_name,
                        "engine": engine_name,
                        "band_label": result["wl_bundle"]["label"],
                        "wl_min": result["wl_bundle"]["wl_min"],
                        "wl_max": result["wl_bundle"]["wl_max"],
                        "t0": float(np.asarray(result["params_opt"]["t0"])[0]),
                        "duration": float(np.asarray(result["params_opt"]["duration"])[0]),
                        "b": float(np.asarray(result["params_opt"]["b"])[0]),
                        "rors": float(np.asarray(result["params_opt"]["rors"])[0]),
                        "depth": float(np.asarray(result["params_opt"]["rors"])[0] ** 2),
                    }
                    if "a_rs" in result["params_opt"]:
                        row["a_rs"] = float(np.asarray(result["params_opt"]["a_rs"])[0])
                    if "cos_i" in result["params_opt"]:
                        row["cos_i"] = float(np.asarray(result["params_opt"]["cos_i"])[0])
                    for harmonic_name in result["odd_harmonics"]:
                        if harmonic_name in result["params_opt"]:
                            row[harmonic_name] = float(np.asarray(result["params_opt"][harmonic_name]).ravel()[0])
                    row.update(result["metrics"])
                    paper_rows.append(row)

            paper_csv_path = output_dir / f"{instrument_full_str}_paper_style_two_band_comparison.csv"
            pd.DataFrame(paper_rows).to_csv(paper_csv_path, index=False)

            if "harmonica" in paper_band1_results:
                paper_band1_harmonica_limb_products = _save_harmonica_map_limb_products(
                    paper_band1_results["harmonica"],
                    output_dir,
                    instrument_full_str,
                    f"{band1_bundle['slug']}_harmonica_map",
                    f"{planet_str} - {band1_bundle['label']} optimx MAP",
                )
            if "harmonica" in paper_band2_results:
                paper_band2_harmonica_limb_products = _save_harmonica_map_limb_products(
                    paper_band2_results["harmonica"],
                    output_dir,
                    instrument_full_str,
                    f"{band2_bundle['slug']}_harmonica_map",
                    f"{planet_str} - {band2_bundle['label']} optimx MAP",
                )

            if run_band_mcmc:
                print(
                    "Running independent two-band white-light MCMC on "
                    f"{band1_bundle['wl_min']:.2f}-{band1_bundle['wl_max']:.2f} um and "
                    f"{band2_bundle['wl_min']:.2f}-{band2_bundle['wl_max']:.2f} um "
                    f"with {band_mcmc_num_warmup} warmup / {band_mcmc_num_samples} samples."
                )
                mcmc_seed = jax.random.fold_in(key_master, 4242)
                mcmc_keys = jax.random.split(mcmc_seed, num=4)

                band1_mcmc_map = paper_band1_results[transit_engine]
                band2_mcmc_map = _run_map_fit(
                    transit_engine,
                    band2_bundle,
                    mcmc_keys[0],
                    prior_t0,
                    prior_b,
                    prior_rprs,
                    prior_depth,
                    prior_dur,
                    harmonica_a_rs,
                    wl_ld_mode_override=paper_wl_ld_mode,
                )
                band1_mcmc_samples = _run_band_mcmc(band1_mcmc_map, mcmc_keys[1])
                band2_mcmc_samples = _run_band_mcmc(band2_mcmc_map, mcmc_keys[2])

                band_mcmc_summary = {
                    "config_path": cfg_path,
                    "transit_engine": transit_engine,
                    "num_warmup": band_mcmc_num_warmup,
                    "num_samples": band_mcmc_num_samples,
                    "bands": [
                        _band_mcmc_summary_row(
                            band1_bundle["label"],
                            band1_bundle,
                            band1_mcmc_samples,
                            transit_engine,
                            periods[0],
                            harmonica_ecc[0],
                            harmonica_omega[0],
                        ),
                        _band_mcmc_summary_row(
                            band2_bundle["label"],
                            band2_bundle,
                            band2_mcmc_samples,
                            transit_engine,
                            periods[0],
                            harmonica_ecc[0],
                            harmonica_omega[0],
                        ),
                    ],
                }

                band_mcmc_summary_json_path = output_dir / f"{instrument_full_str}_two_band_whitelight_mcmc_summary.json"
                with open(band_mcmc_summary_json_path, "w") as fh:
                    json.dump(_serialise(band_mcmc_summary), fh, indent=2, sort_keys=True)

                band_mcmc_summary_csv_path = output_dir / f"{instrument_full_str}_two_band_whitelight_mcmc_summary.csv"
                pd.DataFrame(band_mcmc_summary["bands"]).to_csv(band_mcmc_summary_csv_path, index=False)

                if transit_engine == "harmonica":
                    band_centers = np.array(
                        [
                            0.5 * (float(band1_bundle["wl_min"]) + float(band1_bundle["wl_max"])),
                            0.5 * (float(band2_bundle["wl_min"]) + float(band2_bundle["wl_max"])),
                        ],
                        dtype=float,
                    )
                    band_half_widths = np.array(
                        [
                            0.5 * (float(band1_bundle["wl_max"]) - float(band1_bundle["wl_min"])),
                            0.5 * (float(band2_bundle["wl_max"]) - float(band2_bundle["wl_min"])),
                        ],
                        dtype=float,
                    )

                    combined_rors_samples = np.column_stack(
                        [np.asarray(band1_mcmc_samples["rors_0"]), np.asarray(band2_mcmc_samples["rors_0"])]
                    )
                    combined_harmonic_samples = {}
                    for harmonic_name in odd_harmonics:
                        if harmonic_name in band1_mcmc_samples and harmonic_name in band2_mcmc_samples:
                            combined_harmonic_samples[harmonic_name] = np.column_stack(
                                [
                                    np.asarray(band1_mcmc_samples[harmonic_name]),
                                    np.asarray(band2_mcmc_samples[harmonic_name]),
                                ]
                            )

                    band_mcmc_limb_csv_path = output_dir / f"{instrument_full_str}_two_band_whitelight_mcmc_limb_spectra.csv"
                    band_mcmc_limb_fig_path = output_dir / f"{instrument_full_str}_two_band_whitelight_mcmc_limb_spectra.png"
                    band_mcmc_strings_fig_path = output_dir / f"{instrument_full_str}_two_band_whitelight_mcmc_transmission_strings.png"
                    save_harmonica_limb_products(
                        wavelengths=band_centers,
                        wavelength_err=band_half_widths,
                        rors_samples=combined_rors_samples,
                        harmonic_samples=combined_harmonic_samples,
                        csv_path=str(band_mcmc_limb_csv_path),
                        limb_spectrum_path=str(band_mcmc_limb_fig_path),
                        transmission_strings_path=str(band_mcmc_strings_fig_path),
                        posterior_strings_path=None,
                        title_prefix=f"{planet_str} - Two-band white-light MCMC",
                        bandpass_min=float(np.min(band_centers - band_half_widths)),
                        bandpass_max=float(np.max(band_centers + band_half_widths)),
                    )

                    band_mcmc_band1_string_path = output_dir / f"{instrument_full_str}_{band1_bundle['slug']}_whitelight_mcmc_transmission_string.png"
                    save_harmonica_limb_products(
                        wavelengths=band_centers[0],
                        wavelength_err=band_half_widths[0],
                        rors_samples=np.asarray(band1_mcmc_samples["rors_0"]),
                        harmonic_samples={
                            harmonic_name: np.asarray(band1_mcmc_samples[harmonic_name])
                            for harmonic_name in odd_harmonics
                            if harmonic_name in band1_mcmc_samples
                        },
                        csv_path=str(output_dir / f"{instrument_full_str}_{band1_bundle['slug']}_whitelight_mcmc_limb_spectra.csv"),
                        limb_spectrum_path=str(output_dir / f"{instrument_full_str}_{band1_bundle['slug']}_whitelight_mcmc_limb_spectra.png"),
                        transmission_strings_path=str(band_mcmc_band1_string_path),
                        title_prefix=f"{planet_str} - {band1_bundle['label']} white-light MCMC",
                        bandpass_min=float(band1_bundle["wl_min"]),
                        bandpass_max=float(band1_bundle["wl_max"]),
                    )

                    band_mcmc_band2_string_path = output_dir / f"{instrument_full_str}_{band2_bundle['slug']}_whitelight_mcmc_transmission_string.png"
                    save_harmonica_limb_products(
                        wavelengths=band_centers[1],
                        wavelength_err=band_half_widths[1],
                        rors_samples=np.asarray(band2_mcmc_samples["rors_0"]),
                        harmonic_samples={
                            harmonic_name: np.asarray(band2_mcmc_samples[harmonic_name])
                            for harmonic_name in odd_harmonics
                            if harmonic_name in band2_mcmc_samples
                        },
                        csv_path=str(output_dir / f"{instrument_full_str}_{band2_bundle['slug']}_whitelight_mcmc_limb_spectra.csv"),
                        limb_spectrum_path=str(output_dir / f"{instrument_full_str}_{band2_bundle['slug']}_whitelight_mcmc_limb_spectra.png"),
                        transmission_strings_path=str(band_mcmc_band2_string_path),
                        title_prefix=f"{planet_str} - {band2_bundle['label']} white-light MCMC",
                        bandpass_min=float(band2_bundle["wl_min"]),
                        bandpass_max=float(band2_bundle["wl_max"]),
                    )

    print("Saved pre-MCMC diagnostic outputs:")
    print(f"  figure: {fig_path}")
    print(f"  json:   {json_path}")
    print(f"  csv:    {csv_path}")
    print(f"  compare figure: {comparison_fig_path}")
    print(f"  compare json:   {comparison_json_path}")
    print(f"  compare csv:    {comparison_csv_path}")
    if primary_harmonica_limb_products is not None:
        print(f"  harmonica MAP limb csv:    {primary_harmonica_limb_products['csv']}")
        print(f"  harmonica MAP limb figure: {primary_harmonica_limb_products['limb_figure']}")
        print(f"  harmonica MAP string fig:  {primary_harmonica_limb_products['string_figure']}")
    if paper_protocol_summary is not None:
        print(f"  paper band1 figure: {paper_band1_fig_path}")
        print(f"  paper band2 figure: {paper_band2_fig_path}")
        print(f"  paper json:         {paper_json_path}")
        print(f"  paper csv:          {paper_csv_path}")
        if paper_band1_harmonica_limb_products is not None:
            print(f"  paper band1 limb csv:      {paper_band1_harmonica_limb_products['csv']}")
            print(f"  paper band1 limb figure:   {paper_band1_harmonica_limb_products['limb_figure']}")
            print(f"  paper band1 string fig:    {paper_band1_harmonica_limb_products['string_figure']}")
        if paper_band2_harmonica_limb_products is not None:
            print(f"  paper band2 limb csv:      {paper_band2_harmonica_limb_products['csv']}")
            print(f"  paper band2 limb figure:   {paper_band2_harmonica_limb_products['limb_figure']}")
            print(f"  paper band2 string fig:    {paper_band2_harmonica_limb_products['string_figure']}")
    if band_mcmc_summary is not None:
        print(f"  band mcmc json:     {band_mcmc_summary_json_path}")
        print(f"  band mcmc csv:      {band_mcmc_summary_csv_path}")
        if band_mcmc_limb_csv_path is not None:
            print(f"  band limb csv:      {band_mcmc_limb_csv_path}")
            print(f"  band limb figure:   {band_mcmc_limb_fig_path}")
            print(f"  band strings fig:   {band_mcmc_strings_fig_path}")
            print(f"  band1 string fig:   {band_mcmc_band1_string_path}")
            print(f"  band2 string fig:   {band_mcmc_band2_string_path}")
    print("Key metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    if transit_engine == "harmonica":
        for harmonic_name in primary_result["odd_harmonics"]:
            if harmonic_name in params_opt:
                harmonic_value = float(np.asarray(params_opt[harmonic_name]).ravel()[0])
                print(f"  {harmonic_name}: {harmonic_value:+.6f}")
    if len(engine_results) > 1:
        print("Engine comparison (optimx MAP):")
        for engine_name, result in engine_results.items():
            print(
                "  "
                + f"{engine_name}: "
                + f"chi2_red={result['metrics']['chi2_red_opt']:.4f}, "
                + f"chi2_red_data={result['metrics']['chi2_red_dataerr_opt']:.4f}, "
                + f"RMS={result['metrics']['rms_opt_ppm']:.3f} ppm, "
                + f"MAD={result['metrics']['mad_opt_ppm']:.3f} ppm, "
                + f"AIC={result['metrics']['aic_opt']:.3f}, "
                + f"BIC={result['metrics']['bic_opt']:.3f}"
            )
            if "a1" in result["params_opt"]:
                print(f"    a1={float(np.asarray(result['params_opt']['a1']).ravel()[0]):+.6f}")
        if "harmonica" in engine_results and "jaxoplanet" in engine_results:
            harm = engine_results["harmonica"]["metrics"]
            sph = engine_results["jaxoplanet"]["metrics"]
            print(
                "  delta(harmonica - jaxoplanet): "
                + f"chi2={harm['chi2_opt'] - sph['chi2_opt']:.3f}, "
                + f"chi2_red={harm['chi2_red_opt'] - sph['chi2_red_opt']:.6f}, "
                + f"chi2_red_data={harm['chi2_red_dataerr_opt'] - sph['chi2_red_dataerr_opt']:.6f}, "
                + f"RMS={harm['rms_opt_ppm'] - sph['rms_opt_ppm']:.3f} ppm, "
                + f"BIC={harm['bic_opt'] - sph['bic_opt']:.3f}"
            )
    if paper_band2_results is not None and "harmonica" in paper_band2_results and "jaxoplanet" in paper_band2_results:
        print("Paper-style two-band comparison:")
        print(
            "  geometry band: "
            + f"{paper_protocol_summary['band1']['wl_min']:.2f}-{paper_protocol_summary['band1']['wl_max']:.2f} um "
            + f"({paper_protocol_summary['band1']['n_wave']} pixels)"
        )
        print(
            "  comparison band: "
            + f"{paper_protocol_summary['band2']['wl_min']:.2f}-{paper_protocol_summary['band2']['wl_max']:.2f} um "
            + f"({paper_protocol_summary['band2']['n_wave']} pixels)"
        )
        for engine_name, result in paper_band2_results.items():
            print(
                "  "
                + f"{engine_name}: "
                + f"chi2_red={result['metrics']['chi2_red_opt']:.4f}, "
                + f"chi2_red_data={result['metrics']['chi2_red_dataerr_opt']:.4f}, "
                + f"RMS={result['metrics']['rms_opt_ppm']:.3f} ppm, "
                + f"MAD={result['metrics']['mad_opt_ppm']:.3f} ppm, "
                + f"AIC={result['metrics']['aic_opt']:.3f}, "
                + f"BIC={result['metrics']['bic_opt']:.3f}"
            )
            if "a1" in result["params_opt"]:
                print(f"    a1={float(np.asarray(result['params_opt']['a1']).ravel()[0]):+.6f}")
        harm = paper_band2_results["harmonica"]["metrics"]
        sph = paper_band2_results["jaxoplanet"]["metrics"]
        print(
            "  delta(harmonica - jaxoplanet): "
            + f"chi2={harm['chi2_opt'] - sph['chi2_opt']:.3f}, "
            + f"chi2_red={harm['chi2_red_opt'] - sph['chi2_red_opt']:.6f}, "
            + f"chi2_red_data={harm['chi2_red_dataerr_opt'] - sph['chi2_red_dataerr_opt']:.6f}, "
            + f"RMS={harm['rms_opt_ppm'] - sph['rms_opt_ppm']:.3f} ppm, "
            + f"BIC={harm['bic_opt'] - sph['bic_opt']:.3f}"
        )
    if band_mcmc_summary is not None:
        print("Two-band white-light MCMC summary:")
        for row in band_mcmc_summary["bands"]:
            print(
                "  "
                + f"{row['label']}: "
                + f"rors={row['rors']:.6f} -{row['rors_err_low']:.6f}/+{row['rors_err_high']:.6f}, "
                + f"depth={row['depth'] * 100:.4f}% -{row['depth_err_low'] * 100:.4f}/+{row['depth_err_high'] * 100:.4f}%"
            )
            if "a1" in row:
                print(
                    "    "
                    + f"a1={row['a1']:+.6f}, "
                    + f"a1_frac={row.get('a1_frac', float('nan')):+.6f}"
                )


if __name__ == "__main__":
    main()

import os
import sys
import glob
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.7
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting_lineartrend import plot_map_fits, plot_map_residuals, plot_transmission_spectrum, plot_wavelength_offset_summary
import new_unpack
import argparse
import yaml
import jaxopt
import arviz as az
from jwstdata import SpectroData, process_spectroscopy_data
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
import tinygp
from jaxoplanet.experimental import calc_poly_coeffs

# ---------------------
# Model Functions
# ---------------------
DTYPE = jnp.float64

def _to_f64(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
        return jnp.asarray(x, DTYPE)
    return x

def _tree_to_f64(tree):
    return jax.tree_util.tree_map(_to_f64, tree)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def jax_bin_lightcurve(time, flux, duration, points_per_transit=20):
    dt = duration / points_per_transit
    t_min = jnp.min(time)
    t_max = jnp.max(time)
    total_range = t_max - t_min
    num_bins = jnp.ceil(total_range / dt).astype(int) + 1
    bin_indices = jnp.clip(((time - t_min) / dt).astype(int), 0, num_bins - 1)
    flux_sums = jnp.zeros(num_bins)
    time_sums = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)
    flux_sums = flux_sums.at[bin_indices].add(flux)
    time_sums = time_sums.at[bin_indices].add(time)
    counts = counts.at[bin_indices].add(1.0)
    binned_flux = jnp.where(counts > 0, flux_sums / counts, jnp.nan)
    binned_time = jnp.where(counts > 0, time_sums / counts, jnp.nan)
    return binned_time, binned_flux

def spot_crossing(t, amp, mu, sigma):
    return amp * jnp.exp(-0.5 * (t - mu) **2 / sigma **2)
def get_I_power2(c, alpha, u):
    return 1 - c*(1-jnp.power(u,alpha))

def _compute_transit_model(params, t):
    """Transit Model for one or more planets, using vmap for performance."""
    periods = jnp.atleast_1d(params["period"])
    durations = jnp.atleast_1d(params["duration"])
    t0s = jnp.atleast_1d(params["t0"])
    bs = jnp.atleast_1d(params["b"])
    rorss = jnp.atleast_1d(params["rors"])

    def get_lc(period, duration, t0, b, rors):
        orbit = TransitOrbit(
            period=period,
            duration=duration,
            time_transit=t0,
            impact_param=b,
            radius_ratio=rors
        )
        return limb_dark_light_curve(orbit, params["u"])(t)

    batched_lcs = jax.vmap(get_lc)(periods, durations, t0s, bs, rorss)
    total_flux = jnp.sum(batched_lcs, axis=0)
    return total_flux 
    
def compute_lc_none(params, t):
    return _compute_transit_model(params, t) + 1.0

def compute_lc_linear(params, t):
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend

def compute_lc_quadratic(params, t):
    lc_transit = _compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend

def compute_lc_cubic(params, t):
    lc_transit = _compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2 + params["v3"] * t_norm**3
    return lc_transit + trend

def compute_lc_quartic(params, t):
    lc_transit = _compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2 + params["v3"] * t_norm**3 + params["v4"] * t_norm**4
    return lc_transit + trend

def compute_lc_linear_discontinuity(params, t):
    lc_transit = _compute_transit_model(params, t)
    jump = jnp.where(t > params["t_jump"], params["jump"], 0.0)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + jump
    return lc_transit + trend

def compute_lc_explinear(params, t):
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-(t-jnp.min(t)) / params['tau'])
    return lc_transit + trend

def compute_lc_spot(params, t):
    lc_transit = _compute_transit_model(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend + spot

# --- GP MEAN FUNCTIONS ---
def compute_lc_gp_mean(params, t):
    """The mean function for the simple GP model is just the transit + constant."""
    return _compute_transit_model(params, t) + params["c"]

def compute_lc_linear_gp_mean(params, t):
    """The mean function for the linear + GP model."""
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend

def compute_lc_quadratic_gp_mean(params, t):
    """The mean function for the quadratic + GP model."""
    lc_transit = _compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend

def compute_lc_explinear_gp_mean(params, t):
    """The mean function for the exp-linear + GP model."""
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-(t-jnp.min(t)) / params['tau'])
    return lc_transit + trend

# --- GP BUILDERS ---
def build_gp(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_gp_mean, params))

def build_gp_linear(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_linear_gp_mean, params))

def build_gp_quadratic(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_quadratic_gp_mean, params))

def build_gp_explinear(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_explinear_gp_mean, params))

@jax.jit
def loss(params, t, y, error):
    gp = build_gp(params, t, error)
    return -gp.log_probability(y)

def create_whitelight_model(detrend_type='linear', n_planets=1):
    print(f"Building whitelight model with: detrend_type='{detrend_type}' for {n_planets} planets")
    def _whitelight_model_static(t, yerr, y=None, prior_params=None):

        params = {}
        durations, t0s, bs, rorss = [], [], [], []

        for i in range(n_planets):
            logD = numpyro.sample(f"logD_{i}", dist.Normal(jnp.log(prior_params['duration'][i]), 3e-2))
            durations.append(numpyro.deterministic(f"duration_{i}", jnp.exp(logD)))
            t0s.append(numpyro.sample(f"t0_{i}", dist.Normal(prior_params['t0'][i], 3e-2)))
            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            bs.append(numpyro.deterministic(f'b_{i}', jnp.abs(_b)))
            depths = numpyro.sample(f'depths_{i}', dist.Uniform(1e-5, 0.5))
            rorss.append(numpyro.deterministic(f"rors_{i}", jnp.sqrt(depths)))
        POLY_DEGREE = 12
        MUS = jnp.linspace(0.0, 1.00, 100, endpoint=True)
        #c, alpha = prior_params['u']
        u_theory = prior_params['u']
        c1_mu, c2_mu = u_theory[0], u_theory[1]
        #c1 = numpyro.sample('c1', dist.TruncatedNormal(c1_mu, 0.1, low=0.0, high=1.0))
        #c2 = numpyro.sample('c2', dist.TruncatedNormal(c2_mu, 0.05, low=0.001, high=1.0))
        #power2_profile = get_I_power2(c1, c2, MUS)
        power2_profile = get_I_power2(c1_mu, c2_mu, MUS)
        u_poly = calc_poly_coeffs(MUS, power2_profile,
                              poly_degree = POLY_DEGREE)
        u = numpyro.deterministic('u', u_poly)

        #u = numpyro.sample("u", distx.QuadLDParams())
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter)**2 + yerr**2))

        params = {
            "period": prior_params['period'], "duration": jnp.array(durations), "t0": jnp.array(t0s),
            "b": jnp.array(bs), "rors": jnp.array(rorss), "u": u,
        }

        detrend_components = detrend_type.split('+')

        if any(comp in detrend_components for comp in ['linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', 'gp']):
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))
        if any(comp in detrend_components for comp in ['linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot']):
            params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1))
        if any(comp in detrend_components for comp in ['quadratic', 'cubic', 'quartic']):
            params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1))
        if any(comp in detrend_components for comp in ['cubic', 'quartic']):
            params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1))
        if 'quartic' in detrend_components:
            params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1))
        if 'linear_discontinuity' in detrend_components:
            params['t_jump'] = numpyro.sample('t_jump', dist.Normal(59791.12, 1e-2))
            params['jump'] = numpyro.sample('jump', dist.Normal(0.0, 0.1))
        if 'explinear' in detrend_components:
            params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-5), jnp.log(1.0)))
            params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
            #params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1))
            #params['tau'] = numpyro.sample('tau', dist.HalfNormal(0.1))
        if 'spot' in detrend_components:
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Normal(0.0, 0.01))
            params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(prior_params['spot_guess'], 0.01))
            params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Normal(0.0, 0.01))

        if 'gp' in detrend_components:
            params['GP_log_sigma'] = numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-5), jnp.log(1e3)))
            params['GP_log_rho'] = numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(1e-3), jnp.log(1e2)))

        if detrend_type == 'linear':
            lc_model = compute_lc_linear(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'quadratic':
            lc_model = compute_lc_quadratic(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'cubic':
            lc_model = compute_lc_cubic(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'quartic':
            lc_model = compute_lc_quartic(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'linear_discontinuity':
            lc_model = compute_lc_linear_discontinuity(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'explinear':
            lc_model = compute_lc_explinear(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'spot':
            lc_model = compute_lc_spot(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'none':
            lc_model = compute_lc_none(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'gp':
            gp = build_gp(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        elif detrend_type == 'linear+gp':
            gp = build_gp_linear(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        elif detrend_type == 'quadratic+gp':
            gp = build_gp_quadratic(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        elif detrend_type == 'explinear+gp':
            gp = build_gp_explinear(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model_static

def get_samples(model, key, t, yerr, indiv_y, init_params, **model_kwargs):
    t = _to_f64(t)
    yerr = _to_f64(yerr)
    indiv_y = _to_f64(indiv_y)
    init_params = _tree_to_f64(init_params)
    model_kwargs = _tree_to_f64(model_kwargs)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            model,
            regularize_mass_matrix=False,
            init_strategy=numpyro.infer.init_to_value(values=init_params),
            target_accept_prob=0.9
        ),
        num_warmup=1000,
        num_samples=1000,
        progress_bar=True,
        jit_model_args=True
    )
    mcmc.run(key, t, yerr, y=indiv_y, **model_kwargs)
    return mcmc.get_samples()

def compute_aic(n, residuals, k):
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic

def compute_bic(n, residuals, k):
    """Compute Bayesian Information Criterion."""
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    bic = k * np.log(n) + n * np.log(rss/n)
    return bic

def count_parameters(detrending_type, n_planets=1):
    """Count the number of free parameters for a given detrending type."""
    # Base transit parameters per planet: duration, t0, b, depth
    # Plus: u1, u2 (limb darkening), log_jitter
    n_params = 4 * n_planets + 3

    # Detrending parameters
    if detrending_type == 'none':
        pass  # No additional params
    elif detrending_type == 'linear':
        n_params += 2  # c, v
    elif detrending_type == 'quadratic':
        n_params += 3  # c, v, v2
    elif detrending_type == 'cubic':
        n_params += 4  # c, v, v2, v3
    elif detrending_type == 'quartic':
        n_params += 5  # c, v, v2, v3, v4
    elif detrending_type == 'explinear':
        n_params += 4  # c, v, A, tau
    elif detrending_type == 'linear_discontinuity':
        n_params += 4  # c, v, t_jump, jump
    elif detrending_type == 'spot':
        n_params += 5  # c, v, spot_amp, spot_mu, spot_sigma
    elif detrending_type == 'gp':
        n_params += 3  # c, GP_log_sigma, GP_log_rho
    elif detrending_type == 'linear+gp':
        n_params += 4  # c, v, GP_log_sigma, GP_log_rho
    elif detrending_type == 'quadratic+gp':
        n_params += 5  # c, v, v2, GP_log_sigma, GP_log_rho
    elif detrending_type == 'explinear+gp':
        n_params += 6  # c, v, A, tau, GP_log_sigma, GP_log_rho
    else:
        raise ValueError(f"Unknown detrending_type: {detrending_type}")

    return n_params

def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None):
    if instrument == 'NIRSPEC/G395H':
        mode = "JWST_NIRSpec_G395H"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/G235H':
        mode = "JWST_NIRSpec_G235H"
        wl_min, wl_max = 17000.0, 30600.0
    elif instrument == 'NIRSPEC/G140H':
        mode = "JWST_NIRSpec_G140H"
        wl_min, wl_max = 10000.0, 18000.0
    elif instrument == 'NIRSPEC/G395M':
        mode = "JWST_NIRSpec_G395M"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/PRISM':
        mode = "JWST_NIRSpec_Prism"
        wl_min, wl_max = 5000.0, 55000.0
    elif instrument == 'NIRISS/SOSS':
        mode = f"JWST_NIRISS_SOSSo{order}"
        wl_min, wl_max = 8300.0, 28100.0
    elif instrument == 'MIRI/LRS':
        mode = f'JWST_MIRI_LRS'
        wl_min, wl_max = 50000.0, 120000.0

    wavelengths = np.array(wavelengths)
    wavelength_err = np.array(wavelength_err) if hasattr(wavelength_err, '__len__') else wavelength_err

    U_mu = []

    if hasattr(wavelength_err, '__len__'):
        for i in range(len(wavelengths)):
            wl_angstrom = wavelengths[i] * 1e4
            err_angstrom = wavelength_err[i] * 1e4
            intended_min = wl_angstrom - err_angstrom
            intended_max = wl_angstrom + err_angstrom

            if intended_max > wl_max:
                if err_angstrom > 0:
                    range_min = max(wl_max - err_angstrom * 2, wl_min)
                    range_max = wl_max
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            elif intended_min < wl_min:
                if err_angstrom > 0:
                    range_min = wl_min
                    range_max = min(wl_min + err_angstrom * 2, wl_max)
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            else:
                range_min = intended_min
                range_max = intended_max

            U_mu.append(sld.compute_power2_ld_coeffs(
                wavelength_range=[range_min, range_max],
                mode=mode,
                return_sigmas=False
            ))
        U_mu = jnp.array(U_mu)
    else:
        wl_range_clipped = [max(min(wavelengths)*1e4, wl_min),
                           min(max(wavelengths)*1e4, wl_max)]
        U_mu = sld.compute_power2_ld_coeffs(
            wavelength_range=wl_range_clipped,
            mode=mode,
            return_sigmas=False
        )
        U_mu = jnp.array(U_mu)
    return U_mu

def fit_polynomial(x, y, poly_orders):
    best_order = None
    best_aic = np.inf
    best_coeffs = None
    for deg in poly_orders:
        y_med = np.median(y, axis=0)
        coeffs = np.polyfit(x, y_med, deg)
        pred = np.polyval(coeffs, x)
        aic = compute_aic(len(x), y_med - pred, k=deg+1)
        if aic < best_aic:
            best_aic = aic
            best_order = deg
            best_coeffs = coeffs
    return best_coeffs, best_order, best_aic


def get_robust_sigma(x):
    """Helper to calculate sigma using MAD (robust to outliers)."""
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad

def calculate_beta_metrics(residuals, dt, cut_factor=5.0):
    residuals = np.array(residuals)
    ndata = len(residuals)
    
    # 1. Base White Noise Level (Unbinned)
    # We use this SINGLE value to anchor the theoretical curve
    sigma1 = get_robust_sigma(residuals)
    
    # Time stuff
    cadence_min = dt / 60.0
    
    # Determine Bin Sizes
    max_bin_n = ndata // int(cut_factor) 
    
    # Create unique bin sizes (in points)
    bin_sizes_points = np.unique(np.logspace(0, np.log10(max_bin_n), 300).astype(int))
    
    measured_rms = []
    expected_rms = []
    bin_sizes_min = []
    betas = []
    
    for N in bin_sizes_points:
        # Calculate Binned RMS
        cutoff = ndata - (ndata % N)

        # Binning
        binned_res = residuals[:cutoff].reshape(-1, N).mean(axis=1)
        
        sigma_N_measured = get_robust_sigma(binned_res)
        
        sigma_N_theory = sigma1 / np.sqrt(N)
        
        measured_rms.append(sigma_N_measured)
        expected_rms.append(sigma_N_theory)
        bin_sizes_min.append(N * cadence_min)
        
        betas.append(sigma_N_measured / sigma_N_theory)

    beta_final = np.median(betas)
    
    return beta_final, np.array(bin_sizes_min), np.array(measured_rms), np.array(expected_rms)

def run_beta_monte_carlo(residuals, dt, n_sims=500):
    ndata = len(residuals)
    
    # Estimate the noise from real data to create synthetic data
    sigma1 = get_robust_sigma(residuals)
    
    mc_betas = []
    
    # Get reference bin sizes from the real calculation so arrays match
    _, ref_bin_sizes, _, _ = calculate_beta_metrics(residuals, dt)
    all_sim_rms = np.zeros((n_sims, len(ref_bin_sizes)))
    
    for i in range(n_sims):
        # Generate synthetic white noise using the SAME sigma1
        synth_res = np.random.normal(0, sigma1, ndata)
        
        # Calculate metrics for this synthetic realization
        b_sim, _, rms_sim, _ = calculate_beta_metrics(synth_res, dt)
        
        mc_betas.append(b_sim)
        
        if len(rms_sim) == len(ref_bin_sizes):
            all_sim_rms[i, :] = rms_sim
            
    rms_low_1sig = np.percentile(all_sim_rms, 16, axis=0)
    rms_high_1sig = np.percentile(all_sim_rms, 84, axis=0)
    rms_low_2sig = np.percentile(all_sim_rms, 5, axis=0)
    rms_high_2sig = np.percentile(all_sim_rms, 95, axis=0)

    return np.array(mc_betas), rms_low_1sig, rms_high_1sig, rms_low_2sig, rms_high_2sig
# ---------------------
# Main Analysis
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instrument = cfg['instrument']
    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        nrs = cfg['nrs']
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
    
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    flags = cfg.get('flags', {})
    resolution = cfg.get('resolution', None)
    pixels = cfg.get('pixels', None)
    
    if resolution is None:
        if pixels is None: raise ValueError('Must Specify Resolutions or Pixels')
        bins = pixels
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)
    elif pixels is None:
        bins = resolution
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)

    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']
    mask_integrations_start = outlier_clip.get('mask_integrations_start', None)
    mask_integrations_end = outlier_clip.get('mask_integrations_end', None)

    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    detrending_type = flags.get('detrending_type', 'linear')
    interpolate_trend = flags.get('interpolate_trend', False)
    interpolate_ld = flags.get('interpolate_ld', False)
    fix_ld = flags.get('fix_ld', False)
    need_lowres = flags.get('need_lowres', True)
    mask_start = flags.get('mask_start', False)
    mask_end = flags.get('mask_end', False)
    spot_amp = flags.get('spot_amp', 0.0)
    spot_mu = flags.get('spot_center', 0.0)
    spot_sigma = flags.get('spot_width', 0.0)
    save_trace = flags.get('save_whitelight_trace', False)

    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    periods = jnp.atleast_1d(planet_cfg['period'])
    n_planets = len(periods)
    durations = jnp.atleast_1d(planet_cfg['duration'])
    t0s = jnp.atleast_1d(planet_cfg['t0'])
    bs = jnp.atleast_1d(planet_cfg['b'])
    rors = jnp.atleast_1d(planet_cfg['rprs'])
    depths = rors**2

    PERIOD_FIXED = periods
    PRIOR_DUR = durations
    PRIOR_T0 = t0s
    PRIOR_B = bs
    PRIOR_RPRS = rors
    PRIOR_DEPTH = depths

    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')
    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path
    )

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        mini_instrument = f'nrs{nrs}'
    elif instrument == 'NIRISS/SOSS':
        mini_instrument = f'order{order}'
    else:
        mini_instrument = ''

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    if bins == resolution:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}LR_{high_resolution_bins}HR.pkl'
    elif bins == pixels:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}pix_{high_resolution_bins}pix.pkl'

    lr_bin_str = f'R{low_resolution_bins}' if bins == resolution else f'pix{low_resolution_bins}'
    hr_bin_str = f'R{high_resolution_bins}' if bins == resolution else f'pix{high_resolution_bins}'

    if not os.path.exists(spectro_data_file) or mask_start is not False:
        data = process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start, mask_end, mask_integrations_start, mask_integrations_end)
        data.save(spectro_data_file)
    else:
        data = SpectroData.load(spectro_data_file)
    
    print("Data loaded.")
    print(f"Time: {data.time.shape}")

    COMPUTE_KERNELS = {
        'linear': compute_lc_linear,
        'quadratic': compute_lc_quadratic,
        'cubic': compute_lc_cubic,
        'quartic': compute_lc_quartic,
        'linear_discontinuity': compute_lc_linear_discontinuity,
        'explinear': compute_lc_explinear,
        'spot': compute_lc_spot,
        'gp': compute_lc_gp_mean,
        'none': compute_lc_none,
    }
     
    stringcheck = os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument)
    elif instrument == 'NIRISS/SOSS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order)

    # ----------------------------------------------------
    # WHITELIGHT FITTING - MULTIPLE DETRENDING TYPES
    # ----------------------------------------------------
    # Define detrending types to compare (ignoring config detrending_type)
    detrending_types_to_test = ['linear', 'explinear', 'gp', 'linear+gp', 'explinear+gp']

    # Storage for results across all detrending types
    all_results = {}

    for detrending_type in detrending_types_to_test:
        print(f"\n{'='*80}")
        print(f"FITTING WITH DETRENDING TYPE: {detrending_type}")
        print(f"{'='*80}\n")

        # Check if this fit already exists (skip if not GP and mask exists)
        fit_exists = os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params_{detrending_type}.csv')
        if fit_exists and ('gp' not in detrending_type):
            print(f"Fit for {detrending_type} already exists, skipping...")
            # TODO: Load existing results for comparison
            continue

        plt.scatter(data.wl_time, data.wl_flux)
        plt.savefig(f'{output_dir}/stuff_{detrending_type}.png')
        plt.close()
        print(f'Fitting whitelight with {detrending_type} detrending for outliers and bestfit parameters')

        hyper_params_wl = {
            "duration": PRIOR_DUR,
            "t0": PRIOR_T0,
            'period': PERIOD_FIXED,
            'u': U_mu_wl,
            }
        if 'spot' in detrending_type:
            hyper_params_wl['spot_guess'] = spot_mu
            

        
        init_params_wl = {
            'u': U_mu_wl,
            'c': 1.0,
            'v': 0.0,
            'log_jitter': jnp.log(1e-4),
            'b': PRIOR_B,
            'rors': PRIOR_RPRS
        }
        for i in range(n_planets):
            init_params_wl[f'logD_{i}'] = jnp.log(PRIOR_DUR[i])
            init_params_wl[f't0_{i}'] = PRIOR_T0[i]
            init_params_wl[f'_b_{i}'] = PRIOR_B[i]
            init_params_wl[f'depths_{i}'] = PRIOR_DEPTH[i]

        if 'quadratic' in detrending_type:
            init_params_wl['v2'] = 0.0
        if 'cubic' in detrending_type:
            init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0
        if 'quartic' in detrending_type:
            init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0; init_params_wl['v4'] = 0.0
        if 'explinear' in detrending_type:
            init_params_wl['A'] = 0.001; init_params_wl['tau'] = 0.5
        if 'gp' in detrending_type:
            init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
            init_params_wl['GP_log_rho'] = jnp.log(1)
        if 'linear_discontinuity' in detrending_type:
            init_params_wl['t_jump'] = 59791.12
            init_params_wl['jump'] = -0.001
        whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets)

        # Special logic for Spot Sliding Window - only if purely spot model
        if detrending_type == 'spot':
            whitelight_model_for_run = create_whitelight_model(detrend_type='linear', n_planets=n_planets) # Temp
            # ... [Slider code omitted for brevity as it was specific to pure spot] ...

        if 'gp' in detrending_type:
            print("--- Running Pre-Fit with Linear Detrending to stabilize GP ---")
            whitelight_model_prefit = create_whitelight_model(detrend_type='linear', n_planets=n_planets)
            init_params_prefit = init_params_wl.copy()
            init_params_prefit.pop('GP_log_sigma', None)
            init_params_prefit.pop('GP_log_rho', None)
            soln_prefit = optimx.optimize(whitelight_model_prefit, start=init_params_prefit)(
                key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
            )
            print("--- Initializing GP with Pre-Fit Parameters ---")
            for k in soln_prefit.keys():
                if k in init_params_wl: init_params_wl[k] = soln_prefit[k]

            print("Please make sure config is CPU for GP whitelight fit!")
            init_params_wl['GP_log_sigma'] = jnp.log(5.0 * jnp.nanmedian(data.wl_flux_err))
            init_params_wl['GP_log_rho'] = jnp.log(0.1)
            whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets)
            soln = optimx.optimize(whitelight_model_for_run, start=init_params_wl)(
                key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
            )
        else:
            soln =  optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(whitelight_model_for_run, regularize_mass_matrix=False, init_strategy=numpyro.infer.init_to_value(values=soln), target_accept_prob=0.9),
            num_warmup=1000, num_samples=1000, progress_bar=True, jit_model_args=True
        )
        mcmc.run(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
        inf_data = az.from_numpyro(mcmc)
        if save_trace: az.to_netcdf(inf_data, f'whitelight_trace_{n_planets}planets.nc')
        wl_samples = mcmc.get_samples()
        print(az.summary(inf_data, var_names=None, round_to=7))

            # ------------------------------------------------
            # PARAMETER EXTRACTION (FIXED FOR GP LOGIC)
            # ------------------------------------------------
        bestfit_params_wl = {
            'period': PERIOD_FIXED,
            'u': jnp.nanmedian(wl_samples['u'], axis=0),
        }

        durations_fit, t0s_fit, bs_fit, rors_fit = [], [], [], []
        durations_err, t0s_err, bs_err, rors_err, depths_err = [], [], [], [], []

        for i in range(n_planets):
            durations_fit.append(jnp.nanmedian(wl_samples[f'duration_{i}']))
            t0s_fit.append(jnp.nanmedian(wl_samples[f't0_{i}']))
            bs_fit.append(jnp.nanmedian(wl_samples[f'b_{i}']))
            rors_fit.append(jnp.nanmedian(wl_samples[f'rors_{i}']))
            durations_err.append(jnp.std(wl_samples[f'duration_{i}']))
            t0s_err.append(jnp.std(wl_samples[f't0_{i}']))
            bs_err.append(jnp.std(wl_samples[f'b_{i}']))
            rors_err.append(jnp.std(wl_samples[f'rors_{i}']))
            depths_err.append(jnp.std(wl_samples[f'rors_{i}']**2))

        bestfit_params_wl['duration'] = jnp.array(durations_fit)
        bestfit_params_wl['t0'] = jnp.array(t0s_fit)
        bestfit_params_wl['b'] = jnp.array(bs_fit)
        bestfit_params_wl['rors'] = jnp.array(rors_fit)
        bestfit_params_wl['depths'] = jnp.array(rors_fit)**2

        bestfit_params_wl['duration_err'] = jnp.array(durations_err)
        bestfit_params_wl['t0_err'] = jnp.array(t0s_err)
        bestfit_params_wl['b_err'] = jnp.array(bs_err)
        bestfit_params_wl['rors_err'] = jnp.array(rors_err)
        bestfit_params_wl['depths_err'] = jnp.array(depths_err)
        bestfit_params_wl['error'] = jnp.nanmedian(wl_samples['error'])
            
        if detrending_type != 'none':
            bestfit_params_wl['c'] = jnp.nanmedian(wl_samples['c'])
                # FIX: Check if we are strictly using a pure GP (no trend) or combined
            if detrending_type != 'gp': 
                bestfit_params_wl['v'] = jnp.nanmedian(wl_samples['v'])
                # If we have linear+gp, we might still have V. Check if v exists in samples
            if 'v' in wl_samples and 'v' not in bestfit_params_wl:
                 bestfit_params_wl['v'] = jnp.nanmedian(wl_samples['v'])

        if 'v2' in wl_samples:
            bestfit_params_wl['v2'] = jnp.nanmedian(wl_samples['v2'])
        if 'v3' in wl_samples:
            bestfit_params_wl['v3'] = jnp.nanmedian(wl_samples['v3'])
        if 'v4' in wl_samples:
            bestfit_params_wl['v4'] = jnp.nanmedian(wl_samples['v4'])
        if 'explinear' in detrending_type:
            bestfit_params_wl['A'] = jnp.nanmedian(wl_samples['A'])
            bestfit_params_wl['tau'] = jnp.nanmedian(wl_samples['tau'])
        if 'spot' in detrending_type:
            bestfit_params_wl['spot_amp'] = jnp.nanmedian(wl_samples['spot_amp'])
            bestfit_params_wl['spot_mu'] = jnp.nanmedian(wl_samples['spot_mu'])
            bestfit_params_wl['spot_sigma'] = jnp.nanmedian(wl_samples['spot_sigma'])
        if 'linear_discontinuity' in detrending_type:
            bestfit_params_wl['t_jump'] = jnp.nanmedian(wl_samples['t_jump'])
            bestfit_params_wl['jump'] = jnp.nanmedian(wl_samples['jump'])
    
            # FIX: Correct GP parameter extraction
        if 'gp' in detrending_type:
            bestfit_params_wl['GP_log_sigma'] = jnp.nanmedian(wl_samples['GP_log_sigma'])
            bestfit_params_wl['GP_log_rho'] = jnp.nanmedian(wl_samples['GP_log_rho'])

        spot_trend, jump_trend = None, None
        if 'spot' in detrending_type:
            spot_trend = spot_crossing(data.wl_time, bestfit_params_wl["spot_amp"], bestfit_params_wl["spot_mu"], bestfit_params_wl["spot_sigma"])
        if 'linear_discontinuity' in detrending_type:
            jump_trend = jnp.where(data.wl_time > bestfit_params_wl["t_jump"], bestfit_params_wl["jump"], 0.0)

            # ------------------------------------------------
            # RECONSTRUCT MODEL (FIXED FOR COMBINED GP)
            # ------------------------------------------------
        if 'gp' in detrending_type:
                # Select the correct mean function
            if 'linear' in detrending_type and 'explinear' not in detrending_type:
                gp_mean_func = compute_lc_linear_gp_mean
            elif 'quadratic' in detrending_type:
                gp_mean_func = compute_lc_quadratic_gp_mean
            elif 'explinear' in detrending_type:
                gp_mean_func = compute_lc_explinear_gp_mean
            else:
                gp_mean_func = compute_lc_gp_mean

            wl_kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
            )
            wl_gp = tinygp.GaussianProcess(
                wl_kernel,
                data.wl_time,
                diag=bestfit_params_wl['error']**2,
                mean=partial(gp_mean_func, bestfit_params_wl),
            )
            cond_gp = wl_gp.condition(data.wl_flux, data.wl_time).gp
            mu, var = cond_gp.loc, cond_gp.variance
            wl_transit_model = mu
                
                # Decompose for detrending
            planet_model_only = _compute_transit_model(bestfit_params_wl, data.wl_time)
                # The total trend (parametric + stochastic) is Total - Planet - 1
            trend_flux_total = mu - planet_model_only - 1.0 # Centered around 0
                
                # The "GP Trend" (stochastic part only) is Posterior Mean - Parametric Mean
            parametric_mean_val = gp_mean_func(bestfit_params_wl, data.wl_time)
            gp_stochastic_component = mu - parametric_mean_val

        elif detrending_type == 'linear':
            wl_transit_model = compute_lc_linear(bestfit_params_wl, data.wl_time)
        elif detrending_type == 'quadratic':
            wl_transit_model = compute_lc_quadratic(bestfit_params_wl, data.wl_time)
        elif detrending_type == 'cubic':
            wl_transit_model = compute_lc_cubic(bestfit_params_wl, data.wl_time)
        elif detrending_type == 'quartic':
            wl_transit_model = compute_lc_quartic(bestfit_params_wl, data.wl_time)
        elif detrending_type == 'explinear':
            wl_transit_model = compute_lc_explinear(bestfit_params_wl, data.wl_time)
        elif detrending_type == 'none':
            wl_transit_model = compute_lc_none(bestfit_params_wl, data.wl_time)
        elif detrending_type == 'linear_discontinuity':
            wl_transit_model = compute_lc_linear_discontinuity(bestfit_params_wl, data.wl_time)
        else:
             print('Error with model, not defined!')
             exit()

        wl_residual = data.wl_flux - wl_transit_model
        wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
        wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma
        wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))

        plt.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
        plt.scatter(data.wl_time, data.wl_flux, s=6, c='k', zorder=1, alpha=0.5)

        plt.savefig(f"{output_dir}/11_{instrument_full_str}_{detrending_type}_whitelightmodel.png")
        plt.close()

        plt.scatter(data.wl_time, wl_residual, s=6, c='k')
        plt.title(f'WL Pre-outlier rejection residual ({detrending_type})')
        plt.savefig(f"{output_dir}/12_{instrument_full_str}_{detrending_type}_whitelightresidual.png")
        plt.close()

            # ------------------------------------------------
            # DETRENDED FLUX CALCULATION
            # ------------------------------------------------
        t_masked = data.wl_time[~wl_mad_mask]
        f_masked = data.wl_flux[~wl_mad_mask]
        t_norm_masked = t_masked - jnp.min(t_masked)

            # Calculate the trend to remove based on detrending_type
        if 'gp' in detrending_type:
                # For GP models, use the posterior mean
            planet_model_masked = _compute_transit_model(bestfit_params_wl, t_masked)
            mu_masked = mu[~wl_mad_mask] 
            total_trend_at_points = mu_masked - planet_model_masked
            detrended_flux = f_masked - (total_trend_at_points - 1.0)
            gp_stochastic_at_masked = gp_stochastic_component[~wl_mad_mask]

        elif detrending_type == 'linear':
            trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'quadratic':
            trend = (bestfit_params_wl["c"] + 
                     bestfit_params_wl["v"] * t_norm_masked + 
                     bestfit_params_wl["v2"] * t_norm_masked**2)
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'cubic':
            trend = (bestfit_params_wl["c"] + 
                     bestfit_params_wl["v"] * t_norm_masked + 
                     bestfit_params_wl["v2"] * t_norm_masked**2 + 
                     bestfit_params_wl["v3"] * t_norm_masked**3)
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'quartic':
            trend = (bestfit_params_wl["c"] + 
                     bestfit_params_wl["v"] * t_norm_masked + 
                     bestfit_params_wl["v2"] * t_norm_masked**2 + 
                     bestfit_params_wl["v3"] * t_norm_masked**3 + 
                     bestfit_params_wl["v4"] * t_norm_masked**4)
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'explinear':
            trend = (bestfit_params_wl["c"] + 
                     bestfit_params_wl["v"] * t_norm_masked + 
                     bestfit_params_wl['A'] * jnp.exp(-t_norm_masked / bestfit_params_wl['tau']))
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'linear_discontinuity':
            jump = jnp.where(t_masked > bestfit_params_wl["t_jump"], bestfit_params_wl["jump"], 0.0)
            trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked + jump
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'spot':
            spot = spot_crossing(t_masked, bestfit_params_wl["spot_amp"], 
                                 bestfit_params_wl["spot_mu"], bestfit_params_wl["spot_sigma"])
            trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked + spot
            detrended_flux = f_masked - trend + 1.0

        elif detrending_type == 'none':
            detrended_flux = f_masked 
                
        else:
                # Fallback: assume linear
            trend = bestfit_params_wl["c"] + bestfit_params_wl["v"] * t_norm_masked
            detrended_flux = f_masked - trend + 1.0 

        plt.scatter(t_masked, detrended_flux, c='k', s=6, alpha=0.5)
        plt.title(f'Detrended WLC ({detrending_type}): Sigma {round(wl_sigma_post_clip*1e6)} PPM')
        plt.savefig(f'{output_dir}/14_{instrument_full_str}_{detrending_type}_whitelightdetrended.png')
        plt.close()

        transit_only_model = _compute_transit_model(bestfit_params_wl, t_masked) + 1.0
        residuals_detrended = detrended_flux - transit_only_model

        # ----------------------------------------------------
        # CALCULATE AIC/BIC
        # ----------------------------------------------------
        n_data = len(residuals_detrended)
        n_params = count_parameters(detrending_type, n_planets=n_planets)
        aic = compute_aic(n_data, residuals_detrended, n_params)
        bic = compute_bic(n_data, residuals_detrended, n_params)

        print(f"\nModel Selection Metrics for {detrending_type}:")
        print(f"  N_params: {n_params}")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")

        # ----------------------------------------------------
        # 15_ SUMMARY PLOT (3x2 Grid Layout with Residuals Histogram)
        # ----------------------------------------------------
        fig = plt.figure(figsize=(18, 14))
        b_time, b_flux = jax_bin_lightcurve(jnp.array(data.wl_time),
                                            jnp.array(data.wl_flux),
                                            bestfit_params_wl['duration'])

        # 2. Detrended Data Binning
        b_time_det, b_flux_det = jax_bin_lightcurve(jnp.array(t_masked),
                                                    jnp.array(detrended_flux),
                                                    bestfit_params_wl['duration'])
        b_time_det, b_res_det = jax_bin_lightcurve(jnp.array(t_masked),
                                        jnp.array(residuals_detrended),
                                        bestfit_params_wl['duration'])
        # Style for the binned points
        bin_style = dict(c='darkviolet', edgecolors='darkslateblue', s=40,  zorder=10, label='Binned (8/dur)')

        # Define Main Grid: 3 Rows x 2 Columns
        # Column 1: Light curve panels
        # Column 2: Beta analysis (rows 1-2) + Residuals histogram (row 3)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.5],
                               width_ratios=[1, 1], hspace=0.3, wspace=0.3)

            # --- Column 1, Row 1: Raw Light Curve ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
        ax1.scatter(np.array(b_time), np.array(b_flux), **bin_style)
        ax1.set_title('Raw Light Curve', fontsize=14)
        ax1.set_ylabel('Flux', fontsize=12)
        ax1.tick_params(labelbottom=False)

            # --- Column 1, Row 2: Raw Light Curve + Best-fit Model ---
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
        ax2.scatter(np.array(b_time), np.array(b_flux), **bin_style)
        ax2.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
        ax2.set_title('Raw Light Curve + Best-fit Model', fontsize=14)
        ax2.set_ylabel('Flux', fontsize=12)
        ax2.tick_params(labelbottom=False)

            # --- Column 1, Row 3: Nested Grid (Detrended + Residuals) ---
        gs_nested = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[2, 0], 
            height_ratios=[2, 1],
            hspace=0.0
        )

            # Detrended Light Curve (Top of nested)
        ax3_top = fig.add_subplot(gs_nested[0], sharex=ax1)

        ax3_top.scatter(t_masked, detrended_flux, c='k', s=4, alpha=0.2, label='Detrended Data')
        ax3_top.plot(t_masked, transit_only_model, color="mediumorchid", lw=2, zorder=3, label='Transit Model')
        ax3_top.scatter(np.array(b_time_det), np.array(b_flux_det), **bin_style)
        ax3_top.set_ylabel('Normalized Flux', fontsize=12)
        ax3_top.set_title('Detrended Light Curve', fontsize=14)
        plt.setp(ax3_top.get_xticklabels(), visible=False)

            # Residuals (Bottom of nested)
        ax3_bot = fig.add_subplot(gs_nested[1], sharex=ax3_top)
    

        ax3_bot.scatter(t_masked, residuals_detrended * 1e6, c='k', s=4, alpha=0.2)
        ax3_bot.axhline(0, color='mediumorchid', lw=4, zorder=3, linestyle='--')
        ax3_bot.scatter(np.array(b_time_det), np.array(b_res_det) * 1e6 , **bin_style)
        ax3_bot.set_ylabel('Res. (ppm)', fontsize=10)
        ax3_bot.set_xlabel('Time (BJD)', fontsize=12)

            # ----------------------------------------------------
            # BETA CALCULATION
            # ----------------------------------------------------
        dt = np.median(np.diff(data.wl_time)) * 86400
        residuals_arr = np.array(wl_residual[~wl_mad_mask])

            # Calculate beta metrics
        beta, bin_sizes_min, measured_rms, expected_rms = calculate_beta_metrics(residuals_arr, dt)
        mc_betas, rms_lo_1, rms_hi_1, rms_lo_2, rms_hi_2 = run_beta_monte_carlo(residuals_arr, dt, n_sims=500)

        mu_sim, std_sim = norm.fit(mc_betas)
        z_score = (beta - mu_sim) / std_sim

        print(f"Beta: {beta:.4f}")
        print(f"Measured Beta: {beta:.3f}")
        print(f"MC Mean Beta:  {mu_sim:.3f}")
        print(f"MC Std Dev:    {std_sim:.3f}")
        print(f"Significance:  {z_score:.2f} sigma")

            # --- Column 2, Row 1: RMS vs Bin Size ---
        ax_rms = fig.add_subplot(gs[0, 1])
        ax_rms.loglog(bin_sizes_min, expected_rms * 1e6, 'k--', lw=1.5, label='Theory $1/\sqrt{N}$')
        ax_rms.fill_between(bin_sizes_min, rms_lo_2 * 1e6, rms_hi_2 * 1e6, color='gray', alpha=0.2, label='White Noise ($2\sigma$)')
        ax_rms.fill_between(bin_sizes_min, rms_lo_1 * 1e6, rms_hi_1 * 1e6, color='gray', alpha=0.4, label='White Noise ($1\sigma$)')
        ax_rms.loglog(bin_sizes_min, measured_rms * 1e6, color='teal', lw=2, marker='o', markersize=5, label=f'Data (Beta={beta:.2f})')
        ax_rms.set_xlabel('Bin Size (minutes)', fontsize=12)
        ax_rms.set_ylabel('RMS (ppm)', fontsize=12)
        ax_rms.set_title('Time-Correlated Noise', fontsize=14)
        ax_rms.grid(True, which="both", alpha=0.2)
        ax_rms.legend(fontsize=9)

            # --- Column 2, Row 2: Beta Factor Histogram ---
        ax_beta = fig.add_subplot(gs[1, 1])

            # Plot histogram
        n, bins, patches = ax_beta.hist(mc_betas, bins=30, color='silver', alpha=0.6, density=True, label='Simulated White Noise')

            # Plot Gaussian fit
        xmin, xmax = ax_beta.get_xlim()
        x_plot = np.linspace(xmin, xmax, 100)
        p_plot = norm.pdf(x_plot, mu_sim, std_sim)
        ax_beta.plot(x_plot, p_plot, 'k--', linewidth=2, label='Gaussian Fit')

            # Plot measured beta
        ax_beta.axvline(beta, color='teal', lw=3, label=f'Measured: {beta:.2f}')

            # Add significance text with beta value
        sig_color = 'green' if abs(z_score) < 2.0 else ('orange' if abs(z_score) < 3.0 else 'firebrick')
        ax_beta.text(0.95, 0.85, f"Beta: {beta:.2f}\nSignificance: {z_score:.1f}$\sigma$",
                   transform=ax_beta.transAxes, ha='right', fontsize=14, color=sig_color, fontweight='bold')

        ax_beta.set_xlabel('Beta Factor', fontsize=12)
        ax_beta.set_ylabel('Probability Density', fontsize=12)
        ax_beta.set_title("Beta Significance Test", fontsize=14)
        ax_beta.legend(fontsize=9)

        # --- Column 2, Row 3: Residuals Histogram (vertical) ---
        ax_hist = fig.add_subplot(gs[2, 1])
        residuals_ppm = residuals_detrended * 1e6

        # Plot vertical histogram
        ax_hist.hist(residuals_ppm, bins=50, color='mediumorchid', alpha=0.6, edgecolor='darkslateblue')

        # Add Gaussian fit overlay
        mu_res = np.mean(residuals_ppm)
        sigma_res = np.std(residuals_ppm)
        x_range = np.linspace(np.min(residuals_ppm), np.max(residuals_ppm), 100)
        gaussian_fit = norm.pdf(x_range, mu_res, sigma_res) * len(residuals_ppm) * (residuals_ppm.max() - residuals_ppm.min()) / 50
        ax_hist.plot(x_range, gaussian_fit, 'k--', linewidth=2, label='Gaussian')

        ax_hist.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero')
        ax_hist.set_xlabel('Residuals (ppm)', fontsize=12)
        ax_hist.set_ylabel('Count', fontsize=12)
        ax_hist.set_title(f'Residuals Distribution\n$\\sigma$={sigma_res:.1f} ppm',
                         fontsize=12, fontweight='bold')
        #ax_hist.grid(True, alpha=0.3, axis='y')
        #ax_hist.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/15_{instrument_full_str}_{detrending_type}_whitelight_summary.png', dpi=150)
        plt.close(fig)

        # ----------------------------------------------------
        # STORE RESULTS FOR THIS DETRENDING TYPE
        # ----------------------------------------------------
        all_results[detrending_type] = {
            'bestfit_params': bestfit_params_wl,
            'aic': aic,
            'bic': bic,
            'beta': beta,
            'beta_significance': z_score,
            'n_params': n_params,
            'sigma_post_clip_ppm': wl_sigma_post_clip * 1e6,
            'residuals_detrended': residuals_detrended,
            'wl_mad_mask': wl_mad_mask
        }

        # Save outputs with detrending type in filename
        np.save(f'{output_dir}/{instrument_full_str}_{detrending_type}_whitelight_outlier_mask.npy', arr=wl_mad_mask)

        if 'gp' in detrending_type:
            # Save the stochastic component for spectroscopy
            df_gp = pd.DataFrame({
                'wl_flux': data.wl_flux,
                'gp_flux': mu, # posterior mean
                'gp_err': jnp.sqrt(var),
                'gp_trend': gp_stochastic_component # This is the wiggle (Posterior - Prior Mean)
            })
            # Note: saving full length, not masked, to correspond to data indices
            df_gp.to_csv(f'{output_dir}/{instrument_full_str}_{detrending_type}_whitelight_GP_database.csv')

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
            }
            if detrending_type != 'none':
                row['c'] = bestfit_params_wl['c']
                if 'v' in bestfit_params_wl: row['v'] = bestfit_params_wl['v']

            row['u1'] = bestfit_params_wl['u'][0]
            row['u2'] = bestfit_params_wl['u'][1]

            if 'explinear' in detrending_type:
                row['A'] = bestfit_params_wl['A']
                row['tau'] = bestfit_params_wl['tau']
            elif 'quadratic' in detrending_type:
                row['v2'] = bestfit_params_wl['v2']
            if 'gp' in detrending_type:
                row['GP_log_sigma'] = bestfit_params_wl['GP_log_sigma']
                row['GP_log_rho'] = bestfit_params_wl['GP_log_rho']

            rows.append(row)

        df_params = pd.DataFrame(rows)
        df_params.to_csv(f'{output_dir}/{instrument_full_str}_{detrending_type}_whitelight_bestfit_params.csv', index=False)

        print(f"\n{detrending_type} fit complete!\n")

    # END OF DETRENDING TYPES LOOP

    # ----------------------------------------------------
    # COMPARISON PLOT AND TABLE ACROSS ALL DETRENDING TYPES
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("COMPARISON ACROSS ALL DETRENDING TYPES")
    print("="*80 + "\n")

    # Create comparison table
    comparison_data = []
    for detrend_name, results in all_results.items():
        comparison_data.append({
            'Detrending': detrend_name,
            'N_params': results['n_params'],
            'AIC': results['aic'],
            'BIC': results['bic'],
            'Beta': results['beta'],
            'Beta_Sig': results['beta_significance'],
            'Sigma_PPM': results['sigma_post_clip_ppm']
        })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('BIC')  # Sort by BIC (lower is better)
    df_comparison.to_csv(f'{output_dir}/{instrument_full_str}_detrending_comparison.csv', index=False)

    print(df_comparison.to_string(index=False))
    print(f"\nBest model by BIC: {df_comparison.iloc[0]['Detrending']}")
    print(f"Best model by AIC: {df_comparison.sort_values('AIC').iloc[0]['Detrending']}")

    # Create comparison plot (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Detrending Method Comparison', fontsize=16, fontweight='bold')

    detrend_names = list(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(detrend_names)))

    # Plot 1: Beta factor comparison
    ax = axes[0]
    beta_values = [all_results[d]['beta'] for d in detrend_names]
    bars = ax.bar(range(len(detrend_names)), beta_values, color=colors)
    ax.axhline(1.0, color='red', linestyle='--', label='White Noise', linewidth=2)
    ax.set_xticks(range(len(detrend_names)))
    ax.set_xticklabels(detrend_names, rotation=45, ha='right')
    ax.set_ylabel('Beta Factor')
    ax.set_title('Beta Factor')
   # ax.grid(True, alpha=0.3)
   # ax.legend()
    # Highlight best (closest to 1)
    best_idx = np.argmin(np.abs(np.array(beta_values) - 1.0))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    # Plot 2: Beta significance
    ax = axes[1]
    beta_sig_values = [all_results[d]['beta_significance'] for d in detrend_names]
    bars = ax.bar(range(len(detrend_names)), beta_sig_values, color=colors)
    ax.axhline(2.0, color='orange', linestyle='--', label='2$\\sigma$', linewidth=1.5)
    ax.axhline(3.0, color='red', linestyle='--', label='3$\\sigma$', linewidth=1.5)
    ax.set_xticks(range(len(detrend_names)))
    ax.set_xticklabels(detrend_names, rotation=45, ha='right')
    ax.set_ylabel('Significance (σ)')
    ax.set_title('Beta Significance')
    best_idx = np.argmin(np.abs(np.array(beta_sig_values) - 1.0))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    #ax.grid(True, alpha=0.3)
    #ax.legend()

    # Plot 3: RMS comparison
    ax = axes[2]
    sigma_values = [all_results[d]['sigma_post_clip_ppm'] for d in detrend_names]
    bars = ax.bar(range(len(detrend_names)), sigma_values, color=colors)
    ax.set_xticks(range(len(detrend_names)))
    ax.set_xticklabels(detrend_names, rotation=45, ha='right')
    ax.set_ylabel('RMS (ppm)')
    ax.set_title('Residual RMS')
    #ax.grid(True, alpha=0.3)
    # Highlight best
    best_idx = np.argmin(sigma_values)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instrument_full_str}_detrending_comparison.png', dpi=150)
    plt.close(fig)

    print(f"\nComparison plot saved to: {output_dir}/{instrument_full_str}_detrending_comparison.png")
    print(f"Comparison table saved to: {output_dir}/{instrument_full_str}_detrending_comparison.csv")


if __name__ == "__main__":
    main()


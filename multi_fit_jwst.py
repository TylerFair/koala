import os
import sys
import glob
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
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
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
import tinygp
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

def _noise_binning_stats(residuals, n_bins=30, max_bin=None):
    """
    residuals: array-like, shape (n_channels, n_times)
    Returns: bins, median_sigma, p16, p84, expected_white_sigma
    """
    residuals = np.array(residuals)
    if residuals.ndim == 1:
        residuals = residuals[None, :]
    n_channels, n_times = residuals.shape
    if max_bin is None:
        max_bin = max(1, n_times // 4)
    # logarithmically spaced bin sizes between 1 and max_bin
    bins = np.unique(np.round(np.logspace(0, np.log10(max_bin), n_bins)).astype(int))
    sigma_b_channels = np.full((n_channels, len(bins)), np.nan)
    for i in range(n_channels):
        r = residuals[i, :]
        for j, b in enumerate(bins):
            m = (n_times // b) * b
            if m < b:
                sigma_b_channels[i, j] = np.nan
                continue
            r_trunc = r[:m].reshape(-1, b)
            means = r_trunc.mean(axis=1)
            if means.size > 1:
                sigma_b_channels[i, j] = np.std(means, ddof=1)
            else:
                sigma_b_channels[i, j] = 0.0
    sigma_med = np.nanmedian(sigma_b_channels, axis=0)
    sigma_16 = np.nanpercentile(sigma_b_channels, 16, axis=0)
    sigma_84 = np.nanpercentile(sigma_b_channels, 84, axis=0)
    # expected white-noise scaling using median unbinned sigma across channels
    sigma1_channels = np.nanstd(residuals, axis=1, ddof=1)
    sigma1_med = np.nanmedian(sigma1_channels)
    expected_white = sigma1_med / np.sqrt(bins)
    return bins, sigma_med, sigma_16, sigma_84, expected_white

def plot_noise_binning(residuals, outpath, title=None):
    """
    Create a log-log plot of binned noise vs bin size and save to outpath.
    residuals shape: (n_channels, n_times) or (n_times,)
    """
    bins, sigma_med, sigma_16, sigma_84, expected_white = _noise_binning_stats(residuals)
    plt.figure(figsize=(6,4))
    plt.loglog(bins, sigma_med, 'k-o', label='Measured (median across channels)')
    plt.fill_between(bins, sigma_16, sigma_84, color='gray', alpha=0.3, label='16-84%')
    plt.loglog(bins, expected_white, 'r--', label='White-noise expectation (σ₁/√N)')
    plt.xlabel('Bin size (number of points)')
    plt.ylabel('RMS')
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved noise-binning plot to {outpath}")
def spot_crossing(t, amp, mu, sigma):
    return amp * jnp.exp(-0.5 * (t - mu) **2 / sigma **2)

def _compute_transit_model(params, t):
    """Transit Model for one or more planets, using vmap for performance."""

    # Ensure all planet parameters are arrays
    periods = jnp.atleast_1d(params["period"])
    durations = jnp.atleast_1d(params["duration"])
    t0s = jnp.atleast_1d(params["t0"])
    bs = jnp.atleast_1d(params["b"])
    rorss = jnp.atleast_1d(params["rors"])

    # Define a function to compute a single light curve
    def get_lc(period, duration, t0, b, rors):
        orbit = TransitOrbit(
            period=period,
            duration=duration,
            time_transit=t0,
            impact_param=b,
            radius_ratio=rors
        )
        # 'u' and 't' are the same for all planets
        return limb_dark_light_curve(orbit, params["u"])(t)

    # Vectorize the function over the planet parameters
    batched_lcs = jax.vmap(get_lc)(periods, durations, t0s, bs, rorss)

    # Sum the light curves and adjust for the baseline
    total_flux = jnp.sum(batched_lcs, axis=0)

    return total_flux 
    
def compute_lc_none(params, t):
    """Computes transit with no detrending."""
    return _compute_transit_model(params, t) + 1.0

def compute_lc_linear(params, t):
    """Computes transit + linear trend."""
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend

def compute_lc_explinear(params, t):
    """Computes transit + exponential-linear trend."""
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-(t-jnp.min(t)) / params['tau'])
    return lc_transit + trend

def compute_lc_spot(params, t):
    """Computes transit + spot crossing."""
    lc_transit = _compute_transit_model(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend + spot

def compute_lc_gp_mean(params, t):
    """The mean function for the GP model is just the transit."""
    return _compute_transit_model(params, t) + params["c"]

def compute_lc_gp_spectroscopic(params, t, gp_trend):
    """Computes transit + scaled GP trend for spectroscopic light curves."""
    lc_transit = _compute_transit_model(params, t)
    # The GP trend is scaled by a new amplitude parameter A_gp
    # and added to the transit model with a baseline offset 'c'.
    return lc_transit + params["c"] + params["A_gp"] * gp_trend
def build_gp(params, t):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=jnp.exp(params["logs2"]),
              mean=partial(compute_lc_gp_mean, params))
@jax.jit
def loss(params, t, y):
    gp = build_gp(params, t)
    return -gp.log_probability(y)
def create_whitelight_model(detrend_type='linear', n_planets=1):
    """
    Building a static whitelight model so jax doesn't retrace.
    """
    print(f"Building whitelight model with: detrend_type='{detrend_type}' for {n_planets} planets")
    def _whitelight_model_static(t, yerr, y=None, prior_params=None):

        params = {}

        # Planet parameters - now handling lists
        durations, t0s, bs, rorss = [], [], [], []

        for i in range(n_planets):
            logD = numpyro.sample(f"logD_{i}", dist.Normal(jnp.log(prior_params['duration'][i]), 1e-2))
            durations.append(numpyro.deterministic(f"duration_{i}", jnp.exp(logD)))

            t0s.append(numpyro.sample(f"t0_{i}", dist.Normal(prior_params['t0'][i], 1e-2)))

            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            bs.append(numpyro.deterministic(f'b_{i}', jnp.abs(_b)))

            depths = numpyro.sample(f'depths_{i}', dist.Uniform(1e-5, 0.5))
            rorss.append(numpyro.deterministic(f"rors_{i}", jnp.sqrt(depths)))

        # Shared parameters
        u = numpyro.sample("u", distx.QuadLDParams())
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter)**2 + yerr**2))

        params = {
            "period": prior_params['period'], "duration": jnp.array(durations), "t0": jnp.array(t0s),
            "b": jnp.array(bs), "rors": jnp.array(rorss), "u": u,
        }

        # The returned model will only contain ONE of these blocks.
        if detrend_type == 'linear':
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1))
            lc_model = compute_lc_linear(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)

        elif detrend_type == 'explinear':
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1))
            params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1))
            params['tau'] = numpyro.sample('tau', dist.Normal(0.0, 0.1))
            lc_model = compute_lc_explinear(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'spot':
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1))
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Normal(0.0, 0.01)) # 1 % transit
            params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(prior_params['spot_guess'], 0.01)) #  15 min uncertainty
            params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Normal(0.0, 0.01)) # 15 min half-width
            lc_model = compute_lc_spot(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif detrend_type == 'none':
            lc_model = compute_lc_none(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)

        elif detrend_type == 'gp':
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))
            params['v'] = 0.0

            params['logs2'] = numpyro.sample('logs2', dist.Uniform(2*jnp.log(1e-6), 2*jnp.log(1.0)))
            params['GP_log_sigma']  = numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-5), jnp.log(1e3)))
            params['GP_log_rho']  = numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(1e-3), jnp.log(1e2)))

            gp = build_gp(params, t)

            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model_static

def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free', n_planets=1):
    """
    Build a static vectorized model for individual res calls.
    It must be built in this way or else we deal with numerous if statement problems.
    """
    print(f"Building vectorized model with: detrend='{detrend_type}', ld='{ld_mode}', trend='{trend_mode}' for {n_planets} planets")

    if detrend_type == 'linear':
        compute_lc_kernel = compute_lc_linear
    elif detrend_type == 'explinear':
        compute_lc_kernel = compute_lc_explinear
    elif detrend_type == 'spot':
        compute_lc_kernel = compute_lc_spot
    elif detrend_type == 'gp':
        compute_lc_kernel = compute_lc_gp_mean
    elif detrend_type == 'gp_spectroscopic':
        compute_lc_kernel = compute_lc_gp_spectroscopic
    elif detrend_type == 'none':
        compute_lc_kernel = compute_lc_none
    else:
        raise ValueError(f"Unsupported detrend_type for vectorized model: {detrend_type}")

    def _vectorized_model_static(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                               mu_depths=None, PERIOD=None, trend_fixed=None,
                               ld_interpolated=None, ld_fixed=None,
                               mu_spot_amp=None, mu_spot_mu=None, mu_spot_sigma=None,
                               mu_u_ld=None, gp_trend=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]

        # These are now arrays for multi-planet support
        durations = mu_duration
        t0s = mu_t0
        bs = mu_b

        # Sample depths for each planet and each light curve
        depths = numpyro.sample('depths', dist.Uniform(1e-5, 0.5).expand([num_lcs, n_planets]))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))

        yerr_per_lc = jnp.nanmedian(yerr, axis=1)
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]))
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter**2 + yerr_per_lc**2))
        error_broadcast = total_error[:, None] * jnp.ones_like(t)

        if ld_mode == 'free':
            u = numpyro.sample('u', dist.TruncatedNormal(loc=mu_u_ld, scale=0.2, low=-1.0, high=1.0).to_event(1))
        elif ld_mode == 'fixed':
            u = numpyro.deterministic("u", ld_fixed)
        elif ld_mode == 'interpolated':
            u = numpyro.deterministic("u", ld_interpolated)
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        params = {
            "period": PERIOD, "duration": durations, "t0": t0s, "b": bs, "rors": rors, "u": u,
        }

        in_axes = {"period": None, "duration": None, "t0": None, "b": None, "rors": 0, "u": 0}
        if detrend_type != 'none':
            if trend_mode == 'free':
                params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1).expand([num_lcs]))
                params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1).expand([num_lcs]))
                in_axes.update({'c': 0, 'v': 0})
                if detrend_type == 'explinear':
                    params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    params['tau'] = numpyro.sample('tau', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes.update({'A': 0, 'tau': 0})
                if detrend_type == 'spot':
                    params['spot_amp'] = numpyro.sample('spot_amp', dist.Normal(mu_spot_amp, 0.01).expand([num_lcs]))
                    params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(mu_spot_mu, 0.01).expand([num_lcs]))
                    params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Normal(mu_spot_sigma, 0.01).expand([num_lcs]))
                    in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})
            elif trend_mode == 'fixed':
                trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
                params['v'] = numpyro.deterministic('v', trend_temp[:, 1])
                in_axes.update({'c': 0, 'v': 0})
                if detrend_type == 'explinear':
                    params['A'] = numpyro.deterministic('A', trend_temp[:, 2])
                    params['tau'] = numpyro.deterministic('tau', trend_temp[:, 3])
                    in_axes.update({'A': 0, 'tau': 0})
                if detrend_type == 'spot':
                    params['spot_amp'] = numpyro.deterministic('spot_amp', trend_temp[:, 2])
                    params['spot_mu'] = numpyro.deterministic('spot_mu', trend_temp[:, 3])
                    params['spot_sigma'] = numpyro.deterministic('spot_sigma', trend_temp[:, 4])
                    in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})
            else:
                raise ValueError(f"Unknown trend_mode: {trend_mode}")

        # Adjust vmap for multi-planet
        if detrend_type == 'gp_spectroscopic':
            params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(1e-1, 1e1).expand([num_lcs]))
            in_axes['A_gp'] = 0
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, gp_trend)
        else:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)

        numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model_static

def get_samples(model, key, t, yerr, indiv_y, init_params, **model_kwargs):
    """
    Runs MCMC using NUTS to get posterior samples.
    Accepts additional keyword arguments to pass directly to the model.
    """
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

    #  **model_kwargs unpacks the dictionary into kwargs
    # e.g., if model_kwargs is {'ld_fixed': arr},mcmc.run(..., ld_fixed=arr)
    mcmc.run(key, t, yerr, y=indiv_y, **model_kwargs)

    return mcmc.get_samples()

def compute_aic(n, residuals, k):
    """Computes an approximate AIC value."""
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic

def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None):
    """Gets limb darkening coefficients using boundary ranges for out-of-range wavelengths."""
    if instrument == 'NIRSPEC/G395H':
        mode = "JWST_NIRSpec_G395H"
        wl_min, wl_max = 28700.0, 51700.0 #29000, 52000  # Angstroms
    if instrument == 'NIRSPEC/G395M':
        mode = "JWST_NIRSpec_G395M"
        wl_min, wl_max = 28700.0, 51700.0 #29000, 52000  # Angstroms
    elif instrument == 'NIRSPEC/PRISM':
        mode = "JWST_NIRSpec_Prism"
        wl_min, wl_max = 5000.0, 55000.0
    elif instrument == 'NIRISS/SOSS':
        mode = f"JWST_NIRISS_SOSSo{order}"
        wl_min, wl_max = 8300.0, 28100.0  # Angstroms
    elif instrument == 'MIRI/LRS':
        mode = f'JWST_MIRI_LRS'
        wl_min, wl_max = 50000.0, 120000.0

    wavelengths = np.array(wavelengths)
    wavelength_err = np.array(wavelength_err) if hasattr(wavelength_err, '__len__') else wavelength_err

    U_mu = []

    if hasattr(wavelength_err, '__len__'):  # Array of errors
        for i in range(len(wavelengths)):
            wl_angstrom = wavelengths[i] * 1e4
            err_angstrom = wavelength_err[i] * 1e4

            # Calculate the intended range
            intended_min = wl_angstrom - err_angstrom
            intended_max = wl_angstrom + err_angstrom

            # Check if range exceeds boundaries
            if intended_max > wl_max:
                # Use boundary range
                if err_angstrom > 0:
                    range_min = max(wl_max - err_angstrom * 2, wl_min)
                    range_max = wl_max
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            elif intended_min < wl_min:
                #  logic for lower boundary
                if err_angstrom <= 0:
                    print('Issue here, err_angstrom is', err_angstrom)
                if err_angstrom > 0:
                    range_min = wl_min
                    range_max = min(wl_min + err_angstrom * 2, wl_max)
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            else:
                # Normal case - within bounds
                range_min = intended_min
                range_max = intended_max

            U_mu.append(sld.compute_quadratic_ld_coeffs(
                wavelength_range=[range_min, range_max],
                mode=mode,
                return_sigmas=False
            ))

        U_mu = jnp.array(U_mu)
    else:
        # Single error case - clip the overall range
        wl_range_clipped = [max(min(wavelengths)*1e4, wl_min),
                           min(max(wavelengths)*1e4, wl_max)]
        U_mu = sld.compute_quadratic_ld_coeffs(
            wavelength_range=wl_range_clipped,
            mode=mode,
            return_sigmas=False
        )
        U_mu = jnp.array(U_mu)

    return U_mu




def fit_model_map(t, yerr, indiv_y, init_params,
                 mu_duration, mu_t0, mu_depths, PERIOD,
                 key=None, trend_fixed=None, ld_interpolated=None, ld_fixed=None):
    """Fits the model using MAP optimization."""
    if key is None:
        key = jax.random.PRNGKey(111)

    vmodel = partial(vectorized_model,
                     mu_duration=mu_duration, mu_t0=mu_t0, mu_depths=mu_depths,
                     PERIOD=PERIOD, trend_fixed=trend_fixed, ld_interpolated=ld_interpolated, ld_fixed=ld_fixed)

    soln = optimx.optimize(vmodel, start=init_params)(key, t, yerr, y=indiv_y)

    if trend_fixed is not None:
        soln["c"] = np.array(trend_fixed["c"])
        soln["v"] = np.array(trend_fixed["v"])
    if ld_interpolated is not None:
        soln["u"] = np.array(ld_interpolated)
    if ld_fixed is not None:
        soln["u"] = np.array(ld_fixed)

    return soln

def fit_polynomial(x, y, poly_orders):
    """Fits polynomials and selects the best order using AIC."""
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

def plot_poly_fit(x, y, coeffs, order, xlabel, ylabel, title, save_path):
    """Plots data and its polynomial fit, then saves it."""
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)

    y_med = np.median(y, axis=0)
    y_err = np.std(y, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y_med, yerr=y_err, fmt='o', ls='', ecolor='k', mfc='k', mec='k')
    ax.plot(x_fit, y_fit, '-', label=f'Poly fit (deg {order})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved polynomial fit plot to {save_path}")

def save_results(wavelengths, samples, csv_filename):
    """Computes and saves transmission spectrum results to CSV for multi-planet fits."""
    depth_chain = samples['rors']**2

    # (n_samples, n_lcs, n_planets) -> (n_lcs, n_planets)
    depth_median = np.nanmedian(depth_chain, axis=0)
    depth_err = np.std(depth_chain, axis=0)

    if depth_median.ndim == 1:
        # This case might occur if n_planets=1 and the last dim is squeezed.
        depth_median = depth_median[:, np.newaxis]
        depth_err = depth_err[:, np.newaxis]

    n_planets = depth_median.shape[1]

    # Prepare header
    header_cols = ["wavelength"]
    for i in range(n_planets):
        header_cols.append(f"depth{i:02d}")
        header_cols.append(f"depth_err{i:02d}")
    header = ",".join(header_cols)

    # Prepare data columns
    output_cols = [wavelengths]
    for i in range(n_planets):
        output_cols.append(depth_median[:, i])
        output_cols.append(depth_err[:, i])

    output_data = np.column_stack(output_cols)

    np.savetxt(
        csv_filename, output_data, delimiter=",",
        header=header, comments=""
    )
    print(f"Transmission spectroscopy data saved to {csv_filename}")

# ---------------------
# Main Analysis
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
   #validate_config(cfg)


    instrument = cfg['instrument']
    if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM':
        nrs = cfg['nrs']
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
    elif instrument == 'MIRI/LRS':
        pass
    else:
        raise NotImplementedError('Instrument not implemented yet')
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']


    flags = cfg.get('flags', {})
    resolution = cfg.get('resolution', None)
    pixels = cfg.get('pixels', None)
    if resolution is None:
        if pixels is None:
            raise ValueError('Must Specify Resolutions or Pixels')
        bins = pixels
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)
    elif pixels is None:
        if resolution is None:
            raise ValueError('Must specify resolution or pixels')
        bins = resolution
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)

    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']
    mask_integrations_start = outlier_clip.get('mask_integrations_start', None)
    mask_integrations_end = outlier_clip.get('mask_integrations_end', None)

    # Paths
    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    if not os.path.exists(input_dir):
        print(f"Error: No Planet Folder detected. Need directory: {input_dir}")
        sys.exit(1)
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if fits_file is None:
        print("Error: `fits_file` not specified in the config file.")
        sys.exit(1)
    if not os.path.exists(fits_file):
        print(f"Error: FITS file not found at: {fits_file}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # device (please use gpu!!)
    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    # flags
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

    # binning seperation


    # outlier clipping
    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    # planet priors
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
    # stellar parameters
    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')

    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path
    )

    mini_instrument = 'order'+str(order) if instrument == 'NIRISS/SOSS' else 'nrs'+str(nrs) if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM' else ''
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
        print("Shapes:")
        print(f"Time: {data.time.shape}")
        print(f"White light: {data.wl_time.shape}, {data.wl_flux.shape}")
        print(f"Low res: {data.wavelengths_lr.shape}, {data.flux_lr.shape}")
        print(f"High res: {data.wavelengths_hr.shape}, {data.flux_hr.shape}")
    else:
        data = SpectroData.load(spectro_data_file)
    print("Shapes:")
    print(f"Time: {data.time.shape}")
    print(f"White light: {data.wl_time.shape}, {data.wl_flux.shape}")
    print(f"Low res: {data.wavelengths_lr.shape}, {data.flux_lr.shape}")
    print(f"High res: {data.wavelengths_hr.shape}, {data.flux_hr.shape}, {data.flux_err_hr.shape}")

    if any((t0 <= jnp.min(data.time)) or (t0 >= jnp.max(data.time)) for t0 in PRIOR_T0):
        print('One or more T0 is not in data timeframe, please double check your T0 with the 00_precheck plot!!!')

    COMPUTE_KERNELS = {
    'linear': compute_lc_linear,
    'explinear': compute_lc_explinear,
    'spot': compute_lc_spot,
     'gp': compute_lc_gp_mean,
    'none': compute_lc_none,
    'gp_spectroscopic': compute_lc_gp_spectroscopic}


    stringcheck = os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')

    if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM' or instrument == 'MIRI/LRS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument)
    elif instrument == 'NIRISS/SOSS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order)

    if not stringcheck or (detrending_type == 'gp'):
        if not os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv'):
            plt.scatter(data.wl_time, data.wl_flux)
            plt.savefig(f'{output_dir}/00_{instrument_full_str}_whitelight_precheck.png')
            plt.close()
            #keep_going = input('Whitelight precheck with guess T0 has been created, would you like to continue? (Enter to continue/N to exit)')
            #plt.close()
            #if keep_going.lower == 'n':
            #    exit()
            print('Fitting whitelight for outliers and bestfit parameters')
            hyper_params_wl = {
                "duration": PRIOR_DUR,
                "t0": PRIOR_T0,
                'period': PERIOD_FIXED,
            }
            if detrending_type == 'spot':
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

            if detrending_type == 'explinear':
                init_params_wl['A'] = 0.0
                init_params_wl['tau'] = 0.5
            if detrending_type == 'spot':
                init_params_wl['spot_amp'] = spot_amp
                init_params_wl['spot_mu'] = spot_mu
                init_params_wl['spot_sigma'] = spot_sigma
            elif detrending_type == 'gp':
                init_params_wl['logs2'] = jnp.log(2*jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(0.1)

            if detrending_type == 'gp':
                print("Please make sure config is CPU for GP whitelight fit!")
                print("GP's are currently exceptionally slow if using GPU,")
                print("so it is recommended to fit the whitelight curve")
                print("and then rerun with config swapped to GPU.")
                print("This script will prompt you once the whitelight")
                print("is finished to exit the script (your GP will be saved). Happy fitting!")


            whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets)
            if detrending_type == 'spot':
                whitelight_model_for_run = create_whitelight_model(detrend_type='linear', n_planets=n_planets)



            if detrending_type == 'gp':
                solver = jaxopt.ScipyMinimize(fun=loss)
                init_params = jax.tree_util.tree_map(jnp.asarray, init_params_wl|hyper_params_wl)
                soln = solver.run(init_params, data.wl_time, data.wl_flux)
                init_params_wl['GP_log_sigma'] = soln.params['GP_log_sigma']
                init_params_wl['GP_log_rho'] = soln.params['GP_log_rho']
                init_params_wl['logs2'] = soln.params['logs2']


            soln =  optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            if detrending_type == 'spot':
                amp0     = float(init_params_wl['spot_amp'])
                center0  = float(init_params_wl['spot_mu'])
                width0   = float(init_params_wl['spot_sigma'])
                tmin, tmax = float(np.min(data.time)), float(np.max(data.time))
                AMP_MIN, AMP_MAX         = (0.0, 0.01)
                CENTER_MIN, CENTER_MAX   = (tmin, tmax)
                WIDTH_MIN, WIDTH_MAX     = (0.0, 0.01)
                fig, ax = plt.subplots(figsize=(8, 4))
                plt.subplots_adjust(left=0.10, right=0.98, bottom=0.35)
                soln['spot_amp'] = init_params_wl['spot_amp']
                soln['spot_mu'] = init_params_wl['spot_mu']
                soln['spot_sigma'] = init_params_wl['spot_sigma']
                soln['period'] = hyper_params_wl['period']
                lc_model = compute_lc_spot(soln, data.time)
                ax.scatter(data.time, data.wl_flux, c='k', s=4, alpha=0.6, label='data')
                [model_line] = ax.plot(data.time, lc_model, c='r', lw=1.5, label='model')
                ax.set_xlabel("time")
                ax.set_ylabel("flux")
                axcolor = 'lightgoldenrodyellow'
                ax_amp    = plt.axes([0.10, 0.25, 0.80, 0.03], facecolor=axcolor)
                ax_center = plt.axes([0.10, 0.20, 0.80, 0.03], facecolor=axcolor)
                ax_width  = plt.axes([0.10, 0.15, 0.80, 0.03], facecolor=axcolor)
                s_amp    = Slider(ax_amp,    'spot_amp',    AMP_MIN,    AMP_MAX,    valinit=amp0)
                s_center = Slider(ax_center, 'spot_mu', CENTER_MIN, CENTER_MAX, valinit=center0)
                s_width  = Slider(ax_width,  'spot_sigma',  WIDTH_MIN,  WIDTH_MAX,  valinit=width0)
                reset_ax = plt.axes([0.10, 0.03, 0.14, 0.06])
                save_ax  = plt.axes([0.28, 0.03, 0.20, 0.06])
                cancel_ax= plt.axes([0.51, 0.03, 0.14, 0.06])
                reset_btn = Button(reset_ax, 'Reset')
                save_btn  = Button(save_ax,  'Save / Use These')
                cancel_btn= Button(cancel_ax,'Cancel')
                bestfit = {
                    'amp': amp0,
                    'mu': center0,
                    'sigma': width0,
                    'accepted': False
                }
                def recompute_and_draw(a, c, w):
                    p = dict(soln)
                    p['spot_amp']    = float(a)
                    p['spot_mu'] = float(c)
                    p['spot_sigma']  = float(w)
                    model_line.set_ydata(compute_lc_spot(p, data.time))
                    fig.canvas.draw_idle()
                def on_slider_change(_):
                    recompute_and_draw(s_amp.val, s_center.val, s_width.val)
                def on_reset(_):
                    s_amp.reset(); s_center.reset(); s_width.reset()
                def on_save(_):
                    bestfit['amp']   = float(s_amp.val)
                    bestfit['mu']    = float(s_center.val)
                    bestfit['sigma'] = float(s_width.val)
                    bestfit['accepted'] = True
                    plt.close(fig)
                def on_cancel(_):
                    bestfit['accepted'] = False
                    plt.close(fig)
                s_amp.on_changed(on_slider_change)
                s_center.on_changed(on_slider_change)
                s_width.on_changed(on_slider_change)
                reset_btn.on_clicked(on_reset)
                save_btn.on_clicked(on_save)
                cancel_btn.on_clicked(on_cancel)

                plt.show()
                if bestfit['accepted']:
                    init_params_wl['spot_amp']   = bestfit['amp']
                    init_params_wl['spot_mu']    = bestfit['mu']
                    init_params_wl['spot_sigma'] = bestfit['sigma']

                print(f"Using spot params: amp={init_params_wl['spot_amp']}, mu={init_params_wl['spot_mu']}, sigma={init_params_wl['spot_sigma']}")

                whitelight_model_for_run = create_whitelight_model(detrend_type='spot', n_planets=n_planets)

                soln =  optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(
                    whitelight_model_for_run,
                    regularize_mass_matrix=False,
                    init_strategy=numpyro.infer.init_to_value(values=soln),
                    target_accept_prob=0.9,
                ),
                num_warmup=1000,
                num_samples=1000,
                progress_bar=True,
                jit_model_args=True
            )
            mcmc.run(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            inf_data = az.from_numpyro(mcmc)
            if save_trace:
                az.to_netcdf(inf_data, f'whitelight_trace_{n_planets}planets.nc')
            wl_samples = mcmc.get_samples()

            print(az.summary(inf_data, var_names=None, round_to=7))

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

            if detrending_type != 'none':
                bestfit_params_wl['c'] = jnp.nanmedian(wl_samples['c'])
                bestfit_params_wl['v'] = jnp.nanmedian(wl_samples['v']) if detrending_type != 'gp' else 0.0
            if detrending_type == 'explinear':
                bestfit_params_wl['A'] = jnp.nanmedian(wl_samples['A'])
                bestfit_params_wl['tau'] = jnp.nanmedian(wl_samples['tau'])
            if detrending_type == 'spot':
                bestfit_params_wl['spot_amp'] = jnp.nanmedian(wl_samples['spot_amp'])
                bestfit_params_wl['spot_mu'] = jnp.nanmedian(wl_samples['spot_mu'])
                bestfit_params_wl['spot_sigma'] = jnp.nanmedian(wl_samples['spot_sigma'])
            elif detrending_type == 'gp':
                bestfit_params_wl['logs2'] = jnp.nanmedian(wl_samples['logs2'])
                bestfit_params_wl['GP_log_sigma'] = jnp.nanmedian(wl_samples['GP_log_sigma'])
                bestfit_params_wl['GP_log_rho'] = jnp.nanmedian(wl_samples['GP_log_rho'])


            if detrending_type == 'linear':
                wl_transit_model = compute_lc_linear(bestfit_params_wl, data.wl_time)
            if detrending_type == 'explinear':
                wl_transit_model = compute_lc_explinear(bestfit_params_wl, data.wl_time)
            if detrending_type == 'spot':
                wl_transit_model = compute_lc_spot(bestfit_params_wl, data.wl_time)
            if detrending_type == 'none':
                wl_transit_model = compute_lc_none(bestfit_params_wl, data.wl_time)
            if detrending_type == 'gp':
                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                    data.wl_time,
                    diag=jnp.exp(bestfit_params_wl['logs2']),
                    mean=partial(compute_lc_gp_mean, bestfit_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux, data.wl_time).gp
                mu, var = cond_gp.loc, cond_gp.variance
                wl_transit_model = mu
            wl_residual = data.wl_flux - wl_transit_model


            wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))


            wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma

            wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))

            median_error_wl = np.nanmedian(wl_samples['error'])
            plt.plot(data.wl_time, wl_transit_model, color="r", lw=2, zorder=3)
            plt.errorbar(data.wl_time, data.wl_flux, yerr=median_error_wl, fmt='.', c='k', ms=1)

           # plt.title('WL GP fit')
            plt.savefig(f"{output_dir}/11_{instrument_full_str}_whitelightmodel.png")
            plt.show()
            plt.close()

            plt.errorbar(data.wl_time, wl_residual, yerr=median_error_wl, fmt='.', c='k', ms=2)
            plt.axhline(0, c='r', lw=2)
            plt.axhline(3*wl_sigma, c='b', lw=2, ls='--')
            plt.axhline(-3*wl_sigma, c='b', lw=2, ls='--')
            plt.axhline(4*wl_sigma, c='r', lw=2, ls='--')
            plt.axhline(-4*wl_sigma, c='r', lw=2, ls='--')
            plt.title('WL Pre-outlier rejection residual. Blue, Red Lines are +/- 3, 4 sigma')
            plt.savefig(f"{output_dir}/12_{instrument_full_str}_whitelightresidual.png")
            plt.show()
            plt.close()



            #plt.plot(data.wl_time, wl_transit_model, color="C0", lw=2)
            plt.scatter(data.wl_time, data.wl_flux, c='r', s=6)
            plt.scatter(data.wl_time[~wl_mad_mask], data.wl_flux[~wl_mad_mask], c='k', s=6)
            plt.title(f'Whitelight Outlier Rejection Plot (Removed = Red)')
            plt.savefig(f'{output_dir}/13_{instrument_full_str}_whitelightpostrejection.png')
            plt.show()
            plt.close()

            if detrending_type == 'linear':
                detrended_flux = 1.0 + data.wl_flux[~wl_mad_mask] - (bestfit_params_wl["c"] + bestfit_params_wl["v"] * (data.wl_time[~wl_mad_mask] - jnp.min(data.wl_time[~wl_mad_mask])))
            if detrending_type == 'explinear':
                detrended_flux = 1.0 + data.wl_flux[~wl_mad_mask] - ((bestfit_params_wl["c"] + bestfit_params_wl["v"] * (data.wl_time[~wl_mad_mask] - jnp.min(data.wl_time[~wl_mad_mask]))
                                                                + bestfit_params_wl['A'] * jnp.exp(-(data.wl_time[~wl_mad_mask] - jnp.min(data.wl_time[~wl_mad_mask]))/bestfit_params_wl['tau'])) )
            if detrending_type == 'spot':
                detrended_flux = 1.0 + data.wl_flux[~wl_mad_mask] - (bestfit_params_wl["c"] + bestfit_params_wl["v"] * (data.wl_time[~wl_mad_mask] - jnp.min(data.wl_time[~wl_mad_mask]))
                                                                    + spot_crossing(data.wl_time[~wl_mad_mask], bestfit_params_wl["spot_amp"], bestfit_params_wl["spot_mu"], bestfit_params_wl["spot_sigma"]))
            if detrending_type == 'none':
                detrended_flux = data.wl_flux[~wl_mad_mask]
            if detrending_type == 'gp':
                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                   data.wl_time[~wl_mad_mask],
                    diag=jnp.exp(bestfit_params_wl['logs2']),
                   # mean=partial(compute_lc_from_params, bestfit_params_wl, detrend_type='gp'),
                    mean=partial(compute_lc_gp_mean, bestfit_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux[~wl_mad_mask], data.wl_time[~wl_mad_mask]).gp
                mu, var = cond_gp.loc, cond_gp.variance
               # trend_flux = mu - compute_lc_from_params(bestfit_params_wl, data.wl_time[~wl_mad_mask], 'gp')
                trend_flux = mu - compute_lc_gp_mean(bestfit_params_wl, data.wl_time[~wl_mad_mask])
                detrended_flux = data.wl_flux[~wl_mad_mask] / (trend_flux + 1.0)
                planet_model_only = compute_lc_gp_mean(bestfit_params_wl, data.wl_time[~wl_mad_mask])

            plt.scatter(data.wl_time[~wl_mad_mask], detrended_flux, c='k', s=6)
            plt.title(f'Detrended WLC: Sigma {round(wl_sigma_post_clip*1e6)} PPM')
            plt.savefig(f'{output_dir}/14_{instrument_full_str}_whitelightdetrended.png')
            plt.show()
            plt.close()

            detrended_data = pd.DataFrame({'time': data.wl_time[~wl_mad_mask], 'flux': detrended_flux})
            detrended_data.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_detrended.csv', index=False)
            np.save(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy', arr=wl_mad_mask)
            if detrending_type == 'gp':
                df = pd.DataFrame({'wl_flux': data.wl_flux[~wl_mad_mask], 'gp_flux': mu, 'gp_err': jnp.sqrt(var), 'transit_model_flux': planet_model_only})
                df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            
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
                }
                
                # Add scalar parameters (same for all planets)
                if detrending_type != 'none':
                    row['c'] = bestfit_params_wl['c']
                    row['v'] = bestfit_params_wl['v']
                
                # Add LD coefficients
                row['u1'] = bestfit_params_wl['u'][0]
                row['u2'] = bestfit_params_wl['u'][1]
                
                # Add other detrending parameters if present
                if detrending_type == 'explinear':
                    row['A'] = bestfit_params_wl['A']
                    row['tau'] = bestfit_params_wl['tau']
                elif detrending_type == 'spot':
                    row['spot_amp'] = bestfit_params_wl['spot_amp']
                    row['spot_mu'] = bestfit_params_wl['spot_mu']
                    row['spot_sigma'] = bestfit_params_wl['spot_sigma']
                elif detrending_type == 'gp':
                    row['logs2'] = bestfit_params_wl['logs2']
                    row['GP_log_sigma'] = bestfit_params_wl['GP_log_sigma']
                    row['GP_log_rho'] = bestfit_params_wl['GP_log_rho']
                
                rows.append(row)
        
            df = pd.DataFrame(rows)
            df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv', index=False)
            print(f'Saved whitelight parameters to {output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
            bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')


            DURATION_BASE = bestfit_params_wl_df['duration'].values
            T0_BASE = bestfit_params_wl_df['t0'].values
            B_BASE = bestfit_params_wl_df['b'].values
            RORS_BASE = bestfit_params_wl_df['rors'].values
            DEPTH_BASE = RORS_BASE**2
            if detrending_type == 'spot':
                SPOT_AMP_BASE = bestfit_params_wl_df['spot_amp'].values[0]
                SPOT_MU_BASE = bestfit_params_wl_df['spot_mu'].values[0]
                SPOT_SIGMA_BASE = bestfit_params_wl_df['spot_sigma'].values[0]
            if detrending_type == 'gp':
                exit_link = input('The whitelight fitting has finished, as you are using a GP you should exit now and swap to GPU if available. Would you like to leave (Y/N)')
                if exit_link.lower() == 'y':
                    exit()
        else:
            print(f'GP trends already exist... If you want to refit GP on whitelight please remove {output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
            bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
            DURATION_BASE = bestfit_params_wl_df['duration'].values
            T0_BASE = bestfit_params_wl_df['t0'].values
            B_BASE = bestfit_params_wl_df['b'].values
            RORS_BASE = bestfit_params_wl_df['rors'].values
            DEPTH_BASE = RORS_BASE**2
    else:
        print(f'Whitelight outliers and bestfit parameters already exist, skipping whitelight fit. If you want to fit whitelight please delete {output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
        wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
        bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
        DURATION_BASE = bestfit_params_wl_df['duration'].values
        T0_BASE = bestfit_params_wl_df['t0'].values
        B_BASE = bestfit_params_wl_df['b'].values
        RORS_BASE = bestfit_params_wl_df['rors'].values
        DEPTH_BASE = RORS_BASE**2
        if detrending_type == 'spot':
            SPOT_AMP_BASE = bestfit_params_wl_df['spot_amp'].values[0]
            SPOT_MU_BASE = bestfit_params_wl_df['spot_mu'].values[0]
            SPOT_SIGMA_BASE = bestfit_params_wl_df['spot_sigma'].values[0]
    key_lr, key_hr, key_map_lr, key_mcmc_lr, key_map_hr, key_mcmc_hr, key_prior_pred = jax.random.split(key_master, 7)

    need_lowres_analysis = interpolate_trend or interpolate_ld or need_lowres

    # init vars that might be set in low-res analysis
    trend_fixed_hr = None
    ld_fixed_hr = None
    ld_grid_hr = None
    best_poly_coeffs_c = None
    best_poly_coeffs_v = None
    best_poly_coeffs_u1 = None
    best_poly_coeffs_u2 = None
    best_poly_coeffs_A = None
    best_poly_coeffs_tau = None
    best_poly_coeffs_spot_amp = None
    best_poly_coeffs_spot_mu = None
    best_poly_coeffs_spot_sigma = None



    # --- Low-resolution Analysis ---
    if need_lowres_analysis:
        print(f"\n--- Running Low-Resolution Analysis (Binned to {lr_bin_str}) ---")

        ##### APPLY OUTLIER MASK HERE ####
        time_lr = jnp.array(data.time[~wl_mad_mask])
        flux_lr = jnp.array(data.flux_lr[:, ~wl_mad_mask])
        flux_err_lr = jnp.array(data.flux_err_lr[:, ~wl_mad_mask])
        num_lcs_lr = jnp.array(data.flux_err_lr.shape[0])

        if detrending_type == 'gp':
            gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            gp_trend = jnp.array(gp_df['gp_flux'].values - gp_df['transit_model_flux'].values)
            detrend_type_multiwave = 'gp_spectroscopic'
        else:
            detrend_type_multiwave = detrending_type
            gp_trend = None


        assert time_lr.dtype == 'float64', "time_lr should be float64"
        assert flux_lr.dtype == 'float64', "indiv_y_lr should be float64"
        assert flux_err_lr.dtype == 'float64', "yerr_lr should be float64"
        assert len(time_lr.shape) == 1, "t_lr should be 1D array"
        assert len(flux_lr.shape) == 2, "indiv_y_lr should be 2D array (channels, time)"
        assert len(flux_err_lr.shape) == 2, "yerr_lr should be 2D array (channels, time)"
        assert time_lr.shape[0] == flux_lr.shape[1], "t_lr and indiv_y_lr should have same number of time points"
        assert time_lr.shape[0] == flux_err_lr.shape[1], "t_lr and yerr_lr should have same number of time points"

        print(f"Low-res: {num_lcs_lr} light curves.")

        DEPTHS_BASE_LR = jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))

        if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM' or instrument == 'MIRI/LRS':
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr , instrument)
        elif instrument == 'NIRISS/SOSS':
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr, instrument, order=order)



        if fix_ld is True:
            ld_fixed_lr = U_mu_lr
            print("Using fixed limb darkening coefficients for low-resolution analysis.")
        else:
            ld_fixed_lr = None

        init_params_lr = {
            "u": U_mu_lr,
            "depths": jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))
        }
        if detrend_type_multiwave != 'none':
            init_params_lr['c'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['c'].values[0])
            if detrend_type_multiwave != 'gp_spectroscopic':
                init_params_lr['v'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['v'].values[0])
        if detrend_type_multiwave == 'explinear':
            init_params_lr['A'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['A'].values[0])
            init_params_lr['tau'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['tau'].values[0])
        if detrend_type_multiwave == 'spot':
            init_params_lr['spot_amp'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['spot_amp'].values[0])
            init_params_lr['spot_mu'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['spot_mu'].values[0])
            init_params_lr['spot_sigma'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['spot_sigma'].values[0])
        print("Sampling low-res model using MCMC to find median coefficients...")

        lr_trend_mode = 'free'

        lr_ld_mode = 'free'
        if flags.get('fix_ld', False):
            lr_ld_mode = 'fixed'


        lr_model_for_run = create_vectorized_model(
            detrend_type=detrend_type_multiwave,
            ld_mode=lr_ld_mode,
            trend_mode=lr_trend_mode,
            n_planets=n_planets
        )


        model_run_args_lr = {
        'mu_duration': DURATION_BASE,
        'mu_t0': T0_BASE,
        'mu_b': B_BASE,
        'mu_depths': DEPTHS_BASE_LR,
        'PERIOD': PERIOD_FIXED,
        }
        if lr_ld_mode == 'fixed':
            model_run_args_lr['ld_fixed'] = U_mu_lr
        if lr_ld_mode == 'free':
            model_run_args_lr['mu_u_ld'] = U_mu_lr
        if detrend_type_multiwave == 'spot':
            model_run_args_lr['mu_spot_amp'] = SPOT_AMP_BASE
            model_run_args_lr['mu_spot_mu'] = SPOT_MU_BASE
            model_run_args_lr['mu_spot_sigma'] = SPOT_SIGMA_BASE
        if detrend_type_multiwave == 'gp_spectroscopic':
            model_run_args_lr['gp_trend'] = gp_trend
            init_params_lr['A_gp'] = jnp.ones(num_lcs_lr)


        samples_lr = get_samples(
            model=lr_model_for_run,
            key=key_mcmc_lr,
            t=time_lr,
            yerr=flux_err_lr,
            indiv_y=flux_lr,
            init_params=init_params_lr,
            **model_run_args_lr
        )

        ld_u_lr = np.array(samples_lr["u"])
        if detrend_type_multiwave != 'none':
            trend_c_lr = np.array(samples_lr["c"])
            trend_v_lr = np.array(samples_lr["v"])
        if detrend_type_multiwave == 'explinear':
            trend_A_lr = np.array(samples_lr["A"])
            trend_tau_lr = np.array(samples_lr["tau"])
        if detrend_type_multiwave == 'spot':
            trend_spot_amp_lr = np.array(samples_lr["spot_amp"])
            trend_spot_mu_lr = np.array(samples_lr["spot_mu"])
            trend_spot_sigma_lr = np.array(samples_lr["spot_sigma"])

        map_params_lr = {
            "duration": DURATION_BASE,
            "t0": T0_BASE,
            "b": B_BASE,
            "rors": jnp.nanmedian(samples_lr["rors"], axis=0),
            "u": jnp.nanmedian(ld_u_lr, axis=0),  "period": PERIOD_FIXED,
        }
        if detrend_type_multiwave != 'none':
            map_params_lr['c'] = jnp.nanmedian(samples_lr["c"], axis=0)
            if detrend_type_multiwave != 'gp_spectroscopic':
                map_params_lr['v'] = jnp.nanmedian(samples_lr["v"], axis=0)
        if detrend_type_multiwave == 'explinear':
            map_params_lr['A'] = jnp.nanmedian(samples_lr['A'], axis=0)
            map_params_lr['tau'] = jnp.nanmedian(samples_lr['tau'], axis=0)

        if detrend_type_multiwave == 'spot':
            map_params_lr['spot_amp'] = jnp.nanmedian(samples_lr['spot_amp'], axis=0)
            map_params_lr['spot_mu'] = jnp.nanmedian(samples_lr['spot_mu'], axis=0)
            map_params_lr['spot_sigma'] = jnp.nanmedian(samples_lr['spot_sigma'], axis=0)

        try:
            selected_kernel = COMPUTE_KERNELS[detrend_type_multiwave]
        except KeyError:
            raise ValueError(f"Unknown detrend_type: {detrend_type_multiwave}")

        in_axes_map = {
            'rors': 0,
            'u': 0,
        }
        if detrend_type_multiwave != 'none':
            in_axes_map['c'] = 0
            if detrend_type_multiwave != 'gp_spectroscopic':
                in_axes_map['v'] = 0
        if detrend_type_multiwave == 'explinear':
            in_axes_map.update({'A': 0, 'tau': 0})
        if detrend_type_multiwave == 'spot':
            in_axes_map.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})

        final_in_axes = {k: in_axes_map.get(k, None) for k in map_params_lr.keys()}

        if detrend_type_multiwave == 'gp_spectroscopic':
            map_params_lr['A_gp'] = jnp.nanmedian(samples_lr['A_gp'], axis=0)
            final_in_axes['A_gp'] = 0
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None, None))(map_params_lr, time_lr, gp_trend)
        else:
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None))(map_params_lr, time_lr)

        residuals = flux_lr - model_all
        plot_noise_binning(
        residuals,
        f"{output_dir}/25_{instrument_full_str}_{lr_bin_str}_noisebin.png",
        title=f"Noise binning — Low-res {lr_bin_str}"
        )


        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)

        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas

        time_mask = np.any(point_mask, axis=0)

        n_channels, n_times = flux_lr.shape

        if n_channels > 0:
            nrows = int(np.ceil(np.sqrt(n_channels)))
            ncols = int(np.ceil(n_channels / nrows))
        else:
            nrows, ncols = 1, 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)

        axes = axes.flatten()

        plot_count = 0

        for ch in range(n_channels):
            outlier_mask = point_mask[ch]

            if outlier_mask.any():
                ax = axes[plot_count]
                ax.scatter(time_lr, flux_lr[ch, :], c="k", label="data", s=10)

                ax.scatter(time_lr[outlier_mask], flux_lr[ch, outlier_mask],
                            c="r", label="outliers", s=10)

                plot_count += 1 #
        for i in range(plot_count, nrows * ncols):
            axes[i].axis('off')

        if plot_count > 0:
            plt.tight_layout()
            plt.savefig(f"{output_dir}/21_{instrument_full_str}_lowresoutliers.png")
        else:
            print("No channels with outliers found to plot.")

        plt.close(fig)

        valid = ~time_mask
        time_lr       = time_lr[valid]
        flux_lr = flux_lr[:, valid]
        flux_err_lr    = flux_err_lr[:, valid]
        if gp_trend is not None:
            gp_trend = gp_trend[valid]

        print("Plotting low-resolution fits and residuals...")
        #median_error_lr = np.nanmedian(samples_lr['error'], axis=0)
        #plot_wavelength_offset_summary(time_lr, flux_lr, median_error_lr, data.wavelengths_lr,
        median_total_error_lr = np.nanmedian(samples_lr['total_error'], axis=0)
        plot_wavelength_offset_summary(time_lr, flux_lr, median_total_error_lr, data.wavelengths_lr,
                                     map_params_lr, {"period": PERIOD_FIXED},
                                     f"{output_dir}/22_{instrument_full_str}_{lr_bin_str}_summary.png",
                                     detrend_type=detrend_type_multiwave,
                                     gp_trend=gp_trend)

        # Polynomial Fitting for Interpolation
        poly_orders = [1, 2, 3, 4]
        wl_lr = np.array(data.wavelengths_lr)

        if detrending_type != 'none':
            print("Fitting polynomials to trend coefficients...")
            best_poly_coeffs_c, best_order_c, _ = fit_polynomial(wl_lr, trend_c_lr, poly_orders)
            best_poly_coeffs_v, best_order_v, _ = fit_polynomial(wl_lr, trend_v_lr, poly_orders)
            print(f"Selected polynomial degrees: c={best_order_c}, v={best_order_v}")

            plot_poly_fit(wl_lr, trend_c_lr, best_poly_coeffs_c, best_order_c,
                            "Wavelength (μm)", "Trend coefficient c", "Trend Offset (c) Polynomial Fit",
                            f"{output_dir}/2optional1_{instrument_full_str}_{lr_bin_str}_cinterp.png")
            plot_poly_fit(wl_lr, trend_v_lr, best_poly_coeffs_v, best_order_v,
                            "Wavelength (μm)", "Trend coefficient v", "Trend Slope (v) Polynomial Fit",
                            f"{output_dir}/2optional2_{instrument_full_str}_{lr_bin_str}_vinterp.png")
        if detrend_type_multiwave == 'explinear':
            best_poly_coeffs_A, best_order_A, _ = fit_polynomial(wl_lr, trend_A_lr, poly_orders)
            best_poly_coeffs_tau, best_order_tau, _ = fit_polynomial(wl_lr, trend_tau_lr, poly_orders)
            print(f"Selected polynomial degrees: A={best_order_A}, tau={best_order_tau}")
        if detrend_type_multiwave == 'spot':
            best_poly_coeffs_spot_amp, best_order_spot_amp, _ = fit_polynomial(wl_lr, trend_spot_amp_lr, poly_orders)
            best_poly_coeffs_spot_mu, best_order_spot_mu, _ = fit_polynomial(wl_lr, trend_spot_mu_lr, poly_orders)
            best_poly_coeffs_spot_sigma, best_order_spot_sigma, _ = fit_polynomial(wl_lr, trend_spot_sigma_lr, poly_orders)
            print(f"Selected polynomial degrees: spot amp={best_order_spot_amp}, spot mu={best_order_spot_mu}, spot sigma={best_order_spot_sigma}")


        if interpolate_ld:
            print("Fitting polynomials to limb darkening coefficients...")
            best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 0], poly_orders)
            best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 1], poly_orders)
            print(f"Selected polynomial degrees: u1={best_order_u1}, u2={best_order_u2}")
            plot_poly_fit(wl_lr, ld_u_lr[:,:, 0], best_poly_coeffs_u1, best_order_u1,
                        "Wavelength (μm)", "LD coefficient u1", "Limb Darkening u1 Polynomial Fit",
                        f"{output_dir}/2optional3_{instrument_full_str}_{lr_bin_str}_u1interp.png")
            plot_poly_fit(wl_lr, ld_u_lr[:,:, 1], best_poly_coeffs_u2, best_order_u2,
                        "Wavelength (μm)", "LD coefficient u2", "Limb Darkening u2 Polynomial Fit",
                        f"{output_dir}/2optional4_{instrument_full_str}_{lr_bin_str}_u2interp.png")


        print("Plotting and saving lowres transmission spectrum...")
        plot_transmission_spectrum(wl_lr, samples_lr["rors"],
                            f"{output_dir}/24_{instrument_full_str}_{lr_bin_str}_spectrum")
        save_results(wl_lr, samples_lr, f"{output_dir}/{instrument_full_str}_{lr_bin_str}.csv")

        #oot_mask_lr = (time_lr < T0_BASE - 0.6 * DURATION_BASE) | (time_lr > T0_BASE + 0.6 * DURATION_BASE)    
        in_transit_mask = jnp.zeros_like(time_lr, dtype=bool)
        for t0, duration in zip(T0_BASE, DURATION_BASE):
            in_transit_mask |= (time_lr >= t0 - 0.6 * duration) & (time_lr <= t0 + 0.6 * duration)
    
        oot_mask_lr = ~in_transit_mask

        def calc_rms(y_bin):
            baseline = y_bin[oot_mask_lr]
            return jnp.nanmedian(jnp.abs(baseline - jnp.nanmedian(baseline))) * 1.4826

        rms_vals = jax.vmap(calc_rms)(flux_lr)

        plt.figure(figsize=(8,5))
        plt.scatter(data.wavelengths_lr, rms_vals * 1e6, c='k', label='Measured RMS')
        plt.scatter(data.wavelengths_lr, median_total_error_lr * 1e6, c='r', marker='x', label='Derived Total Error')
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Per Wavelength Noise (ppm)")
        plt.legend()
       # plt.title("Out‑of‑Transit RMS vs Wavelength")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/24_{instrument_full_str}_{lr_bin_str}_rms.png')
        plt.close()
    # --- High-resolution Analysis ---
    print(f"\n--- Running High-Resolution Analysis (Binned to {hr_bin_str}) ---")

    ##### APPLY OUTLIER MASK HERE ####
    time_hr = jnp.array(data.time[~wl_mad_mask])
    flux_hr = jnp.array(data.flux_hr[:, ~wl_mad_mask])
    flux_err_hr = jnp.array(data.flux_err_hr[:, ~wl_mad_mask])

    if detrending_type == 'gp':
        gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
        gp_trend = jnp.array(gp_df['gp_flux'].values - gp_df['transit_model_flux'].values)
        detrend_type_multiwave = 'gp_spectroscopic'
    else:
        detrend_type_multiwave = detrending_type
        gp_trend = None


    if need_lowres_analysis:
        valid = ~time_mask
        time_hr       = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr    = flux_err_hr[:, valid]
        if gp_trend is not None:
            gp_trend = gp_trend[valid]
    assert time_hr.dtype == 'float64', "t_hr should be float64"
    assert flux_hr.dtype == 'float64', "indiv_y_hr should be float64"
    assert flux_err_hr.dtype == 'float64', "yerr_hr should be float64"
    assert len(time_hr.shape) == 1, "t_hr should be 1D array"
    assert len(flux_hr.shape) == 2, "indiv_y_hr should be 2D array (channels, time)"
    assert len(flux_err_hr.shape) == 2, "yerr_hr should be 2D array (channels, time)"
    assert time_hr.shape[0] == flux_hr.shape[1], "t_hr and indiv_y_hr should have same number of time points"
    assert time_hr.shape[0] == flux_err_hr.shape[1], "t_hr and yerr_hr should have same number of time points"

    num_lcs_hr = flux_err_hr.shape[0]
    print(f"High-res: {num_lcs_hr} light curves.")

    DEPTHS_BASE_HR = jnp.tile(DEPTH_BASE, (num_lcs_hr, 1))
    hr_ld_mode = 'free'
    if flags.get('interpolate_ld', False):
        hr_ld_mode = 'interpolated'
    elif flags.get('fix_ld', False):
        hr_ld_mode = 'fixed'

    hr_trend_mode = 'free'
    if flags.get('interpolate_trend', False):
        hr_trend_mode = 'fixed'

    model_run_args_hr = {}
    wl_hr = np.array(data.wavelengths_hr)

    if hr_ld_mode == 'interpolated':
        u1_interp_hr = np.polyval(best_poly_coeffs_u1, wl_hr)
        u2_interp_hr = np.polyval(best_poly_coeffs_u2, wl_hr)
        ld_interpolated_hr = jnp.array(np.column_stack((u1_interp_hr, u2_interp_hr)))
        model_run_args_hr['ld_interpolated'] = ld_interpolated_hr
        U_mu_hr_init = ld_interpolated_hr
        print("HR Run Config: Using INTERPOLATED limb darkening.")

    elif hr_ld_mode == 'fixed':
        if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM' or instrument == 'MIRI/LRS':
            U_mu_hr = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument)
        elif instrument == 'NIRISS/SOSS':
            U_mu_hr = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order)
        model_run_args_hr['ld_fixed'] = U_mu_hr
        U_mu_hr_init = U_mu_hr
        print("HR Run Config: Using FIXED limb darkening.")

    else: # hr_ld_mode == 'free'
        if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM' or instrument == 'MIRI/LRS':
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument)
        elif instrument == 'NIRISS/SOSS':
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order)
        model_run_args_hr['mu_u_ld'] = U_mu_hr_init
        print("HR Run Config: FITTING for limb darkening (free).")

    if hr_trend_mode == 'fixed':
        c_interp_hr = np.polyval(best_poly_coeffs_c, wl_hr)
        v_interp_hr = np.polyval(best_poly_coeffs_v, wl_hr)
        trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr))
        if detrend_type_multiwave == 'explinear':
            A_interp_hr = np.polyval(best_poly_coeffs_A, wl_hr)
            tau_interp_hr = np.polyval(best_poly_coeffs_tau, wl_hr)
            trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr, A_interp_hr, tau_interp_hr))
        if detrend_type_multiwave == 'spot':
            spot_amp_interp_hr = np.polyval(best_poly_coeffs_spot_amp, wl_hr)
            spot_mu_interp_hr = np.polyval(best_poly_coeffs_spot_mu, wl_hr)
            spot_sigma_interp_hr = np.polyval(best_poly_coeffs_spot_sigma, wl_hr)
            trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr, spot_amp_interp_hr, spot_mu_interp_hr, spot_sigma_interp_hr))

        model_run_args_hr['trend_fixed'] = jnp.array(trend_fixed_hr)
        print("HR Run Config: Using FIXED (interpolated) trend.")
    else: # hr_trend_mode == 'free'
        print("HR Run Config: FITTING for trend (free).")

    model_run_args_hr['mu_duration'] = DURATION_BASE
    model_run_args_hr['mu_t0'] = T0_BASE
    model_run_args_hr['mu_b'] = B_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = PERIOD_FIXED
    if detrend_type_multiwave == 'spot':
        model_run_args_hr['mu_spot_amp'] = SPOT_AMP_BASE
        model_run_args_hr['mu_spot_mu'] = SPOT_MU_BASE
        model_run_args_hr['mu_spot_sigma'] = SPOT_SIGMA_BASE
    init_params_hr = {
        "depths": DEPTHS_BASE_HR,
        "u": U_mu_hr_init,
    }
    if hr_trend_mode == 'free':
        if detrend_type_multiwave != 'none':
            init_params_hr["c"] = np.polyval(best_poly_coeffs_c, wl_hr)
            if detrend_type_multiwave != 'gp_spectroscopic':
                init_params_hr["v"] = np.polyval(best_poly_coeffs_v, wl_hr)
        if detrend_type_multiwave == 'explinear':
            init_params_hr["A"] = np.polyval(best_poly_coeffs_A, wl_hr)
            init_params_hr["tau"] = np.polyval(best_poly_coeffs_tau, wl_hr)
        if detrend_type_multiwave == 'spot':
            init_params_hr["spot_amp"] = np.polyval(best_poly_coeffs_spot_amp, wl_hr)
            init_params_hr["spot_mu"] = np.polyval(best_poly_coeffs_spot_mu, wl_hr)
            init_params_hr["spot_sigma"] = np.polyval(best_poly_coeffs_spot_sigma, wl_hr)
    hr_model_for_run = create_vectorized_model(
        detrend_type=detrend_type_multiwave,
        ld_mode=hr_ld_mode,
        trend_mode=hr_trend_mode,
        n_planets=n_planets
    )
    if detrend_type_multiwave == 'gp_spectroscopic':
        model_run_args_hr['gp_trend'] = gp_trend
        init_params_hr['A_gp'] = jnp.ones(num_lcs_hr)

    samples_hr = get_samples(
        model=hr_model_for_run,
        key=key_mcmc_hr,
        t=time_hr,
        yerr=flux_err_hr,
        indiv_y=flux_hr,
        init_params=init_params_hr,
        **model_run_args_hr
    )
    map_params_hr = {
        "duration": DURATION_BASE,
        "t0": T0_BASE,
        "b": B_BASE,
        "rors": jnp.nanmedian(samples_hr["rors"], axis=0),
        "period": PERIOD_FIXED
    }
    # include LD if fitted
    if "u" in samples_hr:
        map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    # include trend terms if present in samples
    if detrend_type_multiwave != 'none' and "c" in samples_hr:
        map_params_hr["c"] = jnp.nanmedian(samples_hr["c"], axis=0)
        if detrend_type_multiwave != 'gp_spectroscopic':
            map_params_hr["v"] = jnp.nanmedian(samples_hr["v"], axis=0)
    if detrend_type_multiwave == 'explinear' and "A" in samples_hr:
        map_params_hr["A"] = jnp.nanmedian(samples_hr["A"], axis=0)
        map_params_hr["tau"] = jnp.nanmedian(samples_hr["tau"], axis=0)
    if detrend_type_multiwave == 'spot' and "spot_amp" in samples_hr:
        map_params_hr["spot_amp"] = jnp.nanmedian(samples_hr["spot_amp"], axis=0)
        map_params_hr["spot_mu"]  = jnp.nanmedian(samples_hr["spot_mu"], axis=0)
        map_params_hr["spot_sigma"] = jnp.nanmedian(samples_hr["spot_sigma"], axis=0)

    # prepare in_axes similar to low-res mapping so jax.vmap works
    in_axes_map_hr = {"rors": 0, "u": 0}
    if detrend_type_multiwave != 'none':
        in_axes_map_hr.update({"c": 0})
        if detrend_type_multiwave != 'gp_spectroscopic':
            in_axes_map_hr.update({"v": 0})
    if detrend_type_multiwave == 'explinear':
        in_axes_map_hr.update({"A": 0, "tau": 0})
    if detrend_type_multiwave == 'spot':
        in_axes_map_hr.update({"spot_amp": 0, "spot_mu": 0, "spot_sigma": 0})

    final_in_axes_hr = {k: in_axes_map_hr.get(k, None) for k in map_params_hr.keys()}

    # compute model and residuals on the HR time grid
    selected_kernel_hr = COMPUTE_KERNELS[detrend_type_multiwave]
    if detrend_type_multiwave == 'gp_spectroscopic':
        map_params_hr['A_gp'] = jnp.nanmedian(samples_hr['A_gp'], axis=0)
        final_in_axes_hr['A_gp'] = 0
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None, None))(map_params_hr, time_hr, gp_trend)
    else:
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None))(map_params_hr, time_hr)
    residuals_hr = np.array(flux_hr - model_all_hr)

    # save noise-binning plot
    plot_noise_binning(
        residuals_hr,
        f"{output_dir}/36_{instrument_full_str}_{hr_bin_str}_noisebin.png",
        title=f"Noise binning — High-res {hr_bin_str}"
    )



    print("Plotting high-resolution fits and residuals...")
    median_total_error_hr = np.nanmedian(samples_hr['total_error'], axis=0)
    plot_wavelength_offset_summary(time_hr, flux_hr, median_total_error_hr, data.wavelengths_hr,
                                    map_params_hr, {"period": PERIOD_FIXED},
                                    f"{output_dir}/34_{instrument_full_str}_{hr_bin_str}_summary.png",
                                    detrend_type=detrend_type_multiwave,
                                    gp_trend=gp_trend)

    print("Plotting and saving final transmission spectrum...")
    plot_transmission_spectrum(wl_hr, samples_hr["rors"],
                            f"{output_dir}/31_{instrument_full_str}_{hr_bin_str}_spectrum")
    save_results(wl_hr, samples_hr,  f"{output_dir}/{instrument_full_str}_{hr_bin_str}.csv")

    if "u" in samples_hr:
        u1_16, u1_median, u1_84 = np.percentile(np.array(samples_hr["u"][:,:,0]), [16, 50, 84], axis=0) # Shape: (n_samples, n_lcs, 2)
        u1_err_low = u1_median - u1_16
        u1_err_high = u1_84 - u1_median
        u1_yerr = np.array([u1_err_low, u1_err_high])

        u2_16, u2_median, u2_84 = np.percentile(np.array(samples_hr["u"][:,:,1]), [16, 50, 84], axis=0)
        u2_err_low = u2_median - u2_16
        u2_err_high = u2_84 - u2_median
        u2_yerr = np.array([u2_err_low, u2_err_high])

        plt.figure(figsize=(10, 6))
        plt.errorbar(wl_hr, u1_median, yerr=u1_yerr, fmt='o', markersize=4,
                    capsize=3, elinewidth=1, markeredgewidth=1, mfc='w',mec='k', label='u1 Median ± 1σ')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('u1')
        plt.tight_layout()
        u1_save_path = f'{output_dir}/3optional1_{instrument_full_str}_{hr_bin_str}_u1.png'
        plt.savefig(u1_save_path)
        print(f"Saved u1 plot with uncertainties to {u1_save_path}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.errorbar(wl_hr, u2_median, yerr=u2_yerr, fmt='o', markersize=4,
                    capsize=3, elinewidth=1, markeredgewidth=1, mfc='w',mec='k', label='u2 Median ± 1σ')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('u2')
        plt.tight_layout()
        u2_save_path = f'{output_dir}/3optional2_{instrument_full_str}_{hr_bin_str}_u2.png'
        plt.savefig(u2_save_path)
        print(f"Saved u2 plot with uncertainties to {u2_save_path}")
        plt.close()

    else:
        print("LD coefficients were fixed—skipping u₁–u₂ plots.")

   # oot_mask = (time_hr < T0_BASE - 0.6 * DURATION_BASE) | (time_hr > T0_BASE + 0.6 * DURATION_BASE)
    in_transit_mask = jnp.zeros_like(time_hr, dtype=bool)
    for t0, duration in zip(T0_BASE, DURATION_BASE):
        in_transit_mask |= (time_hr >= t0 - 0.6 * duration) & (time_hr <= t0 + 0.6 * duration)
    
    oot_mask_hr = ~in_transit_mask

    def calc_rms(y_bin):
        baseline = y_bin[oot_mask]
        return jnp.nanmedian(jnp.abs(baseline - jnp.nanmedian(baseline))) * 1.4826

    rms_vals = jax.vmap(calc_rms)(flux_hr)
    median_total_error_hr = np.nanmedian(samples_hr['total_error'], axis=0)
    plt.figure(figsize=(8,5))
    plt.scatter(wl_hr, rms_vals*1e6, c='k', label='Measured RMS')
    plt.scatter(wl_hr, median_total_error_hr * 1e6, c='r', marker='x', label='Derived Total Error')
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Per-Wavelength Noise (ppm)")
    plt.legend()
   # plt.title("Out‑of‑Transit RMS vs Wavelength")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/32_{instrument_full_str}_{hr_bin_str}_rms.png')
    plt.close()

    #depth_hr = np.nanmedian(samples_hr["depths"], axis=0)
    #depth_chain = np.array(samples_hr["depths"])
    #cov_hr = np.cov(depth_chain, rowvar=False)
    #np.savez(
    #f"{output_dir}/spectrum_native_cov.npz",
    #wavelength=wl_hr,
    #depth=depth_hr,
    #covariance=cov_hr
    #)


    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


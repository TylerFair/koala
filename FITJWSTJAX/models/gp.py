import jax.numpy as jnp
from functools import partial
import tinygp
from .core import compute_transit_model_auto

# --- GP MEAN FUNCTIONS ---
def compute_lc_gp_mean(params, t):
    """The mean function for the simple GP model is just the transit + constant."""
    return compute_transit_model_auto(params, t) + params["c"]

def compute_lc_linear_gp_mean(params, t):
    """The mean function for the linear + GP model."""
    lc_transit = compute_transit_model_auto(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend

def compute_lc_quadratic_gp_mean(params, t):
    """The mean function for the quadratic + GP model."""
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend

def compute_lc_cubic_gp_mean(params, t):
    """The mean function for the cubic + GP model."""
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
    )
    return lc_transit + trend

def compute_lc_quartic_gp_mean(params, t):
    """The mean function for the quartic + GP model."""
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
        + params["v4"] * t_norm**4
    )
    return lc_transit + trend

def compute_lc_explinear_gp_mean(params, t):
    """The mean function for the exp-linear + GP model."""
    lc_transit = compute_transit_model_auto(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-(t-jnp.min(t)) / params['tau'])
    return lc_transit + trend

# --- SPECTROSCOPIC GP FUNCTIONS ---
def compute_lc_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    return lc_transit + params["c"] + params["A_gp"] * gp_trend

def compute_lc_linear_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_quadratic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_cubic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
    )
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_quartic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
        + params["v4"] * t_norm**4
    )
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_explinear_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-(t-jnp.min(t)) / params['tau'])
    return lc_transit + trend + params["A_gp"] * gp_trend

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

def build_gp_cubic(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_cubic_gp_mean, params))

def build_gp_quartic(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_quartic_gp_mean, params))

def build_gp_explinear(params, t, error):
    kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(params['GP_log_rho']),
                sigma=jnp.exp(params['GP_log_sigma']),
            )
    return tinygp.GaussianProcess(kernel, t, diag=error**2,
              mean=partial(compute_lc_explinear_gp_mean, params))


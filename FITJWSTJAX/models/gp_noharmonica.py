import jax.numpy as jnp
from functools import partial
import tinygp
from .core import compute_transit_model

def _trend_time(t):
    return t - jnp.min(t)

def _poly_trend(params, t_norm, order):
    trend = params["c"] + params["v"] * t_norm
    if order >= 2:
        trend = trend + params["v2"] * t_norm**2
    if order >= 3:
        trend = trend + params["v3"] * t_norm**3
    if order >= 4:
        trend = trend + params["v4"] * t_norm**4
    return trend

def compute_lc_gp_mean(params, t):
    return compute_transit_model(params, t) + params["c"]

def compute_lc_linear_gp_mean(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=1)
    return lc_transit + trend

def compute_lc_quadratic_gp_mean(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=2)
    return lc_transit + trend

def compute_lc_cubic_gp_mean(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=3)
    return lc_transit + trend

def compute_lc_quartic_gp_mean(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=4)
    return lc_transit + trend

def compute_lc_explinear_gp_mean(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=1) + params["A"] * jnp.exp(-t_norm / params["tau"])
    return lc_transit + trend

def compute_lc_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_gp"] * gp_trend

def compute_lc_linear_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=1)
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_quadratic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=2)
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_cubic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=3)
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_quartic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=4)
    return lc_transit + trend + params["A_gp"] * gp_trend

def compute_lc_explinear_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model(params, t)
    t_norm = _trend_time(t)
    trend = _poly_trend(params, t_norm, order=1) + params["A"] * jnp.exp(-t_norm / params["tau"])
    return lc_transit + trend + params["A_gp"] * gp_trend

def _build_gp_kernel(params, t, error, mean_fn):
    kernel = tinygp.kernels.quasisep.Matern32(
        scale=jnp.exp(params["GP_log_rho"]),
        sigma=jnp.exp(params["GP_log_sigma"]),
    )
    return tinygp.GaussianProcess(kernel, t, diag=error**2, mean=partial(mean_fn, params))

def build_gp(params, t, error):
    return _build_gp_kernel(params, t, error, compute_lc_gp_mean)

def build_gp_linear(params, t, error):
    return _build_gp_kernel(params, t, error, compute_lc_linear_gp_mean)

def build_gp_quadratic(params, t, error):
    return _build_gp_kernel(params, t, error, compute_lc_quadratic_gp_mean)

def build_gp_cubic(params, t, error):
    return _build_gp_kernel(params, t, error, compute_lc_cubic_gp_mean)

def build_gp_quartic(params, t, error):
    return _build_gp_kernel(params, t, error, compute_lc_quartic_gp_mean)

def build_gp_explinear(params, t, error):
    return _build_gp_kernel(params, t, error, compute_lc_explinear_gp_mean)


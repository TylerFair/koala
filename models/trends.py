import os

import jax.numpy as jnp
from jax.nn import sigmoid
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from .common import compute_transit_model_auto, apply_systematics

def _trend_time(t):
    return t - jnp.min(t)

_JUMP_WIDTH_DAYS = 1e-4
_MAX_JUMP_WIDTH_DAYS = 30.0 / (24.0 * 60.0)
_SPOT_CENTER_SIGMA_DAYS = 0.01


def resolve_whitelight_trend_parameterization(configured="physical"):
    """Resolve the production option, retaining the diagnostic env override."""
    env_value = os.getenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION")
    value = env_value if env_value is not None and env_value.strip() else configured
    value = str(value or "physical").strip().lower()
    if value not in {"physical", "cadence"}:
        raise ValueError(
            "White-light trend parameterization must be 'physical' or "
            "'cadence'."
        )
    return value


def sample_ordered_spot_centers(
    spot_guess,
    spot_guess2,
    cadence,
    *,
    parameterization="physical",
):
    """Sample a canonical left/right pair with the exact unlabeled prior.

    Sorting two labeled components has two preimages.  The density below sums
    both labeled Normal densities and includes the midpoint/log-separation
    Jacobian, so imposing ``spot_mu < spot_mu2`` removes only the redundant
    labeling and preserves the prior on the physical two-template sum.
    """
    parameterization = resolve_whitelight_trend_parameterization(
        parameterization
    )
    cadence = jnp.asarray(cadence, dtype=jnp.float64)
    guess_left = jnp.asarray(spot_guess, dtype=jnp.float64)
    guess_right = jnp.asarray(spot_guess2, dtype=jnp.float64)
    guess_midpoint = 0.5 * (guess_left + guess_right)
    guess_separation = jnp.maximum(jnp.abs(guess_right - guess_left), cadence)
    midpoint_scale = _SPOT_CENTER_SIGMA_DAYS / jnp.sqrt(2.0)

    if parameterization == "cadence":
        midpoint_base = dist.Normal(0.0, midpoint_scale / cadence)
        separation_base = dist.Normal(
            jnp.log(guess_separation / cadence), 1.0
        )
        midpoint_coordinate = numpyro.sample(
            "spot_center_midpoint_offset_cadences", midpoint_base
        )
        log_separation_coordinate = numpyro.sample(
            "log_spot_center_separation_cadences", separation_base
        )
        midpoint = numpyro.deterministic(
            "spot_center_midpoint",
            guess_midpoint + cadence * midpoint_coordinate,
        )
        separation = cadence * jnp.exp(log_separation_coordinate)
        log_abs_det_jacobian = jnp.log(cadence) + jnp.log(separation)
    else:
        midpoint_base = dist.Normal(guess_midpoint, midpoint_scale)
        separation_base = dist.Normal(jnp.log(guess_separation), 1.0)
        midpoint_coordinate = numpyro.sample(
            "spot_center_midpoint", midpoint_base
        )
        log_separation_coordinate = numpyro.sample(
            "log_spot_center_separation", separation_base
        )
        midpoint = midpoint_coordinate
        separation = jnp.exp(log_separation_coordinate)
        log_abs_det_jacobian = jnp.log(separation)

    separation = numpyro.deterministic(
        "spot_center_separation", separation
    )
    spot_mu = midpoint - 0.5 * separation
    spot_mu2 = midpoint + 0.5 * separation

    first_prior = dist.Normal(guess_left, _SPOT_CENTER_SIGMA_DAYS)
    second_prior = dist.Normal(guess_right, _SPOT_CENTER_SIGMA_DAYS)
    direct_log_density = (
        first_prior.log_prob(spot_mu) + second_prior.log_prob(spot_mu2)
    )
    swapped_log_density = (
        first_prior.log_prob(spot_mu2) + second_prior.log_prob(spot_mu)
    )
    ordered_log_density = logsumexp(
        jnp.stack((direct_log_density, swapped_log_density))
    )
    base_log_density = midpoint_base.log_prob(midpoint_coordinate)
    base_log_density += separation_base.log_prob(log_separation_coordinate)
    numpyro.factor(
        "ordered_spot_center_prior",
        ordered_log_density + log_abs_det_jacobian - base_log_density,
    )
    return (
        numpyro.deterministic("spot_mu", spot_mu),
        numpyro.deterministic("spot_mu2", spot_mu2),
    )


def sample_step_width(t, prior_params, mode=None, parameterization=None):
    """Sample or fix the sigmoid width and expose common physical units."""
    mode = str(mode or prior_params.get("step_width_mode", "free")).lower()
    if mode == "fixed":
        width = jnp.asarray(
            prior_params.get("step_width_days", _JUMP_WIDTH_DAYS), dtype=jnp.float64
        )
    elif mode == "free":
        cadence = jnp.median(jnp.diff(jnp.sort(jnp.asarray(t))))
        parameterization = resolve_whitelight_trend_parameterization(
            parameterization
        )
        if parameterization == "cadence":
            log_width_low = jnp.log(0.5)
            log_width_high = jnp.log(_MAX_JUMP_WIDTH_DAYS / cadence)
            log_width_unit_logit = numpyro.sample(
                "log_width_unit_logit", dist.Logistic(0.0, 1.0)
            )
            log_width_cadences = numpyro.deterministic(
                "log_width_cadences",
                log_width_low
                + (log_width_high - log_width_low) * sigmoid(log_width_unit_logit),
            )
            log_width = numpyro.deterministic(
                "log_width", log_width_cadences + jnp.log(cadence)
            )
        elif parameterization == "physical":
            log_width = numpyro.sample(
                "log_width",
                dist.Uniform(
                    jnp.log(0.5 * cadence), jnp.log(_MAX_JUMP_WIDTH_DAYS)
                ),
            )
        width = jnp.exp(log_width)
    else:
        raise ValueError("step_width_mode must be 'free' or 'fixed'.")
    width = numpyro.deterministic("width", width)
    numpyro.deterministic("width_minutes", width * 24.0 * 60.0)
    return width

def _soft_step(t, t_jump, width=_JUMP_WIDTH_DAYS):
    """Smooth unit step with ``width`` expressed in days."""
    return sigmoid((t - t_jump) / width)

def _poly_trend(params, t_norm, order):
    trend = params["c"] + params["v"] * t_norm
    if order >= 2:
        trend = trend + params["v2"] * t_norm**2
    if order >= 3:
        trend = trend + params["v3"] * t_norm**3
    if order >= 4:
        trend = trend + params["v4"] * t_norm**4
    return trend

def spot_crossing(t, amp, mu, sigma):
    sigma = jnp.maximum(jnp.abs(sigma), 1e-6)
    return amp * jnp.exp(-0.5 * (t - mu) ** 2 / sigma ** 2)

def compute_lc_none(params, t):
    return apply_systematics(compute_transit_model_auto(params, t), 1.0)

def compute_lc_linear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    trend = _poly_trend(params, t_norm, order=1)
    return apply_systematics(lc_transit, trend)

def compute_lc_quadratic(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    trend = _poly_trend(params, t_norm, order=2)
    return apply_systematics(lc_transit, trend)

def compute_lc_quadratic_spot(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    trend = _poly_trend(params, t_norm, order=2)
    return apply_systematics(lc_transit, trend + spot)

def compute_lc_cubic(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    trend = _poly_trend(params, t_norm, order=3)
    return apply_systematics(lc_transit, trend)

def compute_lc_quartic(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    trend = _poly_trend(params, t_norm, order=4)
    return apply_systematics(lc_transit, trend)

def compute_lc_linear_discontinuity(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    jump = params["jump"] * _soft_step(
        t, params["t_jump"], params.get("width", _JUMP_WIDTH_DAYS)
    )
    trend = _poly_trend(params, t_norm, order=1) + jump
    return apply_systematics(lc_transit, trend)

def compute_lc_explinear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    trend = _poly_trend(params, t_norm, order=1) + params["A"] * jnp.exp(-t_norm / params["tau"])
    return apply_systematics(lc_transit, trend)

def compute_lc_explinear_spectroscopic(params, t, exp_trend):
    """Exponential-linear trend with a white-light-fixed exponential shape."""
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    return apply_systematics(lc_transit, params["c"] + params["v"] * t_norm + params["A"] * exp_trend)

def compute_lc_spot(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    trend = _poly_trend(params, t_norm, order=1)
    return apply_systematics(lc_transit, trend + spot)

def compute_lc_2spot(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    spot_1 = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    spot_2 = spot_crossing(t, params["spot_amp2"], params["spot_mu2"], params["spot_sigma2"])
    trend = _poly_trend(params, t_norm, order=1)
    return apply_systematics(lc_transit, trend + spot_1 + spot_2)

def compute_lc_spot_linear_discontinuity(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    jump = params["jump"] * _soft_step(
        t, params["t_jump"], params.get("width", _JUMP_WIDTH_DAYS)
    )
    trend = _poly_trend(params, t_norm, order=1) + spot + jump
    return apply_systematics(lc_transit, trend)

def compute_lc_spot_explinear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    trend = _poly_trend(params, t_norm, order=1) + spot + explinear
    return apply_systematics(lc_transit, trend)

def compute_lc_2spot_explinear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    spot_1 = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    spot_2 = spot_crossing(t, params["spot_amp2"], params["spot_mu2"], params["spot_sigma2"])
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    trend = _poly_trend(params, t_norm, order=1) + spot_1 + spot_2 + explinear
    return apply_systematics(lc_transit, trend)

def compute_lc_spot_spectroscopic(params, t, spot_trend):
    lc_transit = compute_transit_model_auto(params, t)
    return apply_systematics(lc_transit, params["c"] + params["A_spot"] * spot_trend)

def compute_lc_quadratic_spot_spectroscopic(params, t, spot_trend):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    trend = _poly_trend(params, t_norm, order=2) + params["A_spot"] * spot_trend
    return apply_systematics(lc_transit, trend)

def compute_lc_2spot_spectroscopic(params, t, spot_trend, spot_trend2):
    lc_transit = compute_transit_model_auto(params, t)
    return apply_systematics(lc_transit, params["c"] + params["A_spot"] * spot_trend + params["A_spot2"] * spot_trend2)

def compute_lc_linear_discontinuity_spectroscopic(params, t, jump_trend):
    lc_transit = compute_transit_model_auto(params, t)
    return apply_systematics(lc_transit, params["c"] + params["A_jump"] * jump_trend)

def compute_lc_spot_linear_discontinuity_spectroscopic(params, t, spot_trend, jump_trend):
    lc_transit = compute_transit_model_auto(params, t)
    return apply_systematics(lc_transit, params["c"] + params["A_spot"] * spot_trend + params["A_jump"] * jump_trend)

def compute_lc_spot_explinear_spectroscopic(params, t, spot_trend):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    return apply_systematics(lc_transit, params["c"] + params["v"] * t_norm + explinear + params["A_spot"] * spot_trend)

def compute_lc_2spot_explinear_spectroscopic(params, t, spot_trend, spot_trend2):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model_auto(params, t)
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    return (
        lc_transit
        + params["c"]
        + params["v"] * t_norm
        + explinear
        + params["A_spot"] * spot_trend
        + params["A_spot2"] * spot_trend2
    )

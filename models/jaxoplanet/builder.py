import os

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import numpyro
import numpyro.distributions as dist
import numpy as np

from ..common import apply_systematics, get_I_power2
from ..cadence_reduction import linear_spectro_trend_coefficient_names
from ..linear_marginalization import marginalized_log_likelihood_and_conditional
from ..ld_parameterization import Power2MaxtedTransform
from ..trend_marginal import build_marginalized_trend_design
from ..detrend import (
    COMPUTE_KERNELS,
    resolve_detrend_kernel,
    _split_components,
    _prepare_power2_poly,
)
from ..gp import GP_HYPERPARAMETER_BOUNDS, resolve_gp_builder, resolve_gp_solver
from ..trends import (
    resolve_whitelight_trend_parameterization,
    sample_step_width,
    spot_crossing,
)


_AUTO_CADENCE_REDUCTION_THRESHOLD = 5000


def _enforce_decorrelated_coefficient_support(coefficients, low, high):
    """Apply a numerically zero outside-support density and safe model values."""
    coefficients = jnp.asarray(coefficients)
    low = jnp.asarray(low, dtype=coefficients.dtype)
    high = jnp.asarray(high, dtype=coefficients.dtype)
    inside = jnp.all(
        jnp.isfinite(coefficients)
        & (coefficients >= low)
        & (coefficients <= high), axis=-1
    )
    # Unit validates its log factor when NumPyro validation is enabled, so an
    # IEEE -inf raises before HMC can reject the point. exp(-1e100) is exactly
    # zero in float64 while remaining a valid finite Unit parameter.
    numpyro.factor(
        "ld_decorrelated_support", jnp.where(inside, 0.0, -1.0e100)
    )
    midpoint = 0.5 * (low + high)
    safe = jnp.nan_to_num(
        coefficients, nan=midpoint, posinf=high, neginf=low
    )
    return jnp.clip(safe, low, high)

from ..harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_cos_i_from_geometry,
    harmonica_duration_from_geometry,
    harmonica_impact_param_from_cos_i,
)
from .core import (
    build_transit_phase_offsets,
    compute_transit_model,
    compute_transit_model_window,
    resolve_jaxoplanet_kernel,
)
from models.limb_darkening_config import GAUSSIAN_LD_WIDTH

NUTS_KWARGS = {
    "dense_mass": False,
    "regularize_mass_matrix": True,
    "target_accept_prob": 0.8,
}


def _phase_flux_interval(conditioning_flux):
    conditioning_flux = jnp.asarray(conditioning_flux, dtype=jnp.float64)
    finfo = jnp.finfo(conditioning_flux.dtype)
    conditioning_flux = jnp.nan_to_num(
        conditioning_flux,
        nan=finfo.tiny,
        posinf=0.1 * finfo.max,
        neginf=finfo.tiny,
    )
    conditioning_flux = jnp.maximum(conditioning_flux, finfo.tiny)
    return conditioning_flux / 5.0, 5.0 * conditioning_flux


def _conditional_phase_flux_from_quantile(
    quantile, conditioning_flux, center, width
):
    """Map a unit quantile to the original positive normal within ratio bounds."""
    quantile = jnp.asarray(quantile, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    width = jnp.asarray(width, dtype=jnp.float64)
    low, high = _phase_flux_interval(conditioning_flux)
    value = dist.TruncatedNormal(
        center, width, low=low, high=high
    ).icdf(quantile)
    # NumPyro's two-sided truncated-normal inverse CDF switches to survival
    # probabilities in the upper tail. Clipping only guards final floating
    # point roundoff; mathematically the inverse is already inside the bounds.
    return jnp.clip(value, low, high)


def phase_flux_to_conditional_quantile(value, conditioning_flux, center, width):
    """Return the unit quantile used to initialize a conditional phase flux."""
    value = jnp.asarray(value, dtype=jnp.float64)
    center = jnp.asarray(center, dtype=jnp.float64)
    width = jnp.asarray(width, dtype=jnp.float64)
    low, high = _phase_flux_interval(conditioning_flux)
    cdf_low = ndtr((low - center) / width)
    cdf_high = ndtr((high - center) / width)
    value_cdf = ndtr((value - center) / width)
    cdf_denominator = jnp.maximum(
        cdf_high - cdf_low, jnp.finfo(value_cdf.dtype).tiny
    )
    cdf_quantile = (value_cdf - cdf_low) / cdf_denominator

    survival_low = ndtr((center - low) / width)
    survival_high = ndtr((center - high) / width)
    survival_value = ndtr((center - value) / width)
    survival_denominator = jnp.maximum(
        survival_low - survival_high, jnp.finfo(value_cdf.dtype).tiny
    )
    survival_quantile = (
        survival_low - survival_value
    ) / survival_denominator
    quantile = jnp.where(low < center, cdf_quantile, survival_quantile)
    eps = jnp.finfo(value_cdf.dtype).eps
    return jnp.clip(quantile, eps, 1.0 - eps)


def _positive_normal_interval_log_mass(conditioning_flux, center, width):
    """Log probability of the physical ratio interval under TN(low=0)."""
    center = jnp.asarray(center, dtype=jnp.float64)
    width = jnp.asarray(width, dtype=jnp.float64)
    low, high = _phase_flux_interval(conditioning_flux)
    log_cdf_high = log_ndtr((high - center) / width)
    log_cdf_low = log_ndtr((low - center) / width)
    log_survival_low = log_ndtr((center - low) / width)
    log_survival_high = log_ndtr((center - high) / width)

    use_cdf = low < center
    log_larger = jnp.where(use_cdf, log_cdf_high, log_survival_low)
    log_smaller = jnp.where(use_cdf, log_cdf_low, log_survival_high)
    ratio = jnp.exp(jnp.minimum(log_smaller - log_larger, 0.0))
    log_interval = log_larger + jnp.log1p(-ratio)
    return log_interval - log_ndtr(center / width)


def _sample_surface_parameters(surface_config, n_planets, num_lcs=None):
    """Sample the compact surface-model priors and expose output-friendly units."""
    config = surface_config or {"model": "transit", "spots": ()}
    model = config.get("model", "transit")
    if model == "transit" and not config.get("spots"):
        return {}
    result = {"_surface_model": model, "_stellar_spots": config.get("spots", ())}
    shape = () if num_lcs is None else (int(num_lcs),)

    def sample_flux(name):
        values = []
        specs = config[f"{name}_spec"]
        for index in range(n_planets):
            values.append(
                sample_parameter(specs[index], f"_{name}_{index}", shape)
            )
        axis = 0 if num_lcs is None else 1
        value = numpyro.deterministic(name, jnp.stack(values, axis=axis))
        numpyro.deterministic(f"{name}_ppm", value * 1e6)
        result[name] = value

    if model == "eclipse":
        sample_flux("eclipse_depth")
    elif model == "phase_curve":
        day_centers = np.asarray(config["dayside_flux"], dtype=float)
        day_widths = np.asarray(
            config["dayside_flux_prior_width"], dtype=float
        )
        night_centers = np.asarray(config["nightside_flux"], dtype=float)
        night_widths = np.asarray(
            config["nightside_flux_prior_width"], dtype=float
        )
        day_values = []
        night_values = []

        def conditional_flux(name, index, conditioning, center, width):
            quantile = numpyro.sample(
                f"_{name}_quantile_{index}",
                dist.Uniform(0.0, 1.0).expand(shape),
            )
            value = _conditional_phase_flux_from_quantile(
                quantile, conditioning, center, width
            )
            value = numpyro.deterministic(f"_{name}_{index}", value)
            numpyro.factor(
                f"_{name}_physical_interval_mass_{index}",
                jnp.sum(
                    _positive_normal_interval_log_mass(
                        conditioning, center, width
                    )
                ),
            )
            return value

        for index in range(n_planets):
            day_free = day_widths[index] > 0.0
            night_free = night_widths[index] > 0.0
            if day_free and night_free:
                day = numpyro.sample(
                    f"_dayside_flux_{index}",
                    dist.TruncatedNormal(
                        day_centers[index], day_widths[index], low=0.0
                    ).expand(shape),
                )
                night = conditional_flux(
                    "nightside_flux",
                    index,
                    day,
                    night_centers[index],
                    night_widths[index],
                )
            elif day_free:
                night = jnp.full(
                    shape, night_centers[index], dtype=jnp.float64
                )
                if night_centers[index] == 0.0:
                    raise ValueError(
                        "a free dayside flux cannot be paired with a fixed zero "
                        "nightside flux under the physical phase-map constraint."
                    )
                day = conditional_flux(
                    "dayside_flux",
                    index,
                    night,
                    day_centers[index],
                    day_widths[index],
                )
            elif night_free:
                day = jnp.full(shape, day_centers[index], dtype=jnp.float64)
                if day_centers[index] == 0.0:
                    raise ValueError(
                        "a free nightside flux cannot be paired with a fixed zero "
                        "dayside flux under the physical phase-map constraint."
                    )
                night = conditional_flux(
                    "nightside_flux",
                    index,
                    day,
                    night_centers[index],
                    night_widths[index],
                )
            else:
                day = jnp.full(shape, day_centers[index], dtype=jnp.float64)
                night = jnp.full(
                    shape, night_centers[index], dtype=jnp.float64
                )
            day_values.append(day)
            night_values.append(night)

        axis = 0 if num_lcs is None else 1
        dayside = numpyro.deterministic(
            "dayside_flux", jnp.stack(day_values, axis=axis)
        )
        nightside = numpyro.deterministic(
            "nightside_flux", jnp.stack(night_values, axis=axis)
        )
        numpyro.deterministic("dayside_flux_ppm", dayside * 1e6)
        numpyro.deterministic("nightside_flux_ppm", nightside * 1e6)
        result["dayside_flux"] = dayside
        result["nightside_flux"] = nightside
        values = []
        specs = config["hotspot_offset_spec"]
        for index in range(n_planets):
            values.append(
                sample_parameter(specs[index], f"_hotspot_offset_{index}", shape)
            )
        axis = 0 if num_lcs is None else 1
        value = numpyro.deterministic(
            "hotspot_offset", jnp.stack(values, axis=axis)
        )
        numpyro.deterministic("hotspot_offset_deg", jnp.rad2deg(value))
        result["hotspot_offset"] = value

    spots = config.get("spots", ())
    if spots:
        contrasts = []
        for index, spot in enumerate(spots):
            center = float(spot["contrast"])
            width = float(spot.get("contrast_prior_width", 0.0))
            if width > 0.0:
                contrast = numpyro.sample(
                    f"_stellar_spot_contrast_{index}",
                    dist.TruncatedNormal(center, width, low=0.0, high=1.0).expand(shape),
                )
            else:
                contrast = jnp.full(shape, center, dtype=jnp.float64)
            contrasts.append(contrast)
        axis = 0 if num_lcs is None else 1
        result["stellar_spot_contrast"] = numpyro.deterministic(
            "stellar_spot_contrast", jnp.stack(contrasts, axis=axis)
        )
        result["stellar_rotation_period"] = jnp.asarray(
            config["stellar_rotation_period"], dtype=jnp.float64
        )
    return result


LD_MAP_COEFFICIENTS = 0
LD_MAP_POWER2_MAXTED = 1
LD_MAP_QUADRATIC_SUMDIFF = 2
LD_MAP_SING = 3


def _finite_log_density(value):
    """Turn support violations into the finite rejection value used by NUTS."""
    return jnp.where(jnp.isfinite(value), value, -1.0e100)


def _ld_variant_data_form(latent, center, scale, low, high, map_code, law,
                          latent_low=None, latent_high=None):
    """Return physical LD coefficients and the exact data-selected log prior.

    ``scale == 0`` denotes fixed coefficients, ``scale < 0`` a uniform
    distribution, and ``scale > 0`` a truncated-normal distribution.  The
    latent always has a standard-normal base density; ``log_correction``
    replaces it with the requested coordinate density.  For fixed LD the
    latent remains an independent standard-normal nuisance variable.
    """
    latent = jnp.asarray(latent, dtype=jnp.float64)
    center = jnp.broadcast_to(jnp.asarray(center, dtype=latent.dtype), latent.shape)
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=latent.dtype), latent.shape)
    low = jnp.broadcast_to(jnp.asarray(low, dtype=latent.dtype), latent.shape)
    high = jnp.broadcast_to(jnp.asarray(high, dtype=latent.dtype), latent.shape)
    code = jnp.asarray(map_code, dtype=jnp.int32)
    base_logp = jnp.sum(dist.Normal(0.0, 1.0).log_prob(latent), axis=-1)
    bounded = latent_low is not None and latent_high is not None
    if bounded:
        latent_low = jnp.broadcast_to(
            jnp.asarray(latent_low, dtype=latent.dtype), latent.shape
        )
        latent_high = jnp.broadcast_to(
            jnp.asarray(latent_high, dtype=latent.dtype), latent.shape
        )
        unit = jax.nn.sigmoid(latent)
        coordinate = latent_low + (latent_high - latent_low) * unit
        coordinate_log_jacobian = jnp.sum(
            jnp.log(latent_high - latent_low)
            + jax.nn.log_sigmoid(latent)
            + jax.nn.log_sigmoid(-latent), axis=-1,
        )
    else:
        coordinate = latent
        coordinate_log_jacobian = jnp.zeros_like(base_logp)

    def coordinate_density(x, lo, hi, loc, scl):
        uniform_logp = jnp.sum(dist.Uniform(lo, hi).log_prob(x), axis=-1)
        safe_scale = jnp.where(scl > 0.0, scl, 1.0)
        normal_logp = jnp.sum(
            dist.TruncatedNormal(loc, safe_scale, low=lo, high=hi).log_prob(x),
            axis=-1,
        )
        fixed = jnp.all(scl == 0.0, axis=-1)
        uniform = jnp.all(scl < 0.0, axis=-1)
        target = jnp.where(uniform, uniform_logp, normal_logp)
        return jnp.where(fixed, base_logp, _finite_log_density(target))

    def coefficients(_):
        fixed = jnp.all(scale == 0.0, axis=-1)
        physical = jnp.where(fixed[..., None], center, coordinate)
        target = coordinate_density(coordinate, low, high, center, scale)
        return physical, jnp.where(fixed, base_logp, target + coordinate_log_jacobian)

    def maxted(_):
        transform = Power2MaxtedTransform()
        if bounded:
            # Bound the unconstrained sampler through the physical coefficient
            # box, then map to h. Pulling the standalone h density back through
            # h(c) cancels its change-of-variables Jacobian exactly.
            physical = coordinate
            h = transform(physical)
            physical_logp = coordinate_density(physical, low, high, center, scale)
            h_logp = physical_logp - transform.log_abs_det_jacobian(physical, h)
            pulled_back = h_logp + transform.log_abs_det_jacobian(physical, h)
            return physical, _finite_log_density(
                pulled_back + coordinate_log_jacobian
            )
        physical = transform.inv(coordinate)
        physical_logp = coordinate_density(physical, low, high, center, scale)
        jacobian = transform.log_abs_det_jacobian(physical, coordinate)
        return physical, _finite_log_density(physical_logp - jacobian)

    def sumdiff(_):
        physical = jnp.stack(
            ((coordinate[..., 0] + coordinate[..., 1]) / 2.0,
             (coordinate[..., 0] - coordinate[..., 1]) / 2.0), axis=-1,
        )
        return physical, coordinate_density(
            coordinate, low, high, center, scale
        ) + coordinate_log_jacobian

    def sing(_):
        limb_l = coordinate[..., 0]
        u_plus = 1.0 - limb_l
        if bounded:
            limb_delta = (2.0 * coordinate[..., 1] - 1.0) * u_plus / 4.0
            conditional_log_jacobian = jnp.log(u_plus / 2.0)
        else:
            limb_delta = coordinate[..., 1]
            conditional_log_jacobian = jnp.zeros_like(u_plus)
        physical = jnp.stack(
            (u_plus - 4.0 * limb_delta, 4.0 * limb_delta), axis=-1
        )
        l_logp = dist.TruncatedNormal(
            center[..., 0], jnp.maximum(scale[..., 0], 1e-12),
            low=low[..., 0], high=high[..., 0],
        ).log_prob(limb_l)
        delta_logp = dist.TruncatedNormal(
            center[..., 1], jnp.maximum(scale[..., 1], 1e-12),
            low=-u_plus / 4.0, high=u_plus / 4.0,
        ).log_prob(limb_delta)
        return physical, _finite_log_density(
            l_logp + delta_logp + conditional_log_jacobian
            + coordinate_log_jacobian
        )

    if law == "power2":
        branches = (coefficients, maxted, coefficients, coefficients)
    elif law == "quadratic":
        branches = (coefficients, coefficients, sumdiff, sing)
    else:
        raise ValueError(f"Unsupported LD law for data form: {law}")
    physical, target_logp = jax.lax.switch(code, branches, operand=None)
    return physical, target_logp - base_logp


def _resolve_builder_kernel(jaxoplanet_kernel, ld_profile, param_method, *, ld_mode=None):
    """Validate the retained model-level kernel choices and resolve the route."""
    del ld_mode
    requested = str(jaxoplanet_kernel).lower()
    degree = 12 if ld_profile == "power2" else 2 if ld_profile == "quadratic" else None
    return resolve_jaxoplanet_kernel(
        requested,
        ld_profile=ld_profile,
        degree=degree,
        keplerian=param_method == "a_rs",
    )


def derive_geometry(wl_samples, period, ecc=0.0, omega=0.0):
    """Derive a consistent geometry bundle from any supported jaxoplanet WL parameterization.

    ``period`` is the fixed period per planet; when the white-light fit
    sampled ``period_{i}`` the posterior draws are used instead.
    """
    period = jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64))
    ecc = jnp.atleast_1d(jnp.asarray(ecc, dtype=jnp.float64))
    omega = jnp.atleast_1d(jnp.asarray(omega, dtype=jnp.float64))
    n_planets = period.shape[0]
    period = [
        wl_samples[f"period_{i}"] if f"period_{i}" in wl_samples else period[i]
        for i in range(n_planets)
    ]

    result = {}
    for i in range(n_planets):
        if f"a_rs_{i}" in wl_samples:
            a_rs_samples = wl_samples[f"a_rs_{i}"]
        elif f"log_a_rs_{i}" in wl_samples:
            a_rs_samples = jnp.exp(wl_samples[f"log_a_rs_{i}"])
        else:
            a_rs_samples = None

        if f"b_{i}" in wl_samples:
            b_samples = wl_samples[f"b_{i}"]
        elif f"_b_{i}" in wl_samples:
            b_samples = jnp.abs(wl_samples[f"_b_{i}"])
        else:
            b_samples = None

        if f"cos_i_{i}" in wl_samples:
            cos_i_samples = wl_samples[f"cos_i_{i}"]
            inc_samples = jnp.arccos(jnp.clip(cos_i_samples, 0.0, 1.0))
        elif f"inc_{i}" in wl_samples:
            inc_samples = wl_samples[f"inc_{i}"]
            cos_i_samples = jnp.cos(inc_samples)
        else:
            cos_i_samples = None
            inc_samples = None

        if f"duration_{i}" in wl_samples:
            dur_samples = wl_samples[f"duration_{i}"]
        elif f"logD_{i}" in wl_samples:
            dur_samples = jnp.exp(wl_samples[f"logD_{i}"])
        else:
            dur_samples = None

        rors_samples = wl_samples[f"rors_{i}"]

        if a_rs_samples is None and dur_samples is not None and b_samples is not None:
            a_rs_samples = harmonica_a_rs_from_duration(
                period[i], dur_samples, b_samples, rors_samples,
                ecc=ecc[i], omega=omega[i],
            )
        if b_samples is None and cos_i_samples is not None and a_rs_samples is not None:
            b_samples = harmonica_impact_param_from_cos_i(
                cos_i_samples, a_rs_samples, ecc=ecc[i], omega=omega[i],
            )
        if cos_i_samples is None and b_samples is not None and a_rs_samples is not None:
            cos_i_samples = harmonica_cos_i_from_geometry(
                b_samples, a_rs_samples, ecc=ecc[i], omega=omega[i],
            )
            inc_samples = jnp.arccos(jnp.clip(cos_i_samples, 0.0, 1.0))
        if dur_samples is None and a_rs_samples is not None and b_samples is not None:
            dur_samples = harmonica_duration_from_geometry(
                period[i], a_rs_samples, b_samples, rors_samples,
                ecc=ecc[i], omega=omega[i],
            )

        if a_rs_samples is None or b_samples is None or cos_i_samples is None or dur_samples is None:
            continue

        result[f'duration_{i}'] = dur_samples
        result[f'b_{i}'] = b_samples
        result[f'a_rs_{i}'] = a_rs_samples
        result[f'cos_i_{i}'] = cos_i_samples
        result[f'inc_{i}'] = inc_samples

    return result


_WHITELIGHT_PARAMETER_PRIOR_KEYS = (
    'period', 't0', 'eclipse_time', 'b', 'rprs', 'duration', 'a_rs', 'ecc', 'omega',
)


from models.priors import spec_distribution as _spec_distribution, sample_parameter


def _validate_parameter_priors(parameter_priors, n_planets, param_method):
    """Check the per-planet parameter specifications at factory time."""
    if not parameter_priors:
        raise ValueError(
            "parameter_priors is required: every planet parameter must carry "
            "a {value, prior, ...} specification (see koala.config)."
        )
    resolved = {}
    for name, specs in parameter_priors.items():
        if name not in _WHITELIGHT_PARAMETER_PRIOR_KEYS:
            raise ValueError(
                f"parameter_priors has no white-light site for planet.{name}."
            )
        specs = tuple(specs)
        if len(specs) != n_planets:
            raise ValueError(
                f"planet.{name}: expected {n_planets} specifications, "
                f"received {len(specs)}."
            )
        for spec in specs:
            if name in {'ecc', 'omega'} and spec.free:
                raise ValueError(f"planet.{name} may only be fixed for now.")
            if spec.free and spec.prior != 'log_uniform':
                _spec_distribution(spec)
        unused = (
            (name == 'duration' and param_method == 'a_rs')
            or (name == 'a_rs' and param_method == 'duration')
        )
        if unused and any(spec.free for spec in specs):
            other = 'a_rs' if name == 'duration' else 'duration'
            raise ValueError(
                f"planet.{name} is free but the white-light model is "
                f"parameterized by {other}; give planet.{other} the prior instead."
            )
        resolved[name] = specs
    for name in ('period', 't0', 'b', 'rprs',
                 'duration' if param_method == 'duration' else 'a_rs'):
        if name not in resolved:
            raise ValueError(f"parameter_priors is missing planet.{name}.")
    return resolved


def create_whitelight_model(detrend_type='linear', n_planets=1, ld_profile='quadratic',
                            ld_mode='gaussian', param_method='duration',
                            jaxoplanet_kernel='auto',
                            surface_config=None,
                            surface_basis=None,
                            ld_parameterization='coefficients',
                            ld_uniform_basis='uplus_uminus',
                            ld_uniform_coefficient_bounds=(0.0, 1.0),
                            step_width_mode='free',
                            trend_parameterization='physical',
                            two_spot_ordering='legacy',
                            ld_variant_as_data=False,
                            gp_solver=None,
                            gp_assume_sorted=False,
                            parameter_priors=None):
    """Jaxoplanet white-light model with duration- or a_rs-based geometry.

    ``parameter_priors`` maps every ``planet`` key (``period``, ``t0``,
    ``b``, ``rprs``, ``duration`` or ``a_rs``, ``ecc``, ``omega``) to one
    ``ParameterSpec`` per planet (see :mod:`koala.config`). A fixed
    specification becomes a deterministic site; a free one is sampled from
    its prior by :func:`models.priors.sample_parameter`.
    """
    if param_method not in ('duration', 'a_rs'):
        raise ValueError(f"Unknown param_method: {param_method}")
    parameter_priors = _validate_parameter_priors(
        parameter_priors, n_planets, param_method
    )
    surface_config = surface_config or {"model": "transit", "spots": ()}
    if (surface_config.get("model") != "transit" or surface_config.get("spots")) and param_method != "a_rs":
        raise ValueError("Eclipses, phase curves, and stellar spots require param_method='a_rs'.")
    basis_surface_valid = (
        (surface_config.get("model") == "transit" and surface_config.get("spots"))
        or (
            surface_config.get("model") in {"eclipse", "phase_curve"}
            and not surface_config.get("spots")
        )
    )
    if surface_basis is not None and not (
        basis_surface_valid
        and not surface_config.get("fit_geometry", True)
        and ld_mode == "fixed"
    ):
        raise ValueError(
            "surface_basis requires a supported stellar-spot or emission model "
            "with fixed geometry and fixed limb darkening."
        )
    if ld_parameterization not in {'coefficients', 'decorrelated'}:
        raise ValueError(
            "ld_parameterization must be 'coefficients' or 'decorrelated'."
        )
    if ld_uniform_basis not in {'uplus_uminus', 'coefficients'}:
        raise ValueError("ld_uniform_basis must be 'uplus_uminus' or 'coefficients'.")
    uniform_coefficient_low, uniform_coefficient_high = map(
        float, ld_uniform_coefficient_bounds
    )
    if uniform_coefficient_low >= uniform_coefficient_high:
        raise ValueError("ld_uniform_coefficient_bounds must have low < high.")
    configured_trend_parameterization = trend_parameterization
    two_spot_ordering = str(two_spot_ordering).strip().lower()
    if two_spot_ordering != 'legacy':
        raise ValueError("two_spot_ordering must be 'legacy'.")
    power2_transform = Power2MaxtedTransform()
    _resolve_builder_kernel(
        jaxoplanet_kernel, ld_profile, param_method, ld_mode=ld_mode
    )
    detrend_components = _split_components(detrend_type)
    # Resolve once at factory time, and only when a GP is actually built, so
    # non-GP trends never touch the solver policy (or its install check).
    resolved_gp_solver = resolve_gp_solver(gp_solver) if 'gp' in detrend_type else None
    if ld_profile == "power2":
        MUS, P = _prepare_power2_poly()

    print(f"Building jaxoplanet whitelight model: detrend='{detrend_type}', "
          f"ld='{ld_mode}', ld_profile='{ld_profile}' for {n_planets} planets")

    def _whitelight_model(t, yerr, y=None, prior_params=None,
                          ld_center=None, ld_scale=None, ld_low=None,
                          ld_high=None, ld_map_code=0, ld_latent_low=None,
                          ld_latent_high=None):
        trend_parameterization = resolve_whitelight_trend_parameterization(
            configured_trend_parameterization
        )
        cadence = jnp.median(jnp.diff(jnp.sort(jnp.asarray(t))))

        durations, t0s, bs, rorss, a_rss, cos_is, incs = [], [], [], [], [], [], []
        periods = []
        eccs = jnp.asarray([spec.value for spec in parameter_priors['ecc']], dtype=jnp.float64)
        omegas = jnp.asarray([spec.value for spec in parameter_priors['omega']], dtype=jnp.float64)
        fit_geometry = bool(surface_config.get("fit_geometry", True))
        if not fit_geometry:
            numpyro.deterministic("_geometry_fixed", jnp.asarray(True))

        def _site(name, i):
            return sample_parameter(parameter_priors[name][i], f"{name}_{i}")

        for i in range(n_planets):
            period_i = _site("period", i)
            periods.append(period_i)
            t0_i = _site("t0", i)
            t0s.append(t0_i)
            if 'eclipse_time' in parameter_priors:
                numpyro.deterministic(f"eclipse_time_{i}", t0_i + 0.5 * period_i)
            rors_i = sample_parameter(parameter_priors["rprs"][i], f"rors_{i}")
            numpyro.deterministic(f"depths_{i}", rors_i ** 2)
            rorss.append(rors_i)
            b_i = _site("b", i)
            bs.append(b_i)

            if param_method == 'duration':
                duration_i = _site("duration", i)
                a_rs_i = numpyro.deterministic(
                    f"a_rs_{i}",
                    harmonica_a_rs_from_duration(
                        period_i, duration_i, b_i, rors_i,
                        ecc=eccs[i], omega=omegas[i],
                    ),
                )
            else:
                a_rs_i = _site("a_rs", i)
                duration_i = numpyro.deterministic(
                    f"duration_{i}",
                    harmonica_duration_from_geometry(
                        period_i, a_rs_i, b_i, rors_i,
                        ecc=eccs[i], omega=omegas[i],
                    ),
                )

            cos_i_i = numpyro.deterministic(
                f"cos_i_{i}",
                harmonica_cos_i_from_geometry(
                    b_i, a_rs_i, ecc=eccs[i], omega=omegas[i],
                ),
            )
            inc_i = numpyro.deterministic(
                f"inc_{i}", jnp.arccos(jnp.clip(cos_i_i, 0.0, 1.0))
            )

            durations.append(duration_i)
            a_rss.append(a_rs_i)
            cos_is.append(cos_i_i)
            incs.append(inc_i)

        if ld_variant_as_data:
            if any(value is None for value in (ld_center, ld_scale, ld_low, ld_high)):
                raise ValueError("LD data form requires center, scale, low, and high arrays.")
            ld_latent = numpyro.sample(
                'ld_variant_latent', dist.Normal(0.0, 1.0).expand([2]).to_event(1)
            )
            coefficients, correction = _ld_variant_data_form(
                ld_latent, ld_center, ld_scale, ld_low, ld_high,
                ld_map_code, ld_profile, ld_latent_low, ld_latent_high,
            )
            numpyro.factor('ld_variant_prior', correction)
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', coefficients)
            else:
                c1 = numpyro.deterministic('c1', coefficients[0])
                c2 = numpyro.deterministic('c2', coefficients[1])
                prof = get_I_power2(c1, c2, MUS)
                u = P @ (1.0 - prof)
        elif ld_profile == 'quadratic':
            if ld_mode in {'gaussian', 'stellarprior'}:
                u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
                if ld_mode == 'stellarprior':
                    if 'u_sigma' not in prior_params:
                        raise ValueError("ld_mode='stellarprior' requires prior_params['u_sigma'].")
                    u_sigma = jnp.asarray(prior_params['u_sigma'], dtype=jnp.float64)
                else:
                    u_sigma = jnp.full_like(u_prior, GAUSSIAN_LD_WIDTH)
                u_sigma = jnp.broadcast_to(u_sigma, u_prior.shape)
                u_sigma = jnp.clip(u_sigma, 1e-6, None)
                coefficient_prior = dist.TruncatedNormal(
                    loc=u_prior, scale=u_sigma, low=0.0, high=1.0
                ).to_event(1)
                u = numpyro.sample("u", coefficient_prior)
            elif ld_mode == 'uniform':
                if ld_uniform_basis == 'uplus_uminus':
                    ld_sumdiff = numpyro.sample(
                        'ld_uplus_uminus',
                        dist.Uniform(jnp.array([-1.0, -2.0]),
                                     jnp.array([2.0, 2.0])).to_event(1),
                    )
                    u1 = numpyro.deterministic('u1', 0.5 * (ld_sumdiff[0] + ld_sumdiff[1]))
                    u2 = numpyro.deterministic('u2', 0.5 * (ld_sumdiff[0] - ld_sumdiff[1]))
                    numpyro.deterministic('l', 1.0 - ld_sumdiff[0])
                    numpyro.deterministic('delta', (ld_sumdiff[0] - ld_sumdiff[1]) / 8.0)
                    u = numpyro.deterministic('u', jnp.stack((u1, u2)))
                else:
                    coefficient_prior = dist.Uniform(
                        uniform_coefficient_low, uniform_coefficient_high
                    ).expand([2]).to_event(1)
                    u = numpyro.sample("u", coefficient_prior)
            elif ld_mode == 'fixed':
                u = numpyro.deterministic("u", jnp.asarray(prior_params['u'], dtype=jnp.float64))
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
        elif ld_profile == 'power2':
            u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
            if ld_mode in {'gaussian', 'stellarprior'}:
                if ld_mode == 'stellarprior':
                    if 'u_sigma' not in prior_params:
                        raise ValueError("ld_mode='stellarprior' requires prior_params['u_sigma'].")
                    u_sigma = jnp.asarray(prior_params['u_sigma'], dtype=jnp.float64)
                else:
                    u_sigma = jnp.full_like(u_prior, GAUSSIAN_LD_WIDTH)
                u_sigma = jnp.broadcast_to(u_sigma, u_prior.shape)
                u_sigma = jnp.clip(u_sigma, 1e-6, None)
                coefficient_prior = dist.TruncatedNormal(
                    u_prior, u_sigma, low=jnp.asarray([0.0, 0.001]), high=1.0
                ).to_event(1)
                if ld_parameterization == 'decorrelated' and ld_mode == 'gaussian':
                    h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, power2_transform))
                    coefficients = _enforce_decorrelated_coefficient_support(
                        power2_transform.inv(h),
                        jnp.asarray([0.0, 0.001]), 1.0
                    )
                    c1 = numpyro.deterministic('c1', coefficients[0])
                    c2 = numpyro.deterministic('c2', coefficients[1])
                else:
                    c1 = numpyro.sample('c1', dist.TruncatedNormal(
                        u_prior[0], u_sigma[0], low=0.0, high=1.0))
                    c2 = numpyro.sample('c2', dist.TruncatedNormal(
                        u_prior[1], u_sigma[1], low=0.001, high=1.0))
            elif ld_mode == 'uniform':
                coefficient_prior = dist.Uniform(0.0, 1.0).expand([2]).to_event(1)
                if ld_parameterization == 'decorrelated':
                    h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, power2_transform))
                    coefficients = _enforce_decorrelated_coefficient_support(
                        power2_transform.inv(h), 0.0, 1.0
                    )
                    c1 = numpyro.deterministic('c1', coefficients[0])
                    c2 = numpyro.deterministic('c2', coefficients[1])
                else:
                    c1 = numpyro.sample('c1', dist.Uniform(0.0, 1.0))
                    c2 = numpyro.sample('c2', dist.Uniform(0.0, 1.0))
            elif ld_mode == 'fixed':
                c1 = numpyro.deterministic('c1', u_prior[0])
                c2 = numpyro.deterministic('c2', u_prior[1])
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
            prof = get_I_power2(c1, c2, MUS)
            u = P @ (1.0 - prof)
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")

        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter) ** 2 + yerr ** 2))

        params = {
            "period": jnp.array(periods),
            "duration": jnp.array(durations),
            "t0": jnp.array(t0s),
            "b": jnp.array(bs),
            "rors": jnp.array(rorss),
            "_jaxoplanet_kernel": jaxoplanet_kernel,
            "_ld_profile": ld_profile,
        }
        params["u"] = u
        if param_method == 'a_rs':
            params["a_rs"] = jnp.array(a_rss)
            params["ecc"] = eccs
            params["omega"] = omegas
        params.update(_sample_surface_parameters(surface_config, n_planets))
        if surface_basis is not None:
            params["_surface_basis"] = surface_basis

        has_offset_term = not detrend_components.isdisjoint(
            {'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', '2spot', 'gp'}
        )
        if has_offset_term:
            params['c'] = numpyro.sample('c', dist.Uniform(0.9, 1.1))

        has_linear = not detrend_components.isdisjoint(
            {'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', '2spot'}
        )
        if has_linear:
            params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1))

        if not detrend_components.isdisjoint({'quadratic', 'cubic', 'quartic'}):
            params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1))
        if not detrend_components.isdisjoint({'cubic', 'quartic'}):
            params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1))
        if 'quartic' in detrend_components:
            params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1))

        if 'linear_discontinuity' in detrend_components:
            t_jump_guess = prior_params.get('t_jump_guess', 0.5 * (jnp.min(t) + jnp.max(t)))
            jump_guess = prior_params.get('jump_guess', 0.0)
            if trend_parameterization == 'cadence':
                t_jump_offset_cadences = numpyro.sample(
                    't_jump_offset_cadences', dist.Normal(0.0, 1e-2 / cadence)
                )
                params['t_jump'] = numpyro.deterministic(
                    't_jump', t_jump_guess + cadence * t_jump_offset_cadences
                )
            else:
                params['t_jump'] = numpyro.sample(
                    't_jump', dist.Normal(t_jump_guess, 1e-2)
                )
            params['jump'] = numpyro.sample('jump', dist.Normal(jump_guess, 0.01))
            params['width'] = sample_step_width(
                t, prior_params, step_width_mode, trend_parameterization
            )

        if 'explinear' in detrend_components:
            params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)))
            params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))

        if not detrend_components.isdisjoint({'spot', '2spot'}):
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Uniform(0.0, 0.1))
            if trend_parameterization == 'cadence':
                spot_mu_offset_cadences = numpyro.sample(
                    'spot_mu_offset_cadences', dist.Normal(0.0, 0.01 / cadence)
                )
                params['spot_mu'] = numpyro.deterministic(
                    'spot_mu',
                    prior_params['spot_guess'] + cadence * spot_mu_offset_cadences,
                )
                spot_sigma_cadences = numpyro.sample(
                    'spot_sigma_cadences',
                    dist.Uniform(1e-4 / cadence, 0.1 / cadence),
                )
                params['spot_sigma'] = numpyro.deterministic(
                    'spot_sigma', cadence * spot_sigma_cadences
                )
            else:
                params['spot_mu'] = numpyro.sample(
                    'spot_mu', dist.Normal(prior_params['spot_guess'], 0.01)
                )
                params['spot_sigma'] = numpyro.sample(
                    'spot_sigma', dist.Uniform(1e-4, 0.1)
                )
        if '2spot' in detrend_components:
            spot_guess2 = prior_params.get('spot_guess2', prior_params['spot_guess'])
            params['spot_amp2'] = numpyro.sample('spot_amp2', dist.Uniform(0.0, 0.1))
            if trend_parameterization == 'cadence':
                spot_sigma2_cadences = numpyro.sample(
                    'spot_sigma2_cadences',
                    dist.Uniform(1e-4 / cadence, 0.1 / cadence),
                )
                spot_mu2_offset_cadences = numpyro.sample(
                    'spot_mu2_offset_cadences', dist.Normal(0.0, 0.01 / cadence)
                )
                params['spot_mu2'] = numpyro.deterministic(
                    'spot_mu2', spot_guess2 + cadence * spot_mu2_offset_cadences
                )
                params['spot_sigma2'] = numpyro.deterministic(
                    'spot_sigma2', cadence * spot_sigma2_cadences
                )
            else:
                params['spot_mu2'] = numpyro.sample(
                    'spot_mu2', dist.Normal(spot_guess2, 0.01)
                )
                params['spot_sigma2'] = numpyro.sample(
                    'spot_sigma2', dist.Uniform(1e-4, 0.1)
                )
        if 'gp' in detrend_components:
            params['GP_log_sigma'] = numpyro.sample('GP_log_sigma', dist.Uniform(*GP_HYPERPARAMETER_BOUNDS['GP_log_sigma']))
            params['GP_log_rho'] = numpyro.sample('GP_log_rho', dist.Uniform(*GP_HYPERPARAMETER_BOUNDS['GP_log_rho']))

        if 'gp' in detrend_type:
            gp_builder = resolve_gp_builder(detrend_type)
            gp = gp_builder(
                params,
                t,
                error,
                gp_solver=resolved_gp_solver,
                assume_sorted=gp_assume_sorted,
            )
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            try:
                lc_model = resolve_detrend_kernel(detrend_type)(params, t)
                numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
            except KeyError:
                raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model


def create_vectorized_model(detrend_type='linear', ld_mode='gaussian', trend_mode='free',
                            n_planets=1, ld_profile='quadratic',
                            param_method='duration', transit_window='off',
                            transit_window_indices=None,
                            cadence_reduction='off',
                            transit_grid='off',
                            transit_grid_nodes=769,
                            transit_grid_non_grazing=False,
                            transit_grid_outer_contact_safe=None,
                            jaxoplanet_kernel='auto',
                            surface_config=None,
                            surface_basis=None,
                            jitter_prior='lognormal',
                            jitter_prior_scale=2.0,
                            jitter_prior_center=0.5,
                            ld_parameterization='coefficients',
                            ld_uniform_basis='uplus_uminus',
                            ld_uniform_coefficient_bounds=(0.0, 1.0),
                            ld_variant_as_data=False,
                            joint_geometry=False):
    """Jaxoplanet spectroscopic model with WL-fixed duration- or a_rs-based geometry.

    ``joint_geometry=True`` replaces the fixed white-light geometry with
    shared sites ``t0``, ``b``, and ``duration`` (or ``a_rs``) sampled once
    per draw for every channel (``in_axes=None``).  Each shared site takes a
    ``Normal(mu_<name>, sigma_<name>)`` prior from the model keyword
    arguments; the period stays fixed.  The static cadence-reduction and
    transit-grid accelerations assume fixed geometry and are disabled.

    ``jitter_prior='lognormal'`` uses
    ``log_jitter ~ Normal(log(jitter_prior_center * median(yerr)),
      jitter_prior_scale)``.  The posterior is likelihood-dominated whenever the
      jitter is identified (posterior widths ~0.1-0.4 in log versus a prior
      width of 2.0 e-folds) and reduces to a smooth Gaussian tail instead of a
      plateau when it is not.  The site name, shape, and downstream
      ``total_error`` deterministic are unchanged.
    """
    if param_method not in {'duration', 'a_rs'}:
        raise ValueError(f"Unknown param_method: {param_method}")
    surface_config = surface_config or {"model": "transit", "spots": ()}
    if (surface_config.get("model") != "transit" or surface_config.get("spots")) and param_method != "a_rs":
        raise ValueError("Eclipses, phase curves, and stellar spots require param_method='a_rs'.")
    basis_surface_valid = (
        (surface_config.get("model") == "transit" and surface_config.get("spots"))
        or (
            surface_config.get("model") in {"eclipse", "phase_curve"}
            and not surface_config.get("spots")
        )
    )
    if surface_basis is not None and not (
        basis_surface_valid
        and not surface_config.get("fit_geometry", True)
        and ld_mode == "fixed"
    ):
        raise ValueError(
            "surface_basis requires a supported stellar-spot or emission model "
            "with fixed geometry and fixed limb darkening."
        )
    if ld_parameterization not in {'coefficients', 'decorrelated'}:
        raise ValueError(
            "ld_parameterization must be 'coefficients' or 'decorrelated'."
        )
    if ld_uniform_basis not in {'uplus_uminus', 'coefficients'}:
        raise ValueError("ld_uniform_basis must be 'uplus_uminus' or 'coefficients'.")
    uniform_coefficient_low, uniform_coefficient_high = map(
        float, ld_uniform_coefficient_bounds
    )
    if uniform_coefficient_low >= uniform_coefficient_high:
        raise ValueError("ld_uniform_coefficient_bounds must have low < high.")
    power2_transform = Power2MaxtedTransform()
    joint_geometry = bool(joint_geometry)
    if joint_geometry and not surface_config.get("fit_geometry", True):
        raise ValueError(
            "joint_geometry=True samples the transit geometry and therefore "
            "requires fit_geometry=True."
        )
    if jitter_prior != 'lognormal':
        raise ValueError("jitter_prior must be 'lognormal'.")
    jitter_prior_scale = float(jitter_prior_scale)
    jitter_prior_center = float(jitter_prior_center)
    if jitter_prior_scale <= 0.0 or jitter_prior_center <= 0.0:
        raise ValueError("jitter_prior_scale and jitter_prior_center must be > 0.")
    if trend_mode not in {'free', 'fixed', 'gaussian_marginalized'}:
        raise ValueError(f"Unknown trend_mode: {trend_mode}")
    _resolve_builder_kernel(
        jaxoplanet_kernel, ld_profile, param_method, ld_mode=ld_mode
    )
    if transit_window not in {'auto', 'off'}:
        raise ValueError("transit_window must be either 'auto' or 'off'.")
    cadence_reduction = str(cadence_reduction).lower()
    if cadence_reduction not in {'auto', 'off'}:
        raise ValueError("cadence_reduction must be either 'auto' or 'off'.")
    transit_grid = str(transit_grid).lower()
    if transit_grid not in {'auto', 'off'}:
        raise ValueError("transit_grid must be either 'auto' or 'off'.")
    transit_grid_nodes = int(transit_grid_nodes)
    if transit_grid_nodes < 13:
        raise ValueError("transit_grid_nodes must be at least 13.")
    # The handoff flag is conservative over the full radius-ratio prior.  A
    # grazing or near-grazing box needs the denser audited default; an explicit
    # larger setting is preserved.
    transit_grid_non_grazing = bool(transit_grid_non_grazing)
    if transit_grid_outer_contact_safe is None:
        # Backward compatibility for pre-widening stage dumps: the old
        # conservative non-grazing proof also guarantees ample outer-contact
        # separation over the radius prior.
        transit_grid_outer_contact_safe = transit_grid_non_grazing
    transit_grid_outer_contact_safe = bool(transit_grid_outer_contact_safe)
    if not transit_grid_non_grazing and ld_profile == 'power2':
        transit_grid_nodes = max(transit_grid_nodes, 2049)
    use_transit_window = (
        transit_window == 'auto'
        and transit_window_indices is not None
        and param_method == 'duration'
    )
    if transit_window_indices is not None:
        transit_window_indices = jnp.asarray(
            transit_window_indices, dtype=jnp.int32
        )
        if transit_window_indices.ndim != 1:
            raise ValueError("transit_window_indices must be one-dimensional.")
    detrend_components = _split_components(detrend_type)
    linear_trend_names = linear_spectro_trend_coefficient_names(detrend_type)
    cadence_reduction_eligible = (
        cadence_reduction == 'auto'
        and use_transit_window
        and n_planets == 1
        and surface_config.get("model") == "transit"
        and not surface_config.get("spots")
        and trend_mode == 'free'
        and linear_trend_names is not None
        and not joint_geometry
    )
    transit_grid_eligible = (
        transit_grid == 'auto'
        and not joint_geometry
        and use_transit_window
        and n_planets == 1
        and param_method == 'duration'
        and surface_config.get("model") == "transit"
        and not surface_config.get("spots")
        and ld_profile in {'power2', 'quadratic'}
        and ld_mode != 'interpolated'
        and transit_grid_outer_contact_safe
    )
    unsupported_non_spectroscopic = {'spot', '2spot', 'linear_discontinuity'}
    if detrend_components & unsupported_non_spectroscopic:
        unsupported = ", ".join(sorted(detrend_components & unsupported_non_spectroscopic))
        raise ValueError(
            "Vectorized model does not support non-spectroscopic spot/jump trends "
            f"({unsupported}). Use the corresponding *_spectroscopic trend family instead."
        )
    compute_lc_kernel = resolve_detrend_kernel(detrend_type)
    if ld_profile == "power2":
        MUS_LD, P_LD = _prepare_power2_poly()

    print(f"Building jaxoplanet vectorized model: detrend='{detrend_type}', "
          f"ld='{ld_mode}', ld_profile='{ld_profile}' for {n_planets} planets"
          + (", shared (joint) geometry sites" if joint_geometry else ""))

    def _vectorized_model(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                          mu_depths=None, PERIOD=None, trend_fixed=None,
                          mu_a_rs=None, mu_ecc=0.0, mu_omega=0.0,
                          ld_interpolated=None, ld_fixed=None,
                          mu_u_ld=None, sigma_u_ld=None, gp_trend=None, spot_trend=None,
                          spot_trend2=None, jump_trend=None,
                          exp_trend=None, fixed_tau=None,
                          trend_design=None,
                          precomputed_yerr_per_lc=None,
                          trend_prior_mean=None, trend_prior_scale=None,
                          likelihood_mask=None, ld_center=None, ld_scale=None,
                          ld_low=None, ld_high=None, ld_map_code=0,
                          ld_latent_low=None, ld_latent_high=None,
                          surface_basis_data=None,
                          oot_reference_beta=None,
                          oot_group_yerr=None,
                          oot_group_count=None,
                          oot_group_reference_sse=None,
                          oot_group_x_reference_residual=None,
                          oot_group_xx=None,
                          sigma_t0=None, sigma_b=None,
                          sigma_duration=None, sigma_a_rs=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]
        use_cadence_reduction = (
            cadence_reduction_eligible
            and int(jnp.shape(t)[0]) > _AUTO_CADENCE_REDUCTION_THRESHOLD
        )
        use_transit_grid = (
            transit_grid_eligible
            and int(jnp.shape(t)[0]) > _AUTO_CADENCE_REDUCTION_THRESHOLD
        )
        def _shared_geometry_site(name, center, scale):
            # One draw per planet, shared by every channel in the fit.
            if center is None or scale is None:
                raise ValueError(
                    f"joint_geometry=True requires mu_{name} and sigma_{name}."
                )
            center = jnp.atleast_1d(jnp.asarray(center, dtype=jnp.float64))
            scale = jnp.broadcast_to(
                jnp.atleast_1d(jnp.asarray(scale, dtype=jnp.float64)),
                center.shape,
            )
            return numpyro.sample(name, dist.Normal(center, scale))

        if joint_geometry:
            t0s = _shared_geometry_site("t0", mu_t0, sigma_t0)
            bs = _shared_geometry_site("b", mu_b, sigma_b)
        else:
            t0s = mu_t0
            bs = mu_b

        if param_method == 'duration':
            duration = (
                _shared_geometry_site("duration", mu_duration, sigma_duration)
                if joint_geometry else mu_duration
            )
            orbital_params = {"duration": duration}
        else:
            if mu_a_rs is None:
                raise ValueError("mu_a_rs must be provided when param_method='a_rs'.")
            a_rs = (
                _shared_geometry_site("a_rs", mu_a_rs, sigma_a_rs)
                if joint_geometry
                else jnp.asarray(mu_a_rs, dtype=jnp.float64)
            )
            orbital_params = {
                "a_rs": a_rs,
                "ecc": jnp.asarray(mu_ecc, dtype=jnp.float64),
                "omega": jnp.asarray(mu_omega, dtype=jnp.float64),
            }

        if surface_config.get("fit_geometry", True):
            rors = numpyro.sample(
                "rors",
                dist.Uniform(jnp.sqrt(1e-5), jnp.sqrt(0.5)).expand([num_lcs, n_planets]),
            )
        else:
            if mu_depths is None:
                raise ValueError("mu_depths is required when fit_geometry is false.")
            # Preserve the channel axis on every vector-model output.  This is
            # required when selective failed lanes move through the joint-NUTS
            # fallback and are merged back into a chunk.
            numpyro.deterministic(
                "_geometry_fixed", jnp.ones((num_lcs,), dtype=jnp.bool_)
            )
            fixed_rors = jnp.sqrt(jnp.asarray(mu_depths, dtype=jnp.float64))
            if fixed_rors.ndim == 1:
                fixed_rors = jnp.broadcast_to(fixed_rors, (num_lcs, n_planets))
            rors = numpyro.deterministic("rors", fixed_rors)
        depths = numpyro.deterministic("depths", rors ** 2)

        yerr_matrix = jnp.atleast_2d(
            jnp.asarray(yerr, dtype=jnp.float64)
        )
        if yerr_matrix.shape[0] == 1 and num_lcs > 1:
            yerr_matrix = jnp.broadcast_to(
                yerr_matrix, (num_lcs, yerr_matrix.shape[1])
            )
        if precomputed_yerr_per_lc is None:
            yerr_per_lc = jnp.nanmedian(yerr_matrix, axis=1)
        else:
            yerr_per_lc = jnp.asarray(
                precomputed_yerr_per_lc, dtype=jnp.float64
            )
        log_jitter = numpyro.sample(
            'log_jitter',
            dist.Normal(
                jnp.log(jitter_prior_center * yerr_per_lc),
                jnp.full((num_lcs,), jitter_prior_scale, dtype=jnp.float64),
            ),
        )
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter ** 2 + yerr_per_lc ** 2))
        # ``total_error`` is a compact per-channel posterior summary.  The
        # likelihood itself must retain the reported time-dependent errors;
        # replacing them with their median changes the statistical weighting
        # whenever a JWST integration has a different uncertainty.
        error_broadcast = jnp.sqrt(
            jitter[:, None] ** 2 + yerr_matrix ** 2
        )

        if ld_variant_as_data:
            if any(value is None for value in (ld_center, ld_scale, ld_low, ld_high)):
                raise ValueError("LD data form requires center, scale, low, and high arrays.")
            ld_latent = numpyro.sample(
                'ld_variant_latent',
                dist.Normal(0.0, 1.0).expand([num_lcs, 2]).to_event(1),
            )
            coefficients, correction = _ld_variant_data_form(
                ld_latent, ld_center, ld_scale, ld_low, ld_high,
                ld_map_code, ld_profile, ld_latent_low, ld_latent_high,
            )
            numpyro.factor('ld_variant_prior', jnp.sum(correction))
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', coefficients)
            else:
                c1 = numpyro.deterministic('c1', coefficients[:, 0])
                c2 = numpyro.deterministic('c2', coefficients[:, 1])
                profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                u = (P_LD @ (1.0 - profs).T).T
        elif ld_mode in {'gaussian', 'stellarprior', 'sing'}:
            if ld_profile == 'quadratic':
                if ld_mode == 'sing':
                    if sigma_u_ld is None:
                        raise ValueError("ld_mode='sing' requires (l, delta) sigma_u_ld.")
                    sing_mu = jnp.asarray(mu_u_ld, dtype=jnp.float64)
                    sing_sigma = jnp.asarray(sigma_u_ld, dtype=jnp.float64)
                    limb_l = numpyro.sample(
                        'limb_l',
                        dist.TruncatedNormal(sing_mu[:, 0], sing_sigma[:, 0], low=0.0, high=1.0),
                    )
                    u_plus = 1.0 - limb_l
                    # c1 >= 0, c1 + 2*c2 >= 0, and c1+c2 <= 1.
                    limb_delta = numpyro.sample(
                        'limb_delta',
                        dist.TruncatedNormal(
                            sing_mu[:, 1], sing_sigma[:, 1],
                            low=-u_plus / 4.0, high=u_plus / 4.0,
                        ),
                    )
                    c1 = numpyro.deterministic('c1', u_plus - 4.0 * limb_delta)
                    c2 = numpyro.deterministic('c2', 4.0 * limb_delta)
                    u = numpyro.deterministic('u', jnp.stack((c1, c2), axis=1))
                else:
                    if ld_mode == 'stellarprior' and sigma_u_ld is None:
                        raise ValueError("ld_mode='stellarprior' requires sigma_u_ld.")
                    u_scale = (
                        jnp.full_like(mu_u_ld, GAUSSIAN_LD_WIDTH)
                        if ld_mode == 'gaussian' else sigma_u_ld
                    )
                    u_prior_dist = dist.TruncatedNormal(
                        loc=mu_u_ld, scale=u_scale, low=0.0, high=1.0
                    ).to_event(1)
                    u = numpyro.sample('u', u_prior_dist)
            elif ld_profile == 'power2':
                if ld_mode == 'stellarprior' and sigma_u_ld is None:
                    raise ValueError("ld_mode='stellarprior' requires sigma_u_ld.")
                if ld_mode == 'gaussian':
                    sigma_u_ld = jnp.full_like(mu_u_ld, GAUSSIAN_LD_WIDTH)
                sigma_u_ld = jnp.asarray(sigma_u_ld, dtype=jnp.float64)
                sigma_u_ld = jnp.broadcast_to(sigma_u_ld, mu_u_ld.shape)
                sigma_u_ld = jnp.clip(sigma_u_ld, 1e-6, None)
                coefficient_prior = dist.TruncatedNormal(
                    mu_u_ld, sigma_u_ld,
                    low=jnp.asarray([0.0, 0.001]), high=1.0,
                ).to_event(1)
                if ld_parameterization == 'decorrelated' and ld_mode == 'gaussian':
                    h = numpyro.sample(
                        'ld_decorrelated',
                        dist.TransformedDistribution(
                            coefficient_prior, power2_transform
                        ),
                    )
                    coefficients = _enforce_decorrelated_coefficient_support(
                        power2_transform.inv(h),
                        jnp.asarray([0.0, 0.001]), 1.0
                    )
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                else:
                    c1 = numpyro.sample('c1', dist.TruncatedNormal(
                        mu_u_ld[:, 0], sigma_u_ld[:, 0], low=0.0, high=1.0))
                    c2 = numpyro.sample('c2', dist.TruncatedNormal(
                        mu_u_ld[:, 1], sigma_u_ld[:, 1], low=0.001, high=1.0))
                profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                u = (P_LD @ (1.0 - profs).T).T
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        elif ld_mode == 'uniform':
            if ld_profile == 'quadratic':
                if ld_uniform_basis == 'uplus_uminus':
                    low = jnp.broadcast_to(jnp.array([-1.0, -2.0]), (num_lcs, 2))
                    high = jnp.broadcast_to(jnp.array([2.0, 2.0]), (num_lcs, 2))
                    ld_sumdiff = numpyro.sample(
                        'ld_uplus_uminus', dist.Uniform(low, high).to_event(1))
                    u1 = numpyro.deterministic(
                        'u1', 0.5 * (ld_sumdiff[:, 0] + ld_sumdiff[:, 1]))
                    u2 = numpyro.deterministic(
                        'u2', 0.5 * (ld_sumdiff[:, 0] - ld_sumdiff[:, 1]))
                    numpyro.deterministic('l', 1.0 - ld_sumdiff[:, 0])
                    numpyro.deterministic(
                        'delta', (ld_sumdiff[:, 0] - ld_sumdiff[:, 1]) / 8.0)
                    u = numpyro.deterministic('u', jnp.stack((u1, u2), axis=1))
                else:
                    coefficient_prior = dist.Uniform(
                        uniform_coefficient_low, uniform_coefficient_high
                    ).expand([num_lcs, 2]).to_event(1)
                    u = numpyro.sample('u', coefficient_prior)
            elif ld_profile == 'power2':
                coefficient_prior = dist.Uniform(0.0, 1.0).expand([num_lcs, 2]).to_event(1)
                if ld_parameterization == 'decorrelated':
                    h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, power2_transform))
                    coefficients = _enforce_decorrelated_coefficient_support(
                        power2_transform.inv(h), 0.0, 1.0
                    )
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                else:
                    coefficients = numpyro.sample('ld_coefficients', coefficient_prior)
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                u = (P_LD @ (1.0 - profs).T).T
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        elif ld_mode == 'fixed':
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', ld_fixed)
            elif ld_profile == 'power2':
                c1_mu = numpyro.deterministic('c1', ld_fixed[:, 0])
                c2_mu = numpyro.deterministic('c2', ld_fixed[:, 1])
                c1, c2 = c1_mu, c2_mu
                profs = get_I_power2(c1_mu[:, None], c2_mu[:, None], MUS_LD[None, :])
                u = (P_LD @ (1.0 - profs).T).T
        elif ld_mode == 'interpolated':
            u = numpyro.deterministic('u', ld_interpolated)
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        params = {
            "period": PERIOD, "t0": t0s, "b": bs,
            "rors": rors,
            "_jaxoplanet_kernel": jaxoplanet_kernel,
            "_ld_profile": ld_profile,
            **orbital_params,
        }
        in_axes = {
            "period": None, "t0": None, "b": None,
            "rors": 0,
            "_jaxoplanet_kernel": None,
            "_ld_profile": None,
        }
        params["u"] = u
        in_axes["u"] = 0
        if param_method == 'duration':
            in_axes["duration"] = None
            phase_offsets, phase_mask = build_transit_phase_offsets(
                t, PERIOD, t0s, orbital_params["duration"]
            )
            params["_transit_phase_offsets"] = phase_offsets
            params["_transit_phase_mask"] = phase_mask
            in_axes["_transit_phase_offsets"] = None
            in_axes["_transit_phase_mask"] = None
        else:
            in_axes["a_rs"] = None
            in_axes["ecc"] = None
            in_axes["omega"] = None
        if use_transit_window:
            params["_transit_window_indices"] = transit_window_indices
            in_axes["_transit_window_indices"] = None
        if use_transit_grid:
            params['_transit_grid_nodes'] = transit_grid_nodes
            in_axes['_transit_grid_nodes'] = None
            params['_transit_grid_contact_fallback_margin'] = (
                0.0
            )
            in_axes['_transit_grid_contact_fallback_margin'] = None
            params['_transit_grid_force_stock_kernel'] = bool(
                ld_profile == 'power2' and not transit_grid_non_grazing
            )
            in_axes['_transit_grid_force_stock_kernel'] = None
            if ld_profile == 'power2':
                params['_transit_grid_c1'] = c1
                params['_transit_grid_c2'] = c2
                in_axes['_transit_grid_c1'] = 0
                in_axes['_transit_grid_c2'] = 0
            else:
                params['_transit_grid_quadratic'] = True
                in_axes['_transit_grid_quadratic'] = None

        surface_params = _sample_surface_parameters(
            surface_config, n_planets, num_lcs=num_lcs
        )
        params.update(surface_params)
        active_surface_basis = (
            surface_basis_data
            if surface_basis_data is not None else surface_basis
        )
        if active_surface_basis is not None:
            if not (
                basis_surface_valid
                and not surface_config.get("fit_geometry", True)
                and ld_mode == "fixed"
            ):
                raise ValueError(
                    "surface_basis_data requires a supported stellar-spot or "
                    "emission model with fixed geometry and fixed limb darkening."
                )
            params["_surface_basis"] = active_surface_basis
            in_axes["_surface_basis"] = (
                0 if jnp.ndim(active_surface_basis.baseline) > 1 else None
            )
        if "_surface_model" in surface_params:
            in_axes["_surface_model"] = None
            in_axes["_stellar_spots"] = None
        for name in (
            "eclipse_depth", "dayside_flux", "nightside_flux",
            "hotspot_offset", "stellar_spot_contrast",
        ):
            if name in surface_params:
                in_axes[name] = 0
        if "stellar_rotation_period" in surface_params:
            in_axes["stellar_rotation_period"] = None

        if trend_mode == 'gaussian_marginalized':
            if y is None:
                raise ValueError(
                    "Gaussian trend marginalization requires observed y values."
                )
            tau = None
            if 'explinear_spectroscopic' in detrend_components:
                if exp_trend is None or fixed_tau is None:
                    raise ValueError("explinear_spectroscopic requires exp_trend and fixed_tau.")
                tau = numpyro.deterministic(
                    'tau', jnp.full((num_lcs,), jnp.asarray(fixed_tau, dtype=jnp.float64))
                )
            elif 'explinear' in detrend_components:
                log_tau = numpyro.sample(
                    'log_tau',
                    dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]),
                )
                tau = numpyro.deterministic('tau', jnp.exp(log_tau))
            transit_model = jax.vmap(
                compute_transit_model,
                in_axes=(in_axes, None),
            )(params, t)
            # F = (1 + transit) * (X @ beta): fold the transit factor into
            # every design column so the model is exactly linear in beta.
            design, coefficient_names = build_marginalized_trend_design(
                detrend_type,
                t,
                num_lcs,
                tau=tau,
                spot_trend=spot_trend,
                spot_trend2=spot_trend2,
                jump_trend=jump_trend,
                exp_trend=exp_trend,
                transit_factor=1.0 + transit_model,
            )
            num_coefficients = len(coefficient_names)
            if trend_prior_mean is None:
                default_mean = jnp.zeros((num_coefficients,), dtype=jnp.float64)
                default_mean = default_mean.at[0].set(1.0)
                trend_prior_mean = jnp.broadcast_to(
                    default_mean, (num_lcs, num_coefficients)
                )
            else:
                trend_prior_mean = jnp.asarray(
                    trend_prior_mean, dtype=jnp.float64
                )
            if trend_prior_scale is None:
                trend_prior_scale = jnp.full(
                    (num_lcs, num_coefficients), 0.1, dtype=jnp.float64
                )
            else:
                trend_prior_scale = jnp.asarray(
                    trend_prior_scale, dtype=jnp.float64
                )
            log_likelihood, conditional = (
                marginalized_log_likelihood_and_conditional(
                    jnp.asarray(y, dtype=jnp.float64),
                    design,
                    error_broadcast,
                    trend_prior_mean,
                    prior_scale=trend_prior_scale,
                )
            )
            numpyro.deterministic('trend_beta_mean', conditional.mean)
            numpyro.deterministic('trend_beta_factor', conditional.factor)
            numpyro.factor('obs_marginalized', jnp.sum(log_likelihood))
            return

        if detrend_type != 'none':
            if trend_mode == 'free':
                params['c'] = numpyro.sample('c', dist.Uniform(0.9, 1.1).expand([num_lcs]))
                params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes.update({'c': 0, 'v': 0})

                poly_order = 1
                if 'quartic' in detrend_type:
                    poly_order = 4
                elif 'cubic' in detrend_type:
                    poly_order = 3
                elif 'quadratic' in detrend_type:
                    poly_order = 2

                if poly_order >= 2:
                    params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v2'] = 0
                if poly_order >= 3:
                    params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v3'] = 0
                if poly_order >= 4:
                    params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v4'] = 0

                if 'explinear_spectroscopic' in detrend_components:
                    if exp_trend is None or fixed_tau is None:
                        raise ValueError("explinear_spectroscopic requires exp_trend and fixed_tau.")
                    params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic(
                        'tau', jnp.full((num_lcs,), jnp.asarray(fixed_tau, dtype=jnp.float64))
                    )
                    in_axes.update({'A': 0, 'tau': 0})
                elif 'explinear' in detrend_components:
                    params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                    in_axes.update({'A': 0, 'tau': 0})

            elif trend_mode == 'fixed':
                trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
                params['v'] = numpyro.deterministic('v', trend_temp[:, 1])
                in_axes.update({'c': 0, 'v': 0})

                poly_order = 1
                if 'quartic' in detrend_type:
                    poly_order = 4
                elif 'cubic' in detrend_type:
                    poly_order = 3
                elif 'quadratic' in detrend_type:
                    poly_order = 2

                if poly_order >= 2:
                    params['v2'] = numpyro.deterministic('v2', trend_temp[:, 2])
                    in_axes['v2'] = 0
                if poly_order >= 3:
                    params['v3'] = numpyro.deterministic('v3', trend_temp[:, 3])
                    in_axes['v3'] = 0
                if poly_order >= 4:
                    params['v4'] = numpyro.deterministic('v4', trend_temp[:, 4])
                    in_axes['v4'] = 0
                if 'linear_discontinuity' in detrend_components:
                    params['t_jump'] = numpyro.deterministic('t_jump', trend_temp[:, 2])
                    params['jump'] = numpyro.deterministic('jump', trend_temp[:, 3])
                    in_axes.update({'t_jump': 0, 'jump': 0})
                elif 'explinear' in detrend_components:
                    params['A'] = numpyro.deterministic('A', trend_temp[:, 2])
                    params['tau'] = numpyro.deterministic('tau', trend_temp[:, 3])
                    in_axes.update({'A': 0, 'tau': 0})
                elif 'spot' in detrend_components and '2spot' not in detrend_components:
                    params['spot_amp'] = numpyro.deterministic('spot_amp', trend_temp[:, 2])
                    params['spot_mu'] = numpyro.deterministic('spot_mu', trend_temp[:, 3])
                    params['spot_sigma'] = numpyro.deterministic('spot_sigma', trend_temp[:, 4])
                    in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})
            else:
                raise ValueError(f"Unknown trend_mode: {trend_mode}")

        # Fixed-shape spectroscopic amplitudes are part of the linear trend
        # basis, so they must exist before the reduced-likelihood early return.
        if trend_mode == 'free':
            if '2spot_spectroscopic' in detrend_components:
                params['A_spot'] = numpyro.sample(
                    'A_spot', dist.Uniform(0.5, 2).expand([num_lcs])
                )
                params['A_spot2'] = numpyro.sample(
                    'A_spot2', dist.Uniform(0.5, 2).expand([num_lcs])
                )
                in_axes['A_spot'] = 0
                in_axes['A_spot2'] = 0
            elif 'spot_spectroscopic' in detrend_components:
                params['A_spot'] = numpyro.sample(
                    'A_spot', dist.Uniform(0.5, 2).expand([num_lcs])
                )
                in_axes['A_spot'] = 0
            if 'linear_discontinuity_spectroscopic' in detrend_components:
                params['A_jump'] = numpyro.sample(
                    'A_jump', dist.Uniform(0.5, 2).expand([num_lcs])
                )
                in_axes['A_jump'] = 0

        resolved_trend_design = trend_design
        if use_cadence_reduction and resolved_trend_design is None:
            centered_time = t - jnp.min(t)
            external_bases = {
                'A': exp_trend,
                'A_spot': spot_trend,
                'A_spot2': spot_trend2,
                'A_jump': jump_trend,
            }
            columns = []
            for name in linear_trend_names:
                if name == 'c':
                    column = jnp.ones_like(t, dtype=jnp.float64)
                elif name == 'v':
                    column = centered_time
                elif name.startswith('v') and name[1:].isdigit():
                    column = centered_time ** int(name[1:])
                else:
                    column = external_bases[name]
                    if column is None:
                        raise ValueError(
                            f"cadence reduction requires the fixed {name} basis."
                        )
                columns.append(jnp.asarray(column, dtype=jnp.float64))
            resolved_trend_design = jnp.stack(columns, axis=1)

        if use_cadence_reduction:
            statistics = (
                oot_reference_beta,
                oot_group_yerr,
                oot_group_count,
                oot_group_reference_sse,
                oot_group_x_reference_residual,
                oot_group_xx,
            )
            if y is None or any(value is None for value in statistics):
                raise ValueError(
                    "cadence_reduction='auto' requires observed y and the "
                    "precomputed out-of-transit sufficient statistics."
                )
            indices = transit_window_indices
            if use_transit_grid:
                from jaxoplanet.core.limb_dark import (
                    light_curve as stock_light_curve,
                )
                from .limb_dark_streamed import (
                    light_curve as streamed_light_curve,
                )
                from .transit_grid import (
                    grid_reduced_log_likelihood,
                )

                selected_kernel = resolve_jaxoplanet_kernel(
                    jaxoplanet_kernel,
                    ld_profile=ld_profile,
                    degree=int(jnp.shape(u)[-1]),
                    keplerian=False,
                )
                grid_kernel = (
                    streamed_light_curve
                    if (
                        selected_kernel == 'streamed'
                        and transit_grid_non_grazing
                    )
                    else stock_light_curve
                )
                active_y = jnp.atleast_2d(
                    jnp.asarray(y, dtype=jnp.float64)
                )[:, indices]
                active_yerr = yerr_matrix[:, indices]
                if likelihood_mask is None:
                    active_mask = jnp.ones_like(active_y, dtype=bool)
                else:
                    active_mask = jnp.broadcast_to(
                        jnp.asarray(likelihood_mask, dtype=bool),
                        jnp.shape(jnp.atleast_2d(y)),
                    )[:, indices]
                reference_beta = jnp.atleast_2d(jnp.asarray(
                    oot_reference_beta, dtype=jnp.float64
                ))
                group_error = jnp.atleast_2d(jnp.asarray(
                    oot_group_yerr, dtype=jnp.float64
                ))
                group_count = jnp.atleast_2d(jnp.asarray(
                    oot_group_count, dtype=jnp.float64
                ))
                group_reference_sse = jnp.atleast_2d(jnp.asarray(
                    oot_group_reference_sse, dtype=jnp.float64
                ))
                group_xr = jnp.asarray(
                    oot_group_x_reference_residual, dtype=jnp.float64
                )
                group_xx = jnp.asarray(oot_group_xx, dtype=jnp.float64)
                if group_xr.ndim == 2:
                    group_xr = group_xr[None, ...]
                    group_xx = group_xx[None, ...]
                design_active = jnp.asarray(
                    resolved_trend_design, dtype=jnp.float64
                )[indices]
                beta = jnp.stack(
                    tuple(params[name] for name in linear_trend_names), axis=1
                )
                if ld_profile == 'power2':
                    transit_theta = jnp.stack(
                        (rors[:, 0], c1, c2), axis=1
                    )
                else:
                    transit_theta = jnp.column_stack((rors[:, 0], u))
                theta = jnp.concatenate(
                    (transit_theta, beta, jitter[:, None]), axis=1
                )
                phase_active = params['_transit_phase_offsets'][0, indices]
                phase_mask_active = params['_transit_phase_mask'][0, indices]

                def lane_log_likelihood(
                    lane_theta,
                    lane_u,
                    lane_y,
                    lane_yerr,
                    lane_mask,
                    lane_reference_beta,
                    lane_group_error,
                    lane_group_count,
                    lane_reference_sse,
                    lane_group_xr,
                    lane_group_xx,
                ):
                    return grid_reduced_log_likelihood(
                        grid_kernel,
                        lane_theta,
                        lane_u,
                        phase_active,
                        phase_mask_active,
                        design_active,
                        lane_y,
                        lane_yerr,
                        lane_mask,
                        lane_reference_beta,
                        lane_group_error,
                        lane_group_count,
                        lane_reference_sse,
                        lane_group_xr,
                        lane_group_xx,
                        duration=mu_duration[0],
                        impact=bs[0],
                        ld_profile=ld_profile,
                        num_nodes=transit_grid_nodes,
                        contact_fallback_margin=(
                            0.0
                        ),
                        order=10,
                    )

                reduced_log_prob = jax.vmap(lane_log_likelihood)(
                    theta,
                    u,
                    active_y,
                    active_yerr,
                    active_mask,
                    reference_beta,
                    group_error,
                    group_count,
                    group_reference_sse,
                    group_xr,
                    group_xx,
                )
                numpyro.factor('obs_active', jnp.sum(reduced_log_prob[:, 0]))
                numpyro.factor(
                    'obs_out_of_window', jnp.sum(reduced_log_prob[:, 1])
                )
                return

            transit_active = jax.vmap(
                compute_transit_model_window,
                in_axes=(in_axes, None),
            )(params, t)
            beta = jnp.stack(
                tuple(params[name] for name in linear_trend_names), axis=1
            )
            design_active = jnp.asarray(
                resolved_trend_design, dtype=jnp.float64
            )[indices]
            trend_active = beta @ design_active.T
            active_model = apply_systematics(transit_active, trend_active)
            active_error = error_broadcast[:, indices]
            active_y = jnp.atleast_2d(
                jnp.asarray(y, dtype=jnp.float64)
            )[:, indices]
            active_log_prob = dist.Normal(
                active_model, active_error
            ).log_prob(active_y)
            if likelihood_mask is not None:
                active_mask = jnp.asarray(likelihood_mask, dtype=bool)
                active_mask = jnp.broadcast_to(
                    active_mask, jnp.shape(jnp.atleast_2d(y))
                )[:, indices]
                active_log_prob = jnp.where(
                    active_mask, active_log_prob, 0.0
                )
            numpyro.factor('obs_active', jnp.sum(active_log_prob))

            reference_beta = jnp.atleast_2d(jnp.asarray(
                oot_reference_beta, dtype=jnp.float64
            ))
            delta = beta - reference_beta
            group_xr = jnp.asarray(
                oot_group_x_reference_residual, dtype=jnp.float64
            )
            group_xx = jnp.asarray(oot_group_xx, dtype=jnp.float64)
            if group_xr.ndim == 2:
                group_xr = group_xr[None, ...]
                group_xx = group_xx[None, ...]
            group_sse = (
                jnp.atleast_2d(jnp.asarray(
                    oot_group_reference_sse, dtype=jnp.float64
                ))
                - 2.0 * jnp.einsum('lp,lgp->lg', delta, group_xr)
                + jnp.einsum('lp,lgpq,lq->lg', delta, group_xx, delta)
            )
            group_count = jnp.atleast_2d(jnp.asarray(
                oot_group_count, dtype=jnp.float64
            ))
            group_error = jnp.atleast_2d(jnp.asarray(
                oot_group_yerr, dtype=jnp.float64
            ))
            group_variance = group_error**2 + jitter[:, None]**2
            group_log_prob = -0.5 * (
                group_sse / group_variance
                + group_count * jnp.log(2.0 * jnp.pi * group_variance)
            )
            numpyro.factor('obs_out_of_window', jnp.sum(group_log_prob))
            return

        if 'explinear_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(
                params, t, exp_trend
            )
        elif 'gp_spectroscopic' in detrend_components:
            params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_gp'] = 0

            if 'linear' in detrend_components and 'v' not in params:
                params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v'] = 0

            poly_order = 1
            if 'quartic' in detrend_components:
                poly_order = 4
            elif 'cubic' in detrend_components:
                poly_order = 3
            elif 'quadratic' in detrend_components:
                poly_order = 2

            if poly_order >= 2 and 'v2' not in params:
                params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v2'] = 0
            if poly_order >= 3 and 'v3' not in params:
                params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v3'] = 0
            if poly_order >= 4 and 'v4' not in params:
                params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v4'] = 0

            if 'explinear' in detrend_components and 'A' not in params:
                params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                in_axes.update({'A': 0, 'tau': 0})

            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, gp_trend)

        elif '2spot_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, spot_trend2)
        elif 'spot_spectroscopic' in detrend_components:
            if 'linear_discontinuity_spectroscopic' in detrend_components:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, jump_trend)
            else:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, spot_trend)
        elif 'linear_discontinuity_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, jump_trend)
        else:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)

        if likelihood_mask is None:
            numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)
        else:
            with numpyro.handlers.mask(mask=jnp.asarray(likelihood_mask, dtype=bool)):
                numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model

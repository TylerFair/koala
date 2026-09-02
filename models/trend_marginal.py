"""Spectroscopic trend bases and posterior reconstruction helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .detrend import _split_components


def marginalized_trend_coefficient_names(detrend_type: str) -> tuple[str, ...]:
    """Return the ordered conditionally-linear coefficients for a trend."""
    components = _split_components(detrend_type)
    if "gp_spectroscopic" in components:
        raise ValueError("Analytic trend marginalization does not support GP trends.")
    if detrend_type == "none":
        raise ValueError("There are no trend coefficients to marginalize for 'none'.")

    if "2spot_spectroscopic" in components:
        names = ["c"]
        if "explinear" in components:
            names.extend(("v", "A"))
        names.extend(("A_spot", "A_spot2"))
        return tuple(names)
    if "spot_spectroscopic" in components:
        names = ["c"]
        # The quadratic+spot kernel retains the polynomial basis.
        if "quadratic" in components:
            names.extend(("v", "v2"))
        elif "explinear" in components:
            names.extend(("v", "A"))
        names.append("A_spot")
        if "linear_discontinuity_spectroscopic" in components:
            names.append("A_jump")
        return tuple(names)
    if "linear_discontinuity_spectroscopic" in components:
        return ("c", "A_jump")

    names = ["c", "v"]
    if "quadratic" in components:
        names.append("v2")
    elif "cubic" in components:
        names.extend(("v2", "v3"))
    elif "quartic" in components:
        names.extend(("v2", "v3", "v4"))
    if "explinear" in components:
        names.append("A")
    return tuple(names)


def build_marginalized_trend_design(
    detrend_type: str,
    t,
    num_channels: int,
    *,
    tau=None,
    spot_trend=None,
    spot_trend2=None,
    jump_trend=None,
    exp_trend=None,
):
    """Build a ``[channel, time, coefficient]`` additive trend design."""
    names = marginalized_trend_coefficient_names(detrend_type)
    t = jnp.asarray(t, dtype=jnp.float64)
    t_norm = t - jnp.min(t)
    shared = {
        "c": jnp.ones_like(t_norm),
        "v": t_norm,
        "v2": t_norm**2,
        "v3": t_norm**3,
        "v4": t_norm**4,
        "A_spot": None if spot_trend is None else jnp.asarray(spot_trend, dtype=jnp.float64),
        "A_spot2": None if spot_trend2 is None else jnp.asarray(spot_trend2, dtype=jnp.float64),
        "A_jump": None if jump_trend is None else jnp.asarray(jump_trend, dtype=jnp.float64),
    }
    columns = []
    for name in names:
        if name == "A":
            if "explinear_spectroscopic" in _split_components(detrend_type):
                template = jnp.asarray(exp_trend, dtype=jnp.float64)
                if template.ndim != 1 or template.shape[0] != t.shape[0]:
                    raise ValueError("explinear_spectroscopic requires exp_trend with shape [time].")
                columns.append(jnp.broadcast_to(template, (num_channels, t.shape[0])))
                continue
            if tau is None:
                raise ValueError("explinear marginalization requires tau.")
            tau_arr = jnp.asarray(tau, dtype=jnp.float64)
            if tau_arr.ndim == 0:
                tau_arr = jnp.broadcast_to(tau_arr, (num_channels,))
            columns.append(jnp.exp(-t_norm[None, :] / tau_arr[:, None]))
            continue
        column = shared[name]
        if column is None:
            raise ValueError(
                f"{detrend_type!r} marginalization requires {name}'s fixed template."
            )
        if column.ndim != 1 or column.shape[0] != t.shape[0]:
            raise ValueError(f"Template for {name} must have shape [time].")
        columns.append(jnp.broadcast_to(column, (num_channels, t.shape[0])))
    return jnp.stack(columns, axis=-1), names


def materialize_marginalized_trend_samples(
    samples,
    coefficient_names,
    key,
    *,
    drop_conditional_arrays: bool = True,
):
    """Draw trend coefficients from saved conditional Gaussian factors.

    One conditional coefficient vector is generated for every nonlinear MCMC
    draw and wavelength channel, producing valid joint posterior draws without
    placing the nuisance coefficients in the HMC state.
    """
    if "trend_beta_mean" not in samples or "trend_beta_factor" not in samples:
        raise KeyError(
            "Marginalized samples require trend_beta_mean and trend_beta_factor."
        )
    mean = jnp.asarray(samples["trend_beta_mean"], dtype=jnp.float64)
    factor = jnp.asarray(samples["trend_beta_factor"], dtype=jnp.float64)
    names = tuple(coefficient_names)
    if mean.shape[-1] != len(names):
        raise ValueError("Coefficient names do not match conditional mean shape.")
    if factor.shape != mean.shape + (mean.shape[-1],):
        raise ValueError("Conditional factor must have shape [..., K, K].")
    standard = jax.random.normal(key, mean.shape, dtype=jnp.float64)
    beta = mean + jnp.einsum("...ij,...j->...i", factor, standard)
    result = dict(samples)
    for index, name in enumerate(names):
        result[name] = beta[..., index]
    if drop_conditional_arrays:
        result.pop("trend_beta_mean", None)
        result.pop("trend_beta_factor", None)
    return result

"""Spectroscopic trend bases and posterior reconstruction helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .detrend import _split_components


def normalized_polynomial_transform(t, num_channels, order, mask=None):
    """Map normalized polynomial coefficients to elapsed-day coefficients.

    Rows excluded from the likelihood never define the coordinate system.
    This is a coordinate change only: every observed integration is retained.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    elapsed = t - jnp.min(t)
    valid = jnp.ones((num_channels, t.size), dtype=bool)
    if mask is not None:
        valid = jnp.broadcast_to(jnp.atleast_2d(mask), valid.shape)
    first = jnp.min(jnp.where(valid, elapsed, jnp.inf), axis=-1)
    last = jnp.max(jnp.where(valid, elapsed, -jnp.inf), axis=-1)
    span = last - first
    offset = jnp.sum(jnp.where(valid, elapsed, 0.), axis=-1) / jnp.sum(valid, axis=-1)
    transform = jnp.zeros((num_channels, order + 1, order + 1), dtype=t.dtype)
    import math
    for column in range(order + 1):
        for row in range(column + 1):
            transform = transform.at[:, row, column].set(
                math.comb(column, row) * (-offset)**(column-row) / span**column)
    return transform



def normalized_linear_trend_transform(t,num_channels,names,mask=None):
    """Normalize the polynomial block, leaving fixed template amplitudes alone."""
    positions=[i for i,name in enumerate(names) if name=='c' or name=='v'
               or name.startswith('v') and name[1:].isdigit()]
    expected=['c']+['v' if k==1 else f'v{k}' for k in range(1,len(positions))]
    if [names[i] for i in positions]!=expected:
        raise ValueError('Linear trend polynomial columns must be in degree order.')
    transform=jnp.broadcast_to(jnp.eye(len(names),dtype=jnp.float64),
                                (num_channels,len(names),len(names)))
    polynomial=normalized_polynomial_transform(t,num_channels,len(positions)-1,mask)
    for row,i in enumerate(positions):
        for column,j in enumerate(positions):
            transform=transform.at[:,i,j].set(polynomial[:,row,column])
    return transform


def marginalized_trend_coefficient_names(detrend_type: str) -> tuple[str, ...]:
    """Return the ordered conditionally-linear coefficients for a trend."""
    components = _split_components(detrend_type)
    if "gp_spectroscopic" in components:
        raise ValueError("Analytic trend marginalization does not support GP trends.")
    if detrend_type == "none":
        raise ValueError("There are no trend coefficients to marginalize for 'none'.")
    if detrend_type == "constant":
        return ("c",)

    if "2spot_spectroscopic" in components:
        names = ["c", "v"]
        if "quadratic" in components:
            names.append("v2")
        elif components & {"explinear", "explinear_spectroscopic"}:
            names.append("A")
        names.extend(("A_spot", "A_spot2"))
        return tuple(names)
    if "spot_spectroscopic" in components:
        names = ["c", "v"]
        # The quadratic+spot kernel retains the polynomial basis.
        if "quadratic" in components:
            names.append("v2")
        elif components & {"explinear", "explinear_spectroscopic"}:
            names.append("A")
        names.append("A_spot")
        if "linear_discontinuity_spectroscopic" in components:
            names.append("A_jump")
        return tuple(names)
    if "linear_discontinuity_spectroscopic" in components:
        return ("c", "v", "A_jump")

    names = ["c", "v"]
    if "quadratic" in components:
        names.append("v2")
    elif "cubic" in components:
        names.extend(("v2", "v3"))
    elif "quartic" in components:
        names.extend(("v2", "v3", "v4"))
    if components & {"explinear", "explinear_spectroscopic"}:
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
    transit_factor=None,
):
    """Build a ``[channel, time, coefficient]`` systematics design.

    The light-curve model is ``(1 + transit) * (X @ beta)``; when
    ``transit_factor`` (``1 + transit``, shape ``[time]`` or
    ``[channel, time]``) is given, every column is multiplied by it so the
    model is exactly linear in ``beta`` and the marginal likelihood stays
    exact.  Without it the design describes the systematics factor alone.
    """
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
        if column.ndim == 1 and column.shape[0] == t.shape[0]:
            columns.append(jnp.broadcast_to(column, (num_channels, t.shape[0])))
        elif column.ndim == 2 and column.shape == (num_channels, t.shape[0]):
            columns.append(column)
        else:
            raise ValueError(
                f"Template for {name} must have shape [time] or [channel, time]."
            )
    design = jnp.stack(columns, axis=-1)
    if transit_factor is not None:
        factor = jnp.asarray(transit_factor, dtype=jnp.float64)
        if factor.ndim == 1 and factor.shape[0] == t.shape[0]:
            factor = jnp.broadcast_to(factor, (num_channels, t.shape[0]))
        elif factor.shape != (num_channels, t.shape[0]):
            raise ValueError(
                "transit_factor must have shape [time] or [channel, time]."
            )
        design = design * factor[..., None]
    return design, names


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

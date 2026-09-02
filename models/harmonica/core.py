import jax
import jax.numpy as jnp
import numpy as np

try:
    from harmonica.jax import (
        harmonica_transit_power2_ld,
        harmonica_transit_quad_ld,
    )
except ImportError:
    def harmonica_transit_power2_ld(*args, **kwargs):
        raise ImportError(
            "harmonica is required to evaluate the harmonica transit model."
        )

    def harmonica_transit_quad_ld(*args, **kwargs):
        raise ImportError(
            "harmonica is required to evaluate the harmonica transit model."
        )

_ALL_ODD_COEFF_SPECS = (("a1", 1), ("a3", 3), ("a5", 5))

# For r(theta) = a0 + a1 cos(theta), the Catwoman-comparable half-area
# contrast is q = (D_evening - D_morning) / (D_evening + D_morning).
HARMONICA_HALF_AREA_CONTRAST_FACTOR = 4.0 / np.pi
HARMONICA_HALF_AREA_CONVEX_Q_LIMIT = 16.0 / (9.0 * np.pi)
HARMONICA_SPECTRO_RORS_MIN = float(np.sqrt(1e-5))
HARMONICA_SPECTRO_RORS_MAX = float(np.sqrt(0.5))


def harmonica_odd_coeff_specs(max_order=1):
    """Return the coeff specs up to *max_order* (1, 3, or 5)."""
    return tuple(s for s in _ALL_ODD_COEFF_SPECS if s[1] <= max_order)


HARMONICA_ODD_COEFF_SPECS = _ALL_ODD_COEFF_SPECS


def harmonica_half_area_q_from_ratio(a1_over_a0):
    """Map the first-harmonic radius ratio to physical half-area contrast.

    This is the forward map for the smooth ``N_c=1`` shape,

    ``q = (4/pi) x / (1 + x**2/2)``, where ``x = a1/a0``.
    """
    ratio = jnp.asarray(a1_over_a0, dtype=jnp.float64)
    return (
        HARMONICA_HALF_AREA_CONTRAST_FACTOR
        * ratio
        / (1.0 + 0.5 * ratio**2)
    )


def harmonica_half_area_ratio_from_q(q):
    """Stably invert physical half-area contrast to ``a1/a0``.

    The rationalized form avoids cancellation at ``q ~= 0``. Inputs used by
    the sampler satisfy ``|q| < HARMONICA_HALF_AREA_CONVEX_Q_LIMIT``, so the
    square-root argument is strictly positive.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    factor = jnp.asarray(
        HARMONICA_HALF_AREA_CONTRAST_FACTOR, dtype=jnp.float64
    )
    root = jnp.sqrt(jnp.maximum(0.0, factor**2 - 2.0 * q**2))
    return 2.0 * q / (factor + root)


def harmonica_half_area_q_from_coefficients(a0, a1):
    """Return physical half-area contrast from smooth-shape coefficients."""
    a0 = jnp.asarray(a0, dtype=jnp.float64)
    a1 = jnp.asarray(a1, dtype=jnp.float64)
    return harmonica_half_area_q_from_ratio(a1 / a0)


def harmonica_half_area_a1_from_q(a0, q):
    """Return the first cosine coefficient for radius ``a0`` and contrast ``q``."""
    a0 = jnp.asarray(a0, dtype=jnp.float64)
    return a0 * harmonica_half_area_ratio_from_q(q)


def harmonica_half_area_coefficients_from_area_radius(area_radius, q):
    """Map total-area radius and physical contrast to smooth coefficients.

    ``area_radius**2`` is the full silhouette area divided by ``pi``. This is
    the radius stored in the usual spectroscopic ``rors``/``depths`` products.
    """
    area_radius = jnp.asarray(area_radius, dtype=jnp.float64)
    ratio = harmonica_half_area_ratio_from_q(q)
    a0 = area_radius / jnp.sqrt(1.0 + 0.5 * ratio**2)
    return a0, a0 * ratio


def harmonica_half_area_area_radius_and_q(a0, a1):
    """Map smooth coefficients to total-area radius and physical contrast."""
    total, evening, morning = harmonica_half_area_depths(a0, a1)
    q = (evening - morning) / (evening + morning)
    return jnp.sqrt(total), q


def harmonica_half_area_depths(a0, a1):
    """Return total, evening, and morning half-area-equivalent depths.

    ``depth_total_area`` is the mean of the two representative limb depths,
    and therefore equals the full silhouette area divided by ``pi``.
    """
    a0 = jnp.asarray(a0, dtype=jnp.float64)
    a1 = jnp.asarray(a1, dtype=jnp.float64)
    total = a0**2 + 0.5 * a1**2
    contrast = HARMONICA_HALF_AREA_CONTRAST_FACTOR * a0 * a1
    return total, total + contrast, total - contrast


def _broadcast_planet_param(param, n_planets, name):
    arr = jnp.atleast_1d(jnp.asarray(param, dtype=jnp.float64))
    if arr.size == 1 and n_planets > 1:
        return jnp.repeat(arr, n_planets)
    if arr.size != n_planets:
        raise ValueError(
            f"`{name}` must be scalar or length {n_planets}, got shape {arr.shape}."
        )
    return arr


def harmonica_cos_i_from_geometry(b, a_rs, ecc=0.0, omega=0.0):
    fac = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(omega))
    return b / (a_rs * fac)


def harmonica_impact_param_from_cos_i(cos_i, a_rs, ecc=0.0, omega=0.0):
    fac = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(omega))
    return a_rs * fac * cos_i


def harmonica_geometry_is_valid(b, a_rs, ecc=0.0, omega=0.0, tol=1e-10):
    cos_i = harmonica_cos_i_from_geometry(b, a_rs, ecc=ecc, omega=omega)
    return (
        jnp.isfinite(cos_i)
        & jnp.isfinite(a_rs)
        & jnp.isfinite(ecc)
        & jnp.isfinite(omega)
        & (a_rs > 0.0)
        & (ecc >= 0.0)
        & (ecc < 1.0)
        & (cos_i >= -1.0 - tol)
        & (cos_i <= 1.0 + tol)
    )


def harmonica_duration_from_geometry(period, a_rs, b, rors, ecc=0.0, omega=0.0):
    cos_i = harmonica_cos_i_from_geometry(b, a_rs, ecc=ecc, omega=omega)
    cos_i = jnp.clip(cos_i, 0.0, 1.0)
    sin_i = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_i ** 2))
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b ** 2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc ** 2))
        / (1.0 + ecc * jnp.sin(omega))
    )
    denom = jnp.maximum(a_rs * sin_i, 1e-12)
    arg = jnp.clip(chord / denom, 0.0, 1.0)
    return (period / jnp.pi) * jnp.arcsin(arg) * speed_factor


def harmonica_a_rs_from_duration(period, duration, b, rors, ecc=0.0, omega=0.0):
    period = jnp.asarray(period, dtype=jnp.float64)
    duration = jnp.asarray(duration, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    rors = jnp.asarray(rors, dtype=jnp.float64)
    ecc = jnp.asarray(ecc, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=jnp.float64)

    fac = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(omega))
    fac = jnp.maximum(fac, 1e-12)
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b ** 2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc ** 2))
        / (1.0 + ecc * jnp.sin(omega))
    )
    angle = jnp.pi * duration / jnp.maximum(period * speed_factor, 1e-12)
    sin_angle = jnp.sin(jnp.clip(angle, 1e-12, 0.5 * jnp.pi - 1e-12))
    a_rs_sq = (chord / jnp.maximum(sin_angle, 1e-12)) ** 2 + (b / fac) ** 2
    return jnp.sqrt(jnp.maximum(a_rs_sq, 1e-12))


def harmonica_duration_from_cos_i(period, a_rs, cos_i, rors, ecc=0.0, omega=0.0):
    cos_i = jnp.clip(jnp.asarray(cos_i, dtype=jnp.float64), 0.0, 1.0)
    b = harmonica_impact_param_from_cos_i(cos_i, a_rs, ecc=ecc, omega=omega)
    sin_i = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_i ** 2))
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b ** 2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc ** 2))
        / (1.0 + ecc * jnp.sin(omega))
    )
    denom = jnp.maximum(a_rs * sin_i, 1e-12)
    arg = jnp.clip(chord / denom, 0.0, 1.0)
    return (period / jnp.pi) * jnp.arcsin(arg) * speed_factor


def _highest_harmonica_odd_order(params):
    for name, order in reversed(HARMONICA_ODD_COEFF_SPECS):
        if name in params:
            return order
    return 1


def _build_harmonica_r_vector(rors, coeffs, highest_order):
    r_terms = [jnp.asarray(rors, dtype=jnp.float64)]
    for order in range(1, highest_order + 1):
        if order % 2 == 1:
            coeff = coeffs.get(f"a{order}", 0.0)
        else:
            coeff = 0.0
        r_terms.extend(
            [
                jnp.asarray(coeff, dtype=jnp.float64),
                jnp.asarray(0.0, dtype=jnp.float64),
            ]
        )
    return jnp.stack(r_terms)


def _broadcast_harmonica_lc_param(param, n_lcs, name):
    arr = jnp.asarray(param, dtype=jnp.float64)
    if arr.ndim == 0:
        return jnp.full((n_lcs,), arr, dtype=jnp.float64)
    if arr.ndim == 1:
        if arr.shape[0] == 1 and n_lcs > 1:
            return jnp.broadcast_to(arr, (n_lcs,))
        if arr.shape[0] != n_lcs:
            raise ValueError(
                f"`{name}` must be scalar or length {n_lcs}, got shape {arr.shape}."
            )
        return arr
    raise ValueError(f"`{name}` must be scalar or 1D, got shape {arr.shape}.")


def _broadcast_harmonica_lc_planet_param(param, n_lcs, n_planets, name):
    arr = jnp.asarray(param, dtype=jnp.float64)
    if arr.ndim == 0:
        return jnp.full((n_lcs, n_planets), arr, dtype=jnp.float64)
    if arr.ndim == 1:
        if arr.shape[0] == n_planets:
            return jnp.broadcast_to(arr[None, :], (n_lcs, n_planets))
        if n_planets == 1 and arr.shape[0] == n_lcs:
            return arr[:, None]
        if arr.shape[0] == 1:
            return jnp.full((n_lcs, n_planets), arr[0], dtype=jnp.float64)
        raise ValueError(
            f"`{name}` must be scalar, length {n_planets}, or shape ({n_lcs}, {n_planets}); "
            f"got shape {arr.shape}."
        )
    if arr.ndim == 2:
        if arr.shape == (n_lcs, n_planets):
            return arr
        if arr.shape == (1, n_planets):
            return jnp.broadcast_to(arr, (n_lcs, n_planets))
        if n_planets == 1 and arr.shape == (n_lcs, 1):
            return arr
    raise ValueError(
        f"`{name}` must be scalar, 1D, or shape ({n_lcs}, {n_planets}); got shape {arr.shape}."
    )


def _broadcast_harmonica_times(t, n_lcs):
    times = jnp.asarray(t, dtype=jnp.float64)
    if times.ndim == 1:
        return jnp.broadcast_to(times[None, :], (n_lcs, times.shape[0]))
    if times.ndim == 2:
        if times.shape[0] == n_lcs:
            return times
        if times.shape[0] == 1:
            return jnp.broadcast_to(times, (n_lcs, times.shape[1]))
    raise ValueError(
        f"`t` must have shape (n_times,) or ({n_lcs}, n_times); got shape {times.shape}."
    )


def _build_harmonica_r_matrix(rors, coeffs, highest_order):
    r_terms = [jnp.asarray(rors, dtype=jnp.float64)]
    zeros = jnp.zeros_like(r_terms[0])
    for order in range(1, highest_order + 1):
        if order % 2 == 1:
            coeff = jnp.asarray(coeffs.get(f"a{order}", 0.0), dtype=jnp.float64)
        else:
            coeff = zeros
        r_terms.extend([coeff, zeros])
    return jnp.stack(r_terms, axis=-1)


def compute_transit_model_harmonica_batched(params, t):
    """
    Spectroscopic Harmonica transit model with one batched call over light curves.

    `t` may be shape `(n_times,)` for shared timestamps or `(n_lcs, n_times)` for
    per-light-curve timestamps. Channel-varying inputs (`rors`, odd coefficients,
    `c_ld`, `alpha_ld`) are broadcast to `(n_lcs, n_planets)` or `(n_lcs,)`.

    Returns the transit signal with shape `(n_lcs, n_times)`.
    """
    periods = jnp.atleast_1d(jnp.asarray(params["period"], dtype=jnp.float64))
    n_planets = periods.shape[0]
    highest_order = _highest_harmonica_odd_order(params)
    active_odd_specs = harmonica_odd_coeff_specs(highest_order)

    ld_profile = "quadratic" if "u1_ld" in params else "power2"
    if ld_profile == "power2":
        ld_reference = jnp.asarray(params["c_ld"], dtype=jnp.float64)
    elif ld_profile == "quadratic":
        ld_reference = jnp.asarray(params["u1_ld"], dtype=jnp.float64)
    else:
        raise ValueError(f"Unsupported Harmonica limb-darkening profile: {ld_profile}")
    n_lcs = 1 if ld_reference.ndim == 0 else ld_reference.shape[0]
    if ld_profile == "power2":
        c_ld = _broadcast_harmonica_lc_param(
            params["c_ld"], n_lcs, "c_ld"
        )
        alpha_ld = _broadcast_harmonica_lc_param(
            params["alpha_ld"], n_lcs, "alpha_ld"
        )
    else:
        u1_ld = _broadcast_harmonica_lc_param(
            params["u1_ld"], n_lcs, "u1_ld"
        )
        u2_ld = _broadcast_harmonica_lc_param(
            params["u2_ld"], n_lcs, "u2_ld"
        )
    raw_times = jnp.asarray(t, dtype=jnp.float64)
    times = (
        raw_times
        if (
            raw_times.ndim == 1
            and jax.default_backend() == "gpu"
            and highest_order <= 1
        )
        else _broadcast_harmonica_times(raw_times, n_lcs)
    )

    rorss = _broadcast_harmonica_lc_planet_param(params["rors"], n_lcs, n_planets, "rors")
    odd_coeffs = {
        name: _broadcast_harmonica_lc_planet_param(
            params.get(name, 0.0), n_lcs, n_planets, name
        )
        for name, _order in active_odd_specs
    }
    t0s = _broadcast_planet_param(params["t0"], n_planets, "t0")
    a_rss = _broadcast_planet_param(params["a_rs"], n_planets, "a_rs")
    eccs = _broadcast_planet_param(params.get("ecc", 0.0), n_planets, "ecc")
    omegas = _broadcast_planet_param(params.get("omega", 0.0), n_planets, "omega")
    if "inc" in params:
        incs = _broadcast_planet_param(params["inc"], n_planets, "inc")
    elif "cos_i" in params:
        cos_is = _broadcast_planet_param(params["cos_i"], n_planets, "cos_i")
        incs = jnp.arccos(jnp.clip(cos_is, 0.0, 1.0 - 1e-9))
    else:
        bs = _broadcast_planet_param(params["b"], n_planets, "b")
        cos_is = harmonica_cos_i_from_geometry(bs, a_rss, ecc=eccs, omega=omegas)
        incs = jnp.arccos(jnp.clip(cos_is, 0.0, 1.0 - 1e-9))

    def get_lc_batched(period, t0, rors, a_rs, ecc, omega, inc, *odd_args):
        coeff_map = {
            name: coeff
            for (name, _order), coeff in zip(active_odd_specs, odd_args)
        }
        r = _build_harmonica_r_matrix(rors, coeff_map, highest_order)
        if ld_profile == "power2":
            flux = harmonica_transit_power2_ld(
                times,
                t0,
                period,
                a_rs,
                inc,
                ecc,
                omega,
                c=c_ld,
                alpha=alpha_ld,
                r=r,
            )
        else:
            flux = harmonica_transit_quad_ld(
                times,
                t0,
                period,
                a_rs,
                inc,
                ecc,
                omega,
                u1=u1_ld,
                u2=u2_ld,
                r=r,
            )
        return flux - 1.0

    vmap_args = [
        periods, t0s,
        jnp.swapaxes(rorss, 0, 1),
        a_rss, eccs, omegas, incs,
        *[jnp.swapaxes(odd_coeffs[name], 0, 1) for name, _order in active_odd_specs],
    ]
    if n_planets == 1:
        return get_lc_batched(
            *[arg[0] if hasattr(arg, "ndim") and arg.ndim > 0 else arg for arg in vmap_args]
        )
    batched_lcs = jax.vmap(get_lc_batched)(*vmap_args)
    return jnp.sum(batched_lcs, axis=0)


def compute_transit_model_harmonica(params, t):
    """
    Transit model using harmonica with native power-2 limb darkening
    and odd-cosine transmission strings up to fifth order.

    Returns the transit *signal* (0 out of transit, negative during transit)
    matching jaxoplanet convention used by the detrending kernels.
    """
    periods = jnp.atleast_1d(params["period"])
    t0s = jnp.atleast_1d(params["t0"])
    rorss = jnp.atleast_1d(params["rors"])
    n_planets = periods.shape[0]
    highest_order = _highest_harmonica_odd_order(params)
    active_odd_specs = harmonica_odd_coeff_specs(highest_order)
    odd_coeffs = {
        name: _broadcast_planet_param(
            params.get(name, jnp.zeros_like(rorss)), n_planets, name
        )
        for name, _order in active_odd_specs
    }
    a_rss = _broadcast_planet_param(params["a_rs"], n_planets, "a_rs")
    eccs = _broadcast_planet_param(params.get("ecc", 0.0), n_planets, "ecc")
    omegas = _broadcast_planet_param(params.get("omega", 0.0), n_planets, "omega")
    if "inc" in params:
        incs = _broadcast_planet_param(params["inc"], n_planets, "inc")
    elif "cos_i" in params:
        cos_is = _broadcast_planet_param(params["cos_i"], n_planets, "cos_i")
        incs = jnp.arccos(jnp.clip(cos_is, 0.0, 1.0 - 1e-9))
    else:
        bs = _broadcast_planet_param(params["b"], n_planets, "b")
        cos_is = harmonica_cos_i_from_geometry(bs, a_rss, ecc=eccs, omega=omegas)
        incs = jnp.arccos(jnp.clip(cos_is, 0.0, 1.0 - 1e-9))
    ld_profile = "quadratic" if "u1_ld" in params else "power2"

    def get_lc(period, t0, rors, a_rs, ecc, omega, inc, *odd_args):
        coeff_map = {
            name: coeff
            for (name, _order), coeff in zip(active_odd_specs, odd_args)
        }
        r = _build_harmonica_r_vector(rors, coeff_map, highest_order)
        if ld_profile == "power2":
            flux = harmonica_transit_power2_ld(
                t,
                t0,
                period,
                a_rs,
                inc,
                ecc,
                omega,
                c=params["c_ld"],
                alpha=params["alpha_ld"],
                r=r,
            )
        elif ld_profile == "quadratic":
            flux = harmonica_transit_quad_ld(
                t,
                t0,
                period,
                a_rs,
                inc,
                ecc,
                omega,
                u1=params["u1_ld"],
                u2=params["u2_ld"],
                r=r,
            )
        else:
            raise ValueError(
                f"Unsupported Harmonica limb-darkening profile: {ld_profile}"
            )
        return flux - 1.0

    vmap_args = [
        periods, t0s, rorss, a_rss, eccs, omegas, incs,
        *[odd_coeffs[name] for name, _order in active_odd_specs],
    ]
    batched_lcs = jax.vmap(get_lc)(*vmap_args)
    return jnp.sum(batched_lcs, axis=0)

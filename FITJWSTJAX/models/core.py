import jax
import jax.numpy as jnp
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from harmonica.jax import harmonica_transit_power2_ld
import numpy as np

# Full catalogue of supported odd-cosine coefficients.
_ALL_ODD_COEFF_SPECS = (("a1", 1), ("a3", 3), ("a5", 5))

def harmonica_odd_coeff_specs(max_order=1):
    """Return the coeff specs up to *max_order* (1, 3, or 5)."""
    return tuple(s for s in _ALL_ODD_COEFF_SPECS if s[1] <= max_order)

# Default: first-order only (fast, sufficient for most limb-limb science).
HARMONICA_ODD_COEFF_SPECS = harmonica_odd_coeff_specs(1)

def _to_f64(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
        return jnp.asarray(x, jnp.float64)
    return x

def _tree_to_f64(tree):
    return jax.tree_util.tree_map(_to_f64, tree)


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
    sin_i = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_i**2))
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b ** 2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc**2))
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

    fac = (1.0 - ecc**2) / (1.0 + ecc * jnp.sin(omega))
    fac = jnp.maximum(fac, 1e-12)
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b**2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc**2))
        / (1.0 + ecc * jnp.sin(omega))
    )
    angle = jnp.pi * duration / jnp.maximum(period * speed_factor, 1e-12)
    sin_angle = jnp.sin(jnp.clip(angle, 1e-12, 0.5 * jnp.pi - 1e-12))
    a_rs_sq = (chord / jnp.maximum(sin_angle, 1e-12)) ** 2 + (b / fac) ** 2
    return jnp.sqrt(jnp.maximum(a_rs_sq, 1e-12))


def harmonica_duration_from_cos_i(period, a_rs, cos_i, rors, ecc=0.0, omega=0.0):
    cos_i = jnp.clip(jnp.asarray(cos_i, dtype=jnp.float64), 0.0, 1.0)
    b = harmonica_impact_param_from_cos_i(cos_i, a_rs, ecc=ecc, omega=omega)
    sin_i = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_i**2))
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b ** 2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc**2))
        / (1.0 + ecc * jnp.sin(omega))
    )
    denom = jnp.maximum(a_rs * sin_i, 1e-12)
    arg = jnp.clip(chord / denom, 0.0, 1.0)
    return (period / jnp.pi) * jnp.arcsin(arg) * speed_factor

def compute_transit_model(params, t):
    """
    Transit Model for one or more planets, using vmap for performance.
    Expects params to contain 'period', 'duration', 't0', 'b', 'rors', 'u'.
    These should be arrays where the 0-th dimension is the planet index,
    except 'u' which is limb darkening parameters.
    """
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


def uses_harmonica_params(params):
    """Detect whether a parameter bundle should be evaluated with harmonica."""
    return all(key in params for key in ("a_rs", "c_ld", "alpha_ld"))


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


def compute_transit_model_harmonica(params, t):
    """
    Transit model using harmonica with native power-2 limb darkening
    and odd-cosine transmission strings up to fifth order.

    Returns the transit *signal* (0 out of transit, negative during transit)
    matching jaxoplanet convention used by the detrending kernels.

    Expects params to contain:
      'period', 't0', 'b', 'rors' - orbital
      'a_rs'    - semi-major axis in stellar radii
      'ecc'     - eccentricity (0 for circular)
      'omega'   - argument of periastron [radians]
      'a1','a3','a5' - odd cosine asymmetry coefficients (optional)
      'c_ld'    - power-2 limb darkening c
      'alpha_ld'- power-2 limb darkening alpha

    Derives inclination from (b, a_rs, ecc, omega).
    """
    periods = jnp.atleast_1d(params["period"])
    t0s = jnp.atleast_1d(params["t0"])
    rorss = jnp.atleast_1d(params["rors"])
    n_planets = periods.shape[0]
    highest_order = _highest_harmonica_odd_order(params)
    odd_coeffs = {
        name: _broadcast_planet_param(
            params.get(name, jnp.zeros_like(rorss)), n_planets, name
        )
        for name, order in HARMONICA_ODD_COEFF_SPECS
        if order <= highest_order
    }
    a_rss = _broadcast_planet_param(params["a_rs"], n_planets, "a_rs")
    eccs = _broadcast_planet_param(params.get("ecc", 0.0), n_planets, "ecc")
    omegas = _broadcast_planet_param(params.get("omega", 0.0), n_planets, "omega")
    if "cos_i" in params:
        cos_is = _broadcast_planet_param(params["cos_i"], n_planets, "cos_i")
    else:
        bs = _broadcast_planet_param(params["b"], n_planets, "b")
        cos_is = harmonica_cos_i_from_geometry(bs, a_rss, ecc=eccs, omega=omegas)
    c_ld = params["c_ld"]
    alpha_ld = params["alpha_ld"]

    def get_lc(period, t0, rors, a_rs, ecc, omega, cos_i, *odd_args):
        inc = jnp.arccos(jnp.clip(cos_i, 0.0, 1.0))
        coeff_map = {
            name: coeff
            for (name, order), coeff in zip(
                [spec for spec in HARMONICA_ODD_COEFF_SPECS if spec[1] <= highest_order],
                odd_args,
            )
        }
        r = _build_harmonica_r_vector(rors, coeff_map, highest_order)
        # harmonica returns normalised flux (1.0 out of transit).
        # Subtract 1.0 to match jaxoplanet signal convention.
        return harmonica_transit_power2_ld(
            t, t0, period, a_rs, inc, ecc, omega,
            c=c_ld, alpha=alpha_ld, r=r,
        ) - 1.0

    vmap_args = [
        periods,
        t0s,
        rorss,
        a_rss,
        eccs,
        omegas,
        cos_is,
        *[odd_coeffs[name] for name, order in HARMONICA_ODD_COEFF_SPECS if order <= highest_order],
    ]
    batched_lcs = jax.vmap(get_lc)(*vmap_args)
    return jnp.sum(batched_lcs, axis=0)


def compute_transit_model_auto(params, t):
    """Dispatch to the harmonica or spherical transit model based on params."""
    if uses_harmonica_params(params):
        return compute_transit_model_harmonica(params, t)
    return compute_transit_model(params, t)


def get_I_power2(c, alpha, u):
    return 1 - c*(1-jnp.power(u,alpha))


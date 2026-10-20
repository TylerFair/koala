import jax
import jax.numpy as jnp
import numpy as np

# Load transit_grid eagerly: its module-level power-2 projection calls
# np.linalg.pinv, which fails if the first import happens inside a jit trace.
from . import transit_grid as _transit_grid  # noqa: F401


_TRANSIT_WINDOW_INDEX_KEY = "_transit_window_indices"
_JAXOPLANET_KERNEL_KEY = "_jaxoplanet_kernel"
_LD_PROFILE_KEY = "_ld_profile"
_TRANSIT_PHASE_OFFSETS_KEY = "_transit_phase_offsets"
_TRANSIT_PHASE_MASK_KEY = "_transit_phase_mask"
def resolve_jaxoplanet_kernel(kernel="auto", *, ld_profile=None, degree=None,
                              keplerian=False):
    """Resolve the retained stock/streamed kernel routes."""
    kernel = str(kernel).lower()
    if kernel not in {"auto", "stock", "streamed", "quadratic_contact_safe"}:
        raise ValueError(
            "jaxoplanet_kernel must be one of {'auto', 'stock', 'streamed', 'quadratic_contact_safe'}; "
            f"received {kernel!r}."
        )
    if kernel == 'auto' and degree in {0, 1, 2} and ld_profile in {None, 'quadratic'}:
        return 'quadratic_contact_safe'
    if kernel == "quadratic_contact_safe":
        if ld_profile not in {None, "quadratic"} or degree not in {0, 1, 2}:
            raise ValueError("quadratic_contact_safe requires a uniform, linear, or quadratic limb law.")
        return kernel
    supported = not keplerian and ld_profile == "power2" and degree == 12
    if not supported:
        return "stock"
    return "streamed" if kernel != "stock" else "stock"


def build_transit_phase_offsets(t, period, t0, duration):
    """Return channel-invariant phase offsets and duration masks.

    This is the exact phase convention used by jaxoplanet 0.1.0's
    ``TransitOrbit.relative_position``.  It is useful in spectroscopic models
    where the geometry is fixed across all wavelength channels.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    periods = jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64))
    t0s = jnp.atleast_1d(jnp.asarray(t0, dtype=jnp.float64))
    durations = jnp.atleast_1d(jnp.asarray(duration, dtype=jnp.float64))
    half_period = 0.5 * periods[:, None]
    dt = jnp.mod(t[None, :] - (t0s[:, None] - half_period), periods[:, None]) - half_period
    mask = jnp.fabs(dt) < 0.5 * durations[:, None]
    return dt, mask


def build_transit_window_indices(t, period, t0, duration):
    """Return the static union of fixed-duration transit windows.

    This helper is intentionally host-side: the returned integer array has a
    static shape and can therefore reduce the amount of compiled light-curve
    work. The caller supplies an envelope covering every allowed contact
    time, not the duration at just the reference radius. A conservative
    tolerance includes boundary cadences; the physical occultation kernel
    returns zero outside contact.
    """
    times = np.asarray(t, dtype=np.float64)
    periods = np.atleast_1d(np.asarray(period, dtype=np.float64))
    t0s = np.atleast_1d(np.asarray(t0, dtype=np.float64))
    durations = np.atleast_1d(np.asarray(duration, dtype=np.float64))

    if times.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    n_planets = max(periods.size, t0s.size, durations.size)

    def _broadcast_planet_array(name, value):
        if value.size == n_planets:
            return value
        if value.size == 1:
            return np.repeat(value, n_planets)
        raise ValueError(
            f"{name} must contain one value or one value per planet; "
            f"received {value.size} values for {n_planets} planets."
        )

    periods = _broadcast_planet_array("period", periods)
    t0s = _broadcast_planet_array("t0", t0s)
    durations = _broadcast_planet_array("duration", durations)
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("period values must be finite and positive.")
    if np.any(~np.isfinite(t0s)):
        raise ValueError("t0 values must be finite.")
    if np.any(~np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("duration values must be finite and non-negative.")

    active = np.zeros(times.shape, dtype=bool)
    scale = max(
        1.0,
        float(np.max(np.abs(times))) if times.size else 1.0,
        float(np.max(np.abs(t0s))),
        float(np.max(np.abs(periods))),
    )
    tolerance = 8.0 * np.finfo(np.float64).eps * scale
    for planet_period, planet_t0, planet_duration in zip(
        periods, t0s, durations
    ):
        half_period = 0.5 * planet_period
        dt = np.mod(times - (planet_t0 - half_period), planet_period) - half_period
        active |= np.abs(dt) <= 0.5 * planet_duration + tolerance

    return np.flatnonzero(active).astype(np.int32, copy=False)


def _safe_separation(squared):
    """Euclidean separation with the symmetric zero derivative at the origin."""
    positive = squared > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def orbit_separation(t, period, t0, a_rs, b, ecc=0.0, omega=0.0, *, circular=False):
    """Exact Keplerian projected distance and foreground mask (angles in radians).

    t0 is inferior conjunction, f=pi/2-omega, matching batman/jaxoplanet.
    The orbit contains no occulting radius and is shared across wavelengths.
    circular=True is for an eccentricity fixed at zero: the true anomaly is
    the mean anomaly, so the Kepler solve is skipped.
    """
    from jaxoplanet.core.kepler import kepler
    f0 = 0.5*jnp.pi - omega
    e0 = 2*jnp.arctan2(jnp.sqrt(1-ecc)*jnp.sin(f0/2),
                       jnp.sqrt(1+ecc)*jnp.cos(f0/2))
    mean = e0-ecc*jnp.sin(e0)+2*jnp.pi*(t-t0)/period
    sinf, cosf = (jnp.sin(mean), jnp.cos(mean)) if circular else kepler(mean, ecc)
    sinarg = sinf*jnp.cos(omega)+cosf*jnp.sin(omega)
    cosarg = cosf*jnp.cos(omega)-sinf*jnp.sin(omega)
    radius = a_rs*(1-ecc**2)/(1+ecc*cosf)
    cosi = b*(1+ecc*jnp.sin(omega))/(a_rs*(1-ecc**2))
    # sin(arg)^2 + cos(arg)^2 == 1. This equivalent form preserves
    # exactly constant separation for a face-on circular orbit (cosi == 1),
    # avoiding spurious partial contacts/derivatives from trig roundoff.
    distance = _safe_separation(radius**2*(cosi**2+(1-cosi**2)*cosarg**2))
    return distance, sinarg > 0


def fixed_orbit_separation(t, period, t0, a_rs, b, eccentricity=0., omega=90.):
    """Shared Keplerian separation; public omega is degrees, as in configs.

    The foreground sentinel preserves the fixed-separation interface used by
    spectroscopic callers. This delegates to the same orbit as direct fits.
    """
    distance, foreground = orbit_separation(
        jnp.asarray(t), period, t0, a_rs, b, eccentricity, jnp.deg2rad(omega))
    return jnp.where(foreground, distance, 1.e3)


def circular_orbit_separation(t, period, t0, a_rs, b):
    return fixed_orbit_separation(t, period, t0, a_rs, b)


def _flux_at_separation(params, distance, radius, kernel):
    """One exact flux implementation for direct and prepared physical orbits."""
    if params.get('_power2_exact', False):
        from .power2_exact import light_curve
        from .power2_table import _streamed_light_curve
        evaluate = _streamed_light_curve if distance.size > 5000 else light_curve
        return evaluate(params['c1'], params['c2'], distance, radius,
                        order=params.get('_power2_quadrature_order'))
    from jaxoplanet.core.limb_dark import light_curve
    coefficients = params['u']
    if kernel == 'quadratic_contact_safe':
        coefficients = jnp.pad(coefficients, (0, 2-int(jnp.size(coefficients))))
        bound = params.get('_quadratic_radius_bound')
        proven = bound is not None and 0 <= float(bound) < .99999
        from .quadratic_safe import fixed_orbit_evaluator
        return fixed_orbit_evaluator(not proven)(coefficients, radius, distance)
    return light_curve(coefficients, distance, radius, order=10)


def _fixed_separation_transit(params, separation, kernel):
    separation = jnp.asarray(separation)
    radius = jnp.ravel(params['rors'])[0]
    signal = _flux_at_separation(params, jnp.asarray(separation), radius, kernel)
    return jnp.where(separation < 1 + radius, signal, 0.)


def _compute_transit_model_keplerian(params, t, *, kernel="stock"):
    values = [jnp.atleast_1d(jnp.asarray(params[name], dtype=jnp.float64))
              for name in ('period', 't0', 'a_rs', 'b', 'rors')]
    ecc = jnp.broadcast_to(jnp.atleast_1d(params.get('ecc', 0.0)), values[0].shape)
    omega = jnp.broadcast_to(jnp.atleast_1d(params.get('omega', 0.0)), values[0].shape)
    circular = bool(params.get('_fixed_circular_orbit', False))
    def one(period, t0, a_rs, b, rors, e, w):
        distance, foreground = orbit_separation(t, period, t0, a_rs, b, e, w, circular=circular)
        return jnp.where(foreground, _flux_at_separation(params, distance, rors, kernel), 0.)
    return jnp.sum(jax.vmap(one)(*values, ecc, omega), axis=0)


def _prepared_transit(params, t, kernel):
    separation = jnp.asarray(params['_transit_separation'], dtype=jnp.float64)
    radii = jnp.atleast_1d(params['rors'])
    if separation.shape != (radii.size, t.size):
        raise ValueError('Prepared separation must have shape (n_planets, n_times).')
    indices = params.get(_TRANSIT_WINDOW_INDEX_KEY)
    if indices is not None:
        if len(indices) == 0:
            return jnp.zeros_like(t, dtype=jnp.float64)
        separation = separation[:, indices]
    def one(radius, z):
        signal = _flux_at_separation(params, z, radius, kernel)
        return jnp.where(z < 1 + radius, signal, 0.)
    signal = jnp.sum(jax.vmap(one)(radii, separation), axis=0)
    if indices is None:
        return signal
    return jnp.zeros_like(t, dtype=signal.dtype).at[indices].set(signal)


def duration_to_orbit(params):
    """Resolve duration once at its defining radius, never a channel radius.

    Standalone/white-light calls use their jointly sampled radius. Shared fits
    must supply a_rs or rors_reference explicitly before varying channel radii.
    """
    if 'a_rs' in params:
        return params
    if jnp.ndim(params['rors']) > 1 and 'rors_reference' not in params:
        raise ValueError('Shared duration evaluation requires a_rs or rors_reference; channel radii cannot define the orbit.')
    from ..harmonica.core import harmonica_a_rs_from_duration
    out = dict(params)
    out['a_rs'] = harmonica_a_rs_from_duration(
        jnp.asarray(out['period']), jnp.asarray(out['duration']),
        jnp.asarray(out['b']), jnp.asarray(out.get('rors_reference', out['rors'])),
        ecc=jnp.asarray(out.get('ecc', 0.)), omega=jnp.asarray(out.get('omega', 0.)))
    return out


def compute_transit_model(params, t, *, kernel=None, ld_profile=None):
    """
    Transit model for one or more planets using jaxoplanet.

    Expects params to contain 'period', 't0', 'b', 'rors', and either
    'duration' or ('a_rs', with optional 'ecc' and 'omega'). These should be
    arrays where the 0-th dimension is the planet index, except 'u' which is
    the limb darkening parameter vector.
    """
    surface_model = params.get("_surface_model", "transit")
    spots = params.get("_stellar_spots", ())
    surface_basis = params.get("_surface_basis")
    from ..configuration_support import validate_power2_support
    profile = (ld_profile if ld_profile is not None else params.get(_LD_PROFILE_KEY,
               'power2' if params.get('_power2_exact', False) else None))
    validate_power2_support(profile, model=surface_model, spots=spots)
    if params.get('_transit_separation') is not None and (
        surface_model != 'transit' or spots or surface_basis is not None
    ):
        raise ValueError('Prepared transit separations cannot drive a surface/spot model.')
    if surface_basis is not None:
        from .surface_basis import (
            compute_emission_basis_model,
            compute_spot_basis_model,
        )
        if hasattr(surface_basis, "uniform"):
            return compute_emission_basis_model(params, surface_basis)
        return compute_spot_basis_model(params, surface_basis)
    if params.get("_transit_separation") is None:
        params = duration_to_orbit(params)
    if surface_model != "transit" or spots:
        if params.get("_transit_separation") is not None:
            raise ValueError("Prepared transit separations cannot drive a surface/spot model.")
        if "a_rs" not in params:
            raise ValueError(
                "Eclipse, phase-curve, and stellar-spot evaluation requires a_rs geometry."
            )
        if "u" not in params:
            raise ValueError("JAXoplanet surface evaluation requires params['u'].")
        from .surface import compute_surface_model
        return compute_surface_model(
            params, t, model=surface_model, spots=spots
        )

    requested_kernel = params.get(_JAXOPLANET_KERNEL_KEY, "auto") if kernel is None else kernel
    profile = params.get(_LD_PROFILE_KEY) if ld_profile is None else ld_profile
    degree = int(jnp.shape(params["u"])[-1]) if "u" in params else None
    selected_kernel = resolve_jaxoplanet_kernel(
        requested_kernel,
        ld_profile=profile,
        degree=degree,
        keplerian="a_rs" in params,
    )

    if "u" not in params:
        raise ValueError("Keplerian transit evaluation requires params['u'].")
    if params.get("_transit_separation") is not None:
        return _prepared_transit(params, t, selected_kernel)
    indices = params.get(_TRANSIT_WINDOW_INDEX_KEY)
    if indices is None:
        return _compute_transit_model_keplerian(params, t, kernel=selected_kernel)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    if indices.ndim != 1:
        raise ValueError("Transit-window indices must be one-dimensional.")
    if indices.shape[0] == 0:
        return jnp.zeros_like(t, dtype=jnp.float64)
    signal = _compute_transit_model_keplerian(params, t[indices], kernel=selected_kernel)
    return jnp.zeros_like(t, dtype=signal.dtype).at[indices].set(signal)


def compute_transit_model_window(params, t):
    """Evaluate and return only the statically selected transit cadences."""
    indices = params.get(_TRANSIT_WINDOW_INDEX_KEY)
    if indices is None:
        raise ValueError("A static transit window is required.")
    indices = jnp.asarray(indices, dtype=jnp.int32)
    active_params = dict(params)
    active_params.pop(_TRANSIT_WINDOW_INDEX_KEY, None)
    if active_params.get("_transit_separation") is not None:
        active_params["_transit_separation"] = active_params["_transit_separation"][:, indices]
    return compute_transit_model(active_params, t[indices])

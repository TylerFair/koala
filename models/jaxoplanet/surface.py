"""Physical eclipse, phase-curve, and spotted-star light curves.

The public functions in this module deliberately use the same dimensionless
geometry as the transit fitter: stellar radius is one, ``a_rs`` is in stellar
radii, flux ratios are fractions, angles are radians, and times share the unit
of ``period``.  Returned model values are additive relative flux (total flux
divided by the unspotted stellar flux, minus one).

This module requires jaxoplanet's experimental ``starry`` implementation.
The imports are local so ordinary transit-only use does not pay its relatively
large import/compilation cost.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=64)
def _unit_spot_template(degree: int, radius: float, latitude: float, longitude: float):
    """Precompute the fixed harmonic geometry; spot contrast is linear."""
    import jax
    from jaxoplanet.starry.ylm import ylm_spot

    # This helper can first be reached while an outer NumPyro model is being
    # traced. Force constant evaluation here and cache a host array so a JAX
    # tracer can never escape through lru_cache into a later compilation.
    with jax.ensure_compile_time_eval():
        unit_spot = ylm_spot(degree)(1.0, radius, latitude, longitude).todense()
        baseline = jnp.zeros_like(unit_spot).at[0].set(1.0)
        return np.asarray(unit_spot - baseline)


def _one(value: Any, name: str):
    """Return a scalar while accepting the fitters' length-one arrays."""
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.size != 1:
        raise ValueError(
            f"{name} must contain one value; physical surface models currently "
            "support one planet at a time."
        )
    return jnp.reshape(value, ())


def build_keplerian_system(
    *,
    period,
    t0,
    a_rs,
    b,
    rors,
    ecc=0.0,
    omega=0.0,
    central_surface=None,
    planet_surface=None,
):
    """Build a physically consistent, effectively massless-planet system.

    JAXoplanet 0.1 requires exactly one of period and semimajor axis on a
    ``Body``.  Inferring the stellar mass from the observed ``period`` and
    ``a_rs`` first lets the body be specified by semimajor axis while retaining
    both observables exactly.  The planet mass is fixed to zero, as is
    appropriate for photometric geometry.
    """
    from jaxoplanet.orbits.keplerian import Body, Central
    from jaxoplanet.starry.orbit import SurfaceSystem

    period = _one(period, "period")
    t0 = _one(t0, "t0")
    a_rs = _one(a_rs, "a_rs")
    b = _one(b, "b")
    rors = _one(rors, "rors")
    ecc = _one(ecc, "ecc")
    omega = _one(omega, "omega")

    central = Central.from_orbital_properties(
        period=period,
        semimajor=a_rs,
        radius=1.0,
        body_mass=0.0,
    )
    body = Body(
        time_transit=t0,
        semimajor=a_rs,
        impact_param=b,
        eccentricity=ecc,
        omega_peri=omega,
        mass=0.0,
        radius=rors,
    )
    return SurfaceSystem(central, central_surface).add_body(body, planet_surface)


def build_spotted_stellar_surface(
    spots: Sequence[Mapping[str, Any]] | None,
    *,
    rotation_period,
    u=(),
    inclination=0.5 * np.pi,
    phase=0.0,
    degree: int = 8,
    contrasts=None,
):
    """Build a limb-darkened stellar surface from simple circular spots.

    Each spot mapping contains ``latitude``, ``longitude``, ``radius``, and
    ``contrast`` in radians/fractional units.  A contrast of zero is invisible
    and one makes the spot center dark.  ``contrasts`` optionally replaces all
    configured contrasts, which keeps fixed spot geometry convenient while
    sampling their contrasts.

    JAXoplanet's ``ylm_spot`` is a smooth, finite spherical-harmonic
    approximation to a circular spot. Multiple maps are added linearly.
    Overlapping high-contrast spots and ringing near very small spot edges can
    make the reconstructed local intensity negative even when each individual
    contrast lies in [0, 1]. Avoid overlapping spots, inspect the rendered map,
    and increase ``degree`` when resolving a small spot matters scientifically.
    """
    from jaxoplanet.starry import Surface, Ylm

    if int(degree) != degree or degree < 1:
        raise ValueError("spot degree must be a positive integer.")
    spots = () if spots is None else tuple(spots)
    if contrasts is not None:
        contrasts = jnp.atleast_1d(jnp.asarray(contrasts, dtype=jnp.float64))
        if contrasts.size != len(spots):
            raise ValueError(
                "stellar_spot_contrast must contain one value per configured spot."
            )

    if not spots:
        u_array = jnp.atleast_1d(jnp.asarray(u))
        return Surface(
            y=Ylm(),
            inc=inclination,
            u=tuple(u_array),
            period=rotation_period,
            phase=phase,
            amplitude=1.0,
            normalize=False,
        )

    size = (int(degree) + 1) ** 2
    coeffs = jnp.zeros(size, dtype=jnp.float64).at[0].set(1.0)
    for index, spot in enumerate(spots):
        missing = {"latitude", "longitude", "radius", "contrast"} - set(spot)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"stellar spot {index} is missing: {names}.")
        contrast = spot["contrast"] if contrasts is None else contrasts[index]
        try:
            template = _unit_spot_template(
                int(degree),
                float(spot["radius"]),
                float(spot["latitude"]),
                float(spot["longitude"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stellar spot latitude, longitude, and radius must be fixed "
                "numeric values; contrast is the inferred spot parameter."
            ) from exc
        # ylm_spot is exactly linear in contrast. Cache the unit-contrast
        # geometry so repeated likelihood traces only scale fixed coefficients.
        coeffs = coeffs + jnp.asarray(contrast) * template

    u_array = jnp.atleast_1d(jnp.asarray(u))
    return Surface(
        y=Ylm.from_dense(coeffs, normalize=False),
        inc=inclination,
        u=tuple(u_array),
        period=rotation_period,
        phase=phase,
        amplitude=1.0,
        normalize=False,
    )


def phase_curve_angle(t, period, t0):
    """Orbital phase angle ``phi = 2 pi (t - t_sec) / P`` with ``t_sec = t0 + P/2``.

    ``phi`` is zero at (circular-orbit) secondary eclipse and ``pi`` at transit.
    """
    period = _one(period, "period")
    t0 = _one(t0, "t0")
    return 2.0 * jnp.pi * (jnp.asarray(t) - t0) / period - jnp.pi


def sinusoid_phase_flux(phi, eclipse_depth, phase_c1, phase_d1):
    """Unocculted planet flux ``L(phi) = f_d + C1 (cos phi - 1) + D1 sin phi``.

    ``f_d = eclipse_depth`` is the planet flux at secondary eclipse (phi = 0),
    so it is also the eclipse depth. The nightside flux (phi = pi) is
    ``f_d - 2 C1``. No positivity constraint is imposed.
    """
    return (
        jnp.asarray(eclipse_depth)
        + jnp.asarray(phase_c1) * (jnp.cos(phi) - 1.0)
        + jnp.asarray(phase_d1) * jnp.sin(phi)
    )


def phase_curve_derived_quantities(eclipse_depth, phase_c1, phase_d1):
    """Derived sinusoidal phase-curve quantities (fractional flux, degrees).

    ``nightside_flux = f_d - 2 C1`` and ``phase_amplitude = sqrt(C1^2 + D1^2)``.
    ``C1 cos phi + D1 sin phi`` peaks at ``phi_max = atan2(D1, C1)``, i.e. a
    time ``phi_max P / 2 pi`` after secondary eclipse. ``phase_offset_deg``
    is ``-phi_max`` in degrees, so a POSITIVE offset means the flux maximum
    occurs BEFORE secondary eclipse (an eastward hotspot), which happens when
    ``D1 < 0``.
    """
    depth = jnp.asarray(eclipse_depth)
    c1 = jnp.asarray(phase_c1)
    d1 = jnp.asarray(phase_d1)
    return {
        "nightside_flux": depth - 2.0 * c1,
        "phase_amplitude": jnp.sqrt(c1 ** 2 + d1 ** 2),
        "phase_offset_deg": jnp.rad2deg(jnp.arctan2(-d1, c1)),
    }


def _planet_surface(params: Mapping[str, Any], model: str):
    from jaxoplanet.starry import Surface

    if model == "transit":
        # A zero-amplitude surface lets SurfaceSystem evaluate spot crossings
        # while the planet itself remains dark.
        return Surface(amplitude=0.0, normalize=False)
    if model == "eclipse":
        return Surface(
            amplitude=_one(params["eclipse_depth"], "eclipse_depth"),
            normalize=False,
        )
    # phase_curve: a uniform unit-flux disk. Its light-curve component is the
    # visible fraction E(t) of the planet disk, which is then scaled by the
    # sinusoidal planet flux L(phi) in compute_surface_model.
    return Surface(amplitude=1.0, normalize=False)


def _uniform_disk_visible_fraction(d, p):
    """Visible fraction of a uniform disk of radius ``p`` behind a unit star.

    ``d`` is the sky-projected centre separation in stellar radii. Uses the
    exact two-circle overlap area, with safe arguments so gradients stay
    finite at contact points.
    """
    d = jnp.asarray(d)
    full = d <= 1.0 - p
    none = d >= 1.0 + p
    partial = ~(full | none)
    ds = jnp.where(partial, d, 1.0)
    k0 = jnp.arccos(jnp.clip((ds**2 + p**2 - 1.0) / (2.0 * ds * p), -1.0, 1.0))
    k1 = jnp.arccos(jnp.clip((ds**2 + 1.0 - p**2) / (2.0 * ds), -1.0, 1.0))
    root = jnp.where(
        partial,
        (-ds + p + 1.0) * (ds + p - 1.0) * (ds - p + 1.0) * (ds + p + 1.0),
        1.0,
    )
    overlap = p**2 * k0 + k1 - 0.5 * jnp.sqrt(jnp.maximum(root, 1e-300))
    hidden = jnp.where(full, 1.0, jnp.where(none, 0.0, overlap / (jnp.pi * p**2)))
    return 1.0 - hidden


def _closed_form_emission_model(params, t, model):
    """Quadratic-LD transit plus uniform-disk eclipse without starry.

    Exactly the native model for an unspotted star with at most quadratic
    limb darkening: the transit uses JAXoplanet's analytic quadratic kernel
    and ``E(t)`` the analytic two-circle overlap, both on the same Keplerian
    orbit as :func:`build_keplerian_system`.
    """
    from jaxoplanet.light_curves import limb_dark_light_curve
    from jaxoplanet.orbits.keplerian import Body, Central, System

    a_rs = _one(params["a_rs"], "a_rs")
    central = Central.from_orbital_properties(
        period=_one(params["period"], "period"), semimajor=a_rs,
        radius=1.0, body_mass=0.0,
    )
    orbit = System(central).add_body(Body(
        time_transit=_one(params["t0"], "t0"), semimajor=a_rs,
        impact_param=_one(params["b"], "b"),
        eccentricity=_one(params.get("ecc", 0.0), "ecc"),
        omega_peri=_one(params.get("omega", 0.0), "omega"),
        mass=0.0, radius=_one(params["rors"], "rors"),
    ))
    t = jnp.asarray(t)
    u = tuple(jnp.ravel(jnp.asarray(params.get("u", ()), dtype=jnp.float64)))
    transit = limb_dark_light_curve(orbit, *u)(t)[..., 0] if u else (
        limb_dark_light_curve(orbit)(t)[..., 0]
    )
    x, y, z = orbit.relative_position(t)
    x, y, z = x[0], y[0], z[0]  # (body, time) -> time
    rors = _one(params["rors"], "rors")
    visible = jnp.where(
        z < 0.0,
        _uniform_disk_visible_fraction(jnp.sqrt(x**2 + y**2), rors),
        1.0,
    )
    if model == "eclipse":
        return transit + _one(params["eclipse_depth"], "eclipse_depth") * visible
    phi = phase_curve_angle(t, params["period"], params["t0"])
    planet_flux = sinusoid_phase_flux(
        phi,
        _one(params["eclipse_depth"], "eclipse_depth"),
        _one(params["phase_c1"], "phase_c1"),
        _one(params["phase_d1"], "phase_d1"),
    )
    return transit + planet_flux * visible


def compute_surface_model(
    params: Mapping[str, Any],
    t,
    *,
    model: str = "phase_curve",
    spots: Sequence[Mapping[str, Any]] | None = None,
    spot_degree: int = 8,
    order: int = 20,
):
    """Evaluate a stellar-normalized eclipse or full phase curve.

    ``model='transit'`` uses a dark planet and is useful when spots are active.
    ``model='eclipse'`` uses a uniform planet with ``eclipse_depth`` total
    flux. ``model='phase_curve'`` is the standard sinusoidal phase curve

        F(t) - 1 = transit(t) + L(phi) E(t),
        L(phi) = eclipse_depth + phase_c1 (cos phi - 1) + phase_d1 sin phi,

    with ``phi = 2 pi (t - t0 - P/2) / P`` (zero at circular-orbit secondary
    eclipse) and ``E(t)`` the visible fraction of a uniform planet disk taken
    from the same native eclipse geometry as ``model='eclipse'``. No
    positivity constraint is placed on ``L``. On eccentric orbits ``phi`` is
    still referenced to ``t0 + P/2`` (mean motion), not the true eclipse time.
    Stellar spot longitude and ``stellar_phase`` are defined at transit epoch
    ``t0`` rather than at the absolute time origin.

    The returned array is total system flux minus one, matching the additive
    signal convention of :func:`models.jaxoplanet.core.compute_transit_model`.
    """
    from jaxoplanet.starry.light_curves import light_curve

    model = str(model).lower()
    if model not in {"transit", "eclipse", "phase_curve"}:
        raise ValueError(
            "surface model must be 'transit', 'eclipse', or 'phase_curve'."
        )
    if (model in {"eclipse", "phase_curve"} and not spots
            and params.get("stellar_spot_contrast") is None
            and jnp.size(jnp.asarray(params.get("u", ()))) <= 2):
        return _closed_form_emission_model(params, t, model)
    if int(order) != order or order < 1:
        raise ValueError("starry integration order must be a positive integer.")

    spot_contrasts = params.get("stellar_spot_contrast")
    rotation_period = params.get("stellar_rotation_period")
    if spots and rotation_period is None:
        raise ValueError(
            "stellar_rotation_period is required when stellar spots are configured."
        )
    if rotation_period is None:
        rotation_period = _one(params["period"], "period")

    rotation_period = _one(rotation_period, "stellar_rotation_period")
    stellar_phase = _one(params.get("stellar_phase", 0.0), "stellar_phase")
    stellar_phase = (
        stellar_phase
        - 2.0 * jnp.pi * _one(params["t0"], "t0") / rotation_period
    )

    star = build_spotted_stellar_surface(
        spots,
        rotation_period=rotation_period,
        u=params.get("u", ()),
        inclination=params.get("stellar_inclination", 0.5 * jnp.pi),
        phase=stellar_phase,
        degree=spot_degree,
        contrasts=spot_contrasts,
    )
    planet = _planet_surface(params, model)
    system = build_keplerian_system(
        period=params["period"],
        t0=params["t0"],
        a_rs=params["a_rs"],
        b=params["b"],
        rors=params["rors"],
        ecc=params.get("ecc", 0.0),
        omega=params.get("omega", 0.0),
        central_surface=star,
        planet_surface=planet,
    )
    components = light_curve(system, order=int(order))(jnp.asarray(t))
    if model == "phase_curve":
        # components[..., 0] is the (transited, possibly spotted) star and
        # components[..., 1] the visible fraction E(t) of a unit uniform disk.
        phi = phase_curve_angle(t, params["period"], params["t0"])
        planet_flux = sinusoid_phase_flux(
            phi,
            _one(params["eclipse_depth"], "eclipse_depth"),
            _one(params["phase_c1"], "phase_c1"),
            _one(params["phase_d1"], "phase_d1"),
        )
        return components[..., 0] + planet_flux * components[..., 1] - 1.0
    return jnp.sum(components, axis=-1) - 1.0


__all__ = [
    "build_keplerian_system",
    "build_spotted_stellar_surface",
    "compute_surface_model",
    "phase_curve_angle",
    "phase_curve_derived_quantities",
    "sinusoid_phase_flux",
]

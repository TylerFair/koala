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

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

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


def validate_phase_curve_fluxes(dayside_flux, nightside_flux) -> None:
    """Validate the physical range of the native degree-one brightness map.

    A non-negative real dipole map requires both fluxes to be non-negative and
    the brighter hemisphere's disk-integrated flux to be no more than five
    times the fainter hemisphere's flux.  This host-side helper is intended for
    configuration and prior validation before JAX tracing starts.
    """
    day = np.asarray(dayside_flux, dtype=float)
    night = np.asarray(nightside_flux, dtype=float)
    day, night = np.broadcast_arrays(day, night)
    if np.any(~np.isfinite(day)) or np.any(~np.isfinite(night)):
        raise ValueError("dayside_flux and nightside_flux must be finite.")
    if np.any(day < 0.0) or np.any(night < 0.0):
        raise ValueError("dayside_flux and nightside_flux must be non-negative.")
    faint = np.minimum(day, night)
    bright = np.maximum(day, night)
    if np.any(bright > 5.0 * faint):
        raise ValueError(
            "the degree-one phase map requires the brighter flux to be at most "
            "five times the fainter flux so surface intensity stays non-negative."
        )


def _planet_surface(params: Mapping[str, Any], model: str):
    from jaxoplanet.starry import Surface, Ylm

    period = _one(params["period"], "period")
    t0 = _one(params["t0"], "t0")
    if model == "transit":
        # A zero-amplitude surface lets SurfaceSystem evaluate spot crossings
        # while the planet itself remains dark.
        return Surface(amplitude=0.0, normalize=False)
    if model == "eclipse":
        return Surface(
            amplitude=_one(params["eclipse_depth"], "eclipse_depth"),
            normalize=False,
        )

    dayside = _one(params["dayside_flux"], "dayside_flux")
    nightside = _one(params.get("nightside_flux", 0.0), "nightside_flux")
    offset = _one(params.get("hotspot_offset", 0.0), "hotspot_offset")
    mean_flux = 0.5 * (dayside + nightside)
    difference = dayside - nightside
    # For a real degree-one Y(1,0) map the disk-integrated modulation is
    # (2/sqrt(3))*coefficient*cos(theta).  This coefficient therefore makes
    # the requested day/night values the exact unocculted extrema.
    dipole = jnp.where(
        mean_flux != 0.0,
        0.25 * jnp.sqrt(3.0) * difference / mean_flux,
        0.0,
    )
    y = Ylm({(0, 0): 1.0, (1, 0): dipole})
    # This surface stays non-negative for max(day, night) <= 5*min(day, night).
    # Builders enforce that bound when constructing sampled priors.
    # At t0 the nightside faces the observer. Positive hotspot_offset means
    # the observed maximum occurs after the nominal secondary-eclipse center.
    map_phase = jnp.pi - 2.0 * jnp.pi * t0 / period - offset
    return Surface(
        y=y,
        period=period,
        phase=map_phase,
        amplitude=mean_flux,
        normalize=False,
    )


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
    flux. ``model='phase_curve'`` uses a mean-motion-synchronous degree-one map whose
    unocculted extrema are ``dayside_flux`` and ``nightside_flux``.  Transits,
    secondary eclipses, eccentric timing, phase modulation, stellar rotation,
    and spot crossings are then evaluated together by JAXoplanet.

    The current planetary map is equator-on, independent of orbital impact
    parameter, so its day/night parameters retain an exact disk-integrated
    meaning. On eccentric orbits it rotates uniformly with the mean period;
    it is a low-order photometric brightness model, not a tidal-spin model.
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
    return jnp.sum(components, axis=-1) - 1.0


__all__ = [
    "build_keplerian_system",
    "build_spotted_stellar_surface",
    "compute_surface_model",
    "validate_phase_curve_fluxes",
]

"""Small, user-facing configuration adapter for JAXoplanet surface models."""

import math

import numpy as np


SURFACE_MODELS = frozenset({"transit", "eclipse", "phase_curve"})


def parse_surface_config(flags, planet, stellar, n_planets, parameter_specs=None,
                         param_method="a_rs"):
    """Normalize friendly ppm/degree inputs to the model's fraction/radian form.

    ``parameter_specs`` are the parsed orbital specifications; the geometry
    counts as fixed when t0, b, rprs, and duration/a_rs are all fixed.
    """
    from koala.config import geometry_is_fixed, parse_planet_surface_specs

    model = str(flags.get("light_curve_model", "transit")).strip().lower()
    if model not in SURFACE_MODELS:
        raise ValueError(
            "flags.light_curve_model must be 'transit', 'eclipse', or "
            f"'phase_curve'; got {model!r}."
        )
    if "fit_geometry" in flags:
        raise ValueError(
            "flags.fit_geometry has been removed: geometry is fixed when "
            "planet.t0, b, rprs, and duration/a_rs all have prior: fixed, "
            "and fitted otherwise."
        )
    if n_planets != 1 and (model != "transit" or stellar.get("spots")):
        raise ValueError(
            "JAXoplanet eclipse, phase-curve, and stellar-spot models currently "
            "support exactly one planet."
        )
    fit_geometry = (
        True if parameter_specs is None
        else not geometry_is_fixed(parameter_specs, param_method)
    )
    result = {"model": model, "spots": ()}
    # Pure transit fits retain their established geometry behavior.  This flag
    # controls geometry only when a planetary surface or stellar map is active.
    if model != "transit":
        result["fit_geometry"] = bool(fit_geometry)

    def flux(name, unit):
        specs = parse_planet_surface_specs(planet, n_planets, [f"{name}_{unit}"])[f"{name}_{unit}"]
        center = np.asarray([spec.value for spec in specs], dtype=float)
        width = np.asarray(
            [spec.sigma if spec.prior == "gaussian" else 0.0 for spec in specs],
            dtype=float,
        )
        if unit == "ppm" and np.any(center < 0.0):
            raise ValueError(f"planet.{name}_ppm must be >= 0.")
        result[name] = center
        result[f"{name}_prior_width"] = width
        result[f"{name}_spec"] = tuple(specs)

    if model == "eclipse":
        flux("eclipse_depth", "ppm")
    elif model == "phase_curve":
        flux("dayside_flux", "ppm")
        flux("nightside_flux", "ppm")
        inferred_flux = (
            (result["dayside_flux_prior_width"] > 0.0)
            | (result["nightside_flux_prior_width"] > 0.0)
        )
        if np.any(
            inferred_flux
            & (
                (result["dayside_flux"] <= 0.0)
                | (result["nightside_flux"] <= 0.0)
            )
        ):
            raise ValueError(
                "Inferred phase curves require strictly positive configured "
                "day- and nightside fluxes. Both may be zero only when both "
                "fluxes are fixed."
            )
        low = np.minimum(result["dayside_flux"], result["nightside_flux"])
        high = np.maximum(result["dayside_flux"], result["nightside_flux"])
        if np.any(~(((high == 0.0) & (low == 0.0)) | ((low > 0.0) & (high <= 5.0 * low)))):
            raise ValueError(
                "The phase map requires nonnegative day/night fluxes whose "
                "larger value is no more than five times the smaller value."
            )
        flux("hotspot_offset", "deg")

    raw_spots = stellar.get("spots", ())
    if raw_spots is None:
        raw_spots = ()
    if not isinstance(raw_spots, (list, tuple)):
        raise ValueError("stellar.spots must be a list of spot mappings.")
    spots = []
    for index, raw in enumerate(raw_spots):
        if not isinstance(raw, dict):
            raise ValueError(f"stellar.spots[{index}] must be a mapping.")
        missing = {
            name for name in ("latitude_deg", "longitude_deg", "radius_deg", "contrast")
            if name not in raw
        }
        if missing:
            raise ValueError(
                f"stellar.spots[{index}] is missing: {', '.join(sorted(missing))}."
            )
        latitude = float(raw["latitude_deg"])
        longitude = float(raw["longitude_deg"])
        radius = float(raw["radius_deg"])
        contrast = float(raw["contrast"])
        width = float(raw.get("contrast_prior_width", 0.0))
        if not all(map(math.isfinite, (latitude, longitude, radius, contrast, width))):
            raise ValueError(f"stellar.spots[{index}] values must be finite.")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"stellar.spots[{index}].latitude_deg must be in [-90, 90].")
        if not 0.0 < radius < 90.0:
            raise ValueError(f"stellar.spots[{index}].radius_deg must be in (0, 90).")
        if not 0.0 <= contrast <= 1.0 or width < 0.0:
            raise ValueError(
                f"stellar.spots[{index}] contrast must be in [0, 1] and its width >= 0."
            )
        spots.append({
            "latitude": math.radians(latitude),
            "longitude": math.radians(longitude),
            "radius": math.radians(radius),
            "contrast": contrast,
            "contrast_prior_width": width,
        })
    if spots:
        result.setdefault("fit_geometry", bool(fit_geometry))
        rotation = float(stellar.get("rotation_period", np.nan))
        if not math.isfinite(rotation) or rotation <= 0.0:
            raise ValueError(
                "stellar.rotation_period must be finite and positive when stellar.spots are used."
            )
        result["stellar_rotation_period"] = rotation
        result["spots"] = tuple(spots)
    return result

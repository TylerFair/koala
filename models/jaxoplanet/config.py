"""Small, user-facing configuration adapter for JAXoplanet surface models."""

import math

import numpy as np


SURFACE_MODELS = frozenset({"transit", "eclipse", "phase_curve"})


def _planet_values(planet, name, n_planets, default=None):
    if name not in planet:
        if default is None:
            raise ValueError(f"planet.{name} is required for this light-curve model.")
        value = default
    else:
        value = planet[name]
    values = np.atleast_1d(np.asarray(value, dtype=float))
    if values.size == 1 and n_planets > 1:
        values = np.repeat(values, n_planets)
    if values.size != n_planets or not np.all(np.isfinite(values)):
        raise ValueError(
            f"planet.{name} must be finite and scalar or length {n_planets}."
        )
    return values


def parse_surface_config(flags, planet, stellar, n_planets):
    """Normalize friendly ppm/degree inputs to the model's fraction/radian form."""
    model = str(flags.get("light_curve_model", "transit")).strip().lower()
    if model not in SURFACE_MODELS:
        raise ValueError(
            "flags.light_curve_model must be 'transit', 'eclipse', or "
            f"'phase_curve'; got {model!r}."
        )
    if n_planets != 1 and (model != "transit" or stellar.get("spots")):
        raise ValueError(
            "JAXoplanet eclipse, phase-curve, and stellar-spot models currently "
            "support exactly one planet."
        )
    fit_geometry = flags.get("fit_geometry", model != "eclipse")
    if not isinstance(fit_geometry, (bool, np.bool_)):
        raise ValueError("flags.fit_geometry must be true or false.")
    raw_spots_for_geometry = stellar.get("spots", ())
    has_spots_for_geometry = (
        isinstance(raw_spots_for_geometry, (list, tuple))
        and len(raw_spots_for_geometry) > 0
    )
    if (
        model == "transit"
        and not has_spots_for_geometry
        and "fit_geometry" in flags
        and not bool(fit_geometry)
    ):
        raise ValueError(
            "flags.fit_geometry: false is supported for eclipse, phase-curve, "
            "or stellar-spot fits; ordinary transit fits always infer geometry."
        )
    result = {"model": model, "spots": ()}
    # Pure transit fits retain their established geometry behavior.  This flag
    # controls geometry only when a planetary surface or stellar map is active.
    if model != "transit":
        result["fit_geometry"] = bool(fit_geometry)

    def flux(name, default=None):
        center = _planet_values(planet, f"{name}_ppm", n_planets, default)
        width = _planet_values(
            planet, f"{name}_prior_width_ppm", n_planets, 0.0
        )
        if np.any(center < 0.0) or np.any(width < 0.0):
            raise ValueError(f"planet.{name}_ppm and its prior width must be >= 0.")
        result[name] = center * 1e-6
        result[f"{name}_prior_width"] = width * 1e-6

    if model == "eclipse":
        flux("eclipse_depth")
    elif model == "phase_curve":
        flux("dayside_flux")
        flux("nightside_flux")
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
        offset = _planet_values(planet, "hotspot_offset_deg", n_planets, 0.0)
        offset_width = _planet_values(
            planet, "hotspot_offset_prior_width_deg", n_planets, 0.0
        )
        if np.any(offset_width < 0.0):
            raise ValueError("planet.hotspot_offset_prior_width_deg must be >= 0.")
        result["hotspot_offset"] = np.deg2rad(offset)
        result["hotspot_offset_prior_width"] = np.deg2rad(offset_width)

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

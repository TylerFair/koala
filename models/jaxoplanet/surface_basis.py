"""Exact fixed-geometry basis for fast stellar-spot inference.

JAXoplanet's spherical-harmonic flux is linear in map coefficients, and
``ylm_spot`` is linear in spot contrast.  With fixed orbit, limb darkening,
spot geometry, and time samples, a spotted transit can therefore be evaluated
exactly as a zero-contrast light curve plus one native unit-contrast template
per spot.  This removes the degree-eight starry graph from the sampler while
retaining the native model used to construct the templates.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .surface import compute_surface_model


class SpotLightCurveBasis(NamedTuple):
    """A JAX Pytree holding an additive baseline and spot differences.

    ``baseline`` has shape ``(n_times,)`` and ``differences`` has shape
    ``(n_spots, n_times)``.  Each difference is the native unit-contrast curve
    minus the native zero-contrast curve.
    """

    baseline: Any
    differences: Any


class EmissionLightCurveBasis(NamedTuple):
    """Exact fixed-geometry eclipse or phase-curve templates.

    ``baseline`` is the transit-only curve and ``uniform`` the visible fraction
    ``E(t)`` of a unit uniform planet disk. ``phase_cos = E (cos phi - 1)`` and
    ``phase_sin = E sin phi`` are ``None`` for an eclipse basis and arrays for
    a sinusoidal phase-curve basis, which is then exactly linear:
    ``baseline + f_d uniform + C1 phase_cos + D1 phase_sin``. All array fields
    have shape ``(n_times,)``.
    """

    baseline: Any
    uniform: Any
    phase_cos: Any | None
    phase_sin: Any | None


_FIXED_KEYS = (
    "period",
    "t0",
    "a_rs",
    "b",
    "rors",
    "u",
    "ecc",
    "omega",
    "stellar_rotation_period",
    "stellar_inclination",
    "stellar_phase",
)


def _host_fixed_params(
    params: Mapping[str, Any], *, require_stellar_rotation: bool
) -> dict[str, jax.Array]:
    required = {"period", "t0", "a_rs", "b", "rors", "u"}
    if require_stellar_rotation:
        required.add("stellar_rotation_period")
    missing = required - set(params)
    if missing:
        raise ValueError(
            "fixed spot-basis parameters are missing: " + ", ".join(sorted(missing))
        )
    fixed = {}
    for name in _FIXED_KEYS:
        if name not in params:
            continue
        value = params[name]
        if isinstance(value, jax.core.Tracer):
            raise ValueError(
                "spot-basis geometry and limb darkening must be fixed before JAX tracing."
            )
        try:
            host = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, jax.errors.TracerArrayConversionError) as exc:
            raise ValueError(
                "spot-basis geometry and limb darkening must be fixed numeric values."
            ) from exc
        if not np.all(np.isfinite(host)):
            raise ValueError(f"fixed spot-basis parameter {name!r} must be finite.")
        fixed[name] = jnp.asarray(host, dtype=jnp.float64)
    return fixed


def prepare_spot_light_curve_basis(
    fixed_params: Mapping[str, Any],
    t,
    *,
    spots: Sequence[Mapping[str, Any]],
    spot_degree: int = 8,
    order: int = 20,
) -> SpotLightCurveBasis:
    """Prepare exact native templates for a fixed-geometry spotted transit.

    This is deliberately a host-side preparation step and must run outside a
    NumPyro/JAX trace.  Orbit parameters, radius ratio, limb darkening, stellar
    rotation, stellar orientation, spot positions/radii, and the time grid are
    fixed. Only spot contrasts may vary when evaluating the resulting basis.
    The returned baseline already includes the spotless transit and follows the
    additive ``total_flux - 1`` convention.
    """
    model = fixed_params.get("_surface_model", "transit")
    if model != "transit":
        raise ValueError("the fixed spot basis currently supports transit mode only.")
    spots = tuple(spots)
    if not spots:
        raise ValueError("at least one stellar spot is required to prepare a spot basis.")
    fixed = _host_fixed_params(fixed_params, require_stellar_rotation=True)
    if isinstance(t, jax.core.Tracer):
        raise ValueError("spot-basis times must be fixed before JAX tracing.")
    times_host = np.asarray(t, dtype=np.float64)
    if times_host.ndim != 1 or not np.all(np.isfinite(times_host)):
        raise ValueError("spot-basis times must be a finite one-dimensional array.")
    times = jnp.asarray(times_host, dtype=jnp.float64)

    contrast_cases = jnp.concatenate(
        [jnp.zeros((1, len(spots))), jnp.eye(len(spots))], axis=0
    )

    def native_curve(contrasts):
        params = {**fixed, "stellar_spot_contrast": contrasts}
        return compute_surface_model(
            params,
            times,
            model="transit",
            spots=spots,
            spot_degree=spot_degree,
            order=order,
        )

    curves = jax.jit(jax.vmap(native_curve))(contrast_cases)
    curves = jax.block_until_ready(curves)
    baseline = curves[0]
    return SpotLightCurveBasis(
        baseline=baseline,
        differences=curves[1:] - baseline[None, :],
    )


def compute_spot_basis_model(
    params: Mapping[str, Any], basis: SpotLightCurveBasis
):
    """Evaluate a prepared spot basis using direct fractional contrasts."""
    if "stellar_spot_contrast" not in params:
        raise ValueError("params['stellar_spot_contrast'] is required for a spot basis.")
    contrasts = jnp.atleast_1d(jnp.asarray(params["stellar_spot_contrast"]))
    differences = jnp.asarray(basis.differences)
    if contrasts.shape[-1] != differences.shape[0]:
        raise ValueError(
            "stellar_spot_contrast must contain one value per prepared spot."
        )
    return jnp.asarray(basis.baseline) + jnp.matmul(contrasts, differences)


def prepare_emission_light_curve_basis(
    fixed_params: Mapping[str, Any],
    t,
    *,
    model: str,
    order: int = 20,
) -> EmissionLightCurveBasis:
    """Prepare exact native templates for fixed-geometry planet emission.

    Geometry, radius ratio, limb darkening, and the time grid must be fixed.
    Eclipse depth or the three phase-curve parameters remain free. Stellar
    spots are intentionally excluded because their product with an emission
    surface belongs to the full native model.
    """
    model = str(model).lower()
    if model not in {"eclipse", "phase_curve"}:
        raise ValueError("emission basis model must be 'eclipse' or 'phase_curve'.")
    if fixed_params.get("_stellar_spots"):
        raise ValueError("the fixed emission basis does not support stellar spots.")
    fixed = _host_fixed_params(fixed_params, require_stellar_rotation=False)
    if isinstance(t, jax.core.Tracer):
        raise ValueError("emission-basis times must be fixed before JAX tracing.")
    times_host = np.asarray(t, dtype=np.float64)
    if times_host.ndim != 1 or not np.all(np.isfinite(times_host)):
        raise ValueError("emission-basis times must be a finite one-dimensional array.")
    times = jnp.asarray(times_host, dtype=jnp.float64)

    def native_curve(depth):
        return compute_surface_model(
            {**fixed, "eclipse_depth": depth},
            times,
            model="eclipse",
            order=order,
        )

    curves = jax.jit(jax.vmap(native_curve))(jnp.array([0.0, 1.0]))
    curves = jax.block_until_ready(curves)
    baseline = curves[0]
    uniform = curves[1] - baseline
    if model == "eclipse":
        return EmissionLightCurveBasis(
            baseline=baseline, uniform=uniform, phase_cos=None, phase_sin=None,
        )

    from .surface import phase_curve_angle

    phi = phase_curve_angle(times, fixed["period"], fixed["t0"])
    return EmissionLightCurveBasis(
        baseline=baseline,
        uniform=uniform,
        phase_cos=uniform * (jnp.cos(phi) - 1.0),
        phase_sin=uniform * jnp.sin(phi),
    )


def compute_emission_basis_model(
    params: Mapping[str, Any], basis: EmissionLightCurveBasis
):
    """Evaluate a prepared eclipse or phase basis in additive-flux units."""
    baseline = jnp.asarray(basis.baseline)
    uniform = jnp.asarray(basis.uniform)
    if "eclipse_depth" not in params:
        raise ValueError("params['eclipse_depth'] is required for an emission basis.")
    model = baseline + jnp.asarray(params["eclipse_depth"]) * uniform
    if basis.phase_cos is None:
        return model

    missing = {"phase_c1", "phase_d1"} - set(params)
    if missing:
        raise ValueError(
            "phase-basis parameters are missing: " + ", ".join(sorted(missing))
        )
    return (
        model
        + jnp.asarray(params["phase_c1"]) * jnp.asarray(basis.phase_cos)
        + jnp.asarray(params["phase_d1"]) * jnp.asarray(basis.phase_sin)
    )


__all__ = [
    "EmissionLightCurveBasis",
    "SpotLightCurveBasis",
    "compute_emission_basis_model",
    "compute_spot_basis_model",
    "prepare_emission_light_curve_basis",
    "prepare_spot_light_curve_basis",
]

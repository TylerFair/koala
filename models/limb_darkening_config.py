"""Canonical public configuration for limb darkening."""

from __future__ import annotations

import math


LD_PROFILES = ("quadratic", "power2", "kipping")
# "kipping" is the quadratic law sampled in Kipping (2013) (q1, q2) coordinates.
LD_PRIORS = ("uniform", "gaussian", "sing", "stellarprior", "fixed", "fixed_u1")
GAUSSIAN_LD_WIDTH = 0.2


def _choice_error(field, value, choices):
    rendered = ", ".join(repr(choice) for choice in choices)
    return ValueError(f"flags.{field} must be one of: {rendered}; got {value!r}.")


def validate_ld_profile(value):
    """Return an exact canonical limb-darkening profile name."""
    if value not in LD_PROFILES:
        raise _choice_error("ld_profile", value, LD_PROFILES)
    return value


def validate_ld_prior(value, *, ld_profile):
    """Return an exact canonical prior name and validate its profile."""
    validate_ld_profile(ld_profile)
    if value not in LD_PRIORS:
        raise _choice_error("ld_prior", value, LD_PRIORS)
    if value == "fixed_u1" and ld_profile != "quadratic":
        raise ValueError("flags.ld_prior='fixed_u1' requires flags.ld_profile='quadratic'.")
    if value == "sing" and ld_profile != "quadratic":
        raise ValueError("flags.ld_prior='sing' requires flags.ld_profile='quadratic'.")
    if value == "stellarprior" and ld_profile not in {"quadratic", "power2"}:
        raise ValueError(
            "flags.ld_prior='stellarprior' requires flags.ld_profile='quadratic' or 'power2'."
        )
    return value


def resolve_ld_prior(value, *, ld_profile, has_stellar_uncertainties=False):
    """Resolve an omitted prior without accepting alternate public names."""
    validate_ld_profile(ld_profile)
    if value is None:
        value = (
            "stellarprior"
            if ld_profile == "power2" and has_stellar_uncertainties
            else "uniform" if ld_profile == "quadratic" else "gaussian"
        )
    return validate_ld_prior(value, ld_profile=ld_profile)


def validate_gaussian_ld_width(value=GAUSSIAN_LD_WIDTH):
    """Positive coefficient-space standard deviation for the Gaussian mode."""
    if isinstance(value, bool):
        raise ValueError('flags.ld_gaussian_sigma must be finite and positive.')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('flags.ld_gaussian_sigma must be finite and positive.') from None
    if not math.isfinite(value) or value <= 0.:
        raise ValueError('flags.ld_gaussian_sigma must be finite and positive.')
    return value


def validate_uniform_ld_bounds(value=(0.0, 1.0)):
    """Two finite increasing bounds for direct quadratic coefficients."""
    message = 'flags.ld_uniform_coefficient_bounds must contain two finite numbers with low < high.'
    try:
        if isinstance(value, (str, bytes)) or len(value) != 2 or any(isinstance(x, bool) for x in value):
            raise ValueError(message)
        low, high = map(float, value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError(message)
    return low, high

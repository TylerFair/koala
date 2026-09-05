"""Canonical public configuration for limb darkening."""

from __future__ import annotations


LD_PROFILES = ("quadratic", "power2")
LD_PRIORS = ("uniform", "gaussian", "sing", "stellarprior", "fixed")
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
    if value == "sing" and ld_profile != "quadratic":
        raise ValueError("flags.ld_prior='sing' requires flags.ld_profile='quadratic'.")
    if value == "stellarprior" and ld_profile != "power2":
        raise ValueError(
            "flags.ld_prior='stellarprior' requires flags.ld_profile='power2'."
        )
    return value


def resolve_ld_prior(value, *, ld_profile, has_stellar_uncertainties=False):
    """Resolve an omitted prior without accepting alternate public names."""
    validate_ld_profile(ld_profile)
    if value is None:
        value = (
            "stellarprior"
            if ld_profile == "power2" and has_stellar_uncertainties
            else "gaussian"
        )
    return validate_ld_prior(value, ld_profile=ld_profile)

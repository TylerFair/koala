"""Configuration loading, validation, and option resolution."""

from koala.console import debug, info

import difflib
import numbers
import warnings
from dataclasses import dataclass
import numpy as np
import yaml
import jax
import jax.numpy as jnp
from models.cadence_reduction import linear_spectro_trend_coefficient_names
from models.limb_darkening_config import resolve_ld_prior, validate_ld_profile
from models.trend_marginal import marginalized_trend_coefficient_names
from .constants import (
    KNOWN_FLAGS,
    PARAMETER_SPEC_PRIORS,
    PLANET_LEGACY_KEYS,
    PLANET_PARAMETERS_FIXED_ONLY,
    PLANET_PARAMETERS_GAUSSIAN_ONLY,
    PLANET_PARAMETER_KEYS,
    PLANET_SURFACE_PARAMETER_SCALES,
    TREND_PRIOR_DEFAULTS,
)

class UnknownFlagWarning(UserWarning):
    """A ``flags`` entry is not consumed by the current configuration API."""


@dataclass(frozen=True)
class ParameterSpec:
    """How one ``planet`` parameter enters the fit.

    Every parameter is written as a mapping with the same keys::

        {value: 0.45, prior: fixed}
        {value: 0.45, prior: uniform, low: 0.0, high: 1.0}
        {value: 0.45, prior: log_uniform, low: 0.01, high: 1.0}
        {value: 0.45, prior: gaussian, sigma: 0.05}
        {value: 0.45, prior: gaussian, sigma: 0.05, low: 0.0}   # truncated

    ``value`` is the fixed value or the starting point of a free parameter
    (and the centre of a gaussian). ``sigma`` belongs to ``gaussian``;
    ``low``/``high`` are required for ``uniform``/``log_uniform`` and
    optional truncation bounds for ``gaussian``.
    """

    name: str
    prior: str
    value: float
    sigma: float = None
    low: float = None
    high: float = None

    @property
    def free(self):
        return self.prior != 'fixed'

    @property
    def fixed(self):
        return self.prior == 'fixed'

    @property
    def center(self):
        """Representative value: the fixed value or the starting point."""
        return float(self.value)

    @property
    def bounded(self):
        return self.low is not None or self.high is not None

    def scaled(self, factor):
        """Return the same specification in different units."""
        factor = float(factor)

        def _s(x):
            return None if x is None else float(x) * factor

        return ParameterSpec(
            self.name, self.prior, _s(self.value), _s(self.sigma), _s(self.low), _s(self.high),
        )

    def describe(self):
        if self.fixed:
            return f"fixed at {self.value!r}"
        if self.prior == 'uniform':
            return f"uniform({self.low!r}, {self.high!r}), start {self.value!r}"
        if self.prior == 'log_uniform':
            return f"log-uniform({self.low!r}, {self.high!r}), start {self.value!r}"
        text = f"gaussian(mu={self.value!r}, sigma={self.sigma!r})"
        if self.bounded:
            text += f" truncated to [{self.low!r}, {self.high!r}]"
        return text


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _spec_number(key, label, value):
    if not _is_number(value):
        raise ValueError(
            f"planet.{key}: '{label}' must be a number, received {value!r}."
        )
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"planet.{key}: '{label}' must be finite.")
    return value


_SPEC_FORM = (
    "{value: <number>, prior: fixed | uniform | log_uniform | gaussian, "
    "sigma: <gaussian width>, low: <bound>, high: <bound>}"
)


def parse_parameter_spec(key, raw):
    """Parse one ``{value, prior, sigma, low, high}`` parameter mapping."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"planet.{key}: every planet parameter is written as a mapping "
            f"{_SPEC_FORM}; received {raw!r}. A fixed value is "
            f"{{value: {raw!r}, prior: fixed}}."
        )
    fields = {str(name).strip().lower(): value for name, value in raw.items()}
    if 'mode' in fields or 'mu' in fields:
        raise ValueError(
            f"planet.{key}: 'mode' and 'mu' are no longer accepted; use "
            f"{_SPEC_FORM} (the gaussian centre is 'value')."
        )
    unknown = set(fields) - {'value', 'prior', 'sigma', 'low', 'high'}
    if unknown:
        raise ValueError(
            f"planet.{key}: unknown keys {sorted(unknown)}; the form is {_SPEC_FORM}."
        )
    for required in ('value', 'prior'):
        if required not in fields:
            raise ValueError(
                f"planet.{key}: '{required}' is required; the form is {_SPEC_FORM}."
            )
    prior = str(fields['prior']).strip().lower()
    if prior not in PARAMETER_SPEC_PRIORS:
        raise ValueError(
            f"planet.{key}: unknown prior {fields['prior']!r}; expected one of "
            f"{list(PARAMETER_SPEC_PRIORS)}."
        )
    value = _spec_number(key, 'value', fields['value'])
    sigma = low = high = None
    if prior == 'fixed':
        extra = {'sigma', 'low', 'high'} & set(fields)
        if extra:
            raise ValueError(
                f"planet.{key}: a fixed parameter takes only 'value'; remove {sorted(extra)}.",
            )
        return ParameterSpec(key, 'fixed', value)
    if key in PLANET_PARAMETERS_FIXED_ONLY:
        raise ValueError(
            f"planet.{key} may only be fixed for now; write "
            f"{{value: {value!r}, prior: fixed}}."
        )
    if key in PLANET_PARAMETERS_GAUSSIAN_ONLY and prior != 'gaussian':
        raise ValueError(
            f"planet.{key} supports only 'fixed' or 'gaussian' priors."
        )
    if prior == 'gaussian':
        if 'sigma' not in fields:
            raise ValueError(f"planet.{key}: a gaussian prior needs 'sigma'.")
        sigma = _spec_number(key, 'sigma', fields['sigma'])
        if sigma <= 0.0:
            raise ValueError(f"planet.{key}: 'sigma' must be > 0, received {sigma!r}.")
        if 'low' in fields:
            low = _spec_number(key, 'low', fields['low'])
        if 'high' in fields:
            high = _spec_number(key, 'high', fields['high'])
    else:
        if 'sigma' in fields:
            raise ValueError(f"planet.{key}: 'sigma' only applies to a gaussian prior.")
        for bound in ('low', 'high'):
            if bound not in fields:
                raise ValueError(
                    f"planet.{key}: a {prior} prior needs both 'low' and 'high'."
                )
        low = _spec_number(key, 'low', fields['low'])
        high = _spec_number(key, 'high', fields['high'])
        if prior == 'log_uniform' and low <= 0.0:
            raise ValueError(
                f"planet.{key}: a log_uniform prior needs low > 0, received {low!r}."
            )
    if low is not None and high is not None and low >= high:
        raise ValueError(
            f'planet.{key}: bounds require low < high, received low={low!r}, high={high!r}.',
        )
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(
            f"planet.{key}: 'value' {value!r} must lie inside [{low!r}, {high!r}].",
        )
    return ParameterSpec(key, prior, value, sigma, low, high)


def _parse_parameter_entries(key, raw):
    """Return one specification per planet for a ``planet`` block entry."""
    if isinstance(raw, (list, tuple)):
        if not raw or not all(isinstance(item, dict) for item in raw):
            raise ValueError(
                f"planet.{key}: a list is only used for several planets and "
                f"must hold one {{value, prior, ...}} mapping per planet; "
                f"received {list(raw)!r}."
            )
        return tuple(parse_parameter_spec(key, item) for item in raw)
    return (parse_parameter_spec(key, raw),)


def reject_legacy_planet_keys(planet_cfg):
    """Raise for removed ``planet`` keys, naming the replacement."""
    for key, replacement in PLANET_LEGACY_KEYS.items():
        if key in planet_cfg:
            raise ValueError(
                f"planet.{key} has been removed; write {replacement} instead."
            )


def parse_planet_parameter_specs(planet_cfg):
    """Parse every orbital entry of ``planet`` into per-planet specifications.

    Returns ``{name: (ParameterSpec, ...)}`` with one entry per planet for
    each of ``PLANET_PARAMETER_KEYS`` present in the block (``ecc`` and
    ``omega`` default to fixed zero).  Single mappings are broadcast to the
    number of planets, which is taken from ``period``. ``eclipse_time`` may
    replace ``t0`` for circular orbits; the derived ``t0`` specification is
    then added (``t0 = eclipse_time - period / 2``).
    """
    reject_legacy_planet_keys(planet_cfg)
    entries = {}
    for key in PLANET_PARAMETER_KEYS:
        if key in planet_cfg:
            entries[key] = _parse_parameter_entries(key, planet_cfg[key])
    for key in ('ecc', 'omega'):
        entries.setdefault(key, (ParameterSpec(key, 'fixed', 0.0),))
    if 'period' not in entries:
        raise KeyError("'planet.period' is required.")
    n_planets = len(entries['period'])
    resolved = {}
    for key, specs in entries.items():
        if len(specs) == 1 and n_planets > 1:
            specs = specs * n_planets
        if len(specs) != n_planets:
            raise ValueError(
                f"planet.{key} must be one mapping or a list of {n_planets}, "
                f"got {len(specs)} entries."
            )
        resolved[key] = tuple(specs)
    if 'eclipse_time' in resolved:
        if 't0' in resolved:
            raise ValueError(
                'planet.t0 and planet.eclipse_time are alternatives; give exactly one of them.',
            )
        resolved['t0'] = tuple(
            _t0_spec_from_eclipse_time(spec, period, ecc)
            for spec, period, ecc in zip(
                resolved['eclipse_time'], resolved['period'], resolved['ecc']
            )
        )
    elif 't0' not in resolved:
        raise KeyError("'planet.t0' (or 'planet.eclipse_time') is required.")
    return resolved


def check_ephemeris_covers_data(time, specs, light_curve_model='transit'):
    """Fail early when no allowed transit/eclipse time falls on the data.

    A wrong time system is the usual cause: Koala works in BJD_TDB - 2400000,
    while papers also quote full BJD or BMJD_TDB = BJD_TDB - 2400000.5.
    With a fixed period the model is periodic, so such an epoch silently puts
    the model event at the wrong orbital phase and the fitted depth goes to zero.
    """
    time = np.asarray(time, dtype=float)
    t_lo, t_hi = float(np.min(time)), float(np.max(time))
    event = 'eclipse' if light_curve_model == 'eclipse' else 'transit'
    shift = 0.5 if event == 'eclipse' else 0.0
    for i, (t0, period) in enumerate(zip(specs['t0'], specs['period'])):
        if abs(t0.value - 0.5 * (t_lo + t_hi)) > 1.0e5:
            raise ValueError(
                f"planet.t0 (or eclipse_time) = {t0.value + shift * period.value:.5f} "
                f"looks like a full BJD, but koala works in BJD_TDB - 2400000 "
                f"(the data run from {t_lo:.3f} to {t_hi:.3f}). Subtract 2400000."
            )
        if light_curve_model == 'phase_curve' or period.free:
            continue
        if t0.prior == 'gaussian':
            lo, hi = t0.value - 5 * t0.sigma, t0.value + 5 * t0.sigma
        else:
            lo = t0.value if t0.low is None else t0.low
            hi = t0.value if t0.high is None else t0.high
        p = float(period.value)
        # 0.05 d of slack keeps partial events whose mid-time lies just outside.
        lo, hi = lo + shift * p - 0.05, hi + shift * p + 0.05
        def covers(offset):
            n = np.ceil((t_lo - hi - offset) / p)  # first window ending after t_lo
            return lo + offset + n * p <= t_hi

        if not covers(0.0):
            hint = (" Shifting the ephemeris by 0.5 d would fit: a BMJD_TDB "
                    "(BJD - 2400000.5) time was probably used instead of BJD_TDB - 2400000; "
                    "add 0.5."
                    if covers(0.5) else
                    " Shifting the ephemeris by 0.5 d would fit: check the time system "
                    "(koala works in BJD_TDB - 2400000)."
                    if covers(-0.5) else "")
            raise ValueError(
                f"Planet {i}: no {event} time allowed by the planet.t0/eclipse_time "
                f"prior ({lo:.5f} to {hi:.5f} plus whole periods) falls inside the "
                f"data ({t_lo:.5f} to {t_hi:.5f}, BJD_TDB - 2400000).{hint}"
            )


def _t0_spec_from_eclipse_time(spec, period_spec, ecc_spec):
    """Shift an ``eclipse_time`` specification to the primary-transit epoch.

    Valid for circular orbits only: t0 = eclipse_time - period / 2.
    """
    if not period_spec.fixed:
        raise ValueError(
            "planet.eclipse_time requires a fixed planet.period."
        )
    if float(ecc_spec.value) != 0.0:
        raise ValueError(
            "planet.eclipse_time assumes a circular orbit; set planet.ecc to "
            "0 or give planet.t0 directly."
        )
    if spec.prior == 'log_uniform':
        raise ValueError(
            'planet.eclipse_time does not support a log_uniform prior: shifting '
            'the time origin changes that density. Give the intended prior on '
            'planet.t0 explicitly, or use fixed, uniform, or gaussian eclipse_time.'
        )
    shift = -0.5 * float(period_spec.value)

    def _s(x):
        return None if x is None else float(x) + shift

    return ParameterSpec(
        't0', spec.prior, _s(spec.value), spec.sigma, _s(spec.low), _s(spec.high),
    )


def parse_planet_surface_specs(planet_cfg, n_planets, keys):
    """Parse emission entries of ``planet`` (ppm/degree units) into model units.

    ``keys`` are the ``PLANET_SURFACE_PARAMETER_SCALES`` names required by the
    active light-curve model. Returns ``{name: (ParameterSpec, ...)}`` with
    the specification scaled to fractional flux or radians.
    """
    reject_legacy_planet_keys(planet_cfg)
    resolved = {}
    for key in keys:
        if key not in planet_cfg:
            raise ValueError(
                f"planet.{key} is required for this light-curve model; write "
                f"{key}: {_SPEC_FORM}."
            )
        specs = _parse_parameter_entries(key, planet_cfg[key])
        if len(specs) == 1 and n_planets > 1:
            specs = specs * n_planets
        if len(specs) != n_planets:
            raise ValueError(
                f"planet.{key} must be one mapping or a list of {n_planets}, "
                f"got {len(specs)} entries."
            )
        scale = PLANET_SURFACE_PARAMETER_SCALES[key]
        resolved[key] = tuple(spec.scaled(scale) for spec in specs)
    return resolved


def geometry_is_fixed(specs, param_method):
    """True when every parameter that changes the orbital light curve is fixed."""
    names = ('period', 't0', 'b', 'rprs', 'duration' if param_method == 'duration' else 'a_rs')
    return all(
        spec.fixed for name in names if name in specs for spec in specs[name]
    )


def planet_parameter_centers(specs, key, default=None):
    """Return the representative value of ``key`` for every planet."""
    if key not in specs:
        if default is None:
            raise KeyError(f"planet.{key} is not configured.")
        n_planets = len(next(iter(specs.values())))
        return np.full(n_planets, float(default), dtype=np.float64)
    return np.asarray([spec.center for spec in specs[key]], dtype=np.float64)


def describe_planet_parameter_specs(specs, title="Planet parameters"):
    """A small table, one row per configured parameter (and planet)."""
    rows = []
    for key, entries in specs.items():
        for index, spec in enumerate(entries):
            label = key if len(entries) == 1 else f"{key}[{index}]"
            rows.append((label, spec.describe()))
    width = max((len(label) for label, _ in rows), default=0)
    lines = [f"{title}:"]
    lines.extend(f"  {label.ljust(width)}  {text}" for label, text in rows)
    return lines


def parse_trend_priors(raw):
    """Parse the optional top-level ``trend_priors`` block.

    Each of ``c``, ``v``, ``A`` and ``tau`` may be given in the planet
    parameter form ``{prior, low, high, sigma, value}`` and replaces the
    built-in white-light prior for that site (``TREND_PRIOR_DEFAULTS``).
    ``value`` (the start) is optional: it defaults to the built-in start, or
    to the prior midpoint when that start lies outside the bounds. Returns
    ``{name: ParameterSpec}`` for the configured keys only, so an absent block
    leaves every prior, site name and start unchanged.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"trend_priors must be a mapping of {sorted(TREND_PRIOR_DEFAULTS)}; received {raw!r}.")
    unknown = set(raw) - set(TREND_PRIOR_DEFAULTS)
    if unknown:
        raise ValueError(
            f"trend_priors: unknown keys {sorted(unknown)}; expected any of {list(TREND_PRIOR_DEFAULTS)}."
        )
    specs = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"trend_priors.{key} must be a mapping {_SPEC_FORM}; received {entry!r}.")
        fields = {str(name).strip().lower(): value for name, value in entry.items()}
        if 'value' not in fields and str(fields.get('prior', '')).strip().lower() != 'fixed':
            start = TREND_PRIOR_DEFAULTS[key]['value']
            low, high = fields.get('low'), fields.get('high')
            if _is_number(low) and _is_number(high) and not (low < start < high):
                prior = str(fields.get('prior', '')).strip().lower()
                start = (float(np.sqrt(low * high)) if prior == 'log_uniform' and low > 0
                         else 0.5 * (float(low) + float(high)))
            fields['value'] = start
        try:
            spec = parse_parameter_spec(key, fields)
        except ValueError as error:
            raise ValueError(str(error).replace(f"planet.{key}", f"trend_priors.{key}")) from None
        if key == 'tau' and spec.value <= 0.0:
            raise ValueError("trend_priors.tau: 'value' (the start) must be > 0 days.")
        specs[key] = spec
    return specs


def trend_prior_tau_range(trend_priors, floor=1e-4):
    """``(low, high)`` days spanned by the configured tau prior (default 1e-3--1e-1)."""
    spec = (trend_priors or {}).get('tau')
    if spec is None:
        return 1e-3, 1e-1
    if spec.fixed:
        return spec.value, spec.value
    if spec.prior == 'gaussian':
        low = spec.value - 3 * spec.sigma if spec.low is None else max(spec.low, spec.value - 3 * spec.sigma)
        high = spec.value + 3 * spec.sigma if spec.high is None else min(spec.high, spec.value + 3 * spec.sigma)
    else:
        low, high = spec.low, spec.high
    high = max(float(high), floor)
    return min(max(float(low), floor), high), high


def _validate_flag_keys(flags):
    """Warn for ignored flag names while preserving legacy configuration use."""
    if not hasattr(flags, 'keys'):
        return
    candidates = sorted(KNOWN_FLAGS)
    for key in flags.keys():
        if key in KNOWN_FLAGS:
            continue
        closest = difflib.get_close_matches(
            str(key), candidates, n=1, cutoff=0.0
        )[0]
        warnings.warn(
            f"Unknown flags key {key!r}; did you mean {closest!r}? "
            "The unknown key is not used.",
            UnknownFlagWarning,
            stacklevel=2,
        )


def _build_gaussian_trend_prior(detrend_type, num_channels, init_params, flags):
    """Build explicit per-channel priors for analytic trend marginalization."""
    names = marginalized_trend_coefficient_names(detrend_type)
    default_means = {
        'c': 1.0,
        'v': 0.0,
        'v2': 0.0,
        'v3': 0.0,
        'v4': 0.0,
        'A': 0.0,
        'A_spot': 1.0,
        'A_spot2': 1.0,
        'A_jump': 1.0,
    }
    default_scales = {
        'c': 0.1,
        'v': 0.1,
        'v2': 0.1,
        'v3': 0.1,
        'v4': 0.1,
        'A': 0.1,
        'A_spot': 0.75,
        'A_spot2': 0.75,
        'A_jump': 0.75,
    }

    means = []
    scales = []
    for name in names:
        # Normalized priors are defined in normalized coordinates. Do not
        # accidentally center them on elapsed-day initialization coefficients.
        source_mean = (default_means[name] if flags.get('spectro_trend_basis') == 'batman_normalized'
                       else init_params.get(name, default_means[name]))
        mean = jnp.asarray(source_mean, dtype=jnp.float64)
        if mean.ndim == 0:
            mean = jnp.broadcast_to(mean, (num_channels,))
        if mean.shape != (num_channels,):
            raise ValueError(
                f"Trend prior mean for {name!r} must be scalar or shape "
                f"({num_channels},); received {mean.shape}."
            )
        scale = jnp.asarray(
            flags.get('trend_prior_scale', default_scales[name]), dtype=jnp.float64,
        )
        if scale.ndim == 0:
            scale = jnp.broadcast_to(scale, (num_channels,))
        if scale.shape != (num_channels,):
            raise ValueError(
                f"Trend prior scale for {name!r} must be scalar or shape "
                f"({num_channels},); received {scale.shape}."
            )
        if not bool(jnp.all(jnp.isfinite(mean))):
            raise ValueError(f"Trend prior mean for {name!r} is non-finite.")
        if not bool(jnp.all(jnp.isfinite(scale) & (scale > 0))):
            raise ValueError(f"Trend prior scale for {name!r} must be finite and > 0.")
        means.append(mean)
        scales.append(scale)
    return jnp.stack(means, axis=1), jnp.stack(scales, axis=1), names


def _has_stellar_ld_uncertainties(stellar_cfg):
    return all(k in stellar_cfg for k in ('teff_sigma', 'logg_sigma', 'feh_sigma'))


def _resolve_ld_prior_mode(flags, stellar_cfg, ld_profile):
    raw_mode = flags.get('ld_prior', None)
    resolved = resolve_ld_prior(
        raw_mode, ld_profile=ld_profile,
        has_stellar_uncertainties=_has_stellar_ld_uncertainties(stellar_cfg),
    )
    if resolved == 'stellarprior':
        if not _has_stellar_ld_uncertainties(stellar_cfg):
            raise ValueError(
                "flags.ld_prior='stellarprior' requires stellar.teff_sigma, "
                "stellar.logg_sigma, and stellar.feh_sigma."
            )
    return resolved


def _resolve_ld_configuration(flags, stellar_cfg):
    """Resolve public LD defaults while preserving explicitly selected treatments."""
    profile = validate_ld_profile(flags.get('ld_profile', 'quadratic'))
    prior = _resolve_ld_prior_mode(flags, stellar_cfg, profile)
    basis = str(flags.get('ld_uniform_basis', 'uplus_uminus')).strip().lower()
    if basis not in {'uplus_uminus', 'coefficients'}:
        raise ValueError("flags.ld_uniform_basis must be 'uplus_uminus' or 'coefficients'.")
    if profile == 'kipping':
        if 'ld_uniform_basis' in flags:
            raise ValueError("flags.ld_uniform_basis cannot be combined with flags.ld_profile='kipping'.")
        profile, basis = 'quadratic', 'kipping'
    return profile, prior, basis


def _resolve_ld_parameterization(ld_prior_mode, ld_profile='power2', requested=None):
    """Choose the sampling coordinates for power-2 limb darkening.

    Power-2 ``gaussian``/``uniform`` priors default to ``'maxted_smooth'``: a
    smooth bijection of Maxted-like coordinates onto the prior support, so the
    sampler sees no hard wall. ``requested`` (``flags.ld_parameterization``) may
    select ``'coefficients'`` (``(c, alpha)`` directly) or ``'decorrelated'``
    (Maxted coordinates with a hard support wall). Smooth and decorrelated
    coordinates preserve the full prior. Direct uniform coefficients retain
    the legacy alpha<40 truncation, so that route omits a small prior tail.
    """
    supports_decorrelated = ld_profile == 'power2' and ld_prior_mode in {'gaussian', 'uniform'}
    requested = 'auto' if requested is None else str(requested).strip().lower()
    if requested == 'auto':
        return 'maxted_smooth' if supports_decorrelated else 'coefficients'
    if requested == 'coefficients':
        return 'coefficients'
    if requested in {'decorrelated', 'maxted_smooth'}:
        if not supports_decorrelated:
            raise ValueError(
                f"flags.ld_parameterization={requested!r} requires power-2 limb "
                "darkening with ld_prior 'gaussian' or 'uniform'."
            )
        return requested
    raise ValueError(
        "flags.ld_parameterization must be 'auto', 'coefficients', 'decorrelated' or 'maxted_smooth'."
    )


def _spectro_detrend_type(detrending_type, fixed_timescale=False):
    """Map WL detrending type to spectroscopic equivalent where applicable."""
    detrend_type = detrending_type
    if fixed_timescale and 'explinear' in detrend_type and 'explinear_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('explinear', 'explinear_spectroscopic')
    if 'gp' in detrend_type and 'gp_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('gp', 'gp_spectroscopic')
    if 'linear_discontinuity' in detrend_type and 'linear_discontinuity_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('linear_discontinuity', 'linear_discontinuity_spectroscopic')
    if 'spot' in detrend_type and 'spot_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('spot', 'spot_spectroscopic')
    return detrend_type


def _has_single_spot_spectroscopic(detrend_type):
    return 'spot_spectroscopic' in detrend_type and '2spot_spectroscopic' not in detrend_type


def load_config(path):
    """Load the shared YAML schema with an actionable top-level error."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping of configuration sections; "
            "start with examples/niriss_soss_order1.yaml."
        )
    return config


def _resolve_stage_vmap_width(
    flags, stage_name, default_width, *, num_cadences, mcmc_kwargs, **context,
):
    """Resolve the documented automatic resident width."""
    if str(default_width).lower() != 'auto':
        return default_width
    from models.channel_batching import (
        SpectroMemoryModel,
        resolve_spectro_auto_width,
    )
    stats = jax.devices()[0].memory_stats() or {}
    bytes_limit = int(stats.get('bytes_limit', 0))
    if bytes_limit <= 0:
        raise ValueError(
            "spectro_chunk_size='auto' requires a backend that reports "
            "memory_stats()['bytes_limit']."
        )
    # Duration-grid interpolation is disabled for physical shared orbits.
    # Its measured smaller footprint must never size the direct evaluator.
    accelerated_long_cadence = False
    model = SpectroMemoryModel(
        intercept_bytes=160000000, bytes_per_lane=0.0,
        bytes_per_lane_cadence=(2500.0 if accelerated_long_cadence else 6000.0) * int(mcmc_kwargs.get("num_chains", 1)),
        bytes_per_draw_lane=128.0 * int(mcmc_kwargs.get("num_chains", 1)),
    )
    if int(num_cadences) > 10_000:
        speed_cap = 40 if accelerated_long_cadence else 4
    else:
        speed_cap = 160
    chains = int(mcmc_kwargs.get("num_chains", 1))
    channel_cap = max(1, speed_cap // chains)
    width = resolve_spectro_auto_width(
        model, bytes_limit=bytes_limit, cadences=int(num_cadences),
        draws=int(mcmc_kwargs['num_samples']), speed_cap=channel_cap, max_channels=channel_cap,
        headroom_fraction=0.25,
    )
    num_channels = int(context.get('num_channels', 0))
    if num_channels > 0:
        width = min(width, num_channels)
        batches = (num_channels + width - 1) // width
        # Keep one compiled width but spread channels evenly. A 42-channel
        # stage then uses 21+21 lanes, rather than 40+2 padded to 40+40.
        width = (num_channels + batches - 1) // batches
    debug(
        f"[spectro auto-width] stage={stage_name}, cadences={num_cadences}, "
        f"bytes_limit={bytes_limit}, headroom=25%, speed_cap={speed_cap}, "
        f"cadence_acceleration={accelerated_long_cadence}, "
        f"selected={width} channels × {chains} chains.",
        flush=True,
    )
    return width


def _resolve_stage_mcmc_kwargs(flags, stage_name, default_warmup=1000, default_samples=1000):
    stage_key = str(stage_name).lower()
    num_warmup = int(flags.get(f"{stage_key}_num_warmup", default_warmup))
    num_samples = int(flags.get(f"{stage_key}_num_samples", default_samples))
    num_chains = int(flags.get(f"{stage_key}_num_chains", flags.get("num_chains", 4)))
    if num_warmup < 0:
        raise ValueError(f"flags.{stage_key}_num_warmup must be >= 0.")
    if num_samples < 1:
        raise ValueError(f"flags.{stage_key}_num_samples must be >= 1.")
    if num_chains < 1:
        raise ValueError(f"flags.{stage_key}_num_chains must be >= 1.")
    return {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "chain_method": "vectorized",
    }


def _resolve_whitelight_laplace_options(is_prism=False, mass_matrix="laplace"):
    """Return the fixed production white-light Laplace policy."""
    return {
        "mass_matrix": mass_matrix,
        "warmup": 200,
        "target_accept": 0.99 if is_prism else 0.9,
        "max_tree_depth": 10,
        "trust_radius": 5.0,
        "hessian_method": "finite_difference",
    }


def _trend_components(detrending_type):
    """Normalize the component names used by white-light sampler policy."""
    return {
        item.strip().lower()
        for item in str(detrending_type).replace("_", "+").split("+")
        if item.strip()
    }


def _resolve_whitelight_mass_matrix(detrending_type, requested=None):
    """Choose the production white-light metric for the requested trend.

    ``requested`` (``flags.whitelight_mass_matrix``) may force ``'laplace'`` or
    ``'adaptive'``; ``None``/``'auto'`` keeps the trend-based default.
    """
    requested = 'auto' if requested is None else str(requested).strip().lower()
    if requested in {'laplace', 'adaptive'}:
        return requested
    if requested != 'auto':
        raise ValueError("flags.whitelight_mass_matrix must be 'auto', 'laplace' or 'adaptive'.")
    components = _trend_components(detrending_type)
    is_complex = bool(
        components.intersection({"spot", "2spot", "discontinuity", "step"})
        or "discontinuity" in str(detrending_type).lower()
        or "step" in str(detrending_type).lower()
    )
    return "adaptive" if "2spot" in components or (
        "+" in str(detrending_type) and is_complex
    ) else "laplace"


def _resolve_whitelight_trend_parameterization(detrending_type):
    """Infer the production trend coordinates from the trend model."""
    components = _trend_components(detrending_type)
    is_step = bool(
        components.intersection({"discontinuity", "step"})
        or "discontinuity" in str(detrending_type).lower()
        or "step" in str(detrending_type).lower()
    )
    return "cadence" if components == {"spot"} or is_step else "physical"


def _resolve_compile_cache_options(flags):
    """Return the persistent JAX compilation-cache directory, or None if disabled.

    The cache is on by default (``.koala_cache/jax``); set
    ``jax_compilation_cache_dir`` to ``false``/``off`` to disable it.
    """
    cache_dir = flags.get("jax_compilation_cache_dir", ".koala_cache/jax")
    if cache_dir is None or cache_dir is False or str(cache_dir).strip().lower() in {
        "", "false", "off", "none",
    }:
        return {"cache_dir": None}
    return {"cache_dir": str(cache_dir)}


def _resolve_ld_prior_cache_options(stellar_cfg):
    """Return the fingerprinted stellar-LD cache settings."""
    return {
        "enabled": bool(stellar_cfg.get("ld_prior_cache", True)),
        "cache_dir": str(
            stellar_cfg.get('ld_prior_cache_dir', '.koala_cache/ld_prior'),
        ),
    }


def _resolve_harmonica_stage_nuts_kwargs(
    flags, stage_prefix, *, default_dense_mass, default_regularize_mass_matrix=True,
    default_max_tree_depth=8, default_target_accept=0.8, is_prism=False, is_explinear=False,
):
    result = {
        "dense_mass": bool(default_dense_mass),
        "regularize_mass_matrix": bool(default_regularize_mass_matrix),
        "max_tree_depth": int(default_max_tree_depth),
        "target_accept_prob": float(default_target_accept),
    }
    sampler = str(flags.get("spectro_sampler", "independent_nuts")).lower()
    # White light is sampled by NumPyro's NUTS constructor. Its Laplace
    # preparation is configured separately by whitelight_laplace_options;
    # independent-runner options belong only to the spectroscopic stages.
    if stage_prefix != "harmonica_wl" and sampler in {"independent_nuts", "independent_hmc"}:
        result = _resolve_jaxoplanet_spectro_nuts_kwargs(
            stage_prefix, result, independent=True, hmc=sampler == 'independent_hmc',
            is_prism=is_prism, is_explinear=is_explinear,
        )
    return result


def _resolve_spectro_mass_matrix(flags):
    """Return ``flags.spectro_mass_matrix``: ``'laplace'`` (default) or ``'adaptive'``."""
    value = str(flags.get('spectro_mass_matrix', 'laplace')).strip().lower()
    if value not in {'laplace', 'adaptive'}:
        raise ValueError("flags.spectro_mass_matrix must be 'laplace' or 'adaptive'.")
    return value


def _resolve_jaxoplanet_spectro_nuts_kwargs(
    stage_prefix, defaults, *, independent=False, hmc=False, is_prism=False,
    is_explinear=False, mass_matrix='laplace',
):
    """Return the fixed production spectroscopic sampler policy.

    ``mass_matrix='adaptive'`` replaces the Laplace metric of independent NUTS
    with per-channel windowed dense-mass adaptation (same posterior target). It
    helps when the MAP lies on the edge of a bounded prior, where the Laplace
    metric is poor.
    """
    result = dict(defaults)
    if not independent:
        return result
    laplace_accept = 0.99 if (is_prism or is_explinear) else 0.95
    result.update(
        dense_mass=True, mass_matrix='laplace', target_accept_prob=laplace_accept,
        laplace_warmup=150, laplace_target_accept=laplace_accept if not hmc else 0.85,
        laplace_start_at_map=True, laplace_hessian_method='finite_difference',
        laplace_fd_relative_step=0.0002, laplace_fd_batch_size=1, laplace_fuse_program=False,
        laplace_trust_radius=5.0, laplace_map_decrement_tolerance=0.0001,
    )
    if hmc:
        result.pop("max_tree_depth", None)
        result.update(num_steps=8, trajectory_jitter=0.25)
    else:
        depth = 6 if is_prism else 5
        result.update(max_tree_depth=depth, laplace_max_tree_depth=depth)
        if mass_matrix == 'adaptive':
            result.update(mass_matrix='adaptive', target_accept_prob=0.9, max_tree_depth=8)
    return result


def _resolve_spectro_joint_geometry(flags):
    """Return ``(enabled, prior_inflation)`` for ``flags.spectro_joint_geometry``.

    When enabled, the spectroscopic stages sample the transit geometry
    (``t0``, ``b``, and ``duration`` or ``a_rs``) as sites shared by every
    channel instead of fixing them to the white-light medians.  The prior on
    each shared site is a Gaussian centred on the white-light posterior median
    whose width is the white-light posterior standard deviation multiplied by
    ``flags.spectro_joint_geometry_prior_inflation`` (default 3).
    """
    enabled = flags.get('spectro_joint_geometry', False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {'1', 'true', 'yes', 'on'}
    enabled = bool(enabled)
    inflation = flags.get('spectro_joint_geometry_prior_inflation', 3.0)
    try:
        inflation = float(inflation)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "flags.spectro_joint_geometry_prior_inflation must be a number."
        ) from error
    if not np.isfinite(inflation) or inflation <= 0.0:
        raise ValueError(
            "flags.spectro_joint_geometry_prior_inflation must be finite and > 0."
        )
    return enabled, inflation


def _resolve_spectro_joint_geometry_sampler(
    spectro_sampler, joint_geometry, nuts_kwargs, *, stage_name='spectroscopic'
):
    """Force the joint sampler when the geometry is shared across channels.

    Shared sites cannot be sampled by the per-channel independent lanes, so a
    configured ``independent_nuts``/``independent_hmc`` backend is overridden
    to ``joint_nuts`` with a warning and the Laplace-metric lane options are
    replaced by the joint-NUTS defaults.
    """
    if not joint_geometry:
        return spectro_sampler, nuts_kwargs
    sampler = str(spectro_sampler).lower()
    if sampler == 'joint_nuts':
        return sampler, nuts_kwargs
    warnings.warn(
        f"flags.spectro_joint_geometry=true requires the joint sampler; "
        f"overriding flags.spectro_sampler={sampler!r} with 'joint_nuts' for "
        f"the {stage_name} stage.",
        UserWarning,
        stacklevel=2,
    )
    from models.jaxoplanet.builder import NUTS_KWARGS as _JAXOPLANET_NUTS_KWARGS
    return 'joint_nuts', _resolve_jaxoplanet_spectro_nuts_kwargs(
        stage_name, _JAXOPLANET_NUTS_KWARGS, independent=False
    )


def _resolve_spectro_joint_geometry_chunk_size(
    chunk_size, num_channels, joint_geometry
):
    """Return the resident width that keeps every channel in one joint fit."""
    if not joint_geometry:
        return chunk_size
    num_channels = int(num_channels)
    if num_channels < 1:
        raise ValueError("Joint geometry requires at least one channel.")
    if chunk_size is None or (
        isinstance(chunk_size, str) and chunk_size.lower() == 'auto'
    ):
        return num_channels
    width = int(chunk_size)
    if width < num_channels:
        raise ValueError(
            "flags.spectro_joint_geometry=true fits every channel of a stage "
            "in one chunk so the geometry is shared across the whole spectrum, "
            f"but flags.spectro_chunk_size/vmap_chunk={width} would split the "
            f"{num_channels} channels. Raise it to at least {num_channels} "
            "or remove it."
        )
    return num_channels


def _resolve_spectro_fixed_timescale_trends(flags, detrending_type, transit_engine):
    """Select a fixed white-light ramp shape or the existing free-tau model."""
    value = flags.get('spectro_fixed_timescale_trends', True)
    if not isinstance(value, bool):
        raise ValueError('flags.spectro_fixed_timescale_trends must be Boolean.')
    if value and 'explinear' in detrending_type and transit_engine != 'jaxoplanet':
        raise ValueError("Fixed-timescale spectroscopy requires transit_engine='jaxoplanet'; "
                         "set flags.spectro_fixed_timescale_trends=false for sampled tau.")
    return value

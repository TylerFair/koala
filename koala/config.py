"""Configuration loading, validation, and option resolution."""

import difflib
import numbers
import warnings
from dataclasses import dataclass
import numpy as np
import yaml
import jax
import jax.numpy as jnp
from models.channel_batching import SpectroMemoryModel, resolve_spectro_auto_width
from models.cadence_reduction import linear_spectro_trend_coefficient_names
from models.limb_darkening_config import resolve_ld_prior
from models.trend_marginal import marginalized_trend_coefficient_names
from .constants import *

class UnknownFlagWarning(UserWarning):
    """A ``flags`` entry is not consumed by the current configuration API."""


@dataclass(frozen=True)
class ParameterSpec:
    """How one ``planet`` parameter enters the white-light fit.

    ``mode`` is ``'fixed'`` or ``'free'``.  A free parameter written as a bare
    number keeps ``prior=None``, which means the engine's historical default
    prior; an explicit ``[free, prior, ...]`` form sets ``prior`` to
    ``'uniform'`` (``low``/``high``), ``'gaussian'`` (``mu``/``sigma``) or
    ``'truncated_gaussian'`` (all four).  ``value`` holds the fixed value or
    the bare-number initial guess.
    """

    name: str
    mode: str
    prior: str = None
    value: float = None
    mu: float = None
    sigma: float = None
    low: float = None
    high: float = None
    explicit: bool = False

    @property
    def free(self):
        return self.mode == 'free'

    @property
    def fixed(self):
        return self.mode == 'fixed'

    @property
    def explicit_prior(self):
        """True when a free parameter carries a user-specified prior."""
        return self.free and self.prior is not None

    @property
    def center(self):
        """Representative value: fixed value, prior mean, or interval midpoint."""
        if self.value is not None:
            return float(self.value)
        if self.prior in {'gaussian', 'truncated_gaussian'}:
            return float(self.mu)
        return 0.5 * (float(self.low) + float(self.high))

    def describe(self):
        if self.fixed:
            return f"fixed at {self.value!r}"
        if self.prior is None:
            return f"free (default prior, initial value {self.value!r})"
        if self.prior == 'uniform':
            return f"free, uniform({self.low!r}, {self.high!r})"
        if self.prior == 'gaussian':
            return f"free, gaussian(mu={self.mu!r}, sigma={self.sigma!r})"
        return (
            f"free, truncated_gaussian(mu={self.mu!r}, sigma={self.sigma!r}, "
            f"low={self.low!r}, high={self.high!r})"
        )


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _spec_number(key, label, value):
    if not _is_number(value):
        raise ValueError(
            f"planet.{key}: {label} must be a number, received {value!r}."
        )
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"planet.{key}: {label} must be finite.")
    return value


def _finish_free_spec(key, prior, fields):
    """Validate the prior fields of an explicit free specification."""
    if prior not in PARAMETER_SPEC_PRIORS:
        raise ValueError(
            f"planet.{key}: unknown prior {prior!r}; expected one of "
            f"{list(PARAMETER_SPEC_PRIORS)}."
        )
    mu = sigma = low = high = None
    if prior in {'gaussian', 'truncated_gaussian'}:
        mu = _spec_number(key, 'mu', fields['mu'])
        sigma = _spec_number(key, 'sigma', fields['sigma'])
        if sigma <= 0.0:
            raise ValueError(f"planet.{key}: sigma must be > 0, received {sigma!r}.")
    if prior in {'uniform', 'truncated_gaussian'}:
        low = _spec_number(key, 'low', fields['low'])
        high = _spec_number(key, 'high', fields['high'])
        if low >= high:
            raise ValueError(
                f"planet.{key}: prior bounds require low < high, received "
                f"low={low!r}, high={high!r}."
            )
    if key in PLANET_PARAMETERS_FIXED_ONLY:
        raise ValueError(
            f"planet.{key} may only be fixed for now; write "
            f"[fixed, <value>] or a bare number."
        )
    return ParameterSpec(
        key, 'free', prior, None, mu, sigma, low, high, explicit=True,
    )


_LIST_FORMS = {
    'uniform': '[free, uniform, low, high]',
    'gaussian': '[free, gaussian, mu, sigma]',
    'truncated_gaussian': '[free, truncated_gaussian, mu, sigma, low, high]',
}


def _parse_list_spec(key, raw):
    mode = str(raw[0]).strip().lower()
    if mode not in PARAMETER_SPEC_MODES:
        raise ValueError(
            f"planet.{key}: unknown mode {raw[0]!r}; expected 'fixed' or 'free'."
        )
    if mode == 'fixed':
        if len(raw) != 2:
            raise ValueError(
                f"planet.{key}: the fixed form is [fixed, value], received {raw!r}."
            )
        return ParameterSpec(
            key, 'fixed', None, _spec_number(key, 'value', raw[1]), explicit=True,
        )
    if len(raw) < 2:
        raise ValueError(
            f"planet.{key}: the free form is [free, prior, ...], received {raw!r}."
        )
    prior = str(raw[1]).strip().lower()
    if prior not in _LIST_FORMS:
        raise ValueError(
            f"planet.{key}: unknown prior {raw[1]!r}; expected one of "
            f"{list(PARAMETER_SPEC_PRIORS)}."
        )
    arity = {'uniform': 4, 'gaussian': 4, 'truncated_gaussian': 6}[prior]
    if len(raw) != arity:
        raise ValueError(
            f"planet.{key}: expected {_LIST_FORMS[prior]}, received {raw!r}."
        )
    values = raw[2:]
    if prior == 'uniform':
        fields = {'low': values[0], 'high': values[1]}
    elif prior == 'gaussian':
        fields = {'mu': values[0], 'sigma': values[1]}
    else:
        fields = {'mu': values[0], 'sigma': values[1], 'low': values[2], 'high': values[3]}
    return _finish_free_spec(key, prior, fields)


_DICT_FIELD_ALIASES = {
    'lo': 'low', 'min': 'low', 'lower': 'low',
    'hi': 'high', 'max': 'high', 'upper': 'high',
    'mean': 'mu', 'loc': 'mu', 'std': 'sigma', 'scale': 'sigma',
}


def _parse_dict_spec(key, raw):
    fields = {}
    for name, value in raw.items():
        canonical = _DICT_FIELD_ALIASES.get(str(name).lower(), str(name).lower())
        fields[canonical] = value
    if 'mode' not in fields:
        raise ValueError(
            f"planet.{key}: a mapping specification needs 'mode' "
            "('fixed' or 'free')."
        )
    mode = str(fields.pop('mode')).strip().lower()
    if mode not in PARAMETER_SPEC_MODES:
        raise ValueError(
            f"planet.{key}: unknown mode {mode!r}; expected 'fixed' or 'free'."
        )
    if mode == 'fixed':
        unexpected = set(fields) - {'value'}
        if unexpected or 'value' not in fields:
            raise ValueError(
                f"planet.{key}: the fixed mapping form is "
                "{mode: fixed, value: <number>}."
            )
        return ParameterSpec(
            key, 'fixed', None, _spec_number(key, 'value', fields['value']),
            explicit=True,
        )
    if 'prior' not in fields:
        raise ValueError(
            f"planet.{key}: a free mapping needs 'prior' ('uniform', "
            "'gaussian' or 'truncated_gaussian')."
        )
    prior = str(fields.pop('prior')).strip().lower()
    required = {
        'uniform': {'low', 'high'},
        'gaussian': {'mu', 'sigma'},
        'truncated_gaussian': {'mu', 'sigma', 'low', 'high'},
    }.get(prior)
    if required is None:
        raise ValueError(
            f"planet.{key}: unknown prior {prior!r}; expected one of "
            f"{list(PARAMETER_SPEC_PRIORS)}."
        )
    missing = required - set(fields)
    unexpected = set(fields) - required
    if missing or unexpected:
        raise ValueError(
            f"planet.{key}: prior '{prior}' takes exactly {sorted(required)}; "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}."
        )
    return _finish_free_spec(key, prior, fields)


def parse_parameter_spec(key, raw):
    """Parse one bare-number, list, or mapping parameter specification."""
    if _is_number(raw):
        mode = 'fixed' if key in PLANET_PARAMETERS_FIXED_BY_DEFAULT else 'free'
        return ParameterSpec(key, mode, None, _spec_number(key, 'value', raw))
    if isinstance(raw, (list, tuple)):
        if not raw or not isinstance(raw[0], str):
            raise ValueError(
                f"planet.{key}: expected [fixed, value] or [free, prior, ...], "
                f"received {list(raw)!r}."
            )
        return _parse_list_spec(key, list(raw))
    if isinstance(raw, dict):
        return _parse_dict_spec(key, raw)
    raise ValueError(
        f"planet.{key}: expected a number, [fixed, value], "
        f"[free, prior, ...] or a mapping, received {raw!r}."
    )


def _parse_parameter_entries(key, raw):
    """Return one specification per planet for a ``planet`` block entry."""
    if isinstance(raw, (list, tuple)) and raw and not isinstance(raw[0], str):
        return tuple(parse_parameter_spec(key, item) for item in raw)
    return (parse_parameter_spec(key, raw),)


def parse_planet_parameter_specs(planet_cfg):
    """Parse every orbital entry of ``planet`` into per-planet specifications.

    Returns ``{name: (ParameterSpec, ...)}`` with one entry per planet for
    each of ``PLANET_PARAMETER_KEYS`` present in the block (``ecc`` and
    ``omega`` default to fixed zero).  Scalars are broadcast to the number of
    planets, which is taken from ``period`` when present.
    """
    entries = {}
    for key in PLANET_PARAMETER_KEYS:
        if key in planet_cfg:
            entries[key] = _parse_parameter_entries(key, planet_cfg[key])
    for key in ('ecc', 'omega'):
        entries.setdefault(key, (ParameterSpec(key, 'fixed', None, 0.0),))
    if 'period' in entries:
        n_planets = len(entries['period'])
    else:
        n_planets = max(len(specs) for specs in entries.values())
    resolved = {}
    for key, specs in entries.items():
        if len(specs) == 1 and n_planets > 1:
            specs = specs * n_planets
        if len(specs) != n_planets:
            raise ValueError(
                f"planet.{key} must be scalar or length {n_planets}, "
                f"got {len(specs)} entries."
            )
        resolved[key] = tuple(specs)
    return resolved


def planet_parameter_centers(specs, key, default=None):
    """Return the representative value of ``key`` for every planet."""
    if key not in specs:
        if default is None:
            raise KeyError(f"planet.{key} is not configured.")
        n_planets = len(next(iter(specs.values())))
        return np.full(n_planets, float(default), dtype=np.float64)
    return np.asarray([spec.center for spec in specs[key]], dtype=np.float64)


def describe_planet_parameter_specs(specs):
    """One human-readable line per configured parameter."""
    lines = []
    for key, entries in specs.items():
        described = [spec.describe() for spec in entries]
        if len(set(described)) == 1:
            lines.append(f"planet.{key}: {described[0]}")
        else:
            lines.append(
                f"planet.{key}: "
                + "; ".join(f"planet {i}: {text}" for i, text in enumerate(described))
            )
    return lines


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
        source_mean = init_params.get(name, default_means[name])
        mean = jnp.asarray(source_mean, dtype=jnp.float64)
        if mean.ndim == 0:
            mean = jnp.broadcast_to(mean, (num_channels,))
        if mean.shape != (num_channels,):
            raise ValueError(
                f"Trend prior mean for {name!r} must be scalar or shape "
                f"({num_channels},); received {mean.shape}."
            )
        scale = jnp.asarray(
            default_scales[name],
            dtype=jnp.float64,
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
        raw_mode,
        ld_profile=ld_profile,
        has_stellar_uncertainties=_has_stellar_ld_uncertainties(stellar_cfg),
    )
    if resolved == 'stellarprior':
        if not _has_stellar_ld_uncertainties(stellar_cfg):
            raise ValueError(
                "flags.ld_prior='stellarprior' requires stellar.teff_sigma, "
                "stellar.logg_sigma, and stellar.feh_sigma."
            )
    return resolved


def _resolve_ld_parameterization(ld_prior_mode, ld_profile='power2'):
    """Infer the retained coefficient or Maxted-decorrelated coordinates."""
    if ld_profile == 'power2' and ld_prior_mode in {'gaussian', 'uniform'}:
        return 'decorrelated'
    return 'coefficients'


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
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_stage_vmap_width(
    flags,
    stage_name,
    default_width,
    *,
    num_cadences,
    mcmc_kwargs,
    **context,
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
    accelerated_long_cadence = (
        int(num_cadences) > 5_000
        and str(flags.get('spectro_cadence_reduction', 'auto')).lower()
        == 'auto'
        and str(flags.get('spectro_transit_grid', 'auto')).lower() == 'auto'
        and context.get('sampler_backend') == 'independent_nuts'
        and context.get('transit_engine') == 'jaxoplanet'
        and context.get('ld_profile') in {'power2', 'quadratic'}
        and context.get('ld_mode') != 'interpolated'
        and bool(context.get(
            'transit_grid_outer_contact_safe',
            context.get('transit_grid_non_grazing', False),
        ))
        and context.get('trend_inference') == 'sampled_uniform'
        and linear_spectro_trend_coefficient_names(
            context.get('detrend_type')
        ) is not None
        and context.get('param_method') == 'duration'
        and int(context.get('n_planets', 0)) == 1
        and context.get('transit_window') == 'auto'
    )
    model = SpectroMemoryModel(
        intercept_bytes=160_000_000,
        bytes_per_lane=0.0,
        bytes_per_lane_cadence=(2_500.0 if accelerated_long_cadence else 6_000.0),
        bytes_per_draw_lane=128.0,
    )
    if int(num_cadences) > 10_000:
        speed_cap = 40 if accelerated_long_cadence else 4
    else:
        speed_cap = 160
    width = resolve_spectro_auto_width(
        model,
        bytes_limit=bytes_limit,
        cadences=int(num_cadences),
        draws=int(mcmc_kwargs['num_samples']),
        speed_cap=speed_cap,
        max_channels=speed_cap,
        headroom_fraction=0.25,
    )
    print(
        f"[spectro auto-width] stage={stage_name}, cadences={num_cadences}, "
        f"bytes_limit={bytes_limit}, headroom=25%, speed_cap={speed_cap}, "
        f"cadence_acceleration={accelerated_long_cadence}, "
        f"selected={width} lanes.",
        flush=True,
    )
    return width


def _resolve_stage_mcmc_kwargs(flags, stage_name, default_warmup=1000, default_samples=1000):
    stage_key = str(stage_name).lower()
    num_warmup = int(flags.get(f"{stage_key}_num_warmup", default_warmup))
    num_samples = int(flags.get(f"{stage_key}_num_samples", default_samples))
    if num_warmup < 0:
        raise ValueError(f"flags.{stage_key}_num_warmup must be >= 0.")
    if num_samples < 1:
        raise ValueError(f"flags.{stage_key}_num_samples must be >= 1.")
    return {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
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


def _resolve_whitelight_mass_matrix(detrending_type):
    """Choose the production white-light metric for the requested trend."""
    components = {
        item.strip().lower()
        for item in str(detrending_type).replace("_", "+").split("+")
        if item.strip()
    }
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
    components = {
        item.strip().lower()
        for item in str(detrending_type).replace("_", "+").split("+")
        if item.strip()
    }
    is_step = bool(
        components.intersection({"discontinuity", "step"})
        or "discontinuity" in str(detrending_type).lower()
        or "step" in str(detrending_type).lower()
    )
    return "cadence" if components == {"spot"} or is_step else "physical"


def _resolve_compile_cache_options(flags):
    """Return the optional advanced persistent-cache directory."""
    cache_dir = flags.get("jax_compilation_cache_dir")
    return {"cache_dir": None if cache_dir in {None, ""} else str(cache_dir)}


def _resolve_ld_prior_cache_options(stellar_cfg):
    """Return the fingerprinted stellar-LD cache settings."""
    return {
        "enabled": bool(stellar_cfg.get("ld_prior_cache", True)),
        "cache_dir": str(stellar_cfg.get(
            "ld_prior_cache_dir",
            ".koala_cache/ld_prior",
        )),
    }


def _resolve_harmonica_stage_nuts_kwargs(
    flags,
    stage_prefix,
    *,
    default_dense_mass,
    default_regularize_mass_matrix=True,
    default_max_tree_depth=8,
    default_target_accept=0.8,
    is_prism=False,
    is_explinear=False,
):
    result = {
        "dense_mass": bool(default_dense_mass),
        "regularize_mass_matrix": bool(default_regularize_mass_matrix),
        "max_tree_depth": int(default_max_tree_depth),
        "target_accept_prob": float(default_target_accept),
    }
    sampler = str(flags.get("spectro_sampler", "independent_nuts")).lower()
    if sampler in {"independent_nuts", "independent_hmc"}:
        result = _resolve_jaxoplanet_spectro_nuts_kwargs(
            stage_prefix,
            result,
            independent=True,
            hmc=sampler == "independent_hmc",
            is_prism=is_prism,
            is_explinear=is_explinear,
        )
    return result


def _resolve_jaxoplanet_spectro_nuts_kwargs(
    stage_prefix,
    defaults,
    *,
    independent=False,
    hmc=False,
    is_prism=False,
    is_explinear=False,
):
    """Return the fixed production spectroscopic sampler policy."""
    result = dict(defaults)
    if not independent:
        return result
    laplace_accept = 0.99 if (is_prism or is_explinear) else 0.95
    result.update(
        dense_mass=True,
        mass_matrix="laplace",
        target_accept_prob=laplace_accept,
        laplace_warmup=150,
        laplace_target_accept=laplace_accept if not hmc else 0.85,
        laplace_start_at_map=True,
        laplace_hessian_method="finite_difference",
        laplace_fd_relative_step=2.0e-4,
        laplace_fd_batch_size=1,
        laplace_fuse_program=False,
        laplace_trust_radius=5.0,
        laplace_map_decrement_tolerance=1.0e-4,
    )
    if hmc:
        result.pop("max_tree_depth", None)
        result.update(num_steps=8, trajectory_jitter=0.25)
    else:
        depth = 6 if is_prism else 5
        result.update(max_tree_depth=depth, laplace_max_tree_depth=depth)
    return result

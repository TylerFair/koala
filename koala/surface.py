"""Surface helpers."""

import re
import jax
import jax.numpy as jnp
import numpy as np
from .constants import _JAXOPLANET_STATIC_EVAL_KEYS, _JAXOPLANET_DYNAMIC_EVAL_KEYS


def _attach_jaxoplanet_eval_metadata(
    params,
    t,
    *,
    jaxoplanet_kernel,
    ld_profile,
    transit_window_optimization="off",
):
    """Attach the physical orbit and routing metadata to diagnostics.

    Drop cached phase/window arrays: diagnostics evaluate the complete current
    cadence axis and channel contacts can differ from the reference duration.
    """
    out = dict(params)
    for name in (*_JAXOPLANET_STATIC_EVAL_KEYS, *_JAXOPLANET_DYNAMIC_EVAL_KEYS):
        out.pop(name, None)
    out["_jaxoplanet_kernel"] = str(jaxoplanet_kernel)
    out["_ld_profile"] = str(ld_profile)
    if (ld_profile == 'power2' and 'c1' in out and 'c2' in out
            and out.get('_surface_model', 'transit') == 'transit'
            and not out.get('_stellar_spots')):
        out['_power2_exact'] = True

    from models.jaxoplanet.core import duration_to_orbit
    if "duration" in out:
        out = duration_to_orbit(out)
    if transit_window_optimization not in {'auto', 'off'}:
        raise ValueError("transit_window_optimization must be one of {'auto', 'off'}.")
    # Diagnostics evaluate the full current cadence axis. A WL duration mask
    # can truncate a channel with a different radius and must not be reused.
    return out


def _attach_surface_eval_metadata(params, surface_config):
    """Attach static map geometry and fixed/default surface values for plotting."""
    out = dict(params)
    if not surface_config or (
        surface_config.get("model", "transit") == "transit"
        and not surface_config.get("spots")
    ):
        return out
    out["_surface_model"] = surface_config["model"]
    out["_stellar_spots"] = surface_config.get("spots", ())
    for name in ("eclipse_depth", "phase_c1", "phase_d1"):
        indexed = sorted(
            (
                (int(key.rsplit("_", 1)[1]), value)
                for key, value in out.items()
                if re.fullmatch(rf"_{name}_\d+", key)
            ),
            key=lambda item: item[0],
        )
        if name not in out and indexed:
            out[name] = jnp.stack([value for _, value in indexed])
        if name in surface_config and name not in out:
            out[name] = jnp.asarray(surface_config[name], dtype=jnp.float64)
    if surface_config.get("spots"):
        out["stellar_rotation_period"] = jnp.asarray(
            surface_config["stellar_rotation_period"], dtype=jnp.float64
        )
        if "stellar_spot_contrast" not in out:
            indexed = sorted(
                (
                    (int(key.rsplit("_", 1)[1]), value)
                    for key, value in out.items()
                    if re.fullmatch(r"_stellar_spot_contrast_\d+", key)
                ),
                key=lambda item: item[0],
            )
            out["stellar_spot_contrast"] = (
                jnp.stack([value for _, value in indexed])
                if indexed else jnp.asarray(
                    [spot["contrast"] for spot in surface_config["spots"]],
                    dtype=jnp.float64,
                )
            )
    return out


def _spectroscopic_surface_config(surface_config, parameter_specs):
    """Resolve radius freedom after white-light orbital geometry is handed off.

    The white-light flag describes all orbital parameters. Spectroscopy holds
    those at the handoff and only the radius specification controls whether
    the channel model needs variable geometry.
    """
    config = dict(surface_config)
    if config.get("model", "transit") != "transit" or config.get("spots"):
        config["fit_geometry"] = any(spec.free for spec in parameter_specs["rprs"])
    return config


def _seed_surface_spectroscopic_init(init_params, surface_config, num_lcs):
    """Seed every free surface latent at its physical user prior center."""
    if not surface_config:
        return init_params
    from models.priors import latent_init_site
    for name in ("eclipse_depth", "phase_c1", "phase_d1"):
        for planet, spec in enumerate(surface_config.get(f"{name}_spec", ())):
            init = latent_init_site(spec, f"_{name}_{planet}")
            if init is not None:
                site, value = init
                init_params[site] = jnp.full(int(num_lcs), value, dtype=jnp.float64)
    for spot, spec in enumerate(surface_config.get("spots", ())):
        if float(spec.get("contrast_prior_width", 0.0)) > 0.0:
            init_params[f"_stellar_spot_contrast_{spot}"] = jnp.full(
                int(num_lcs), float(spec["contrast"]), dtype=jnp.float64
            )
    return init_params


def _surface_polynomial_ld(u, ld_profile):
    """Return polynomial limb-darkening rows for the starry surface.

    starry takes polynomial coefficients directly, so quadratic ``(u1, u2)``
    pass through. Power-2 ``(c, alpha)`` must first be projected onto the same
    degree-12 polynomial the jaxoplanet builders use; handing them to starry
    unconverted silently produces a wrong intensity profile.
    """
    u = np.asarray(u, dtype=float)
    if ld_profile != "power2":
        return u
    from models.common import get_I_power2
    from models.detrend import _prepare_power2_poly

    mus, projection = _prepare_power2_poly()
    rows = np.atleast_2d(u)
    if rows.shape[-1] != 2:
        raise ValueError("power2 limb darkening needs (c, alpha) per channel.")
    profiles = np.asarray(get_I_power2(rows[:, :1], rows[:, 1:], np.asarray(mus)[None, :]))
    poly = np.asarray(np.asarray(projection) @ (1.0 - profiles).T).T
    return poly[0] if u.ndim == 1 else poly


def _prepare_fixed_surface_basis(
    surface_config, t, u, *, period, t0, a_rs, b, rors, ecc, omega,
    ld_profile="quadratic",
):
    """Precompute an exact native linear basis for a fixed surface model."""
    from models.configuration_support import validate_power2_support
    validate_power2_support(ld_profile, model=surface_config.get('model', 'transit'),
                            spots=surface_config.get('spots', ()))
    if surface_config.get("fit_geometry", True):
        return None
    u = _surface_polynomial_ld(u, ld_profile)
    from models.jaxoplanet.surface_basis import (
        EmissionLightCurveBasis,
        SpotLightCurveBasis,
        prepare_emission_light_curve_basis,
        prepare_spot_light_curve_basis,
    )

    model = surface_config.get("model", "transit")
    spots = surface_config.get("spots", ())
    if not ((model == "transit" and spots) or (model in {"eclipse", "phase_curve"} and not spots)):
        return None
    base = {
        "_surface_model": model,
        "period": period,
        "t0": t0,
        "a_rs": a_rs,
        "b": b,
        "rors": rors,
        "ecc": ecc,
        "omega": omega,
    }
    if spots:
        base["stellar_rotation_period"] = surface_config["stellar_rotation_period"]

    def prepare_one(coefficients):
        fixed = {**base, "u": coefficients}
        if spots:
            return prepare_spot_light_curve_basis(
                fixed, t, spots=spots
            )
        return prepare_emission_light_curve_basis(
            fixed, t, model=model
        )

    def stack_bases(bases):
        return type(bases[0])(*(
            None if field is None else jnp.stack([
                basis[index] for basis in bases
            ])
            for index, field in enumerate(bases[0])
        ))

    u_grid = np.asarray(u, dtype=float)
    if u_grid.ndim == 1:
        return prepare_one(u_grid)
    if u_grid.ndim != 2:
        raise ValueError("Fixed surface-basis limb darkening must have one row per channel.")
    # A shared explicit LD law is the common fast path.  For a wavelength
    # dependent fixed grid, prepare each exact native basis once and stack it;
    # the sampler still sees only the small linear contrast model.
    if np.allclose(u_grid, u_grid[0], rtol=0.0, atol=0.0):
        shared = prepare_one(u_grid[0])
        return type(shared)(*(
            None if field is None else jnp.broadcast_to(
                field, (u_grid.shape[0],) + field.shape
            )
            for field in shared
        ))
    return stack_bases([prepare_one(row) for row in u_grid])


def _scan_tau_grid(tau_range):
    """Explinear timescales tried by the eclipse scan (``tau_range`` in days).

    The default 1e-3--1e-1 d range keeps the historical 25-point grid; a wider
    configured ``trend_priors.tau`` range keeps the same log spacing.
    """
    low, high = (float(x) for x in tau_range)
    if low == 1e-3 and high == 1e-1:
        return np.geomspace(1e-3, 1e-1, 25)
    if high <= low:
        return np.asarray([high])
    return np.geomspace(low, high, max(25, int(np.ceil(12.5 * np.log10(high / low))) + 1))


def scan_eclipse_time(
    surface_config, t, flux, flux_err, t0_spec, depth_spec, detrending_type,
    *, period, a_rs, b, rors, ecc, omega, tau_range=(1e-3, 1e-1),
):
    """Locate the secondary eclipse by brute force before gradient fitting.

    A shallow eclipse gives no timing gradient once the model eclipse and the
    data eclipse stop overlapping, so an optimizer started more than about one
    eclipse duration away keeps the wrong time and drives the depth to zero.
    For each trial epoch in the ``t0`` prior this solves the linear problem
    ``flux = trend + depth * eclipse`` (depth kept inside its prior) and returns
    ``(t0, depth, delta_chi2)`` for the best epoch, where ``delta_chi2`` is its
    improvement over a model with no eclipse. Returns ``None`` if ``t0`` is
    fixed or the surface model is not an eclipse or phase curve.
    """
    model = surface_config.get("model")
    if t0_spec.fixed or model not in {"eclipse", "phase_curve"}:
        return None
    from models.jaxoplanet.surface_basis import prepare_emission_light_curve_basis

    t = np.asarray(t, dtype=float)
    y = np.asarray(flux, dtype=float)
    w = 1.0 / np.broadcast_to(np.asarray(flux_err, dtype=float), y.shape)
    if t0_spec.prior == "gaussian":
        lo, hi = t0_spec.value - 5 * t0_spec.sigma, t0_spec.value + 5 * t0_spec.sigma
        lo = lo if t0_spec.low is None else max(lo, t0_spec.low)
        hi = hi if t0_spec.high is None else min(hi, t0_spec.high)
    else:
        lo, hi = t0_spec.low, t0_spec.high
    ref = float(t0_spec.value)

    # Unit-depth eclipse template on a fine grid of offsets from ``ref``;
    # a trial epoch shifts it rigidly (the orbit is fixed apart from timing).
    cadence = float(np.median(np.diff(np.sort(t))))
    step = min(cadence, 2.0 / 1440.0)
    grid = np.arange(t.min() - (hi - ref) - step, t.max() + (ref - lo) + 2 * step, step)
    fixed = dict(_surface_model="eclipse", period=period, t0=ref, a_rs=a_rs, b=b,
                 rors=rors, ecc=ecc, omega=omega, u=np.zeros(2))
    # ``uniform`` is 1 while the planet is visible and 0 in full eclipse.
    template = np.asarray(
        prepare_emission_light_curve_basis(fixed, grid, model="eclipse").uniform
    ).reshape(-1) - 1.0
    duration = step * np.count_nonzero(template < -1e-6)
    if duration <= 0:
        return None
    if model == "phase_curve":
        # A full-orbit phase curve contains transits, which this linear scan
        # does not model; drop every cadence a trial epoch could put in transit.
        reach = 0.5 * duration + max(hi - ref, ref - lo) + step
        offset = np.mod(t - ref + 0.5 * period, period) - 0.5 * period
        keep = np.abs(offset) > reach
        if not np.any(keep):
            return None
        t, y, w = t[keep], y[keep], w[keep]

    tn = t - t.min()
    trend = [np.ones_like(t), tn / max(tn.max(), 1e-12)]
    if any(name in detrending_type for name in ("quadratic", "cubic", "quartic", "gp", "spot")):
        trend.append(trend[1] ** 2)
    if model == "phase_curve":
        phase = 2 * np.pi * (t - ref) / period
        trend += [np.cos(phase), np.sin(phase)]
    trend_sets = [np.column_stack(trend)]
    if "explinear" in detrending_type:
        trend_sets = [np.column_stack(trend + [np.exp(-tn / tau)])
                      for tau in _scan_tau_grid(tau_range)]

    def chi2(design, target):
        coef, *_ = np.linalg.lstsq(design * w[:, None], target * w, rcond=None)
        return np.sum(((target - design @ coef) * w) ** 2), coef

    null = min(chi2(X, y)[0] for X in trend_sets)
    d_lo = 0.0 if depth_spec.low is None else float(depth_spec.low)
    d_hi = np.inf if depth_spec.high is None else float(depth_spec.high)
    best = (np.inf, ref, float(depth_spec.value))
    for trial in np.arange(lo, hi + 0.5 * step, max(step, duration / 20.0)):
        shape = np.interp(t - (trial - ref), grid, template)
        if not np.any(shape < -1e-6):
            continue  # this epoch puts the whole eclipse outside the data
        for X in trend_sets:
            value, coef = chi2(np.column_stack([X, shape]), y)
            depth = coef[-1]
            if not d_lo <= depth <= d_hi:
                depth = float(np.clip(depth, d_lo, d_hi))
                value = chi2(X, y - depth * shape)[0]
            if value < best[0]:
                best = (value, float(trial), float(depth))
    if not np.isfinite(best[0]):
        return None
    return best[1], best[2], float(null - best[0])


def _surface_basis_on_time_mask(basis, mask):
    """Select cadence leaves while preserving the exact surface-basis type."""
    if basis is None:
        return None
    mask = jnp.asarray(mask, dtype=bool)
    return type(basis)(*(
        None if field is None else field[..., mask]
        for field in basis
    ))


def _select_transit_eval_params(
    params,
    transit_engine,
    param_method,
    *,
    t=None,
    jaxoplanet_kernel=None,
    ld_profile=None,
    transit_window_optimization="off",
    surface_config=None,
):
    out = dict(params)
    if transit_engine != 'jaxoplanet':
        return out
    if param_method == 'a_rs':
        out.pop('duration', None)
    if t is not None and jaxoplanet_kernel is not None and ld_profile is not None:
        out = _attach_jaxoplanet_eval_metadata(
            out,
            t,
            jaxoplanet_kernel=jaxoplanet_kernel,
            ld_profile=ld_profile,
            transit_window_optimization=transit_window_optimization,
        )
    out = _attach_surface_eval_metadata(out, surface_config)
    return out

import jax
import jax.numpy as jnp
import numpy as np


_TRANSIT_WINDOW_INDEX_KEY = "_transit_window_indices"
_JAXOPLANET_KERNEL_KEY = "_jaxoplanet_kernel"
_LD_PROFILE_KEY = "_ld_profile"
_TRANSIT_PHASE_OFFSETS_KEY = "_transit_phase_offsets"
_TRANSIT_PHASE_MASK_KEY = "_transit_phase_mask"
_VALID_JAXOPLANET_KERNELS = frozenset(
    {
        "auto",
        "stock",
        "streamed",
        "fused",
        "native_power2",
        "quadratic_specialized",
        "quadratic_local_jvp",
    }
)


def resolve_jaxoplanet_kernel(kernel="auto", *, ld_profile=None, degree=None,
                              keplerian=False):
    """Resolve kernel routing, conservatively falling back to stock.

    The streamed and fused implementations are deliberately limited to the
    validated degree-12 power-2 polynomial approximation on duration-based
    orbits. ``auto`` remains on the lower-risk streamed implementation; the
    fused contraction is opt-in because its GPU benefit depends strongly on
    cadence count and batch shape.
    ``native_power2`` is an explicitly opt-in float64-only evaluator of the
    Power-2 law's native ``c, alpha`` coefficients.  It is limited to the
    duration orbit.  Both direct-u1/u2 specializations are opt-in and limited to degree-two
    quadratic duration-based orbits. ``quadratic_local_jvp`` is tuned for
    short fixed-LD GPU windows; it remains mathematically valid for free LD,
    but that use has no promised speedup. Keplerian and otherwise unsupported
    calls remain on stock jaxoplanet even when another kernel is requested.
    """
    kernel = str(kernel).lower()
    if kernel not in _VALID_JAXOPLANET_KERNELS:
        raise ValueError(
            "jaxoplanet_kernel must be one of "
            "{'auto', 'stock', 'streamed', 'fused', "
            "'native_power2', 'quadratic_specialized', "
            "'quadratic_local_jvp'}; "
            f"received {kernel!r}."
        )
    if kernel == "native_power2":
        if not bool(jax.config.x64_enabled):
            raise RuntimeError(
                "jaxoplanet_kernel='native_power2' requires JAX float64 "
                "(jax_enable_x64=True); its <=2 ppm validation does not "
                "cover float32 execution."
            )
        return "native_power2" if not keplerian and ld_profile == "power2" else "stock"
    if kernel in {"quadratic_specialized", "quadratic_local_jvp"}:
        return (
            kernel
            if not keplerian and ld_profile == "quadratic" and degree == 2
            else "stock"
        )
    supported = not keplerian and ld_profile == "power2" and degree == 12
    if not supported:
        return "stock"
    if kernel == "fused":
        return "fused"
    return "streamed" if kernel != "stock" else "stock"


def build_transit_phase_offsets(t, period, t0, duration):
    """Return channel-invariant phase offsets and duration masks.

    This is the exact phase convention used by jaxoplanet 0.1.0's
    ``TransitOrbit.relative_position``.  It is useful in spectroscopic models
    where the geometry is fixed across all wavelength channels.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    periods = jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64))
    t0s = jnp.atleast_1d(jnp.asarray(t0, dtype=jnp.float64))
    durations = jnp.atleast_1d(jnp.asarray(duration, dtype=jnp.float64))
    half_period = 0.5 * periods[:, None]
    dt = jnp.mod(t[None, :] - (t0s[:, None] - half_period), periods[:, None]) - half_period
    mask = jnp.fabs(dt) < 0.5 * durations[:, None]
    return dt, mask


def build_transit_window_indices(t, period, t0, duration):
    """Return the static union of fixed-duration transit windows.

    This helper is intentionally host-side: the returned integer array has a
    static shape and can therefore reduce the amount of compiled light-curve
    work. A small conservative tolerance includes boundary cadences; evaluating
    an extra boundary cadence is harmless because ``TransitOrbit`` still masks
    it to zero.
    """
    times = np.asarray(t, dtype=np.float64)
    periods = np.atleast_1d(np.asarray(period, dtype=np.float64))
    t0s = np.atleast_1d(np.asarray(t0, dtype=np.float64))
    durations = np.atleast_1d(np.asarray(duration, dtype=np.float64))

    if times.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    n_planets = max(periods.size, t0s.size, durations.size)

    def _broadcast_planet_array(name, value):
        if value.size == n_planets:
            return value
        if value.size == 1:
            return np.repeat(value, n_planets)
        raise ValueError(
            f"{name} must contain one value or one value per planet; "
            f"received {value.size} values for {n_planets} planets."
        )

    periods = _broadcast_planet_array("period", periods)
    t0s = _broadcast_planet_array("t0", t0s)
    durations = _broadcast_planet_array("duration", durations)
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("period values must be finite and positive.")
    if np.any(~np.isfinite(t0s)):
        raise ValueError("t0 values must be finite.")
    if np.any(~np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("duration values must be finite and non-negative.")

    active = np.zeros(times.shape, dtype=bool)
    scale = max(
        1.0,
        float(np.max(np.abs(times))) if times.size else 1.0,
        float(np.max(np.abs(t0s))),
        float(np.max(np.abs(periods))),
    )
    tolerance = 8.0 * np.finfo(np.float64).eps * scale
    for planet_period, planet_t0, planet_duration in zip(
        periods, t0s, durations
    ):
        half_period = 0.5 * planet_period
        dt = np.mod(times - (planet_t0 - half_period), planet_period) - half_period
        active |= np.abs(dt) <= 0.5 * planet_duration + tolerance

    return np.flatnonzero(active).astype(np.int32, copy=False)


def _keplerian_body_mass(period, a_rs):
    """Return the pseudo-body mass that makes semimajor and period consistent."""
    from jaxoplanet.orbits.keplerian import constants

    period = jnp.asarray(period, dtype=jnp.float64)
    a_rs = jnp.asarray(a_rs, dtype=jnp.float64)
    return 4.0 * jnp.pi**2 * a_rs**3 / (constants.G * period**2)


def _compute_transit_model_keplerian(params, t):
    from jaxoplanet.light_curves import limb_dark_light_curve
    from jaxoplanet.orbits.keplerian import Body, Central, System

    periods = jnp.atleast_1d(jnp.asarray(params["period"], dtype=jnp.float64))
    a_rss = jnp.atleast_1d(jnp.asarray(params["a_rs"], dtype=jnp.float64))
    t0s = jnp.atleast_1d(jnp.asarray(params["t0"], dtype=jnp.float64))
    bs = jnp.atleast_1d(jnp.asarray(params["b"], dtype=jnp.float64))
    rorss = jnp.atleast_1d(jnp.asarray(params["rors"], dtype=jnp.float64))
    eccs = jnp.atleast_1d(jnp.asarray(params.get("ecc", 0.0), dtype=jnp.float64))
    omegas = jnp.atleast_1d(jnp.asarray(params.get("omega", 0.0), dtype=jnp.float64))

    if eccs.size == 1 and periods.size > 1:
        eccs = jnp.repeat(eccs, periods.size)
    if omegas.size == 1 and periods.size > 1:
        omegas = jnp.repeat(omegas, periods.size)

    central = Central(radius=1.0, mass=0.0)
    light_curves = []
    for i in range(int(periods.shape[0])):
        body_mass = _keplerian_body_mass(periods[i], a_rss[i])
        orbit = System(central=central).add_body(
            Body(
                time_transit=t0s[i],
                semimajor=a_rss[i],
                impact_param=bs[i],
                eccentricity=eccs[i],
                omega_peri=omegas[i],
                mass=body_mass,
                radius=rorss[i],
            )
        )
        lc = jnp.asarray(limb_dark_light_curve(orbit, params["u"])(t))
        if lc.ndim == 1:
            light_curves.append(lc)
        else:
            light_curves.append(jnp.sum(lc, axis=-1))

    return jnp.sum(jnp.stack(light_curves, axis=0), axis=0)


def _compute_transit_model_duration(params, t, *, kernel="stock"):
    from jaxoplanet.light_curves import limb_dark_light_curve
    from jaxoplanet.orbits.transit import TransitOrbit

    periods = jnp.atleast_1d(params["period"])
    t0s = jnp.atleast_1d(params["t0"])
    bs = jnp.atleast_1d(params["b"])
    rorss = jnp.atleast_1d(params["rors"])
    durations = jnp.atleast_1d(params["duration"])

    phase_offsets = params.get(_TRANSIT_PHASE_OFFSETS_KEY)
    phase_mask = params.get(_TRANSIT_PHASE_MASK_KEY)

    if kernel == "native_power2":
        from .experimental_power2_native import light_curve as native_light_curve

        if "c1" not in params or "c2" not in params:
            raise ValueError(
                "jaxoplanet_kernel='native_power2' requires direct Power-2 "
                "coefficients params['c1'] and params['c2']."
            )
        c1 = jnp.asarray(params["c1"], dtype=jnp.float64)
        c2 = jnp.asarray(params["c2"], dtype=jnp.float64)
        if c1.size != 1 or c2.size != 1:
            raise ValueError(
                "A single light curve must provide scalar Power-2 c1 and c2 "
                "coefficients to native_power2."
            )
        c1 = jnp.reshape(c1, ())
        c2 = jnp.reshape(c2, ())

        if phase_offsets is None:
            phase_offsets, phase_mask = build_transit_phase_offsets(
                t, periods, t0s, durations
            )
        else:
            phase_offsets = jnp.asarray(phase_offsets, dtype=jnp.float64)
            if phase_offsets.ndim != 2 or phase_offsets.shape[0] != periods.shape[0]:
                raise ValueError(
                    "Precomputed transit phase offsets must have shape "
                    "(n_planets, n_times)."
                )
            if phase_mask is None:
                phase_mask = jnp.fabs(phase_offsets) < 0.5 * durations[:, None]
            else:
                phase_mask = jnp.asarray(phase_mask, dtype=bool)
                if phase_mask.shape != phase_offsets.shape:
                    raise ValueError(
                        "Precomputed transit phase mask shape must match offsets."
                    )

        def get_native_lc(duration, b, rors, dt, mask):
            speed = 2.0 * jnp.sqrt(
                jnp.maximum(0.0, jnp.square(1.0 + rors) - jnp.square(b))
            ) / duration
            separation = jnp.sqrt(jnp.square(speed * dt) + jnp.square(b))
            signal = native_light_curve(
                c1, c2, separation, rors, order=16
            )
            return jnp.where(mask, signal, 0.0)

        batched_lcs = jax.vmap(get_native_lc)(
            durations, bs, rorss, phase_offsets, phase_mask
        )
        return jnp.sum(batched_lcs, axis=0)

    if phase_offsets is not None:
        from jaxoplanet.core.limb_dark import light_curve as stock_light_curve
        from .limb_dark_streamed import light_curve as streamed_light_curve

        phase_offsets = jnp.asarray(phase_offsets, dtype=jnp.float64)
        if phase_offsets.ndim != 2 or phase_offsets.shape[0] != periods.shape[0]:
            raise ValueError(
                "Precomputed transit phase offsets must have shape "
                "(n_planets, n_times)."
            )
        if phase_mask is None:
            phase_mask = jnp.fabs(phase_offsets) < 0.5 * durations[:, None]
        else:
            phase_mask = jnp.asarray(phase_mask, dtype=bool)
            if phase_mask.shape != phase_offsets.shape:
                raise ValueError("Precomputed transit phase mask shape must match offsets.")

        if kernel == "streamed":
            lc_kernel = streamed_light_curve
        elif kernel == "fused":
            from .limb_dark_fused import light_curve as lc_kernel
        elif kernel == "quadratic_specialized":
            from .limb_dark_quadratic import light_curve as lc_kernel
        elif kernel == "quadratic_local_jvp":
            from .limb_dark_quadratic_local_jvp import light_curve as lc_kernel
        else:
            lc_kernel = stock_light_curve

        def get_lc_from_phase(duration, b, rors, dt, mask):
            speed = 2 * jnp.sqrt(
                jnp.maximum(0, jnp.square(1 + rors) - jnp.square(b))
            ) / duration
            separation = jnp.sqrt(jnp.square(speed * dt) + jnp.square(b))
            lc = lc_kernel(params["u"], separation, rors, order=10)
            return jnp.where(mask, lc, 0)

        batched_lcs = jax.vmap(get_lc_from_phase)(
            durations, bs, rorss, phase_offsets, phase_mask
        )
        return jnp.sum(batched_lcs, axis=0)

    if kernel == "streamed":
        from .limb_dark_streamed import limb_dark_light_curve
    elif kernel == "fused":
        from .limb_dark_fused import limb_dark_light_curve
    elif kernel == "quadratic_specialized":
        from .limb_dark_quadratic import limb_dark_light_curve
    elif kernel == "quadratic_local_jvp":
        from .limb_dark_quadratic_local_jvp import limb_dark_light_curve
    else:
        from jaxoplanet.light_curves import limb_dark_light_curve

    def get_lc(period, duration, t0, b, rors):
        orbit = TransitOrbit(
            period=period,
            duration=duration,
            time_transit=t0,
            impact_param=b,
            radius_ratio=rors,
        )
        return limb_dark_light_curve(orbit, params["u"])(t)

    batched_lcs = jax.vmap(get_lc)(periods, durations, t0s, bs, rorss)
    return jnp.sum(batched_lcs, axis=0)


def compute_transit_model(params, t, *, kernel=None, ld_profile=None):
    """
    Transit model for one or more planets using jaxoplanet.

    Expects params to contain 'period', 't0', 'b', 'rors', and either
    'duration' or ('a_rs', with optional 'ecc' and 'omega'). These should be
    arrays where the 0-th dimension is the planet index, except 'u' which is
    the limb darkening parameter vector. ``native_power2`` instead consumes
    scalar direct Power-2 coefficients ``c1`` and ``c2``.
    """
    surface_model = params.get("_surface_model", "transit")
    spots = params.get("_stellar_spots", ())
    surface_basis = params.get("_surface_basis")
    if surface_basis is not None:
        from .surface_basis import (
            compute_emission_basis_model,
            compute_spot_basis_model,
        )
        if hasattr(surface_basis, "uniform"):
            return compute_emission_basis_model(params, surface_basis)
        return compute_spot_basis_model(params, surface_basis)
    if surface_model != "transit" or spots:
        if "a_rs" not in params:
            raise ValueError(
                "Eclipse, phase-curve, and stellar-spot evaluation requires a_rs geometry."
            )
        if "u" not in params:
            raise ValueError("JAXoplanet surface evaluation requires params['u'].")
        from .surface import compute_surface_model
        return compute_surface_model(
            params, t, model=surface_model, spots=spots
        )

    requested_kernel = params.get(_JAXOPLANET_KERNEL_KEY, "auto") if kernel is None else kernel
    profile = params.get(_LD_PROFILE_KEY) if ld_profile is None else ld_profile
    degree = int(jnp.shape(params["u"])[-1]) if "u" in params else None
    selected_kernel = resolve_jaxoplanet_kernel(
        requested_kernel,
        ld_profile=profile,
        degree=degree,
        keplerian="a_rs" in params,
    )

    if "a_rs" in params:
        if "u" not in params:
            raise ValueError(
                "Keplerian/a_rs transit evaluation requires polynomial or "
                "quadratic params['u']; native_power2 supports only the "
                "duration parameterization."
            )
        return _compute_transit_model_keplerian(params, t)

    if selected_kernel != "native_power2" and "u" not in params:
        raise ValueError(
            "The selected jaxoplanet transit kernel requires params['u']; "
            "direct c1/c2 coefficients are accepted only by native_power2."
        )

    indices = params.get(_TRANSIT_WINDOW_INDEX_KEY)
    if indices is None:
        return _compute_transit_model_duration(params, t, kernel=selected_kernel)

    indices = jnp.asarray(indices, dtype=jnp.int32)
    if indices.ndim != 1:
        raise ValueError("Transit-window indices must be one-dimensional.")
    if indices.shape[0] == 0:
        return jnp.zeros_like(t, dtype=jnp.result_type(t, jnp.float64))

    active_params = params
    if _TRANSIT_PHASE_OFFSETS_KEY in params:
        active_params = dict(params)
        active_params[_TRANSIT_PHASE_OFFSETS_KEY] = params[_TRANSIT_PHASE_OFFSETS_KEY][:, indices]
        if _TRANSIT_PHASE_MASK_KEY in params:
            active_params[_TRANSIT_PHASE_MASK_KEY] = params[_TRANSIT_PHASE_MASK_KEY][:, indices]
    active_flux = _compute_transit_model_duration(
        active_params, t[indices], kernel=selected_kernel
    )
    return jnp.zeros_like(t, dtype=active_flux.dtype).at[indices].set(active_flux)

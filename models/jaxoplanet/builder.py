
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np

from ..model_building import (
    polynomial_order, sample_whitelight_trend, sample_sing_coefficients,
    sample_power2_coefficients, sample_quadratic_uniform, sample_kipping_coefficients, sample_quadratic_fixed_u1,
    ld_prior_scale,
    _enforce_decorrelated_coefficient_support,
)
from ..model_building import posterior_geometry, sample_orbital_geometry
from ..limb_darkening_config import GAUSSIAN_LD_WIDTH, validate_gaussian_ld_width, validate_uniform_ld_bounds
from ..configuration_support import validate_power2_support
from ..common import apply_systematics, get_I_power2
from ..cadence_reduction import linear_spectro_trend_coefficient_names
from ..linear_marginalization import (
    marginalized_log_likelihood_and_conditional, marginalized_log_likelihood_with_oot,
)
from ..ld_parameterization import Power2MaxtedTransform
from ..trend_marginal import build_marginalized_trend_design, normalized_polynomial_transform
from ..detrend import resolve_detrend_kernel, _split_components, _prepare_power2_poly
from ..gp import resolve_gp_builder, resolve_gp_solver
from ..trends import resolve_whitelight_trend_parameterization
# Load transit_grid eagerly: its module-level power-2 projection calls
# np.linalg.pinv, which fails if the first import happens inside a jit trace.
from . import transit_grid as _transit_grid  # noqa: F401


_AUTO_CADENCE_REDUCTION_THRESHOLD = 5000


from .core import (
    build_transit_phase_offsets,
    compute_transit_model,
    compute_transit_model_window,
    resolve_jaxoplanet_kernel,
)
from .config import surface_science_sites

# Transit-grid projections need 64-bit precision before their arrays are created.
jax.config.update("jax_enable_x64", True)

NUTS_KWARGS = {
    "dense_mass": False,
    "regularize_mass_matrix": True,
    "target_accept_prob": 0.8,
}


def _sample_surface_parameters(surface_config, n_planets, num_lcs=None):
    """Sample the compact surface-model priors and expose output-friendly units."""
    config = surface_config or {"model": "transit", "spots": ()}
    model = config.get("model", "transit")
    if model == "transit" and not config.get("spots"):
        return {}
    result = {"_surface_model": model, "_stellar_spots": config.get("spots", ())}
    shape = () if num_lcs is None else (int(num_lcs),)

    def sample_surface_site(name):
        specs = config[f"{name}_spec"]
        values = [sample_parameter(specs[index], f"_{name}_{index}", shape)
                  for index in range(n_planets)]
        value = numpyro.deterministic(name, jnp.stack(values, axis=0 if num_lcs is None else 1))
        numpyro.deterministic(f"{name}_ppm", value * 1e6)
        result[name] = value
        return value

    if model == "eclipse":
        sample_surface_site("eclipse_depth")
    elif model == "phase_curve":
        from .surface import phase_curve_derived_quantities

        depth = sample_surface_site("eclipse_depth")
        c1 = sample_surface_site("phase_c1")
        d1 = sample_surface_site("phase_d1")
        derived = phase_curve_derived_quantities(depth, c1, d1)
        numpyro.deterministic("nightside_flux_ppm", derived["nightside_flux"] * 1e6)
        numpyro.deterministic("phase_amplitude_ppm", derived["phase_amplitude"] * 1e6)
        numpyro.deterministic("phase_offset_deg", derived["phase_offset_deg"])

    spots = config.get("spots", ())
    if spots:
        contrasts = []
        for index, spot in enumerate(spots):
            center = float(spot["contrast"])
            width = float(spot.get("contrast_prior_width", 0.0))
            if width > 0.0:
                contrast = numpyro.sample(
                    f"_stellar_spot_contrast_{index}",
                    dist.TruncatedNormal(center, width, low=0.0, high=1.0).expand(shape),
                )
            else:
                contrast = jnp.full(shape, center, dtype=jnp.float64)
            contrasts.append(contrast)
        axis = 0 if num_lcs is None else 1
        result["stellar_spot_contrast"] = numpyro.deterministic(
            "stellar_spot_contrast", jnp.stack(contrasts, axis=axis)
        )
        result["stellar_rotation_period"] = jnp.asarray(
            config["stellar_rotation_period"], dtype=jnp.float64
        )
    return result


LD_MAP_COEFFICIENTS = 0
LD_MAP_POWER2_MAXTED = 1
LD_MAP_QUADRATIC_SUMDIFF = 2
LD_MAP_SING = 3


def _finite_log_density(value):
    """Turn support violations into the finite rejection value used by NUTS."""
    return jnp.where(jnp.isfinite(value), value, -1.0e100)


def _ld_variant_data_form(latent, center, scale, low, high, map_code, law,
                          latent_low=None, latent_high=None):
    """Return physical LD coefficients and the exact data-selected log prior.

    ``scale == 0`` denotes fixed coefficients, ``scale < 0`` a uniform
    distribution, and ``scale > 0`` a truncated-normal distribution.  The
    latent always has a standard-normal base density; ``log_correction``
    replaces it with the requested coordinate density.  For fixed LD the
    latent remains an independent standard-normal nuisance variable.
    """
    latent = jnp.asarray(latent, dtype=jnp.float64)
    center = jnp.broadcast_to(jnp.asarray(center, dtype=latent.dtype), latent.shape)
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=latent.dtype), latent.shape)
    low = jnp.broadcast_to(jnp.asarray(low, dtype=latent.dtype), latent.shape)
    high = jnp.broadcast_to(jnp.asarray(high, dtype=latent.dtype), latent.shape)
    code = jnp.asarray(map_code, dtype=jnp.int32)
    base_logp = jnp.sum(dist.Normal(0.0, 1.0).log_prob(latent), axis=-1)
    bounded = latent_low is not None and latent_high is not None
    if bounded:
        latent_low = jnp.broadcast_to(
            jnp.asarray(latent_low, dtype=latent.dtype), latent.shape
        )
        latent_high = jnp.broadcast_to(
            jnp.asarray(latent_high, dtype=latent.dtype), latent.shape
        )
        unit = jax.nn.sigmoid(latent)
        coordinate = latent_low + (latent_high - latent_low) * unit
        coordinate_log_jacobian = jnp.sum(
            jnp.log(latent_high - latent_low)
            + jax.nn.log_sigmoid(latent)
            + jax.nn.log_sigmoid(-latent), axis=-1,
        )
    else:
        coordinate = latent
        coordinate_log_jacobian = jnp.zeros_like(base_logp)

    def coordinate_density(x, lo, hi, loc, scl):
        uniform_logp = jnp.sum(dist.Uniform(lo, hi).log_prob(x), axis=-1)
        safe_scale = jnp.where(scl > 0.0, scl, 1.0)
        normal_logp = jnp.sum(
            dist.TruncatedNormal(loc, safe_scale, low=lo, high=hi).log_prob(x), axis=-1,
        )
        fixed = jnp.all(scl == 0.0, axis=-1)
        uniform = jnp.all(scl < 0.0, axis=-1)
        target = jnp.where(uniform, uniform_logp, normal_logp)
        # These densities replace an unconstrained Normal site's base density.
        # NumPyro log_prob does not mask support with validation disabled;
        # enforce the declared box even for the unbounded data-form API.
        supported = jnp.all(jnp.isfinite(x) & (x >= lo) & (x <= hi), axis=-1)
        target = jnp.where(supported, _finite_log_density(target), -1.0e100)
        return jnp.where(fixed, base_logp, target)

    def supported_box(x, lo, hi):
        return jnp.all(jnp.isfinite(x) & (x >= lo) & (x <= hi), axis=-1)

    def coefficients(_):
        fixed = jnp.all(scale == 0.0, axis=-1)
        physical = jnp.where(fixed[..., None], center, coordinate)
        target = coordinate_density(coordinate, low, high, center, scale)
        return (physical, jnp.where(fixed, base_logp, target + coordinate_log_jacobian),
                fixed | supported_box(coordinate, low, high))

    def maxted(_):
        transform = Power2MaxtedTransform()
        if bounded:
            # Bound the unconstrained sampler through the physical coefficient
            # box, then map to h. Pulling the standalone h density back through
            # h(c) cancels its change-of-variables Jacobian exactly.
            physical = coordinate
            h = transform(physical)
            physical_logp = coordinate_density(physical, low, high, center, scale)
            h_logp = physical_logp - transform.log_abs_det_jacobian(physical, h)
            pulled_back = h_logp + transform.log_abs_det_jacobian(physical, h)
            return (physical, _finite_log_density(
                pulled_back + coordinate_log_jacobian
            ), supported_box(physical, low, high))
        physical = transform.inv(coordinate)
        physical_logp = coordinate_density(physical, low, high, center, scale)
        jacobian = transform.log_abs_det_jacobian(physical, coordinate)
        return (physical, _finite_log_density(physical_logp - jacobian),
                supported_box(physical, low, high))

    def sumdiff(_):
        physical = jnp.stack(
            ((coordinate[..., 0] + coordinate[..., 1]) / 2.0,
             (coordinate[..., 0] - coordinate[..., 1]) / 2.0), axis=-1,
        )
        return (physical, coordinate_density(
            coordinate, low, high, center, scale
        ) + coordinate_log_jacobian, supported_box(coordinate, low, high))

    def sing(_):
        limb_l = coordinate[..., 0]
        u_plus = 1.0 - limb_l
        if bounded:
            limb_delta = (2.0 * coordinate[..., 1] - 1.0) * u_plus / 4.0
            conditional_log_jacobian = jnp.log(u_plus / 2.0)
        else:
            limb_delta = coordinate[..., 1]
            conditional_log_jacobian = jnp.zeros_like(u_plus)
        physical = jnp.stack(
            (u_plus - 4.0 * limb_delta, 4.0 * limb_delta), axis=-1
        )
        l_logp = dist.TruncatedNormal(center[..., 0], jnp.maximum(scale[..., 0], 1e-12), low=low[..., 0], high=high[..., 0]).log_prob(
            limb_l,
        )
        delta_logp = dist.TruncatedNormal(center[..., 1], jnp.maximum(scale[..., 1], 1e-12), low=-u_plus / 4.0, high=u_plus / 4.0).log_prob(
            limb_delta,
        )
        supported = (jnp.isfinite(limb_l) & jnp.isfinite(limb_delta)
                     & (limb_l >= low[..., 0]) & (limb_l <= high[..., 0])
                     & (u_plus > 0.) & (limb_delta >= -u_plus / 4.)
                     & (limb_delta <= u_plus / 4.))
        target = _finite_log_density(
            l_logp + delta_logp + conditional_log_jacobian + coordinate_log_jacobian,
        )
        return physical, target, supported

    if law == "power2":
        branches = (coefficients, maxted, coefficients, coefficients)
    elif law == "quadratic":
        branches = (coefficients, coefficients, sumdiff, sing)
    else:
        raise ValueError(f"Unsupported LD law for data form: {law}")
    physical, target_logp, supported = jax.lax.switch(code, branches, operand=None)
    # A true support violation has zero probability. Keep -inf outside the
    # numerical finite-rejection helper so initialisation cannot accept an
    # out-of-prior Normal auxiliary draw as a merely low-density start.
    return physical, jnp.where(supported, target_logp - base_logp, -jnp.inf)


def _resolve_builder_kernel(jaxoplanet_kernel, ld_profile, param_method, *, ld_mode=None):
    """Validate the retained model-level kernel choices and resolve the route."""
    del ld_mode
    requested = str(jaxoplanet_kernel).lower()
    degree = 12 if ld_profile == "power2" else 2 if ld_profile == "quadratic" else None
    return resolve_jaxoplanet_kernel(
        requested, ld_profile=ld_profile, degree=degree, keplerian=param_method == 'a_rs',
    )


def derive_geometry(wl_samples, period, ecc=0.0, omega=0.0):
    """Derive orbital geometry from the supported white-light parameterizations."""
    return posterior_geometry(wl_samples, period, ecc, omega)


_WHITELIGHT_PARAMETER_PRIOR_KEYS = (
    'period', 't0', 'eclipse_time', 'b', 'rprs', 'duration', 'a_rs', 'ecc', 'omega',
)


from models.priors import spec_distribution as _spec_distribution, sample_parameter


def _validate_parameter_priors(parameter_priors, n_planets, param_method):
    """Check the per-planet parameter specifications at factory time."""
    if not parameter_priors:
        raise ValueError(
            "parameter_priors is required: every planet parameter must carry "
            "a {value, prior, ...} specification (see koala.config)."
        )
    resolved = {}
    for name, specs in parameter_priors.items():
        if name not in _WHITELIGHT_PARAMETER_PRIOR_KEYS:
            raise ValueError(
                f"parameter_priors has no white-light site for planet.{name}."
            )
        specs = tuple(specs)
        if len(specs) != n_planets:
            raise ValueError(
                f'planet.{name}: expected {n_planets} specifications, received {len(specs)}.',
            )
        for spec in specs:
            if name in {'ecc', 'omega'} and spec.free:
                raise ValueError(f"planet.{name} may only be fixed for now.")
            if spec.free and spec.prior != 'log_uniform':
                _spec_distribution(spec)
        unused = (
            (name == 'duration' and param_method == 'a_rs')
            or (name == 'a_rs' and param_method == 'duration')
        )
        if unused and any(spec.free for spec in specs):
            other = 'a_rs' if name == 'duration' else 'duration'
            raise ValueError(
                f"planet.{name} is free but the white-light model is "
                f"parameterized by {other}; give planet.{other} the prior instead."
            )
        resolved[name] = specs
    for name in ('period', 't0', 'b', 'rprs',
                 'duration' if param_method == 'duration' else 'a_rs'):
        if name not in resolved:
            raise ValueError(f"parameter_priors is missing planet.{name}.")
    return resolved


def _validate_model_options(
    param_method, surface_config, surface_basis, ld_mode,
    ld_parameterization, ld_uniform_basis, ld_uniform_coefficient_bounds,
):
    """Validate shared geometry, surface-basis, and limb-darkening choices."""
    if param_method not in ('duration', 'a_rs'):
        raise ValueError(f"Unknown param_method: {param_method}")
    surface_config = surface_config or {"model": "transit", "spots": ()}
    basis_surface_valid = (
        (surface_config.get("model") == "transit" and surface_config.get("spots"))
        or (
            surface_config.get("model") in {"eclipse", "phase_curve"}
            and not surface_config.get("spots")
        )
    )
    if surface_basis is not None and not (
        basis_surface_valid
        and not surface_config.get("fit_geometry", True)
        and ld_mode == "fixed"
    ):
        raise ValueError(
            "surface_basis requires a supported stellar-spot or emission model "
            "with fixed geometry and fixed limb darkening."
        )
    if ld_parameterization not in {'coefficients', 'decorrelated', 'maxted_smooth'}:
        raise ValueError(
            "ld_parameterization must be 'coefficients', 'decorrelated' or 'maxted_smooth'."
        )
    if ld_uniform_basis not in {'uplus_uminus', 'coefficients', 'kipping'}:
        raise ValueError("ld_uniform_basis must be 'uplus_uminus', 'coefficients' or 'kipping'.")
    uniform_coefficient_low, uniform_coefficient_high = validate_uniform_ld_bounds(
        ld_uniform_coefficient_bounds)
    return surface_config, basis_surface_valid, uniform_coefficient_low, uniform_coefficient_high


def create_whitelight_model(
    detrend_type='linear', n_planets=1, ld_profile='quadratic', ld_mode='gaussian',
    param_method='duration', jaxoplanet_kernel='auto', surface_config=None, surface_basis=None,
    ld_parameterization='coefficients', ld_uniform_basis='uplus_uminus',
    ld_uniform_coefficient_bounds=(0.0, 1.0), step_width_mode='free',
    trend_parameterization='physical', two_spot_ordering='legacy', ld_variant_as_data=False,
    gp_solver=None, gp_assume_sorted=False, parameter_priors=None,
    power2_evaluation='auto', trend_priors=None, gaussian_ld_width=GAUSSIAN_LD_WIDTH,
):
    """Jaxoplanet white-light model with duration- or a_rs-based geometry.

    ``parameter_priors`` maps every ``planet`` key (``period``, ``t0``,
    ``b``, ``rprs``, ``duration`` or ``a_rs``, ``ecc``, ``omega``) to one
    ``ParameterSpec`` per planet (see :mod:`koala.config`). A fixed
    specification becomes a deterministic site; a free one is sampled from
    its prior by :func:`models.priors.sample_parameter`.
    """
    if ld_mode == "fixed_u1" and ld_profile != "quadratic":
        raise ValueError("fixed_u1 requires quadratic limb darkening.")
    gaussian_ld_width = validate_gaussian_ld_width(gaussian_ld_width)
    surface_config, basis_surface_valid, uniform_coefficient_low, uniform_coefficient_high = (
        _validate_model_options(
            param_method, surface_config, surface_basis, ld_mode,
            ld_parameterization, ld_uniform_basis, ld_uniform_coefficient_bounds,
        )
    )
    validate_power2_support(ld_profile, model=surface_config.get('model', 'transit'),
                            spots=surface_config.get('spots', ()),
                            evaluation=power2_evaluation, interpolated=ld_mode == 'interpolated')
    if power2_evaluation == 'auto':
        power2_evaluation = ('exact' if surface_config.get('model') == 'transit'
                             and not surface_config.get('spots')
                             and ld_mode != 'interpolated' else 'polynomial')
    if power2_evaluation not in {'polynomial', 'exact'}:
        raise ValueError("power2_evaluation must be 'auto', 'polynomial', or 'exact'.")
    exact_power2 = ld_profile == 'power2' and power2_evaluation == 'exact'
    if exact_power2 and (surface_config.get('model') != 'transit' or surface_config.get('spots')):
        raise ValueError("Direct power-2 evaluation currently requires an unspotted transit.")
    configured_trend_parameterization = trend_parameterization
    two_spot_ordering = str(two_spot_ordering).strip().lower()
    if two_spot_ordering != 'legacy':
        raise ValueError("two_spot_ordering must be 'legacy'.")
    resolved_kernel = _resolve_builder_kernel(
        jaxoplanet_kernel, ld_profile, param_method, ld_mode=ld_mode
    )
    detrend_components = _split_components(detrend_type)
    # Resolve once at factory time, and only when a GP is actually built, so
    # non-GP trends never touch the solver policy (or its install check).
    resolved_gp_solver = resolve_gp_solver(gp_solver) if 'gp' in detrend_type else None
    if ld_profile == "power2":
        MUS, P = _prepare_power2_poly()


    parameter_priors = _validate_parameter_priors(
        parameter_priors, n_planets, param_method
    )

    def _whitelight_model(t, yerr, y=None, prior_params=None,
                          ld_center=None, ld_scale=None, ld_low=None,
                          ld_high=None, ld_map_code=0, ld_latent_low=None,
                          ld_latent_high=None):
        trend_parameterization = resolve_whitelight_trend_parameterization(
            configured_trend_parameterization
        )

        geometry = sample_orbital_geometry(
            parameter_priors, n_planets, param_method,
            geometry_fixed=not surface_config.get('fit_geometry', True),
        )

        if ld_variant_as_data:
            if any(value is None for value in (ld_center, ld_scale, ld_low, ld_high)):
                raise ValueError("LD data form requires center, scale, low, and high arrays.")
            ld_latent = numpyro.sample(
                'ld_variant_latent', dist.Normal(0.0, 1.0).expand([2]).to_event(1)
            )
            coefficients, correction = _ld_variant_data_form(
                ld_latent, ld_center, ld_scale, ld_low, ld_high,
                ld_map_code, ld_profile, ld_latent_low, ld_latent_high,
            )
            numpyro.factor('ld_variant_prior', correction)
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', coefficients)
            else:
                c1 = numpyro.deterministic('c1', coefficients[0])
                c2 = numpyro.deterministic('c2', coefficients[1])
                prof = get_I_power2(c1, c2, MUS)
                u = P @ (1.0 - prof)
        elif ld_profile == 'quadratic':
            if ld_mode in {'gaussian', 'stellarprior'}:
                u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
                u_sigma = ld_prior_scale(
                    u_prior, prior_params.get('u_sigma'), ld_mode,
                    sigma_name="prior_params['u_sigma']", gaussian_width=gaussian_ld_width,
                )
                if ld_uniform_basis == 'kipping':
                    u = sample_kipping_coefficients((), u_prior, u_sigma)
                else:
                    coefficient_prior = dist.TruncatedNormal(
                        loc=u_prior, scale=u_sigma, low=0.0, high=1.0
                    ).to_event(1)
                    u = numpyro.sample("u", coefficient_prior)
            elif ld_mode == 'uniform':
                u = sample_quadratic_uniform(
                    (), ld_uniform_basis, uniform_coefficient_low, uniform_coefficient_high,
                )
            elif ld_mode == 'fixed_u1':
                u = sample_quadratic_fixed_u1(prior_params['u'], (), uniform_coefficient_low, uniform_coefficient_high)
            elif ld_mode == 'fixed':
                u = numpyro.deterministic("u", jnp.asarray(prior_params['u'], dtype=jnp.float64))
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
        elif ld_profile == 'power2':
            c1, c2 = sample_power2_coefficients(
                jnp.asarray(prior_params['u'], dtype=jnp.float64), prior_params.get('u_sigma'),
                ld_mode, decorrelated=ld_parameterization == 'decorrelated',
                maxted_smooth=ld_parameterization == 'maxted_smooth',
                sigma_name="prior_params['u_sigma']", gaussian_width=gaussian_ld_width,
            )
            prof = get_I_power2(c1, c2, MUS)
            u = P @ (1.0 - prof)
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")

        jitter_spec = (trend_priors or {}).get('jitter')
        if jitter_spec is None:
            log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
            jitter = jnp.exp(log_jitter)
        else:
            from models.priors import sample_parameter
            jitter = sample_parameter(jitter_spec, 'jitter')
        error = numpyro.deterministic('error', jnp.sqrt(jitter ** 2 + yerr ** 2))

        params = {name: geometry[name] for name in ('period', 'duration', 't0', 'b', 'rors')}
        params.update(_jaxoplanet_kernel=jaxoplanet_kernel, _ld_profile=ld_profile, u=u)
        if jaxoplanet_kernel == 'quadratic_contact_safe':
            bounds = [abs(spec.value) if spec.fixed else
                      spec.high if spec.low is not None and spec.low >= 0. else None
                      for spec in parameter_priors['rprs']]
            if all(bound is not None for bound in bounds):
                params['_quadratic_radius_bound'] = max(map(float, bounds))
        if exact_power2:
            params.update(c1=c1, c2=c2, _power2_exact=True)
        params.update({name: geometry[name] for name in ('a_rs', 'ecc', 'omega')})
        params['_fixed_circular_orbit'] = all(spec.value == 0 for spec in parameter_priors['ecc'])
        params.update(_sample_surface_parameters(surface_config, n_planets))
        if surface_basis is not None:
            params["_surface_basis"] = surface_basis

        params.update(sample_whitelight_trend(
            t, prior_params, detrend_components, trend_parameterization,
            step_width_mode, two_spot_ordering, trend_priors=trend_priors,
        ))

        if 'gp' in detrend_type:
            gp_builder = resolve_gp_builder(detrend_type)
            gp = gp_builder(
                params, t, error, gp_solver=resolved_gp_solver, assume_sorted=gp_assume_sorted,
            )
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            try:
                lc_model = resolve_detrend_kernel(detrend_type)(params, t)
                numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
            except KeyError:
                raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model


def create_vectorized_model(
    detrend_type='linear', ld_mode='gaussian', trend_mode='free', n_planets=1,
    ld_profile='quadratic', param_method='duration', transit_window='off',
    transit_window_indices=None, cadence_reduction='off', transit_grid='off',
    transit_grid_nodes=769, transit_grid_non_grazing=False,
    transit_grid_outer_contact_safe=None, jaxoplanet_kernel='auto', surface_config=None,
    surface_basis=None, jitter_prior='lognormal', jitter_prior_scale=2.0,
    jitter_prior_center=0.5, ld_parameterization='coefficients',
    ld_uniform_basis='uplus_uminus', ld_uniform_coefficient_bounds=(0.0, 1.0),
    ld_variant_as_data=False, joint_geometry=False,
    power2_evaluation='auto', power2_table=None, trend_basis='elapsed_days',
    trend_bounds=None, gaussian_ld_width=GAUSSIAN_LD_WIDTH,
):
    """Jaxoplanet spectroscopy with a wavelength-independent Keplerian orbit.

    ``joint_geometry=True`` replaces the fixed white-light geometry with
    shared sites ``t0``, ``b``, and reference-radius ``duration`` (or ``a_rs``) sampled once
    per draw for every channel (``in_axes=None``).  Each shared site takes a
    ``Normal(mu_<name>, sigma_<name>)`` prior from the model keyword
    arguments; the period stays fixed.  The static cadence-reduction and
    transit-grid accelerations assume fixed geometry and are disabled.

    ``jitter_prior='lognormal'`` uses
    ``log_jitter ~ Normal(log(jitter_prior_center * median(yerr)),
      jitter_prior_scale)``.  The posterior is likelihood-dominated whenever the
      jitter is identified (posterior widths ~0.1-0.4 in log versus a prior
      width of 2.0 e-folds) and reduces to a smooth Gaussian tail instead of a
      plateau when it is not.  The site name, shape, and downstream
      ``total_error`` deterministic are unchanged.
    """
    if ld_mode == "fixed_u1" and ld_profile != "quadratic":
        raise ValueError("fixed_u1 requires quadratic limb darkening.")
    gaussian_ld_width = validate_gaussian_ld_width(gaussian_ld_width)
    surface_config, basis_surface_valid, uniform_coefficient_low, uniform_coefficient_high = (
        _validate_model_options(
            param_method, surface_config, surface_basis, ld_mode,
            ld_parameterization, ld_uniform_basis, ld_uniform_coefficient_bounds,
        )
    )
    joint_geometry = bool(joint_geometry)
    validate_power2_support(ld_profile, model=surface_config.get('model', 'transit'),
                            spots=surface_config.get('spots', ()),
                            evaluation=power2_evaluation, interpolated=ld_mode == 'interpolated')
    if power2_evaluation == 'auto':
        power2_evaluation = ('exact' if surface_config.get('model') == 'transit'
                             and not surface_config.get('spots')
                             and ld_mode != 'interpolated' else 'polynomial')
    if power2_evaluation not in {'polynomial', 'exact'}:
        raise ValueError("power2_evaluation must be 'auto', 'polynomial', or 'exact'.")
    exact_power2 = ld_profile == 'power2' and power2_evaluation == 'exact'
    if exact_power2 and (surface_config.get('model') != 'transit' or surface_config.get('spots')
                         or ld_mode == 'interpolated'):
        raise ValueError("Direct power-2 evaluation requires an unspotted transit with physical coefficients.")
    if power2_table is not None and (not exact_power2 or joint_geometry or n_planets != 1
                                    or param_method != 'duration'):
        raise ValueError("Power-2 tables require exact evaluation and one planet with fixed duration geometry.")
    if joint_geometry and not surface_config.get("fit_geometry", True):
        raise ValueError(
            "joint_geometry=True samples the transit geometry and therefore "
            "requires fit_geometry=True."
        )
    if jitter_prior != 'lognormal':
        raise ValueError("jitter_prior must be 'lognormal'.")
    jitter_prior_scale = float(jitter_prior_scale)
    jitter_prior_center = float(jitter_prior_center)
    if jitter_prior_scale <= 0.0 or jitter_prior_center <= 0.0:
        raise ValueError("jitter_prior_scale and jitter_prior_center must be > 0.")
    if trend_mode not in {'free', 'fixed', 'gaussian_marginalized'}:
        raise ValueError(f"Unknown trend_mode: {trend_mode}")
    resolved_kernel = _resolve_builder_kernel(
        jaxoplanet_kernel, ld_profile, param_method, ld_mode=ld_mode
    )
    if transit_window not in {'auto', 'off'}:
        raise ValueError("transit_window must be either 'auto' or 'off'.")
    cadence_reduction = str(cadence_reduction).lower()
    if cadence_reduction not in {'auto', 'off'}:
        raise ValueError("cadence_reduction must be either 'auto' or 'off'.")
    transit_grid = str(transit_grid).lower()
    if transit_grid not in {'auto', 'off'}:
        raise ValueError("transit_grid must be either 'auto' or 'off'.")
    transit_grid_nodes = int(transit_grid_nodes)
    if transit_grid_nodes < 13:
        raise ValueError("transit_grid_nodes must be at least 13.")
    # The handoff flag is conservative over the full radius-ratio prior.  A
    # grazing or near-grazing box needs the denser audited default; an explicit
    # larger setting is preserved.
    transit_grid_non_grazing = bool(transit_grid_non_grazing)
    if transit_grid_outer_contact_safe is None:
        # Backward compatibility for pre-widening stage dumps: the old
        # conservative non-grazing proof also guarantees ample outer-contact
        # separation over the radius prior.
        transit_grid_outer_contact_safe = transit_grid_non_grazing
    transit_grid_outer_contact_safe = bool(transit_grid_outer_contact_safe)
    if not transit_grid_non_grazing and ld_profile == 'power2':
        transit_grid_nodes = max(transit_grid_nodes, 2049)
    use_transit_window = (
        transit_window == 'auto'
        and transit_window_indices is not None
        and not joint_geometry
    )
    if transit_window_indices is not None:
        transit_window_indices = jnp.asarray(
            transit_window_indices, dtype=jnp.int32
        )
        if transit_window_indices.ndim != 1:
            raise ValueError("transit_window_indices must be one-dimensional.")
    detrend_components = _split_components(detrend_type)
    if trend_basis not in {'elapsed_days', 'batman_normalized'}:
        raise ValueError("trend_basis must be 'elapsed_days' or 'batman_normalized'.")
    if trend_basis == 'batman_normalized' and (detrend_type=='none' or trend_mode=='fixed'):
        raise ValueError('batman_normalized requires sampled or marginalized polynomial coefficients.')
    from ..spectro_trends import validate_spectral_trend_bounds
    trend_bounds=validate_spectral_trend_bounds(trend_bounds)
    if trend_bounds is not None and trend_mode!='free' and set(trend_bounds)-{'log_tau'}:
        raise ValueError('trend_bounds applies only to sampled coefficients; use trend_prior_scale for Gaussian integrals.')
    if 'log_tau' in (trend_bounds or {}) and 'explinear' not in detrend_components:
        raise ValueError('log_tau bounds require a sampled-timescale explinear trend.')
    linear_trend_names = linear_spectro_trend_coefficient_names(detrend_type)
    cadence_reduction_eligible = (
        cadence_reduction == 'auto'
        and use_transit_window
        and n_planets == 1
        and surface_config.get("model") == "transit"
        and not surface_config.get("spots")
        and trend_mode == 'free'
        and linear_trend_names is not None
        and not joint_geometry
    )
    transit_grid_eligible = (
        False  # Duration interpolation is incompatible with shared Keplerian orbits.
        and not joint_geometry
        and use_transit_window
        and n_planets == 1
        and param_method == 'duration'
        and surface_config.get("model") == "transit"
        and not surface_config.get("spots")
        and ld_profile in {'power2', 'quadratic'}
        and ld_mode != 'interpolated'
        and transit_grid_outer_contact_safe
    )
    unsupported_non_spectroscopic = {'spot', '2spot', 'linear_discontinuity'}
    if detrend_components & unsupported_non_spectroscopic:
        unsupported = ", ".join(sorted(detrend_components & unsupported_non_spectroscopic))
        raise ValueError(
            "Vectorized model does not support non-spectroscopic spot/jump trends "
            f"({unsupported}). Use the corresponding *_spectroscopic trend family instead."
        )
    compute_lc_kernel = resolve_detrend_kernel(detrend_type)
    # Timing options travel with existing model metadata, preserving the
    # constructor's positional API and later native-geometry extensions.
    free_t0 = bool(surface_config.get('spectro_free_t0', False))
    t0_half_width_days = float(surface_config.get('spectro_t0_half_width_days', 1.0 / 48.0))
    if free_t0:
        if joint_geometry:
            raise ValueError('free_t0 and joint_geometry are mutually exclusive.')
        if surface_config.get('model') != 'transit' or surface_config.get('spots'):
            raise ValueError('free_t0 requires an unspotted transit.')
        if not np.isfinite(t0_half_width_days) or t0_half_width_days <= 0:
            raise ValueError('t0_half_width_days must be finite and positive.')
        if power2_table is not None:
            raise ValueError('free_t0 cannot use a fixed-orbit power-2 table.')
    # With the CUDA kernel the fused score also beats the full likelihood on short light
    # curves; with JAX alone it does not, so CPU runs keep the full likelihood there.
    quadratic_score_route = (
        free_t0 and resolved_kernel == 'quadratic_contact_safe' and ld_profile == 'quadratic'
        and surface_config.get('fit_geometry', True)
    )
    if quadratic_score_route:
        from .quadratic_cuda import accelerated
        quadratic_score_route = accelerated()
    if ld_profile == "power2":
        MUS_LD, P_LD = _prepare_power2_poly()


    def _vectorized_model(
        t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None, mu_depths=None, PERIOD=None,
        trend_fixed=None, mu_a_rs=None, mu_ecc=0.0, mu_omega=0.0, ld_interpolated=None,
        ld_fixed=None, mu_u_ld=None, sigma_u_ld=None, gp_trend=None, spot_trend=None,
        spot_trend2=None, jump_trend=None, exp_trend=None, fixed_tau=None, trend_design=None,
        precomputed_yerr_per_lc=None, trend_prior_mean=None, trend_prior_scale=None,
        likelihood_mask=None, ld_center=None, ld_scale=None, ld_low=None, ld_high=None,
        ld_map_code=0, ld_latent_low=None, ld_latent_high=None, surface_basis_data=None,
        oot_reference_beta=None, oot_group_yerr=None, oot_group_count=None,
        oot_group_reference_sse=None, oot_group_x_reference_residual=None, oot_group_xx=None,
        sigma_t0=None, sigma_b=None, sigma_duration=None, sigma_a_rs=None,
        rors_reference=None, transit_separation=None,
    ):
        num_lcs = jnp.atleast_2d(yerr).shape[0]
        use_cadence_reduction = cadence_reduction_eligible and (
            int(jnp.shape(t)[0]) > _AUTO_CADENCE_REDUCTION_THRESHOLD
            or (quadratic_score_route and oot_reference_beta is not None)
        )
        use_transit_grid = (
            transit_grid_eligible
            and int(jnp.shape(t)[0]) > _AUTO_CADENCE_REDUCTION_THRESHOLD
        )
        def _shared_geometry_site(name, center, scale):
            # One draw per planet, shared by every channel in the fit.
            if center is None or scale is None:
                raise ValueError(
                    f"joint_geometry=True requires mu_{name} and sigma_{name}."
                )
            center = jnp.atleast_1d(jnp.asarray(center, dtype=jnp.float64))
            scale = jnp.broadcast_to(
                jnp.atleast_1d(jnp.asarray(scale, dtype=jnp.float64)), center.shape,
            )
            return numpyro.sample(name, dist.Normal(center, scale))

        if joint_geometry:
            t0s = _shared_geometry_site("t0", mu_t0, sigma_t0)
            bs = _shared_geometry_site("b", mu_b, sigma_b)
        else:
            t0s = mu_t0
            bs = mu_b

        if rors_reference is not None:
            rors_reference = jnp.atleast_1d(jnp.asarray(rors_reference, dtype=jnp.float64))
            if rors_reference.shape != (n_planets,):
                raise ValueError('rors_reference must contain one shared radius per planet, not per-channel radii.')
        if param_method == 'duration':
            # Duration is defined at ONE white-light/config reference radius.
            # It can be shared/sampled, but never redefined at a channel radius.
            if rors_reference is None and (joint_geometry or mu_a_rs is None):
                raise ValueError("duration spectroscopy requires rors_reference (white-light/config rprs), or param_method='a_rs' with mu_a_rs.")
            duration = (_shared_geometry_site("duration", mu_duration, sigma_duration)
                        if joint_geometry else mu_duration)
            from ..harmonica.core import harmonica_a_rs_from_duration
            if not joint_geometry and mu_a_rs is not None:
                a_rs = jnp.asarray(mu_a_rs, dtype=jnp.float64)
            else:
                a_rs = harmonica_a_rs_from_duration(
                    jnp.asarray(PERIOD), jnp.asarray(duration), jnp.asarray(bs),
                    jnp.asarray(rors_reference), ecc=jnp.asarray(mu_ecc), omega=jnp.asarray(mu_omega))
        else:
            if mu_a_rs is None:
                raise ValueError("mu_a_rs must be provided when param_method='a_rs'.")
            a_rs = (_shared_geometry_site("a_rs", mu_a_rs, sigma_a_rs)
                    if joint_geometry else jnp.asarray(mu_a_rs, dtype=jnp.float64))
        if joint_geometry:
            numpyro.deterministic("orbit_a_rs", a_rs)
        orbital_params = {"a_rs": a_rs, "ecc": jnp.asarray(mu_ecc, dtype=jnp.float64),
                          "omega": jnp.asarray(mu_omega, dtype=jnp.float64)}

        if surface_config.get("fit_geometry", True):
            rors = numpyro.sample(
                "rors",
                dist.Uniform(jnp.sqrt(1e-5), jnp.sqrt(0.5)).expand([num_lcs, n_planets]),
            )
        else:
            if mu_depths is None:
                raise ValueError("mu_depths is required when fit_geometry is false.")
            # Preserve the channel axis on every vector-model output.  This is
            # required when selective failed lanes move through the joint-NUTS
            # fallback and are merged back into a chunk.
            numpyro.deterministic(
                "_geometry_fixed", jnp.ones((num_lcs,), dtype=jnp.bool_)
            )
            fixed_rors = jnp.sqrt(jnp.asarray(mu_depths, dtype=jnp.float64))
            if fixed_rors.ndim == 1:
                fixed_rors = jnp.broadcast_to(fixed_rors, (num_lcs, n_planets))
            rors = numpyro.deterministic("rors", fixed_rors)
        depths = numpyro.deterministic("depths", rors ** 2)

        yerr_matrix = jnp.atleast_2d(
            jnp.asarray(yerr, dtype=jnp.float64)
        )
        if yerr_matrix.shape[0] == 1 and num_lcs > 1:
            yerr_matrix = jnp.broadcast_to(
                yerr_matrix, (num_lcs, yerr_matrix.shape[1])
            )
        if precomputed_yerr_per_lc is None:
            yerr_per_lc = jnp.nanmedian(yerr_matrix, axis=1)
        else:
            yerr_per_lc = jnp.asarray(
                precomputed_yerr_per_lc, dtype=jnp.float64
            )
        log_jitter = numpyro.sample(
            'log_jitter',
            dist.Normal(
                jnp.log(jitter_prior_center * yerr_per_lc),
                jnp.full((num_lcs,), jitter_prior_scale, dtype=jnp.float64),
            ),
        )
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter ** 2 + yerr_per_lc ** 2))
        # ``total_error`` is a compact per-channel posterior summary.  The
        # likelihood itself must retain the reported time-dependent errors;
        # replacing them with their median changes the statistical weighting
        # whenever a JWST integration has a different uncertainty.
        error_broadcast = jnp.sqrt(
            jitter[:, None] ** 2 + yerr_matrix ** 2
        )

        if ld_variant_as_data:
            if any(value is None for value in (ld_center, ld_scale, ld_low, ld_high)):
                raise ValueError("LD data form requires center, scale, low, and high arrays.")
            ld_latent = numpyro.sample(
                'ld_variant_latent', dist.Normal(0.0, 1.0).expand([num_lcs, 2]).to_event(1),
            )
            coefficients, correction = _ld_variant_data_form(
                ld_latent, ld_center, ld_scale, ld_low, ld_high,
                ld_map_code, ld_profile, ld_latent_low, ld_latent_high,
            )
            numpyro.factor('ld_variant_prior', jnp.sum(correction))
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', coefficients)
            else:
                c1 = numpyro.deterministic('c1', coefficients[:, 0])
                c2 = numpyro.deterministic('c2', coefficients[:, 1])
                profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                u = (P_LD @ (1.0 - profs).T).T
        elif ld_mode == 'interpolated':
            u = numpyro.deterministic('u', ld_interpolated)
        elif ld_profile == 'power2':
            c1, c2 = sample_power2_coefficients(
                ld_fixed if ld_mode == 'fixed' else mu_u_ld, sigma_u_ld, ld_mode,
                shape=(num_lcs,), decorrelated=ld_parameterization == 'decorrelated',
                maxted_smooth=ld_parameterization == 'maxted_smooth',
                joint_uniform=True, gaussian_width=gaussian_ld_width,
            )
            profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
            u = (P_LD @ (1.0 - profs).T).T
        elif ld_profile == 'quadratic':
            if ld_mode == 'sing':
                c1, c2, u = sample_sing_coefficients(mu_u_ld, sigma_u_ld)
            elif ld_mode in {'gaussian', 'stellarprior'}:
                scale = ld_prior_scale(mu_u_ld, sigma_u_ld, ld_mode, clipped=False, gaussian_width=gaussian_ld_width)
                if ld_uniform_basis == 'kipping':
                    u = sample_kipping_coefficients((num_lcs,), mu_u_ld, scale)
                else:
                    u = numpyro.sample('u', dist.TruncatedNormal(
                        loc=mu_u_ld, scale=scale, low=0.0, high=1.0,
                    ).to_event(1))
            elif ld_mode == 'uniform':
                u = sample_quadratic_uniform(
                    (num_lcs,), ld_uniform_basis, uniform_coefficient_low, uniform_coefficient_high,
                )
            elif ld_mode == 'fixed_u1':
                u = sample_quadratic_fixed_u1(ld_fixed, (num_lcs,), uniform_coefficient_low, uniform_coefficient_high)
            elif ld_mode == 'fixed':
                u = numpyro.deterministic('u', ld_fixed)
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")

        params = {
            "period": PERIOD, "t0": t0s, "b": bs,
            "rors": rors,
            "_jaxoplanet_kernel": jaxoplanet_kernel,
            "_ld_profile": ld_profile,
            **orbital_params,
        }
        in_axes = {
            "period": None, "t0": None, "b": None,
            "rors": 0,
            "_jaxoplanet_kernel": None,
            "_ld_profile": None,
        }
        params["u"] = u
        in_axes["u"] = 0
        if free_t0:
            # A small relative coordinate avoids subtracting large epochs in
            # the sampler. The physical orbit stays fixed at the white fit.
            dt0 = numpyro.sample('dt0', dist.Uniform(
                -t0_half_width_days, t0_half_width_days,
            ).expand([num_lcs, n_planets]))
            params['t0'] = numpyro.deterministic('t0', jnp.asarray(mu_t0) + dt0)
            in_axes['t0'] = 0

        if jaxoplanet_kernel == 'quadratic_contact_safe' and surface_config.get('fit_geometry', True):
            params['_quadratic_radius_bound'] = float(0.5 ** 0.5)
            in_axes['_quadratic_radius_bound'] = None
        if exact_power2:
            params.update(c1=c1, c2=c2, _power2_exact=True)
            in_axes.update(c1=0, c2=0, _power2_exact=None)
            if power2_table is not None:
                params['_power2_table'] = power2_table
                in_axes['_power2_table'] = None
                rlo, rhi = power2_table['radius_bounds']
                alo, ahi = power2_table['exponent_bounds']
                params['_power2_use_table'] = jnp.all(jnp.abs(bs) == power2_table['impact']) & jnp.all(
                    (rors >= rlo) & (rors <= rhi)
                    & (c2[:, None] >= alo) & (c2[:, None] <= ahi)
                )
                in_axes['_power2_use_table'] = None
        in_axes.update(a_rs=None, ecc=None, omega=None)
        if transit_separation is not None:
            if free_t0 or joint_geometry or surface_config.get('model') != 'transit' or surface_config.get('spots'):
                raise ValueError('Prepared separations require fixed geometry and an unspotted transit.')
            params['_transit_separation'] = jnp.atleast_2d(jnp.asarray(transit_separation))
            in_axes['_transit_separation'] = None
        if use_transit_window:
            params["_transit_window_indices"] = transit_window_indices
            in_axes["_transit_window_indices"] = None
        if use_transit_grid:
            params['_transit_grid_nodes'] = transit_grid_nodes
            in_axes['_transit_grid_nodes'] = None
            params['_transit_grid_contact_fallback_margin'] = (
                0.0
            )
            in_axes['_transit_grid_contact_fallback_margin'] = None
            params['_transit_grid_force_stock_kernel'] = bool(
                ld_profile == 'power2' and not transit_grid_non_grazing
            )
            in_axes['_transit_grid_force_stock_kernel'] = None
            if ld_profile == 'power2':
                params['_transit_grid_c1'] = c1
                params['_transit_grid_c2'] = c2
                in_axes['_transit_grid_c1'] = 0
                in_axes['_transit_grid_c2'] = 0
            else:
                params['_transit_grid_quadratic'] = True
                in_axes['_transit_grid_quadratic'] = None

        surface_params = _sample_surface_parameters(
            surface_config, n_planets, num_lcs=num_lcs
        )
        params.update(surface_params)
        active_surface_basis = (
            surface_basis_data
            if surface_basis_data is not None else surface_basis
        )
        if active_surface_basis is not None:
            if not (
                basis_surface_valid
                and not surface_config.get("fit_geometry", True)
                and ld_mode == "fixed"
            ):
                raise ValueError(
                    "surface_basis_data requires a supported stellar-spot or "
                    "emission model with fixed geometry and fixed limb darkening."
                )
            params["_surface_basis"] = active_surface_basis
            in_axes["_surface_basis"] = (
                0 if jnp.ndim(active_surface_basis.baseline) > 1 else None
            )
        if "_surface_model" in surface_params:
            in_axes["_surface_model"] = None
            in_axes["_stellar_spots"] = None
        for name in (
            "eclipse_depth", "phase_c1", "phase_d1", "stellar_spot_contrast",
        ):
            if name in surface_params:
                in_axes[name] = 0
        if "stellar_rotation_period" in surface_params:
            in_axes["stellar_rotation_period"] = None

        if trend_mode == 'gaussian_marginalized':
            if y is None:
                raise ValueError(
                    "Gaussian trend marginalization requires observed y values."
                )
            reduce_gaussian = (
                cadence_reduction == 'auto' and use_transit_window
                and n_planets == 1 and not joint_geometry
                and detrend_type in {'constant', 'linear', 'quadratic'}
                and surface_config.get('model') == 'transit'
                and not surface_config.get('spots')
                and all(value is not None for value in (
                    oot_reference_beta, oot_group_yerr, oot_group_count,
                    oot_group_reference_sse, oot_group_x_reference_residual, oot_group_xx))
            )
            tau = None
            if 'explinear_spectroscopic' in detrend_components:
                if exp_trend is None or fixed_tau is None:
                    raise ValueError("explinear_spectroscopic requires exp_trend and fixed_tau.")
                tau = numpyro.deterministic(
                    'tau', jnp.full((num_lcs,), jnp.asarray(fixed_tau, dtype=jnp.float64))
                )
            elif 'explinear' in detrend_components:
                log_tau = numpyro.sample(
                    'log_tau', dist.Uniform(*(trend_bounds or {}).get('log_tau',(jnp.log(0.001),jnp.log(0.1)))).expand([num_lcs]),
                )
                tau = numpyro.deterministic('tau', jnp.exp(log_tau))
            if reduce_gaussian:
                transit_model = jax.vmap(compute_transit_model_window, in_axes=(in_axes, None))(params, t)
                design, coefficient_names = build_marginalized_trend_design(detrend_type, t, num_lcs)
                design = design[:, transit_window_indices, :] * (1. + transit_model)[..., None]
            elif (use_transit_window and n_planets == 1
                    and surface_config.get('model') == 'transit'
                    and not surface_config.get('spots')):
                # The marginal integral still contains every observation, but
                # the transit is identically zero outside the same envelope
                # already used by sampled-trend inference. Avoid evaluating
                # its expensive derivatives on guaranteed empty cadences.
                active = jax.vmap(compute_transit_model_window, in_axes=(in_axes, None))(params, t)
                transit_model = jnp.zeros((num_lcs, t.size), dtype=jnp.float64)
                transit_model = transit_model.at[:, transit_window_indices].set(active)
            else:
                transit_model = jax.vmap(compute_transit_model, in_axes=(in_axes, None))(params, t)
            # F = (1 + transit) * (X @ beta): fold the transit factor into
            # every design column so the model is exactly linear in beta.
            if not reduce_gaussian:
                design, coefficient_names = build_marginalized_trend_design(
                    detrend_type, t, num_lcs, tau=tau, spot_trend=spot_trend,
                    spot_trend2=spot_trend2, jump_trend=jump_trend, exp_trend=exp_trend,
                    transit_factor=1.0 + transit_model,
                )
            num_coefficients = len(coefficient_names)
            transform = None
            if trend_basis == 'batman_normalized':
                from ..trend_marginal import normalized_linear_trend_transform
                transform = normalized_linear_trend_transform(
                    t, num_lcs, coefficient_names, likelihood_mask)
                design = jnp.matmul(design, transform)
            if trend_prior_mean is None:
                default_mean = jnp.zeros((num_coefficients,), dtype=jnp.float64)
                default_mean = default_mean.at[0].set(1.0)
                trend_prior_mean = jnp.broadcast_to(
                    default_mean, (num_lcs, num_coefficients)
                )
            else:
                trend_prior_mean = jnp.asarray(
                    trend_prior_mean, dtype=jnp.float64
                )
            if trend_prior_scale is None:
                trend_prior_scale = jnp.full(
                    (num_lcs, num_coefficients), 0.1, dtype=jnp.float64
                )
            else:
                trend_prior_scale = jnp.asarray(
                    trend_prior_scale, dtype=jnp.float64
                )
            observed = jnp.asarray(y, dtype=jnp.float64)
            noise_scale = error_broadcast
            if reduce_gaussian:
                reported = jnp.broadcast_to(jnp.atleast_2d(yerr), jnp.shape(jnp.atleast_2d(observed)))[:, transit_window_indices]
                if likelihood_mask is not None:
                    valid = jnp.broadcast_to(jnp.asarray(likelihood_mask,dtype=bool), jnp.shape(jnp.atleast_2d(observed)))[:, transit_window_indices]
                    reported = jnp.where(valid, reported, 1.)
                noise_scale = jnp.sqrt(reported**2 + jitter[:, None]**2)
            masked_count = 0.0
            if likelihood_mask is not None and not reduce_gaussian:
                # Padded/excluded cadences carry no information: give them a
                # zero row and unit noise, then remove their constant
                # -log(2 pi)/2 normalization, so the marginal likelihood is
                # exactly that of the unmasked integrations.
                mask = jnp.broadcast_to(
                    jnp.asarray(likelihood_mask, dtype=bool), jnp.shape(jnp.atleast_2d(observed))
                )
                observed = jnp.where(mask, observed, 0.0)
                design = jnp.where(mask[..., None], design, 0.0)
                noise_scale = jnp.where(mask, noise_scale, 1.0)
                masked_count = jnp.sum(~mask, axis=-1).astype(jnp.float64)
            if reduce_gaussian:
                log_likelihood, conditional = marginalized_log_likelihood_with_oot(
                    observed, design, noise_scale, trend_prior_mean, trend_prior_scale,
                    indices=transit_window_indices, jitter=jitter,
                    reference_beta=oot_reference_beta, group_yerr=oot_group_yerr,
                    group_count=oot_group_count, group_reference_sse=oot_group_reference_sse,
                    group_x_reference_residual=oot_group_x_reference_residual,
                    group_xx=oot_group_xx, transform=transform,
                    likelihood_mask=likelihood_mask,
                )
                # The reduced integral counts only actual observations.
                masked_count = 0.
            else:
                log_likelihood, conditional = marginalized_log_likelihood_and_conditional(
                    observed, design, noise_scale,
                    trend_prior_mean, prior_scale=trend_prior_scale,
                )
            log_likelihood = log_likelihood + 0.5 * masked_count * jnp.log(2.0 * jnp.pi)
            mean, factor = conditional.mean, conditional.factor
            if transform is not None:
                # Save both coordinate systems for audits, but reconstruct
                # ordinary c/v/v2 outputs in the existing elapsed-day basis.
                numpyro.deterministic('trend_normalized_mean', mean)
                numpyro.deterministic('trend_normalized_factor', factor)
                mean = jnp.einsum('...ij,...j->...i', transform, mean)
                factor = jnp.matmul(transform, factor)
            numpyro.deterministic('trend_beta_mean', mean)
            numpyro.deterministic('trend_beta_factor', factor)
            numpyro.factor('obs_marginalized', jnp.sum(log_likelihood))
            return

        if detrend_type != 'none':
            order = (0 if detrend_type in {'constant','gp_spectroscopic'}
                     else polynomial_order(detrend_components, 1))
            names = ['c'] + ['v' if degree == 1 else f'v{degree}' for degree in range(1, order + 1)]
            if trend_mode == 'fixed':
                trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                if 'explinear' in detrend_components:
                    names += ['A', 'tau']
                for index, name in enumerate(names):
                    # The fixed exponential columns retain their original indices.
                    column = {'A': 2, 'tau': 3}.get(name, index)
                    params[name] = numpyro.deterministic(name, trend_temp[:, column])
            elif trend_mode == 'free':
                if trend_basis == 'batman_normalized':
                    # Sample the reference's independent box in x=(t-mean)/span,
                    # then expose elapsed-day coefficients to all existing
                    # kernels, sufficient statistics and plotting consumers.
                    transform = normalized_polynomial_transform(
                        t, num_lcs, order, likelihood_mask)
                    beta = {}
                    for name in names:
                        bounds = (0.5, 1.5) if name == 'c' else (-1.0, 1.0)
                        bounds = (trend_bounds or {}).get(name, bounds)
                        beta[name] = numpyro.sample('trend_' + name, dist.Uniform(*bounds).expand([num_lcs]))
                    normalized = jnp.stack([beta[name] for name in names], axis=-1)
                    physical = jnp.einsum('...ij,...j->...i', transform, normalized)
                    for index, name in enumerate(names):
                        params[name] = numpyro.deterministic(name, physical[:, index])
                else:
                    for name in names:
                        bounds = (0.9, 1.1) if name == 'c' else (-0.1, 0.1)
                        bounds = (trend_bounds or {}).get(name, bounds)
                        params[name] = numpyro.sample(name, dist.Uniform(*bounds).expand([num_lcs]))
                if 'explinear_spectroscopic' in detrend_components:
                    if exp_trend is None or fixed_tau is None:
                        raise ValueError("explinear_spectroscopic requires exp_trend and fixed_tau.")
                    params['A'] = numpyro.sample('A', dist.Uniform(*(trend_bounds or {}).get('A',(-0.1,0.1))).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic(
                        'tau', jnp.full((num_lcs,), jnp.asarray(fixed_tau, dtype=jnp.float64))
                    )
                    names += ['A', 'tau']
                elif 'explinear' in detrend_components:
                    params['A'] = numpyro.sample('A', dist.Uniform(*(trend_bounds or {}).get('A',(-0.1,0.1))).expand([num_lcs]))
                    log_tau = numpyro.sample('log_tau', dist.Uniform(*(trend_bounds or {}).get('log_tau',(jnp.log(1e-3),jnp.log(1e-1)))).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                    names += ['A', 'tau']
            else:
                raise ValueError(f"Unknown trend_mode: {trend_mode}")
            in_axes.update(dict.fromkeys(names, 0))

        # Fixed-shape amplitudes must exist before the reduced-likelihood return.
        if trend_mode == 'free':
            names = []
            if '2spot_spectroscopic' in detrend_components:
                names += ['A_spot', 'A_spot2']
            elif 'spot_spectroscopic' in detrend_components:
                names.append('A_spot')
            if 'linear_discontinuity_spectroscopic' in detrend_components:
                names.append('A_jump')
            for name in names:
                params[name] = numpyro.sample(name, dist.Uniform(*(trend_bounds or {}).get(name, (0.5, 2))).expand([num_lcs]))
                in_axes[name] = 0

        resolved_trend_design = trend_design
        if use_cadence_reduction and resolved_trend_design is None:
            centered_time = t - jnp.min(t)
            external_bases = {
                'A': exp_trend,
                'A_spot': spot_trend,
                'A_spot2': spot_trend2,
                'A_jump': jump_trend,
            }
            columns = []
            for name in linear_trend_names:
                if name == 'c':
                    column = jnp.ones_like(t, dtype=jnp.float64)
                elif name == 'v':
                    column = centered_time
                elif name.startswith('v') and name[1:].isdigit():
                    column = centered_time ** int(name[1:])
                else:
                    column = external_bases[name]
                    if column is None:
                        raise ValueError(
                            f"cadence reduction requires the fixed {name} basis."
                        )
                columns.append(jnp.asarray(column, dtype=jnp.float64))
            resolved_trend_design = jnp.stack(columns, axis=1)

        if use_cadence_reduction:
            statistics = (
                oot_reference_beta,
                oot_group_yerr,
                oot_group_count,
                oot_group_reference_sse,
                oot_group_x_reference_residual,
                oot_group_xx,
            )
            if y is None or any(value is None for value in statistics):
                raise ValueError(
                    "cadence_reduction='auto' requires observed y and the "
                    "precomputed out-of-transit sufficient statistics."
                )
            indices = transit_window_indices
            # Sampled power-2 coefficients use the same transit VJP as the
            # unreduced model.  Fusing that VJP into the likelihood changes
            # rounding in the ill-conditioned degree-12 LD sensitivities.
            # The ordinary reduced branch below still uses the transit grid.
            use_fused_grid = use_transit_grid and (
                ld_profile != 'power2'
                or (ld_mode == 'fixed' and not ld_variant_as_data)
            ) and not exact_power2
            if use_fused_grid:
                from jaxoplanet.core.limb_dark import (
                    light_curve as stock_light_curve,
                )
                from .limb_dark_streamed import (
                    light_curve as streamed_light_curve,
                )
                from .transit_grid import (
                    grid_reduced_log_likelihood,
                )

                selected_kernel = resolve_jaxoplanet_kernel(
                    jaxoplanet_kernel, ld_profile=ld_profile, degree=int(jnp.shape(u)[-1]),
                    keplerian=False,
                )
                grid_kernel = (
                    streamed_light_curve
                    if (
                        selected_kernel == 'streamed'
                        and transit_grid_non_grazing
                    )
                    else stock_light_curve
                )
                active_y = jnp.atleast_2d(
                    jnp.asarray(y, dtype=jnp.float64)
                )[:, indices]
                # Square before gathering: jaxlib 0.6.2 constant folding can
                # mis-layout square(gather(constant)) for multiple channels,
                # silently mixing their reported variances.  The non-grid
                # branch likewise forms the full variance before gathering.
                active_variance = (yerr_matrix ** 2)[:, indices]
                if likelihood_mask is None:
                    active_mask = jnp.ones_like(active_y, dtype=bool)
                else:
                    active_mask = jnp.broadcast_to(
                        jnp.asarray(likelihood_mask, dtype=bool), jnp.shape(jnp.atleast_2d(y)),
                    )[:, indices]
                reference_beta = jnp.atleast_2d(jnp.asarray(
                    oot_reference_beta, dtype=jnp.float64
                ))
                group_error = jnp.atleast_2d(jnp.asarray(
                    oot_group_yerr, dtype=jnp.float64
                ))
                group_count = jnp.atleast_2d(jnp.asarray(
                    oot_group_count, dtype=jnp.float64
                ))
                group_reference_sse = jnp.atleast_2d(jnp.asarray(
                    oot_group_reference_sse, dtype=jnp.float64
                ))
                group_xr = jnp.asarray(
                    oot_group_x_reference_residual, dtype=jnp.float64
                )
                group_xx = jnp.asarray(oot_group_xx, dtype=jnp.float64)
                if group_xr.ndim == 2:
                    group_xr = group_xr[None, ...]
                    group_xx = group_xx[None, ...]
                design_active = jnp.asarray(
                    resolved_trend_design, dtype=jnp.float64
                )[indices]
                beta = jnp.stack(
                    tuple(params[name] for name in linear_trend_names), axis=1
                )
                if ld_profile == 'power2':
                    transit_theta = jnp.stack(
                        (rors[:, 0], c1, c2), axis=1
                    )
                else:
                    transit_theta = jnp.column_stack((rors[:, 0], u))
                theta = jnp.concatenate(
                    (transit_theta, beta, jitter[:, None]), axis=1
                )
                phase_active = params['_transit_phase_offsets'][0, indices]
                phase_mask_active = params['_transit_phase_mask'][0, indices]

                def lane_log_likelihood(
                    lane_theta, lane_u, lane_y, lane_variance, lane_mask, lane_reference_beta,
                    lane_group_error, lane_group_count, lane_reference_sse, lane_group_xr,
                    lane_group_xx,
                ):
                    return grid_reduced_log_likelihood(
                        grid_kernel, lane_theta, lane_u, phase_active, phase_mask_active,
                        design_active, lane_y, lane_variance, lane_mask, lane_reference_beta,
                        lane_group_error, lane_group_count, lane_reference_sse, lane_group_xr,
                        lane_group_xx, duration=mu_duration[0], impact=bs[0],
                        ld_profile=ld_profile, num_nodes=transit_grid_nodes,
                        contact_fallback_margin=0.0, order=10,
                    )

                reduced_log_prob = jax.vmap(lane_log_likelihood)(
                    theta, u, active_y, active_variance, active_mask, reference_beta, group_error,
                    group_count, group_reference_sse, group_xr, group_xx,
                )
                numpyro.factor('obs_active', jnp.sum(reduced_log_prob[:, 0]))
                numpyro.factor(
                    'obs_out_of_window', jnp.sum(reduced_log_prob[:, 1])
                )
                return

            # The fused score takes one fixed orbit shared by every channel.
            use_quadratic_score = (
                free_t0 and resolved_kernel == 'quadratic_contact_safe'
                and ld_profile == 'quadratic' and n_planets == 1
                and not joint_geometry and surface_config.get('fit_geometry', True)
                and all(jnp.size(value) == 1 for value in (PERIOD, a_rs, bs, mu_ecc, mu_omega))
            )
            if use_quadratic_score:
                from .quadratic_cuda import log_likelihood

                beta = jnp.stack(
                    tuple(params[name] for name in linear_trend_names), axis=1
                )
                design_active = jnp.asarray(
                    resolved_trend_design, dtype=jnp.float64
                )[indices]
                active_y = jnp.atleast_2d(jnp.asarray(y, dtype=jnp.float64))[:, indices]
                # Square before gathering, as in the ordinary reduced model.
                active_variance = (yerr_matrix ** 2)[:, indices]
                active_mask = (jnp.ones_like(active_y, dtype=bool)
                    if likelihood_mask is None else jnp.broadcast_to(
                        jnp.asarray(likelihood_mask, dtype=bool), jnp.shape(jnp.atleast_2d(y))
                    )[:, indices])
                theta = jnp.concatenate(
                    (rors[:, :1], u, params['t0'][:, :1], beta, jitter[:, None]), axis=1
                )
                orbit = jnp.stack([
                    jnp.ravel(value)[0] for value in (PERIOD, a_rs, bs, mu_ecc, mu_omega)
                ])
                lane_log_prob = jax.vmap(
                    log_likelihood, in_axes=(0, None, 0, 0, 0, None, None)
                )(theta, t[indices], active_y, active_variance, active_mask, design_active, orbit)
                numpyro.factor('obs_active', jnp.sum(lane_log_prob))
            else:
                transit_active = jax.vmap(compute_transit_model_window, in_axes=(in_axes, None))(
                    params, t,
                )
                beta = jnp.stack(
                    tuple(params[name] for name in linear_trend_names), axis=1
                )
                design_active = jnp.asarray(
                    resolved_trend_design, dtype=jnp.float64
                )[indices]
                trend_active = beta @ design_active.T
                active_model = apply_systematics(transit_active, trend_active)
                active_error = error_broadcast[:, indices]
                active_y = jnp.atleast_2d(
                    jnp.asarray(y, dtype=jnp.float64)
                )[:, indices]
                active_log_prob = dist.Normal(
                    active_model, active_error
                ).log_prob(active_y)
                if likelihood_mask is not None:
                    active_mask = jnp.asarray(likelihood_mask, dtype=bool)
                    active_mask = jnp.broadcast_to(
                        active_mask, jnp.shape(jnp.atleast_2d(y))
                    )[:, indices]
                    active_log_prob = jnp.where(
                        active_mask, active_log_prob, 0.0
                    )
                numpyro.factor('obs_active', jnp.sum(active_log_prob))

            reference_beta = jnp.atleast_2d(jnp.asarray(
                oot_reference_beta, dtype=jnp.float64
            ))
            delta = beta - reference_beta
            group_xr = jnp.asarray(
                oot_group_x_reference_residual, dtype=jnp.float64
            )
            group_xx = jnp.asarray(oot_group_xx, dtype=jnp.float64)
            if group_xr.ndim == 2:
                group_xr = group_xr[None, ...]
                group_xx = group_xx[None, ...]
            group_sse = (
                jnp.atleast_2d(jnp.asarray(
                    oot_group_reference_sse, dtype=jnp.float64
                ))
                - 2.0 * jnp.einsum('lp,lgp->lg', delta, group_xr)
                + jnp.einsum('lp,lgpq,lq->lg', delta, group_xx, delta)
            )
            group_count = jnp.atleast_2d(jnp.asarray(
                oot_group_count, dtype=jnp.float64
            ))
            group_error = jnp.atleast_2d(jnp.asarray(
                oot_group_yerr, dtype=jnp.float64
            ))
            group_variance = group_error**2 + jitter[:, None]**2
            group_log_prob = -0.5 * (
                group_sse / group_variance
                + group_count * jnp.log(2.0 * jnp.pi * group_variance)
            )
            numpyro.factor('obs_out_of_window', jnp.sum(group_log_prob))
            return

        if 'gp_spectroscopic' in detrend_components:
            if gp_trend is None:
                raise ValueError("gp_spectroscopic requires the fixed white-light GP template.")
            params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(*(trend_bounds or {}).get('A_gp',(0.5,2))).expand([num_lcs]))
            in_axes['A_gp'] = 0
        if 'explinear_spectroscopic' in detrend_components:
            if 'gp_spectroscopic' in detrend_components:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes,None,None,None))(
                    params,t,gp_trend,exp_trend)
            elif '2spot_spectroscopic' in detrend_components:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes,None,None,None,None))(
                    params,t,spot_trend,spot_trend2,exp_trend)
            elif 'spot_spectroscopic' in detrend_components:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes,None,None,None))(
                    params,t,spot_trend,exp_trend)
            else:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes,None,None))(
                    params,t,exp_trend)
        elif 'gp_spectroscopic' in detrend_components and detrend_components & {
                'spot_spectroscopic', '2spot_spectroscopic', 'linear_discontinuity_spectroscopic'}:
            # spot and step templates under the GP template (models.gp.compute_lc_event_gp_spectroscopic)
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None, None, None))(
                params, t, gp_trend,
                spot_trend if detrend_components & {'spot_spectroscopic', '2spot_spectroscopic'} else None,
                spot_trend2 if '2spot_spectroscopic' in detrend_components else None,
                jump_trend if 'linear_discontinuity_spectroscopic' in detrend_components else None)
        elif 'gp_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes,None,None))(params,t,gp_trend)
        elif '2spot_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, spot_trend2)
        elif 'spot_spectroscopic' in detrend_components:
            if 'linear_discontinuity_spectroscopic' in detrend_components:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, jump_trend)
            else:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, spot_trend)
        elif 'linear_discontinuity_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, jump_trend)
        else:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)

        if likelihood_mask is None:
            numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)
        else:
            with numpyro.handlers.mask(mask=jnp.asarray(likelihood_mask, dtype=bool)):
                numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    _vectorized_model._koala_ld_mode = ld_mode
    _vectorized_model.surface_science_sites = surface_science_sites(surface_config)
    if free_t0:
        _vectorized_model.surface_science_sites = tuple(
            _vectorized_model.surface_science_sites or ()
        ) + ('dt0',)
    # These supported models have parameter-independent latent constraints.
    # Per-channel prior locations/scales and observations remain dynamic; only
    # fully supplied initial-value transforms may reuse their traced supports.
    _vectorized_model._koala_fixed_initialization_supports = bool(
        not joint_geometry and not ld_variant_as_data and ld_mode != 'sing'
        and surface_config.get('model') == 'transit' and not surface_config.get('spots')
        and detrend_type in {'none','constant','linear','quadratic','cubic','quartic'}
    )
    return _vectorized_model

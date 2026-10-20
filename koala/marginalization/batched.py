"""Exact augmented targets for batching polynomial/LD candidates together.

Unused coordinates have proper independent priors and integrate to one.
They allow one static sampler program per limb-darkening law. Every lane
keeps its own geometry, observations, MAP, metric, random key and adaptation.
This module is opt-in until full posterior and end-to-end gates establish a
benefit; the ordinary pipeline and grid runner remain the production route.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from models.jaxoplanet.core import compute_transit_model, build_transit_phase_offsets
from models.priors import sample_parameter


def create_batched_model(*, law="quadratic", stage="highres", parameter_priors=None,
                         ld_uniform_basis="uplus_uminus", ld_bounds=(0., 1.),
                         power2_grid_nodes=None, power2_parameterization='coefficients',
                         transit_window_indices=None, power2_table_shape=None,
                         power2_quadrature_order=128, step_trend=False,
                         step_parameterization='cadence',trend_basis='elapsed_days',trend_bounds=None):
    from models.spectro_trends import validate_spectral_trend_bounds
    trend_bounds=validate_spectral_trend_bounds(trend_bounds)
    if trend_basis not in {'elapsed_days','batman_normalized'}:
        raise ValueError('Batched trend_basis must be elapsed_days or batman_normalized.')
    if trend_bounds and (stage=='white' or set(trend_bounds)-{'c','v','v2','A_jump'}):
        raise ValueError('Batched spectral bounds support c/v/v2/A_jump; use ordinary grid fitting for other priors.')
    if law not in {"quadratic", "power2"} or stage not in {"white", "highres"}:
        raise ValueError("Batched targets require quadratic/power2 and white/highres.")
    if not isinstance(step_trend, bool):
        raise ValueError('step_trend must be Boolean.')
    if step_parameterization not in {'physical','cadence'}:
        raise ValueError('Step coordinates must be physical or cadence.')
    if ld_uniform_basis not in {"coefficients", "uplus_uminus"}:
        raise ValueError("Unsupported quadratic uniform basis.")
    low, high = map(float, ld_bounds)
    if not low < high:
        raise ValueError("LD bounds require low < high.")
    if stage == "white" and parameter_priors is None:
        raise ValueError("White-light targets need explicit planet parameter priors.")
    if law == "power2":
        low, high = 0., 1.
    if power2_grid_nodes is not None and int(power2_grid_nodes) < 4097:
        raise ValueError('Batched power-2 grids require at least 4097 nodes.')
    if int(power2_quadrature_order) != power2_quadrature_order or power2_quadrature_order < 64:
        raise ValueError('Power-2 quadrature requires an integer order of at least 64.')
    if power2_parameterization not in {'coefficients', 'maxted_smooth'}:
        raise ValueError('Batched power-2 coordinates must be coefficients or maxted_smooth.')
    smooth = law == 'power2' and power2_parameterization == 'maxted_smooth'
    if power2_table_shape is not None:
        if law != 'power2' or stage != 'highres' or power2_grid_nodes is None:
            raise ValueError('Power-2 tables require fixed high-resolution geometry and a grid.')
        power2_table_shape = tuple(map(int, power2_table_shape))
        if len(power2_table_shape) != 3 or min(power2_table_shape) < 1 or power2_table_shape[-1] != power2_grid_nodes:
            raise ValueError('Invalid power-2 table shape.')
    if transit_window_indices is not None:
        window = np.asarray(transit_window_indices, dtype=int)
        if window.ndim != 1 or len(window) == 0 or np.any(window < 0) or np.any(np.diff(window) <= 0):
            raise ValueError('Transit indices must be nonempty, nonnegative, sorted and unique.')
        window = jnp.asarray(window)
    if law == "power2":
        from models.ld_parameterization import Power2MaxtedUniform, Power2MaxtedUniformTransform
        maxted_prior = Power2MaxtedUniform()
    if smooth:
        transform = Power2MaxtedUniformTransform()
    if law == "quadratic" and ld_uniform_basis == "uplus_uminus":
        ld_low, ld_high = jnp.array([-1., -2.]), jnp.array([2., 2.])
    else:
        ld_low, ld_high = jnp.full(2, low), jnp.full(2, high)

    def model(t, yerr, y=None, *, quadratic_trend, free_ld, ld_fixed,
              PERIOD=None, mu_duration=None, mu_t0=None, mu_b=None, mu_a_rs=None,
              mu_ecc=0., mu_omega=0., rors_reference=None,
              likelihood_mask=None, power2_table_bank=None, table_index=None,
              t_jump_guess=None, jump_guess=None, jump_template_bank=None, step_index=None):
        errors = jnp.atleast_2d(jnp.asarray(yerr, dtype=jnp.float64))
        lanes = errors.shape[0]
        shape = (lanes,)
        quadratic = jnp.broadcast_to(jnp.asarray(quadratic_trend), shape)
        free = jnp.broadcast_to(jnp.asarray(free_ld), shape)
        if smooth:
            base = dist.Normal(0., 1.).expand((lanes, 2)).to_event(1)
            latent = numpyro.sample('ld_coordinates', base)
            coords = transform(latent)
            # Uniform in the Maxted coordinates (h1, h2) (koala's power-2 uniform prior).
            transformed_logp = (maxted_prior.log_prob(coords)
                               +transform.log_abs_det_jacobian(latent, coords))
            numpyro.factor('ld_variant_prior', jnp.sum(jnp.where(
                free, transformed_logp-base.log_prob(latent), 0.)))
        elif law == "power2":
            # Direct (c, alpha) coordinates with hard walls and the same Maxted-uniform density.
            raw = numpyro.sample("ld_coordinates", dist.Uniform(jnp.array([0., 0.]), jnp.array([1.5, 40.]))
                                 .expand((lanes, 2)).to_event(1))
            inside = jnp.isfinite(maxted_prior.log_prob(raw))
            coords = jnp.where(inside[:, None], raw, .5)
            numpyro.factor('ld_variant_prior', jnp.sum(jnp.where(
                free, jnp.where(inside, maxted_prior.log_prob(coords)+jnp.log(60.), -1e100), 0.)))
        else:
            coords = numpyro.sample("ld_coordinates", dist.Uniform(ld_low, ld_high)
                                    .expand((lanes, 2)).to_event(1))
        if law == "quadratic" and ld_uniform_basis == "uplus_uminus":
            coefficients = jnp.stack(((coords[:, 0]+coords[:, 1])/2,
                                      (coords[:, 0]-coords[:, 1])/2), axis=-1)
        else:
            coefficients = coords
        coefficients = jnp.where(free[:, None], coefficients, jnp.asarray(ld_fixed))
        if law == "quadratic":
            numpyro.deterministic("u", coefficients)
        else:
            numpyro.deterministic("c1", coefficients[:, 0])
            numpyro.deterministic("c2", coefficients[:, 1])
        defaults=({'c':(.5,1.5),'v':(-1.,1.),'v2':(-1.,1.)} if trend_basis=='batman_normalized'
                  else {'c':(.9,1.1),'v':(-.1,.1),'v2':(-.1,.1)})
        bounds=[(trend_bounds or {}).get(name,defaults[name]) for name in ('c','v','v2')]
        trend = numpyro.sample("trend_coordinates", dist.Uniform(
            jnp.asarray([b[0] for b in bounds]),jnp.asarray([b[1] for b in bounds]))
            .expand((lanes, 3)).to_event(1))
        # The unused quadratic coordinate has a normalized independent prior
        # and integrates to one in linear/step variants, in either basis.
        active=trend.at[:,2].set(jnp.where(quadratic & (not step_trend),trend[:,2],0.))
        if trend_basis=='batman_normalized':
            from models.trend_marginal import normalized_polynomial_transform
            trend_transform=normalized_polynomial_transform(t,lanes,2,likelihood_mask)
            physical=jnp.einsum('...ij,...j->...i',trend_transform,active)
        else:physical=active
        c = numpyro.deterministic("c", physical[:, 0])
        v = numpyro.deterministic("v", physical[:, 1])
        v2 = numpyro.deterministic("v2", physical[:, 2])
        if step_trend:
            if stage == 'white':
                from models.trends import _MAX_JUMP_WIDTH_DAYS
                guess = .5*(jnp.min(t)+jnp.max(t)) if t_jump_guess is None else jnp.asarray(t_jump_guess)
                amplitude = 0. if jump_guess is None else jnp.asarray(jump_guess)
                jump = numpyro.sample('jump',dist.Normal(amplitude,.01).expand(shape))
                cadence = jnp.median(jnp.diff(jnp.sort(t)))
                if step_parameterization == 'cadence':
                    offset = numpyro.sample('t_jump_offset_cadences',dist.Normal(0.,.01/cadence).expand(shape))
                    epoch = numpyro.deterministic('t_jump',guess+cadence*offset)
                    width_unit = numpyro.sample('log_width_unit_logit',dist.Logistic(0.,1.).expand(shape))
                    lower,upper = jnp.log(.5),jnp.log(_MAX_JUMP_WIDTH_DAYS/cadence)
                    log_width_cadences = numpyro.deterministic('log_width_cadences',
                        lower+(upper-lower)*jax.nn.sigmoid(width_unit))
                    log_width = numpyro.deterministic('log_width',log_width_cadences+jnp.log(cadence))
                else:
                    epoch = numpyro.sample('t_jump',dist.Normal(guess,.01).expand(shape))
                    log_width = numpyro.sample('log_width',dist.Uniform(jnp.log(.5*cadence),
                        jnp.log(_MAX_JUMP_WIDTH_DAYS)).expand(shape))
                width = numpyro.deterministic('width',jnp.exp(log_width))
                numpyro.deterministic('width_minutes',width*24*60)
                step = jump[:,None]*jax.nn.sigmoid((t[None,:]-epoch[:,None])/width[:,None])
            else:
                if jump_template_bank is None or step_index is None:
                    raise ValueError('A spectroscopic step requires its white-light template bank and lane indices.')
                amplitude = numpyro.sample('A_jump',dist.Uniform(*(trend_bounds or {}).get('A_jump',(.5,2.))).expand(shape))
                step = amplitude[:,None]*jnp.asarray(jump_template_bank)[jnp.asarray(step_index,dtype=int)]
        if stage == "white":
            geometry = {}
            for name in ("period", "t0", "b", "duration", "rprs"):
                site = "rors_0" if name == "rprs" else f"{name}_0"
                value = sample_parameter(parameter_priors[name][0], site, shape)
                geometry[name] = value[:, None]
            radius = geometry.pop("rprs")
            numpyro.deterministic("rors", radius)
            log_jitter = numpyro.sample("log_jitter", dist.Uniform(
                jnp.log(1e-5), jnp.log(1e-2)).expand(shape))
        else:
            radius = numpyro.sample("rors", dist.Uniform(jnp.sqrt(1e-5),
                jnp.sqrt(.5)).expand((lanes, 1)))
            geometry = {name: jnp.broadcast_to(jnp.asarray(value), (lanes, 1))
                for name, value in (("period", PERIOD), ("duration", mu_duration),
                                    ("t0", mu_t0), ("b", mu_b))}
            median_error = jnp.nanmedian(errors, axis=1)
            log_jitter = numpyro.sample("log_jitter", dist.Normal(
                jnp.log(.5*median_error), jnp.full(shape, 2.)))
            numpyro.deterministic("total_error", jnp.sqrt(
                jnp.exp(2*log_jitter)+median_error**2))
        numpyro.deterministic("depths", radius**2)

        transit_time = t if transit_window_indices is None else t[window]
        indices = jnp.zeros(lanes, dtype=int) if table_index is None else jnp.asarray(table_index, dtype=int)
        if stage == 'white':
            from models.harmonica.core import harmonica_a_rs_from_duration
            e = parameter_priors['ecc'][0].value
            w = parameter_priors['omega'][0].value
            orbit_a = harmonica_a_rs_from_duration(geometry['period'], geometry['duration'],
                        geometry['b'], radius, ecc=e, omega=w)
        else:
            if mu_a_rs is None:
                if rors_reference is None:
                    raise ValueError('Batched spectroscopy requires white-light mu_a_rs or rors_reference.')
                from models.harmonica.core import harmonica_a_rs_from_duration
                mu_a_rs = harmonica_a_rs_from_duration(
                    geometry['period'],geometry['duration'],geometry['b'],rors_reference,
                    ecc=mu_ecc,omega=mu_omega)
            orbit_a = jnp.broadcast_to(jnp.asarray(mu_a_rs), (lanes, 1))
            e, w = mu_ecc, mu_omega
        def transit(period, t0, b, duration, r, ld, index, a_rs):
            offsets, mask = build_transit_phase_offsets(transit_time, period, t0, duration)
            params = dict(period=period, t0=t0, b=b, duration=duration, rors=r,
                a_rs=a_rs, ecc=e, omega=w,
                _ld_profile=law, _transit_phase_offsets=offsets,
                _transit_phase_mask=mask)
            if law == "quadratic":
                params.update(u=ld, _jaxoplanet_kernel="quadratic_contact_safe")
                if stage == "highres":
                    params['_quadratic_radius_bound'] = .5**.5
                else:
                    spec = parameter_priors['rprs'][0]
                    bound = (abs(spec.value) if spec.fixed else spec.high
                             if spec.low is not None and spec.low >= 0. else None)
                    if bound is not None:
                        params['_quadratic_radius_bound'] = float(bound)
            else:
                params.update(u=jnp.zeros(12), c1=ld[0], c2=ld[1],
                              _power2_exact=True, _power2_quadrature_order=int(power2_quadrature_order))
                nodes = power2_grid_nodes
                if stage == 'white' and len(t) > 5000:
                    nodes = 4097 if nodes is None else nodes
                if nodes is not None:
                    params['_transit_grid_nodes'] = int(nodes)
                if power2_table_shape is not None:
                    if power2_table_bank is None:
                        raise ValueError('A table-enabled model requires its shared table bank.')
                    table = {name: jnp.asarray(value)[index] for name, value in power2_table_bank.items()}
                    # The compressed evaluator only needs this array's shape.
                    # Its values vanish during compilation; banks store the
                    # small factors once, without copying them into each lane.
                    table['coefficients'] = jnp.zeros(power2_table_shape, dtype=jnp.uint8)
                    params.update(_power2_table=table, _power2_table_strict_fallback=True)
            return compute_transit_model(params, transit_time)

        signal = jax.vmap(transit)(geometry["period"], geometry["t0"],
            geometry["b"], geometry["duration"], radius, coefficients, indices, orbit_a)
        if transit_window_indices is not None:
            signal = jnp.zeros_like(errors).at[:, window].set(signal)
        time = t-jnp.min(t)
        # A fixed step template retains the linear background in both stages.
        # Its auxiliary quadratic coordinate was removed before conversion so
        # normalized coordinates cannot leak that unused draw into c or v.
        baseline = c[:, None]+v[:, None]*time+v2[:, None]*time**2
        if step_trend:
            baseline = baseline+step
        mean = (1+signal)*baseline
        error = jnp.sqrt(errors**2+jnp.exp(2*log_jitter[:, None]))
        if likelihood_mask is None:
            numpyro.sample("obs", dist.Normal(mean, error), obs=y)
        else:
            with numpyro.handlers.mask(mask=jnp.asarray(likelihood_mask)):
                numpyro.sample("obs", dist.Normal(mean, error), obs=y)

    model._koala_channel_varying_kwargs = BATCHED_CHANNEL_KWARGS
    model._koala_fixed_initialization_supports = True
    return model


BATCHED_CHANNEL_KWARGS = ("quadratic_trend", "free_ld", "ld_fixed", "PERIOD",
                          "mu_duration", "mu_t0", "mu_b", "mu_a_rs", "rors_reference", "table_index",
                          "t_jump_guess", "jump_guess", "step_index")

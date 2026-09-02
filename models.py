import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
import tinygp
from functools import partial

def spot_crossing(t, amp, mu, sigma):
    return amp * jnp.exp(-0.5 * (t - mu) **2 / sigma **2)

def _compute_transit_model(params, t):
    """Transit Model for one or more planets, using vmap for performance."""
    periods = jnp.atleast_1d(params["period"])
    durations = jnp.atleast_1d(params["duration"])
    t0s = jnp.atleast_1d(params["t0"])
    bs = jnp.atleast_1d(params["b"])
    rorss = jnp.atleast_1d(params["rors"])

    def get_lc(period, duration, t0, b, rors):
        orbit = TransitOrbit(
            period=period,
            duration=duration,
            time_transit=t0,
            impact_param=b,
            radius_ratio=rors
        )
        return limb_dark_light_curve(orbit, params["u"])(t)

    batched_lcs = jax.vmap(get_lc)(periods, durations, t0s, bs, rorss)
    return jnp.sum(batched_lcs, axis=0)

def compute_mean_model(params, t, detrend_type, gp_trend=None, spot_trend=None, jump_trend=None):
    """Computes the mean model."""
    model = _compute_transit_model(params, t)
    components = detrend_type.split('+')
    parametric_models = [c for c in components if c not in ['gp', 'gp_spectroscopic', 'spot_spectroscopic', 'linear_discontinuity_spectroscopic']]
    if len(parametric_models) > 1:
        raise ValueError("Cannot combine multiple parametric models.")
    parametric_model = parametric_models[0] if parametric_models else 'none'
    
    trend = 0.0
    t_norm = t - jnp.min(t)

    if parametric_model != 'none' or 'gp' in components:
        trend += params.get('c', 1.0)
    else:
        model += 1.0

    if parametric_model == 'linear':
        trend += params['v'] * t_norm
    elif parametric_model == 'quadratic':
        trend += params['v'] * t_norm + params['v2'] * t_norm**2
    elif parametric_model == 'cubic':
        trend += params['v'] * t_norm + params['v2'] * t_norm**2 + params['v3'] * t_norm**3
    elif parametric_model == 'quartic':
        trend += params['v'] * t_norm + params['v2'] * t_norm**2 + params['v3'] * t_norm**3 + params['v4'] * t_norm**4
    elif parametric_model == 'linear_discontinuity':
        trend += params['v'] * t_norm + jnp.where(t > params["t_jump"], params["jump"], 0.0)
    elif parametric_model == 'explinear':
        trend += params['v'] * t_norm + params['A'] * jnp.exp(-t_norm / params['tau'])
    elif parametric_model == 'spot':
        trend += params['v'] * t_norm + spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    
    if 'gp_spectroscopic' in components:
        trend += params['A_gp'] * gp_trend
    if 'spot_spectroscopic' in components:
        trend += params['A_spot'] * spot_trend
    if 'linear_discontinuity_spectroscopic' in components:
        trend += params['A_jump'] * jump_trend
        
    return model + trend

def build_gp(params, t, error, detrend_type):
    kernel = tinygp.kernels.quasisep.Matern32(
        scale=jnp.exp(params['GP_log_rho']),
        sigma=jnp.exp(params['GP_log_sigma']),
    )
    def mean_function(t_gp):
        return compute_mean_model(params, t_gp, detrend_type)
    return tinygp.GaussianProcess(kernel, t, diag=error**2, mean=mean_function)

def create_whitelight_model(detrend_type='linear', n_planets=1):
    """Builds a static whitelight model."""
    print(f"Building whitelight model with: detrend_type='{detrend_type}' for {n_planets} planets")
    components = detrend_type.split('+')
    
    def _whitelight_model_static(t, yerr, y=None, prior_params=None):
        params = {}
        durations, t0s, bs, rorss = [], [], [], []
        for i in range(n_planets):
            logD = numpyro.sample(f"logD_{i}", dist.Normal(jnp.log(prior_params['duration'][i]), 3e-2))
            durations.append(numpyro.deterministic(f"duration_{i}", jnp.exp(logD)))
            t0s.append(numpyro.sample(f"t0_{i}", dist.Normal(prior_params['t0'][i], 3e-2)))
            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            bs.append(numpyro.deterministic(f'b_{i}', jnp.abs(_b)))
            depths = numpyro.sample(f'depths_{i}', dist.Uniform(1e-5, 0.5))
            rorss.append(numpyro.deterministic(f"rors_{i}", jnp.sqrt(depths)))

        u = numpyro.sample("u", distx.QuadLDParams())
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter)**2 + yerr**2))
        params.update({"period": prior_params['period'], "duration": jnp.array(durations), "t0": jnp.array(t0s), "b": jnp.array(bs), "rors": jnp.array(rorss), "u": u})

        if any(c for c in components if c not in ['gp', 'none']) or 'gp' in components:
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))
        if any(c in components for c in ['linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot']):
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1))
        if 'quadratic' in components:
            params['v2'] = numpyro.sample('v2', dist.Normal(0.0, 0.1))
        if 'cubic' in components:
            params.update({'v2': numpyro.sample('v2', dist.Normal(0.0, 0.1)), 'v3': numpyro.sample('v3', dist.Normal(0.0, 0.1))})
        if 'quartic' in components:
            params.update({'v2': numpyro.sample('v2', dist.Normal(0.0, 0.1)), 'v3': numpyro.sample('v3', dist.Normal(0.0, 0.1)), 'v4': numpyro.sample('v4', dist.Normal(0.0, 0.1))})
        if 'linear_discontinuity' in components:
            params.update({'t_jump': numpyro.sample('t_jump', dist.Normal(59791.12, 0.1)), 'jump': numpyro.sample('jump', dist.Normal(0.0, 0.1))})
        if 'explinear' in components:
            params.update({'A': numpyro.sample('A', dist.Normal(0.0, 0.1)), 'tau': numpyro.sample('tau', dist.Normal(0.0, 0.1))})
        if 'spot' in components:
            params.update({'spot_amp': numpyro.sample('spot_amp', dist.Normal(0.0, 0.01)), 'spot_mu': numpyro.sample('spot_mu', dist.Normal(prior_params['spot_guess'], 0.01)), 'spot_sigma': numpyro.sample('spot_sigma', dist.Normal(0.0, 0.01))})

        if 'gp' in components:
            params.update({'GP_log_sigma': numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-5), jnp.log(1e3))), 'GP_log_rho': numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(1e-3), jnp.log(1e2)))})
            gp = build_gp(params, t, error, detrend_type)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            lc_model = compute_mean_model(params, t, detrend_type)
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
    return _whitelight_model_static

def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free', n_planets=1):
    """Builds a static vectorized model."""
    print(f"Building vectorized model with: detrend='{detrend_type}', ld='{ld_mode}', trend='{trend_mode}' for {n_planets} planets")
    components = detrend_type.split('+')

    def _vectorized_model_static(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None, mu_depths=None, PERIOD=None, trend_fixed=None, ld_interpolated=None, ld_fixed=None, mu_spot_amp=None, mu_spot_mu=None, mu_spot_sigma=None, mu_u_ld=None, gp_trend=None, spot_trend=None, jump_trend=None):
        num_lcs = jnp.atleast_2d(yerr).shape[0]
        depths = numpyro.sample('depths', dist.Uniform(1e-5, 0.5).expand([num_lcs, n_planets]))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))
        yerr_per_lc = jnp.nanmedian(yerr, axis=1)
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]))
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter**2 + yerr_per_lc**2))
        error_broadcast = total_error[:, None] * jnp.ones_like(t)

        if ld_mode == 'free':
            u = numpyro.sample('u', dist.TruncatedNormal(loc=mu_u_ld, scale=0.2, low=-1.0, high=1.0).to_event(1))
        elif ld_mode == 'fixed':
            u = numpyro.deterministic("u", ld_fixed)
        elif ld_mode == 'interpolated':
            u = numpyro.deterministic("u", ld_interpolated)
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        params = {"period": PERIOD, "duration": mu_duration, "t0": mu_t0, "b": mu_b, "rors": rors, "u": u}
        in_axes = {"period": None, "duration": None, "t0": None, "b": None, "rors": 0, "u": 0}

        if any(c for c in components if c not in ['gp', 'gp_spectroscopic', 'none']) or 'gp' in components or 'gp_spectroscopic' in components:
            param_defs = { 'linear': [('v', dist.Normal(0.0, 0.1))], 'quadratic': [('v', dist.Normal(0.0, 0.1)), ('v2', dist.Normal(0.0, 0.1))], 'cubic': [('v', dist.Normal(0.0, 0.1)), ('v2', dist.Normal(0.0, 0.1)), ('v3', dist.Normal(0.0, 0.1))], 'quartic': [('v', dist.Normal(0.0, 0.1)), ('v2', dist.Normal(0.0, 0.1)), ('v3', dist.Normal(0.0, 0.1)), ('v4', dist.Normal(0.0, 0.1))], 'linear_discontinuity': [('v', dist.Normal(0.0, 0.1)), ('t_jump', dist.Normal(jnp.median(t), 0.1)), ('jump', dist.Normal(0.0, 0.1))], 'explinear': [('v', dist.Normal(0.0, 0.1)), ('A', dist.Normal(0.0, 0.1)), ('tau', dist.Normal(0.0, 0.1))], 'spot': [('v', dist.Normal(0.0, 0.1)), ('spot_amp', dist.Normal(mu_spot_amp, 0.01)), ('spot_mu', dist.Normal(mu_spot_mu, 0.01)), ('spot_sigma', dist.Normal(mu_spot_sigma, 0.01))] }
            if trend_mode == 'free':
                params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1).expand([num_lcs]))
                in_axes['c'] = 0
                for comp in components:
                    if comp in param_defs:
                        for p_name, p_dist in param_defs[comp]:
                            params[p_name] = numpyro.sample(p_name, p_dist.expand([num_lcs]))
                            in_axes[p_name] = 0
            elif trend_mode == 'fixed':
                trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                param_names_ordered = []
                for comp in ['linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot']:
                    if comp in components:
                        if comp == 'linear': param_names_ordered.extend(['v'])
                        if comp == 'quadratic': param_names_ordered.extend(['v', 'v2'])
                        if comp == 'cubic': param_names_ordered.extend(['v', 'v2', 'v3'])
                        if comp == 'quartic': param_names_ordered.extend(['v', 'v2', 'v3', 'v4'])
                        if comp == 'linear_discontinuity': param_names_ordered.extend(['v', 't_jump', 'jump'])
                        if comp == 'explinear': param_names_ordered.extend(['v', 'A', 'tau'])
                        if comp == 'spot': param_names_ordered.extend(['v', 'spot_amp', 'spot_mu', 'spot_sigma'])
                
                params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
                in_axes['c'] = 0
                unique_params = []
                [unique_params.append(p) for p in param_names_ordered if p not in unique_params]
                for i, p_name in enumerate(unique_params):
                    params[p_name] = numpyro.deterministic(p_name, trend_temp[:, i + 1])
                    in_axes[p_name] = 0
            else:
                raise ValueError(f"Unknown trend_mode: {trend_mode}")

        if 'gp_spectroscopic' in components:
            params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(1e-1, 1e1).expand([num_lcs]))
            in_axes['A_gp'] = 0
        if 'spot_spectroscopic' in components:
            params['A_spot'] = numpyro.sample('A_spot', dist.Uniform(1e-1, 1e1).expand([num_lcs]))
            in_axes['A_spot'] = 0
        if 'linear_discontinuity_spectroscopic' in components:
            params['A_jump'] = numpyro.sample('A_jump', dist.Uniform(1e-1, 1e1).expand([num_lcs]))
            in_axes['A_jump'] = 0
            
        vmap_kernel = partial(compute_mean_model, detrend_type=detrend_type, gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend)
        y_model = jax.vmap(vmap_kernel, in_axes=(in_axes, None))(params, t)
        numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model_static


"""Physical power-2 likelihoods and gradients across sampled trend families."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from koala.config import parse_planet_parameter_specs
from models.jaxoplanet.builder import create_whitelight_model
from models.jaxoplanet.power2_exact import light_curve

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('trend', ['constant', 'linear', 'quadratic', 'explinear',
                                  'spot', 'linear_discontinuity'])
@pytest.mark.parametrize('prior', ['fixed', 'gaussian', 'uniform'])
def test_default_physical_law_with_sampled_trends(trend, prior):
    times = jnp.linspace(-.09, .09, 43) + 2e-5*jnp.sin(jnp.arange(43))
    errors = jnp.linspace(.001, .0013, times.size)
    observed = jnp.ones_like(times)
    specs = parse_planet_parameter_specs({key:dict(value=value, prior='fixed')
        for key,value in dict(period=3.,duration=.12,t0=0.,b=.45,rprs=.146).items()})
    model = create_whitelight_model(detrend_type=trend, ld_profile='power2',
                                   ld_mode=prior, parameter_priors=specs)
    prior_params = dict(u=jnp.array([.8,.1]), u_sigma=jnp.array([.1,.1]),
                        c=1., v=0., spot_guess=0.)
    first = handlers.trace(handlers.seed(model, 71)).get_trace(
        times, errors, y=observed, prior_params=prior_params)
    free = {name:site['value'] for name,site in first.items()
            if site['type']=='sample' and not site['is_observed']}
    if prior == 'uniform':
        # Direct power-2 coordinates are hard-walled to the Maxted-uniform support; the independent box draws of
        # c1 and c2 can fall outside it, so evaluate the parity at an interior point.
        free.update(c1=jnp.asarray(.6), c2=jnp.asarray(.7))

    def candidate(parameters):
        trace = handlers.trace(handlers.condition(model, data=parameters)).get_trace(
            times, errors, y=observed, prior_params=prior_params)
        return trace['obs']['fn'].log_prob(observed).sum()

    def oracle(parameters):
        c = parameters.get('c1', prior_params['u'][0])
        alpha = parameters.get('c2', prior_params['u'][1])
        radius, impact, duration = .146, .45, .12
        axis_squared = impact**2+((1+radius)**2-impact**2)/jnp.sin(jnp.pi*duration/3.)**2
        separation = jnp.sqrt(impact**2+(axis_squared-impact**2)*jnp.sin(2*jnp.pi*times/3.)**2)
        transit = light_curve(c,alpha,separation,radius)
        elapsed = times-times.min()
        baseline = parameters['c']
        if trend != 'constant':
            baseline += parameters['v']*elapsed
        if trend == 'quadratic':
            baseline += parameters['v2']*elapsed**2
        if trend == 'explinear':
            baseline += parameters['A']*jnp.exp(-elapsed/jnp.exp(parameters['log_tau']))
        if trend == 'spot':
            baseline += parameters['spot_amp']*jnp.exp(-.5*(times-parameters['spot_mu'])**2
                /jnp.maximum(jnp.abs(parameters['spot_sigma']),1e-6)**2)
        if trend == 'linear_discontinuity':
            # The production physical width prior stores log width.
            width = jnp.exp(parameters['log_width'])
            baseline += parameters['jump']*jax.nn.sigmoid((times-parameters['t_jump'])/width)
        variance = errors**2+jnp.exp(2*parameters['log_jitter'])
        residual = observed-(1+transit)*baseline
        return (-.5*(residual**2/variance+jnp.log(2*jnp.pi*variance))).sum()

    actual, actual_gradient = jax.jit(jax.value_and_grad(candidate))(free)
    expected, expected_gradient = jax.jit(jax.value_and_grad(oracle))(free)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=1e-12)
    assert actual_gradient.keys() == expected_gradient.keys()
    for name in free:
        np.testing.assert_allclose(actual_gradient[name], expected_gradient[name],
                                   atol=2e-7, rtol=2e-6, err_msg=name)

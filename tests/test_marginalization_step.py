"""Native-target density, flux and gradient parity for the sigmoid variant."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers
import pytest

from koala.config import parse_planet_parameter_specs
from koala.marginalization.batched import create_batched_model
from models.jaxoplanet.builder import create_whitelight_model,create_vectorized_model

jax.config.update('jax_enable_x64',True)


def trace(model,values,t,errors,y,kwargs):
    return handlers.trace(handlers.seed(handlers.substitute(model,data=values),0)).get_trace(
        t,errors,y=y,**kwargs)

def logp(tr):
    return sum(jnp.sum(site['fn'].log_prob(site['value'])) for site in tr.values()
               if site.get('type') == 'sample')


@pytest.mark.parametrize('stage',['white','highres'])
@pytest.mark.parametrize('law',['quadratic','power2'])
@pytest.mark.parametrize('free',[False,True])
@pytest.mark.parametrize('parameterization',['physical','cadence'])
def test_sigmoid_target_matches_native_flux_density_and_all_parameter_gradients(stage,law,free,parameterization):
    t = jnp.asarray([-.08,-.0500000001,-.05,-.0499999999,-.025,0.,.025,
                     .0499999999,.05,.0500000001,.08])
    errors = jnp.full((1,len(t)),.001)
    y = jnp.ones_like(errors)
    fixed = jnp.array([[.4,.2]])
    basis = 'uplus_uminus' if law=='quadratic' else 'coefficients'
    smooth = law=='power2'
    if smooth:
        from models.ld_parameterization import Power2MaxtedUniformTransform
        transform = Power2MaxtedUniformTransform()
    specs = parse_planet_parameter_specs({name:dict(value=value,prior='fixed') for name,value in
        dict(period=3.,t0=0.,duration=.1,b=.71,rprs=.1,ecc=0.,omega=0.).items()})
    options = dict(ld_profile=law,ld_mode='uniform' if free else 'fixed',
                   ld_uniform_basis=basis,jaxoplanet_kernel='quadratic_contact_safe' if law=='quadratic' else 'auto')
    native = create_whitelight_model(detrend_type='linear_discontinuity',parameter_priors=specs,trend_parameterization=parameterization,**options) if stage=='white' else create_vectorized_model(detrend_type='linear_discontinuity_spectroscopic',**options)
    augmented = create_batched_model(law=law,stage=stage,step_trend=True,
        ld_uniform_basis=basis,parameter_priors=specs if stage=='white' else None,
        step_parameterization=parameterization,power2_parameterization='maxted_smooth' if smooth else 'coefficients',
        power2_quadrature_order=64)
    bank = (-.0005*jax.nn.sigmoid((t-.018)/.004))[None,:]
    kwargs = dict(quadratic_trend=jnp.array([False]),free_ld=jnp.array([free]),ld_fixed=fixed)
    if stage=='white':
        kwargs.update(t_jump_guess=jnp.array([.017]),jump_guess=jnp.array([-.0005]))
    else:
        kwargs.update(PERIOD=jnp.array([[3.]]),rors_reference=jnp.array([.1]), mu_duration=jnp.array([[.1]]),
            mu_t0=jnp.array([[0.]]),mu_b=jnp.array([[.71]]),jump_template_bank=bank,step_index=jnp.array([0]))
    def compare(p):
        jitter,c,v,v2,first,second,epoch,jump,log_width,amplitude = p
        coordinates = jnp.array([first,second])
        latent = transform.inv(coordinates) if smooth else coordinates
        values = dict(log_jitter=jitter[None],trend_coordinates=jnp.array([[c,v,v2]]),
                      ld_coordinates=latent[None,:])
        nvalues = dict(c=c,log_jitter=jitter)
        if stage=='white':
            values['jump'] = jump[None]
            nvalues.update(v=v,jump=jump)
            if parameterization=='cadence':
                values.update(t_jump_offset_cadences=epoch[None],log_width_unit_logit=log_width[None])
                nvalues.update(t_jump_offset_cadences=epoch,log_width_unit_logit=log_width)
            else:
                values.update(t_jump=epoch[None],log_width=log_width[None])
                nvalues.update(t_jump=epoch,log_width=log_width)
            if free:
                nvalues.update(dict(ld_uplus_uminus=coordinates) if law=='quadratic' else dict(c1=first,c2=second))
            nkwargs = dict(prior_params=dict(u=fixed[0],t_jump_guess=.017,jump_guess=-.0005))
            nerror,ny = errors[0],y[0]
        else:
            values.update(rors=jnp.array([[.1]]),A_jump=amplitude[None])
            nvalues['v'] = v
            nvalues = {name:value[None] for name,value in nvalues.items()}
            nvalues.update(rors=jnp.array([[.1]]),A_jump=amplitude[None])
            if free:
                nvalues['ld_uplus_uminus' if law=='quadratic' else 'ld_coefficients'] = coordinates[None,:]
            nkwargs = dict(ld_fixed=fixed,PERIOD=jnp.array([3.]),rors_reference=jnp.array([.1]), mu_duration=jnp.array([.1]),
                mu_t0=jnp.array([0.]),mu_b=jnp.array([.71]),jump_trend=bank[0])
            nerror,ny = errors,y
        actual = trace(augmented,values,t,errors,y,kwargs)
        expected = trace(native,nvalues,t,nerror,ny,nkwargs)
        ghost = dist.Uniform(-.1,.1).log_prob(v2)
        if free and smooth:
            ghost += transform.log_abs_det_jacobian(latent,coordinates)
        elif not free:
            ghost += (dist.Normal(0.,1.).log_prob(latent).sum() if smooth else
                      dist.Uniform(jnp.array([-1.,-2.]),jnp.array([2.,2.])).log_prob(coordinates).sum())
        return jnp.array([logp(actual)-ghost,logp(expected)]),jnp.stack([
            actual['obs']['fn'].loc.reshape(-1),expected['obs']['fn'].loc.reshape(-1)])
    point = jnp.array([-9.,1.,.003,.002,.3,.2,.019 if parameterization=='physical' else .2,
                      -.0006,np.log(.016) if parameterization=='physical' else .1,1.1])
    density,flux = jax.jit(compare)(point)
    np.testing.assert_allclose(flux[0],flux[1],atol=2e-12,rtol=0)
    np.testing.assert_allclose(density[0],density[1],atol=2e-8,rtol=0)
    gradient = jax.jit(jax.jacrev(lambda p:compare(p)[0]))(point)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[0],gradient[1],atol=2e-7,rtol=2e-6)

def test_sigmoid_bank_and_lane_kwargs_preserve_template_ownership():
    from koala.marginalization.batched_fitting import _lane_kwargs
    bank = np.array([[0.,.001],[0.,-.003]])
    kwargs = dict(jump_template_bank=bank,step_index=np.array([1,0]),free_ld=np.array([False,True]))
    one = _lane_kwargs(kwargs,slice(1,2))
    assert one['jump_template_bank'] is bank
    np.testing.assert_array_equal(one['step_index'],[0])


@pytest.mark.parametrize('law',['quadratic','power2'])
def test_sigmoid_full_geometry_gradients_at_central_grazing_and_exact_contacts(law):
    t = jnp.array([-.08,-.0500000001,-.05,-.0499999999,-.025,0.,.025,
                   .0499999999,.05,.0500000001,.08])
    errors,y = jnp.full((1,len(t)),.001),jnp.ones((1,len(t)))
    bank = (-.0005*jax.nn.sigmoid((t-.018)/.004))[None,:]
    augmented = create_batched_model(law=law,step_trend=True,ld_uniform_basis='coefficients')
    native = create_vectorized_model(ld_profile=law,ld_mode='fixed',
        detrend_type='linear_discontinuity_spectroscopic',jaxoplanet_kernel='quadratic_contact_safe' if law=='quadratic' else 'auto')
    def targets(p):
        r,b,duration,epoch = p
        geometry = dict(PERIOD=jnp.array([[3.]]),rors_reference=jnp.array([.1]), mu_duration=duration.reshape(1,1),
                        mu_t0=epoch.reshape(1,1),mu_b=b.reshape(1,1))
        values = dict(rors=r.reshape(1,1),log_jitter=jnp.array([-9.]),
            trend_coordinates=jnp.array([[1.,.003,.002]]),ld_coordinates=jnp.array([[.3,.2]]),
            A_jump=jnp.array([1.1]))
        original = dict(rors=r.reshape(1,1),log_jitter=jnp.array([-9.]),c=jnp.array([1.]),v=jnp.array([.003]),A_jump=jnp.array([1.1]))
        actual = trace(augmented,values,t,errors,y,dict(geometry,ld_fixed=jnp.array([[.4,.2]]),
            quadratic_trend=jnp.array([False]),free_ld=jnp.array([False]),
            jump_template_bank=bank,step_index=jnp.array([0])))
        expected = trace(native,original,t,errors,y,dict(ld_fixed=jnp.array([[.4,.2]]),
            jump_trend=bank[0],**{name:value[0] for name,value in geometry.items()}))
        ghost = dist.Uniform(-.1,.1).log_prob(.003)+(2*dist.Uniform(0.,1.).log_prob(.3) if law=='quadratic' else
                 dist.Uniform(0.,1.5).log_prob(.3)+dist.Uniform(0.,40.).log_prob(.2))
        return jnp.array([logp(actual)-ghost,logp(expected)]),jnp.stack([
            actual['obs']['fn'].loc,expected['obs']['fn'].loc])
    points = jnp.array([[r,b,.1,0.] for r in (.0032,.11,.707) for b in
        (0.,.17,.49,.93,1-r,1-r-1e-10,1-r+1e-10,1+r,1+r-1e-10,1+r+1e-10,1.8)])
    densities,fluxes = jax.jit(jax.vmap(targets))(points)
    gradients = jax.jit(jax.vmap(jax.jacrev(lambda p:targets(p)[0])))(points)
    assert np.isfinite(gradients).all()
    np.testing.assert_allclose(densities[:,0],densities[:,1],atol=2e-8,rtol=0)
    np.testing.assert_allclose(fluxes[:,0],fluxes[:,1],atol=2e-12,rtol=0)
    np.testing.assert_allclose(gradients[:,0],gradients[:,1],atol=2e-7,rtol=2e-6)

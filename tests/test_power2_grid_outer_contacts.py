"""Every contact neighborhood retains full orbital and LD derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet.power2_exact import light_curve
from models.jaxoplanet.power2_table import duration_light_curve,prepare_table
jax.config.update('jax_enable_x64',True)

@pytest.mark.slow
@pytest.mark.parametrize('impact',[0.,.4498,.93])
@pytest.mark.parametrize('radius',[.01,.1457,.5])
def test_neighborhoods_separately_preserve_six_parameter_jacobian(impact,radius):
    def reference(p,data):
        r,c,a,b,duration,epoch=p;phase=data-epoch
        squared=b*b+4*jnp.maximum(0.,(1+r)**2-b*b)*phase**2/duration**2
        z=jnp.where(squared>0,jnp.sqrt(jnp.where(squared>0,squared,1.)),0.)
        return jnp.where(jnp.abs(phase)<duration/2,light_curve(c,a,z,r,order=128),0.)
    def candidate(p,data):
        r,c,a,b,duration,epoch=p;phase=data-epoch
        return duration_light_curve(r,c,a,phase,jnp.abs(phase)<duration/2,
            duration=duration,impact=b,num_nodes=4097,order=64)
    expected=jax.jit(lambda p,t:(reference(p,t),jax.jacfwd(reference)(p,t)))
    actual=jax.jit(lambda p,t:(candidate(p,t),jax.jacfwd(candidate)(p,t),jax.jacrev(candidate)(p,t)))
    for duration in (.01,.11693087083333333,1.):
      epoch=.013
      contacts=[-duration/2,duration/2,0.]
      if impact<1-radius:
        inner=duration/2*np.sqrt(((1-radius)**2-impact**2)/((1+radius)**2-impact**2))
        contacts.extend([-inner,inner])
      for delta in (0.,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4):
       # Each distance is a distinct call. A nearer contact cannot trigger
       # a physical fallback that hides a farther interpolation error.
       times=jnp.asarray(np.r_[np.asarray(contacts)+epoch-delta,np.asarray(contacts)+epoch+delta])
       for c,a in ((0.,0.),(1.,.001),(.6,.4),(1.,1.)):
        p=jnp.array([radius,c,a,impact,duration,epoch]);e,d=expected(p,times);v,g,reverse=actual(p,times)
        np.testing.assert_allclose(v,e,atol=2e-9,rtol=0)
        np.testing.assert_allclose(g,d,atol=2e-7,rtol=0)
        np.testing.assert_allclose(reverse,d,atol=2e-7,rtol=0)


def test_dense_contact_data_retains_conservative_physical_fallback():
    radius=.5;impact=0.;duration=.01
    inner=duration/6
    phases=jnp.linspace(inner-1e-10,inner+1e-10,33)
    def evaluate(r):
        actual=duration_light_curve(r,1.,.001,phases,jnp.ones(33,dtype=bool),
            duration=duration,impact=impact,num_nodes=4097,order=64)
        z=jnp.sqrt(4*(1+r)**2*phases**2/duration**2)
        expected=light_curve(1.,.001,z,r,order=128)
        return jnp.stack((actual,expected))
    values=jax.jit(evaluate)(radius);gradients=jax.jit(jax.jacfwd(evaluate))(radius)
    np.testing.assert_allclose(values[0],values[1],atol=2e-12,rtol=0)
    np.testing.assert_allclose(gradients[0],gradients[1],atol=2e-7,rtol=0)

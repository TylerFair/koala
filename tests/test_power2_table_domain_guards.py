"""A shared routing hint cannot widen an independently audited table domain."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet.power2_table import prepare_table,duration_light_curve
from models.jaxoplanet.power2_exact import light_curve

@pytest.fixture(scope='module')
def table():
    return prepare_table(impact=.45,radius_bounds=(.12,.18),num_nodes=129)

@pytest.mark.parametrize('point,impact,hint,missing',[
    ([.145,-.2,.4],.45,None,False),
    ([.145,1.2,.4],.45,None,False),
    ([.145,-.2,.4],.45,True,False),
    ([.145,1.2,.4],.45,True,False),
    ([.3,.6,.4],.45,True,False),
    ([.145,.6,2.],.45,True,False),
    ([.145,.6,.4],.2,True,False),
    ([.145,.6,.4],.45,True,True),
    ([.145,.6,.4],.45,False,False),
])
def test_hint_never_overrides_radius_ld_or_geometry_domain(table,point,impact,hint,missing):
    active=dict(table,impact=-1.) if missing else table
    phases=jnp.linspace(-.049,.048,31)
    mask=jnp.ones(phases.shape,dtype=bool)
    def candidate(p):
        return duration_light_curve(*p,phases,mask,duration=.1,impact=impact,
            num_nodes=129,table=active,use_table=hint,strict_table_domain=True)
    def reference(p):
        r,c,alpha=p
        z=jnp.sqrt(impact**2+4*((1+r)**2-impact**2)*phases**2/.1**2)
        return light_curve(c,alpha,z,r,order=128)
    p=jnp.asarray(point)
    actual=jax.jit(lambda p:(candidate(p),jax.jacrev(candidate)(p)))(p)
    expected=jax.jit(lambda p:(reference(p),jax.jacfwd(reference)(p)))(p)
    np.testing.assert_allclose(actual[0],expected[0],atol=2e-12,rtol=0)
    np.testing.assert_allclose(actual[1],expected[1],atol=2e-7,rtol=0)

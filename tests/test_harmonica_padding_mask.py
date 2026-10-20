"""Production native Harmonica must preserve the unpadded target exactly."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density
from koala.data import _pad_spectro_cadences_exact
from models.harmonica.builder import create_vectorized_model

jax.config.update('jax_enable_x64', True)

@pytest.mark.parametrize('law,ld', [('quadratic', [.4,.2]),('power2',[.6,.5])])
def test_native_padding_preserves_full_target_and_gradient(law,ld):
    model=create_vectorized_model(detrend_type='quadratic',ld_mode='fixed',ld_profile=law)
    t=jnp.linspace(-.08,.08,17)
    y=jnp.ones((2,len(t)))
    errors=jnp.stack([jnp.linspace(.0003,.0008,len(t)),jnp.linspace(.0006,.001,len(t))])
    kw=dict(PERIOD=jnp.array([3.325271]),mu_t0=jnp.array([0.]),mu_b=jnp.array([.45]),
            harmonica_a_rs=jnp.array([13.6]),ld_fixed=jnp.array([ld,ld]))
    point=dict(rors=jnp.array([[.18],[.19]]),a1_frac=jnp.array([[.01],[-.01]]),
               c=jnp.array([1.,1.]),v=jnp.array([.001,-.002]),v2=jnp.array([.001,.003]),
               log_jitter=jnp.array([-9.,-9.]))
    tr=handlers.trace(handlers.seed(handlers.substitute(model,data=point),0)).get_trace(t,errors,y=y,**kw)
    p={n:s['value'] for n,s in tr.items() if s['type']=='sample' and not s['is_observed']}
    pt,py,pe,mask=_pad_spectro_cadences_exact(t,y,errors,multiple=32)
    def target(params,padded):
        return log_density(model,(pt,pe) if padded else (t,errors),
            dict(y=py,likelihood_mask=mask,**kw) if padded else dict(y=y,**kw),params)[0]
    va,ga=jax.jit(jax.value_and_grad(lambda p:target(p,False)))(p)
    vb,gb=jax.jit(jax.value_and_grad(lambda p:target(p,True)))(p)
    np.testing.assert_allclose(va,vb,atol=1e-7,rtol=0)
    for name in ga:np.testing.assert_allclose(ga[name],gb[name],atol=1e-7,rtol=1e-6)
    padded_trace=handlers.trace(handlers.seed(handlers.substitute(model,data=p),0)).get_trace(
        pt,pe,y=py,likelihood_mask=mask,**kw)
    np.testing.assert_allclose(tr['total_error']['value'],padded_trace['total_error']['value'],atol=1e-15,rtol=0)

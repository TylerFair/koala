"""Exercise the installed native backend, without substituting a mock."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.harmonica.core import compute_transit_model_harmonica_batched, compute_transit_model_harmonica

@pytest.mark.parametrize('law', ['quadratic', 'power2'])
@pytest.mark.parametrize('shared', [True, False])
def test_native_channel_time_axes_match_scalar_flux_and_gradients(law, shared):
    t=jnp.linspace(-.06,.06,31)
    times=t if shared else jnp.stack([t,t+.00013,t-.00017])
    p=dict(period=jnp.array([3.325271]),t0=jnp.array([0.]),a_rs=jnp.array([13.6]),b=jnp.array([.45]),
           ecc=jnp.array([.05]),omega=jnp.array([.7]),rors=jnp.array([[.17],[.18],[.19]]),a1=jnp.array([[.001],[-.002],[.0015]]))
    if law=='quadratic':p.update(u1_ld=jnp.array([.3,.4,.5]),u2_ld=jnp.array([.2,.1,.15]))
    else:p.update(c_ld=jnp.array([.5,.6,.7]),alpha_ld=jnp.array([.4,.6,.8]))
    def batched(r):return compute_transit_model_harmonica_batched(dict(p,rors=r[:,None]),times)
    def scalar(r):
        result=[]
        for i in range(3):
            lane={k:(v[i] if k in ['rors','a1','u1_ld','u2_ld','c_ld','alpha_ld'] else v) for k,v in p.items()}
            lane['rors']=r[i:i+1]
            result.append(compute_transit_model_harmonica(lane,t if shared else times[i]))
        return jnp.stack(result)
    r=p['rors'][:,0];weights=jnp.reshape(jnp.linspace(-1.,2.,93),(3,31))
    actual=jax.jit(batched)(r);expected=jax.jit(scalar)(r)
    assert actual.shape==(3,31)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    ga=jax.jit(jax.grad(lambda r:jnp.sum(weights*batched(r))))(r)
    ge=jax.jit(jax.grad(lambda r:jnp.sum(weights*scalar(r))))(r)
    np.testing.assert_allclose(ga,ge,rtol=1e-6,atol=1e-7)

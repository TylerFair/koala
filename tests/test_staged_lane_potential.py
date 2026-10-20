"""Staging preserves the full unconstrained target and dynamic model inputs."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
from models.independent_nuts import _stage_potential_generator


def test_staging_preserves_density_derivatives_and_dynamic_inputs():
    def model(t, err, y=None, offset=0., scale=1., law='linear'):
        q = numpyro.sample('q', dist.Beta(2., 3.))
        noise = numpyro.sample('noise', dist.LogNormal(-1., .5))
        shape = t if law == 'linear' else t*t
        numpyro.sample('obs', dist.Normal(offset+scale*q*shape,
                                         jnp.sqrt(err**2+noise**2)), obs=y)
    t = jnp.linspace(-1., 1., 11)
    err = jnp.full_like(t, .2)
    args = dict(y=.1+.3*t, offset=jnp.array(.1), scale=jnp.array(.8), law='quadratic')
    info = initialize_model(jax.random.PRNGKey(17), model, dynamic_args=True,
                            init_strategy=init_to_value(values={'q': .4, 'noise': .2}),
                            model_args=(t, err), model_kwargs=args)
    generate = _stage_potential_generator(info.potential_fn, {'law': 'quadratic'})
    for shift in (0., .3):
        changed = {**args, 'y': args['y']+shift, 'offset': args['offset']-.5*shift,
                   'scale': args['scale']+shift}
        original = info.potential_fn(t, err, **changed)
        staged = generate(t, err, **changed)
        point = {'q': jnp.array(-.2+shift), 'noise': jnp.array(-1.3-shift)}
        for derivative in (lambda f: f, jax.grad, jax.hessian):
            expected = derivative(original)(point)
            actual = derivative(staged)(point)
            for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-11)
    # The same staged function accepts a batch of independent positions.
    points = {name: jnp.stack([value, value+.1]) for name, value in point.items()}
    np.testing.assert_allclose(jax.vmap(staged)(points), jax.vmap(original)(points),
                               rtol=1e-11, atol=1e-11)

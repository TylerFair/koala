import jax
import jax.numpy as jnp
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
import numpy as np

def _to_f64(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
        return jnp.asarray(x, jnp.float64)
    return x

def _tree_to_f64(tree):
    return jax.tree_util.tree_map(_to_f64, tree)

def compute_transit_model(params, t):
    """
    Transit Model for one or more planets, using vmap for performance.
    Expects params to contain 'period', 'duration', 't0', 'b', 'rors', 'u'.
    These should be arrays where the 0-th dimension is the planet index,
    except 'u' which is limb darkening parameters.
    """
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
    total_flux = jnp.sum(batched_lcs, axis=0)
    return total_flux

def get_I_power2(c, alpha, u):
    return 1 - c*(1-jnp.power(u,alpha))

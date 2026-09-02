import jax
import jax.numpy as jnp
import numpy as np


def _to_f64(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
        return jnp.asarray(x, jnp.float64)
    return x


def _tree_to_f64(tree):
    return jax.tree_util.tree_map(_to_f64, tree)


def get_I_power2(c, alpha, u):
    return 1 - c * (1 - jnp.power(u, alpha))


def compute_transit_model_auto(params, t):
    """Dispatch to harmonica or jaxoplanet transit model based on params keys."""
    has_harmonica_power2_ld = all(
        key in params for key in ("c_ld", "alpha_ld")
    )
    has_harmonica_quadratic_ld = all(
        key in params for key in ("u1_ld", "u2_ld")
    )
    if "a_rs" in params and (
        has_harmonica_power2_ld or has_harmonica_quadratic_ld
    ):
        from .harmonica.core import compute_transit_model_harmonica
        return compute_transit_model_harmonica(params, t)
    from .jaxoplanet.core import compute_transit_model
    return compute_transit_model(params, t)

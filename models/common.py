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


def apply_systematics(lc_transit, trend):
    """Combine a transit signal with a systematics model multiplicatively.

    Koala's light-curve convention is ``F(t) = (1 + lc_transit) * trend``:
    ``lc_transit`` is the transit *signal* (0 out of transit, negative in
    transit, as returned by jaxoplanet and Harmonica) and ``trend`` is the
    systematics factor (baseline ``c`` near 1 plus polynomial, exponential,
    spot, step, or GP-template terms).  Every model builder and every
    reconstruction of a detrended light curve (``flux / trend``) uses this
    convention.
    """
    return (1.0 + lc_transit) * trend


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
    # Surface signals are normalized stellar-system flux minus one.  The
    # multiplicative systematics convention (``apply_systematics``) scales
    # them by the fitted baseline automatically.
    return compute_transit_model(params, t)

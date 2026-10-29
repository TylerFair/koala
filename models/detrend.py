import jax.numpy as jnp
import numpy as np

from .trends import (
    compute_lc_linear,
    compute_lc_quadratic,
    compute_lc_quadratic_spot,
    compute_lc_cubic,
    compute_lc_quartic,
    compute_lc_linear_discontinuity,
    compute_lc_explinear,
    compute_lc_explinear_spectroscopic,
    compute_lc_spot,
    compute_lc_2spot,
    compute_lc_none,
    compute_lc_constant,
    compute_lc_spot_spectroscopic,
    compute_lc_quadratic_spot_spectroscopic,
    compute_lc_2spot_spectroscopic,
    compute_lc_quadratic_2spot_spectroscopic,
    compute_lc_linear_discontinuity_spectroscopic,
    compute_lc_spot_linear_discontinuity,
    compute_lc_spot_explinear,
    compute_lc_2spot_explinear,
    compute_lc_spot_linear_discontinuity_spectroscopic,
    compute_lc_spot_explinear_spectroscopic,
    compute_lc_2spot_explinear_spectroscopic,
)
from .gp import (
    compute_lc_gp_mean,
    compute_lc_gp_spectroscopic,
    compute_lc_linear_gp_spectroscopic,
    compute_lc_quadratic_gp_spectroscopic,
    compute_lc_cubic_gp_spectroscopic,
    compute_lc_quartic_gp_spectroscopic,
    compute_lc_explinear_gp_spectroscopic,
    compute_lc_event_gp_spectroscopic,
)

COMPUTE_KERNELS = {
    'constant': compute_lc_constant,
    'linear': compute_lc_linear,
    'quadratic': compute_lc_quadratic,
    'cubic': compute_lc_cubic,
    'quartic': compute_lc_quartic,
    'linear_discontinuity': compute_lc_linear_discontinuity,
    'explinear': compute_lc_explinear,
    'explinear_spectroscopic': compute_lc_explinear_spectroscopic,
    'explinear_spectroscopic+gp_spectroscopic': compute_lc_explinear_gp_spectroscopic,
    'spot_spectroscopic+explinear_spectroscopic': compute_lc_spot_explinear_spectroscopic,
    '2spot_spectroscopic+explinear_spectroscopic': compute_lc_2spot_explinear_spectroscopic,
    'spot': compute_lc_spot,
    '2spot': compute_lc_2spot,
    'gp': compute_lc_gp_mean,
    'none': compute_lc_none,
    'gp_spectroscopic': compute_lc_gp_spectroscopic,
    'linear+gp_spectroscopic': compute_lc_linear_gp_spectroscopic,
    'quadratic+gp_spectroscopic': compute_lc_quadratic_gp_spectroscopic,
    'cubic+gp_spectroscopic': compute_lc_cubic_gp_spectroscopic,
    'quartic+gp_spectroscopic': compute_lc_quartic_gp_spectroscopic,
    'explinear+gp_spectroscopic': compute_lc_explinear_gp_spectroscopic,
    # spot crossings and steps stay parametric under a GP (their white-light templates, as without the GP)
    'spot_spectroscopic+gp_spectroscopic': compute_lc_event_gp_spectroscopic,
    '2spot_spectroscopic+gp_spectroscopic': compute_lc_event_gp_spectroscopic,
    'linear_discontinuity_spectroscopic+gp_spectroscopic': compute_lc_event_gp_spectroscopic,
    'spot_spectroscopic+linear_discontinuity_spectroscopic+gp_spectroscopic': compute_lc_event_gp_spectroscopic,
    'spot_spectroscopic': compute_lc_spot_spectroscopic,
    '2spot_spectroscopic': compute_lc_2spot_spectroscopic,
    'quadratic+2spot_spectroscopic': compute_lc_quadratic_2spot_spectroscopic,
    'linear_discontinuity_spectroscopic': compute_lc_linear_discontinuity_spectroscopic,
    'spot+linear_discontinuity': compute_lc_spot_linear_discontinuity,
    'spot+explinear': compute_lc_spot_explinear,
    '2spot+explinear': compute_lc_2spot_explinear,
    'spot_spectroscopic+linear_discontinuity_spectroscopic': compute_lc_spot_linear_discontinuity_spectroscopic,
    'spot_spectroscopic+explinear': compute_lc_spot_explinear_spectroscopic,
    '2spot_spectroscopic+explinear': compute_lc_2spot_explinear_spectroscopic,
}

COMPOSITE_KERNELS = {
    frozenset(name.split('+')): kernel
    for name, kernel in COMPUTE_KERNELS.items() if '+' in name
}
COMPOSITE_KERNELS.update({
    frozenset({'quadratic', 'spot'}): compute_lc_quadratic_spot,
    frozenset({'quadratic', 'spot_spectroscopic'}): compute_lc_quadratic_spot_spectroscopic,
})


_DETREND_TYPE_ALIASES = {
    'quadratic_spot': 'quadratic+spot',
    'quadratic_spot_spectroscopic': 'quadratic+spot_spectroscopic',
}


def _canonical_detrend_type(detrend_type):
    return _DETREND_TYPE_ALIASES.get(detrend_type, detrend_type)


def _split_components(detrend_type):
    return set(_canonical_detrend_type(detrend_type).split('+'))


def resolve_detrend_kernel(detrend_type):
    if detrend_type in COMPUTE_KERNELS:
        return COMPUTE_KERNELS[detrend_type]
    detrend_components = frozenset(_split_components(detrend_type))
    if detrend_components in COMPOSITE_KERNELS:
        return COMPOSITE_KERNELS[detrend_components]
    raise KeyError(detrend_type)


def _prepare_power2_poly(degree=12, n_mu=300):
    mus = jnp.linspace(0.0, 1.0, n_mu, endpoint=True)
    x = jnp.vander(1.0 - mus, N=degree + 1, increasing=True)[:, 1:]
    p = jnp.asarray(np.linalg.pinv(np.asarray(x)))
    return mus, p

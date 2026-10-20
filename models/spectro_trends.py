"""Initial coordinates for normalized spectroscopic polynomial priors."""
import math
import numpy as np


def normalized_trend_initial_values(time, initial):
    """Convert elapsed-day starts through every supported polynomial order.

    No integration is averaged into another; the mean defines the origin.
    """
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or not time.size or not np.all(np.isfinite(time)):
        raise ValueError('Normalized trend requires distinct finite integration times.')
    span = np.ptp(time)
    if span <= 0:
        raise ValueError('Normalized trend requires distinct finite integration times.')
    offset = np.mean(time - time.min())
    names = ['c', 'v', 'v2', 'v3', 'v4']
    order = max((index for index, name in enumerate(names) if name in initial), default=0)
    c = np.asarray(initial.get('c', 1.0))
    physical = [np.asarray(initial.get(name, np.zeros_like(c)))
                if index else c for index, name in enumerate(names[:order + 1])]
    result = dict(initial)
    for column, name in enumerate(names[:order + 1]):
        result['trend_' + name] = span**column * sum(
            math.comb(row, column) * offset**(row - column) * physical[row]
            for row in range(column, order + 1))
    return result


def resolve_spectral_trend_defaults(flags):
    """Resolve public sampled-polynomial defaults before cache fingerprinting.

    Explicit elapsed-day priors and non-polynomial template/GP families retain
    their declared coordinates. Gaussian integration remains an explicit mode.
    """
    resolved = dict(flags)
    trend = str(resolved.get('detrending_type', 'linear'))
    orders = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4}
    sampled = resolved.get('trend_inference', 'sampled_uniform') == 'sampled_uniform'
    basis = 'batman_normalized' if sampled and trend in orders else 'elapsed_days'
    resolved.setdefault('spectro_trend_basis', basis)
    if sampled and resolved['spectro_trend_basis'] == 'batman_normalized' and trend in orders:
        names = ['c'] + ['v' if i == 1 else f'v{i}' for i in range(1, orders[trend] + 1)]
        bounds = {name: [-10., 10.] for name in names}
        bounds.update(resolved.get('spectro_trend_bounds') or {})
        resolved['spectro_trend_bounds'] = bounds
    return resolved
SPECTRAL_TREND_BOUND_NAMES=frozenset({'c','v','v2','v3','v4','A','log_tau',
                                    'A_spot','A_spot2','A_jump','A_gp'})

def validate_spectral_trend_bounds(bounds):
    """Explicit sampled-coefficient bounds, including saved-white templates."""
    if bounds is None:return None
    if not hasattr(bounds,'items'):
        raise ValueError('trend_bounds must map spectral coefficient names to finite ordered pairs.')
    result={}
    for name,pair in bounds.items():
        try:
            values=np.asarray(pair,dtype=float)
            valid=(name in SPECTRAL_TREND_BOUND_NAMES and values.shape==(2,)
                   and np.all(np.isfinite(values)) and values[0]<values[1]
                   and all(not isinstance(x,(bool,np.bool_)) for x in pair))
        except (TypeError,ValueError):valid=False
        if not valid:
            raise ValueError('trend_bounds requires finite ordered spectral coefficient bound pairs; '
                             f'invalid {name!r}: {pair!r}.')
        result[name]=tuple(map(float,values))
    return result

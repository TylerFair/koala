"""Small shared banks of independently validated fixed-geometry tables."""
from pathlib import Path
import json
import numpy as np

from .fitting import _key


def pack_tables(tables, num_nodes):
    """Pad Chebyshev and compressed ranks, preserving every coefficient.

    Missing entries receive an impossible negative impact, so the physical
    fallback always handles their lanes. No interpolation crosses geometries.
    """
    present = [t for t in tables if t is not None]
    if not present:
        return None, None
    nr = max(t['coefficients'].shape[0] for t in present)
    na = max(t['coefficients'].shape[1] for t in present)
    factors = []
    for table in tables:
        if table is None:
            factors.append(None)
        elif 'moment_left' in table:
            factors.append((table['moment_left'], table['moment_right']))
        else:
            a, b, nodes = table['coefficients'].shape
            factors.append((np.eye(a*b).reshape(a, b, a*b),
                            table['coefficients'].reshape(a*b, nodes)))
    rank = max(pair[0].shape[2] for pair in factors if pair is not None)
    count = len(tables)
    bank = dict(moment_left=np.zeros((count, nr, na, rank)),
        moment_right=np.zeros((count, rank, num_nodes)),
        area_coefficients=np.zeros((count, nr, num_nodes)),
        radius_bounds=np.tile([.01, .02], (count, 1)),
        exponent_bounds=np.tile([0., 1.], (count, 1)), impact=np.full(count, -1.))
    for index, (table, pair) in enumerate(zip(tables, factors)):
        if table is None:
            continue
        left, right = pair
        a, b, k = left.shape
        if right.shape[1] != num_nodes:
            raise ValueError('All bank tables must have the same node count.')
        bank['moment_left'][index, :a, :b, :k] = left
        bank['moment_right'][index, :k] = right
        bank['area_coefficients'][index, :a] = table['area_coefficients']
        for name in ('radius_bounds', 'exponent_bounds', 'impact'):
            bank[name][index] = table[name]
    return bank, (nr, na, num_nodes)


def prepare_bank(medians, coefficients, free_ld, channels, cache_dir, code):
    """Cache audited banks; rejected domains retain direct cadence integration."""
    from models.jaxoplanet.power2_table import prepare_table, validated_grid_size
    from .scoring import _save_npz
    nodes = 4097
    inputs = (medians, coefficients, free_ld, channels, nodes, code)
    path = Path(cache_dir) / f'power2_tables_{_key(inputs)}.npz'
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists():
        report = json.loads(metadata.read_text())
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved), tuple(report['shape']), dict(report, cache_reused=True)
    tables, domains = [], []
    for member, (impact, radius) in enumerate(zip(medians['b_0'], medians['rors_0'])):
        impact = abs(float(impact))
        lower = max(np.sqrt(1e-5), float(radius)-.08)
        upper = min(np.sqrt(.5), float(radius)+.08, 1-impact-.005)
        maximum = max(1. if free_ld[member] else 0.,
                      float(np.max(coefficients[member*channels:(member+1)*channels, 1])))
        maximum = max(maximum, 1.)
        table = None
        reason = None
        if not (0 < lower < radius < upper < 1-impact):
            reason = 'No non-grazing table domain brackets the fitted radius.'
        else:
            try:
                validated_grid_size(impact=impact, radius_bounds=(lower, upper),
                                    exponent_bounds=(0., maximum), minimum_nodes=nodes)
                table = prepare_table(impact=impact, radius_bounds=(lower, upper),
                                      exponent_bounds=(0., maximum), num_nodes=nodes)
            except ValueError as error:
                reason = str(error)
        tables.append(table)
        domains.append(dict(impact=impact, radius_bounds=[lower, upper],
                            exponent_bounds=[0., maximum], validated=table is not None, reason=reason))
    bank, shape = pack_tables(tables, nodes)
    report = dict(shape=shape, domains=domains, cache_reused=False,
                  fallback='Direct physical cadence integration outside each audited domain.')
    metadata.write_text(json.dumps(report, indent=2))
    if bank is not None:
        _save_npz(path, **bank)
    return bank, shape, report

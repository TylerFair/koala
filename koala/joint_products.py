"""Combined spectra of a joint run: transit depth and transit time across all datasets.

No JAX imports, so the figures can also be made afterwards from a finished run:

    python -m koala.joint_products results/WASP-107_joint
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Display resolution of the binned joint figure. Spectral channels only, never integrations.
DISPLAY_RESOLUTION = {'NIRISS': 50.0, 'NIRSpec': 100.0}
DISPLAY_GROUP = {'MIRI': 2}
# MIRI/LRS beyond 10 um is too noisy to show next to the near-infrared; the CSVs keep it.
DISPLAY_MAX_WAVELENGTH = {'MIRI': 10.0}
# Changes in each channel's fitted depth and transit time when only the leading (morning) or
# only the trailing (evening) half of the planet grows by one stellar radius.
RESPONSE_COLUMNS = ('d_depth_dRL', 'd_depth_dRT', 'd_dt0_dRL_s_per_Rstar', 'd_dt0_dRT_s_per_Rstar')
EVENING_COLOR = '#c0392b'
MORNING_COLOR = '#2166ac'


def _instrument(filename):
    name = filename.upper()
    if 'MIRI' in name:
        return 'MIRI'
    if 'NIRISS' in name or 'SOSS' in name:
        return 'NIRISS'
    if 'NIRSPEC' in name:
        return 'NIRSpec'
    if 'NIRCAM' in name:
        return 'NIRCam'
    return 'other'


def _label(filename):
    """A readable label from a koala output name, e.g. 'NIRSpec G395H nrs1'."""
    words = Path(filename).name.split('_')[1:]
    keep = [w for w in words if w and not w.startswith(('Rreference', 'Rnative', 'pix'))
            and w not in ('time', 'shift.csv') and not (w[0] == 'R' and w[1:].isdigit())]
    return ' '.join(keep).replace('NIRSPEC', 'NIRSpec').replace('order', 'order ').replace('  ', ' ').strip()


def collect_joint_spectra(output_dir):
    """Stack every dataset's timing table under a joint output directory."""
    tables = []
    for member in sorted(p for p in Path(output_dir).iterdir() if p.is_dir() and p.name != 'joint'):
        for path in sorted(member.glob('*_time_shift.csv')):
            table = pd.read_csv(path)
            table.insert(0, 'dataset', member.name)
            table.insert(1, 'instrument', _instrument(path.name))
            table.insert(2, 'label', _label(path.name))
            tables.append(table)
    if not tables:
        return None
    combined = pd.concat(tables, ignore_index=True)
    return combined.sort_values('wavelength', kind='stable').reset_index(drop=True)


def _edges(part):
    low = part.wave_lo_um if 'wave_lo_um' in part else part.wavelength - part.wavelength_err
    high = part.wave_hi_um if 'wave_hi_um' in part else part.wavelength + part.wavelength_err
    return low.to_numpy(float), high.to_numpy(float)


def _groups(low, high, instrument, resolution, group):
    """Labels for runs of consecutive channels that reach the target width (or count)."""
    labels, current, start = np.zeros(len(low), dtype=int), 0, 0
    for i in range(len(low)):
        labels[i] = current
        if instrument in resolution:
            # Close the bin at whichever edge comes nearest the target width lambda / R.
            target = 0.5 * (high[i] + low[start]) / resolution[instrument]
            width = high[i] - low[start]
            wider = high[i + 1] - low[start] if i + 1 < len(low) else np.inf
            done = abs(width - target) <= abs(wider - target)
        else:
            done = i - start + 1 >= group.get(instrument, 1)
        if done:
            current, start = current + 1, i + 1
    return labels


def display_bin(table, resolution=None, group=None):
    """Inverse-variance bin consecutive channels of each dataset, for display only.

    NIRISS channels are combined to about R = 50 and NIRSpec to about R = 100, MIRI channels in
    pairs. Depth and timing are averaged with their own weights; their correlation is ignored,
    which is fine for a figure but not for analysis.
    """
    resolution = DISPLAY_RESOLUTION if resolution is None else resolution
    group = DISPLAY_GROUP if group is None else group
    rows = []
    for (dataset, instrument), part in table.groupby(['dataset', 'instrument'], sort=False):
        part = part.sort_values('wavelength')
        low, high = _edges(part)
        for _, sub in part.groupby(_groups(low, high, instrument, resolution, group)):
            sub_low, sub_high = _edges(sub)
            wd, wt = 1 / sub.depth_err**2, 1 / sub.t0_err_s**2
            lo, hi = float(sub_low.min()), float(sub_high.max())
            row = dict(
                dataset=dataset, instrument=instrument, label=sub.label.iloc[0], channels=len(sub),
                wavelength=0.5 * (lo + hi), wavelength_err=0.5 * (hi - lo), wave_lo_um=lo, wave_hi_um=hi,
                depth=float(np.sum(wd * sub.depth) / np.sum(wd)), depth_err=float(np.sum(wd) ** -0.5),
                dt0_s=float(np.sum(wt * sub.dt0_s) / np.sum(wt)), t0_err_s=float(np.sum(wt) ** -0.5))
            for column, weight in zip(RESPONSE_COLUMNS, (wd, wd, wt, wt)):
                if column in sub:
                    row[column] = float(np.sum(weight * sub[column]) / np.sum(weight))
            rows.append(row)
    return pd.DataFrame(rows).sort_values('wavelength', kind='stable').reset_index(drop=True)


# Number of grid bins two visits must share before their relative depth and timing offsets are fitted.
IVW_MIN_SHARED_BINS = 3


def _ivw_grid(table, resolution, group):
    """Bin edges at the display resolution of the finest instrument covering each wavelength.

    Instruments without a target resolution (MIRI) use their own channel resolution divided by their
    display group size, so a stretch covered only by them is binned as in the binned figure.
    """
    reach = {}
    for instrument, part in table.groupby('instrument'):
        low, high = _edges(part)
        target = resolution.get(instrument)
        if target is None:
            target = float(np.median(part.wavelength / (high - low))) / group.get(instrument, 1)
        reach[instrument] = (low.min(), high.max(), float(np.clip(np.nan_to_num(target, nan=100.0), 1.0, 1e5)))
    edges = [min(low for low, _, _ in reach.values())]
    end = max(high for _, high, _ in reach.values())
    while edges[-1] < end:
        here = [target for low, high, target in reach.values() if low <= edges[-1] < high]
        edges.append(edges[-1] * (1 + 1 / max(here)) if here else
                     min(low for low, _, _ in reach.values() if low > edges[-1]))
    return np.array(edges)


def _visits(table):
    """One label per visit: datasets from the same exposure share the visit's white-light epoch."""
    if 't0_bjd' not in table:
        return table.dataset.to_numpy()
    epoch = (table.t0_bjd - table.dt0_s / 86400.0).round(4)
    names = {e: '+'.join(sorted(table.dataset[epoch == e].unique())) for e in epoch.unique()}
    return epoch.map(names).to_numpy()


def _ivw(values, errors):
    weights = 1.0 / np.asarray(errors, float) ** 2
    return float(np.sum(weights * values) / np.sum(weights)), float(np.sum(weights) ** -0.5)


def _overlap_offsets(cells, column, error):
    """Per-visit constants that line up overlapping visits, from weighted least squares.

    `cells` has one row per visit and grid bin. In every bin that two or more visits share, the
    model is value = spectrum(bin) + offset(visit). Visits linked through at least
    IVW_MIN_SHARED_BINS shared bins form a group, whose offsets have zero mean when weighted by each
    visit's precision in the shared bins. A visit that overlaps no other keeps a zero offset.
    """
    cells = cells[cells.groupby('bin').visit.transform('nunique') > 1]
    visits = sorted(cells.visit.unique())
    in_bin = [set(s) for _, s in cells.groupby('bin').visit]
    parent = {v: v for v in visits}

    def root(v):
        while parent[v] != v:
            v = parent[v]
        return v

    for i, a in enumerate(visits):
        for b in visits[i + 1:]:
            if sum(a in s and b in s for s in in_bin) >= IVW_MIN_SHARED_BINS:
                parent[root(b)] = root(a)
    rows = []
    for group in sorted({root(v) for v in visits}):
        members = [v for v in visits if root(v) == group]
        if len(members) < 2:
            continue
        part = cells[cells.visit.isin(members)]
        part = part[part.groupby('bin').visit.transform('nunique') > 1]
        bins = np.array(sorted(part.bin.unique()))
        # Unknowns: one value per shared bin, then an offset for every visit but the first (zero).
        design = np.zeros((len(part), len(bins) + len(members) - 1))
        design[np.arange(len(part)), np.searchsorted(bins, part.bin)] = 1.0
        for k, visit in enumerate(members[1:]):
            design[(part.visit == visit).to_numpy(), len(bins) + k] = 1.0
        value, weight = part[column].to_numpy(), 1.0 / part[error].to_numpy() ** 2
        normal = design.T @ (design * weight[:, None])
        solution = np.linalg.solve(normal, design.T @ (weight * value))
        chi2 = float(np.sum(weight * (value - design @ solution) ** 2))
        offset = np.concatenate([[0.0], solution[len(bins):]])
        covariance = np.zeros((len(members), len(members)))
        covariance[1:, 1:] = np.linalg.inv(normal)[len(bins):, len(bins):]
        # Move the zero to the precision-weighted mean of the group.
        precision = np.array([weight[(part.visit == v).to_numpy()].sum() for v in members])
        centre = np.eye(len(members)) - precision[None, :] / precision.sum()
        offset, covariance = centre @ offset, centre @ covariance @ centre.T
        for k, visit in enumerate(members):
            rows.append(dict(visit=visit, offset=offset[k], offset_err=np.sqrt(max(covariance[k, k], 0.0)),
                             shared_bins=len(bins), chi2=chi2, dof=len(part) - design.shape[1]))
    return pd.DataFrame(rows, columns=['visit', 'offset', 'offset_err', 'shared_bins', 'chi2', 'dof'])


def ivw_combine(table, resolution=None, group=None):
    """Average overlapping datasets channel by channel on one wavelength grid, for display.

    Every channel goes to the bin of a common grid that holds its central wavelength, and all
    channels of all datasets in a bin are averaged with inverse-variance weights. The grid has the
    display resolution of the finest instrument covering each wavelength, as in `display_bin`:
    R ~ 100 where NIRSpec covers it, R ~ 50 where only NIRISS does, MIRI channels in pairs.
    Depths and transit times of different visits can differ by a constant: instruments disagree in
    absolute depth, and each visit's times are measured from its own white-light epoch. Before averaging, each visit's depths and times are shifted by a constant
    fitted where visits overlap (see `_overlap_offsets`); the second table returned lists these
    offsets, their errors and the chi-square of the overlap. As in `display_bin`, depth and time
    are averaged with their own weights and their correlation is ignored. The errors do not include
    the uncertainty of the offsets, which moves a whole visit up or down together. Returns
    (None, None) when no two visits share a grid bin.
    """
    resolution = DISPLAY_RESOLUTION if resolution is None else resolution
    group = DISPLAY_GROUP if group is None else group
    table = table.assign(visit=_visits(table))
    edges = _ivw_grid(table, resolution, group)
    table['bin'] = np.clip(np.searchsorted(edges, table.wavelength, side='right') - 1, 0, len(edges) - 2)
    if not (table.groupby('bin').visit.nunique() > 1).any():
        return None, None
    # Each visit on the common grid, to measure how the visits line up where they overlap.
    cells = []
    for (visit, index), part in table.groupby(['visit', 'bin']):
        depth, depth_err = _ivw(part.depth, part.depth_err)
        dt0, t0_err = _ivw(part.dt0_s, part.t0_err_s)
        cells.append(dict(visit=visit, bin=index, depth=depth, depth_err=depth_err, dt0_s=dt0, t0_err_s=t0_err))
    cells = pd.DataFrame(cells)
    fitted = {'depth': _overlap_offsets(cells, 'depth', 'depth_err').set_index('visit'),
              'dt0_s': _overlap_offsets(cells, 'dt0_s', 't0_err_s').set_index('visit')}
    offsets = pd.DataFrame({'visit': sorted(table.visit.unique())})
    for column, name, scale in (('depth', 'depth_offset_ppm', 1e6), ('dt0_s', 't0_offset_s', 1.0)):
        found = fitted[column]
        offsets[name] = offsets.visit.map(found.offset).fillna(0.0) * scale
        offsets[name.replace('offset', 'offset_err')] = offsets.visit.map(found.offset_err).fillna(0.0) * scale
        offsets[name.split('_offset')[0] + '_overlap_chi2'] = offsets.visit.map(found.chi2)
        offsets[name.split('_offset')[0] + '_overlap_dof'] = offsets.visit.map(found.dof)
        table[column] = table[column] - table.visit.map(found.offset).fillna(0.0)
    rows = []
    for index, sub in table.groupby('bin'):
        wd, wt = 1 / sub.depth_err**2, 1 / sub.t0_err_s**2
        depth, depth_err = _ivw(sub.depth, sub.depth_err)
        dt0, t0_err = _ivw(sub.dt0_s, sub.t0_err_s)
        lo, hi = float(edges[index]), float(edges[index + 1])
        row = dict(
            dataset='+'.join(sorted(sub.dataset.unique())), instrument='+'.join(sorted(sub.instrument.unique())),
            label=' + '.join(sorted(sub.label.unique())), visits=sub.visit.nunique(), datasets=sub.dataset.nunique(),
            channels=len(sub),
            wavelength=0.5 * (lo + hi), wavelength_err=0.5 * (hi - lo), wave_lo_um=lo, wave_hi_um=hi,
            depth=depth, depth_err=depth_err, dt0_s=dt0, t0_err_s=t0_err)
        for column, weight in zip(RESPONSE_COLUMNS, (wd, wd, wt, wt)):
            if column in sub:
                row[column] = float(np.sum(weight * sub[column]) / np.sum(weight))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('wavelength', kind='stable').reset_index(drop=True), offsets


# Wavelength axis for spectra that include MIRI: linear up to 5 um, logarithmic beyond, where a
# factor e in wavelength takes 1.5 um of axis. MIRI/LRS (5-10 um) then takes about a fifth of a
# 0.6-10 um axis instead of half of it.
_LOG_BEYOND = 5.0
_LOG_WIDTH = 1.5


def _hybrid_forward(wavelength):
    wavelength = np.clip(np.asarray(wavelength, dtype=float), 1e-3, None)
    return np.where(wavelength <= _LOG_BEYOND, wavelength,
                    _LOG_BEYOND + _LOG_WIDTH * np.log(np.maximum(wavelength, _LOG_BEYOND) / _LOG_BEYOND))


def _hybrid_inverse(position):
    position = np.asarray(position, dtype=float)
    return np.where(position <= _LOG_BEYOND, position,
                    _LOG_BEYOND * np.exp((np.maximum(position, _LOG_BEYOND) - _LOG_BEYOND) / _LOG_WIDTH))


def _limit_to_bulk(ax, wavelength, value, error):
    """Show the central 98% of points; mark the few beyond it at the panel edge.

    A handful of very noisy channels (e.g. the red end of MIRI/LRS) would otherwise squash the
    rest of the spectrum. Every value stays in the CSV next to the figure.
    """
    from .plotting_style import SPECTRUM_COLOR
    low, high = _bulk_range(value, error)
    ax.set_ylim(low, high)
    _mark_beyond(ax, wavelength, value, low, high, SPECTRUM_COLOR)


def _bulk_range(value, error):
    # The range follows the precise channels, so a few noisy ones cannot swamp a good spectrum.
    value, error = np.asarray(value, dtype=float), np.asarray(error, dtype=float)
    precise = error <= 3 * np.nanmedian(error)
    low, high = np.nanpercentile(value[precise] if precise.sum() >= 5 else value, [1, 99])
    pad = max(0.12 * (high - low), 2 * float(np.nanmedian(error)))
    return low - pad, high + pad


def _mark_beyond(ax, wavelength, value, low, high, color, alpha=1.0):
    for mask, edge, marker in ((value > high, high, '^'), (value < low, low, 'v')):
        if np.any(mask):
            ax.plot(wavelength[mask], np.full(mask.sum(), edge), linestyle='none', marker=marker, ms=7,
                    mfc='white', mec=color, mew=1.5, alpha=alpha, clip_on=False, zorder=3)


def _wavelength_axis(top, bottom, out):
    """Shared wavelength axis of two stacked panels."""
    from matplotlib.ticker import FixedLocator, NullLocator
    from .plotting_style import style_axis

    low = float(np.nanmin(out.wavelength - out.wavelength_err))
    high = float(np.nanmax(out.wavelength + out.wavelength_err))
    if high > 5.5 and low < _LOG_BEYOND:
        # Near-infrared plus MIRI: linear up to 5 um, logarithmic beyond (see _LOG_BEYOND).
        bottom.set_xscale('function', functions=(_hybrid_forward, _hybrid_inverse))
        # Reach the 10 um tick when MIRI ends just short of it.
        high = max(high, 10.4) if high > 8.0 else high
        bottom.set_xlim(low - 0.02 * (_LOG_BEYOND - low), high)
        ticks = [t for t in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0) if low <= t <= high * 1.01]
        for ax in (top, bottom):
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            # Mark where the axis turns logarithmic.
            ax.axvline(_LOG_BEYOND, color='0.75', lw=0.8, ls=':', zorder=0)
        bottom.set_xticklabels([f'{t:g}' for t in ticks])
        for ax in (top, bottom):
            style_axis(ax, x_locator=False)
    else:
        pad = 0.02 * (high - low)
        bottom.set_xlim(low - pad, high + pad)
        for ax in (top, bottom):
            style_axis(ax)


def _draw(table, stem, label=None):
    """Two panels on one wavelength axis: transit depth, then the transit-time offset."""
    from .plotting_style import (
        DATA_MARKER_STYLE, SPECTRUM_COLOR, ZERO_LINE_COLOR, apply_publication_style,
        save_figure, style_axis,
    )
    import matplotlib.pyplot as plt

    limit = table.instrument.map(DISPLAY_MAX_WAVELENGTH).fillna(np.inf) if 'instrument' in table else np.inf
    out = table.assign(depth_ppm=table.depth * 1e6, depth_err_ppm=table.depth_err * 1e6,
                       shown=table.wavelength <= limit)
    out.to_csv(f'{stem}.csv', index=False)
    out = out[out.shown]
    apply_publication_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={'height_ratios': [1.25, 1]},
    )
    style = dict(color=SPECTRUM_COLOR, ecolor=SPECTRUM_COLOR, zorder=2, **DATA_MARKER_STYLE)
    for ax, value, error in ((top, out.depth_ppm, out.depth_err_ppm), (bottom, out.dt0_s, out.t0_err_s)):
        ax.errorbar(out.wavelength, value, xerr=out.wavelength_err, yerr=error, **style)
        _limit_to_bulk(ax, out.wavelength.to_numpy(), value.to_numpy(), error.to_numpy())
    bottom.axhline(0, color=ZERO_LINE_COLOR, ls='--', lw=1.0, zorder=0)
    top.set_ylabel('Transit depth [ppm]')
    bottom.set_ylabel(r'$t_0 - t_{0,\mathrm{ref}}$ [s]')
    bottom.set_xlabel(r'Wavelength [$\mu$m]')
    if label:
        top.text(0.02, 0.95, label, transform=top.transAxes, ha='left', va='top', fontsize=15, color='0.3')

    _wavelength_axis(top, bottom, out)
    fig.align_ylabels((top, bottom))
    for extension in ('png', 'pdf'):
        save_figure(fig, f'{stem}.{extension}', close=False)
    plt.close(fig)
    return Path(f'{stem}.png')


def limb_spectra(table, offsets=None, draws=4000, seed=0):
    """Morning and evening transit depths from each channel's depth and transit time.

    A channel's spherical-planet fit returns a depth and a transit time. To first order, a planet
    whose leading (morning) and trailing (evening) halves have radii R_L and R_T gives

        depth - depth(R) = d_depth_dRL (R_L - R) + d_depth_dRT (R_T - R),
        dt0 - offset     = d_dt0_dRL (R_L - R) + d_dt0_dRT (R_T - R),

    with the responses koala computes for every channel. Taking R = sqrt(depth) and solving gives
    R_L and R_T; the depths reported are R_L**2 and R_T**2, the depth of a planet made of that half.

    `offsets` maps a dataset to its visit offset in seconds: the transit time of a planet with
    equal halves, relative to the visit's white-light time. The default, zero, assumes the two
    halves are equal when averaged over the white-light band. Another offset moves the evening
    depths up and the morning depths down by nearly the same amount at every wavelength, so the
    shapes of features carry over but their overall morning-evening difference does not. Errors
    are the 16th and 84th percentiles of Gaussian draws of depth and time with their correlation;
    they are conditional on the white-light orbit and epoch, like the depths and times themselves.
    The two depths share one timing measurement with opposite signs, so their errors are strongly
    anticorrelated (`corr_morning_evening`); mirror-image wiggles in the two spectra are expected
    from noise alone. Returns None when the table has no radius responses.
    """
    if not set(RESPONSE_COLUMNS) <= set(table.columns):
        return None
    offsets = offsets or {}
    rng = np.random.default_rng(seed)
    depth, depth_err = table.depth.to_numpy(float), table.depth_err.to_numpy(float)
    t0, t0_err = table.dt0_s.to_numpy(float), table.t0_err_s.to_numpy(float)
    corr = (table.corr_depth_t0.fillna(0.0).to_numpy(float) if 'corr_depth_t0' in table
            else np.zeros(len(table)))
    offset = table.dataset.map(offsets).fillna(0.0).to_numpy(float) if 'dataset' in table else 0.0
    a, b, c, d = (table[column].to_numpy(float) for column in RESPONSE_COLUMNS)
    determinant = a * d - b * c
    # Correlated Gaussian draws of (depth, time), shape (draws, channels).
    z1, z2 = rng.standard_normal((2, draws, len(table)))
    depth_draws = depth + depth_err * z1
    time_draws = t0 - offset + t0_err * (corr * z1 + np.sqrt(1 - corr**2) * z2)
    radius = np.sqrt(np.clip(depth_draws, 0.0, None))
    # The depth equation has zero left-hand side, since R is the channel's own fitted radius.
    # Datasets written without radius responses give NaN.
    with np.errstate(invalid='ignore', divide='ignore'):
        leading = radius + (-b * time_draws) / determinant
        trailing = radius + (a * time_draws) / determinant
    out = table.copy()
    out['visit_offset_s'] = offset
    for name, half in (('morning', leading), ('evening', trailing)):
        low, mid, high = np.percentile(half**2, [16, 50, 84], axis=0) * 1e6
        out[f'depth_{name}_ppm'] = mid
        out[f'depth_{name}_err_minus_ppm'] = mid - low
        out[f'depth_{name}_err_plus_ppm'] = high - mid
    # Both halves share one timing measurement with opposite signs, so their errors are strongly
    # anticorrelated; a retrieval of the two spectra needs this correlation.
    morning, evening = leading**2 - np.mean(leading**2, axis=0), trailing**2 - np.mean(trailing**2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['corr_morning_evening'] = np.mean(morning * evening, axis=0) / (
            np.std(leading**2, axis=0) * np.std(trailing**2, axis=0))
    difference = trailing - leading
    out['radius_evening_minus_morning_Rstar'] = np.median(difference, axis=0)
    out['radius_evening_minus_morning_err_Rstar'] = 0.5 * np.diff(np.percentile(difference, [16, 84], axis=0), axis=0)[0]
    return out


def _draw_limbs(table, stem):
    """Evening and morning depths in two stacked panels, the evening in front above and the morning below."""
    from .plotting_style import DATA_MARKER_STYLE, apply_publication_style, save_figure
    import matplotlib.pyplot as plt

    limit = table.instrument.map(DISPLAY_MAX_WAVELENGTH).fillna(np.inf) if 'instrument' in table else np.inf
    out = table.assign(shown=table.wavelength <= limit)
    out.to_csv(f'{stem}.csv', index=False)
    out = out[out.shown]
    apply_publication_style()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    series = {
        name: (out[f'depth_{name}_ppm'].to_numpy(),
               np.vstack([out[f'depth_{name}_err_minus_ppm'], out[f'depth_{name}_err_plus_ppm']]), color)
        for name, color in (('evening', EVENING_COLOR), ('morning', MORNING_COLOR))
    }
    values = np.concatenate([value for value, _, _ in series.values()])
    errors = np.concatenate([error.mean(axis=0) for _, error, _ in series.values()])
    low, high = _bulk_range(values, errors)
    wavelength = out.wavelength.to_numpy()
    for ax, front in ((top, 'evening'), (bottom, 'morning')):
        for name, (value, error, color) in series.items():
            alpha, zorder = (1.0, 3) if name == front else (0.25, 2)
            ax.errorbar(wavelength, value, xerr=out.wavelength_err, yerr=error, color=color, ecolor=color,
                        alpha=alpha, zorder=zorder, label=f'{name.capitalize()} limb', **DATA_MARKER_STYLE)
            _mark_beyond(ax, wavelength, value, low, high, color, alpha)
        ax.set_ylabel('Transit depth [ppm]')
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(f'{front.capitalize()} limb'), 1 - labels.index(f'{front.capitalize()} limb')]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left', frameon=False)
    top.set_ylim(low, high)
    bottom.set_xlabel(r'Wavelength [$\mu$m]')
    _wavelength_axis(top, bottom, out)
    fig.align_ylabels((top, bottom))
    for extension in ('png', 'pdf'):
        save_figure(fig, f'{stem}.{extension}', close=False)
    plt.close(fig)
    return Path(f'{stem}.png')


def plot_joint_spectra(output_dir):
    """Write depth and timing figures for every finished dataset, and the joint ones for two or more.

    joint/joint_spectrum.*         every channel of every dataset on one wavelength axis;
    joint/joint_spectrum_binned.*  the same, display-binned (NIRISS R~50, NIRSpec R~100, MIRI pairs);
    joint/datasets/<dataset>.*     each dataset on its own, in the same style (also written as
                                   <dataset>/<prefix>_depth_and_t0.* next to its other products).
    joint/limb_spectra.*           morning and evening depths from each channel's depth and time
                                   (see limb_spectra), and limb_spectra_binned.* from the binned ones.
    joint/joint_spectrum_ivw.*     when visits overlap in wavelength: all datasets averaged on one
                                   grid (see ivw_combine), with limb_spectra_ivw.* and the fitted
                                   visit offsets in ivw_offsets.csv.
    Each figure has a CSV of its values; the `shown` column marks what is plotted.
    """
    table = collect_joint_spectra(output_dir)
    if table is None:
        return None
    joint_dir = Path(output_dir) / 'joint'
    (joint_dir / 'datasets').mkdir(parents=True, exist_ok=True)
    limbs = limb_spectra(table)
    if limbs is not None:
        _draw_limbs(limbs, str(joint_dir / 'limb_spectra'))
        _draw_limbs(limb_spectra(display_bin(table)), str(joint_dir / 'limb_spectra_binned'))
    for dataset, part in table.groupby('dataset', sort=False):
        figure = _draw(part, str(joint_dir / 'datasets' / dataset), label=part.label.iloc[0])
        # The same figure next to the dataset's other products.
        for source in Path(output_dir, dataset).glob('*_time_shift.csv'):
            _draw(part, str(source).replace('_time_shift.csv', '_depth_and_t0'), label=part.label.iloc[0])
    if table['dataset'].nunique() < 2:
        return figure
    _draw(display_bin(table), str(joint_dir / 'joint_spectrum_binned'))
    plot_ivw_spectra(table, joint_dir)
    return _draw(table, str(joint_dir / 'joint_spectrum'))


def plot_ivw_spectra(table, joint_dir):
    """The combined depth, timing and limb figures of overlapping datasets; None without overlap."""
    combined, offsets = ivw_combine(table)
    if combined is None:
        return None
    offsets.to_csv(Path(joint_dir) / 'ivw_offsets.csv', index=False)
    limbs = limb_spectra(combined)
    if limbs is not None:
        _draw_limbs(limbs, str(Path(joint_dir) / 'limb_spectra_ivw'))
    return _draw(combined, str(Path(joint_dir) / 'joint_spectrum_ivw'))


if __name__ == '__main__':
    for directory in sys.argv[1:]:
        print(plot_joint_spectra(directory))

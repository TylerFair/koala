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
            rows.append(dict(
                dataset=dataset, instrument=instrument, label=sub.label.iloc[0], channels=len(sub),
                wavelength=0.5 * (lo + hi), wavelength_err=0.5 * (hi - lo), wave_lo_um=lo, wave_hi_um=hi,
                depth=float(np.sum(wd * sub.depth) / np.sum(wd)), depth_err=float(np.sum(wd) ** -0.5),
                dt0_s=float(np.sum(wt * sub.dt0_s) / np.sum(wt)), t0_err_s=float(np.sum(wt) ** -0.5)))
    return pd.DataFrame(rows).sort_values('wavelength', kind='stable').reset_index(drop=True)


# Hybrid wavelength axis for spectra that include MIRI: log over 0.6-3 um, linear over 3-5 um
# (NIRSpec), log over 5-12 um, taking 22%, 56% and 22% of the width.
_HYBRID = (0.6, 3.0, 5.0, 12.0)
_HYBRID_WIDTHS = (0.22, 0.56, 0.22)


def _hybrid_forward(wavelength):
    start, first, second, end = _HYBRID
    left, middle, right = _HYBRID_WIDTHS
    wavelength = np.clip(np.asarray(wavelength, dtype=float), 1e-3, None)
    return np.where(
        wavelength <= first, left * np.log(wavelength / start) / np.log(first / start),
        np.where(wavelength <= second, left + middle * (wavelength - first) / (second - first),
                 left + middle + right * np.log(wavelength / second) / np.log(end / second)))


def _hybrid_inverse(position):
    start, first, second, end = _HYBRID
    left, middle, right = _HYBRID_WIDTHS
    position = np.asarray(position, dtype=float)
    return np.where(
        position <= left, start * (first / start) ** (position / left),
        np.where(position <= left + middle, first + (position - left) / middle * (second - first),
                 second * (end / second) ** ((position - left - middle) / right)))


def _limit_to_bulk(ax, wavelength, value, error):
    """Show the central 98% of points; mark the few beyond it at the panel edge.

    A handful of very noisy channels (e.g. the red end of MIRI/LRS) would otherwise squash the
    rest of the spectrum. Every value stays in the CSV next to the figure.
    """
    from .plotting_style import SPECTRUM_COLOR
    # The range follows the precise channels, so a few noisy ones cannot swamp a good spectrum.
    precise = error <= 3 * np.nanmedian(error)
    low, high = np.nanpercentile(value[precise] if precise.sum() >= 5 else value, [1, 99])
    pad = max(0.12 * (high - low), 2 * float(np.nanmedian(error)))
    low, high = low - pad, high + pad
    ax.set_ylim(low, high)
    for mask, edge, marker in ((value > high, high, '^'), (value < low, low, 'v')):
        if np.any(mask):
            ax.plot(wavelength[mask], np.full(mask.sum(), edge), linestyle='none', marker=marker, ms=7,
                    mfc='white', mec=SPECTRUM_COLOR, mew=1.5, clip_on=False, zorder=3)


def _draw(table, stem, label=None):
    """Two panels on one wavelength axis: transit depth, then the transit-time offset."""
    from .plotting_style import (
        DATA_MARKER_STYLE, SPECTRUM_COLOR, ZERO_LINE_COLOR, apply_publication_style,
        save_figure, style_axis,
    )
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

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

    low = float(np.nanmin(out.wavelength - out.wavelength_err))
    high = float(np.nanmax(out.wavelength + out.wavelength_err))
    if high > 5.5 and low < 3.0:
        # Near-infrared plus MIRI: log over 0.6-3 um, linear over 3-5 um, log over 5-12 um.
        bottom.set_xscale('function', functions=(_hybrid_forward, _hybrid_inverse))
        bottom.set_xlim(min(low, 0.58), high)
        ticks = [t for t in (1.0, 2.0, 3.0, 4.0, 5.0, 10.0) if low <= t <= high * 1.01]
        for ax in (top, bottom):
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
        bottom.set_xticklabels([f'{t:g}' for t in ticks])
        for ax in (top, bottom):
            style_axis(ax, x_locator=False)
    else:
        pad = 0.02 * (high - low)
        bottom.set_xlim(low - pad, high + pad)
        for ax in (top, bottom):
            style_axis(ax)
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
    Each figure has a CSV of its values; the `shown` column marks what is plotted.
    """
    table = collect_joint_spectra(output_dir)
    if table is None:
        return None
    joint_dir = Path(output_dir) / 'joint'
    (joint_dir / 'datasets').mkdir(parents=True, exist_ok=True)
    for dataset, part in table.groupby('dataset', sort=False):
        figure = _draw(part, str(joint_dir / 'datasets' / dataset), label=part.label.iloc[0])
        # The same figure next to the dataset's other products.
        for source in Path(output_dir, dataset).glob('*_time_shift.csv'):
            _draw(part, str(source).replace('_time_shift.csv', '_depth_and_t0'), label=part.label.iloc[0])
    if table['dataset'].nunique() < 2:
        return figure
    _draw(display_bin(table), str(joint_dir / 'joint_spectrum_binned'))
    return _draw(table, str(joint_dir / 'joint_spectrum'))


if __name__ == '__main__':
    for directory in sys.argv[1:]:
        print(plot_joint_spectra(directory))

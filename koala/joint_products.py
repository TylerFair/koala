"""Combined spectra of a joint run: transit depth and transit time across all datasets.

No JAX imports, so the figure can also be made afterwards from a finished run:

    python -m koala.joint_products results/WASP-107_joint
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def collect_joint_spectra(output_dir):
    """Stack every dataset's timing table under a joint output directory."""
    tables = []
    for member in sorted(p for p in Path(output_dir).iterdir() if p.is_dir() and p.name != 'joint'):
        for path in sorted(member.glob('*_time_shift.csv')):
            table = pd.read_csv(path)
            table.insert(0, 'dataset', member.name)
            tables.append(table)
    if not tables:
        return None
    combined = pd.concat(tables, ignore_index=True)
    return combined.sort_values('wavelength', kind='stable').reset_index(drop=True)


def _limit_to_bulk(ax, wavelength, value, error):
    """Show the central 98% of points; mark the few beyond it at the panel edge.

    A handful of very noisy channels (e.g. the red end of MIRI/LRS) would otherwise squash the
    rest of the spectrum. Every value stays in joint_spectrum.csv.
    """
    from .plotting_style import SPECTRUM_COLOR
    low, high = np.nanpercentile(value, [1, 99])
    pad = max(0.12 * (high - low), 2 * float(np.nanmedian(error)))
    low, high = low - pad, high + pad
    ax.set_ylim(low, high)
    for mask, edge, marker in ((value > high, high, '^'), (value < low, low, 'v')):
        if np.any(mask):
            ax.plot(wavelength[mask], np.full(mask.sum(), edge), linestyle='none', marker=marker, ms=7,
                    mfc='white', mec=SPECTRUM_COLOR, mew=1.5, clip_on=False, zorder=3)


def plot_joint_spectra(output_dir):
    """Write joint/joint_spectrum.{png,pdf,csv} when more than one dataset has a timing table.

    The top panel is the transit depth and the bottom panel the offset of each channel's transit
    time from its visit's white-light time, for every dataset on one wavelength axis.
    """
    from .plotting_style import (
        DATA_MARKER_STYLE, SPECTRUM_COLOR, ZERO_LINE_COLOR, apply_publication_style,
        save_figure, style_axis,
    )
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    table = collect_joint_spectra(output_dir)
    if table is None or table['dataset'].nunique() < 2:
        return None
    joint_dir = Path(output_dir) / 'joint'
    joint_dir.mkdir(parents=True, exist_ok=True)
    out = table.assign(depth_ppm=table.depth * 1e6, depth_err_ppm=table.depth_err * 1e6)
    out.to_csv(joint_dir / 'joint_spectrum.csv', index=False)

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

    low = float(np.nanmin(out.wavelength - out.wavelength_err))
    high = float(np.nanmax(out.wavelength + out.wavelength_err))
    if high > 5.5:
        # MIRI reaches 12 um; a log axis keeps the near-infrared readable.
        bottom.set_xscale('log')
        bottom.set_xlim(low * 0.97, high * 1.03)
        ticks = [t for t in (0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12) if low <= t <= high]
        for ax in (top, bottom):
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(ScalarFormatter())
        for ax in (top, bottom):
            style_axis(ax, x_locator=False)
    else:
        pad = 0.02 * (high - low)
        bottom.set_xlim(low - pad, high + pad)
        for ax in (top, bottom):
            style_axis(ax)
    fig.align_ylabels((top, bottom))
    for extension in ('png', 'pdf'):
        save_figure(fig, joint_dir / f'joint_spectrum.{extension}', close=False)
    plt.close(fig)
    return joint_dir / 'joint_spectrum.png'


if __name__ == '__main__':
    for directory in sys.argv[1:]:
        print(plot_joint_spectra(directory))

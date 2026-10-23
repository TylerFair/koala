"""Draw the figures on the three-instrument WASP-39 b page.

Run from the repository root:
    python tools/docs/three_instruments_figures.py                    # the data figure only
    python tools/docs/three_instruments_figures.py --run results/WASP-39_three_instruments
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from generate_tutorial_figures import COLOR, OUT, ROOT, finish, style

PERIOD, T0 = 4.05528043, 59787.555           # BJD_TDB - 2400000
DATA = [  # file, extension of the flux (time is the last one), wavelengths kept, label, transit number
    ('WASP-39_soss_binned8.fits', 3, (0.85, 2.85), 'NIRISS/SOSS order 1', 0),
    ('WASP-39_g395h_nrs1_binned8.fits', 3, (2.9, 5.0), 'NIRSpec/G395H NRS1', 1),
    ('WASP-39_g395h_nrs2_binned8.fits', 3, (2.9, 5.0), 'NIRSpec/G395H NRS2', 1),
    ('WASP-39_miri_lrs.fits', 3, (5.0, 12.0), 'MIRI/LRS', 50),
]


def white_light(name, flux_ext, wavelengths):
    with fits.open(ROOT / 'examples' / 'data' / name) as hdul:
        wave = hdul[flux_ext - 2].data
        flux = hdul[flux_ext].data
        time = hdul[-1].data + 0.5                    # BMJD_TDB in the file
    keep = np.isfinite(wave) & (wave >= wavelengths[0]) & (wave <= wavelengths[1])
    white = np.nansum(flux[:, keep], axis=1)
    return time, white / np.nanmedian(white)


def data_figure():
    fig, axes = plt.subplots(len(DATA), 1, figsize=(6.9, 7.4), sharex=True, layout='constrained')
    for ax, (name, ext, wavelengths, label, epoch) in zip(axes, DATA):
        time, white = white_light(name, ext, wavelengths)
        hours = (time - (T0 + epoch * PERIOD)) * 24
        ax.plot(hours, white, '.', ms=2, color=COLOR, alpha=0.5, rasterized=True)
        ax.set_title(label, loc='left')
        ax.set_ylabel('Flux')
    axes[-1].set_xlabel('Hours from the predicted mid-transit')
    finish(fig, 'three_instruments_data.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, help='output directory of examples/wasp39_three_instruments.yaml')
    args = parser.parse_args()
    style()
    data_figure()
    if args.run is not None:
        # koala's own joint spectrum, as written by the run.
        shutil.copyfile(args.run / 'figures' / 'joint_spectrum.png', OUT / 'three_instruments_spectrum.png')


if __name__ == '__main__':
    main()

"""Write the compact NIRSpec/G395H and MIRI/LRS example files for the three-instrument case study.

Usage: python tools/docs/make_example_data.py SOURCE_DIR

SOURCE_DIR holds the exoTEDRF extractions WASP-39_nrs{1,2}_box_spectra_fullres_G395H.fits and
WASP-39_MIRI-LRS_box_spectra_fullres.fits. Every integration is kept. The G395H files are summed
in groups of eight adjacent detector columns, like examples/data/WASP-39_soss_binned8.fits; the
MIRI/LRS file keeps every column between 4.9 and 12.1 µm and nothing else.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

OUT = Path(__file__).resolve().parents[2] / 'examples' / 'data'
EXTENSIONS = ('WAVE', 'WAVE ERR', 'FLUX', 'FLUX ERR', 'TIME')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with fits.open(path) as hdul:
        return hdul[0].header.copy(), [np.asarray(hdul[i].data) for i in range(1, 6)]


def bin_columns(wave, wave_err, flux, error, factor):
    """Sum flux in groups of ``factor`` columns; centres are the mean, widths span the outer edges."""
    n = wave.size // factor * factor
    groups = lambda a: a[..., :n].reshape(a.shape[:-1] + (n // factor, factor))
    with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', RuntimeWarning)
        centre = np.nanmean(groups(wave), axis=-1)
        low = np.nanmin(groups(wave - wave_err), axis=-1)
        high = np.nanmax(groups(wave + wave_err), axis=-1)
    return centre, (high - low) / 2, groups(flux).sum(-1), np.sqrt((groups(error) ** 2).sum(-1))


def write(path, header, arrays, history):
    primary = fits.PrimaryHDU(header=header)
    for line in history:
        primary.header['HISTORY'] = line
    hdus = [primary] + [fits.ImageHDU(a, name=name) for a, name in zip(arrays, EXTENSIONS)]
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)
    print(f'{path.name}: {path.stat().st_size / 1e6:.1f} MB, sha256 {sha(path)}')


def main(source):
    source = Path(source)
    for detector in ('nrs1', 'nrs2'):
        name = source / f'WASP-39_{detector}_box_spectra_fullres_G395H.fits'
        header, (wave, wave_err, flux, error, time) = read(name)
        binned = bin_columns(wave, wave_err, flux, error, 8)
        header.update(INST=('NIRSPEC/G395H', 'Instrument used to acquire the data'),
                      DETECTOR=(detector.upper(), 'NIRSpec detector'),
                      CONTENTS=('Teaching subset; binned by 8 columns', 'Description'),
                      PROGID=(1366, 'JWST ERS program'), BINFACT=(8, 'Adjacent detector columns per sample'),
                      ORIGNPIX=(wave.size, 'Input detector columns'), SRCFILE=(name.name, 'Local source extraction'),
                      SRCSHA=(sha(name)[:32], 'Start of source SHA256'))
        write(OUT / f'WASP-39_g395h_{detector}_binned8.fits', header, (*binned, time),
              ['Flux summed in groups of 8 adjacent detector columns.',
               f'Flux errors combined in quadrature; all {time.size} integrations retained.'])
    name = source / 'WASP-39_MIRI-LRS_box_spectra_fullres.fits'
    header, (wave, wave_err, flux, error, time) = read(name)
    # koala's exoTEDRF reader drops five padding columns at each end, so keep five more each side.
    inside = np.flatnonzero((wave > 4.9) & (wave < 12.1))
    keep = np.zeros(wave.size, bool)
    keep[inside[0] - 5:inside[-1] + 6] = True
    header.update(CONTENTS=('Teaching subset; 4.9 to 12.1 um', 'Description'),
                  PROGID=(2783, 'JWST DDT program'), OBSNUM=(1, 'JWST program observation'),
                  ORIGNPIX=(wave.size, 'Input detector columns'), SRCFILE=(name.name, 'Local source extraction'),
                  SRCSHA=(sha(name)[:32], 'Start of source SHA256'))
    write(OUT / 'WASP-39_miri_lrs.fits', header,
          (wave[keep], wave_err[keep], flux[:, keep], error[:, keep], time),
          [f'Only the {keep.sum()} columns around 4.9 to 12.1 um kept; nothing binned.',
           f'All {time.size} integrations retained.'])


if __name__ == '__main__':
    main(sys.argv[1])

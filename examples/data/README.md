# Example data

`WASP-39_soss_binned8.fits` is a compact, real NIRISS/SOSS time-series
extraction for trying Koala. It contains all 537 integrations and both
spectral orders from the source extraction. Every group of eight adjacent
detector columns was combined to reduce the file from 35.3 MB to about 4.4 MB:

- wavelength centers are the mean of the finite input centers;
- wavelength errors span the outer edges of the combined columns;
- fluxes are summed;
- independent flux errors are added in quadrature.

The source FITS header identifies the target as WASP-39, the instrument as
NIRISS/SOSS, the extraction as a 30-pixel box, and the reduction pipeline as
exoTEDRF. The observation is from the public JWST Transiting Exoplanet
Community Early Release Science program 1366, observation 21. STScI identifies
that observation as the program's public NIRISS/SOSS time series:

- [STScI note on public JWST time-series observations](https://jwst-docs.stsci.edu/methods-and-roadmaps/jwst-time-series-observations/jwst-time-series-observations-tso-saturation)
- [exoTEDRF documentation](https://exotedrf.readthedocs.io/)

This reduced file is supplied as a software example, not as a new science data
release. Use MAST and the relevant program publications for archival products,
calibration provenance, and scientific citation. Do not use this compact
eight-column product for an independent published spectrum.

SHA-256: `aaafe700a62d9065a94bb32c1ac5fda4a9278536bb1b7fc4cf2a47912c3a6721`

The primary header records the transformation and source filename. To inspect
the file:

```python
from astropy.io import fits

fits.info("examples/data/WASP-39_soss_binned8.fits")
```

## NIRSpec/G395H and MIRI/LRS

The three-instrument case study adds three more compact files, made from
exoTEDRF extractions with `tools/docs/make_example_data.py`. All
integrations are kept in every file.

- `WASP-39_g395h_nrs1_binned8.fits` and `WASP-39_g395h_nrs2_binned8.fits`:
  NIRSpec/G395H NRS1 and NRS2 from the Transiting Exoplanet Community ERS
  program 1366 (465 integrations), with every eight adjacent detector columns
  combined in the same way as the SOSS file.
- `WASP-39_miri_lrs.fits`: MIRI/LRS from DDT program 2783, observation 1
  (1779 integrations), with every detector column kept between 4.9 and
  12.1 µm and nothing combined.

The time in all four files is the JWST BMJD_TDB; koala adds the half day to
work in BJD_TDB - 2400000. As with the SOSS file, these are software examples,
not a science data release.

SHA-256:

- `WASP-39_g395h_nrs1_binned8.fits`: `805b8ea63ca2e3e4f31c4cfe7a28ed38bce093b504dc16fea2184e677920565c`
- `WASP-39_g395h_nrs2_binned8.fits`: `f5c4d7f3669677b399487bb271ef26a731488d02b14061dc110ad58b9e949340`
- `WASP-39_miri_lrs.fits`: `259ed992dd3c024fdc963c28a7d34ec4a08ce16002bfd99929ba5d731ecdaf2a`

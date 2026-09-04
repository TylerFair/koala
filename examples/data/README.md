# Example data

`WASP-39_soss_binned8.fits` is a compact, real NIRISS/SOSS time-series
extraction for trying penumbra. It contains all 537 integrations and both
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

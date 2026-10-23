# Instruments and input files

koala starts from extracted spectra: a flux and uncertainty for every integration and wavelength, and the time of each integration.
It doesn't reduce raw JWST data itself, so you'll need a reduction from a pipeline like exoTEDRF, Eureka! or SPARTA first.

## Supported instruments

Set `instrument` to one of these names (capitalization doesn't matter).
Some modes also need the detector or the SOSS order:

| `instrument` | Also set | Wavelengths kept (µm) |
|---|---|---|
| `NIRISS/SOSS` | `order: 1` or `2` | the whole order |
| `NIRSPEC/PRISM` | `nrs: 1` or `2` | 0.6–5.0 |
| `NIRSPEC/G395H`, `NIRSPEC/G395M` | `nrs: 1` or `2` | 2.9–5.0 |
| `NIRSPEC/G235H`, `NIRSPEC/G235M` | `nrs: 1` or `2` | 1.66–3.1 |
| `NIRSPEC/G140H`, `NIRSPEC/G140M` (F100LP) | `nrs: 1` or `2` | 1.0–1.8 (G140H), 0.97–1.87 (G140M) |
| `NIRSPEC/G140H-F070`, `NIRSPEC/G140M-F070` | `nrs: 1` or `2` | 0.82–1.27 (G140H), 0.7–1.27 (G140M) |
| `NIRCAM/F322W2` | | 2.4–4.1 |
| `NIRCAM/F444W` | | 3.8–5.0 |
| `MIRI/LRS` | | 5.0–12.0 |

Wavelengths outside these ranges are dropped when the file is read.
You can narrow the range further with `wavelength_filter` (see [configuring a fit](configuration.md#wavelength-channels)).
MIRI/MRS and the imaging modes aren't supported.

Each SOSS order and each NIRSpec detector is its own fit, with its own YAML file.
To measure one orbit from several of them at once, see [WASP-39 b with both SOSS orders](../case_studies/transmission_soss.md#more-orders-detectors-and-visits).

## Input files

koala reads three kinds of file, and works out which one it has from the file itself:

| Pipeline | File |
|---|---|
| exoTEDRF | the extracted-spectra FITS file from stage 3 (`*_box_spectra_fullres.fits` and similar), with both SOSS orders in one file |
| Eureka! | the stage 3 `*_SpecData.h5` (one column per wavelength) or the stage 4 `*_LCData.h5` (already binned in wavelength) |
| SPARTA | the pickle written by `gather_and_filter.py` |

If you'd rather say which one it is, set `input_format` to `exotedrf`, `eureka` or `sparta`.
When koala reads the file, it prints the number of integrations, the wavelength range and what it found, so check those first.

A few things are worth knowing about the inputs:

- Times should be barycentric (BJD_TDB). koala works in BJD_TDB − 2400000, and converts the BMJD_TDB (BJD_TDB − 2400000.5) that JWST files carry, or full BJDs, when it reads them. If no transit allowed by your YAML falls inside the data, koala stops with an error, which is usually a sign that the two use different zero points.
- koala fits every integration you give it. Integrations with a non-finite time, flux or uncertainty are dropped.
- Pixels a pipeline has flagged as bad (Eureka! and SPARTA) are kept but given a huge uncertainty, so they carry no weight in the fit.

# koala

koala fits JWST transits, eclipses and phase curves, first in white light and then in wavelength channels.
It supports NIRISS/SOSS, NIRSpec/PRISM and the G140H/M, G235H/M and G395H/M gratings, NIRCam/F322W2 and F444W grisms, and MIRI/LRS.
It reads exoTEDRF extracted-spectra FITS, SPARTA filtered pickles, and Eureka! Stage 3 `SpecData.h5` or Stage 4 `LCData.h5` files, and fits them with JAX and NumPyro on CPUs or GPUs.

Start here: [Installation](install.md) → [Your first fit](quickstart.md) → [Case studies](case_studies/index.md) → [Reference](reference/index.md).

```{toctree}
:maxdepth: 1

install
quickstart
case_studies/index
reference/index
```

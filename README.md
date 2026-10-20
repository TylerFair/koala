<p align="center">
  <img src="docs/_static/koala_logo.png" width="480" alt="koala">
</p>

# koala

[![Tests](https://github.com/TylerFair/koala/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/TylerFair/koala/actions/workflows/tests.yml)

koala fits JWST transit and eclipse time series with JAX and NumPyro.
It fits the white-light curve first, then each wavelength channel, and writes a transmission or emission spectrum with the fitted light curves.
You can run it on a CPU or GPU, and every integration contributes to the fit.

![WASP-39 b transmission spectrum from the bundled NIRISS/SOSS example](docs/_static/soss_spectrum.png)

## Installation

Clone the repository to get koala and the example data:

```bash
git clone https://github.com/TylerFair/koala.git
cd koala
python -m pip install .
```

Start here: [Installation](https://koala-jwst.readthedocs.io/en/latest/install.html) → [Your first fit](https://koala-jwst.readthedocs.io/en/latest/quickstart.html) → [Case studies](https://koala-jwst.readthedocs.io/en/latest/case_studies/index.html) → [Reference](https://koala-jwst.readthedocs.io/en/latest/reference/index.html).

The first fit uses the bundled WASP-39 b NIRISS/SOSS observation and runs from a notebook or the command line.

## Citing

Please cite the repository and record the commit used for your analysis.
See [citing koala](https://koala-jwst.readthedocs.io/en/latest/citing.html) for the methods and dependencies to cite.

## Contributing

Bug reports and contributions are welcome on [GitHub](https://github.com/TylerFair/koala).
The [development guide](docs/testing.md) explains how to run the tests, and the license terms are in [LICENSE](LICENSE) and [licenses/](licenses/).

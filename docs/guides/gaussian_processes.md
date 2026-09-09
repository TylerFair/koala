# Gaussian processes

Appending `+gp` to a trend (for example `linear+gp`) adds a Matérn-3/2
Gaussian process from [tinygp](https://tinygp.readthedocs.io/) to the
white-light fit. The GP absorbs correlated residual structure that the mean
trend does not describe; the spectroscopic channels then use the saved
white-light trend as a fixed regressor.

```yaml
flags:
  detrending_type: linear+gp
```

Add a GP only after choosing an adequate mean trend, and compare its inferred
depth with a simpler accepted model: a GP can trade against the transit shape
when its timescale overlaps ingress or egress. GP trends cannot be combined
with `trend_inference: gaussian_marginalized`.

## Installation

`requirements.txt` installs tinygp from the Koala fork at a pinned commit,
which contains the parallel associative-scan solver that Koala uses on GPUs:

```bash
python -m pip install 'tinygp @ git+https://github.com/TylerFair/tinygp.git@96d110c8fb4350b0daf50f7f1f86a9196b82fcd4'
```

An environment that still has tinygp 0.3.x from PyPI keeps working with the
serial solver.

## Solver selection

The serial and parallel solvers evaluate the same model; the choice only
changes the algorithm. Set `KOALA_GP_SOLVER` before starting Python, or the
optional top-level configuration key `gp_solver` for one run:

| Value | Meaning |
|---|---|
| `auto` | Default. Parallel on a GPU or TPU when the installed tinygp supports it; serial on a CPU. |
| `serial` | Always use the serial quasiseparable solver. |
| `parallel` | Require the parallel solver; raises an error with installation guidance if it is unavailable. |

To compare the two on your own data, fit the same configuration twice into
separate output directories:

```bash
KOALA_GP_SOLVER=serial python fit_jwst.py -c config_serial.yaml
KOALA_GP_SOLVER=parallel python fit_jwst.py -c config_parallel.yaml
```

The two `*_whitelight_GP_database.csv` tables (`gp_flux`, `gp_err`,
`gp_trend`) and the white-light best-fit parameters should agree to within
sampling noise.

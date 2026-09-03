# Examples

Each YAML file is a complete configuration for one dataset. Copy the closest
one, point `path`, `input_dir` and `fits_file` at your extracted spectra, and
run it:

```bash
python fit_jwst.py -c examples/niriss_soss_order1.yaml
```

| File | What it shows |
|---|---|
| `niriss_soss_order1.yaml` | NIRISS/SOSS order 1 with a linear baseline. Start here. |
| `nirspec_g395m.yaml` | NIRSpec detector selection and masking a bad stretch of the time series. |
| `nirspec_prism.yaml` | PRISM at native resolution with an exponential ramp, and why its chunk width is small. |
| `limb_darkening_stack.yaml` | Fitting one dataset under several limb-darkening treatments and marginalising over them. |
| `plot_spectrum.py` | Reading the output spectrum CSV. |

The examples set only what the dataset requires. Sampler settings, chunk
widths and prior parameterisations are left at their defaults, which are
documented in the configuration guide; override them only when you have a
reason to.

Everything not shown here — trend models, limb-darkening prescriptions,
sampler behaviour, output files — is in the [documentation](../docs).

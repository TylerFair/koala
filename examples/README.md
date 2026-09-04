# Examples

The two executed notebooks are the most direct walkthroughs. They write a YAML
configuration, run the fitter, read the resulting CSV products, and render the
white-light curve and spectrum with the shared publication style. Their saved
outputs were produced on a V100 GPU, while the configuration source retains
`host_device: "cpu"` so a machine without a GPU can rerun them (more slowly).

Each YAML file remains available as a complete configuration for one dataset.
Copy the closest one, point `path`, `input_dir` and `fits_file` at your extracted
spectra, and run it:

```bash
python fit_jwst.py -c examples/niriss_soss_order1.yaml
```

| File | What it shows |
|---|---|
| `niriss_soss_order1.ipynb` | Executed WASP-39 SOSS order-1 fit, output inspection, and embedded publication plots. |
| `nirspec_g395m.ipynb` | Executed HAT-P-18 G395M/NRS1 fit, detector masking, and embedded publication plots. |
| `niriss_soss_order1.yaml` | NIRISS/SOSS order 1 with a linear baseline. Start here. |
| `nirspec_g395m.yaml` | NIRSpec detector selection and masking a bad stretch of the time series. |
| `nirspec_prism.yaml` | PRISM at native resolution with an exponential ramp, and why its chunk width is small. |
| `limb_darkening_stack.yaml` | Fitting one dataset under several limb-darkening treatments and marginalising over them. |
| `plot_spectrum.py` | Reading the output spectrum CSV. |

The examples set only what the dataset requires. Sampler settings and prior
parameterisations are left at their defaults. The native-cadence PRISM example
is the one exception: it lowers the advanced resident-channel width to fit that
long time series comfortably on a V100.

Everything not shown here — trend models, limb-darkening prescriptions,
sampler behaviour, output files — is in the [documentation](../docs).

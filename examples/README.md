# Examples

These are configuration files to start from.
Run them from the top of the repository, for example:

```bash
koala -c examples/niriss_soss_order1.yaml
```

The WASP-39 b examples use the NIRISS/SOSS observation in `data/`, so they run as they are:

- `niriss_soss_order1.yaml` is the fit in [Your first fit](https://koala-jwst.readthedocs.io/en/latest/quickstart.html).
- `joint_soss_orders.yaml` fits one orbit to both SOSS orders (with `joint_soss_order1.yaml` and `joint_soss_order2.yaml`), as in the [both-orders case study](https://koala-jwst.readthedocs.io/en/latest/case_studies/transmission_soss.html).

The eclipse, phase-curve and spotted-star examples use generated data, made with `python tools/example_phase_curves.py eclipse` (or `phase_curve`, or `stellar_spots`):

- `eclipse.yaml` is a synthetic MIRI/LRS eclipse of a rocky planet.
- `phase_curve.yaml` is a synthetic NIRSpec/G395M phase curve.
- `stellar_spots.yaml` is a transit across a rotating spotted star.

The others are set up for real observations whose extracted spectra aren't included, so you'll need to point `path`, `input_dir` and `input_file` at your own:

- `nirspec_g395h.yaml`: WASP-121 b with G395H, a quadratic trend and power-2 limb darkening with stellar-model priors.
- `nirspec_g395m.yaml`: HAT-P-18 b with G395M, leaving out a range of integrations.
- `nirspec_prism.yaml`: WASP-6 b with PRISM at the native resolution and an exponential ramp.
- `nircam_f322w2.yaml`: HAT-P-18 b with the NIRCam F322W2 grism.
- `limb_darkening_stack.yaml`: a list of limb-darkening models to combine, for the [model stacking case study](https://koala-jwst.readthedocs.io/en/latest/case_studies/model_stacking.html).

`plot_spectrum.py` reads a spectrum CSV and plots it.

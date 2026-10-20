# Examples

Start with `niriss_soss_order1.yaml`.
It includes a complete fit to the bundled WASP-39 b observation and uses numeric $R=100$ bins, so you don't need a separate wavelength-grid file:

```bash
cp examples/niriss_soss_order1.yaml my_transit.yaml
# edit the data paths and target parameters
koala -c my_transit.yaml
# Equivalent: python -m koala -c my_transit.yaml
```

The SOSS example reads `data/WASP-39_soss_binned8.fits` and downloads the ExoTiC-LD files it needs on first use.
It uses the default quadratic limb darkening with uniform priors on the sum and difference of the coefficients.
For the other observed targets, supply your own exoTEDRF, SPARTA, or Eureka! extraction.
Set `path`, `input_dir`, `input_file`, and `stellar.ld_data_path` to your local copies.
The input format is detected automatically.
The eclipse, phase-curve, and stellar-spot examples have separate instructions for generating synthetic data.

Choose the example closest to your observation:

- `niriss_soss_order1.yaml` fits a SOSS order-1 transit with a linear trend.
- `nirspec_g395m.yaml` selects a NIRSpec detector and masks an integration interval.
- `nirspec_g395h.yaml` fits G395H/NRS1 with a quadratic baseline.
- `nirspec_prism.yaml` fits native-grid PRISM with an exponential ramp and small GPU batches.
- The [Harmonica case study](../docs/case_studies/harmonica.md) fits an asymmetric planetary boundary.
- `limb_darkening_stack.yaml` compares limb-darkening choices for the same observation.
- `eclipse.yaml` fits a synthetic rocky-planet MIRI/LRS eclipse.
- `phase_curve.yaml` fits a generated thermal phase curve.
- `stellar_spots.yaml` fits a generated transit with rotating Starry spots.
- `plot_spectrum.py` reads and plots a spectrum CSV.

The YAML files specify the dataset and scientific choices.
They generally keep the sampler defaults, with explicit overrides where the example needs them.
Copy the closest example and change the settings your observation requires.

To fit the initial eight-model grid with white light and high resolution, use:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization run my_transit.yaml \
  --workspace analysis/my_transit_grid --execution batched --method stacking
```

The [combining models tutorial](../docs/tutorials/combining_models.md) explains the comparison.
The [model-grid reference](../docs/reference/model_marginalization.md) describes supported models, caching, weights, and diagnostic checks.

For a complete analysis, follow the [SOSS case study](../docs/case_studies/transmission_soss.md).
The tutorials cover [limb darkening](../docs/tutorials/limb_darkening.md) and [eclipses, phase curves, and stellar spots](../docs/tutorials/eclipses_and_phase_curves.md).
The [configuration reference](../docs/configuration.md) lists the available settings.

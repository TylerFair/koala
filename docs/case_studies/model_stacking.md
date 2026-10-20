# Stacking limb-darkening models

Here we'll compare limb-darkening treatments for HAT-P-18 b observed with NIRSpec/G395M.
Each treatment gets a complete fit, and we'll combine the spectra using their predictive weights.
The HAT-P-18 extraction is not bundled, so you'll need to supply it.
For a grid using the bundled SOSS data, start with [combining models](../tutorials/combining_models.md).

## Define the candidates

First, set your extraction and atmosphere-cache paths in `examples/nirspec_g395m.yaml`.
The {download}`matrix YAML <../../examples/limb_darkening_stack.yaml>` supplies overrides to this base configuration.
For example, these entries compare fixed power-2 coefficients with uniform quadratic coefficients:

```yaml
variants:
  - name: fixed_power2
    overrides:
      flags:
        ld_profile: power2
        ld_prior: fixed
  - name: uniform_quadratic
    overrides:
      flags:
        ld_profile: quadratic
        ld_prior: uniform
```

Keep the data, masks, and wavelength grid identical across candidates.
Include models you would be willing to use on their own.
The distributed matrix also includes Sing-calibrated quadratic and stellar-prior power-2 candidates.
Each candidate runs its own white-light fit because limb darkening changes the inferred geometry.

## Fit the matrix

Now prepare and fit the candidates, then combine their spectra:

```bash
python -m tools.stacking.run_matrix examples/nirspec_g395m.yaml \
  examples/limb_darkening_stack.yaml --workspace model_stack
bash model_stack/run_all.sh
python -m tools.stacking.stack_spectra model_stack/analysis.yaml \
  --output model_stack/results/
```

This archived comparison shows the fixed power-2, uniform quadratic, and Sing-calibrated quadratic candidates:

```{figure} ../_static/model_stacking.png
:alt: HAT-P-18 b stacked transmission spectrum and wavelength-dependent model weights

The aligned mixture and its 16th–84th percentile interval, with channel weights below.
This archived figure has three candidates.
The distributed matrix adds the stellar-prior power-2 fit.
```

## Interpret the mixture

For each channel, leave-one-out predictive densities determine nonnegative weights.
The output mixes posterior depth draws using those weights.
This workflow removes each model's achromatic depth offset by default.
It also saves the absolute, unaligned mixture.

A small weight means that the candidate adds little to the predictive mixture in that channel.
The result depends on the candidate set.
Pointwise leave-one-out calculations assume conditionally independent integrations, so inspect correlated residuals before interpreting the weights.

Read `*_stacked.csv` for `wavelength`, `depth`, `depth_err_lo`, `depth_err_hi`, and each candidate's `*_weight`.
Depths are fractional, so multiply by $10^6$ for ppm.
Use `absolute_depth` and its errors when you want the unaligned mixture.
The separate `koala.marginalization` workflow retains absolute offsets by default (see the [model-grid reference](../reference/model_marginalization.md)).

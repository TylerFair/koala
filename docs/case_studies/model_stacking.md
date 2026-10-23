# Combining models

The spectrum you get depends on the trend and limb darkening you choose, and often the data don't clearly prefer one choice.
Rather than picking one, we can fit several and combine their spectra, giving each model a weight according to how well it predicts the data.
koala does this with [Bayesian stacking](https://arxiv.org/abs/1704.02030) (Yao et al. 2018), using leave-one-out predictive scores for every integration.
In this case study we'll do it for the WASP-39 b SOSS observation from [your first fit](../quickstart.md).

## Fitting the models

By default, koala combines two trends (linear and quadratic), two limb-darkening laws (quadratic and power-2) and two treatments of the coefficients (fixed and uniform), which makes eight models.
Run them all, and combine them, with:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization run examples/niriss_soss_order1.yaml \
  --workspace analysis/soss_grid --execution batched --method stacking
```

Each model gets its own white-light fit, since the trend and limb darkening can change the orbit, followed by its own channel fits.
This takes several times longer than a single fit, so it's best run on a GPU or on a cluster node.
If it's interrupted, the same command picks up where it left off.

All the models have to be compared on exactly the same data, so this turns off koala's usual outlier clipping (which would remove different integrations for different models).
If your data need clipping, leave out the bad integrations in the base configuration instead, with `exclude_integrations` or `exclude_times`.

## Reading the result

The combined spectrum is the `*_spectrum.csv` table in `analysis/soss_grid/marginalized/`, with the same columns as the spectrum of a single fit.
`state/posterior.npz` in the same directory has the weights, scores and depth draws of every model, and `state/diagnostics.json` records the checks.

By default every wavelength channel gets its own set of weights, so a model can dominate at some wavelengths and not others.
A small weight just means that model adds little to the combined prediction in that channel.
The weights only rank the models you gave it, though, and can't tell you whether a model you didn't include would have done better.

Before trusting the result, check `state/diagnostics.json`.
The leave-one-out scores are approximated with Pareto-smoothed importance sampling, and if that approximation fails for some integrations, koala refuses to write the final spectrum.
The scores also treat the integrations as independent, so if your residuals show correlated noise, the weights should be read with that in mind.

## An example with HAT-P-18 b

Here's a comparison of limb-darkening treatments for HAT-P-18 b with NIRSpec/G395M (that extraction isn't included with koala):

```{figure} ../_static/model_stacking.png
:alt: The combined HAT-P-18 b spectrum and the weight of each limb-darkening treatment at each wavelength

The combined spectrum and its 16th–84th percentile range (top), and the weight of each model in each channel (bottom).
The three models are fixed power-2 coefficients, uniform quadratic coefficients, and Sing-calibrated quadratic coefficients.
```

This one was made with a slightly different set of tools, which lets you list the models in a YAML file:

```bash
python -m tools.stacking.run_matrix examples/nirspec_g395m.yaml \
  examples/limb_darkening_stack.yaml --workspace model_stack
bash model_stack/run_all.sh
python -m tools.stacking.stack_spectra model_stack/analysis.yaml --output model_stack/results/
```

[`examples/limb_darkening_stack.yaml`](https://github.com/TylerFair/koala/blob/main/examples/limb_darkening_stack.yaml) lists the models as changes to the base configuration, for example:

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

One difference to watch for: this route removes each model's overall depth offset before combining them by default (`*_stacked.csv` has `depth` for that, and `absolute_depth` for the combination without it), while `koala.marginalization` keeps the offsets.

The [model grid reference](../reference/model_grids.md) covers the other weighting methods, custom grids and the command-line options.

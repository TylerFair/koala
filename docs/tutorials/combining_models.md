# Combining models

A transmission spectrum can depend on the trend and limb darkening we choose.
Instead of picking one fit immediately, we can fit several plausible models and compare how well they predict the data.
koala can then combine their depth posteriors using those predictive weights.
Here we'll set up the comparison using the bundled SOSS observation.
The full grid takes more work than a single fit.

## Choose the candidates

First, consider the choices you want to compare.
The default grid combines two trends, two limb-darkening laws, and two priors:

- Linear and quadratic trends.
- Quadratic and power-2 limb darkening.
- Fixed coefficients and uniform coefficient priors.

This gives eight models.
Each gets its own white-light fit, since changing the model can also change the inferred geometry.
Each then fits the high-resolution channels with that geometry held fixed.
The grid skips the optional low-resolution stage.

## Fit and combine

From the repository root, run:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization run examples/niriss_soss_order1.yaml \
  --workspace analysis/soss_grid --execution batched --method stacking
```

The command prepares the candidates, fits them, and calculates pointwise leave-one-out predictive scores with PSIS-LOO.
Stacking chooses nonnegative weights that maximize the predictive density of their mixture.
The resulting depth distribution includes draws from the candidate posteriors.
It retains their absolute depth offsets.

All candidates must score the same times, fluxes, uncertainties, and wavelengths.
The grid keeps the base configuration's common exclusions and disables model-dependent sigma clipping.
If you need to exclude bad data, choose a common mask before running the comparison.

## Read the mixture

Now open `marginalized/spectrum.csv` inside the workspace.
It contains the combined depths and intervals.
The accompanying `posterior.npz` saves model names, weights, scores, Pareto diagnostics, and depth draws.
Check `diagnostics.json` before interpreting the spectrum.
High or nonfinite Pareto values block the final result because they make the predictive comparison unreliable.

A small model weight means that the candidate adds little to the predictive mixture in that channel.
The weights are conditional on the models you supplied.
They cannot tell you whether an omitted model would have been better.

By default, each wavelength channel has its own weights.
Rows of the resulting draws are channel marginals, so don't treat a row as one coherent draw of the entire spectrum.
For a shared candidate choice and posterior row across wavelengths, use `--pooling global` with stacking or pseudo-BMA+.
The fits still condition on each candidate's white-light geometry.

## Try another comparison

You can rescore an accepted grid without fitting it again:

```bash
JAX_ENABLE_X64=1 python -m koala.marginalization score \
  analysis/soss_grid/analysis_batched.yaml --method pseudo_bma_plus
```

Pseudo-BMA+ averages predictive weights over a Bayesian bootstrap.
The [model-grid reference](../reference/model_marginalization.md) describes this and the hierarchical alternatives, supported models, and resume behavior.
The current scores assume conditionally independent integrations.
Correlated residuals need attention before interpreting them.

The [model-stacking case study](../case_studies/model_stacking.md) uses a separate matrix workflow to compare limb-darkening choices for HAT-P-18 b.
That workflow removes achromatic offsets by default and also saves the unaligned mixture.
Keep this difference in mind when comparing its output with the absolute-depth mixture here.

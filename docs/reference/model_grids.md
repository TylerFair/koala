# Model grids

`python -m koala.marginalization` fits a grid of models to one observation and combines their spectra by their predictive weights.
The [combining models case study](../case_studies/model_stacking.md) goes through an example, and this page has the details.

## Commands

```bash
# prepare, fit and combine in one go
JAX_ENABLE_X64=1 python -m koala.marginalization run visit.yaml \
  --workspace analysis/visit_grid --execution batched --method stacking

# or one step at a time
python -m koala.marginalization prepare visit.yaml --workspace analysis/visit_grid
JAX_ENABLE_X64=1 python -m koala.marginalization fit analysis/visit_grid/analysis.yaml --execution batched
JAX_ENABLE_X64=1 python -m koala.marginalization score analysis/visit_grid/analysis_batched.yaml --method pseudo_bma_plus
```

`visit.yaml` is an ordinary koala configuration that already works as a single fit.
Running `fit` again finishes an interrupted grid, and `score` recombines a finished grid with a different method without fitting anything again.

## Choosing the models

The default grid is linear and quadratic trends, quadratic and power-2 limb darkening, and fixed and uniform coefficients, eight models in all.
`--trends` and `--treatments` change the axes:

```bash
python -m koala.marginalization prepare visit.yaml --workspace analysis/grid36 \
  --trends constant,linear,quadratic,cubic,quartic,explinear \
  --treatments fixed,uniform,gaussian
```

For models that don't fit on a simple grid, list them in a YAML file as changes to the base configuration:

```yaml
dataset: visit
variants:
  - name: informative_power2_linear
    overrides:
      flags:
        detrending_type: linear
        ld_profile: power2
        ld_prior: stellarprior
```

`--execution batched` fits several models at once and is the fastest choice, but it only handles one unspotted planet described by its transit duration, with a linear, quadratic or step (`linear_discontinuity`) trend and fixed or uniform limb darkening.
Anything else uses `--execution sequential`, which fits the models one after another with the ordinary koala pipeline.
On a CPU, the grid uses four cores unless you set `--cpu-cores` (`0` uses all of them).

## Weighting methods

| `--method` | |
|---|---|
| `stacking` | the weights that best predict held-out integrations (Yao et al. 2018) |
| `pseudo_bma_plus` | weights from each model's predictive score, averaged over a Bayesian bootstrap (Yao et al. 2018) |
| `hierarchical` | channel weights partially pooled across wavelength and sampled (Yao et al. 2022) |
| `hierarchical_map` | a faster version of `hierarchical`, with the pooling strength fixed |

By default, every wavelength channel gets its own weights (`--pooling channel`), so each channel of the combined spectrum is a mixture of the models' posteriors for that channel.
That means a single draw from the result isn't one consistent spectrum across wavelength.
`--pooling global` uses one set of weights for the whole spectrum instead, with `stacking` or `pseudo_bma_plus`.

The `hierarchical` prior treats the first model in the list as the reference, which with three or more models makes the prior weights slightly unequal (for three equally good models, about 0.30, 0.35 and 0.35).
If you use it, check how much the result depends on the order of the models.

## Checks

All the models are scored on exactly the same integrations, so the grid turns off sigma clipping and keeps only the exclusions in the base configuration.
The leave-one-out scores come from Pareto-smoothed importance sampling, and if the Pareto $\hat{k}$ of any integration is above $\min(0.7, 1 - 1/\log_{10} S)$, with $S$ posterior draws, koala doesn't write the combined spectrum.
`--allow-unreliable` writes it anyway, marked as exploratory.
Every model also has to pass the usual convergence checks.
The scores treat the integrations as independent, so correlated residuals make the weights less reliable.

## Outputs

| File | |
|---|---|
| `marginalized/spectrum.csv` | the combined depths and their intervals |
| `marginalized/posterior.npz` | model names, weights, scores, Pareto diagnostics, each model's depths, and the combined draws |
| `marginalized/diagnostics.json` | the checks, and how the result was put together |
| `analysis.yaml`, `analysis_batched.yaml` | the grid, for `fit` and `score` |

The depths in `spectrum.csv` keep each model's absolute depth, offsets and all.

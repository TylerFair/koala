# Marginalize over model choices

A transmission spectrum can depend on reasonable analysis choices: limb-darkening prior, baseline trend, or treatment of a detector event. Model stacking carries disagreement between accepted fits into the reported spectrum instead of silently choosing one.

This tutorial uses three HAT-P-18 b NIRSpec/G395M fits. They share the same data, linear trend, wavelength grid, and sampler policy; only limb darkening changes:

- fixed power-2;
- broad uniform quadratic;
- Sing-calibrated quadratic.

Every candidate must be a scientifically defensible, converged fit. Stacking does not repair a bad model or a failed chain.

## 1. Define the candidates

Start with one working dataset configuration, then describe only the differences in a matrix:

```yaml
dataset: hatp18_nrs1_g395m
analysis_stage: all

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

  - name: sing_quadratic
    overrides:
      stellar:
        ld_mu_min: 0.2
      flags:
        ld_profile: quadratic
        ld_prior: sing
```

The complete {download}`example matrix <../../examples/limb_darkening_stack.yaml>` also includes a power-2 `stellarprior` reference fit. Remove that row to reproduce the three-model figure below.

Change one scientific axis at a time. If the matrix changes limb darkening, keep the trend, masks, wavelength bins, and sampling requirements fixed. Each variant needs its own white-light fit because limb darkening can change the inferred geometry.

## 2. Create and run the fits

The matrix helper creates one complete YAML per candidate, a portable analysis manifest, and a shell script that runs the fits:

```bash
python tools/stacking/run_matrix.py \
  examples/nirspec_g395m.yaml \
  examples/limb_darkening_stack.yaml \
  --workspace model_stack
```

The base configuration supplies the data paths and target parameters. Edit it and confirm that one ordinary fit can read your data before creating the matrix. The generated candidate configurations write to `model_stack/fits/`; saved likelihood inputs go to `model_stack/stage_inputs/`.

Run every candidate:

```bash
bash model_stack/run_all.sh
```

The script uses `python` from the active environment. To select another interpreter, set `PYTHON`:

```bash
PYTHON=/path/to/python bash model_stack/run_all.sh
```

The script contains absolute executable and configuration paths, so it can be launched from another working directory. After the fits finish, `analysis.yaml` resolves fit products relative to its own location; the completed workspace can then be moved and stacked elsewhere.

Before combining anything, confirm that every candidate has:

- the same cadences and wavelength grid;
- an accepted sampler diagnostic for every included channel;
- pointwise likelihood information from the same predictive unit;
- no unexplained white-light residual structure.

## 3. Compute the predictive stack

Once the fits and saved stage inputs are complete:

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
python tools/stacking/stack_spectra.py model_stack/analysis.yaml \
  --stage high_resolution \
  --output model_stack/results \
  --label hatp18_nrs1_g395m \
  --n-out 20000
```

For each wavelength channel, PSIS-LOO estimates how well each model predicts a held-out cadence. Stacking chooses non-negative weights that sum to one and maximize the predictive density of their mixture. The output spectrum is formed by mixing posterior depth draws with those channel-specific weights.

This is predictive model averaging. Bayesian evidence asks a different question and depends on the complete parameter priors; BIC and Laplace evidence are approximations to evidence, not substitutes for PSIS-LOO stacking. Do not compare their numerical weights as though they were the same quantity.

The arrays file also records pseudo-BMA+ weights as a secondary comparison. The publication figure uses stacking weights so that its meaning stays unambiguous.

## 4. Separate gray offsets from spectral shape

Changing limb darkening can shift an entire spectrum through the fitted reference radius. By default the stacker estimates one achromatic depth offset per model, aligns the candidates, mixes their posterior draws, and restores the model-weighted mean level. This prevents a nearly constant radius shift from looking like wavelength-dependent atmospheric uncertainty.

The output also retains the absolute, unaligned candidates and stack. Use those columns when the absolute reference-radius level matters, and always state whether the published spectrum is aligned.

## 5. Read the result

```{image} ../_static/model_stacking_hatp18_fitted_sing_uplus_v2.png
:alt: HAT-P-18 b NIRSpec G395M stacked transmission spectrum and model weights
:width: 900px
:align: center
```

Panel a shows the offset-aligned stacked spectrum with its 16th--84th percentile interval. The faded lines are candidate medians. Their separation shows which wavelength regions depend on the limb-darkening treatment.

Panel b shows the per-channel depth precision (half the 16th--84th percentile width, in ppm) of each candidate model and of the stack. Where the stack tracks the most precise candidate, the mixture is not inflating the error; where it sits above every candidate, disagreement between the models is being propagated into the stacked uncertainty.

Panel c shows the stacking weights. A zero weight does not mean that a fit failed; it means that model did not improve the optimal predictive mixture in that channel. Rapid channel-to-channel changes deserve scrutiny because cadence-level LOO assumes the residuals are conditionally independent.

:::{dropdown} Details of the illustrated run
All 428,064 pointwise diagnostics have Pareto $\hat k<0.7$ and the global maximum is 0.537. Mean weights are 0.673 for fixed power-2, 0.120 for broad-uniform quadratic, and 0.208 for Sing quadratic. After gray-offset alignment, the median disagreement ratio is 1.010 and its largest value is 2.084. The stack differs from the staged power-2 `stellarprior` production spectrum by $-9.8\pm17.7$ ppm on average, with a median uncertainty ratio of 0.991.

The broad-uniform fit uses $u_+\in[-1,2]$ and $u_-\in[-2,2]$, rather than the older independent $u_1,u_2\in[0,1]$ box. The Sing calibration fits broad $u_+$ and $u_-$ coordinates at low resolution, transforms them to $(l,\delta)$, and pools ESS-inflated channel uncertainties. It measured $\Delta l=+0.00725\pm0.01745$ and $\Delta\delta=+0.00349\pm0.00366$; all 11 calibration channels had bulk ESS above 235 and zero divergences, so the fitted correction was used instead of the fallback.
:::

The companion diagnostics figure written beside the numerical products compares absolute and aligned intervals, plots maximum Pareto $\hat k$, and reports disagreement ratios. Treat a channel above $\hat k=0.7$ as a prompt for exact refits, moment matching, or a better grouped predictive unit.

## Outputs worth keeping

| Product | Use |
|---|---|
| `*_stacked.csv` | Portable aligned and absolute spectrum |
| `*_stacking.png` | Spectrum and channel weights |
| `*_stacking_diagnostics.png` | Offset, Pareto-$\hat k$, and disagreement checks |
| `*_diagnostics.json` | Machine-readable offsets and weight summaries |
| `*_stacking_arrays.npz` | Mixture draws, weights, and LOO results |
| `*_pointwise_loglik.npz` | Reusable pointwise likelihood archive for each model |

Pointwise LOO is only appropriate when a cadence is the relevant predictive unit and residual dependence has been modeled adequately. For time-correlated residuals, use blocked LOO or leave-future-out prediction before interpreting the weights. The result is also conditional on the candidate set: include the plausible alternatives you would have been willing to publish individually.

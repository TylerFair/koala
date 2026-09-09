# Marginalize over model choices

A transmission spectrum can depend on reasonable analysis choices:
limb-darkening prior, baseline trend, or the treatment of a detector event.
Model stacking carries the disagreement between accepted fits into the
reported spectrum instead of silently choosing one.

This guide uses three HAT-P-18 b NIRSpec/G395M fits that share the data,
trend, wavelength grid, and sampler policy and differ only in limb darkening:
fixed power-2, broad uniform quadratic, and Sing-calibrated quadratic.

## 1. Define the candidates

Describe the differences from one working configuration in a matrix:

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

The complete {download}`example matrix <../../examples/limb_darkening_stack.yaml>`
also includes a power-2 `stellarprior` fit. Change one scientific axis at a
time and keep everything else fixed.

## 2. Create and run the fits

```bash
python tools/stacking/run_matrix.py \
  examples/nirspec_g395m.yaml \
  examples/limb_darkening_stack.yaml \
  --workspace model_stack
bash model_stack/run_all.sh
```

The helper writes one complete YAML per candidate under `model_stack/fits/`,
saves the likelihood inputs under `model_stack/stage_inputs/`, and generates
the run script (set `PYTHON=/path/to/python` to choose the interpreter).

## 3. Stack the spectra

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
python tools/stacking/stack_spectra.py model_stack/analysis.yaml \
  --stage high_resolution \
  --output model_stack/results \
  --label hatp18_nrs1_g395m \
  --n-out 20000
```

For each wavelength channel, PSIS-LOO estimates how well each model predicts
a held-out cadence, and stacking chooses non-negative weights that maximize
the predictive density of the mixture. The stacked spectrum mixes the
posterior depth draws with those channel-specific weights. By default one
achromatic depth offset per model is removed before mixing, so a gray radius
shift is not mistaken for wavelength-dependent uncertainty; the absolute,
unaligned columns are also written.

## 4. Read the result

```{image} ../_static/model_stacking_hatp18_fitted_sing_uplus_v2.png
:alt: HAT-P-18 b NIRSpec G395M stacked transmission spectrum and model weights
:width: 900px
:align: center
```

Panel a shows the aligned stacked spectrum with its 16th--84th percentile
interval over the candidate medians. Panel b compares the per-channel depth
precision of each candidate with the stack: where the stack sits above every
candidate, model disagreement is being propagated into the uncertainty. Panel
c shows the stacking weights; a zero weight means that model did not improve
the predictive mixture in that channel, not that its fit failed.

Pointwise LOO assumes the residuals are conditionally independent per cadence
and the result is conditional on the candidate set, so include every
alternative you would have been willing to publish on its own.

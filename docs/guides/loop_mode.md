# Limb-darkening loop mode

Loop mode is a developer tool for comparing several limb-darkening priors
while reusing compiled spectroscopic runners. It is not needed for an ordinary
fit or for the model-stacking tutorial.

The working command consumes previously prepared spectroscopic stage dumps:

```bash
python tools/loop_fit.py \
  --output-dir /path/to/my_loop \
  --stage uniform_quadratic=/path/to/inputs/uniform.pkl \
  --stage sing_quadratic=/path/to/inputs/sing.pkl
```

Each item is `VARIANT=/absolute/path/to/stage.pkl`. Uniform quadratic must
precede Sing when its low-resolution posterior supplies the Sing offset
calibration. Each variant writes a `spectrum.csv`, posterior samples, and a
`summary.json`; the root directory receives `loop_summary.json`.

The `--config` interface can generate a six-variant plan with `--plan-only`,
but direct execution from that plan is intentionally disabled because the
main fitter does not yet expose the required in-process staged API. Use the
regular CLI for standalone fits:

```bash
python fit_jwst.py -c config.yaml
```

Loop output directories are expected to be new. The tool refuses to replace
generated plan files, which keeps comparisons replayable.

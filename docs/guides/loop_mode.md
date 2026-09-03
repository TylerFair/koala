# Limb-darkening loop mode

Loop mode fits several limb-darkening assumptions from replayable stage-input
dumps in one Python process. Models with the same limb-darkening law share one
callable and one resident sampler runner; the prior centres, scales, bounds,
and coordinate map are array inputs to that callable.

Pass each stage as `VARIANT=PATH`. Uniform quadratic must precede Sing when the
Sing centres are calibrated from its low-resolution posterior.

```bash
python tools/loop_fit.py \
  --output-dir /scratch/my_loop \
  --stage uniform_quadratic=/scratch/inputs/low_resolution_inputs.pkl \
  --stage sing_quadratic=/scratch/inputs/sing_low_resolution_inputs.pkl
```

Each variant and stage receives a `spectrum.csv` and `summary.json`. The root
`loop_summary.json` records execution and compilation events. Existing files
are never replaced, so use a new output directory for each invocation.

The current driver consumes spectroscopic stage dumps. White-light fits and
their geometry handoffs must be prepared before creating those dumps.

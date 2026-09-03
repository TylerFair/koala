# Limb-darkening loop mode

Loop mode fits several limb-darkening assumptions from replayable stage-input
dumps in one Python process. Models with the same limb-darkening law share one
callable and one resident sampler runner; the prior centres, scales, bounds,
and coordinate map are array inputs to that callable.

Bounded coordinate variants are pulled back from per-variant lower and upper
bounds through a smooth sigmoid map. This keeps every proposal in the valid
coordinate domain while preserving the standalone prior, including its change
of variables. Spectroscopic fits use the normal production quality gate and
selectively retry failing channels with HMC and then adaptive NUTS; the spectrum
records the sampler retained for each channel.

The primary NUTS and HMC runners use a shared resident channel width. Adaptive
fallback lanes are processed in fixed groups of four, so one small joint-NUTS
program can be reused when different variants fail the gate in different
numbers of channels. Padding lanes are discarded from the saved posterior.

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

The current command-line driver consumes spectroscopic stage dumps. A reusable
white-light runner exists, but constructing its model arguments and feeding its
per-variant geometry into the stage dumps is not yet connected to the command
line. White-light fits and their geometry handoffs must therefore be prepared
before creating those dumps.

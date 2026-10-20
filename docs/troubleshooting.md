# Troubleshooting

If a fit looks wrong, start with the data and initial model.
Then inspect the residuals and sampling diagnostics.
For the bundled SOSS example, these files are a useful starting point:

```bash
ls results/WASP-39_SOSS_ORDER1/00_*check.png
ls results/WASP-39_SOSS_ORDER1/*whitelight_timeseries.csv
```

Check that the event falls within the observation and that the initial curve overlaps it.
An incorrect time origin can move an eclipse entirely out of the data.
A more flexible trend won't fix that.

## Sampling diagnostics

A good-looking light curve does not tell us whether the posterior was sampled well.
Open the diagnostic records alongside the residual and noise-binning plots:

```python
import json
from pathlib import Path

out = Path("results/WASP-39_SOSS_ORDER1")
for path in sorted((out / "chunks").glob("*.diagnostics.json")):
    print(path.name)
    print(json.dumps(json.loads(path.read_text()), indent=2))
```

Effective sample size (ESS) measures how much independent information remains in correlated draws.
The default uses four independent chains with 1,000 retained draws each.
Multi-chain checks require rank-normalized split R-hat below 1.01 and at least 400 combined bulk and tail effective samples for each checked parameter.
That is 400 across the chains.
Divergences indicate that a sampling trajectory could not follow the posterior accurately.
The default permits none.

The white-light check includes every free parameter.
Spectroscopic checks include transit depth, or the free emission parameters when the radius is fixed.
With free channel times, they also include the relative epoch.
Limb-darkening, trend, and jitter diagnostics are recorded under `nuisance_*` without triggering continuation.
Inspect those entries if you plan to use those posteriors themselves.
A coefficient close to a prior boundary can mix slowly even when the depth is well measured.

Read the recorded outcome and sampler for every channel, including retries.
Spectroscopic limits can be changed with `spectro_min_depth_ess` and `spectro_max_divergences`.
Retries may use the alternate HMC or NUTS sampler.
Lowering a threshold does not resolve an unconstrained eclipse time or a poor model.

Look for broad or multimodal posteriors, parameters pressed against prior bounds, and isolated large spectral uncertainties.
Fixed parameters don't need an ESS.
A saved checkpoint alone doesn't establish convergence.

## Ask for help

If you can't resolve a problem, [open an issue](https://github.com/TylerFair/koala/issues).
Include the YAML, software versions, relevant log excerpt, and a small input you can share that reproduces it.

## Reference

| Symptom | Likely cause | What to try |
|---|---|---|
| Eclipse absent from the model | Incorrect epoch or displaced start | Check BMJD_TDB times and the initial-model plots. See [eclipse timing](case_studies/rocky_eclipse.md#when-the-eclipse-is-not-recovered). |
| Ephemeris error mentions 0.5 d | YAML used BJD − 2400000 | Convert to BJD_TDB − 2400000.5. Configuration epochs are not converted automatically. |
| Shallow eclipse timing wanders | Data do not constrain the epoch | Fix `eclipse_time` and geometry from transits, then fit depth, baseline, and noise. |
| Emission depth resembles a transit depth | Reading `depths` or `depth00` | Read `eclipse_depth_ppm` in the emission product. |
| Depth changes with ramp model | Trend absorbs part of the eclipse | Compare simpler trends and, if appropriate, trim settling with `exclude_times`. |
| Structured residuals | Missing baseline, spot, or extraction feature | Compare justified [trend models](tutorials/systematics.md) with identical masks and priors. |
| GPU allocation runs on CPU | JAX cannot see the device | Check `jax.devices()` inside the allocation and the [JAX installation](install.md#nvidia-gpus). |
| GPU runs out of memory | Too many channels fitted together | Lower `spectro_chunk_size`. Joint geometry requires all channels in one chunk. |
| First batch takes much longer | JAX compilation | Let the first shape compile and use a persistent compilation cache. |
| ExoTiC-LD cannot download files | Compute node has no network | Populate `ld_data_path` on a connected machine and copy the cache. |
| YAML key is ignored | Misspelled or removed option | Read the warning and the [configuration tables](configuration.md#reference). |
| Interrupted run recomputes a batch | Data or model changed | Resume with the original YAML and inputs, or use a new directory for a changed analysis. |

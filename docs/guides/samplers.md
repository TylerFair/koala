# Samplers and convergence

You normally do not need to configure a sampler. White light uses NumPyro
NUTS, and wavelength channels use independent NUTS with a local Laplace mass
matrix:

```yaml
flags:
  spectro_sampler: independent_nuts
```

The setting above is the default and may be omitted.

## What happens during a run

Each wavelength channel has its own chain and adaptation state. Channels are
evaluated together in GPU batches, but one difficult channel does not change
the posterior of its neighbours. A completed batch is checkpointed before the
next begins.

The fitter checks transit-depth effective sample size and divergences. If the
primary exact sampler fails that gate, it retries the affected work with the
alternate exact HMC/NUTS path and records the sampler retained for each
channel. Messages containing `PASSED ... GATE`, `depth ESS below`, or
`divergences` explain the decision; the JSON files in `chunks/` preserve it.

## When to change the sampler

| Value | Use |
|---|---|
| `independent_nuts` | Default exact sampler. Start here. |
| `independent_hmc` | Exact fixed-work alternative for comparison or a known difficult workload. |
| `joint_nuts` | Compatibility backend that adapts a whole channel batch together. |

Changing the primary sampler is rarely the first response to a poor channel.
Inspect the light curve, uncertainties, limb-darkening boundary, and trend
choice first. The automatic exact-sampler retry already handles many numerical
failures.

## Reading the log

The first batch of a new static shape includes JAX compilation and is usually
the slowest. Later equal-width batches reuse the compiled runner. A smaller
final batch can compile once more because its array shape differs.

`COMPUTING` means that no compatible checkpoint was found. A resumed run loads
only checkpoints whose fingerprint matches the current data, model, and
sampler settings.

Do not judge convergence from runtime or a `SAVED checkpoint` message alone.
Use the diagnostic JSON and check:

- the depth ESS passed the configured gate;
- the accepted chain has no more than the permitted divergences;
- the sampler recorded for the channel matches the successful retry, if one
  occurred;
- the spectrum does not contain isolated, very large depth uncertainties.

## Reproducibility

Set a seed for a published analysis:

```yaml
flags:
  random_seed: 555
```

`FIT_JWST_SEED` overrides the YAML seed. Channel keys are derived
deterministically from the master seed and channel range, so resuming does not
change later chains. Archive the YAML, software versions, diagnostics, and GPU
model with the spectrum.

If both exact sampler routes fail, investigate the affected data and model.
Lowering the convergence gate only hides the failure.

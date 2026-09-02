# Uniform limb-darkening latent reparameterization test

Test an alternative computational parameterization that preserves the current
physical quadratic limb-darkening prior exactly:

```text
z_l, z_delta ~ Normal(0, 1)
l = Phi(z_l)
delta = (1 - l) * (2 * Phi(z_delta) - 1) / 4
```

This must be compared against the existing prior
`l ~ Uniform(0, 1)` and
`delta | l ~ Uniform(-(1-l)/4, +(1-l)/4)`. It is not a Kipping-q prior and
must not change the induced distribution over physical limb-darkening
coefficients.

For identical SOSS channels and seeds, compare:

- recovered depth and limb-darkening posterior quantiles;
- prior-predictive distributions in `(l, delta, c1, c2)`;
- MAP Hessian condition number and eigenvalue repair;
- divergences, maximum/mean leapfrog steps, tree-depth saturation, and ESS;
- compilation, warmup, and retained-sampling wall time;
- cadence-level PSIS-LOO Pareto-k values for the eventual LD-treatment stack.

Start with a small representative channel set, including at least one weakly
identified limb-darkening channel, before running the full SOSS spectrum.

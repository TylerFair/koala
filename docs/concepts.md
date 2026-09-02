# How the fit works

{{ project }} models an extracted JWST time series as a transit signal plus an additive baseline model. The basic spectroscopic model for wavelength channel $j$ is

$$
f_j(t) = T(t; r_j, \boldsymbol{g}, \boldsymbol{u}_j)
       + S_j(t; \boldsymbol{\beta}_j) + \epsilon_j(t),
$$

where $T$ is the transit light curve, $r_j=R_{p,j}/R_\star$ is the channel radius ratio, $\boldsymbol{g}$ is the shared orbital geometry, $\boldsymbol{u}_j$ describes limb darkening, and $S_j$ is the channel systematics trend. The likelihood uses the uncertainty supplied for every integration. An additional channel jitter is added in quadrature.

```yaml
flags:
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  whitelight_geometry_estimator: posterior_median
```

## White light first

The first fit sums the spectral time series into a white-light curve. This curve has much higher signal-to-noise than an individual wavelength channel. It therefore constrains transit time, duration or scaled semimajor axis, impact parameter, and broadband radius ratio more efficiently.

The white-light model also measures the shape of a visit-level systematics component. For example, an `explinear` white-light fit samples both the exponential amplitude and its decay time. The white-light posterior is saved before spectroscopic fitting starts.

Its median geometry is passed to both spectroscopic stages. The production setting is:

```yaml
flags:
  whitelight_geometry_estimator: posterior_median
```

The handoff fixes all channel fits to one coherent geometry. It prevents low-signal channels from trading wavelength-dependent depth against impact parameter or duration. It also avoids fitting hundreds of copies of parameters that describe the same orbit.

The consequence is important: the per-channel chains condition on the selected geometry. They do not marginalize independently over white-light geometry uncertainty. The selected values are recorded in `*_whitelight_geometry_handoff.json`.

The ordinary white-light summary remains available in `*_whitelight_bestfit_params.csv`.

## Low and high spectral resolution

After white light, {{ project }} can run a low-resolution stage and a high-resolution stage. The low-resolution stage is requested with `need_lowres: true`. It is useful when limb-darkening or trend coefficients will be interpolated to the finer grid.

It is also required for the fitted shared offset in the Sing limb-darkening prescription. The high-resolution stage fits the requested `resolution.high` grid. A numerical value means constant resolving power.

`native` retains the input channelization. `reference` uses the configured reference wavelength grid. Each stage writes its own spectrum, detailed channel table, diagnostic plots, and checkpoints.

## Channels, chunks, and lanes

A channel is one wavelength-bin light curve. A chunk is a contiguous or planned group of channels sent through one compiled sampler call. The word *lane* describes one channel position inside that vectorized call.

With `vmap_chunk: 40`, at most 40 independent channel chains reside on the GPU together. They use the same compiled program but retain separate data, priors, states, and random keys. The chains are statistically independent conditional on the shared white-light geometry.

Chunking controls device memory and compilation reuse. It does not average channels or change the spectral grid. An R=100 fit with 137 channels and width 40 runs widths 40, 40, 40, and 17.

The equal-width calls reuse the first compiled executable. Each completed chunk is checkpointed immediately.

## Limb darkening at JWST precision

Limb darkening changes ingress, egress, and the curved bottom of a transit. Ground-based broad-band photometry often has insufficient precision to distinguish small errors in that shape from noise. JWST spectroscopic light curves can resolve the mismatch channel by channel.

An incorrect limb profile can therefore move the inferred radius ratio in a wavelength-dependent way. The effect is strongest where the stellar intensity profile changes rapidly with wavelength and for non-central transits. `stellar-informed` does not mean fixed.

{{ project }} asks ExoTiC-LD for Stagger-grid stellar intensities at the target wavelength bins. It fits the requested power-2 coefficients to those intensities. It repeats the calculation across the configured uncertainties in effective temperature, surface gravity, and metallicity.

The resulting coefficient scatter supplies the Gaussian prior width. `ld_prior_min_sigma` imposes a numerical floor so a nominally tiny grid scatter does not become an effectively fixed coefficient. The data can move the coefficients within this propagated prior.

See [Limb darkening](guides/limb_darkening.md) for the laws and alternatives.

## Local Laplace metrics

Transit depth, limb darkening, jitter, and trend coefficients can have different numerical scales and can be correlated. An unscaled sampler explores such a posterior inefficiently. Before sampling a channel, {{ project }} locates a local posterior mode and evaluates its curvature.

In plain terms, it measures which parameter combinations are narrow, broad, or tilted near the best-fitting point. If $H_j$ is the Hessian of the negative log posterior for channel $j$, the local covariance approximation is

$$
\boldsymbol{\Sigma}_j \approx H_j^{-1}.
$$

The sampler uses this inverse-Hessian information as its mass metric. It does not replace the posterior with a Gaussian. NUTS or HMC still evaluates the exact configured model and likelihood at every step.

The metric only changes how proposals move through parameter space. Finite-difference Hessians are the production default because they are robust for the supported light-curve kernels. The metric is computed independently for every wavelength channel.

This is why the method is called Laplace-metric independent NUTS or HMC.

## Quality gates

A chain can finish without producing a trustworthy posterior. Two checks guard the spectroscopic result. The effective sample size, or ESS, estimates how many independent draws the correlated chain contains.

A divergence marks a numerical integration failure in Hamiltonian dynamics. The production gate requires depth ESS of at least 400 and zero divergences.

```yaml
flags:
  spectro_min_depth_ess: 400
  spectro_max_divergences: 0
```

When a chunk fails, {{ project }} does not silently write the poor chain as the final posterior. It retries the affected work with the alternate exact sampler. Independent NUTS swaps to Laplace-metric HMC with eight integration steps.

HMC swaps back to NUTS. The alternate trajectory can avoid a failure caused by pathological tree growth or by a fixed-step trajectory that is too rigid. The log records the original gate result, the swap, and the replacement diagnostics.

Checkpoint diagnostics preserve the same information in JSON. If both methods fail repeatedly, the correct response is to inspect the channel data, trend choice, limb prior, and cadence masks. Do not lower the gate merely to obtain a file.

## A practical mental model

Think of the white-light stage as measuring the common transit clock and chord. Think of the low-resolution stage as an optional calibration bridge. Think of each high-resolution channel as a small conditional inference problem.

Chunking places many of those small problems on the accelerator at once. Laplace preconditioning gives each problem its own sensible coordinate scale. The ESS/divergence gate decides whether its numerical posterior is usable.

The final transmission spectrum is the collection of accepted channel-depth posteriors in wavelength order.

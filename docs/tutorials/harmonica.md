# Harmonica limb asymmetry

This tutorial fits the WASP-94 b NIRISS/SOSS order-1 extraction with a first-order Harmonica transmission string. You will infer the odd boundary coefficient with fixed quadratic limb darkening and Laplace-metric independent NUTS. At the end, you will have the total transmission spectrum, correlated evening/morning limb spectra, and posterior transmission-string shapes.

```bash
cp configs_harmonica/WASP-94_soss_order1_config.yaml wasp94_harmonica.yaml
python fit_jwst.py -c wasp94_harmonica.yaml
```

Use these sampler and shape settings:

```yaml
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: delta_r
  harmonica_spectro_odd_frac_sigma: 0.1
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  ld_profile: quadratic
  ld_prior: fixed
```

`delta_r` samples the radius contrast represented by the odd transmission-string coefficient. Order 1 fits `a0` and `a1`; orders 3 and 5 can add `a3` and `a5`. After the fit, inspect `*_limb_spectra.csv`, `*_limb_spectra.png`, `*_transmission_strings.png`, and `*_transmission_string_posterior.png`. The CSV reports area-equivalent evening/leading and morning/trailing depths, endpoint depths, `a0`, active odd coefficients, and their percentile errors. The accompanying `*_limb_posterior_samples.npz` preserves their draw-by-draw covariance.

## Dataset and complete configuration

This example fits WASP-94 b with NIRISS/SOSS order 1. The checked configuration uses $R=20$ for its bridge and $R=50$ for its final asymmetric spectrum.

```yaml
planet:
  name: WASP-94                  # Output prefix.
  period: 3.9502001              # Days.
  duration: 0.189675             # Days.
  t0: 60595.21518                # Same convention as FITS time.
  b: 0.15
  rprs: 0.107
  a_rs: 7.299998525              # Required a/Rstar geometry.
  ecc: 0.0
  omega: 0.0                    # Radians.
stellar:
  feh: 0.37
  teff: 6217
  logg: 4.27
  ld_model: stagger
  ld_data_path: ../exotic_ld_data
instrument: NIRISS/SOSS
order: 1
path: /scratch/midway3/tfairnington/
input_dir: FITS
output_dir: ASYMM_WASP-94_SOSS_ORDER1_FIXEDLD_QUADRATIC_LINEAR
fits_file: WASP-94_box_spectra_fullres.fits
resolution:
  high: 50                       # Final asymmetric spectrum.
  low: 20
  reference_grid: prism_template.csv
flags:
  transit_engine: harmonica
  harmonica_max_order: 1         # a0 plus a1.
  harmonica_spectro_parameterization: delta_r
  harmonica_spectro_fit_jitter: false
  harmonica_spectro_odd_frac_sigma: 0.1
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  vmap_chunk: 4
  detrending_type: linear
  interpolate_trend: false
  interpolate_ld: false
  fix_ld: true                   # Legacy spelling of fixed LD.
  ld_prior: fixed
  need_lowres: false
  ld_profile: quadratic
  mask_start: jnp.min(t)
  mask_end: jnp.min(t) + 0.0208
outlier_clip:
  whitelight_sigma: 5
  spectroscopic_sigma: 5
host_device: gpu
```

The explicit `ld_prior: fixed` line states the same choice as the older `fix_ld: true` switch. The geometry includes `a_rs`, `ecc`, and `omega` because this engine uses the scaled-semimajor-axis parameterization.

## Transmission strings

A circular planet has one radius ratio. Harmonica replaces that circle by a polar boundary $r(\theta)$ in stellar-radius units. For the supported odd-cosine expansion,

$$
r(\theta)=a_0+a_1\cos\theta+a_3\cos3\theta+a_5\cos5\theta.
$$

`harmonica_max_order: 1` retains only $a_0$ and $a_1$. $a_0$ is the mean transmission-string radius. $a_1$ shifts area between the two limbs and changes ingress relative to egress.

The `delta_r` parameterization samples the endpoint radius difference. For order 1, the endpoints are $a_0+a_1$ and $a_0-a_1$, so `endpoint_delta_r` is $2a_1$. For higher supported orders it is $2(a_1+a_3+a_5)$.

Odd modes are used because they encode leading/trailing asymmetry. The configured `harmonica_spectro_odd_frac_sigma` controls the odd-mode prior scale relative to the radius.

## Run

```bash
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
python fit_jwst.py -c wasp94_harmonica.yaml
```

## Log walkthrough

The following lines are representative of a completed Harmonica fit:

```text
Building harmonica whitelight model: detrend='linear', ld='fixed'
               (quadratic), max_order=1, param_method='a_rs'
Using harmonica white-light NUTS settings: dense_mass=True
Running chunked MCMC: ... channels in blocks of 4 (mode=serial)
Checkpoint directory: .../chunks
  chunk 0:4 - gradient diagnostic PASSED
  chunk 0:4 - COMPUTING (4 channels)
  chunk 0:4 - SAVED checkpoint
Saved limb spectra to ..._R50_limb_spectra.csv
Saved joint limb posterior samples to ..._R50_limb_posterior_samples.npz
Saved transmission string plot to ..._transmission_strings.png
Analysis complete!
```

The engine name, LD law, maximum order, and geometry parameterization should match the YAML. The limb CSV and NPZ messages confirm that asymmetric products were written.

## White-light fit

```{image} ../_static/harmonica_wasp94_whitelight.png
:alt: WASP-94 Harmonica white-light fit
:width: 760px
:align: center
```

Ingress and egress carry the asymmetry information. Inspect them separately rather than judging only the flat transit bottom.

## Total transmission spectrum

```{image} ../_static/harmonica_wasp94_spectrum.png
:alt: WASP-94 Harmonica total transmission spectrum
:width: 760px
:align: center
```

The ordinary spectrum retains the fitted model-radius summary. Use the limb products for the representative morning/evening depths.

## Limb spectra

```{image} ../_static/harmonica_wasp94_limb_spectra.png
:alt: WASP-94 representative evening and morning limb spectra
:width: 760px
:align: center
```

Harmonica measures $\theta=0$ along the orbital-velocity direction. The output convention maps $\theta=0$ to the evening/leading hemisphere and $\theta=\pi$ to morning/trailing. The representative depths are half-area equivalents, not the limb-darkened decrement at one exposure.

Define $Q=a_1^2+a_3^2+a_5^2$ and $C=a_1-a_3/3+a_5/5$. Then

$$
D_\mathrm{evening}=a_0^2+Q/2+4a_0C/\pi,
$$

$$
D_\mathrm{morning}=a_0^2+Q/2-4a_0C/\pi,
$$

and $D_\mathrm{total}=(D_\mathrm{evening}+D_\mathrm{morning})/2$. Endpoint depths are reported separately and must not be substituted for these area quantities.

## Read all three spectra

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

out = Path("/scratch/midway3/tfairnington/ASYMM_WASP-94_SOSS_ORDER1_FIXEDLD_QUADRATIC_LINEAR")
total = pd.read_csv(out / "WASP-94_NIRISS_SOSS_order1_R50.csv")
limbs = pd.read_csv(out / "WASP-94_NIRISS_SOSS_order1_R50_limb_spectra.csv")
joint = np.load(out / "WASP-94_NIRISS_SOSS_order1_R50_limb_posterior_samples.npz")
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(limbs.wavelength, limbs.depth_evening_median,
            yerr=[limbs.depth_evening_err_lo, limbs.depth_evening_err_hi],
            fmt="r.", label="evening / leading")
ax.errorbar(limbs.wavelength, limbs.depth_morning_median,
            yerr=[limbs.depth_morning_err_lo, limbs.depth_morning_err_hi],
            fmt="b.", label="morning / trailing")
ax.plot(total.wavelength, total.depth_ppm00, color="0.4", label="model radius")
ax.set(xlabel="Wavelength [micron]", ylabel="Depth [ppm]")
ax.legend()
fig.tight_layout()
plt.show()
print(joint["depth_evening"].shape)  # draw, wavelength
```

Use the NPZ arrays for differences or joint retrievals because the two limb depths are correlated within each draw.

## Output checklist

- `16_*_whitelight_limb_spectra.png`: white-light limb summary.

- `17_*_whitelight_transmission_string.png`: white-light boundary posterior.

- `26_*_R20_limb_spectra.png` or `35_*_R50_limb_spectra.png`: representative limbs.

- `27_*_transmission_strings.png` or `36_*_transmission_strings.png`: median boundaries by wavelength.

- `28_*_transmission_string_posterior.png` or `37_*`: boundary draws for one channel.

- `*_limb_spectra.csv`: marginal summaries and convention metadata.

- `*_limb_posterior_samples.npz`: joint samples in fractional stellar area.

- `*_bestfit_params.csv`: `a1` and any higher active odd coefficient.

## Common problems

**No limb products:** confirm `transit_engine: harmonica` and that odd samples are present. **Nonphysical shapes:** use the configured odd prior and inspect transmission-string plots; higher odd orders add flexibility and require stronger data. **Ingress/egress residuals:** check ephemeris, cadence masks, and limb darkening before attributing them to planetary asymmetry.

**NaN channels:** remove invalid wavelength bins before sampling. **Interrupted run:** repeat the unchanged command; compatible chunk checkpoints load automatically. **Gate failure:** inspect the affected channel, reduce `vmap_chunk`, and allow the exact alternate sampler retry. Do not interpret a failed chain.

**Apparent mirror spectra:** this is expected when a tightly constrained total area combines with a weakly constrained asymmetry; propagate the joint posterior.

## Next steps

Read [How the fit works](../concepts.md) for the shared geometry and quality gate, then use [Limb darkening](../guides/limb_darkening.md) before changing the fixed quadratic coefficients. The [Samplers](../guides/samplers.md) guide covers alternate exact sampling, and [Outputs](../guides/outputs.md) defines every limb-spectrum CSV and NPZ field.

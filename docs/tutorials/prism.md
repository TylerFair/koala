# NIRSpec/PRISM notes

PRISM uses the standard transit workflow, but two features deserve attention: its long time series can make native-channel fits expensive, and early integrations often show a settling ramp.

Start from the compact example:

```bash
cp examples/nirspec_prism.yaml prism.yaml
python fit_jwst.py -c prism.yaml
```

The example selects NRS1, native wavelength bins, and an exponential-plus-linear trend:

```yaml
instrument: NIRSPEC/PRISM
nrs: 1

resolution:
  high: native

flags:
  detrending_type: explinear
  spectro_chunk_size: 40
```

## Choose the wavelength grid

| Setting | When to use it |
|---|---|
| `high: native` | Preserve the extraction grid when channels have adequate signal and compute is available |
| `high: 100` | Start with a smaller, easier spectrum at constant resolving power |
| `high: reference` | Match a supplied comparison or retrieval grid |

Native PRISM channel counts can be large. Reducing `spectro_chunk_size` lowers
GPU memory use without changing the model or wavelength grid. Binning to lower
resolving power changes the scientific product and can improve per-channel
constraints.

## Native-cadence acceleration

Koala automatically uses two complementary long-cadence paths for the default
single-planet, duration-geometry, power-2 model:

- Outside the transit windows, the fixed-timescale exponential-linear trend is
  linear in its three channel coefficients. Koala groups identical reported
  uncertainties and evaluates the same Gaussian likelihood from precomputed
  sufficient statistics. Masks are applied before those statistics are made.
- Inside the windows, Koala evaluates the limb-darkened kernel on 769 nodes
  split at second/third contact, then uses local quintic interpolation in a
  Chebyshev coordinate. Every original cadence retains its own flux prediction,
  uncertainty, and residual.

The interpolation audit covered the full spectroscopic priors for radius ratio
and power-2 coefficients and the HAT-P-65 white-light handoff range. Its worst
flux error was 0.00234 ppm at the fitted impact parameter (below the 0.01 ppm
limit); the corresponding maxima at impact parameters 0.03 and 0.21 were
0.00209 and 0.00217 ppm. Contact nodes and separate stencils prevent
interpolation across the limited-smoothness boundaries.

On a Tesla V100, a native HAT-P-65 value-plus-gradient evaluation for 40 lanes
fell from 9.60 ms to 1.57 ms (6.10x), with compiler temporary storage falling
from 656 MB to 91 MB. A production 1,000-warmup/1,000-draw, 40-lane NUTS check
including compilation fell from 308 s to 167 s. This deliberately used a
coarser 513-node candidate as a conservative posterior test; it had zero
divergences in both cases, and posterior median shifts were at most 0.011
reference sigma.

The complete 40,715-cadence HAT-P-65 NRS1 workflow, including white light, 42
R20 channels, and 369 native channels, fell from 3,407 s to 1,270 s on the same V100 with the
same code (2.7x end to end; the two native spectra agree to 0.03 sigma in
every channel). Its peak resident memory was 5.66 GiB. The R20 result had
zero divergences, and its median depth-posterior shift from the archived direct
fit was 0.0031 reference sigma with a median uncertainty ratio of 1.0003.

The controls are explicit when you need an audit run:

```yaml
flags:
  spectro_cadence_reduction: auto  # or off
  spectro_transit_grid: auto       # or off
  spectro_transit_grid_nodes: 769
```

`auto` activates only above 5,000 cadences and only for a compatible model, so
short SOSS/G395H analyses retain their original arithmetic and outputs.

## Check the ramp

The `explinear` model fits

$$
S(t)=c+v(t-t_\min)+A\exp[-(t-t_\min)/\tau].
$$

White light determines the decay time; each spectral channel gets its own ramp amplitude. The pre-transit baseline must constrain that timescale. Compare with a simpler trend if the ramp trades against ingress or depth.

```{image} ../_static/prism_hatp65_whitelight.png
:alt: Example HAT-P-65 b PRISM white-light fit with an early-time ramp
:width: 760px
:align: center
```

The figure is an example fit from the project archive. Your extraction, masks, and binning determine the exact result.

## Inspect the spectrum

```{image} ../_static/prism_hatp65_spectrum.png
:alt: Example HAT-P-65 b native-grid PRISM transmission spectrum
:width: 760px
:align: center
```

PRISM photon counts and saturation risk change strongly with wavelength. Exclude saturated or invalid bins in the extraction or configured masks; do not replace them with finite placeholder fluxes. Inspect depth uncertainty, sampler diagnostics, and residual noise together.

If the native run is too large, change to a numeric resolving power and use the same validation sequence. See [systematics trends](../guides/trends.md) for ramp alternatives and [GPUs and clusters](../guides/gpu_and_clusters.md) for staged runs.

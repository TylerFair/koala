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
  vmap_chunk: 4
```

## Choose the wavelength grid

| Setting | When to use it |
|---|---|
| `high: native` | Preserve the extraction grid when channels have adequate signal and compute is available |
| `high: 100` | Start with a smaller, easier spectrum at constant resolving power |
| `high: reference` | Match a supplied comparison or retrieval grid |

Native PRISM channel counts can be large. Reducing `vmap_chunk` lowers GPU memory use without changing the model or wavelength grid. Binning to lower resolving power changes the scientific product and can improve per-channel constraints.

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

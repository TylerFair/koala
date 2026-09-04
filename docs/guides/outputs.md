# Output files

List a completed directory with:

```bash
find OUTPUT -maxdepth 2 -type f | sort
```

| Product | Contents |
|---|---|
| `*_whitelight_timeseries.csv` | `time_bjd`, hours from `t0`, flux/error, best-fit model, residual and ppm residual, outlier flag; optional transit/trend/GP columns |
| Resolution spectrum `*.csv` | wavelength center/error and depth, depth error, depth ppm, and ppm error for each planet (`00`, `01`, …) |
| `*_bestfit_params.csv` | wavelength, radius ratio and depth summaries, LD, trend, jitter, and active Harmonica parameters |
| `*_wavelengths.csv` | `wavelength_center`, `wavelength_err` |
| `*_lightcurves_wide.csv` | `time`, then `flux_NNN`, `flux_err_NNN` pairs |
| `*_noisebin.csv` | bin size and measured p16/median/p84 RMS plus expected white RMS |
| `chunks/*.pkl` | Fingerprinted posterior checkpoint for a channel range |
| `chunks/*.diagnostics.json` | ESS, divergence, adaptation, and sampler diagnostics |
| `*_limb_spectra.csv` | Harmonica area/endpoint depths and transmission-string coefficients |
| `*_limb_posterior_samples.npz` | Correlated Harmonica posterior arrays |

```python
import pandas as pd
spectrum = pd.read_csv("OUTPUT/target_instrument_R100.csv")
depth = spectrum["depth_ppm00"]
uncertainty = spectrum["depth_err_ppm00"]
```

Numbered PNGs show the white-light fit, spectroscopic offset summaries, spectra, noise binning, and Harmonica limb products when enabled. JSON manifests bind cached science products to the configuration and source fingerprint.

## White-light tables

`*_whitelight_timeseries.csv` has these verified columns:

| Column | Meaning |
|---|---|
| `time_bjd` | Input time |
| `time_from_t0_hr` | Hours relative to the reference transit time |
| `flux` | Normalized white-light flux |
| `flux_err` | Reported uncertainty |
| `bestfit_model` | Evaluated posterior summary model |
| `residual` | `flux - bestfit_model` |
| `residual_ppm` | Residual multiplied by $10^6$ |
| `is_outlier` | Integer outlier-mask flag |
| `transit_model` | Transit component, when materialized |
| `trend_model` | Additive trend component, when materialized |
| `detrended_flux` | `flux - trend_model`, when available |
| `gp_flux` | GP predictive flux, for GP fits |
| `gp_err` | GP predictive uncertainty |
| `gp_trend` | GP stochastic trend component |

`*_whitelight_bestfit_params.csv` contains the fitted white-light parameter summaries. `*_whitelight_geometry_handoff.json` records the geometry passed to wavelength channels and its source metadata. `*_whitelight_outlier_mask.npy` stores the time mask.

`whitelight_mcmc_diagnostics.json` stores convergence and numerical diagnostics.

## Transmission spectrum CSV

The primary spectrum has two wavelength columns followed by four columns per planet.

| Column | Meaning |
|---|---|
| `wavelength` | Bin center in microns |
| `wavelength_err` | Bin half-width/error in microns |
| `depth00` | Median of `rors**2` for planet 0 |
| `depth_err00` | Standard deviation of `rors**2` |
| `depth_ppm00` | `depth00 * 1e6` |
| `depth_err_ppm00` | `depth_err00 * 1e6` |

Additional planets repeat the suffix as `01`, `02`, and so on.

## Detailed parameter CSV

`*_bestfit_params.csv` starts with:

| Column family | Meaning |
|---|---|
| `wavelength`, `wavelength_err` | Channel metadata |
| `rors` | Radius-ratio central value |
| `rors_err`, `rors_err_low`, `rors_err_high` | Standard and percentile errors |
| `depth` | Radius ratio squared |
| `depth_err`, `depth_err_low`, `depth_err_high` | Depth errors |
| `depth_ppm` | Depth in ppm |
| `depth_err_ppm`, `depth_err_low_ppm`, `depth_err_high_ppm` | Ppm errors |
| `depth_error_ratio_to_median` | Channel depth error divided by median channel error |
| `depth_error_outlier_gt5x` | True when that ratio exceeds five |

LD families add `u1`, `u2`, `c1`, or `c2`, each with `_err`, `_err_low`, and `_err_high` columns. Trend families add any active `c`, `v`, `v2`, `v3`, `v4`, `A`, `tau`, `t_jump`, `jump`, spot, GP, or template-scale columns with the same error suffixes. Harmonica adds active `a1`, `a3`, and `a5` summaries.

## Light-curve export

`*_wavelengths.csv` contains `wavelength_center` and `wavelength_err`. `*_lightcurves_wide.csv` contains `time`, followed by pairs `flux_000`, `flux_err_000`, `flux_001`, `flux_err_001`, and so on. The channel number matches the row number in the wavelength file.

## Noise-binning CSV

| Column | Meaning |
|---|---|
| `bin_size_points` | Number of integrations per temporal bin |
| `measured_rms` | Median channel RMS, normally ppm |
| `measured_rms_p16` | 16th percentile across channels |
| `measured_rms_p84` | 84th percentile across channels |
| `expected_white_rms` | Median unbinned RMS divided by square root of bin size |

## Read a chunk posterior

Checkpoints are Python pickle dictionaries, not NumPy `.npy` files. Load only checkpoint files produced by a trusted run.

```python
from pathlib import Path
import pickle
import numpy as np

chunk = next(Path("OUTPUT/chunks").glob("*_chunk_*.pkl"))
with chunk.open("rb") as handle:
    payload = pickle.load(handle)

# Current checkpoints contain samples plus fingerprint/diagnostic metadata.
samples = payload["samples"] if "samples" in payload else payload
for name, value in samples.items():
    arr = np.asarray(value)
    print(name, arr.shape)

rors = np.asarray(samples["rors"])
print(rors.shape)                # [draw, channel, planet] or [draw, channel]
depth_ppm = rors**2 * 1e6
median_depth = np.nanmedian(depth_ppm, axis=0)
```

The leading axis is retained posterior draw. The next axis is channel within the checkpoint. Parameter-specific trailing axes include planet or LD coefficient dimensions.

The checkpoint filename identifies its global channel range. Do not concatenate files by lexical ordering alone; the fitter uses explicit ranges and manifests.

## Harmonica limb CSV

The file contains wavelength metadata, `planet_index`, schema version, and an angular convention string. Schema v3 reports neutral terminator indices because the fit alone cannot identify physical morning/evening hemispheres. The `one`/`two` index is arbitrary; mapping it to physical hemispheres requires external orbital-geometry knowledge. It reports median, lower error, and upper error for:

- `rp_one` and `rp_two` in stellar-radius units;

- `depth_one` and `depth_two` in ppm;

- `depth_total_area` in ppm;

- terminator-one and terminator-two endpoint radii and depths;

- `asymmetry_coefficient` and `endpoint_delta_r` in stellar-radius units;

- `depth_a0` in ppm and `a0` in stellar-radius units;

- every active `a1`, `a3`, and `a5` coefficient.

Optional `bandpass_min` and `bandpass_max` columns appear when supplied. The NPZ companion contains `sample_axes="draw,wavelength"`, units, wavelengths, `a0`, active coefficients, and draw-level radius/depth products. When loading a schema-v2 CSV, the compatibility reader maps old morning columns to index one and old evening columns to index two in memory; users must supply any physical hemisphere interpretation.

## Numbered figures

Numbers 00, 11, 12, 14, and 15 belong to white-light preparation and summaries. Numbers 22--28 belong to low-resolution products. Numbers 31--37 belong to high-resolution products.

The exact set depends on engine, trend, and enabled stages. Use the CSVs for numerical work; the PNGs are diagnostics and presentation summaries.

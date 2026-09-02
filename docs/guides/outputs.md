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


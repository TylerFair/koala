# A synthetic thermal phase curve

Let's fit a synthetic NIRSpec/G395M observation covering a transit, secondary eclipse, and the planet's changing brightness around its orbit.
We'll hold the orbit fixed and measure the phase-curve coefficients in each wavelength channel.
The injected planet has a 300 ppm nightside and reaches maximum brightness 18° before eclipse.

## Make the data and fit

First, generate the observation and run the example:

```bash
python tools/example_phase_curves.py phase_curve
koala -c examples/phase_curve.yaml
```

The {download}`complete YAML <../../examples/phase_curve.yaml>` fixes the geometry and fits a linear trend.
It uses an $R=20$ coarse stage before fitting the native channels.
The planet's emitted flux is

$$L(\phi)=f_d+C_1(\cos\phi-1)+D_1\sin\phi,$$

with $\phi=0$ at secondary eclipse for this circular orbit.
The [phase-curve tutorial](../tutorials/eclipses_and_phase_curves.md#sinusoidal-phase-curve) describes the model and sign conventions.
The configuration gives these starting values and priors:

```yaml
planet:
  eclipse_depth_ppm: {value: 1000.0, prior: uniform, low: 0.0, high: 3000.0}
  phase_c1_ppm: {value: 350.0, prior: uniform, low: -1500.0, high: 1500.0}
  phase_d1_ppm: {value: -100.0, prior: uniform, low: -1500.0, high: 1500.0}
flags:
  light_curve_model: phase_curve
  transit_engine: jaxoplanet
  detrending_type: linear
```

These values are starting points inside uniform priors, not recovered measurements.
Now we can compare the fit with the injected signal:

```{figure} ../_static/phase_recovery.png
:alt: Synthetic phase-curve fit and recovered eclipse-depth and nightside emission spectra

The detrended phase curve, native-channel eclipse and nightside fluxes, and peak offsets.
The transit is included in the fit but extends below the plotted flux range.
```

## Read the emission table

The emission table contains the fitted coefficients and derived quantities:

```python
from pathlib import Path
import pandas as pd

out = Path("results/synthetic_phase_curve")
emission = pd.read_csv(next(out.glob("*_Rnative_emission.csv")))
print(emission[["wavelength", "eclipse_depth_ppm", "phase_c1_ppm", "phase_d1_ppm",
                "nightside_flux_ppm", "phase_amplitude_ppm", "phase_offset_deg"]])
```

At 4.03 µm, the recovered eclipse depth is $985^{+35}_{-36}$ ppm for an injected 1031 ppm.
The nightside flux is $256\pm34$ ppm for an injected 300 ppm.
The brightness maximum occurs $16.8\pm1.9$ degrees before eclipse, compared with the injected 18°.
Across all 14 channels, the mean normalized residual is within $\pm0.4\sigma$ for every quantity.

Each quantity has `_err_low` and `_err_high` columns.
A positive `phase_offset_deg` puts maximum brightness before secondary eclipse (an eastward hotspot).
It is defined as $-\operatorname{atan2}(D_1,C_1)$.
The nightside flux is $f_d-2C_1$.

The model places no limit on the day/night contrast or positivity of the planet flux.
A nightside consistent with zero, or even below zero, is allowed and should be read as an upper limit.
If your observation covers only an eclipse, it generally cannot determine both phase coefficients.
Use the [uniform eclipse model](rocky_eclipse.md) for that case.

The [trend reference](../tutorials/systematics.md) covers polynomials, exponential ramps, Gaussian processes and spot crossings.
The [limb-darkening reference](../tutorials/limb_darkening.md) covers quadratic and power-2 laws with fixed, Gaussian, uniform and atmosphere-based coefficient priors.

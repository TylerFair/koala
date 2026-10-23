# A thermal phase curve

A phase curve follows a planet around most or all of its orbit, so we see the transit, the eclipse, and the planet's brightness changing in between as its hot dayside turns towards us and away again.
In this case study we'll fit a synthetic NIRSpec/G395M phase curve and measure, at each wavelength, the eclipse depth, the nightside brightness and how far the brightest point is shifted from the substellar point.
The injected planet has a 300 ppm nightside and is brightest 18° before eclipse.

## The model

koala describes the planet's brightness with a cosine and a sine of the orbital phase,

$$L(\phi) = f_d + C_1(\cos\phi - 1) + D_1 \sin\phi,$$

where $\phi = 0$ at secondary eclipse for a circular orbit, $f_d$ is the planet's flux at eclipse (`eclipse_depth_ppm`), and $C_1$ and $D_1$ are `phase_c1_ppm` and `phase_d1_ppm`.
The full light curve is the transit plus this planet flux times the visible fraction of the planet's disk (which is 0 in eclipse), all multiplied by the trend.
Three numbers follow from these:

| Column | |
|---|---|
| `nightside_flux_ppm` | $f_d - 2C_1$, the planet's flux at transit |
| `phase_amplitude_ppm` | $\sqrt{C_1^2 + D_1^2}$ |
| `phase_offset_deg` | $-\mathrm{atan2}(D_1, C_1)$, positive when the planet is brightest before eclipse (a hotspot shifted east) |

Nothing stops the nightside from coming out negative, so read a nightside consistent with zero as an upper limit.

## Making the data and fitting it

First, generate the observation and run the example:

```bash
python tools/example_phase_curves.py phase_curve
koala -c examples/phase_curve.yaml
```

The {download}`configuration <../../examples/phase_curve.yaml>` holds the orbit fixed and fits a linear trend, with these priors on the planet's brightness:

```yaml
planet:
  eclipse_depth_ppm: {value: 1000.0, prior: uniform, low: 0.0, high: 3000.0}
  phase_c1_ppm: {value: 350.0, prior: uniform, low: -1500.0, high: 1500.0}
  phase_d1_ppm: {value: -100.0, prior: uniform, low: -1500.0, high: 1500.0}
flags:
  light_curve_model: phase_curve
```

`phase_c1_ppm` and `phase_d1_ppm` can be negative, so give them a uniform or Gaussian prior rather than a log-uniform one.

```{figure} ../_static/phase_recovery.png
:alt: The fitted synthetic phase curve and the recovered eclipse depth, nightside flux and offset at each wavelength

The phase curve after removing the trend (the transit is in the fit, but below the plotted range), and the recovered eclipse depth, nightside flux and offset in each channel.
```

## Reading the results

The results are in the emission table:

```python
from pathlib import Path
import pandas as pd

out = Path("results/synthetic_phase_curve")
emission = pd.read_csv(next(out.glob("*_Rnative_emission.csv")))
print(emission[["wavelength", "eclipse_depth_ppm", "nightside_flux_ppm", "phase_offset_deg"]])
```

At 4.03 µm, the eclipse depth is $985^{+36}_{-35}$ ppm (we put in 1031 ppm), the nightside $258^{+32}_{-36}$ ppm (300 ppm), and the brightest point $16.8 \pm 1.9$° before eclipse (18°).
Across all 14 channels, every quantity is within 0.5σ of the truth on average, and the channels scatter around it by about as much as their uncertainties say they should.
Each column has `_err_low` and `_err_high` errors alongside it.

If your observation covers only an eclipse, it can't usually pin down both $C_1$ and $D_1$, and the [uniform eclipse model](rocky_eclipse.md) is a better choice.

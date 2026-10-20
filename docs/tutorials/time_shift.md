# Timing spectra

We usually hold the transit time fixed when fitting the wavelength channels.
Here we'll let each channel have its own transit time and measure the offset from the white-light transit.
The planet's orbit otherwise stays fixed.
You can try this with a copy of the bundled SOSS configuration.

## Fit the channel times

First, copy the example so the timing fit has its own configuration:

```bash
cp examples/niriss_soss_order1.yaml my_timing.yaml
```

Then merge these settings into the copy:

```yaml
flags:
  spectro_free_t0: true
  spectro_t0_half_width_s: 1800
```

Choose a separate `output_dir` and run it:

```bash
koala -c my_timing.yaml
```

Each channel gets a uniform epoch-offset prior within 1800 seconds of the white-light transit time.
That is the default half-width.
The period, semimajor axis, inclination, eccentricity, and argument of periastron stay at the white-light solution.
Both depth and relative epoch must pass the sampling checks.

It works with the jaxoplanet transit model, including spot-crossing trends, but not with the starry surface-spot model, and it cannot be combined with `spectro_joint_geometry`.
Transit-time grids stay off.
The calculation accounts for the full allowed epoch and radius ranges when deciding which times can be outside transit.

## Read the offsets

Now open `*_time_shift.csv` and the `*_t0_spectrum.png` or `.pdf` plot.
Positive offsets mean later transits.
The table includes the correlation between depth and transit time, which helps you see whether the two measurements trade off.
These uncertainties are conditional on the white-light orbit and epoch.
They do not propagate their shared uncertainty.

After each channel is sampled, koala also works out how far that channel's transit time moves when one half of the planet grows.
This is `time_shift_coefficient`, in seconds per stellar radius.
It is calculated for that channel's own fitted radius, limb darkening and trend, with the white-light orbit held fixed, so you do not need a separate calibration for each planet.
It takes a fraction of a second per channel.
Read the sign convention in `*_time_shift.json` before interpreting it as a limb asymmetry.

The white-light transit time is fitted, so a radius difference that is the same at every wavelength moves the white-light time and does not show up in `dt0_s`.
When you fit timing spectra in a retrieval, give each visit its own free time offset.
If you want to fit the boundary itself, see [asymmetric limbs with Harmonica](harmonica.md).

## Reference

### Timing columns

The main spectrum gains these columns when free channel times are enabled.
The separate timing CSV has one row per channel and planet.
For multiple planets, the main spectrum uses the existing two-digit planet suffixes.

| Column | Meaning |
|---|---|
| `depth`, `depth_err` | Transit depth and central 68-percent interval half-width. |
| `t0`, `t0_bjd` | Transit epoch in full BJD_TDB. |
| `t0_err_s` | Central 68-percent timing interval half-width in seconds. |
| `dt0_s` | Offset from the white-light transit time in seconds. |
| `corr_depth_t0` | Posterior correlation between depth and epoch. |
| `time_shift_coefficient` | Change in transit time, in seconds, per stellar radius of growth of the trailing (evening) half. |
| `d_dt0_dRL_s_per_Rstar`, `d_dt0_dRT_s_per_Rstar` | Change in transit time when only the leading or only the trailing half grows. A larger leading half gives an earlier transit. |
| `d_depth_dRL`, `d_depth_dRT` | The same for the fitted depth. |
| `antisymmetry_check` | `abs(d_dt0_dRL + d_dt0_dRT) / abs(d_dt0_dRT)`. When it is small, one coefficient describes both halves. |
| `time_shift_coefficient_analytic` | The circular half-disk formula below, for comparison. NaN where it is undefined. |
| `time_shift_coefficient_difference_percent` | `100 * (exact / analytic - 1)`. |
| `radius_response_relative_error` | How well the two half responses add up to the model's own radius derivative. |
| `n_retained` | Number of integrations used in the calculation. |

### From a radius difference to a time shift

koala splits the planet into a leading (morning) and a trailing (evening) half, divided perpendicular to the planet's motion on the sky.
It integrates the stellar intensity along the edge of each half, using the fitted limb darkening and trend, which gives the change in the light curve when that half grows.
It then asks how a spherical-planet fit to that channel would absorb the change.
With the model Jacobian $J$ in all fitted parameters, the weights $W$ from the per-integration errors and median jitter, and the prior curvature $P$, the response of the fitted parameters is

$$
\frac{d\theta}{dR_h} = (J^{\rm T} W J + P)^{-1} J^{\rm T} W \frac{dF}{dR_h}.
$$

The epoch and depth parts of this are the columns above.
Every retained integration is used, and the orbit can be eccentric or grazing.
For small radius differences,

$$
\Delta t_0 = \frac{\partial t_0}{\partial R_L}\,\delta R_L + \frac{\partial t_0}{\partial R_T}\,\delta R_T + \text{visit offset}.
$$

This is exact to first order about a spherical planet.
Large differences between the limbs can be nonlinear.
The per-channel timing errors do not include the shared white-light orbit and epoch uncertainty.

For comparison, the analytic half-disk formula is

$$
\Delta t_0 = \frac{C}{v}(R_{\rm trailing}-R_{\rm leading}) + \text{visit offset},
$$

with

$$
s=\sqrt{1-\frac{1-b^2}{(a/R_*)^2}},\qquad
\theta=\arcsin(b/s),
$$
$$
C=\frac{3(\pi^2/4-\theta^2+\cos^2\theta)}{16\cos\theta},\qquad
v=\frac{2\pi(a/R_*)s}{P}.
$$

Here the radii are in stellar-radius units and $P$ is in seconds.
It assumes a circular orbit and a chord that crosses the whole star, so it is reported as NaN for eccentric or grazing transits.
For a uniform star, a central transit and a small planet it reduces to $C = 3(\pi^2+4)/64$.
For the WASP-94 A b channels the two agree to about half a percent.

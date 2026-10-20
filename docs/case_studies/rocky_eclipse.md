# A rocky-planet eclipse with MIRI/LRS

Let's fit a synthetic MIRI/LRS observation of ROCKY-1 b.
The example injects a flat 100 ppm eclipse into 14 wavelength channels.
We'll recover the emitted flux, then look at a shallower eclipse where the timing is harder to measure.

## Make the data and fit

First, generate the input file and run the bundled configuration:

```bash
python tools/example_phase_curves.py eclipse
koala -c examples/eclipse.yaml
```

The {download}`complete YAML <../../examples/eclipse.yaml>` uses a linear baseline and a uniform eclipse-time window.
The scaled semimajor axis and impact parameter have Gaussian priors.
The radius ratio and period are fixed.
The final grid is `native`, with no coarse stage.
Since the star is not occulted during secondary eclipse, the stellar limb-darkening coefficients are fixed to zero.

```{figure} ../_static/eclipse_recovery.png
:alt: Fitted synthetic MIRI eclipse and the recovered emission spectrum

The white-light eclipse and residuals, followed by the native-channel emission spectrum.
The dashed line marks the injected 100 ppm planet flux.
The white-light depth is $94.3^{+7.3}_{-7.2}$ ppm.
```

The recovered white-light depth is close to the injection.
Now read `*_Rnative_emission.csv` for each channel's `eclipse_depth_ppm` and its lower and upper errors.
The generic `depths` column in the white-light table is $(R_p/R_*)^2$.
Use the emission table when you want the planet's emitted flux.

## When the eclipse is not recovered

A shallow eclipse on a ramp is harder.
This separate experiment has 2,000 integrations, a 3,000 ppm exponential ramp, a 70 ppm eclipse, and 350 ppm white-light scatter per integration:

```{figure} ../_static/eclipse_time_scan.png
:alt: A shallow eclipse on an exponential ramp and a timing scan, with the true epoch and a half-day ephemeris mistake marked

The summed white-light curve and a nonnegative-depth least-squares scan over eclipse time.
The injection is at BMJD_TDB 60001.3081322.
The ramp timescale is estimated outside the injected event and held fixed during the scan.
The baseline, slope, and ramp amplitude are refitted at each epoch.
The best conditional depth is $66.8\pm19.9$ ppm, with $\Delta\chi^2=11.30$.
This scan gives neither a global detection significance nor a koala posterior.
```

First, inspect `00_*_preopt_init_check.png` and `*init_vs_opt_check.png`.
The model eclipse needs to overlap the observation and the suspected flux decrement.
Check the time origin before changing the model.
koala uses BMJD_TDB = BJD_TDB − 2400000.5.
An epoch quoted as “BJD − 2400000” is 0.5 d late on this clock.
Full BJD data are converted automatically, but configuration epochs are not.

A starting epoch more than roughly one eclipse duration away can leave the model without a useful timing gradient.
koala scans the allowed eclipse-time window by linear least squares and starts from its best epoch.
The log reports this as `Eclipse-time scan: ...`.
That scan can find a starting point, but it cannot add timing information to the data.

Below about $5\sigma$ in white light, the eclipse time can be poorly constrained and slow to sample.
koala prints a warning in this case.
A GP or a ramp with a free timescale can also absorb part of a shallow eclipse.
Compare a simpler trend and, if appropriate, trim initial settling with [time exclusions](../tutorials/systematics.md#exclude-time-ranges).

## Use the transit ephemeris

For the shallow experiment, fixing timing and geometry gives a white-light depth of $68\pm20$ ppm.
All 14 native channels are consistent with the injected 70 ppm.
Allowing the epoch to move over $\pm2$ hours also admits solutions with almost no eclipse at many possible times.

For this observation, replace the eclipse configuration's `planet` block with:

```yaml
planet:
  name: ROCKY-1
  period: {value: 2.6162644, prior: fixed}
  eclipse_time: {value: 60001.3081322, prior: fixed}
  a_rs: {value: 17.04, prior: fixed}
  b: {value: 0.208, prior: fixed}
  rprs: {value: 0.0318, prior: fixed}
  ecc: {value: 0.0, prior: fixed}
  eclipse_depth_ppm: {value: 70.0, prior: uniform, low: 0.0, high: 1000.0}
```

This changes the orbital assumptions without changing the input file.
The `eclipse_time` parameter requires a fixed period and zero eccentricity.
koala derives `t0 = eclipse_time - period / 2`.
With timing and geometry supplied by the transit analysis, we can concentrate on the depth, baseline, and noise.

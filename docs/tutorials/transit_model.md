# How koala computes a transit

To calculate a transit, we first find the planet's position and then work out how much starlight it blocks.
The planet's radius and the star's limb darkening can change with wavelength.
The orbit is shared by all channels.
Let's follow that calculation through the bundled SOSS fit.

## Start with white light

We sum the flux over wavelength to make the white-light curve.
This gives us the orbital geometry for the channel fits:

```{figure} ../_static/workflow.png
:alt: White-light fit supplies orbital geometry to wavelength channel fits, which produce a spectrum

The white-light fit measures the orbit, and the channel fits measure the spectrum.
These curves are schematic.
```

By default, koala holds the period, reference epoch, impact parameter, and duration or scaled semimajor axis at their white-light posterior medians.
Each channel has its own radius ratio, limb-darkening coefficients when free, trend, and noise.
The resulting spectrum is conditional on this one orbital solution.
[Joint spectroscopic geometry](planet_parameters.md#joint-spectroscopic-geometry) includes shared geometric uncertainty within a spectroscopic stage.

## Add the trend

For each channel, the mean normalized flux is

$$F(t) = [1 + s(t)]\,T(t),$$

where $s(t)$ is the astrophysical signal and $T(t)$ is the instrument trend.
A transit makes $s(t)$ negative.
An emitting planet adds positive flux outside secondary eclipse.
The likelihood uses the reported flux errors and a fitted white-noise term.
A [Gaussian process](systematics.md#gaussian-processes) can describe additional time-correlated structure.

Now look at the wavelength-channel curves in [your first fit](../quickstart.md#four-views-of-the-fit).
They have been divided by their fitted trends.
Their depths can differ even though they use the same orbit.

## Choose the wavelength grid

To fit an optional coarse spectrum before the final grid, use:

```yaml
resolution:
  low: 20
  high: 100
```

Here $R=\lambda/\Delta\lambda$.
Omit `low` to go straight to the final grid, or use `high: native` to keep the input wavelength channels.
These choices combine wavelengths only.
koala never averages integrations in time.

The [SOSS case study](../case_studies/transmission_soss.md) shows how to change the grid and compare the spectra.
For the mathematical calculation, keep reading.

## Reference

### The planet's position

koala uses `jaxoplanet.core.kepler` to solve Kepler's equation.
In stellar-radius units,

$$
M(t)=M_0+\frac{2\pi(t-t_0)}{P},\qquad E-e\sin E=M,
$$
$$
f=2\operatorname{atan2}\!\left(\sqrt{1+e}\sin(E/2),
                              \sqrt{1-e}\cos(E/2)\right),\qquad
r=\frac{(a/R_*)(1-e^2)}{1+e\cos f},
$$
$$
z=r\sqrt{\cos^2(f+\omega)+\cos^2 i\,\sin^2(f+\omega)}.
$$

Here $M_0$ corresponds to $f_0=\pi/2-\omega$ at the reference epoch, and
$\cos i=b(1+e\sin\omega)/[(a/R_*)(1-e^2)]$.
A foreground mask prevents a planet behind the star from producing a transit.
For a fixed circular orbit, $f=M$, so we can skip the Kepler solve.

With fixed geometry, koala calculates the projected separation $z(t)$ once and shares it across channels.
If you supply a duration, koala first converts it to $a/R_*$ using the white-light reference radius.
The channel radii can then vary without changing the orbit.
A joint-geometry fit recalculates the separation as the orbit changes.
With [free channel epochs](time_shift.md), each channel uses $z(t-\Delta t_0)$ while keeping $P,a/R_*,b,e,\omega$ fixed.

### The blocked light

The flux calculation takes the projected separation, radius ratio, and limb-darkening coefficients.
The default quadratic calculation handles contact points with finite derivatives.
Ordinary unspotted power-2 transits use direct integration of the intensity law.
[Eclipses, phase curves, and spotted surfaces](eclipses_and_phase_curves.md) use their surface models.

Older koala fits used a straight-line speed $v=2\sqrt{(1+k)^2-b^2}/T_{14}$.
Recomputing that speed with each channel's radius ratio $k$ gave each wavelength a different orbit and stretched spectral features.
The current calculation keeps the orbit independent of channel radius.
Spectra fitted with the older model need a new fit, especially when duration was fixed and limb darkening was free.
koala does not reuse the old posterior checkpoints or saved geometry for this calculation.

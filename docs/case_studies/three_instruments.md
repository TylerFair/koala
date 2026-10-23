# WASP-39 b with NIRISS, NIRSpec and MIRI

WASP-39 b has been observed with all four of JWST's instruments, so it's a good planet for putting several of them together.
Here we'll take three of those observations, NIRISS/SOSS from 0.85 to 2.8 µm, NIRSpec/G395H from 2.9 to 5 µm and MIRI/LRS from 5 to 12 µm, fit one orbit to all of them at once, and then measure the spectrum from each.

Each instrument needs slightly different choices, for the wavelength channels and for the trend, and that's most of what this page is about.
If you haven't yet, start with [your first fit](../quickstart.md) and [both SOSS orders](transmission_soss.md), which this builds on.

## The data

The three observations were taken on three different visits:

| Instrument | Program | Integrations | File in `examples/data/` |
|---|---|---|---|
| NIRISS/SOSS, order 1 | ERS 1366 | 537 | `WASP-39_soss_binned8.fits` |
| NIRSpec/G395H, NRS1 and NRS2 | ERS 1366 | 465 | `WASP-39_g395h_nrs1_binned8.fits`, `WASP-39_g395h_nrs2_binned8.fits` |
| MIRI/LRS | DDT 2783 | 1779 | `WASP-39_miri_lrs.fits` |

All of them are exoTEDRF extractions.
As with the SOSS file, every eight neighbouring columns of the G395H detectors have been added together to keep the files small, and the MIRI/LRS file keeps every column from 4.9 to 12.1 µm.
Every integration is still there.

Here are their white-light curves, before any fitting:

```{figure} ../_static/three_instruments_data.png
:alt: The white-light curves of the SOSS, G395H NRS1 and NRS2, and MIRI/LRS observations of WASP-39 b
```

SOSS is the simplest, with a flat baseline either side of the transit.
G395H has a sudden drop in flux about 10 minutes after mid-transit, on both detectors.
That's a known feature of this observation: one of the telescope's mirror segments shifted slightly, which changed the shape of the star's image on the detector ([Alderson et al. 2023](https://doi.org/10.1038/s41586-022-05591-3)).
MIRI/LRS starts with a ramp as the detector settles, and drifts down slowly for the rest of the visit.

## One file for each dataset

As in the two-order fit, each dataset has its own configuration file, and a short joint file ties them together.
The planet and the star are the same in every file, apart from the expected transit time of each visit.

**The instrument.**
SOSS needs the order and NIRSpec the detector, so the two G395H files differ only in `nrs: 1` and `nrs: 2` (and the file name):

```yaml
instrument: NIRSPEC/G395H
nrs: 1
```

**The wavelength channels.**
Each dataset uses a different kind of channel, because the right choice depends on the instrument and on what you want to do with the spectrum.
SOSS uses a resolving power:

```yaml
resolution:
  high: 100
```

which gives channels of the same $\lambda/\Delta\lambda$ everywhere, the usual choice for a spectrum you want to compare across instruments.
G395H fits every wavelength in the file as its own channel:

```yaml
resolution:
  high: native
```

That's worth it when you want the shape of narrow features, such as the CO₂ band at 4.3 µm, or when you plan to bin the spectrum yourself later (in a retrieval, say).
It's also the slowest choice: here it gives 364 G395H channels, against 115 for SOSS at $R = 100$.
MIRI/LRS uses channels of six detector columns each:

```yaml
pixels:
  high: 6
```

That suits MIRI/LRS, whose resolving power climbs from about 40 at 5 µm to about 160 at 10 µm, so one constant $R$ doesn't fit the whole band.
The last choice, `reference` (not used here), takes the bin edges from a CSV file given as `resolution.reference_grid`.
Use it when you want exactly the same channels as another spectrum, such as a published one or a PRISM observation of the same planet.

**The trend.**
SOSS has the default straight line.
For G395H, the trend is a straight line with a step, starting from a rough time and size for the step:

```yaml
flags:
  detrending_type: linear_discontinuity
  t_jump_guess: 59791.6205      # BJD_TDB - 2400000
  jump_guess: -0.0015
```

koala fits the time and size of the step in the white light, and then fits the size of the step in every channel, with its time held fixed.
For MIRI/LRS, the trend is an exponential ramp plus a straight line (`explinear`), and we leave out the first 20 minutes, where the ramp is steepest:

```yaml
exclude_times: [["min(t)", "min(t) + 0.0139"]]   # 20 minutes, in days
flags:
  detrending_type: explinear
```

The G395H files also leave out the first two integrations.
The [trends page](../guide/trends.md) lists the other trends you can choose from.

**The star.**
Nothing to change: the default limb darkening is fitted freely in every channel, and koala works out the starting values from the 3D Stagger stellar models for each instrument.

## Tying them together

The joint file is [`examples/wasp39_three_instruments.yaml`](https://github.com/TylerFair/koala/blob/main/examples/wasp39_three_instruments.yaml):

```yaml
joint:
  members: [wasp39_soss_order1.yaml, wasp39_g395h_nrs1.yaml, wasp39_g395h_nrs2.yaml, wasp39_miri_lrs.yaml]
planet:
  a_rs: {value: 11.4, prior: uniform, low: 5.0, high: 25.0}
  b: {value: 0.45, prior: uniform, low: 0.0, high: 1.0}
output_dir: ../results/WASP-39_three_instruments
```

All four white-light curves share `a_rs` and `b`.
koala groups the members into visits by their observing times, so the two G395H detectors, which were read out at the same time, share one transit time, and SOSS and MIRI/LRS each get their own.
The radius ratio, limb darkening, trend and jitter are fitted separately for each dataset.

## Running the fit

From the top of the repository:

```bash
koala -c examples/wasp39_three_instruments.yaml
```

This took about 25 minutes on 15 cores of an Intel Xeon node, ten of them for the 364 native-resolution G395H channels.
koala reads the four datasets first, then fits their white-light curves together:

```text
Planet parameters:
  period      fixed at 4.05528043
  t0 visit 1  uniform(59787.505, 59787.605), start 59787.555
  t0 visit 2  uniform(59791.56, 59791.66), start 59791.61
  t0 visit 3  uniform(59990.269, 59990.369), start 59990.319
  a_rs        uniform(5.0, 25.0), start 11.4
  b           uniform(0.0, 1.0), start 0.45
  rprs        uniform(0.01, 0.5), start 0.146
Limb darkening: quadratic law with uniform priors on u1 + u2 and u1 - u2
Joint white light: 4 observations sharing a_rs and b, 4 chains of 1000 draws
sample: 100%|██████████████████████████████████████████████████| 1200/1200 [02:31<00:00,  7.90it/s]
White-light convergence checks need more draws; extending the sampling.
sample: 100%|██████████████████████████████████████████████████| 1000/1000 [01:24<00:00, 11.77it/s]
  Converged: R-hat <= 1.003, bulk effective sample size >= 870, no divergences
  a_rs             11.377 ± 0.014
  b                0.4521 ± 0.0029
  t0 visit 1       59787.556748 ± 0.000014 BJD_TDB - 2400000
  t0 visit 2       59791.612056 ± 0.000014 BJD_TDB - 2400000
  t0 visit 3       59990.320812 ± 0.000034 BJD_TDB - 2400000
  rprs SOSS order 1 0.14583 ± 0.00011
  rprs G395H NRS1  0.14602 ± 0.00011
  rprs G395H NRS2  0.14710 ± 0.00013
  rprs MIRI/LRS    0.14529 ± 0.00020
  Residual scatter 109 ppm (SOSS order 1), 114 ppm (G395H NRS1), 173 ppm (G395H NRS2), 544 ppm (MIRI/LRS); 1 integration clipped at 5σ
```

The four white-light curves give $a/R_* = 11.377 \pm 0.014$ and $b = 0.4521 \pm 0.0029$.
That agrees with the two SOSS orders on their own ($11.390 \pm 0.021$ and $0.4516 \pm 0.0042$, in [the two-order fit](transmission_soss.md)), with uncertainties about a third smaller, since all three instruments see the same orbit.
Each visit has its own transit time, and each dataset its own white-light radius ratio, which differ because they average over different wavelengths: NRS2, for example, covers the CO₂ band.
The scatter of the residuals goes from 109 ppm for SOSS to 544 ppm for MIRI/LRS, which collects far fewer photons in each of its shorter integrations.

Then koala fits the channels of each dataset in turn, with the orbit and that visit's transit time held fixed, and all 520 channels pass the convergence checks.

## The spectrum

koala puts the four spectra on one wavelength axis in `figures/joint_spectrum.png`:

```{figure} ../_static/three_instruments_spectrum.png
:alt: The transmission spectrum of WASP-39 b from 0.85 to 12 µm, from SOSS, G395H and MIRI/LRS
```

The water bands in SOSS are the same as in [your first fit](../quickstart.md).
G395H adds the large CO₂ band at 4.3 µm and a smaller bump at 4.05 µm, which [Alderson et al. (2023)](https://doi.org/10.1038/s41586-022-05591-3) identified as SO₂.
In MIRI/LRS the depth rises around 7.7 µm, where [Powell et al. (2024)](https://doi.org/10.1038/s41586-024-07040-9) found more SO₂, and falls beyond 10 µm, where the uncertainties also grow quickly as the light runs out.
The native G395H channels scatter much more than the others, because each one is only eight detector columns wide; `figures/joint_spectrum_binned.png` averages them into wider bins for display.

The axis is linear up to 5 µm and logarithmic beyond (the dotted line), so that the MIRI/LRS channels, which cover more than half of the wavelength range, don't squeeze the rest of the spectrum.
The numbers are in `joint_spectrum.csv`, and each dataset's own directory has its spectrum, light curves and figures as usual.

## Things to check

A shared orbit removes one reason for the instruments to disagree, but not all of them.
Different visits can catch the star with a different coverage of spots, and the instruments and their reductions can leave small offsets of their own, so a step in depth between two instruments isn't necessarily the planet.
Before reading anything into one, it's worth checking:

- each dataset's `figures/whitelight.png`, to see that the trend has caught the step in G395H and the ramp in MIRI/LRS,
- each dataset's `figures/<grid>_residual_map.png`, where anything the trend missed shows up as a stripe across all wavelengths,
- that the two G395H detectors agree on either side of the gap between them, near 3.8 µm.

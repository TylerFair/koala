# WASP-39 b with both SOSS orders

NIRISS/SOSS records two spectral orders at the same time.
In [your first fit](../quickstart.md) we only used order 1, which covers 0.85–2.8 µm.
Order 2 extends the spectrum down to 0.6 µm, so here we'll fit both orders of the same WASP-39 b observation.

We could fit each order on its own, but the two orders see exactly the same transit, and two separate fits would give two slightly different orbits.
The spectra would then each be measured with a different orbit, which can leave an offset between them that has nothing to do with the planet.
Instead, we'll fit one orbit to both white-light curves, and then fit the channels of each order with that orbit.

## Setting up the fit

A joint fit is made of ordinary configuration files, one for each order, and a short file that ties them together.
The member files are [`examples/joint_soss_order1.yaml`](https://github.com/TylerFair/koala/blob/main/examples/joint_soss_order1.yaml), which is the same as the first fit, and [`examples/joint_soss_order2.yaml`](https://github.com/TylerFair/koala/blob/main/examples/joint_soss_order2.yaml), which differs only in these lines:

```yaml
order: 2
wavelength_filter:
  wl_min_o2: 0.6
  wl_max_o2: 0.85
```

Order 2 overlaps order 1 above 0.85 µm, where order 1 has far more signal, so we only use it below that.
The joint file, `examples/joint_soss_orders.yaml`, is:

```yaml
joint:
  members: [joint_soss_order1.yaml, joint_soss_order2.yaml]
planet:
  a_rs: {value: 11.4, prior: uniform, low: 5.0, high: 25.0}
  b: {value: 0.45, prior: uniform, low: 0.0, high: 1.0}
output_dir: ../results/WASP-39_joint
```

The two orders share the scaled semi-major axis `a_rs` and the impact parameter `b`, with the priors given here.
Since their observing times overlap, koala treats them as one visit, so they share the transit time too.
Everything else is fitted separately for each order: the radius ratio, limb darkening, trend and jitter.
(The paths in `members` and `output_dir` are relative to the joint file.)

## Running the fit

From the top of the repository:

```bash
koala -c examples/joint_soss_orders.yaml
```

This took about 7 minutes on 16 cores of an Intel Xeon node.
koala fits the two white-light curves together first, then the channels of order 1, then those of order 2.
The joint orbit is in `results/WASP-39_joint/joint_params.csv`, and each order gets its own directory (`joint_soss_order1/` and `joint_soss_order2/`) with the usual [output files](../guide/outputs.md).

The joint fit gives $a/R_* = 11.390 \pm 0.021$, $b = 0.4516 \pm 0.0043$ and a transit time of $59787.056747 \pm 0.000012$ BMJD_TDB.
The white-light radius ratios are $0.14591 \pm 0.00016$ in order 1 and $0.14527 \pm 0.00020$ in order 2, which differ because the two orders cover different wavelengths.

## The spectrum

Now we can put the two orders together:

```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

out = Path("results/WASP-39_joint")
fig, ax = plt.subplots(figsize=(9, 4))
for order in (1, 2):
    spec = pd.read_csv(next((out / f"joint_soss_order{order}").glob("*_R100_spectrum.csv")))
    ax.errorbar(spec["wavelength"], spec["depth_ppm00"],
                xerr=spec["wavelength_err"], yerr=spec["depth_err_ppm00"],
                fmt="o", ms=3, label=f"order {order}")
ax.legend()
ax.set_xlabel("wavelength (µm)")
ax.set_ylabel("transit depth (ppm)")
```

```{figure} ../_static/soss_both_orders.png
:alt: The WASP-39 b transmission spectrum from SOSS orders 1 and 2 at R = 100
```

Order 2 carries the spectrum down to 0.6 µm, and the two orders line up where they meet at 0.85 µm.
The high point at 0.77 µm sits on the potassium doublet, which Feinstein et al. (2023) found in these same data.
The first order-2 channel, at 0.6 µm, is at the very edge of the order where there's little light, which is why its uncertainty is so large.
koala also makes its own version of this figure, `figures/joint_spectrum.png`.

## Fitting every column

Our example file has every eight detector columns added together, which leaves 246 usable wavelengths in order 1.
To fit each of them as its own channel, change the `resolution` block of a member file to:

```yaml
resolution:
  high: native
```

That's about twice as many channels as at $R = 100$, so the channel fits take roughly twice as long.
For a real analysis you'd start from an extraction at the detector's full resolution, and choose the binning from there.

## More orders, detectors and visits

The same set-up works for any combination of observations of the same planet: SOSS orders, the two NIRSpec detectors, different instruments, or several visits.
List one configuration per dataset under `members`.
Datasets whose observing times overlap are grouped into one visit and share a transit time, and separate visits each get their own.
To group them yourself, give `visits` as lists of member file names without the `.yaml`, for example `visits: [[soss_order1, soss_order2], [nrs1]]`.

If you'd like a dataset to use the joint orbit without contributing to it (a noisier visit, say), put its configuration under `apply_to` instead of `members`.
It gets its own white-light fit with `a_rs` and `b` held at the joint values, and then its channels are fitted as usual.

There are a few limits for now: the orbit has to be circular, the period has to be fixed and the same in every file, and only `a_rs` and `b` can be shared.

## Including the orbit's uncertainty in the spectrum

Normally the channels are fitted with the orbit held at the white-light result, so the depth uncertainties don't include the uncertainty in the orbit.
That's usually fine, because the orbit mostly moves every channel up or down together, and barely changes the shape of the spectrum.
If you do want it included in a single-dataset fit, `spectro_joint_geometry` fits the orbit again, together with all the channels at once (this can't be combined with a joint fit like the one above yet):

```yaml
flags:
  spectro_joint_geometry: true
  spectro_sampler: joint_nuts
```

The orbit then has Gaussian priors centred on the white-light result, three times wider than the white-light uncertainties.
Because every channel is fitted at the same time, this needs a lot more memory.
Afterwards, `*_joint_geometry.csv` compares the orbit from the channels with the white-light one.

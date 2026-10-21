# A joint timing spectrum

Use SOSS orders 1 and 2 to measure one white-light orbit, then fit a depth and transit time in every channel of each order.
This example uses the bundled WASP-39 b extraction. Both orders cover the same visit, so they share a white-light epoch.

## Prepare the members

From the repository root, make two member files and a joint configuration:

```python
from pathlib import Path
import yaml

for order in (1, 2):
    cfg = yaml.safe_load(Path("examples/niriss_soss_order1.yaml").read_text())
    cfg["order"] = order
    cfg["resolution"] = {"high": 20}
    cfg["flags"]["spectro_free_t0"] = True
    cfg["flags"]["spectro_t0_half_width_s"] = 1800
    Path(f"timing_order{order}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
```

Save the following as `timing_joint.yaml`:

```yaml
joint:
  members: [timing_order1.yaml, timing_order2.yaml]
  shared: [a_rs, b]
  visits: auto
planet:
  a_rs: {value: 11.5, prior: uniform, low: 5.0, high: 25.0}
  b: {value: 0.45, prior: uniform, low: 0.0, high: 1.0}
output_dir: results/WASP-39_timing
host_device: cpu
```

Run it with:

```bash
koala -c timing_joint.yaml
```

The circular orbit has a fixed period and shared `a_rs` and `b`. Each order has its own white-light radius, limb darkening, trend and noise.
Spectroscopy fixes the orbit and visit epoch at their joint posterior medians, then fits a free channel epoch within ±1800 s of that epoch.
For several visits or instruments, add their member files to `joint.members`; overlapping time ranges form visits automatically.
See [joint white light](../tutorials/planet_parameters.md#joint-white-light-across-observations) for explicit visit groups and prior rules.

## Read depths and offsets

The joint orbit and its diagnostics are in `results/WASP-39_timing/joint/`.
Member products are in `timing_order1/` and `timing_order2/` under the same output directory.
Each contains the ordinary depth spectrum, a `*_time_shift.csv` table and a `*_t0_spectrum.png` figure.
`joint/datasets/` has a two-panel figure for each member, with the transit depth above and the time offset below.
With two or more members, `joint/joint_spectrum.png` puts every channel on one wavelength axis, and
`joint/joint_spectrum_binned.png` shows the same channels combined in wavelength for display.
When two visits cover the same wavelengths, `joint/joint_spectrum_ivw.png` averages all datasets with inverse-variance weights.
Datasets fitted on the same channels share display bins, so the same channel of two visits is always averaged together.
Its timing panel has its zero at the weighted-mean transit time over 1.22-1.30 µm, between the 1.15 and 1.4 µm water bands.
Each visit is first shifted by a constant depth and time offset fitted where the visits overlap; `joint/ivw_offsets.csv` lists them with the chi-square of the overlap.
`joint/limb_spectra_ivw.png` gives the morning and evening depths of the combined spectrum.
These are display products: their errors leave out the uncertainty of the offsets, so a retrieval should fit the members with their own offsets.
Make these figures again for a finished run with `python -m koala.joint_products results/WASP-39_timing`.

```python
from pathlib import Path
import pandas as pd

out = Path("results/WASP-39_timing")
for member in ("timing_order1", "timing_order2"):
    timing = pd.read_csv(next((out / member).glob("*_R20_time_shift.csv")))
    print(member)
    print(timing[["wavelength", "depth", "depth_err", "dt0_s", "t0_err_s",
                  "corr_depth_t0"]])
```

Positive `dt0_s` means a later transit than the visit's white-light epoch.
Depth and timing errors are conditional on the shared orbit and epoch; they do not propagate their uncertainty.
Use `corr_depth_t0` to check their correlation, and retain a free time offset per visit in a retrieval.
The [timing reference](../tutorials/time_shift.md) gives the radius-response calibration and sign conventions.

This example uses a linear trend and the default uniform quadratic limb darkening.
[Trends](../tutorials/systematics.md) also include polynomials, ramps, Gaussian processes and spot crossings.
[Limb darkening](../tutorials/limb_darkening.md) includes quadratic and power-2 laws with fixed, Gaussian, uniform and atmosphere-based priors; check the reference for compatible combinations.

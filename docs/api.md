# Command line and Python

Use the command line for a complete fit.
The YAML file supplies the data, orbital priors, and model choices:

```bash
koala -c examples/niriss_soss_order1.yaml
# Equivalent module form:
python -m koala -c examples/niriss_soss_order1.yaml
koala --help
```

Both forms use the same configuration loader and CPU or GPU startup.
The required option is `-c` or `--config`.
See [configuration](configuration.md) for the settings.

## Read a configuration

If you want to inspect a configuration in Python, use:

```python
from koala.config import load_config, parse_parameter_spec

config = load_config("examples/niriss_soss_order1.yaml")
period = parse_parameter_spec("period", config["planet"]["period"])
print(period.value, period.prior)
```

Loading the YAML alone does not prepare data or run a fit.
It just gives us the settings to inspect.

## Read a spectrum

CSV and NPZ products are the interface for downstream analysis.
For example, we can read the spectrum from the bundled observation with:

```python
from pathlib import Path
import pandas as pd

out = Path("results/WASP-39_SOSS_ORDER1")
spectrum = pd.read_csv(next(out.glob("*_R100.csv")))
print(spectrum[["wavelength", "depth_ppm00", "depth_err_ppm00"]])
```

The [output reference](outputs.md) lists the fields and units.
Model builders and sampler internals are not a versioned public API.

## Reference

| Function | Behavior |
|---|---|
| `load_config(path)` | Returns the YAML mapping. Raises `ValueError` for an empty, scalar, or sequence document and names a complete starter example. |
| `parse_parameter_spec(name, mapping)` | Checks a `{value, prior, ...}` entry and returns its value, prior, and bounds. Raises `ValueError` for an invalid entry. |

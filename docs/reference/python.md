# Running koala from Python

The command line and Python do the same thing.
From a terminal:

```bash
koala -c config.yaml
```

(`python -m koala -c config.yaml` is the same.)
Add `-v` to see everything koala does on the screen, rather than only in `koala.log`.

From Python or a notebook:

```python
import koala

koala.fit("config.yaml")
```

`koala.fit` takes the path to a YAML file, runs the fit, and shows its progress in the notebook.
Call it in a fresh kernel, before anything imports JAX, so that koala can set JAX up to use all of the CPU cores (on a GPU this doesn't matter).
`koala.fit("config.yaml", verbose=True)` prints everything, like `-v`.

## Reading a configuration

To check a configuration without running it:

```python
from koala.config import load_config, parse_parameter_spec

config = load_config("examples/niriss_soss_order1.yaml")
period = parse_parameter_spec("period", config["planet"]["period"])
print(period.value, period.prior)
```

`load_config` returns the YAML as a dictionary, and `parse_parameter_spec` checks one `{value, prior, ...}` entry and raises an error explaining what's wrong if it isn't valid.

## Working with the results

The output files are the way to use koala's results in your own code.
They're all CSV or NumPy files, described in [output files](../guide/outputs.md).
The functions inside koala that build the models and run the samplers aren't a stable interface yet, and may change between versions.

# API reference

The command-line interface is the stable entry point:

```bash
python fit_jwst.py -c config.yaml
```

```{automodule} models.jaxoplanet.builder
:members:
:undoc-members:
```

```{automodule} models.independent_nuts
:members:
```

```{automodule} models.independent_hmc
:members:
```

```{automodule} models.laplace_is
:members:
```

```{automodule} models.sing_ld
:members:
```

```{automodule} models.trends
:members:
```

```{automodule} models.harmonica.builder
:members:
```

## Using the builders

The builders return NumPyro model callables used by the CLI.

The command-line pipeline supplies normalized data, geometry, limb priors, and trend templates.

Direct callers must supply arrays with compatible JAX float64 dtypes.

The builder APIs expose implementation-level controls and can change more readily than the YAML interface.

Use `fit_jwst.py -c` for normal analyses.

## Jaxoplanet builder

`models.jaxoplanet.builder` constructs white-light and vectorized channel models.

It supports quadratic and power-2 limb profiles.

It also selects the stock or optimized light-curve kernel.

The vectorized model retains time-dependent reported errors in the likelihood.

```python
from models.jaxoplanet.builder import create_whitelight_model

model = create_whitelight_model(
    detrend_type="linear",
    ld_mode="fixed",
    ld_profile="quadratic",
)
```

The returned callable still needs the full prior/data arguments documented by its signature.

## Independent NUTS

`models.independent_nuts` creates and caches equal-width lane runners.

It returns posterior samples and diagnostics to the chunk router.

The cache separates static compilation options from dynamic arrays.

```python
from models.independent_nuts import build_independent_nuts_runner

print(build_independent_nuts_runner.__doc__)
```

Use the CLI to obtain fingerprinted checkpoints and automatic quality gates.

## Independent HMC

`models.independent_hmc` implements fixed-step independent HMC.

Its public runner accepts the same channel-factorized model family and Laplace mass option.

```python
from models.independent_hmc import get_samples_independent_hmc

print(get_samples_independent_hmc.__name__)
```

The CLI production alternate fixes eight steps and applies the same depth gate.

## Laplace importance sampling

`models.laplace_is` exposes proposal preparation, adaptive importance rounds, diagnostics, and fallback bookkeeping.

```python
from models.laplace_is import build_laplace_is_runner

print(build_laplace_is_runner.__name__)
```

This is an opt-in approximate backend.

## Sing utilities

The conversion functions are pure NumPy utilities.

```python
from models.sing_ld import quadratic_to_sing, sing_to_quadratic

l, delta = quadratic_to_sing(0.3, 0.2)
u1, u2 = sing_to_quadratic(l, delta)
print(l, delta, u1, u2)
```

`estimate_gray_offset` expects matching `[channel, 2]` arrays.

`write_offset_artifact` creates a new fingerprinted JSON file and refuses to replace an existing one.

## Trend functions

`models.trends` contains deterministic additive light-curve evaluators.

```python
from models.trends import spot_crossing
import jax.numpy as jnp

t = jnp.linspace(0.0, 0.1, 100)
template = spot_crossing(t, amp=0.001, mu=0.05, sigma=0.005)
```

The functions expect the transit and trend keys named by the selected model.

## Harmonica builder

`models.harmonica.builder` constructs white-light and channel transmission-string models.

It supports odd orders 1, 3, and 5 through the configured maximum.

It also supports `delta_r`, fractional, and half-area spectroscopic parameterizations where allowed.

Direct callers must provide consistent `a_rs`, eccentricity, and argument of periastron geometry.

## Import behavior

Set JAX environment variables before importing these modules.

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -c \
  "import models.jaxoplanet.builder, models.independent_nuts, models.independent_hmc"
```

The documentation build mocks heavy dependencies so API signatures can render on Read the Docs.

The production environment must install the real scientific packages.

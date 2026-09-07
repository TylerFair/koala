# API reference

The supported user interface is the command line:

```bash
python fit_jwst.py -c config.yaml
```

Use the [configuration guide](guides/configuration.md) for YAML options and
[reading the results](guides/outputs.md) for output schemas.

The modules below are useful for extending the fitter. They are internal
building blocks rather than a versioned Python library API, so direct callers
must supply normalized arrays, priors, geometry, and compatible JAX float64
dtypes.

## Pipeline package

The root `fit_jwst.py` file is a command-line and import-compatibility façade.
The implementation is organized under `koala`:

- `koala.pipeline`: CLI parsing and white-light → low-resolution →
  high-resolution orchestration.
- `koala.config` and `koala.constants`: YAML/flag resolution and stable schema
  constants.
- `koala.data` and `koala.artifacts`: data preparation, identities, atomic
  writes, and artifact manifests.
- `koala.sampling` and `koala.geometry`: MCMC/chunk orchestration and the
  validated white-light geometry handoff.
- `koala.limb_darkening`: stellar, uniform, and Sing limb-darkening priors.
- `koala.white_light` and `koala.spectroscopy`: the inference stages.
- `koala.outputs`, `koala.harmonica_products`, and `koala.surface`: result
  products and surface/eclipse/phase-curve support.

Existing imports from `fit_jwst` remain available for compatibility. New code
should import a helper from its owning `koala` module.

## Model builders

[models.jaxoplanet.builder](https://github.com/TylerFair/jwst-lightcurves/blob/main/models/jaxoplanet/builder.py): `create_whitelight_model`, `create_vectorized_model`.

[models.harmonica.builder](https://github.com/TylerFair/jwst-lightcurves/blob/main/models/harmonica/builder.py): `create_whitelight_model`, `create_vectorized_model`.

## Sampler backends

[models.independent_nuts](https://github.com/TylerFair/jwst-lightcurves/blob/main/models/independent_nuts.py): `build_independent_nuts_runner`.

[models.independent_hmc](https://github.com/TylerFair/jwst-lightcurves/blob/main/models/independent_hmc.py): `get_samples_independent_hmc`.

`koala.sampling` supplies the pipeline-level chunking, checkpoint, fallback,
and diagnostics orchestration around these backends.


The CLI adds checkpoint fingerprints, convergence gates, retries, and output
tables around these backends. Calling a backend directly does not reproduce
those safeguards.

## Limb-darkening and trend utilities

[models.sing_ld](https://github.com/TylerFair/jwst-lightcurves/blob/main/models/sing_ld.py): `quadratic_to_sing`, `sing_to_quadratic`, `estimate_gray_offset`.

[models.trends](https://github.com/TylerFair/jwst-lightcurves/blob/main/models/trends.py): `spot_crossing`.

`koala.limb_darkening` supplies the pipeline prior builders, including
`get_or_build_power2_ld_prior`. Harmonica product readers and transformations
live in `koala.harmonica_products`.

For example, the deterministic spot template can be evaluated independently:

```python
import jax.numpy as jnp
from models.trends import spot_crossing

t = jnp.linspace(0.0, 0.1, 100)
template = spot_crossing(t, amp=0.001, mu=0.05, sigma=0.005)
```

Set `JAX_ENABLE_X64=1` and the desired `JAX_PLATFORMS` value before importing
JAX or any model module.

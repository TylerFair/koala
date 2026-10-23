# Contributing and testing

Contributions are very welcome, whether that's a bug report, a fix, or a new feature.
If you're planning something big, it's worth opening an issue first so we can talk about it.

## Setting up

Install koala in editable mode, along with the test dependencies, in a fresh environment:

```bash
git clone https://github.com/TylerFair/koala.git
cd koala
python -m pip install -e . pytest pytest-xdist batman-package
```

batman is only used by the tests, as an independent check of koala's transit models.
The exact versions the continuous integration uses are in `tests/requirements-ci.txt`, and you can install against them with `PIP_CONSTRAINT=tests/requirements-ci.txt`.

## Running the tests

Most of the tests run on a CPU:

```bash
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPLBACKEND=Agg
pytest -n auto -m "not slow and not gpu"
```

The `slow` tests take more than about 30 seconds each, and the `gpu` tests need a CUDA device.
To run everything, drop the `-m` option (`pytest -n 8 --dist loadfile` works well on a cluster node).

On GitHub, the tests run on every pull request, split into eight groups with `tests/run_ci.py` so each fits in a small runner.
To reproduce one of those groups locally, including its memory limits, run:

```bash
python tests/run_ci.py --shard 0 --shards 8 --output test-results/shard-0
```

## Changing the numerics

koala is used to measure spectra, so a change to a light-curve model, likelihood or sampler has to be checked carefully, and we'd ask for a few things in a pull request that does that:

- The flux and its gradient should match the existing code, or an independent calculation, to the precision of that calculation (for the transit models that's about $10^{-12}$ in flux, and $10^{-7} + 10^{-6}|g|$ in the gradient $g$).
- The batman comparisons in the test suite should still pass without loosening their tolerances. batman itself is only accurate to about $10^{-9}$ for the quadratic law, which the tests account for.
- For anything that could change a fitted depth, an injection test through the normal `koala` command, with any bias in the depths below $0.1\sigma$.

`tests/reference_quadratic_integration.py` is an independent numerical integration of the quadratic transit, which we use to check both koala and batman where they disagree.

## The documentation

These pages are Markdown files in `docs/`, built with Sphinx.
To build them locally:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W -b html docs docs/_build/html
```

The figures are made by the scripts in `tools/docs/`, which explain how to redraw them.

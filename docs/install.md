# Installation

Use Python 3.10 or later. Run these commands in a terminal:

```bash
git clone https://github.com/TylerFair/koala.git
cd koala
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For NVIDIA GPUs with CUDA 12, install the matching pinned JAX build:

```bash
python -m pip install "jax[cuda12]==0.6.2"
```

Quadratic transits automatically use a CUDA accelerator when the CUDA toolkit's
`nvcc` compiler is on `PATH` (or under `CUDA_HOME` or `CUDA_PATH`) and supports the
device. The first use builds it in `$XDG_CACHE_HOME/koala/quadratic_cuda`, or
`~/.cache/koala/quadratic_cuda` when `XDG_CACHE_HOME` is unset. Later fits reuse
that build. Package installation does not run a compiler.

If CUDA compilation is unavailable, koala uses the JAX closed form. It prints
which implementation is selected. CPU evaluation and higher derivatives use
JAX; the CUDA accelerator uses the same flux, derivatives and likelihood.
With the accelerator, every trend that is linear in its coefficients (up to eight
terms) is fitted in the same CUDA pass as the transit, and spectroscopic channels
with a free epoch use it at any light-curve length. Integrations outside the
transit window then enter through exact summary statistics, as they already do on
any device for light curves longer than 5000 integrations.

Check the installation:

```bash
python -c "from importlib.metadata import version; print('koala', version('koala-jwst'))"
```

You're ready for [your first fit](quickstart.md).

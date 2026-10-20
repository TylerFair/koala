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

Check the installation:

```bash
python -c "from importlib.metadata import version; print('koala', version('koala-jwst'))"
```

You're ready for [your first fit](quickstart.md).

# Installation

koala needs Python 3.10 or newer, Git, and a C++ compiler.
The compiler is for [Harmonica](https://github.com/DavoGrant/harmonica), one of koala's dependencies, which is built when you install.
Linux systems usually have one already.
On a Mac, run `xcode-select --install` first if you haven't before.

We recommend installing koala into a fresh environment.
With conda, that looks like:

```bash
conda create -n koala python=3.11
conda activate koala
git clone https://github.com/TylerFair/koala.git
cd koala
python -m pip install .
```

(If you don't use conda, `python -m venv .venv` followed by `source .venv/bin/activate` works just as well.)
The install takes a couple of minutes, most of which is spent compiling Harmonica.
Cloning the repository also gets you the example data and configurations used in these docs.

To check that it worked, run:

```bash
koala --help
```

That's all you need to fit on a CPU, and you're ready for [your first fit](quickstart.md).

## Running on a GPU

koala uses an NVIDIA GPU automatically if JAX can see one.
To give JAX that ability, install its CUDA build in the same environment:

```bash
python -m pip install "jax[cuda12]==0.6.2"
```

Then check that JAX finds the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

This should list a `CudaDevice`.
On a cluster, run that check (and your fits) on a GPU node rather than the login node.
If you have the CUDA compiler `nvcc` available, koala also compiles a small CUDA kernel for quadratic limb-darkened transits the first time it needs one, and keeps it in `~/.cache/koala`.
Without `nvcc` the GPU fit uses plain JAX, which gives the same answer a little more slowly.

## Stellar-model files

koala uses [ExoTiC-LD](https://exotic-ld.readthedocs.io) to get limb darkening from stellar models.
The first fit for a given star downloads the stellar-model and instrument-throughput files it needs (only a few MB for the first-fit example) into `~/.cache/koala/exotic_ld_data`, and later fits reuse them.
If your compute nodes can't reach the internet, run one fit somewhere that can and copy the files across (see [troubleshooting](guide/troubleshooting.md)).

koala is tested on Linux with Python 3.10 and 3.11.

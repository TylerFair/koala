# Installation

Let's install koala in a separate Python environment.
You'll need Python 3.10 or later (Python 3.11 is also tested).
Cloning the repository gives you the example configurations and bundled data:

```bash
git clone https://github.com/TylerFair/koala.git
cd koala
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
koala --help
```

The distribution is called `koala-jwst`.
Installation needs Git and network access.
koala itself is pure Python, but its Harmonica dependency builds a C++ extension, so you'll also need a C++ compiler.
The requirements install JAX, NumPyro, jaxoplanet, ExoTiC-LD, tinygp, and the [Harmonica fork](https://github.com/TylerFair/harmonica).
Harmonica is pinned to commit `c8e9233bfc99ba55edbffc0dcc4e562b163a8d21` and builds with its Eigen and pybind11 submodules.

## NVIDIA GPUs

Follow the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) to choose a CUDA wheel for your driver.
For a compatible CUDA 12 system, use:

```bash
python -m pip install --upgrade "jax[cuda12]"
python -c "import jax; print(jax.devices())"
```

Then set `host_device: gpu` in your YAML file.
On a cluster, run the device check inside a GPU allocation.
Native Windows does not provide CUDA JAX wheels, so use a supported Linux environment such as WSL2.
Harmonica evaluates koala's quadratic and power-2 transit models with JAX on CPUs and GPUs.
It doesn't need a separate CUDA extension build.

## Limb-darkening data

The bundled example downloads atmosphere and throughput files on first use.
These settings choose the stellar grid and its cache directory:

```yaml
stellar:
  ld_model: stagger
  ld_data_path: exotic_ld_data
```

If your compute node has no network access, populate the ExoTiC-LD cache on a connected machine first.
Then copy it to the cluster and point `ld_data_path` to that directory.
Explicit `stellar.ld_coefficients` with `ld_prior: fixed` don't need atmosphere interpolation.

## Clusters

Submit fits through the scheduler.
Keep the compilation cache and output files on a filesystem visible to the allocated node.
You can resume an interrupted fit with the same command and YAML file.
If a GPU runs out of memory, reduce `flags.spectro_chunk_size`.

Now you're ready to [fit the bundled observation](quickstart.md).

# Harmonica modified for Koala

This directory contains the Harmonica source used by Koala, including its
quadratic and power-2 JAX models and optional CUDA kernel. It is installed by
the repository-root package build:

```bash
python -m pip install -e .
```

The Python import remains `harmonica`; the root distribution is `koala-jwst`.
The standalone build in this directory retains the `planet-harmonica` name
for compatibility, but is not needed when installing Koala. The local source directory is named
`harmonica_modified`.

A C++ compiler is required. Pip supplies setuptools and pybind11 in its isolated
build environment. Eigen headers are included, so no Git submodule checkout is
needed. For CUDA builds, install the CUDA toolkit and set
`HARMONICA_ENABLE_CUDA=1` before installation. CPU builds are the default.

## Source and licenses

- Harmonica fork: https://github.com/TylerFair/harmonica_bell-main
- Vendored revision: `5ba99b99f7bbf4af47ce0a7f205e935304245de9`
- Original project: https://github.com/DavoGrant/harmonica
- Harmonica license: [MIT](LICENSE)
- Eigen source: https://gitlab.com/libeigen/eigen
- Eigen version: 3.4.0
- Eigen revision: `3147391d946bb4b6c68edd901f2add6ac1f31f8c`
- Eigen licenses: [COPYING.README](vendor/eigen/COPYING.README) and accompanying
  `COPYING.*` files. Only the `Eigen/` header tree needed by the extension is
  included.

Koala vendors the runtime Python/C++/CUDA sources and build inputs. Upstream
experiment outputs, old implementations, and build artifacts are excluded.
Local packaging changes remove developer-specific import paths and include
headers and CUDA sources in source distributions. The incomplete Eigen copy
in the fork is replaced by the complete Eigen 3.4.0 header tree.

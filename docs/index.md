# {{ project }}

{{ project }} fits JWST transit light curves from extracted box-spectrum FITS files. It fits a white-light curve first, passes the orbital geometry to wavelength channels, and writes light-curve diagnostics and transmission spectra.

## How it works

The white-light fit measures the shared transit geometry and visit-level systematics at high signal-to-noise. Spectroscopic fits then hold the posterior-median geometry fixed and infer one radius ratio, limb profile, jitter term, and trend per wavelength channel. Channels run in GPU-resident chunks, and every exact posterior must pass an effective-sample-size and divergence gate before it is accepted. Read [How the fit works](concepts.md) before choosing a limb prior or sampler.

```bash
conda activate jaxoplanet
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=gpu
cp configs_fiducial_stellarinformed/WASP-39_soss_order1_config.yaml config.yaml
# Edit path, input_dir, output_dir, and fits_file in config.yaml.
python fit_jwst.py -c config.yaml
ls /path/from/config/output_dir
python -c "import pandas as pd; print(pd.read_csv('/path/to/spectrum.csv').head())"
```

Features include:

- JWST NIRISS/SOSS orders 1 and 2.
- NIRSpec G395H, G395M, G140H, G235H, and PRISM, including NRS1/NRS2 selection.
- White-light, low-resolution, and high-resolution spectroscopic stages.
- Stellar-informed, Sing, fixed, wide-Gaussian, and free limb darkening.
- Polynomial, spot, discontinuity, exponential-ramp, and GP systematics.
- Laplace-metric HMC samplers implemented in JAX for GPUs.
- Harmonica transmission-string fits for limb asymmetry.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
concepts
faq
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/soss_order1
tutorials/nirspec_g395h
tutorials/prism
tutorials/harmonica
tutorials/executed_notebooks
tutorials/synthetic_surfaces
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/limb_darkening
guides/phase_curves
guides/trends
guides/samplers
guides/configuration
guides/outputs
guides/gpu_and_clusters
guides/model_stacking
guides/loop_mode
api
citing
```

## Choose a starting point

New users should run the [quickstart](quickstart.md), then read [How the fit works](concepts.md). SOSS users can begin with the [order-1 tutorial](tutorials/soss_order1.md). NIRSpec users can begin with [G395H](tutorials/nirspec_g395h.md) or [PRISM](tutorials/prism.md).

Asymmetric ingress/egress analyses should begin with the [Harmonica tutorial](tutorials/harmonica.md).

## Analysis sequence

1. Prepare a box-spectrum FITS extraction.

2. Choose the detector or SOSS order.

3. Choose low- and high-resolution wavelength grids.

4. Choose the limb-darkening law and prior.

5. Select the simplest trend supported by the white-light baseline.

6. Fit white light and inspect its residuals.

7. Run any required low-resolution calibration stage.

8. Sample high-resolution channels in checkpointed chunks.

9. Check ESS and divergences.

10. Read the accepted depth posteriors from the spectrum CSV.

## Supported observing modes

NIRISS/SOSS is selected with `instrument: NIRISS/SOSS` and an `order`. NIRSpec modes are selected with their full instrument string and `nrs` detector number. G395H and G395M cover the long-wavelength NIRSpec detectors.

G140H and G235H use the same detector-aware configuration structure. PRISM supports native or constant-resolving-power channel grids. NRS1 and NRS2 are fitted separately.

SOSS orders 1 and 2 are fitted separately.

## Principal products

Every run writes white-light diagnostic plots and a time-series table. Spectroscopic stages write one transmission-spectrum CSV per resolution. Detailed tables include radius ratio, depth, limb darkening, jitter, and active trend parameters.

Chunk checkpoints make long GPU runs resumable. JSON diagnostics record numerical quality. Harmonica adds limb spectra, transmission-string figures, and joint limb posterior arrays.

## Configuration philosophy

The YAML is the analysis record. Keep target values, stellar values, extraction paths, model choices, and sampler controls together. Use a new output directory for a scientifically distinct configuration.

Set a random seed explicitly for a published analysis. Archive the geometry handoff and diagnostics beside the final spectrum.

## Glossary

**White light:** the wavelength-summed transit time series. **Channel:** one wavelength-bin light curve. **Chunk:** a group of channels evaluated by one compiled sampler call.

**Lane:** one independent channel position inside a vectorized chunk. **Bridge stage:** the optional low-resolution spectroscopic fit. **Geometry handoff:** fixed orbital quantities selected from white light.

**Laplace metric:** local inverse-Hessian scaling used by NUTS or HMC. **Gate:** the ESS and divergence criteria required before accepting a posterior. **Reference grid:** an externally supplied wavelength grid used for binning.

**Transmission string:** Harmonica's angle-dependent planet radius boundary.

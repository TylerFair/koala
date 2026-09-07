# Examples

For self-contained examples beyond transits, run
`python tools/surface_publication_example.py all`. The
[executed surface tutorial](../docs/tutorials/synthetic_surfaces.md) shows
synthetic eclipse and phase-curve spectra, a thermal map with uncertainty,
stellar-spot recovery, corner plots, and light-curve residual diagnostics.
Each analysis saves PNG/PDF figures, posterior samples, and numerical tables.

Begin with `niriss_soss_order1.yaml`. It is the smallest complete configuration and uses numeric $R=100$ bins, so it does not depend on a separate wavelength-grid file:

```bash
cp examples/niriss_soss_order1.yaml my_transit.yaml
# edit the data paths and target parameters
python fit_jwst.py -c my_transit.yaml
```

The repository does not bundle the extracted FITS data or ExoTiC-LD atmosphere grids. Point `path`, `input_dir`, `fits_file`, and `stellar.ld_data_path` at your local copies before running.

| Example | Use it for |
|---|---|
| `niriss_soss_order1.yaml` | A first SOSS order-1 transit with a linear trend |
| `nirspec_g395m.yaml` | NIRSpec detector selection and masking a cadence interval |
| `nirspec_g395h.yaml` | A G395H/NRS1 fit with a quadratic baseline |
| `nirspec_prism.yaml` | Native-grid PRISM with an exponential ramp and small GPU batches |
| `harmonica_soss_order1.yaml` | Advanced asymmetric-ingress/egress fitting |
| `limb_darkening_stack.yaml` | Repeating one fit under several limb-darkening choices |
| `eclipse.yaml` | A generated secondary-eclipse injection and recovery |
| `wasp39_eclipse_nrs1.yaml`, `wasp39_eclipse_nrs2.yaml` | The worked WASP-39 b G395H eclipse at $R=300$ |
| `phase_curve.yaml` | A generated full-orbit thermal phase curve |
| `stellar_spots.yaml` | A generated transit with physical rotating Starry spots |
| `plot_spectrum.py` | Reading and plotting a spectrum CSV |
| `niriss_soss_order1.ipynb` | Saved end-to-end SOSS walkthrough |
| `nirspec_g395m.ipynb` | Saved end-to-end G395M walkthrough |

The YAML files set the scientific and dataset-specific choices. They leave most sampler settings at validated defaults. Copy the closest example and change only the fields your dataset or analysis requires.

The [first-transit tutorial](../docs/tutorials/soss_order1.md) explains the complete workflow. The focused tutorials cover [trend choice](../docs/guides/trends.md), [limb darkening](../docs/guides/limb_darkening.md), [eclipses, phase curves, and stellar spots](../docs/guides/phase_curves.md), and [model stacking](../docs/guides/model_stacking.md).

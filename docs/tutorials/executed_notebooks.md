# Executed notebooks

Two saved notebooks show a complete fit, output-table inspection, and plotting:

- {download}`NIRISS/SOSS order 1 <../../examples/niriss_soss_order1.ipynb>`
- {download}`NIRSpec/G395M NRS1 <../../examples/nirspec_g395m.ipynb>`

They include their saved output, so you can read the workflow without installing Koala or obtaining the original FITS files. To rerun one, set `PENUMBRA_EXAMPLE_DATA_ROOT` to the directory containing the extracted FITS file and `PENUMBRA_NOTEBOOK_RESULT_ROOT` to a new results directory. The SOSS notebook defaults to the compact file in `examples/data/`.

The notebooks use reduced tutorial sampling and $R=20$ spectra to keep the walkthrough manageable. Use the adjacent YAML files and the standard convergence settings for a scientific analysis.

# Harmonica boundary and limb products

Harmonica fits a planetary boundary whose halves can differ.
See the [limb-spectrum case study](../case_studies/harmonica.md) for a runnable configuration and an archived WASP-94 b figure.
It accepts `duration` or `a_rs` geometry and supports one planet.
Power-2 and supported quadratic treatments include fixed, uniform and fixed-$u_1$ coefficients.
The [configuration reference](../configuration.md#harmonica-flags) lists boundary orders, parameterizations and priors.

## Read the two limbs

The ordinary spectrum CSV reports the total area-equivalent transit depth.
The file `*_limb_spectra.csv` adds two indexed halves and boundary summaries.
The matching `*_limb_posterior_samples.npz` keeps their paired posterior draws.
The [output reference](../outputs.md#harmonica) distinguishes half-area depths, endpoint depths and the total silhouette area, and gives their units.

The halves follow the boundary convention stored in the product.
Assigning morning and evening requires the orbital orientation.
Use paired draws when computing a limb difference, since the two depths are correlated.
Subtracting independent error bars loses that information.

# Harmonica limb asymmetry

The WASP-94 b SOSS order-1 configuration supplies the required geometry and uses a first-order transmission string:

```bash
cp configs_harmonica/WASP-94_soss_order1_config.yaml wasp94_harmonica.yaml
python fit_jwst.py -c wasp94_harmonica.yaml
```

Use these sampler and shape settings:

```yaml
flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: delta_r
  harmonica_spectro_odd_frac_sigma: 0.1
  spectro_sampler: independent_nuts
  spectro_mass_matrix: laplace
  ld_profile: quadratic
  ld_prior: fixed
```

`delta_r` samples the radius contrast represented by the odd transmission-string coefficient. Order 1 fits `a0` and `a1`; orders 3 and 5 can add `a3` and `a5`.

After the fit, inspect `*_limb_spectra.csv`, `*_limb_spectra.png`, `*_transmission_strings.png`, and `*_transmission_string_posterior.png`. The CSV reports area-equivalent evening/leading and morning/trailing depths, endpoint depths, `a0`, active odd coefficients, and their percentile errors. The accompanying `*_limb_posterior_samples.npz` preserves their draw-by-draw covariance.


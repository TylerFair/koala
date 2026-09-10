# Fit limb asymmetry with Harmonica

The standard transit model gives the planet one radius at each wavelength.
The Harmonica engine instead fits an asymmetric boundary, so ingress and
egress constrain separate spectra for the two halves of the terminator.

## Configuration

Start from `examples/harmonica_soss_order1.yaml` (WASP-94 b, NIRISS/SOSS
order 1). The settings that select the model are:

```yaml
planet:
  a_rs: {value: 7.299998525, prior: log_uniform, low: 2.0, high: 100.0}
  ecc: {value: 0.0, prior: fixed}
  omega: {value: 0.0, prior: fixed}

flags:
  transit_engine: harmonica
  harmonica_max_order: 1
  harmonica_spectro_parameterization: delta_r
  harmonica_spectro_odd_frac_sigma: 0.1
  ld_profile: quadratic
  ld_prior: fixed
```

`harmonica_max_order: 1` fits the mean radius and one asymmetry term; the
`delta_r` parameterization samples the radius difference between the two
limbs directly. Harmonica samples `a_rs` (not `duration`, which is then
`fixed` and only used to locate the transit window) and works with one
planet and either limb-darkening law.

## Run

```bash
python fit_jwst.py -c examples/harmonica_soss_order1.yaml
```

## Result

```{image} ../_static/harmonica_wasp94_whitelight.png
:alt: WASP-94 b Harmonica white-light fit
:width: 760px
:align: center
```

```{image} ../_static/harmonica_wasp94_limb_spectra_v3.png
:alt: Representative spectra for the two terminator halves of WASP-94 b
:width: 760px
:align: center
```

The two limb spectra are written to `*_limb_spectra.csv`, with correlated
posterior draws in `*_limb_posterior_samples.npz`; the ordinary spectrum CSV
still holds the total area-equivalent depth. Koala labels the halves
terminator one and two; mapping them to morning and evening limbs depends on
your orbital convention. Because the two depths are correlated, compute limb
differences from the joint draws rather than from the marginal errors.

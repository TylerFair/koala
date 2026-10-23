# Troubleshooting

## The fit didn't converge

koala won't accept a posterior until its chains agree.
For every free white-light parameter, and for the depth in every channel, it asks for a rank-normalized $\hat{R}$ below 1.01, at least 400 effective samples summed over the four chains, and no divergent transitions.
A channel that falls short first gets more draws, and then a second try with a different sampler.
If it still fails, koala tells you which channels at the end of the run, and the details are in `koala.log` and the `state/chunks/*.diagnostics.json` files.

The usual causes are:

- The model doesn't describe the data, for example a trend that's too simple or a bad stretch of integrations. Look at the residuals.
- A parameter the data can't constrain, like the time of a shallow eclipse (see the [eclipse case study](../case_studies/rocky_eclipse.md)).
- Limb-darkening coefficients or trend coefficients pushed up against the edge of their prior.

Lowering the thresholds (`spectro_min_depth_ess`, `spectro_max_divergences`) will make the message go away, but it won't fix the fit.

## The transit or eclipse isn't where the model is

Start with `figures/whitelight_start.png` in the output directory, which shows the starting model on top of the data.
If the model's transit or eclipse is nowhere near the one in the data, the time system is the usual culprit.
koala works in BJD_TDB − 2400000, and times quoted as BMJD_TDB (BJD_TDB − 2400000.5) are half a day off from that.
koala converts the times in the data files, but never the times in your YAML.

## Installation problems

Installing needs Git and internet access, so check that Git can reach GitHub.
If Harmonica fails to build, check that `g++` or `clang++` works.
A fresh environment fixes most other problems, especially after changing the versions of JAX or NumPy.

If JAX doesn't see your GPU, check `python -c "import jax; print(jax.devices())"` on the GPU node itself, and that you installed `jax[cuda12]` (see [installation](../install.md)).

In a notebook, `koala.fit` has to set up JAX before anything else does.
If you've already imported JAX (or something that imports it), restart the kernel and call `koala.fit` first.

## The compute nodes are offline

The stellar-model files for limb darkening are downloaded the first time they're needed.
If your compute nodes have no internet access, run a fit once on a machine that does (a short white-light fit with `flags.analysis_stage: whitelight` is enough), then copy `~/.cache/koala/exotic_ld_data` (or wherever `stellar.ld_data_path` points) across, or point `ld_data_path` at a shared filesystem.

## Other things that come up

| What you see | What to try |
|---|---|
| An error that mentions a 0.5 d offset | The YAML times are in BMJD_TDB rather than BJD_TDB − 2400000. Add 0.5 d. |
| A GPU runs out of memory | Lower `flags.spectro_chunk_size` so fewer channels are fitted at once. |
| The first batch of channels is much slower than the rest | That's JAX compiling the model. Later batches, and later runs, reuse it. |
| A warning that a key in `flags` is unknown | It's probably misspelled. The warning suggests the closest name. |
| A rerun starts some of the work again | The data or settings changed, so koala won't reuse the old results. Use a new `output_dir` for a different analysis. |
| The depth changes a lot when you add a ramp | The ramp is trading off against the signal. Compare simpler trends, or leave out the start of the observation. |

If none of this helps, please [open an issue](https://github.com/TylerFair/koala/issues) with your YAML file, `koala.log`, and the versions of koala, JAX and NumPyro.

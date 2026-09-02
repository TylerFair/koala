# Model stacking for transmission spectra

A transmission spectrum can depend on reasonable choices made in the light-curve model: the limb-darkening prior, the baseline trend, or whether an instrumental step is included. Model stacking carries that uncertainty into the reported depth instead of selecting one assumption and treating it as known.

Run the same wavelength bins under every candidate model, using identical sampler settings. For each posterior draw, evaluate the Normal log likelihood separately for every cadence. PSIS-LOO converts those values into an out-of-sample log predictive density for each cadence. In each wavelength channel, stacking chooses non-negative weights summing to one that maximize

\[
\sum_i \log\left(\sum_m w_m p_m(y_i\mid y_{-i})\right).
\]

The final depth posterior is a mixture: choose a model using its channel weight, then choose a depth draw from that model. Its interval naturally becomes wider when well-supported models disagree. The output `disagreement` is the stacked 68-percent half-width divided by the narrowest single-model half-width. Values above one identify model-sensitive channels.

The predictive unit is one light-curve point, because the intended task is predicting another cadence from the same wavelength channel. This assumes residuals are conditionally independent. Time-correlated residuals require a block or grouped LOO scheme; pointwise LOO would otherwise be optimistic.

Pseudo-BMA+ is a cheaper comparison. It exponentiates summed LOO scores and uses a Bayesian bootstrap over cadences to reduce overconfident weights. It can still collapse onto one model and does not optimize mixture predictions directly. Laplace BMA uses approximate marginal likelihoods and answers a different question: posterior probability of each complete model, including its prior volume. In this prototype it is restricted to the white-light fit and should not be presented as channel-specific evidence.

Use `tools/stacking/run_matrix.py` to create variant configs and queue scripts. After all fits finish, use `tools/stacking/stack_spectra.py` to make the pointwise likelihood archives, diagnostics, stacked CSV, and three-panel figure. Always inspect Pareto k. A channel with any important point above 0.7 needs more robust LOO (moment matching, exact refits, or a grouped predictive task) before its weights are trusted.

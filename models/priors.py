"""NumPyro sampling of ``koala.config.ParameterSpec`` planet parameters.

Every planet parameter shares one vocabulary (``fixed``, ``uniform``,
``log_uniform``, ``gaussian`` with optional truncation). This module turns a
specification into the NumPyro site(s) for it so the JAXoplanet and Harmonica
white-light builders, and the surface (emission) priors, all behave the same
way.

Site naming: a fixed parameter becomes a deterministic site ``<site>``; a
uniform or gaussian prior samples ``<site>`` directly; a log-uniform prior
samples ``log_<site>`` and exposes ``<site>`` as a deterministic. The legacy
names ``logD_<i>`` (duration) and ``log_a_rs_<i>`` are kept for those two
parameters so downstream readers keep working.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


_LOG_SITE_NAMES = {'duration': 'logD', 'a_rs': 'log_a_rs'}


def log_site_name(site):
    """Name of the log-space latent site for ``site`` (``'duration_0'`` etc.)."""
    stem, _, index = site.rpartition('_')
    if stem in _LOG_SITE_NAMES and index.isdigit():
        return f"{_LOG_SITE_NAMES[stem]}_{index}"
    return f"log_{site}"


def spec_distribution(spec):
    """NumPyro distribution for a free specification (never log-uniform)."""
    if spec.prior == 'uniform':
        return dist.Uniform(float(spec.low), float(spec.high))
    if spec.prior == 'gaussian':
        if spec.low is None and spec.high is None:
            return dist.Normal(float(spec.value), float(spec.sigma))
        return dist.TruncatedNormal(
            float(spec.value), float(spec.sigma),
            low=None if spec.low is None else float(spec.low),
            high=None if spec.high is None else float(spec.high),
        )
    raise ValueError(
        f"planet.{spec.name}: prior {spec.prior!r} has no direct distribution."
    )


def sample_parameter(spec, site, shape=()):
    """Sample or fix one parameter and return its value at ``site``.

    ``shape`` expands the prior (for per-channel spectroscopic sites).
    """
    shape = tuple(shape)
    if spec.fixed:
        value = jnp.full(shape, float(spec.value), dtype=jnp.float64) if shape \
            else jnp.asarray(float(spec.value), dtype=jnp.float64)
        return numpyro.deterministic(site, value)
    if spec.prior == 'log_uniform':
        log_value = numpyro.sample(
            log_site_name(site),
            dist.Uniform(jnp.log(float(spec.low)), jnp.log(float(spec.high))).expand(shape),
        )
        return numpyro.deterministic(site, jnp.exp(log_value))
    return numpyro.sample(site, spec_distribution(spec).expand(shape))


def latent_init_site(spec, site):
    """``(name, value)`` of the latent site to initialise, or ``None`` if fixed."""
    if spec.fixed:
        return None
    if spec.prior == 'log_uniform':
        return log_site_name(site), jnp.log(float(spec.value))
    return site, jnp.asarray(float(spec.value), dtype=jnp.float64)

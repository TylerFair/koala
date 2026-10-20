"""Geometry helpers."""

import os
import json
import hashlib
import uuid
import re
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from models.mcmc_diagnostics import PosteriorSamples, pool_chains, rank_diagnostics
from .constants import (
    WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION,
    WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION,
)
from .artifacts import _update_checkpoint_hash


def _whitelight_science_sites(parameter_specs, surface_config, param_method):
    """Map diagnostic labels to free orbital and physical surface sites."""
    from models.priors import latent_init_site
    from models.jaxoplanet.config import surface_science_sites

    sites = {}
    names = ('period', 't0', 'b', 'rprs',
             'duration' if param_method == 'duration' else 'a_rs')
    for name in names:
        for index, spec in enumerate(parameter_specs[name]):
            stem = 'rors' if name == 'rprs' else name
            init = latent_init_site(spec, f'{stem}_{index}')
            if init is not None:
                label = stem if index == 0 else f'{stem}_{index}'
                sites[label] = init[0]
    sites.update({name.removeprefix('_'): name for name in surface_science_sites(surface_config)})
    return sites


def _geometry_chain_quality(grouped_samples, science_sites=None):
    """Check free science parameters; constant draws never establish fixed status.

    Production passes the configured sites. Legacy callers without metadata
    conservatively check every recognizable science site present in the draws.
    """
    aliases = {
        "period": ("period_0", "period"),
        "t0": ("t0_0", "t0"),
        "b": ("b_0", "b"),
        "duration": ("logD_0", "duration_0", "duration", "logD"),
        "rors": ("rors_0", "rors"),
    }
    if "_geometry_fixed" in grouped_samples:
        aliases = {}
    for name in grouped_samples:
        if re.fullmatch(
            r"_(?:eclipse_depth|phase_c1|phase_d1|stellar_spot_contrast)_\d+",
            name,
        ):
            aliases[name.removeprefix("_")] = (name,)
    if science_sites is not None:
        aliases = {label: (name,) for label, name in science_sites.items()}
    ess = {}
    rhat = {}
    for label, names in aliases.items():
        name = next((candidate for candidate in names if candidate in grouped_samples), None)
        if name is None:
            if science_sites is not None:
                ess[label] = np.nan
            continue
        values = jnp.asarray(grouped_samples[name], dtype=jnp.float64)
        if values.ndim < 2:
            values = values.reshape((1, values.shape[0]))
        if bool(jnp.all(jnp.isfinite(values))) and bool(
            jnp.all(values == values[0, 0])
        ):
            ess[label] = 0.0
            continue
        if values.shape[0] > 1:
            metrics = rank_diagnostics({label: values})
            ess[label] = float(np.min(metrics['bulk_ess'][label]))
            rhat[label] = float(np.max(metrics['rhat'][label]))
            continue
        site_ess = numpyro.diagnostics.effective_sample_size(values)
        ess_values = np.asarray(jax.device_get(site_ess), dtype=float)
        ess[label] = (
            float(np.min(ess_values))
            if np.all(np.isfinite(ess_values)) else np.nan
        )
        if values.shape[0] > 1:
            site_rhat = numpyro.diagnostics.split_gelman_rubin(values)
            rhat_values = np.asarray(jax.device_get(site_rhat), dtype=float)
            rhat[label] = (
                float(np.max(rhat_values))
                if np.all(np.isfinite(rhat_values)) else np.nan
            )
    return ess, rhat


def _continue_mcmc_until_geometry_gate(
    mcmc, rng_key, run_args, run_kwargs, *, min_ess=400.0, max_divergences=0,
    max_extra_blocks=3, failfast_ess=50.0, science_sites=None,
):
    """Append post-warmup draws until the white-light geometry gate passes."""
    sample_blocks = [jax.device_get(mcmc.get_samples(group_by_chain=True))]
    extra_blocks = [jax.device_get(mcmc.get_extra_fields(group_by_chain=True))]
    extra_count = 0
    quality_blocks = []
    while True:
        grouped = {
            name: np.concatenate([np.asarray(block[name]) for block in sample_blocks], axis=1)
            for name in sample_blocks[0]
        }
        grouped_extra = {
            name: np.concatenate([np.asarray(block[name]) for block in extra_blocks], axis=1)
            for name in extra_blocks[0]
        }
        ess, rhat = (
            _geometry_chain_quality(grouped) if science_sites is None
            else _geometry_chain_quality(grouped, science_sites=science_sites)
        )
        chains = int(next(iter(grouped.values())).shape[0])
        tail_ess = {}
        if chains > 1:
            if science_sites is not None:
                selected = {label: grouped[name] for label, name in science_sites.items() if name in grouped}
            else:
                selected = {name: value for name, value in grouped.items()
                            if name in {'t0_0','b_0','rors_0','logD_0','duration_0'}}
            metrics = rank_diagnostics(selected)
            tail_ess = {name: float(np.min(value)) for name, value in metrics['tail_ess'].items()}
        divergences = int(np.sum(np.asarray(grouped_extra.get("diverging", []))))
        no_free_science = science_sites is not None and not science_sites
        finite_ess = bool(ess) and bool(np.all(np.isfinite(list(ess.values()))))
        passes = no_free_science or (finite_ess and min(ess.values()) >= float(min_ess))
        if chains > 1 and not no_free_science:
            passes = passes and bool(rhat) and all(np.isfinite(value) and value < 1.01 for value in rhat.values())
            passes = passes and bool(tail_ess) and all(np.isfinite(value) and value >= float(min_ess) for value in tail_ess.values())
        passes = passes and divergences <= int(max_divergences)
        print(
            "White-light quality gate: "
            f"bulk_ESS={ess}, tail_ESS={tail_ess or 'n/a'}, rhat={rhat or 'n/a'}, divergences={divergences}, "
            f"pass={passes}, extra_blocks={extra_count}."
        )
        quality_blocks.append({
            "block": extra_count, "bulk_ess": ess, "tail_ess": tail_ess,
            "rhat": rhat, "divergences": divergences, "passed": passes,
            "draws_per_chain": int(next(iter(grouped.values())).shape[1]),
        })
        failfast = bool(
            extra_count == 0 and ess and (min(ess.values()) < float(failfast_ess)),
        )
        if failfast:
            print(
                "WHITE-LIGHT QUALITY GATE FAIL-FAST: first-block minimum "
                f"science ESS is below {float(failfast_ess):g}; skipping "
                "extension blocks."
            )
        # Divergences accumulate over blocks, so more draws from the same
        # chain cannot bring the count back under the limit.
        divergence_stop = divergences > int(max_divergences)
        if divergence_stop:
            print(
                f"White-light chain has {divergences} divergent transition(s); "
                "extension blocks cannot remove them, so none are drawn."
            )
        if passes or failfast or divergence_stop or extra_count >= int(max_extra_blocks):
            flat = PosteriorSamples(pool_chains(grouped), num_chains=chains)
            flat_extra = pool_chains(grouped_extra)
            return flat, grouped, flat_extra, {
                "whitelight_quality_blocks": quality_blocks,
                "whitelight_geometry_bulk_ess": ess,
                "whitelight_geometry_tail_ess": tail_ess,
                "num_chains": chains,
                "draws_per_chain": int(next(iter(grouped.values())).shape[1]),
                "whitelight_geometry_rhat": rhat,
                "whitelight_quality_gate_passed": passes,
                "whitelight_failfast_triggered": failfast,
                "whitelight_divergences": divergences,
                "whitelight_divergence_stop": divergence_stop,
                "whitelight_extra_blocks": extra_count,
                "whitelight_total_retained_draws": int(
                    next(iter(flat.values())).shape[0]
                ),
            }
        extra_count += 1
        print(
            "WHITE-LIGHT QUALITY GATE NOT MET; continuing the same warmed "
            f"chain with retained block {extra_count}/{max_extra_blocks}."
        )
        mcmc.post_warmup_state = mcmc.last_state
        rng_key, block_key = jax.random.split(rng_key)
        continuation_kwargs = dict(run_kwargs)
        continuation_kwargs.pop("init_params", None)
        mcmc.run(block_key, *run_args, **continuation_kwargs)
        sample_blocks.append(jax.device_get(mcmc.get_samples(group_by_chain=True)))
        extra_blocks.append(jax.device_get(mcmc.get_extra_fields(group_by_chain=True)))


def _posterior_num_draws(posterior_samples):
    """Validate a flattened posterior sample mapping and return its draw count."""
    if not posterior_samples:
        raise ValueError("Posterior samples are empty.")
    draw_counts = {}
    for name, values in posterior_samples.items():
        array = np.asarray(jax.device_get(values))
        if array.ndim == 0:
            raise ValueError(
                f"Posterior site '{name}' is missing a leading draw dimension."
            )
        draw_counts[name] = int(array.shape[0])
    unique_counts = set(draw_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            f'Posterior sites have inconsistent draw counts: {draw_counts}.',
        )
    num_draws = unique_counts.pop()
    if num_draws < 1:
        raise ValueError("Posterior samples contain no retained draws.")
    return num_draws


def _geometry_from_white_light_medians(bestfit_params, period):
    """Build the compatibility handoff from the reported posterior medians."""
    periods = np.atleast_1d(np.asarray(period, dtype=np.float64))
    geometry = {"period": periods.tolist()}
    geometry.update({
        name: np.atleast_1d(np.asarray(bestfit_params[name], dtype=float)).tolist()
        for name in ('duration', 't0', 'b', 'a_rs', 'cos_i', 'rors')
    })
    geometry["inclination"] = np.arccos(
        np.clip(np.asarray(geometry["cos_i"], dtype=float), 0.0, 1.0)
    ).tolist()
    lengths = {name: len(values) for name, values in geometry.items()}
    if set(lengths.values()) != {len(periods)}:
        raise RuntimeError(
            f'White-light median geometry has inconsistent planet dimensions: {lengths}.',
        )
    if not all(np.all(np.isfinite(values)) for values in geometry.values()):
        raise RuntimeError("White-light median geometry contains non-finite values.")
    return geometry


def _selected_geometry_primitives(samples, num_planets):
    """Serialize the primitive coordinates that identify a selected draw."""
    names = []
    for planet_index in range(int(num_planets)):
        names.extend(
            f"{base}_{planet_index}"
            for base in (
                "period", "t0", "rors", "_b", "logD", "log_a_rs", "duration",
                "b", "a_rs", "cos_i", "inc",
            )
        )
    primitives = {}
    for name in names:
        if name not in samples:
            continue
        value = np.asarray(jax.device_get(samples[name]))
        primitives[name] = (
            float(value.reshape(-1)[0])
            if value.size == 1
            else value.reshape(-1).astype(float).tolist()
        )
    return primitives


def _whitelight_geometry_handoff_fingerprint(payload):
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION,
            "target_revision": WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION,
            "payload": payload,
        },
    )
    return hasher.hexdigest()


def _write_whitelight_geometry_handoff(path, payload):
    """Atomically persist a self-validating white-light geometry handoff."""
    payload = dict(payload)
    payload["schema_version"] = WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION
    payload["target_revision"] = WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint_sha256"
    }
    payload["artifact_fingerprint_sha256"] = (
        _whitelight_geometry_handoff_fingerprint(fingerprint_payload)
    )
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return payload


def _load_whitelight_geometry_handoff(
    path, *, expected_posterior_fingerprint, expected_estimator,
):
    """Load a handoff only if its schema, fingerprint, and target all match."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = {
        'schema_version': WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION,
        'target_revision': WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION,
        'posterior_fingerprint_sha256': expected_posterior_fingerprint,
        'estimator': expected_estimator,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        return None
    expected_artifact_fingerprint = payload.get("artifact_fingerprint_sha256")
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint_sha256"
    }
    if expected_artifact_fingerprint != _whitelight_geometry_handoff_fingerprint(
        fingerprint_payload
    ):
        return None
    geometry = payload.get("geometry")
    required = {
        "period", "duration", "t0", "b", "a_rs", "cos_i",
        "inclination", "rors",
    }
    if not isinstance(geometry, dict) or not required.issubset(geometry):
        return None
    try:
        lengths = {
            len(np.atleast_1d(np.asarray(geometry[name], dtype=float)))
            for name in required
        }
        finite = all(
            (np.all(np.isfinite(np.asarray(geometry[name], dtype=float))) for name in required),
        )
    except (TypeError, ValueError):
        return None
    if len(lengths) != 1 or next(iter(lengths), 0) < 1 or not finite:
        return None
    if any(
        payload.get(name) is not None
        for name in (
            "selected_flat_draw_index", "selected_chain_index",
            "selected_draw_index", "summed_data_log_likelihood",
        )
    ):
        return None
    return payload




def _joint_geometry_sites(param_method):
    """Return the shared geometry sites for a spectroscopic parameterisation."""
    if param_method == "duration":
        return ("t0", "b", "duration")
    if param_method == "a_rs":
        return ("t0", "b", "a_rs")
    raise ValueError(f"Unknown param_method: {param_method}")


def _whitelight_geometry_prior(
    bestfit_params_wl_df, wl_geometry_handoff, *, param_method, prior_inflation
):
    """Build the shared-geometry prior from the white-light posterior.

    The centres are the handoff medians consumed by the fixed-geometry path;
    the widths are the white-light posterior standard deviations recorded in
    the white-light bestfit table (``t0_err``, ``b_err``, ``duration_err``,
    ``a_rs_err``) multiplied by ``prior_inflation``.  Returns a mapping with
    ``t0``/``b``/``duration`` or ``a_rs`` centres and matching
    ``sigma_<name>`` widths, each a float64 array with one entry per planet.
    """
    prior_inflation = float(prior_inflation)
    if not np.isfinite(prior_inflation) or prior_inflation <= 0.0:
        raise ValueError("prior_inflation must be finite and > 0.")
    geometry = wl_geometry_handoff["geometry"]
    num_planets = len(np.atleast_1d(np.asarray(geometry["t0"], dtype=float)))
    columns = set(getattr(bestfit_params_wl_df, "columns", ()))
    prior = {}
    for name in _joint_geometry_sites(param_method):
        center = np.atleast_1d(np.asarray(geometry[name], dtype=np.float64))
        err_column = f"{name}_err"
        if err_column not in columns:
            raise ValueError(
                "flags.spectro_joint_geometry=true requires the white-light "
                f"posterior standard deviation column {err_column!r} in the "
                "white-light bestfit table; it is missing (was the white-light "
                "geometry fixed?)."
            )
        std = np.atleast_1d(
            np.asarray(bestfit_params_wl_df[err_column].values, dtype=np.float64)
        )
        if std.shape[0] != num_planets or center.shape[0] != num_planets:
            raise ValueError(
                f"White-light {name} median/std have inconsistent planet "
                f"dimensions ({center.shape[0]}, {std.shape[0]}) for "
                f"{num_planets} planets."
            )
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(std)):
            raise ValueError(f"White-light {name} median/std are not finite.")
        if np.any(std <= 0.0):
            raise ValueError(
                f"White-light posterior std for {name} must be > 0 to build "
                "the shared-geometry prior; the white-light posterior appears "
                "degenerate."
            )
        prior[name] = center
        prior[f"sigma_{name}"] = prior_inflation * std
    return prior

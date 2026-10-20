"""Immutable constants and artifact schema revisions."""

from models.harmonica.core import _ALL_ODD_COEFF_SPECS

TREND_PARAMS = [
    'c', 'v', 'v2', 'v3', 'v4',
    'A', 'tau',
    'spot_amp', 'spot_mu', 'spot_sigma',
    'spot_amp2', 'spot_mu2', 'spot_sigma2',
    't_jump', 'jump',
    'A_gp', 'A_spot', 'A_spot2', 'A_jump'
]


SURFACE_PARAMS = (
    'eclipse_depth', 'phase_c1', 'phase_d1', 'stellar_spot_contrast',
)




FLAG_TIERS = {
    'public': frozenset({
        'detrending_type',
        'exclude_integrations',
        'exclude_times',
        'jump_guess',
        'ld_prior',
        'ld_gaussian_sigma',
        'ld_profile',
        'light_curve_model',
        'mask_end',
        'mask_start',
        'spot_amp',
        'spot_amp2',
        'spot_amp_2',
        'spot_center',
        'spot_center2',
        'spot_center_2',
        'spot_width',
        'spot_width2',
        'spot_width_2',
        't_jump_guess',
    }),
    'advanced': frozenset({
        'num_chains',
        'analysis_stage',
        'chunk_mode',
        'chunk_parallel_job_count',
        'chunk_parallel_job_index',
        'cpu_host_devices',
        'harmonica_max_order',
        'harmonica_spectro_fit_jitter',
        'harmonica_spectro_odd_frac_sigma',
        'harmonica_spectro_parameterization',
        'jax_compilation_cache_dir', 'grid_prime_samplers', 'grid_cache_memory_mb',
        'ld_parameterization',
        'ld_uniform_basis', 'ld_uniform_coefficient_bounds',
        'need_lowres',
        'plots',
        'random_seed',
        'spectro_chunk_size',
        'spectro_cadence_reduction',
        'spectro_fixed_timescale_trends',
        'spectro_joint_geometry',
        'spectro_free_t0',
        'spectro_t0_half_width_s',
        'spectro_mass_matrix',
        'spectro_trend_basis',
        'spectro_trend_bounds',
        'spectro_joint_geometry_prior_inflation',
        'spectro_transit_grid',
        'spectro_transit_grid_nodes',
        'spectro_max_divergences',
        'spectro_min_depth_ess',
        'spectro_sampler',
        'transit_engine',
        'whitelight_mass_matrix',
        'whitelight_reuse_samples',
        'trend_inference',
        'trend_prior_scale',
        'vmap_chunk',
    }) | frozenset(
        f'{stage}_{suffix}'
        for stage in ('whitelight', 'lowres', 'highres')
        for suffix in ('num_warmup', 'num_samples', 'num_chains')
    ),
    'internal': frozenset(),
}


PUBLIC_FLAGS = FLAG_TIERS['public']


ADVANCED_FLAGS = FLAG_TIERS['advanced']


INTERNAL_FLAGS = FLAG_TIERS['internal']


KNOWN_FLAGS = frozenset().union(*FLAG_TIERS.values())


# Orbital parameters of the ``planet`` block. Every one is written as a
# ``{value: ..., prior: ...}`` mapping; see ``koala.config.parse_parameter_spec``.
PLANET_PARAMETER_KEYS = (
    'period', 't0', 'eclipse_time', 'duration', 'a_rs', 'b', 'rprs', 'ecc', 'omega',
)


# Surface (emission) parameters of the ``planet`` block, with the factor that
# converts the user's unit into the model's internal unit.
PLANET_SURFACE_PARAMETER_SCALES = {
    'eclipse_depth_ppm': 1e-6,
    'phase_c1_ppm': 1e-6,
    'phase_d1_ppm': 1e-6,
}


# Parameters that may only be ``fixed`` for now.
PLANET_PARAMETERS_FIXED_ONLY = frozenset({'ecc', 'omega'})


# Parameters whose free prior must be gaussian.
PLANET_PARAMETERS_GAUSSIAN_ONLY = frozenset()


PARAMETER_SPEC_PRIORS = ('fixed', 'uniform', 'log_uniform', 'gaussian')


# Keys of the optional top-level ``trend_priors`` block (white-light
# systematics, ``trend = c + v*t + A*exp(-t/tau)`` with t from the first
# timestamp) and the built-in prior each one replaces. ``value`` is the
# default start; an omitted ``value`` falls back to it, or to the prior
# midpoint when it lies outside the configured bounds.
TREND_PRIOR_DEFAULTS = {
    'c': {'prior': 'uniform', 'low': 0.9, 'high': 1.1, 'value': 1.0},
    'v': {'prior': 'uniform', 'low': -0.1, 'high': 0.1, 'value': 0.0},
    **{f'v{k}': {'prior': 'uniform', 'low': -0.1, 'high': 0.1, 'value': 0.0} for k in (2,3,4)},
    'A': {'prior': 'uniform', 'low': -0.1, 'high': 0.1, 'value': 0.001},
    'tau': {'prior': 'log_uniform', 'low': 1e-3, 'high': 1e-1, 'value': 1e-2},
    # White-light white-noise jitter added in quadrature to the flux errors.
    'jitter': {'prior': 'log_uniform', 'low': 1e-5, 'high': 1e-2, 'value': 1e-4},
}


# Removed ``planet`` keys and the replacement each error message names.
PLANET_LEGACY_KEYS = {
    't0_prior_width_days': "t0: {value: ..., prior: gaussian, sigma: <days>}",
    'a_rs_prior_min': "a_rs: {value: ..., prior: log_uniform, low: ..., high: ...}",
    'a_rs_prior_max': "a_rs: {value: ..., prior: log_uniform, low: ..., high: ...}",
    'eclipse_depth_prior_width_ppm': "eclipse_depth_ppm: {value: ..., prior: gaussian, sigma: <ppm>, low: 0}",
    'dayside_flux_prior_width_ppm': "eclipse_depth_ppm (planet flux at secondary eclipse), phase_c1_ppm and phase_d1_ppm mappings",
    'nightside_flux_prior_width_ppm': "eclipse_depth_ppm (planet flux at secondary eclipse), phase_c1_ppm and phase_d1_ppm mappings",
    'hotspot_offset_prior_width_deg': "eclipse_depth_ppm (planet flux at secondary eclipse), phase_c1_ppm and phase_d1_ppm mappings",
    'dayside_flux_ppm': "eclipse_depth_ppm (planet flux at secondary eclipse) with phase_c1_ppm and phase_d1_ppm",
    'nightside_flux_ppm': "phase_c1_ppm (the nightside flux is now derived as eclipse_depth - 2*phase_c1)",
    'hotspot_offset_deg': "phase_c1_ppm and phase_d1_ppm (the offset is now derived from atan2)",
}


_JUMP_WIDTH_DAYS = 1e-4


JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS = (
    "mu_depths",
    "trend_fixed",
    "ld_interpolated",
    "ld_fixed",
    "mu_u_ld",
    "sigma_u_ld",
    "precomputed_yerr_per_lc",
    "trend_prior_mean",
    "trend_prior_scale",
    "surface_basis_data",
    "oot_reference_beta",
    "oot_group_yerr",
    "oot_group_count",
    "oot_group_reference_sse",
    "oot_group_x_reference_residual",
    "oot_group_xx",
)


HARMONICA_CHANNEL_VARYING_MODEL_KWARGS = (
    "mu_depths",
    "ld_interpolated",
    "ld_fixed",
    "mu_u_ld",
    "sigma_u_ld",
)


CHUNK_CHECKPOINT_SCHEMA_VERSION = 5


CHUNK_CHECKPOINT_TARGET_REVISION = "bounded-surface-priors-science-gates-v8-power2-exact-v1-multichain-v1-spectro-mask-depth-gates-v1-keplerian-reference-radius-v1-normalized-polynomial-priors-v1-fixed-ramp-amplitude-v1-template-linear-contract-v1-quadratic-uniform-uplus-uminus-default-v1"


SAMPLING_WORKLOAD_SCHEMA_VERSION = 1


SCIENCE_ARTIFACT_SCHEMA_VERSION = 1


SCIENCE_ARTIFACT_TARGET_REVISION = "surface-period-priors-science-gates-2026-09-12-v3-power2-exact-v1-multichain-v1-spectro-mask-depth-gates-v1-keplerian-reference-radius-v1-normalized-polynomial-priors-v1-fixed-ramp-amplitude-v1-template-linear-contract-v1-quadratic-uniform-uplus-uminus-default-v1"


WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION = 1


WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION = "retained-draw-data-loglik-v2-power2-exact-v1-multichain-v1-spectro-mask-depth-gates-v1-keplerian-reference-radius-v1-quadratic-uniform-uplus-uminus-default-v1"


SPECTRO_DATA_TARGET_REVISION = "createdatacube-multi-epoch-mask-2026-09-09-v3"


POWER2_LD_CACHE_TARGET_REVISION = "stellar-grid-direct-power2-v2"




HARMONICA_INIT_ODD_COEFF = 1e-4


HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION = 3


HARMONICA_LIMB_PRODUCT_CONVENTION = (
    "one:theta=[pi/2,3pi/2],endpoint=pi;"
    "two:theta=[-pi/2,pi/2],endpoint=0;"
    "indices_require_external_geometry_mapping"
)


HARMONICA_ODD_HARMONICS = tuple(name for name, _ in _ALL_ODD_COEFF_SPECS)


_VALID_HARMONICA_SPECTRO_PARAMETERIZATIONS = frozenset(
    {'fractional', 'delta_r', 'half_area'}
)


_JAXOPLANET_STATIC_EVAL_KEYS = frozenset(
    {"_jaxoplanet_kernel", "_ld_profile", "_power2_exact"}
)


_SURFACE_STATIC_EVAL_KEYS = frozenset(
    {"_surface_model", "_stellar_spots"}
)


_JAXOPLANET_DYNAMIC_EVAL_KEYS = (
    "_transit_phase_offsets",
    "_transit_phase_mask",
    "_transit_window_indices",
)

"""Immutable constants and artifact schema revisions."""

from models.limb_darkening_config import LD_PRIORS
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
    'eclipse_depth', 'dayside_flux', 'nightside_flux',
    'hotspot_offset', 'stellar_spot_contrast',
)


LD_PRIOR_MODES = set(LD_PRIORS)


FLAG_TIERS = {
    'public': frozenset({
        'detrending_type',
        'fit_geometry',
        'jump_guess',
        'ld_prior',
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
        'analysis_stage',
        'chunk_mode',
        'chunk_parallel_job_count',
        'chunk_parallel_job_index',
        'harmonica_max_order',
        'harmonica_spectro_fit_jitter',
        'harmonica_spectro_odd_frac_sigma',
        'harmonica_spectro_parameterization',
        'jax_compilation_cache_dir',
        'ld_uniform_basis',
        'need_lowres',
        'plots',
        'random_seed',
        'spectro_chunk_size',
        'spectro_cadence_reduction',
        'spectro_transit_grid',
        'spectro_transit_grid_nodes',
        'spectro_max_divergences',
        'spectro_min_depth_ess',
        'spectro_sampler',
        'transit_engine',
        'trend_inference',
        'vmap_chunk',
    }),
    'internal': frozenset(
        f'{stage}_{suffix}'
        for stage in ('whitelight', 'lowres', 'highres')
        for suffix in ('num_warmup', 'num_samples')
    ),
}


PUBLIC_FLAGS = FLAG_TIERS['public']


ADVANCED_FLAGS = FLAG_TIERS['advanced']


INTERNAL_FLAGS = FLAG_TIERS['internal']


KNOWN_FLAGS = frozenset().union(*FLAG_TIERS.values())


# Orbital parameters of the ``planet`` block that accept the unified
# ``[fixed, value]`` / ``[free, prior, ...]`` specification.
PLANET_PARAMETER_KEYS = (
    'period', 't0', 'duration', 'a_rs', 'b', 'rprs', 'ecc', 'omega',
)


# Bare numbers keep the historical behaviour: these are fixed, the rest free.
PLANET_PARAMETERS_FIXED_BY_DEFAULT = frozenset({'period', 'ecc', 'omega'})


# Parameters that may only be ``fixed`` for now.
PLANET_PARAMETERS_FIXED_ONLY = frozenset({'ecc', 'omega'})


PARAMETER_SPEC_MODES = ('fixed', 'free')


PARAMETER_SPEC_PRIORS = ('uniform', 'gaussian', 'truncated_gaussian')


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


CHUNK_CHECKPOINT_SCHEMA_VERSION = 4


CHUNK_CHECKPOINT_TARGET_REVISION = "phase-flux-quantile-v7"


SAMPLING_WORKLOAD_SCHEMA_VERSION = 1


SCIENCE_ARTIFACT_SCHEMA_VERSION = 1


SCIENCE_ARTIFACT_TARGET_REVISION = "whitelight-trend-routing-2026-09-03-v2"


WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION = 1


WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION = "retained-draw-data-loglik-v2"


SPECTRO_DATA_TARGET_REVISION = "createdatacube-multi-epoch-mask-2026-09-09-v3"


POWER2_LD_CACHE_TARGET_REVISION = "stellar-grid-direct-power2-v2"


SAMPLER_STAGE_INPUT_SCHEMA_VERSION = 1


MCMC_KWARGS = {"num_warmup": 1000, "num_samples": 1000}


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
    {"_jaxoplanet_kernel", "_ld_profile"}
)


_SURFACE_STATIC_EVAL_KEYS = frozenset(
    {"_surface_model", "_stellar_spots"}
)


_JAXOPLANET_DYNAMIC_EVAL_KEYS = (
    "_transit_phase_offsets",
    "_transit_phase_mask",
    "_transit_window_indices",
)

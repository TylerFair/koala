"""Fail early when a requested law would silently use an unaudited projection."""


def validate_power2_support(ld_profile, *, model='transit', spots=(),
                            evaluation='auto', interpolated=False):
    """Power-2 inference requires the physical exact-kernel route.

    Degree-12 surface projections are retained as diagnostic functions, but
    their measured errors do not certify an inference configuration class.
    Quadratic surface models and exact unspotted power-2 remain available.
    """
    if ld_profile != 'power2':
        return
    if model != 'transit' or spots:
        kind = ('spotted transit' if model == 'transit' else model)
        raise ValueError(
            f'Power-2 {kind} is unsupported: the degree-12 surface projection '
            'has no flux/gradient accuracy certificate for this configuration. '
            'Use quadratic limb darkening for surface/spot models, or the exact '
            'power-2 kernel for an unspotted transit.')
    if evaluation == 'polynomial':
        raise ValueError(
            'Polynomial power-2 transits are unsupported: use '
            "power2_evaluation='exact' (or 'auto') with physical (c, alpha) coefficients.")
    if interpolated:
        raise ValueError(
            'Projected/interpolated polynomial coefficients do not identify an exact '
            'power-2 law. Supply physical (c, alpha) coefficients with fixed, gaussian, '
            'stellarprior, or uniform LD; use quadratic LD for polynomial interpolation.')

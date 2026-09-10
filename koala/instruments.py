"""Registry of the JWST instrument modes Koala supports.

Each entry maps a Koala ``instrument`` string to the ExoTiC-LD throughput
mode, the wavelength range (in angstroms) over which that throughput is
defined, the wavelength window (in microns) kept when the extracted spectra
are read, and the configuration key that selects the detector or order.
Only modes with an ExoTiC-LD throughput are supported; MIRI/MRS and the
imaging modes are not.
"""

_NIRSPEC = {
    'NIRSPEC/PRISM': ('JWST_NIRSpec_Prism', (5000.0, 55000.0), (0.6, 5.0)),
    'NIRSPEC/G395H': ('JWST_NIRSpec_G395H', (28700.0, 51700.0), (2.9, 5.0)),
    'NIRSPEC/G395M': ('JWST_NIRSpec_G395M', (28700.0, 51700.0), (2.9, 5.0)),
    'NIRSPEC/G235H': ('JWST_NIRSpec_G235H', (16600.0, 30700.0), (1.66, 3.1)),
    'NIRSPEC/G235M': ('JWST_NIRSpec_G235M', (16600.0, 31200.0), (1.66, 3.1)),
    'NIRSPEC/G140H': ('JWST_NIRSpec_G140H-f100', (10000.0, 18000.0), (1.0, 1.8)),
    'NIRSPEC/G140M': ('JWST_NIRSpec_G140M-f100', (9700.0, 18600.0), (0.97, 1.87)),
    'NIRSPEC/G140H-F070': ('JWST_NIRSpec_G140H-f070', (8250.0, 12700.0), (0.82, 1.27)),
    'NIRSPEC/G140M-F070': ('JWST_NIRSpec_G140M-f070', (7000.0, 12700.0), (0.7, 1.27)),
}

INSTRUMENTS = {}
for _name, (_mode, _ld_bounds, _data_range) in _NIRSPEC.items():
    INSTRUMENTS[_name] = {
        'ld_mode': _mode,
        'ld_bounds': _ld_bounds,
        'data_range': _data_range,
        'detector_key': 'nrs',
    }
INSTRUMENTS['NIRISS/SOSS'] = {
    'ld_mode': {1: 'JWST_NIRISS_SOSSo1', 2: 'JWST_NIRISS_SOSSo2'},
    'ld_bounds': {1: (8300.0, 28100.0), 2: (6300.0, 8500.0)},
    'data_range': None,
    'detector_key': 'order',
}
INSTRUMENTS['NIRCAM/F322W2'] = {
    'ld_mode': 'JWST_NIRCam_F322W2',
    'ld_bounds': (24000.0, 42100.0),
    'data_range': (2.4, 4.1),
    'detector_key': None,
}
INSTRUMENTS['NIRCAM/F444W'] = {
    'ld_mode': 'JWST_NIRCam_F444',
    'ld_bounds': (37300.0, 49900.0),
    'data_range': (3.8, 5.0),
    'detector_key': None,
}
INSTRUMENTS['MIRI/LRS'] = {
    'ld_mode': 'JWST_MIRI_LRS',
    'ld_bounds': (50000.0, 120000.0),
    'data_range': (5.0, 12.0),
    'detector_key': None,
}

# Alternative spellings accepted in configuration files.
INSTRUMENT_ALIASES = {
    'NIRSPEC/G140H-F100': 'NIRSPEC/G140H',
    'NIRSPEC/G140M-F100': 'NIRSPEC/G140M',
    'NIRCAM/F444': 'NIRCAM/F444W',
    'NIRCAM/F332W2': 'NIRCAM/F322W2',  # common typo
}

SUPPORTED_INSTRUMENTS = tuple(INSTRUMENTS)


def normalize_instrument(instrument):
    """Return the canonical Koala name for ``instrument``.

    Matching ignores case and accepts the aliases in ``INSTRUMENT_ALIASES``.
    Raises ``ValueError`` for anything outside the registry.
    """
    key = str(instrument).strip().upper()
    key = INSTRUMENT_ALIASES.get(key, key)
    if key not in INSTRUMENTS:
        raise ValueError(
            f"Unsupported instrument {instrument!r}. Supported instruments: "
            f"{', '.join(SUPPORTED_INSTRUMENTS)}."
        )
    return key


def instrument_spec(instrument):
    return INSTRUMENTS[normalize_instrument(instrument)]


def is_nirspec(instrument):
    return normalize_instrument(instrument).startswith('NIRSPEC/')


def is_niriss(instrument):
    return normalize_instrument(instrument) == 'NIRISS/SOSS'


def detector_key(instrument):
    """Configuration key naming the detector (``nrs``), order, or ``None``."""
    return instrument_spec(instrument)['detector_key']


def resolve_detector(instrument, cfg):
    """Return ``(nrs, order)`` for ``instrument`` from a configuration."""
    key = detector_key(instrument)
    if key is None:
        return None, None
    if key not in cfg:
        raise KeyError(f"'{key}' is required for instrument {instrument}.")
    value = int(cfg[key])
    if key == 'order':
        if value not in (1, 2):
            raise ValueError(f"NIRISS/SOSS order must be 1 or 2, got {value}.")
        return None, value
    if value not in (1, 2):
        raise ValueError(f"NIRSpec nrs must be 1 or 2, got {value}.")
    return value, None


def detector_label(instrument, nrs=None, order=None):
    """Short detector tag used in output file names (``nrs1``, ``order2``, '')."""
    key = detector_key(instrument)
    if key == 'nrs':
        return f'nrs{nrs}'
    if key == 'order':
        return f'order{order}'
    return ''


def ld_mode_and_bounds(instrument, order=None):
    """ExoTiC-LD mode name and its (min, max) wavelength bounds in angstroms."""
    spec = instrument_spec(instrument)
    mode, bounds = spec['ld_mode'], spec['ld_bounds']
    if isinstance(mode, dict):
        if order is None:
            raise ValueError(f"{instrument} requires an order for limb darkening.")
        order = int(order)
        if order not in mode:
            raise ValueError(f"{instrument} order must be one of {sorted(mode)}, got {order}.")
        return mode[order], bounds[order][0], bounds[order][1]
    return mode, bounds[0], bounds[1]


def data_wavelength_range(instrument):
    """Wavelength window in microns kept when reading extracted spectra, or None."""
    return instrument_spec(instrument)['data_range']

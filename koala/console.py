"""Human progress on stdout, complete diagnostics in the run log."""
import contextlib
import datetime
import io
import math
import logging
import os
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from contextvars import ContextVar

logger = logging.getLogger('koala')
logger.setLevel(logging.DEBUG)
logger.propagate = False
_pending = []


def koala_version():
    try:
        return version('koala-jwst')
    except PackageNotFoundError:
        # Source checkouts need not be installed as a distribution.
        import re
        from pathlib import Path
        project = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        if project.exists():
            match = re.search(r'^version = "([^"]+)"', project.read_text(), re.MULTILINE)
            if match:
                return match.group(1)
        return 'unknown'


class _StdoutHandler(logging.StreamHandler):
    def emit(self, record):
        self.stream = sys.stdout
        super().emit(record)


class _Pending(logging.Handler):
    def emit(self, record):
        if not any(isinstance(h, logging.FileHandler) and not getattr(h, '_koala_parent_log', False)
                   for h in logger.handlers):
            _pending.append(record)


logger.addHandler(_Pending())
_default_console = _StdoutHandler(sys.stdout)
_default_console._koala_console = True
_default_console.setLevel(logging.INFO)
_default_console.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(_default_console)


def configure(verbose=False, output_dir=None, config_path=None):
    """Configure a session without adding anything to the science configuration."""
    if joint_display.get() is None:
        _stellar_notices.clear()
    verbose = verbose or os.environ.get('KOALA_VERBOSE') == '1'
    for handler in list(logger.handlers):
        if (getattr(handler, '_koala_console', False) or
                (output_dir is None and isinstance(handler, logging.FileHandler)
                 and not getattr(handler, '_koala_parent_log', False))):
            logger.removeHandler(handler)
            handler.close()
    console = _StdoutHandler(sys.stdout)
    console._koala_console = True
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    if not sys.stdout.isatty():
        os.environ.setdefault('TQDM_MININTERVAL', '30')
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and not getattr(handler, '_koala_parent_log', False):
                logger.removeHandler(handler)
                handler.close()
        file_handler = logging.FileHandler(os.path.join(output_dir, 'koala.log'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(file_handler)
        file_handler.handle(logger.makeRecord('koala', logging.INFO, '', 0,
            f'Run {datetime.datetime.now().isoformat()} | koala {koala_version()} | {config_path or "Python session"}', (), None))
        for record in _pending:
            file_handler.handle(record)
        _pending.clear()
    warnings.showwarning = _showwarning
    arviz_loggers = [logging.getLogger('arviz')]
    arviz_module = sys.modules.get('arviz')
    if arviz_module is not None and hasattr(arviz_module, '_log'):
        arviz_loggers.append(arviz_module._log)
    for arviz in arviz_loggers:
        if not any(isinstance(f, _ArvizFilter) for f in arviz.filters):
            arviz.addFilter(_ArvizFilter())


_original_showwarning = warnings.showwarning


def _log_only(message, pathname='', lineno=0):
    record = logger.makeRecord('koala', logging.DEBUG, pathname, lineno, message, (), None)
    for handler in logger.handlers:
        if isinstance(handler, (logging.FileHandler, _Pending)):
            handler.handle(record)


def _showwarning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    detail = f'{category.__name__}: {text} ({filename}:{lineno})'
    if category is FutureWarning and text.lstrip().startswith('ArviZ is undergoing a major refactor'):
        _log_only(detail, filename, lineno)
        return
    if issubclass(category, UserWarning) and 'layout engine that is incompatible with subplots_adjust' in text:
        _log_only(detail, filename, lineno)
        return
    debug(detail)
    _original_showwarning(message, category, filename, lineno, file=file, line=line)


class _ArvizFilter(logging.Filter):
    def filter(self, record):
        if 'Shape validation failed' in record.getMessage() and 'input_shape: (1,' in record.getMessage():
            _log_only(record.getMessage(), record.pathname, record.lineno)
            return False
        return True


def debug(*values, sep=' ', end='\n', flush=False, **kwargs):
    logger.debug(sep.join(str(value) for value in values).rstrip('\n'))


def info(*values, sep=' ', end='\n', flush=False, **kwargs):
    logger.info(sep.join(str(value) for value in values).rstrip('\n'))


def progress(iterable=None, **kwargs):
    from tqdm.auto import tqdm
    kwargs.setdefault('file', sys.stdout)
    kwargs.setdefault('mininterval', 0.1 if sys.stdout.isatty() else 30.0)
    return tqdm(iterable, **kwargs)


@contextlib.contextmanager
def stellar_output(model, teff, logg, feh):
    """Translate ExoTiC-LD's coverage notice; retain its complete output."""
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            yield
    finally:
        output = stream.getvalue()
        if output:
            debug(output)
            if 'nearest' in output.lower() and 'coverage' in output.lower():
                message = (f'The {model} grid does not cover Teff = {teff:g} K, '
                           f'log g = {logg:g}, [Fe/H] = {feh:g} exactly; using the nearest grid point.')
                if model not in _stellar_notices:
                    _stellar_notices.add(model)
                    info(message)
            else:
                info(output.rstrip())


_stellar_notices = set()


def notice_once(key, message):
    """Print ``message`` only the first time in a run (or in a joint fit)."""
    if key not in _stellar_notices:
        _stellar_notices.add(key)
        info(message)


@contextlib.contextmanager
def quiet_sampling():
    token = _spectroscopy.set(True)
    try:
        yield
    finally:
        _spectroscopy.reset(token)


_spectroscopy = ContextVar('koala_spectroscopy', default=False)


def sampling_options(options):
    """Change presentation only, after the original options have been hashed."""
    return {**(options or {}), 'progress_bar': False} if _spectroscopy.get() else dict(options or {})


def channel_batches(indices, ranges):
    """Advance once a complete batch has been computed or restored."""
    with progress(total=sum(ranges[i][1] - ranges[i][0] for i in indices), desc='Channels') as bar:
        for index in indices:
            yield index
            bar.update(ranges[index][1] - ranges[index][0])


def white_light_verdict(diagnostics):
    passed = diagnostics['whitelight_quality_gate_passed']
    rhats = diagnostics.get('whitelight_geometry_rhat', {})
    ess = diagnostics.get('whitelight_geometry_bulk_ess', {})
    parts = []
    if rhats:
        parts.append(f'R-hat <= {max(rhats.values()):.3f}')
    if ess:
        parts.append(f'bulk effective sample size >= {min(ess.values()):.0f}')
    divergences = diagnostics.get('whitelight_divergences', 0)
    parts.append('no divergences' if divergences == 0 else f'{divergences} divergences')
    info(('  Converged: ' if passed else '  Convergence checks failed: ') + ', '.join(parts))


def spectroscopy_verdict(wavelengths, failed, *, extra_draws=0, refitted=0):
    notes = []
    if extra_draws:
        notes.append(f'{extra_draws} needed extra draws')
    if refitted:
        notes.append(f'{refitted} {"was" if refitted == 1 else "were"} refitted with a second sampler')
    suffix = ' (' + ', '.join(notes) + ')' if notes else ''
    if len(failed):
        waves = ', '.join(f'{float(wavelengths[i]):.3f}' for i in failed)
        info(f'{len(failed)} of {len(wavelengths)} channels did not pass the convergence checks: {waves} µm')
        info('See koala.log and the diagnostics files for details.')
    else:
        info(f'All {len(wavelengths)} channels passed the convergence checks!' + suffix)


def finish_run(output_dir, started):
    import time
    if joint_display.get() is not None:
        return
    info('')
    output_dir = os.path.normpath(str(output_dir))
    info(f'Figures: {output_dir}/figures/')
    info(f'Full log: {os.path.join(output_dir, "koala.log")}')
    elapsed = int(time.perf_counter() - started)
    info(f'Done in {elapsed_time(elapsed)}.')


def single_batch_progress(function):
    """Give unbatched spectroscopy the same stage bar as batched fits."""
    import functools
    import inspect
    signature = inspect.signature(function)
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if bound.arguments.get('use_chunked', False):
            return function(*args, **kwargs)
        count = int(bound.arguments['indiv_y'].shape[0])
        with progress(total=count, desc='Channels') as bar:
            result = function(*args, **kwargs)
            bar.update(count)
            return result
    return wrapped


def elapsed_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds} s'
    if seconds < 3600:
        return f'{seconds // 60} min {seconds % 60} s'
    return f'{seconds // 3600} h {(seconds % 3600) // 60:02d} min'


def grid_description(label):
    """Translate artifact labels without changing the labels used on disk."""
    label = str(label)
    if 'native' in label.lower():
        return 'at native resolution'
    if 'reference' in label.lower():
        return 'on the reference grid'
    if label == 'whitelight':
        return 'for the white-light curve'
    if label.startswith('pix'):
        return f'in bins of {label[3:]} columns'
    if label.startswith('R'):
        return f'at R = {label[1:]}'
    return 'on the configured wavelength grid'


def parameter_summary(label, value, uncertainty, unit=''):
    """Two significant figures in the error; matching decimals in the value."""
    if math.isfinite(uncertainty) and uncertainty > 0:
        rounded_error = float(f'{uncertainty:.2g}')
        place = 1 - math.floor(math.log10(rounded_error))
        decimals = max(0, place)
        value_text = f'{round(value, place):.{decimals}f}'
        error_text = f'{rounded_error:.{decimals}f}'
    else:
        value_text, error_text = f'{value:.7g}', f'{uncertainty:.2g}'
    return f'  {label:<16} {value_text} ± {error_text}' + (f' {unit}' if unit else '')


_reading_path = ContextVar('koala_reading_path', default=None)


@contextlib.contextmanager
def reading(path, enabled=True):
    (info if enabled else debug)(f'Reading {os.path.normpath(str(path))}')
    token = _reading_path.set(path)
    try:
        yield
    finally:
        _reading_path.reset(token)


def announce_reading(path):
    if _reading_path.get() is None:
        info(f'Reading {os.path.normpath(str(path))}')


def sampling_history_notes(attempts, channels, min_ess, max_divergences):
    """Count unique channels needing extensions or retries from recorded gates."""
    extended, refitted = set(), set()
    previous = None
    for attempt in attempts:
        if 'input_local_lanes' in attempt:
            refitted.update(attempt['input_local_lanes'])
        if attempt.get('sampling_extension_blocks', 0) and previous is not None:
            group = previous.get('input_local_lanes', list(range(channels)))
            chains = int(previous.get('num_chains', 1))
            checks = []
            bulk = previous.get('science_ess_per_channel') or {}
            if not bulk and previous.get('depth_ess_per_channel') is not None:
                bulk = {'depth': previous['depth_ess_per_channel']}
            checks.extend((values, min_ess, False) for values in bulk.values())
            if chains > 1:
                for key, threshold, upper in (
                    ('science_tail_ess_per_channel', min_ess, False),
                    ('science_rhat_per_channel', 1.01, True),
                    ('science_min_chain_ess_per_channel', min_ess / chains, False),
                ):
                    checks.extend((values, threshold, upper)
                                  for values in (previous.get(key) or {}).values())
            for values, threshold, upper in checks:
                for index, value in enumerate(values):
                    if index < len(group) and (not math.isfinite(value)
                            or (value >= threshold if upper else value < threshold)):
                        extended.add(group[index])
        previous = attempt
    return len(extended), len(refitted)


@contextlib.contextmanager
def parent_log():
    """Keep the joint run's complete log while members open their own logs."""
    handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    for handler in handlers:
        handler._koala_parent_log = True
    try:
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


@contextlib.contextmanager
def exotic_output():
    """Keep ExoTiC-LD's per-file download bars in the log, preserving warnings."""
    stream = io.StringIO()
    try:
        with contextlib.redirect_stderr(stream):
            yield
    finally:
        import re
        downloads = False
        for line in re.split(r'[\r\n]+', stream.getvalue()):
            if not line:
                continue
            if line.lstrip().startswith('Downloading '):
                _log_only(line)
                downloads = True
            else:
                sys.stderr.write(line + '\n')
        if downloads and 'downloads' not in _stellar_notices:
            _stellar_notices.add('downloads')
            info('Downloaded stellar limb-darkening data.')


# Presentation state stays outside configuration and scientific artifact identities.
joint_display = ContextVar('koala_joint_display', default=None)


@contextlib.contextmanager
def joint_presentation(state):
    token = joint_display.set(state)
    try:
        yield
    finally:
        joint_display.reset(token)


def residual_summary(scatter, clipped, sigma, label=None):
    state = joint_display.get()
    if state is not None and state.get('collect_residuals'):
        state.setdefault('residuals', []).append((scatter, clipped, sigma, label))
        debug(f'Residual scatter {scatter:.0f} ppm ({label}); ' + clipping_description(clipped, sigma))
        return
    info(f'  Residual scatter {scatter:.0f} ppm; ' + clipping_description(clipped, sigma))


def clipping_description(clipped, sigma):
    if not clipped:
        return 'no integrations clipped'
    return f'{clipped} integration{"s" if clipped != 1 else ""} clipped at {sigma:g}σ'


def install_warning_routing():
    """Install before numerical imports, retaining the import notice in the log."""
    warnings.showwarning = _showwarning

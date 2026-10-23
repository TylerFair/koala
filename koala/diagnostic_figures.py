"""Display-only summaries of already evaluated light curves and posteriors."""
import numpy as np
import matplotlib.pyplot as plt
from . import plotting_style as style


def display_bins(x, y, error=None, count=60):
    """Average points for display only; never return data to a fit."""
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    if not len(order):
        return np.array([]), np.array([]), np.array([])
    groups = np.array_split(order, min(count, len(order)))
    bx, by, be = [], [], []
    for group in groups:
        if not len(group):
            continue
        bx.append(np.mean(x[group]))
        by.append(np.mean(y[group]))
        be.append(np.sqrt(np.sum(np.asarray(error)[group] ** 2)) / len(group)
                  if error is not None else np.std(y[group]) / np.sqrt(len(group)))
    return np.asarray(bx), np.asarray(by), np.asarray(be)


def _points(ax, x, y, error, good, color):
    ax.plot(x[good], y[good], '.', color=color, alpha=.18, ms=2, rasterized=True)
    if np.any(~good):
        ax.plot(x[~good], y[~good], 'x', color='.65', ms=3, mew=.7)
    bx, by, be = display_bins(x[good], y[good], error[good])
    ax.errorbar(bx, by, yerr=be, fmt='o', ms=3, color=color, mfc='white', lw=.8, mew=1)
    _limits_from(ax, y[good])


def _limits_from(ax, values, pad=.12):
    low, high = np.nanmin(values), np.nanmax(values)
    span = (high - low) or abs(high) or 1.
    ax.set_ylim(low - pad*span, high + pad*span)


def white_light(time, flux, error, full_model, transit_model, filename, *, label, t0, outlier, event="transit"):
    style.apply_publication_style()
    x = (np.asarray(time) - float(t0)) * 24
    flux, full_model, transit_model = map(np.asarray, (flux, full_model, transit_model))
    error = np.broadcast_to(np.asarray(error), flux.shape)
    good = ~np.asarray(outlier, dtype=bool)
    trend = full_model / transit_model
    fig, axes = plt.subplots(3, 1, figsize=style.figsize(7.0), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1.3]}, constrained_layout=True)
    color = style.accent_color_for_label(label)
    for ax, values, errors, model, ylabel in (
        (axes[0], flux, error, full_model, 'Normalized flux'),
        (axes[1], flux / trend, error / np.abs(trend), transit_model, 'Detrended flux'),
        (axes[2], (flux - full_model) * 1e6, error * 1e6, np.zeros_like(flux), 'Residual [ppm]'),
    ):
        _points(ax, x, values, errors, good, color)
        ax.plot(x, model, color=style.MODEL_COLOR, lw=1.2)
        ax.set_ylabel(ylabel)
        style.style_axis(ax)
    axes[-1].set_xlabel(time_axis_label(event))
    style.figure_title(axes[0], label, 'White light')
    style.save_figure(fig, filename)
    return fig


def white_light_start(time, flux, error, initial, optimized, filename, *, label, t0, event="transit"):
    style.apply_publication_style()
    x = (np.asarray(time) - float(t0)) * 24
    flux, initial, optimized = map(np.asarray, (flux, initial, optimized))
    error = np.broadcast_to(np.asarray(error), flux.shape)
    fig, axes = plt.subplots(2, 1, figsize=style.figsize(5.2), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
    color = style.accent_color_for_label(label)
    _points(axes[0], x, flux, error, np.ones(len(x), bool), color)
    axes[0].plot(x, initial, '--', color=color, lw=1.2, label='Starting model')
    axes[0].plot(x, optimized, color=style.MODEL_COLOR, lw=1.2, label='Optimized start')
    axes[0].legend()
    axes[0].set_ylabel('Normalized flux')
    _points(axes[1], x, (flux-optimized)*1e6, error*1e6, np.ones(len(x), bool), color)
    axes[1].axhline(0, color=style.ZERO_LINE_COLOR, lw=1)
    axes[1].set(xlabel=time_axis_label(event), ylabel='Residual [ppm]')
    style.figure_title(axes[0], label, 'White-light start')
    for ax in axes:
        style.style_axis(ax)
    style.save_figure(fig, filename)
    return fig


def spectrum(wavelengths, rors_samples, filename, *, label, wavelength_err=None):
    style.apply_publication_style()
    depths = np.asarray(rors_samples) ** 2 * 1e6
    if depths.ndim == 2:
        depths = depths[..., None]
    fig, axes = plt.subplots(depths.shape[-1], 1, figsize=style.figsize(3.6 * depths.shape[-1]), squeeze=False,
                             constrained_layout=True)
    for planet, ax in enumerate(axes[:, 0]):
        lo, med, hi = np.nanpercentile(depths[..., planet], [16, 50, 84], axis=0)
        ax.errorbar(wavelengths, med, xerr=wavelength_err, yerr=[med-lo, hi-med],
                    **{**style.DATA_MARKER_STYLE, 'color': style.accent_color_for_label(label)})
        ax.set_ylabel('Depth [ppm]')
        ax.set_xlabel('Wavelength [µm]')
        style.style_axis(ax)
        if depths.shape[-1] > 1:
            ax.set_title(f'Planet {planet+1}', fontsize=style.TITLE_SIZE)
    style.figure_title(axes[0, 0], label, 'Spectrum' if depths.shape[-1] == 1 else 'Spectrum | Planet 1')
    style.save_figure(fig, filename)
    return fig


def channel_diagnostics(time, flux, model, wavelengths, trends, filename, *, label, t0, event="transit"):
    style.apply_publication_style()
    x = (np.asarray(time) - float(t0)) * 24
    flux, model, trends = map(np.asarray, (flux, model, trends))
    waves = np.asarray(wavelengths)
    targets = np.linspace(waves.min(), waves.max(), min(12, len(waves)))
    indices = np.unique([np.argmin(abs(waves-w)) for w in targets])
    # Use independent offsets for detrended flux and ppm residuals.
    detrended = flux / trends
    transit = model / trends
    amplitude = float(np.nanmax(np.nanmax(transit[indices], axis=1)-np.nanmin(transit[indices], axis=1)))
    scatter = float(np.nanpercentile(abs(detrended[indices]-transit[indices]), 98))
    spacing = max(amplitude, 2*scatter, .0001) * 1.35
    resid = (flux-model) * 1e6
    rstep = max(float(np.nanpercentile(abs(resid[indices]), 99)) * 3, 100)
    fig, (left, right) = plt.subplots(1, 2, figsize=style.figsize(max(4.5, .5*len(indices)+1.3)),
                                     gridspec_kw={'width_ratios': [1.6, 1]}, constrained_layout=True)
    for row, index in enumerate(indices):
        color = style.accent_color_for_label(label)
        y = detrended[index] - row*spacing
        left.plot(x, y, '.', ms=2, color=color, alpha=.18, rasterized=True)
        bx, by, _ = display_bins(x, y)
        left.plot(bx, by, 'o', ms=2.8, color=color, mfc='white')
        left.plot(x, transit[index]-row*spacing, color=style.MODEL_COLOR, lw=1.2)
        left.annotate(f'{waves[index]:.2f} µm', (1.01, 1-row*spacing), xycoords=('axes fraction', 'data'),
                      ha='left', va='center', fontsize=style.FONT_SIZE, annotation_clip=False)
        yres = resid[index]-row*rstep
        right.plot(x, yres, '.', ms=2, color=color, alpha=.18, rasterized=True)
        bx, by, _ = display_bins(x, yres)
        right.plot(bx, by, 'o', ms=2.8, color=color, mfc='white')
        right.axhline(-row*rstep, color='.5', lw=.7, ls='--')
        right.annotate(f'{np.nanstd(resid[index]):.0f} ppm', (1.01, -row*rstep),
                       xycoords=('axes fraction', 'data'), ha='left', va='center', fontsize=style.FONT_SIZE,
                       annotation_clip=False)
    for ax in (left, right):
        ax.margins(x=.05, y=.06)
        ax.set_xlabel(time_axis_label(event))
        style.style_axis(ax)
    right.set_yticks([])
    left.set_ylabel('Detrended flux + offset')
    right.set_ylabel('Residuals, offset')
    fig.suptitle(style.figure_title_text(label, 'Wavelength channels'), fontsize=style.TITLE_SIZE)
    style.save_figure(fig, filename)
    return fig


def residual_map(time, wavelengths, residuals, filename, *, label, t0, wavelength_err=None, event="transit"):
    style.apply_publication_style()
    x = (np.asarray(time)-float(t0))*24
    waves, residuals = np.asarray(wavelengths), np.asarray(residuals)*1e6
    time_order, wave_order = np.argsort(x), np.argsort(waves)
    x, waves = x[time_order], waves[wave_order]
    residuals = residuals[np.ix_(wave_order, time_order)]
    limit = max(float(np.nanpercentile(abs(residuals), 98)), 1.)
    fig, ax = plt.subplots(figsize=style.figsize(4.2), constrained_layout=True)
    if len(waves) == 1:
        # A single channel is still a visible wavelength strip.
        edges = np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]
        half_width = .01 if wavelength_err is None else float(np.asarray(wavelength_err).ravel()[0])
        image = ax.pcolormesh(edges, [waves[0]-half_width, waves[0]+half_width], residuals,
                              shading='flat', cmap=style.residual_colormap(),
                              vmin=-limit, vmax=limit, rasterized=True)
    else:
        image = ax.pcolormesh(x, waves, residuals, shading='nearest', cmap=style.residual_colormap(),
                              vmin=-limit, vmax=limit, rasterized=True)
    colorbar = fig.colorbar(image, ax=ax, label='Residual [ppm]')
    colorbar.ax.tick_params(direction='in')
    ax.set(xlabel=time_axis_label(event), ylabel='Wavelength [µm]')
    style.figure_title(ax, label, 'Residuals')
    style.style_axis(ax)
    style.save_figure(fig, filename)
    return fig


def limb_darkening(wavelengths, params, initial, filename, *, label, law):
    style.apply_publication_style()
    names = ('c1', 'c2') if law == 'power2' else ('u1', 'u2')
    initial = np.asarray(initial)
    if initial.ndim == 1:
        initial = np.broadcast_to(initial, (len(wavelengths), len(names)))
    fig, axes = plt.subplots(2, 1, figsize=style.figsize(5.0), sharex=True, constrained_layout=True)
    for index, (ax, name) in enumerate(zip(axes, names)):
        fitted = name in params
        values = params[name] if fitted else initial[:, index]
        errors = params.get(name+'_err') if fitted else np.zeros(len(wavelengths))
        ax.errorbar(wavelengths, values, yerr=errors, label='Fitted' if fitted else 'Fixed',
                    **{**style.DATA_MARKER_STYLE, 'color': style.accent_color_for_label(label)})
        ax.plot(wavelengths, initial[:, index], color=style.MODEL_COLOR, lw=1.2, label='Stellar model start')
        ax.set_ylabel(('c', 'α')[index] if law == 'power2' else name)
        style.style_axis(ax)
    axes[0].legend()
    axes[-1].set_xlabel('Wavelength [µm]')
    style.figure_title(axes[0], label, 'Limb darkening')
    style.save_figure(fig, filename)
    return fig


def noise(residuals, filename, *, label, white=None, wavelengths=None):
    from .outputs import _noise_binning_stats
    style.apply_publication_style()
    fig, ax = plt.subplots(figsize=style.figsize(4.4), constrained_layout=True)
    residuals = np.asarray(residuals)
    indices = np.unique(np.linspace(0, len(residuals)-1, min(4, len(residuals))).astype(int))
    curves = [((f'{float(wavelengths[i]):.2f} µm' if wavelengths is not None else f'Channel {i+1}'),
               residuals[i]) for i in indices]
    if white is not None:
        curves.insert(0, ('White light', np.asarray(white)))
    for index, (name, values) in enumerate(curves):
        bins, rms, _, _, expected, _ = _noise_binning_stats(values)
        color = style.model_palette_colors(len(curves))[index]
        ax.loglog(bins, rms*1e6, 'o-', color=color, ms=3, lw=1, label=name)
        ax.loglog(bins, expected*1e6, '--', color=color, lw=.8, alpha=.6)
    ax.plot([], [], '--', color='.4', lw=.8, label='White noise')
    ax.set(xlabel='Bin size [integrations]', ylabel='Residual RMS [ppm]')
    ax.legend(ncol=2)
    style.figure_title(ax, label, 'Noise')
    style.style_axis(ax, x_locator=False, y_locator=False)
    style.save_figure(fig, filename)
    return fig


def event_epoch(t0, period, model):
    """Display eclipse hours about the secondary event for a circular orbit."""
    epoch = float(np.asarray(t0).ravel()[0])
    return epoch + (float(np.asarray(period).ravel()[0])/2 if model == 'eclipse' else 0.)


def time_axis_label(event):
    return 'Hours from mid-eclipse' if event == 'eclipse' else 'Hours from mid-transit'


def grid_title(grid):
    """'R = 100', 'native resolution', ... for figure titles."""
    from .console import grid_description
    if not grid or grid == 'whitelight':
        return 'White light'
    text = grid_description(grid)
    for start in ('at ', 'on the ', 'in '):
        text = text.removeprefix(start)
    return text

"""Plot documentation figures from completed Koala CSV products.

    python tools/docs/plot_run_outputs.py --runs /path/to/docs_runs_20260927

The directory must contain eclipse/, phase_curve/, niriss_soss_order1/.
No model functions, posterior pickles, or fitting routines are imported.
Compact source tables and a checksum manifest are copied alongside the figures,
so maintainers can redraw archived documentation without rerunning Koala:

    python tools/docs/plot_run_outputs.py --archive docs/_static/data/runs
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_tutorial_figures import style, COLOR

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT=ROOT/'docs'/'_static'


def one(directory, pattern):
    paths=sorted(directory.glob(pattern))
    if len(paths)!=1:
        raise ValueError(f'Expected one {pattern} in {directory}, found {len(paths)}')
    return paths[0]


def save(fig, out, name):
    fig.savefig(out/name,dpi=180,facecolor='white')
    plt.close(fig)


def time_axis(table, epoch=None):
    if epoch is None:
        return table['time_from_t0_hr'].to_numpy()
    return (table['time_bjd'].to_numpy()-epoch)*24


def binned(x, y, n=65):
    groups=np.array_split(np.arange(len(x)),min(n,len(x)))
    return np.array([np.mean(x[g]) for g in groups]), np.array([np.mean(y[g]) for g in groups])


def white_panel(ax, table, epoch=None, residual=False, ppm=False):
    x=time_axis(table,epoch)
    good=~table['is_outlier'].astype(bool).to_numpy()
    y=table['residual_ppm'].to_numpy() if residual else table['flux'].to_numpy()
    if ppm and not residual: y=(y-1)*1e6
    ax.plot(x[good],y[good],'.',ms=3,color=COLOR,alpha=.3)
    bx,by=binned(x[good],y[good])
    ax.plot(bx,by,'o',ms=3,color=COLOR)
    if residual:
        ax.axhline(0,color='.3',lw=1)
        ax.set_ylabel('Residual (ppm)')
    else:
        model=table['bestfit_model'].to_numpy()
        if ppm: model=(model-1)*1e6
        ax.plot(x,model,color='k',lw=1.6)
        ax.set_ylabel('Relative flux (ppm)' if ppm else 'Normalized flux')
    return x


def soss(directory, out):
    ts=pd.read_csv(one(directory,'*_whitelight_timeseries.csv'))
    spectrum=pd.read_csv(one(directory,'*_R100.csv'))
    for name,residual in [('soss_whitelight.png',False),('soss_residuals.png',True)]:
        fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
        white_panel(ax,ts,residual=residual)
        ax.set_xlabel('Time from transit center (h)')
        save(fig,out,name)
    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
    ax.errorbar(spectrum.wavelength,spectrum.depth_ppm00,
        xerr=spectrum.wavelength_err,yerr=spectrum.depth_err_ppm00,
        fmt='o',ms=3.5,mfc='white',mec=COLOR,ecolor=COLOR,elinewidth=.9)
    ax.set(xlabel='Wavelength (µm)',ylabel='Transit depth (ppm)')
    save(fig,out,'soss_spectrum.png')
    curves=pd.read_csv(one(directory,'*_R100_lightcurves_wide.csv'))
    params=pd.read_csv(one(directory,'*_R100_bestfit_params.csv'))
    wl=pd.read_csv(one(directory,'*_whitelight_bestfit_params.csv'))
    t=curves['time'].to_numpy(); hours=(t-float(wl.t0.iloc[0]))*24
    x=t-t.min()
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    indices=np.unique(np.linspace(0,len(params)-1,5,dtype=int))
    colors=plt.cm.magma(np.linspace(.18,.72,len(indices)))
    for j,(i,color) in enumerate(zip(indices,colors)):
        trend=float(params.c.iloc[i])+float(params.v.iloc[i])*x
        flux=curves[f'flux_{i:03d}'].to_numpy()/trend
        offset=-.032*j
        ax.plot(hours,flux+offset,'.',ms=2,color=color,alpha=.3)
        bx,by=binned(hours,flux)
        ax.plot(bx,by+offset,'o',ms=3,color=color)
        ax.text(hours.max()+.12,1+offset,f'{params.wavelength.iloc[i]:.2f} µm',
            ha='left',va='center',fontsize=10,color=color)
    ax.set(xlabel='Time from transit center (h)',ylabel='Detrended flux + offset',
        xlim=(hours.min()-.1,hours.max()+1.1))
    save(fig,out,'soss_detrended.png')
    return {'channels':len(spectrum),'integrations':len(ts),
        'wavelength_min_um':float(spectrum.wavelength.min()),
        'wavelength_max_um':float(spectrum.wavelength.max()),
        'white_residual_rms_ppm':float(ts.loc[~ts.is_outlier.astype(bool),'residual_ppm'].std(ddof=0))}


def eclipse(directory,out):
    ts=pd.read_csv(one(directory,'*_whitelight_timeseries.csv'))
    emission=pd.read_csv(one(directory,'*_Rnative_emission.csv'))
    fig,axes=plt.subplots(3,1,figsize=(9,8),layout='constrained',
        gridspec_kw={'height_ratios':[1.5,1,1.5]})
    epoch=60001.3081322
    white_panel(axes[0],ts,epoch,ppm=True)
    white_panel(axes[1],ts,epoch,residual=True)
    axes[1].set_xlabel('Time from injected eclipse (h)')
    axes[2].errorbar(emission.wavelength,emission.eclipse_depth_ppm,
        xerr=emission.wavelength_err,
        yerr=np.array([emission.eclipse_depth_ppm_err_low,emission.eclipse_depth_ppm_err_high]),
        fmt='o',mfc='white',ms=4,color=COLOR,elinewidth=1)
    axes[2].axhline(100,color='k',ls='--',lw=1,label='Injected')
    axes[2].set(xlabel='Wavelength (µm)',ylabel='Eclipse depth (ppm)')
    axes[2].legend(frameon=False)
    save(fig,out,'eclipse_recovery.png')
    return {'channels':len(emission),'integrations':len(ts)}


def phase(directory,out):
    ts=pd.read_csv(one(directory,'*_whitelight_timeseries.csv'))
    emission=pd.read_csv(one(directory,'*_Rnative_emission.csv'))
    fig,axes=plt.subplots(3,1,figsize=(9,8),layout='constrained',
        gridspec_kw={'height_ratios':[1.4,1.3,1]})
    phase=(ts.time_bjd-60000)/2
    trend=ts.trend_model.to_numpy()
    flux=(ts.flux.to_numpy()/trend-1)*1e6
    model=(ts.bestfit_model.to_numpy()/trend-1)*1e6
    # Clip the display only: the transit was included in the actual fit.
    axes[0].plot(phase,flux,'.',ms=2,color=COLOR,alpha=.3)
    axes[0].plot(phase,model,color='k',lw=1.5)
    above=model>-200
    low=min(-150,float(np.nanpercentile(flux[above],1))-100)
    high=float(np.nanpercentile(flux[above],99))+150
    axes[0].set(xlabel='Orbital phase from transit',ylabel='Relative flux (ppm)',ylim=(low,high))
    axes[0].text(.5,.05,'Transit extends below the plotted range',transform=axes[0].transAxes,
        ha='center',fontsize=9,color='.3')
    for key,label,color in [('eclipse_depth_ppm','At eclipse (dayside)',COLOR),('nightside_flux_ppm','Nightside','#397b88')]:
        axes[1].errorbar(emission.wavelength,emission[key],
            yerr=np.array([emission[key+'_err_low'],emission[key+'_err_high']]),
            fmt='o',ms=4,mfc='white',color=color,elinewidth=1,label=label)
    axes[1].set(ylabel='Planet / star flux (ppm)')
    axes[1].legend(frameon=False,ncol=2)
    key='phase_offset_deg'
    axes[2].errorbar(emission.wavelength,emission[key],
        yerr=np.array([emission[key+'_err_low'],emission[key+'_err_high']]),
        fmt='o',mfc='white',ms=4,color=COLOR,elinewidth=1)
    axes[2].set(xlabel='Wavelength (µm)',ylabel='Peak offset before eclipse (deg)')
    save(fig,out,'phase_recovery.png')
    return {'channels':len(emission),'integrations':len(ts)}


def archive_inputs(runs, archive):
    """Copy only tabular products used in figures; never cache a partial run."""
    patterns={
        'niriss_soss_order1':['*_whitelight_timeseries.csv','*_whitelight_bestfit_params.csv',
            '*_R100.csv','*_R100_lightcurves_wide.csv','*_R100_bestfit_params.csv'],
        'eclipse':['*_whitelight_timeseries.csv','*_whitelight_bestfit_params.csv',
            '*_whitelight_bestfit_params_emission.csv','*_Rnative_emission.csv'],
        'phase_curve':['*_whitelight_timeseries.csv','*_whitelight_bestfit_params.csv',
            '*_whitelight_bestfit_params_emission.csv','*_Rnative_emission.csv']}
    files=[(name,one(runs/name,pattern)) for name,items in patterns.items() for pattern in items]
    manifest=[]
    for name,src in files:
        dest=archive/name/src.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve()!=dest.resolve(): shutil.copyfile(src,dest)
        manifest.append({'file':str(dest.relative_to(archive)),
            'source':src.name,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
            'columns':pd.read_csv(src,nrows=0).columns.tolist()})
    for name in patterns:
        cfg=runs/(name+'.yaml')
        if cfg.exists(): shutil.copyfile(cfg,archive/(name+'.yaml'))
    (archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    inventory={}
    for name in patterns:
        entries=[]
        for path in sorted((runs/name).rglob('*')):
            if not path.is_file(): continue
            entry={'file':str(path.relative_to(runs/name)), 'bytes':path.stat().st_size}
            if path.suffix=='.csv':
                entry['columns']=pd.read_csv(path,nrows=0).columns.tolist()
            elif path.suffix=='.npy':
                arr=np.load(path,mmap_mode='r',allow_pickle=False)
                entry.update(shape=list(arr.shape),dtype=str(arr.dtype))
            elif path.suffix=='.npz':
                with np.load(path,allow_pickle=False) as arr: entry['arrays']=arr.files
            entries.append(entry)
        inventory[name]=entries
    (archive/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--runs',type=Path)
    group.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source=archive_inputs(args.runs,args.output/'data'/'runs') if args.runs else args.archive
    style()
    summary={}
    for name,function in [('niriss_soss_order1',soss),('eclipse',eclipse),('phase_curve',phase)]:
        summary[name]=function(source/name,args.output)
    (args.output/'data'/'run_summary.json').parent.mkdir(parents=True,exist_ok=True)
    (args.output/'data'/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

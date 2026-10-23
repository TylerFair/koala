"""Draw the documentation schematics and the MIRI eclipse-time experiment.

Run from the repository root (no Koala fit or JAX import):
    python tools/docs/generate_tutorial_figures.py

To rebuild the compact white-light input from the original FITS file:
    python tools/docs/generate_tutorial_figures.py --eclipse-fits PATH

Only NumPy, Matplotlib, and (for FITS ingestion) Astropy are required.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from koala.plotting_style import apply_publication_style, LIGHTCURVE_COLOR

OUT = ROOT / 'docs' / '_static'
COLOR = LIGHTCURVE_COLOR

def style():
    apply_publication_style()
    plt.rcParams.update({'text.usetex': False, 'font.size': 12,
        'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.dpi': 180,
        'savefig.dpi': 180, 'savefig.bbox': None})

def finish(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor='white')
    plt.close(fig)

def trend_gallery():
    x=np.linspace(0,1,600); q=x-.5
    spot=.7*np.exp(-.5*((x-.43)/.05)**2)
    spot2=.45*np.exp(-.5*((x-.70)/.04)**2)
    ramp=.95*np.exp(-x/.12)+.25*x
    step=.6/(1+np.exp(-(x-.55)/.005))
    # Illustrative basis shapes only; amplitudes are deliberately arbitrary.
    curves={'none': np.zeros_like(x),'constant': np.full_like(x,.3),
        'linear': .65*q,'quadratic': 2*q*q-.2,'cubic': 4*q**3-.2*q,
        'quartic': 7*q**4-1.5*q*q,
        'explinear': ramp,'linear_discontinuity': .3*q+step,
        'spot': .2*q+spot,'2spot': .2*q+spot+spot2,
        'quadratic+spot': 2*q*q+spot,'spot+linear_discontinuity': .2*q+spot+step,
        'spot+explinear': ramp+spot,'2spot+explinear': ramp+spot+spot2}
    gp=.13*np.sin(12*x)+.20*np.sin(5*x+.7)
    curves['gp']=gp
    for base in ['linear','quadratic','cubic','quartic','explinear']:
        curves[base+'+gp']=curves[base]+gp
    fig,axes=plt.subplots(5,4,figsize=(9,8),layout='constrained')
    for ax,(name,y) in zip(axes.flat,curves.items()):
        ax.axhline(0,color='.8',lw=.7)
        ax.plot(x,y,color=COLOR,lw=1.9)
        ax.set_title(name,fontsize=9,loc='left')
        ax.set(xticks=[],yticks=[],xlim=(0,1),ylim=(-.5,1.4))
    fig.supxlabel('Time through the visit',fontsize=12)
    fig.supylabel('Trend relative to unity (illustrative amplitudes)',fontsize=12)
    finish(fig,'trend_gallery.png')

def harmonica_limbs():
    # Schema-v2 physical labels are mapped to neutral angular indices.
    # The original CSV convention is retained verbatim in the archive.
    data=np.genfromtxt(OUT/'data'/'wasp94_limbs.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    fig=plt.figure(figsize=(9,6.5),layout='constrained')
    grid=fig.add_gridspec(2,2)
    total=fig.add_subplot(grid[0,:])
    one=fig.add_subplot(grid[1,0])
    two=fig.add_subplot(grid[1,1],sharey=one)
    for ax,key,label,color in [(total,'depth_total_area','Total area',COLOR),
            (one,'depth_morning','Terminator one',COLOR),
            (two,'depth_evening','Terminator two','#397b88')]:
        ax.errorbar(data['wavelength'],data[key+'_median'],
            yerr=np.array([data[key+'_err_lo'],data[key+'_err_hi']]),
            fmt='o',ms=3,mfc='white',color=color,elinewidth=.8)
        ax.set_title(label,fontsize=12,loc='left')
        ax.set_xlabel('Wavelength (µm)')
    total.set_ylabel('Transit depth (ppm)')
    one.set_ylabel('Limb depth (ppm)')
    two.tick_params(labelleft=False)
    finish(fig,'harmonica_limbs.png')


def stacking(path=None):
    archive=OUT/'data'/'hatp18_stacked.csv'
    if path:
        import shutil
        shutil.copyfile(path,archive)
    data=np.genfromtxt(archive,delimiter=',',names=True)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,
        gridspec_kw={'height_ratios':[2,1]},layout='constrained')
    axes[0].errorbar(data['wavelength'],data['depth']*1e6,
        yerr=np.array([data['depth_err_lo'],data['depth_err_hi']])*1e6,
        fmt='o',ms=3,mfc='white',mec=COLOR,ecolor=COLOR,elinewidth=.7)
    axes[0].set(ylabel='Transit depth (ppm)')
    keys=[k for k in data.dtype.names if k.endswith('_weight')]
    labels=['Fixed power-2','Uniform quadratic','Sing quadratic']
    colors=[COLOR,'#397b88','#b16e24']
    axes[1].stackplot(data['wavelength'],*[data[k] for k in keys],labels=labels,colors=colors,alpha=.8)
    axes[1].set(xlabel='Wavelength (µm)',ylabel='Model weight',ylim=(0,1))
    axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.35),ncol=3)
    finish(fig,'model_stacking.png')


def occulted_fraction(time, epoch, period=2.6162644, a=17.04, b=.208, r=.0318):
    """Exact overlap of two circles for a circular, uniform-brightness eclipse."""
    phase=2*np.pi*(time-epoch)/period
    z=np.hypot(a*np.sin(phase),b*np.cos(phase))
    overlap=np.zeros_like(z)
    overlap[z <= 1-r]=1
    partial=(z > 1-r)&(z < 1+r)
    d=z[partial]
    term1=r*r*np.arccos(np.clip((d*d+r*r-1)/(2*d*r),-1,1))
    term2=np.arccos(np.clip((d*d+1-r*r)/(2*d),-1,1))
    term3=.5*np.sqrt(np.maximum(0,(-d+r+1)*(d+r-1)*(d-r+1)*(d+r+1)))
    overlap[partial]=(term1+term2-term3)/(np.pi*r*r)
    return np.where(np.cos(phase)>0,overlap,0)

def eclipse_scan(fits_path=None):
    path=OUT/'data'/'realistic_eclipse_white.csv'
    if fits_path:
        from astropy.io import fits
        with fits.open(fits_path) as hdus:
            t=np.asarray(hdus['TIME'].data,dtype=float)
            valid=np.all(np.isfinite(hdus['FLUX'].data) & np.isfinite(hdus['FLUX_ERROR'].data),axis=0)
            flux=np.sum(hdus['FLUX'].data[:,valid],axis=1)
            err=np.sqrt(np.sum(hdus['FLUX_ERROR'].data[:,valid]**2,axis=1))
        norm=np.median(flux)
        np.savetxt(path,np.column_stack([t,flux/norm,err/norm]),delimiter=',',
            header='time_bmjd_tdb,flux,flux_err',comments='')
    d=np.genfromtxt(path,delimiter=',',names=True)
    t,y,err=d['time_bmjd_tdb'],d['flux'],d['flux_err']
    true=60001.3081322
    x=t-t.min()
    # Estimate the exponential timescale from the baseline outside the true
    # event. The scan then holds this timescale fixed, fitting baseline,
    # slope, ramp amplitude, and eclipse depth by weighted least squares.
    oot=occulted_fraction(t,true)==0
    candidates=np.geomspace(.003,.3,160)
    best=(np.inf,None)
    for tau in candidates:
        X=np.column_stack([np.ones_like(t),x,np.exp(-x/tau)])
        coef=np.linalg.lstsq(X[oot]/err[oot,None],y[oot]/err[oot],rcond=None)[0]
        rss=np.sum(((y[oot]-X[oot]@coef)/err[oot])**2)
        if rss<best[0]: best=(rss,tau)
    tau=best[1]
    X=np.column_stack([np.ones_like(t),x,np.exp(-x/tau)])
    A=X/err[:,None]
    Q,_=np.linalg.qr(A,mode='reduced')
    yw=y/err
    residual=yw-Q@(Q.T@yw)
    null_chi2=float(residual@residual)
    epochs=np.unique(np.r_[np.linspace(t.min()-.03,t.max()+.03,700),true,true+.5])
    delta=[]; depths=[]; errors=[]
    for epoch in epochs:
        e=-occulted_fraction(t,epoch)/err
        e=e-Q@(Q.T@e)
        info=float(e@e)
        depth=max(0,float(e@residual)/info) if info>1e-12 else 0
        delta.append(depth*depth*info)
        depths.append(depth*1e6)
        errors.append(1e6/np.sqrt(info) if info>1e-12 else np.nan)
    delta=np.array(delta)
    best_i=int(np.argmax(delta)); best_epoch=epochs[best_i]
    template=-occulted_fraction(t,best_epoch)
    model_X=np.column_stack([X,template])
    coef=np.linalg.lstsq(model_X/err[:,None],yw,rcond=None)[0]
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    hours=(t-true)*24
    axes[0].plot(hours,(y-1)*1e6,'.',ms=2,color=COLOR,alpha=.24,rasterized=True)
    groups=np.array_split(np.arange(len(t)),80)
    axes[0].plot([hours[g].mean() for g in groups],[(y[g].mean()-1)*1e6 for g in groups],
        'o',ms=3,color=COLOR,label='Binned white light')
    axes[0].plot(hours,(model_X@coef-1)*1e6,color='k',lw=1.6,label='Ramp + eclipse fit')
    axes[0].axvline(0,color='.35',ls='--',lw=1)
    axes[0].set(xlabel='Time from injected eclipse (h)',ylabel='Relative flux (ppm)')
    axes[0].legend(frameon=False)
    axes[1].plot((epochs-true)*24,delta,color=COLOR,lw=1.7)
    axes[1].axvline(0,color='k',ls='--',label='Injected eclipse')
    axes[1].axvline(12,color='#b16e24',ls=':',lw=2,label='Wrong time origin (+0.5 d)')
    axes[1].axvspan(hours.min(),hours.max(),color=COLOR,alpha=.07,label='Observed time range')
    axes[1].set(xlabel='Trial eclipse time relative to injected epoch (h)',
        ylabel=r'$\chi^2_{\mathrm{baseline}}-\chi^2_{\mathrm{eclipse}}$',xlim=(hours.min()-.5,13))
    axes[1].legend(frameon=False,loc='upper right',fontsize=9)
    finish(fig,'eclipse_time_scan.png')
    result={'source':'realistic_eclipse_white.csv','n_integrations':len(t),'injected_epoch_bmjd_tdb':true,
        'injected_depth_ppm':70.,'baseline_tau_days':float(tau),
        'best_epoch_bmjd_tdb':float(best_epoch),'best_depth_ppm':float(depths[best_i]),
        'conditional_depth_error_ppm':float(errors[best_i]),
        'max_delta_chi2':float(delta[best_i]),'baseline_chi2':null_chi2,
        'method':'Nonnegative depth; baseline timescale estimated outside injected eclipse; conditional timing scan, not a global detection significance.'}
    (OUT/'data'/'eclipse_scan_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(OUT/'data'/'eclipse_time_scan.csv',np.column_stack([epochs,delta,depths,errors]),
        delimiter=',',header='epoch_bmjd_tdb,delta_chi2,depth_ppm,depth_error_ppm',comments='')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eclipse-fits',type=Path)
    parser.add_argument('--stacked-csv',type=Path)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'data').mkdir(exist_ok=True)
    style(); trend_gallery()
    harmonica_limbs(); stacking(args.stacked_csv); eclipse_scan(args.eclipse_fits)

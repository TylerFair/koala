#!/usr/bin/env python3
"""Seeded posterior-level validation plots for the 20260902 parity campaign."""
from __future__ import annotations

import argparse, json, pickle, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path('/scratch/midway3/tfairnington/accel_parity')
STAGED = Path('/scratch/midway3/tfairnington/accel_parity_staged')
CHUNK_RE = re.compile(r'.+_chunk_(\d+)_(\d+)\.pkl$')
COLORS = {'saved':'0.55', 'A':'#2878b5', 'B':'#e17c05', 'D':'black'}
NAMES = {'saved':'saved', 'A':'Laplace NUTS', 'B':'Laplace HMC-8', 'D':'control D'}

def q(x):
    x=np.asarray(x).reshape(-1); p=np.percentile(x,[16,50,84])
    return p[1], (p[2]-p[0])/2

def load_chunks(output, nchan):
    groups={}
    for p in Path(output,'chunks').glob('*.pkl'):
        m=CHUNK_RE.match(p.name)
        if m: groups.setdefault(p.name[:m.start(1)-6],[]).append((int(m[1]),int(m[2]),p))
    valid=[]
    for prefix, chunks in groups.items():
        chunks.sort(); cursor=0; maps=[]
        for lo,hi,p in chunks:
            if lo != cursor: break
            with p.open('rb') as f: maps.append(pickle.load(f))
            cursor=hi
        if cursor==nchan:
            sites=set(maps[0])
            if all(set(x)==sites for x in maps):
                valid.append((prefix,{s:np.concatenate([np.asarray(x[s]) for x in maps],axis=1) for s in sites}))
    if len(valid)!=1: raise RuntimeError(f'{output}: expected one {nchan}-channel group, got {[x[0] for x in valid]}')
    return valid[0][1]

def staged_reference(source_config):
    p=STAGED/(Path(source_config).stem+'_reference.pkl')
    with p.open('rb') as f: return pickle.load(f), p

def completed_results():
    by={}
    for p in sorted(ROOT.glob('*/**/result_final.json')):
        d=json.loads(p.read_text()); labels={x['label']:x for x in d.get('candidates',[]) if x.get('status')=='completed'}
        entry=by.setdefault(d['dataset'],{'source_config':d['source_config'],'labels':{},'results':[]})
        entry['labels'].update(labels); entry['results'].append(str(p))
    return {k:v for k,v in by.items() if {'A','B'} <= set(v['labels']) and (STAGED/(Path(v['source_config']).stem+'_reference.pkl')).exists()}

def seeded_selection(pool, seed, count=6):
    rng=np.random.default_rng(seed); names=np.array(sorted(pool)); selected=[]
    # Guarantee requested mode coverage when available, then fill randomly.
    for token in ['SOSS_ORDER1','G395H','SOSS_ORDER2','G395M','PRISM']:
        choices=[x for x in names if token in x and x not in selected]
        if choices: selected.append(str(rng.choice(choices)))
    remaining=[x for x in names if x not in selected]
    if len(selected)<count: selected += list(rng.choice(remaining,count-len(selected),replace=False))
    return selected[:count]

def channel_values(a, site, ch):
    x=np.asarray(a[site])[:,ch]
    return x.reshape(x.shape[0],-1)[:,0]

def depth_values(a,ch):
    if 'depths' in a: return channel_values(a,'depths',ch)*1e6
    return channel_values(a,'rors',ch)**2*1e6

def fit_offset_slope(w, delta, sigma):
    x=w-np.average(w,weights=1/sigma**2); X=np.column_stack([np.ones(len(x)),x]); W=1/sigma**2
    beta=np.linalg.solve(X.T@(W[:,None]*X),X.T@(W*delta)); return beta

def plot_dataset(name, waves, samples, channels, out):
    common=set.intersection(*(set(x) for x in samples.values()))
    sites=['depth']+[s for s in ['c','v','c1','c2','A_spot','A_jump','A'] if s in common]
    ncol=len(sites); fig=plt.figure(figsize=(3.15*ncol,15),layout='constrained')
    gs=GridSpec(6,ncol,figure=fig,height_ratios=[1,1,1,1,1.45,0.85])
    order=[x for x in ['saved','A','B','D'] if x in samples]
    for row,ch in enumerate(channels):
        for col,site in enumerate(sites):
            ax=fig.add_subplot(gs[row,col]); vals={k:(depth_values(v,ch) if site=='depth' else channel_values(v,site,ch)) for k,v in samples.items()}
            lo=min(np.percentile(v,.5) for v in vals.values()); hi=max(np.percentile(v,99.5) for v in vals.values()); bins=np.linspace(lo,hi,31)
            ax.hist(vals['saved'],bins=bins,density=True,color='0.75',alpha=.55,label='saved')
            notes=[]; rm,rs=q(vals['saved'])
            for k in order:
                m,s=q(vals[k]); style='--' if k=='D' else '-'
                if k!='saved': ax.hist(vals[k],bins=bins,density=True,histtype='step',lw=1.5,ls=style,color=COLORS[k],label=NAMES[k])
                shift=(m-rm)/rs if rs else np.nan
                notes.append(f'{k}: {m:.5g}±{s:.2g} [{shift:+.2f}σ]')
            ax.text(.02,.98,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=6.1)
            ax.set_yticks([]); ax.set_title(f'ch {ch}, λ={waves[ch]:.4g} µm\n{site}' if row==0 else site,fontsize=9)
            if row==0 and col==ncol-1: ax.legend(fontsize=7,loc='lower right')
    # Full spectrum with residual inset below it.
    split=max(1,(ncol+1)//2); ax=fig.add_subplot(gs[4,:split]); axr=fig.add_subplot(gs[5,:split],sharex=ax)
    med={}; sig={}
    for k,a in samples.items():
        arr=np.stack([depth_values(a,i) for i in range(len(waves))],axis=1); p=np.percentile(arr,[16,50,84],axis=0); med[k]=p[1]; sig[k]=(p[2]-p[0])/2
    span=np.ptp(waves); offsets={'saved':0,'A':-.0015*span,'B':.0015*span,'D':0}
    for k in order:
        beta=fit_offset_slope(waves,med[k]-med['saved'],sig['saved']) if k!='saved' else (0,0)
        label=NAMES[k] if k=='saved' else f'{NAMES[k]}: Δ={beta[0]:+.2f} ppm, slope={beta[1]:+.2f} ppm/µm'
        fmt='o' if k!='D' else 'none'; ax.errorbar(waves+offsets[k],med[k],yerr=sig[k],fmt=fmt,ms=2,lw=.7,color=COLORS[k],label=label)
        if k!='saved': axr.errorbar(waves+offsets[k],med[k]-med['saved'],yerr=sig['saved'],fmt='.',ms=2,lw=.5,color=COLORS[k])
    ax.set_ylabel('depth (ppm)'); ax.legend(fontsize=7,ncol=1); ax.set_title('Transmission spectrum')
    axr.axhline(0,color='.5',lw=.7); axr.set(xlabel='wavelength (µm)',ylabel='candidate − saved\n(ppm)')
    ar=fig.add_subplot(gs[4:,split:]);
    for k in ['A','B']:
        ar.plot(waves,sig[k]/sig['saved'],'.',ms=3,color=COLORS[k],label=NAMES[k])
    ar.axhline(1,color='.5',lw=.8); ar.set(xlabel='wavelength (µm)',ylabel='σ(candidate) / σ(saved)',title='Error-bar ratios'); ar.legend()
    fig.suptitle(name,fontsize=13); fig.savefig(out,dpi=150); plt.close(fig)
    rows=[]
    for k in ['A','B']:
        for site in ['depth','c','v','c1','c2']:
            if site!='depth' and site not in common: continue
            for ch in range(len(waves)):
                rv=depth_values(samples['saved'],ch) if site=='depth' else channel_values(samples['saved'],site,ch)
                cv=depth_values(samples[k],ch) if site=='depth' else channel_values(samples[k],site,ch)
                rm,rs=q(rv); cm,cs=q(cv); rows.append((name,k,site,(cm-rm)/rs,cs/rs,len(rv),len(cv)))
    return rows

def summary_plot(rows,out):
    sites=['depth','c','v','c1','c2']; fig,axs=plt.subplots(2,5,figsize=(16,6.5),layout='constrained')
    for j,site in enumerate(sites):
        subset=[r for r in rows if r[2]==site]; ns=[]
        for k in ['A','B']:
            z=np.array([r[3] for r in subset if r[1]==k]); rr=np.array([r[4] for r in subset if r[1]==k])
            axs[0,j].hist(z,bins=35,density=True,histtype='step',lw=1.5,color=COLORS[k],label=NAMES[k]); axs[1,j].hist(rr,bins=35,density=True,histtype='step',lw=1.5,color=COLORS[k])
            ns += [(r[5],r[6]) for r in subset if r[1]==k]
        if ns:
            sd=np.median([np.sqrt(np.pi/2*(1/nr+1/nc)) for nr,nc in ns]); x=np.linspace(-4*sd,4*sd,300); pdf=np.exp(-.5*(x/sd)**2)/(sd*np.sqrt(2*np.pi)); axs[0,j].plot(x,pdf,'k--',lw=1,label=f'MC Gaussian σ={sd:.3f}')
            sr=np.median([np.sqrt(1/(2*(nr-1))+1/(2*(nc-1))) for nr,nc in ns]); x=np.linspace(1-4*sr,1+4*sr,300); pdf=np.exp(-.5*((x-1)/sr)**2)/(sr*np.sqrt(2*np.pi)); axs[1,j].plot(x,pdf,'k--',lw=1)
        axs[0,j].axvline(0,color='.6',lw=.7); axs[1,j].axvline(1,color='.6',lw=.7); axs[0,j].set_title(site); axs[1,j].set_xlabel('σ ratio')
    axs[0,0].set_ylabel('density: median shift / saved σ'); axs[1,0].set_ylabel('density: error-bar ratio'); axs[0,0].legend(fontsize=7)
    fig.suptitle('Posterior parity across six seeded datasets'); fig.savefig(out,dpi=150); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260902); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False); pool=completed_results(); chosen=seeded_selection(pool,args.seed); rng=np.random.default_rng(args.seed)
    allrows=[]; records=[]
    for name in chosen:
        info=pool[name]; ref,refp=staged_reference(info['source_config']); waves=np.asarray(ref['wavelengths']); sam={'saved':ref['samples']}
        for label in ['A','B']: sam[label]=load_chunks(info['labels'][label]['output'],len(waves))
        if 'D' in info['labels']:
            sam['D']=load_chunks(info['labels']['D']['output'],len(waves))
        channels=np.sort(rng.choice(len(waves),4,replace=False)).tolist(); png=args.output/(name+'.png')
        allrows += plot_dataset(name,waves,sam,channels,png); records.append((name,channels,refp,info['labels'].keys()))
    summary_plot(allrows,args.output/'summary.png')
    lines=['# Posterior validation plots','',f'- Random seed: `{args.seed}`',f'- Eligible completed A+B datasets: `{len(pool)}`',f'- Selection: mode-stratified seeded draw (one each from available SOSS order 1, G395H, SOSS order 2, G395M, PRISM; remaining slots sampled without replacement).','', '| Dataset | Channels (zero-based) | Optional D |','|---|---|---|']
    for n,ch,_,labs in records: lines.append(f'| {n} | {", ".join(map(str,ch))} | {"yes" if "D" in labs else "no"} |')
    lines += ['','## Aggregate numbers','', '| Candidate | Site | N channels | median shift/saved σ | 95% | median σ ratio | 5–95% |','|---|---|---:|---:|---:|---:|---:|']
    for k in ['A','B']:
        for site in ['depth','c','v','c1','c2']:
            z=np.array([r[3] for r in allrows if r[1]==k and r[2]==site]); sr=np.array([r[4] for r in allrows if r[1]==k and r[2]==site])
            if len(z): lines.append(f'| {NAMES[k]} | {site} | {len(z)} | {np.median(z):+.4f} | [{np.percentile(z,2.5):+.4f}, {np.percentile(z,97.5):+.4f}] | {np.median(sr):.4f} | [{np.percentile(sr,5):.4f}, {np.percentile(sr,95):.4f}] |')
    lines += ['','The dashed references in `summary.png` use the finite-draw normal approximations: median-shift SD `sqrt(pi/2 * (1/N_saved + 1/N_candidate))`; sigma-ratio SD `sqrt(1/(2(N_saved-1)) + 1/(2(N_candidate-1)))`. These are Monte-Carlo references, not fitted claims.','', '## Reproduce','', '```bash','JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \\','  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/plot_validation_posteriors.py \\','  --seed 20260902 --output acceleration_reports/validation_plots_reproduction','```','']
    (args.output/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'chosen':chosen,'outputs':len(chosen)+2},indent=2))
if __name__=='__main__': main()

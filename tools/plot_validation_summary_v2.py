#!/usr/bin/env python3
"""ESS-aware aggregate posterior validation; writes only new v2 artifacts."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from plot_validation_posteriors import (
    ROOT, NAMES, COLORS, completed_results, seeded_selection, staged_reference,
    load_chunks, depth_values, q,
)

NOISE=Path('/scratch/midway3/tfairnington/accel_stage_inputs/references')
SITES=['depth','c','v','c1','c2']

def matrix(samples, site):
    if site=='depth':
        return np.stack([depth_values(samples,i) for i in range(np.asarray(samples['rors']).shape[1])],axis=1)
    x=np.asarray(samples[site]); return x.reshape(x.shape[0],x.shape[1],-1)[...,0]

def bulk_ess(x):
    # Checkpoints contain one retained chain; ArviZ receives [chain, draw, channel].
    ds=az.from_dict(posterior={'x':np.asarray(x)[None,:,:]})
    return np.asarray(az.ess(ds,method='bulk')['x'],dtype=float)

def noise_rows():
    out={s:{'shift':[],'ratio':[]} for s in SITES}
    # Only high-resolution production nulls match the plotted high-res products.
    paths=sorted(NOISE.glob('*_high_resolution_*_noise_floor.json'))
    aliases={'depths[0]':'depth'}
    for p in paths:
        for r in json.loads(p.read_text()).get('pairwise_rows',[]):
            site=aliases.get(r['site'],r['site'])
            if site not in out: continue
            # JSON stores absolute median shifts. Mirror them to form a signed null.
            z=float(r['abs_median_shift_pooled_sigma'])
            out[site]['shift'] += [-z,z]
            out[site]['ratio'].append(float(r['sigma_ratio']))
    return out,paths

def gaussian_mixture(x, scales, center=0.):
    scales=np.maximum(np.asarray(scales,dtype=float),1e-12)
    return np.mean(np.exp(-.5*((x[:,None]-center)/scales[None,:])**2)/(np.sqrt(2*np.pi)*scales[None,:]),axis=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seed',type=int,default=20260902); a=ap.parse_args()
    png=a.output_dir/'summary_v2.png'; readme=a.output_dir/'README_v2.md'
    if png.exists() or readme.exists(): raise FileExistsError('v2 output already exists; refusing overwrite')
    pool=completed_results(); chosen=seeded_selection(pool,a.seed); rows=[]; ess_rows=[]
    for name in chosen:
        info=pool[name]; ref,_=staged_reference(info['source_config']); n=len(ref['wavelengths'])
        samples={'saved':ref['samples'],'A':load_chunks(info['labels']['A']['output'],n),'B':load_chunks(info['labels']['B']['output'],n)}
        common=set.intersection(*(set(v) for v in samples.values()))
        for site in SITES:
            if site!='depth' and site not in common: continue
            arrays={k:matrix(v,site) for k,v in samples.items()}; esses={k:bulk_ess(v) for k,v in arrays.items()}
            for k in ['saved','A','B']:
                ess_rows.extend((name,k,site,float(x)) for x in esses[k])
            for k in ['A','B']:
                for ch in range(n):
                    rm,rs=q(arrays['saved'][:,ch]); cm,cs=q(arrays[k][:,ch])
                    rows.append((name,k,site,(cm-rm)/rs,cs/rs,esses['saved'][ch],esses[k][ch]))
    null,noise_paths=noise_rows(); fig,axs=plt.subplots(2,5,figsize=(16,7.1),layout='constrained')
    for j,site in enumerate(SITES):
        sub=[r for r in rows if r[2]==site]
        zall=np.asarray([r[3] for r in sub]); rall=np.asarray([r[4] for r in sub]); nz=np.asarray(null[site]['shift']); nr=np.asarray(null[site]['ratio'])
        zlim=max(.22,np.nanpercentile(np.abs(np.r_[zall,nz]),99.5)*1.12); rlo,rhi=np.nanpercentile(np.r_[rall,nr],[.5,99.5]); pad=.1*(rhi-rlo); rlo-=pad; rhi+=pad
        zbins=np.linspace(-zlim,zlim,38); rbins=np.linspace(rlo,rhi,38)
        if len(nz): axs[0,j].hist(nz,bins=zbins,density=True,color='.72',alpha=.55,label=f'production null (n={len(nz)})')
        if len(nr): axs[1,j].hist(nr,bins=rbins,density=True,color='.72',alpha=.55)
        top=[]; bottom=[]
        for k in ['A','B']:
            ss=[r for r in sub if r[1]==k]; z=np.asarray([r[3] for r in ss]); ratio=np.asarray([r[4] for r in ss])
            sdz=np.sqrt(np.pi/2*(1/np.asarray([r[5] for r in ss])+1/np.asarray([r[6] for r in ss])))
            sdr=np.sqrt(1/(2*np.asarray([r[5] for r in ss]))+1/(2*np.asarray([r[6] for r in ss])))
            axs[0,j].hist(z,bins=zbins,density=True,histtype='step',lw=1.4,color=COLORS[k],label=NAMES[k])
            x=np.linspace(-zlim,zlim,500); axs[0,j].plot(x,gaussian_mixture(x,sdz),color=COLORS[k],ls='--',lw=1)
            axs[1,j].hist(ratio,bins=rbins,density=True,histtype='step',lw=1.4,color=COLORS[k])
            x=np.linspace(rlo,rhi,500); axs[1,j].plot(x,gaussian_mixture(x,sdr,1),color=COLORS[k],ls='--',lw=1)
            top.append(f'{k}: med {np.median(z):+.3f}; 5–95% [{np.percentile(z,5):+.3f},{np.percentile(z,95):+.3f}]')
            bottom.append(f'{k}: med {np.median(ratio):.3f}; 5–95% [{np.percentile(ratio,5):.3f},{np.percentile(ratio,95):.3f}]')
        axs[0,j].text(.02,.98,'\n'.join(top),transform=axs[0,j].transAxes,va='top',fontsize=7)
        axs[1,j].text(.02,.98,'\n'.join(bottom),transform=axs[1,j].transAxes,va='top',fontsize=7)
        axs[0,j].axvline(0,color='.45',lw=.7); axs[1,j].axvline(1,color='.45',lw=.7); axs[0,j].set_title(site); axs[1,j].set_xlabel('candidate σ / saved σ')
    axs[0,0].set_ylabel('density: median shift / saved σ'); axs[1,0].set_ylabel('density: sigma ratio'); axs[0,0].legend(fontsize=6.5,loc='lower left')
    fig.suptitle('Posterior parity: ESS-aware MC mixtures and production seed nulls'); fig.savefig(png,dpi=150); plt.close(fig)
    lines=['# v2: ESS-aware summary addendum','',f'- Seed and six datasets: unchanged from `README.md` (`{a.seed}`).','- Output: `summary_v2.png` (150 dpi).','- Dashed curves: averages of per-channel Gaussian references using actual ArviZ bulk ESS.','- Grey fill: matching high-resolution production seed-to-seed null rows. The source JSON stores absolute median shifts, so shift values are mirrored about zero for display. Sigma ratios are plotted as stored.','', '## Median bulk ESS by site','', '| Chain | Site | N channel-datasets | Median ESS | 5–95% ESS |','|---|---|---:|---:|---:|']
    for k in ['saved','A','B']:
        for site in SITES:
            e=np.asarray([r[3] for r in ess_rows if r[1]==k and r[2]==site])
            if len(e): lines.append(f'| {NAMES.get(k,k)} | {site} | {len(e)} | {np.median(e):.1f} | [{np.percentile(e,5):.1f}, {np.percentile(e,95):.1f}] |')
    lines += ['','## Production-null inputs','']+[f'- `{p}`' for p in noise_paths]+['','## Reproduce','', '```bash','JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \\','  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python tools/plot_validation_summary_v2.py \\','  --seed 20260902 --output-dir acceleration_reports/validation_plots_reproduction','```','']
    readme.write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps({'png':str(png),'readme':str(readme),'datasets':chosen},indent=2))

if __name__=='__main__': main()

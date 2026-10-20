"""Read-only WASP-94 bundle comparison. Run it on a compute node.

Usage: python tools/time_shift_validation/compare_wasp94_bundle.py BUNDLE SOURCE_DATA OUTPUT_PREFIX
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from koala.time_shift_response import projected_position,half_radius_responses,solve_response
from models.jaxoplanet.core import compute_transit_model

jax.config.update('jax_enable_x64',True)
COLUMNS=['d_dt0_dRL_s_per_Rstar','d_dt0_dRT_s_per_Rstar','d_depth_dRL','d_depth_dRT']


def main(bundle, source, output):
    all_rows=[];summary={}
    for detector in ['soss_o1','g395h_nrs1','g395h_nrs2']:
        meta=json.loads((bundle/f'wasp94_{detector}_timeshift.json').read_text())
        with (bundle/f'wasp94_{detector}_timeshift.csv').open() as f:
            ref={row['bin_id']:row for row in csv.DictReader(f) if row['bin_id']!='white'}
        file=source/f'{detector}.npz'
        data=np.load(file)
        # Whole-file hashes prove timestamps, error arrays and masks are exactly
        # those used by the independent calibration, including heavily masked bins.
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        for info in meta['per_bin']:
            if info['bin_id']!='white':
                assert digest==info['source_sha256'][f'data/{detector}.npz']
        epoch=meta['white_light_epoch']['bjd_tdb']-2400000.5
        t=jnp.asarray(data['time']-epoch)
        x=jnp.asarray((data['time']-np.mean(data['time']))/np.ptp(data['time']))
        orbit=(meta['period_days'],0.,meta['a_rs']['value'],meta['b'],float(meta['eccentricity']),0.)
        def model(theta,fixed_u1):
            r,offset,u2=theta[:3]
            p=dict(period=jnp.array([orbit[0]]),t0=jnp.array([offset/86400]),a_rs=jnp.array([orbit[2]]),
                   b=jnp.array([orbit[3]]),ecc=jnp.array([orbit[4]]),omega=jnp.array([orbit[5]]),
                   rors=jnp.array([r]),u=jnp.array([fixed_u1,u2]),_jaxoplanet_kernel='quadratic_contact_safe')
            trend=sum(theta[3+k]*x**k for k in range(len(theta)-3))
            return (1+compute_transit_model(p,t))*trend
        derivative=jax.jit(jax.jacfwd(model))
        position=jax.jit(lambda tt,offset:projected_position(tt,orbit[0],offset/86400,*orbit[2:])[0])
        runtimes=[];parity=[]
        for info in meta['per_bin']:
            bid=info['bin_id']
            if bid=='white':continue
            row=ref[bid];i=int(bid)
            start=time.perf_counter()
            u=info['ld_coefficients'];r=np.sqrt(float(row['depth']))
            theta=np.array([r,float(row['dt0_s']),u[1],*info['baseline_coefficients']])
            jac=np.asarray(derivative(theta,u[0]))
            p,v=jax.jvp(lambda tt:position(tt,theta[1]),(t,),(jnp.ones_like(t),))
            h=half_radius_responses(p,v,r,u)
            trend=sum(theta[3+k]*np.asarray(x)**k for k in range(len(theta)-3))
            h*=trend[:,None]
            mask=np.asarray(data['mask'][i],bool)
            assert int(mask.sum())==info['n_used']
            error=np.sqrt(data['err'][i]**2+info['jitter_fraction']**2)
            assert info['u1_fixed'], 'This validation adapter expects the bundle fixed-u1 treatment'
            result=solve_response(jac[mask],h[mask],error[mask])
            values=[*result[1],*(2*r*result[0])]
            record=dict(detector=detector,bin_id=bid,n_used=int(mask.sum()),
                        generating_RL=info['measured_radii'][0],generating_RT=info['measured_radii'][1],
                        spherical_radius=float(r),linearity_max_frac_dev=float(row['linearity_max_frac_dev']))
            for key,val in zip(COLUMNS,values):
                expected=float(row[key]);record[key]=float(val);record[key+'_reference']=expected
                record[key+'_difference_percent']=float(100*(val/expected-1))
            record['seconds']=time.perf_counter()-start;runtimes.append(record['seconds'])
            record['parity_relative']=float(np.linalg.norm(h[mask].sum(1)-jac[mask,0])/np.linalg.norm(jac[mask,0]))
            parity.append(record['parity_relative']);all_rows.append(record)
        det_rows=[r for r in all_rows if r['detector']==detector]
        summary[detector]=dict(n_bins=len(det_rows),source_file=str(file),source_sha256=digest,
                              cost_median_s=float(np.median(runtimes)),cost_max_s=float(max(runtimes)),
                              radius_sum_max_relative=float(max(parity)),columns={})
        for key in COLUMNS:
            diffs=np.abs([r[key+'_difference_percent'] for r in det_rows])
            summary[detector]['columns'][key]=dict(median_percent=float(np.median(diffs)),max_percent=float(max(diffs)),
                                                 max_bin_id=det_rows[int(np.argmax(diffs))]['bin_id'],
                                                 agreement_within_0_3_percent=bool(np.all(diffs <= .3)))
        print(detector,json.dumps(summary[detector]),flush=True)
    Path(str(output)+'.json').write_text(json.dumps(summary,indent=2)+'\n')
    with Path(str(output)+'.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_rows)
    if not all(column['agreement_within_0_3_percent'] for det in summary.values() for column in det['columns'].values()):
        raise SystemExit('FAIL: spherical-point responses differ from the delivered unequal-limb-point Jacobians beyond 0.3%; see comparison JSON/CSV.')

if __name__=='__main__':main(*(Path(s) for s in sys.argv[1:]))

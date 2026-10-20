"""Diagnose bundle/spherical operating points with independent full 2-D refits."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from reference_half_disk import two_half_flux
from koala.time_shift_response import projected_position,half_radius_responses,solve_response
from models.jaxoplanet.core import compute_transit_model
jax.config.update('jax_enable_x64',True)


def main(root,output):
    bundle=root/'calibration/wasp94';results=[]
    for det,bid in [('g395h_nrs1','0'),('soss_o1','100'),('g395h_nrs2','60')]:
        meta=json.loads((bundle/f'wasp94_{det}_timeshift.json').read_text())
        info=next(b for b in meta['per_bin'] if b['bin_id']==bid)
        row=next(r for r in csv.DictReader((bundle/f'wasp94_{det}_timeshift.csv').open()) if r['bin_id']==bid)
        data=np.load(root/'wasp94_final/data'/f'{det}.npz');i=int(bid)
        mask=data['mask'][i];original_time=data['time']
        epoch=meta['white_light_epoch']['bjd_tdb']-2400000.5
        t=jnp.asarray((original_time-epoch)[mask])
        x=jnp.asarray(((original_time-np.mean(original_time))/np.ptp(original_time))[mask])
        u=info['ld_coefficients'];r=np.sqrt(float(row['depth']));offset=float(row['dt0_s'])
        base=np.asarray(sum(c*x**k for k,c in enumerate(info['baseline_coefficients'])))
        orbit=(meta['period_days'],0.,meta['a_rs']['value'],meta['b'],float(meta['eccentricity']),0.)
        error=np.sqrt(data['err'][i,mask]**2+info['jitter_fraction']**2)
        def model(theta):
            p=dict(period=jnp.array([orbit[0]]),t0=jnp.array([theta[1]/86400]),a_rs=jnp.array([orbit[2]]),
                   b=jnp.array([orbit[3]]),ecc=jnp.array([orbit[4]]),omega=jnp.array([0.]),
                   rors=jnp.array([theta[0]]),u=jnp.array([u[0],theta[2]]),_jaxoplanet_kernel='quadratic_contact_safe')
            trend=sum(theta[3+k]*x**k for k in range(len(theta)-3))
            return (1+compute_transit_model(p,t))*trend
        signal=jax.jit(model);derivative=jax.jit(jax.jacfwd(model))
        theta=np.array([r,offset,u[1],*info['baseline_coefficients']])
        jac=np.asarray(derivative(theta))
        sphere_orbit=(orbit[0],offset/86400,*orbit[2:])
        p,v=jax.jvp(lambda tt:projected_position(tt,*sphere_orbit)[0],(t,),(jnp.ones_like(t),))
        h=half_radius_responses(p,v,r,u)*base[:,None]
        predicted=solve_response(jac,h,error)
        predicted=np.array([*predicted[1],*(2*r*predicted[0])])
        ref=np.array([float(row[k]) for k in ['d_dt0_dRL_s_per_Rstar','d_dt0_dRT_s_per_Rstar','d_depth_dRL','d_depth_dRT']])
        result=dict(detector=det,bin_id=bid,measured_radii=info['measured_radii'],spherical_prediction=predicted.tolist(),bundle=ref.tolist())
        start=time.perf_counter()
        for mode,radii,gen_orbit in [('spherical',[r,r],sphere_orbit),('measured',info['measured_radii'],orbit)]:
            def refit(rr):
                flux=two_half_flux(np.asarray(t),gen_orbit,*rr,u)*base
                fit=least_squares(lambda z:(np.asarray(signal(z))-flux)/error,theta,
                    jac=lambda z:np.asarray(derivative(z))/error[:,None],x_scale='jac',
                    bounds=([.01,-1800.,-1.,*([-10.]*(len(theta)-3))],[.5,1800.,2.,*([10.]*(len(theta)-3))]),
                    ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=200)
                assert fit.success
                return np.array([fit.x[0]**2,fit.x[1]])
            epsilon=2e-5;fds=[]
            for limb in range(2):
                delta=np.zeros(2);delta[limb]=epsilon
                fds.append((refit(np.array(radii)+delta)-refit(np.array(radii)-delta))/(2*epsilon))
            value=np.array([fds[0][1],fds[1][1],fds[0][0],fds[1][0]])
            target=predicted if mode=='spherical' else ref
            result[mode+'_independent']=value.tolist()
            result[mode+'_difference_percent']=(100*(value/target-1)).tolist()
        result['seconds']=time.perf_counter()-start
        results.append(result)
        print(json.dumps(result),flush=True)
        output.write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main(*(Path(p) for p in sys.argv[1:]))

"""Independent stellar-surface area integration, without elliptic integrals."""
import json, pathlib, runpy, sys
import numpy as np
from scipy.integrate import quad
import batman
fixture=runpy.run_path('tests/test_spectro_free_t0.py')
_, times, trace=fixture['traced'](True)
times=np.asarray(times); flux=np.asarray(trace['obs']['fn'].loc)

def area_flux(z,k,u,eps=2e-13,nested=False):
    if z>=1+k: return 1.,0.
    u1,u2=u; lo=max(-1.,z-k); hi=min(1.,z+k)
    crossing=(1+z*z-k*k)/(2*z) if z else hi
    bounds=sorted(set([lo,hi]+([crossing] if lo<crossing<hi else [])))
    def column(x):
        a=max(0.,1-x*x); y=np.sqrt(max(0.,min(a,k*k-(x-z)**2)))
        if a==0: return 0.
        if nested:
            def intensity(theta):
                yy=y*np.sin(theta); mu=np.sqrt(max(0.,a-yy*yy))
                return (1-u1*(1-mu)-u2*(1-mu)**2)*y*np.cos(theta)
            return 2*quad(intensity,0,np.pi/2,epsabs=eps,epsrel=eps)[0]
        return 2*y*(1-u1-u2-u2*a)+2*u2*y**3/3+(u1+2*u2)*(y*np.sqrt(max(0.,a-y*y))+a*np.arcsin(min(1.,y/np.sqrt(a))))
    integrals=[quad(column,a,b,epsabs=eps,epsrel=eps,limit=200) for a,b in zip(bounds[:-1],bounds[1:])]
    norm=np.pi*(1-u1/3-u2/6)
    return 1-sum(x[0] for x in integrals)/norm,sum(x[1] for x in integrals)/norm
rows=[]
for c,(k,offset,u) in enumerate(zip([.12,.18],[-20,30],[[.3,.2],[.4,.1]])):
    p=batman.TransitParams();p.t0=offset/86400;p.per=4.;p.a=9.;p.rp=k
    p.inc=np.degrees(np.arccos(.35/9));p.ecc=0.;p.w=90.;p.u=u;p.limb_dark='quadratic'
    ref=batman.TransitModel(p,times).light_curve(p)
    phase=2*np.pi*(times-offset/86400)/4
    zs=np.sqrt(9.**2*np.sin(phase)**2+.35**2*np.cos(phase)**2)
    truth=np.array([area_flux(z,k,u)[0] for z in zs])
    index=int(np.argmax(np.abs(flux[c]-ref)))
    direct,err=area_flux(zs[index],k,u,eps=2e-13,nested=True)
    tight,_=area_flux(zs[index],k,u,eps=2e-14,nested=True)
    rows.append(dict(channel=c,radius=k,epoch_seconds=offset,ld=u,worst_index=index,time=float(times[index]),separation=float(zs[index]),koala=float(flux[c,index]),batman=float(ref[index]),surface_integral=float(direct),surface_integral_tight=float(tight),nested_vs_analytic=float(direct-truth[index]),tight_vs_default=float(tight-direct),quadrature_error_estimate=err,max_koala_vs_integral=float(np.max(np.abs(flux[c]-truth))),max_batman_vs_integral=float(np.max(np.abs(ref-truth))),worst_koala_error=float(flux[c,index]-direct),worst_batman_error=float(ref[index]-direct)))
out=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parent/'reference_data/quadratic_surface_integral.json';out.write_text(json.dumps(rows,indent=2)+'\n');print(out.read_text())
assert all(abs(r['nested_vs_analytic'])<1e-12 and abs(r['tight_vs_default'])<1e-12 for r in rows)

assert all(r['max_koala_vs_integral'] < 1e-12 for r in rows)
assert all(r['max_batman_vs_integral'] > 1e-9 for r in rows)

"""Independent 2-D quadratic-intensity cubature for timing-response tests.

Integrate Cartesian slices in the instantaneous velocity frame. The inner
intensity integral is analytic; adaptive quadrature integrates the outer slice.
This shares neither boundary quadrature nor JAX orbit code with production.
"""
import numpy as np
from scipy.integrate import quad


def sky_orbit(t, period, t0, a, b, ecc=0., omega=0.):
    f0 = np.pi/2-omega
    e0 = 2*np.arctan2(np.sqrt(1-ecc)*np.sin(f0/2),np.sqrt(1+ecc)*np.cos(f0/2))
    mean = e0-ecc*np.sin(e0)+2*np.pi*(np.asarray(t)-t0)/period
    anomaly = mean.copy()
    for _ in range(20):
        anomaly -= (anomaly-ecc*np.sin(anomaly)-mean)/(1-ecc*np.cos(anomaly))
    f = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(anomaly/2),np.sqrt(1-ecc)*np.cos(anomaly/2))
    argument = f+omega
    rho = a*(1-ecc*np.cos(anomaly))
    cosi = b*(1+ecc*np.sin(omega))/(a*(1-ecc**2))
    p = np.column_stack((-rho*np.cos(argument),rho*np.sin(argument)*cosi))
    rate = 2*np.pi/period*np.sqrt(1-ecc**2)/(1-ecc*np.cos(anomaly))**2
    dr = a*ecc*np.sin(anomaly)*2*np.pi/period/(1-ecc*np.cos(anomaly))
    v = np.column_stack((-dr*np.cos(argument)+rho*np.sin(argument)*rate,
                         (dr*np.sin(argument)+rho*np.cos(argument)*rate)*cosi))
    return p, v, np.sin(argument)>0


def two_half_flux(time, orbit, leading, trailing, u):
    p, v, foreground = sky_orbit(time, *orbit)
    direction = v/np.linalg.norm(v,axis=1)[:,None]
    parallel = np.sum(p*direction,axis=1)
    perpendicular = p[:,0]*(-direction[:,1])+p[:,1]*direction[:,0]
    u1,u2 = u
    norm = np.pi*(1-u1/3-u2/6)
    def primitive(y,h):
        mu = np.sqrt(max(0.,h*h-y*y))
        return ((1-u1-u2)*y+(u1+2*u2)/2*(y*mu+h*h*np.arcsin(np.clip(y/h,-1,1)))
                -u2*(h*h*y-y**3/3))
    def area(px,py,r,sign):
        lo,hi = (0.,r) if sign>0 else (-r,0.)
        lo,hi = max(lo,-1-px), min(hi,1-px)
        if lo>=hi:
            return 0.
        def height(x):
            yp = np.sqrt(max(0.,r*r-x*x))
            h = np.sqrt(max(0.,1-(px+x)**2))
            ylo,yhi = max(py-yp,-h),min(py+yp,h)
            if ylo>=yhi or h==0:
                return 0.
            return primitive(yhi,h)-primitive(ylo,h)
        # Break at the circle intersections, where clipping changes branch.
        z=np.hypot(px,py); points=[]
        if abs(1-r)<z<1+r:
            d=(1-r*r+z*z)/(2*z)
            h=np.sqrt(max(0.,1-d*d))
            for sign_y in (-1,1):
                x=(d-z)*px/z+sign_y*h*(-py/z)
                if lo<x<hi:points.append(x)
        return quad(height,lo,hi,points=points,epsabs=2e-12,epsrel=2e-10,limit=120)[0]
    out=np.ones(len(p))
    for i in np.flatnonzero(foreground & (np.linalg.norm(p,axis=1)<1+max(leading,trailing))):
        out[i]-=(area(parallel[i],perpendicular[i],leading,1)
                 +area(parallel[i],perpendicular[i],trailing,-1))/norm
    return out

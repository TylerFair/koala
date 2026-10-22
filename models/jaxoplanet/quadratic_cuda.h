// Scalar closed-form quadratic flux and Gaussian score.
// See licenses/exoplanet-core.txt for the upstream MIT license.
// CEL and mu moment adapted from exoplanet-core commit
// 156cca94f455541f3d37d04af97479798022937b, Copyright 2018-2021 Dan Foreman-Mackey.
#pragma once
#include <cmath>
#include <cstdint>
#include <cfloat>
#ifdef __CUDACC__
#define KOALA_INLINE __host__ __device__ inline
#else
#define KOALA_INLINE inline
#endif
namespace koala_quad {
constexpr double pi = 3.141592653589793238462643383279502884;
struct Parts { double value, dr, dz, du1, du2; };
KOALA_INLINE double maximum(double a, double b) { return a>b?a:b; }
KOALA_INLINE double minimum(double a, double b) { return a<b?a:b; }
KOALA_INLINE void cel3(double kc, double p, double a, double b,
                    double& third, double& E, double& C) {
  p=sqrt(p); b/=p;
  double ee=kc,m=1.,p1=1.,a2=1.,a3=1.,b2=kc*kc,b3=0.;
  for (int i=0;i<8;++i) {
    if(i) { kc=2*sqrt(ee); ee=kc*m; }
    double g=ee/p,g1=ee/p1;
    double an=a+b/p,bn=2*(b+a*g);
    double a2n=a2+b2/p1,b2n=2*(b2+a2*g1);
    double a3n=a3+b3/p1,b3n=2*(b3+a3*g1);
    a=an;b=bn;a2=a2n;b2=b2n;a3=a3n;b3=b3n;
    p+=g;p1+=g1;m+=kc;
  }
  double common=.5*pi/(m*(m+p1));
  third=.5*pi*(a*m+b)/(m*(m+p)); E=common*(a2*m+b2); C=common*(a3*m+b3);
}
KOALA_INLINE void mu_moment(double b,double r,double& moment,double& dr,double& db) {
  if(b==0.) {
    moment=-2*pi*expm1(1.5*log1p(-r*r))/3;
    dr=2*pi*r*sqrt(1-r*r);db=0.;return;
  }
  if(b+r==1.) {
    double root=sqrt(r*(1-r));
    moment=(2*acos(1-2*r)-4*(3+2*r-8*r*r)*root/3)/3;
    dr=8*r*root;db=-dr/3;return;
  }
  double bm=b-r,bp=b+r,diff=(1+bm)*(1-bm),outside=(1+bp)*(1-bp),fourbr=4*b*r;
  bool contained=outside>0.,coincident=b==r;
  double kc2=contained?outside/diff:-outside/fourbr;
  double kc=sqrt(maximum(kc2,DBL_MIN)),ratio=bm/bp,mu=3*ratio/diff;
  double p=contained?ratio*ratio*outside/diff:bm*bm*kc2;
  double aa=contained?1+mu:0.,zz=contained?p+mu:3*kc2*bm*bp;
  if(coincident) {p=1.;aa=1.;zz=kc2;}
  double third,E,C;cel3(kc,p,aa,zz,third,E,C);
  if(coincident) {
    double m=4*r*r;
    double le=r<.5?pi+2*((2*m-3)*E-m*C)/3:pi+(-m*E+(2*m-3)*C)/(3*r);
    moment=le/3;dr=r<.5?4*r*E:2*C;db=r<.5?4*r*(E-2*C)/3:2*(C-2*E)/3;
    return;
  }
  double root=sqrt(diff),invbr=1/sqrt(b*r);
  double lam=contained?2*root*(outside*third-(4-7*r*r-b*b)*E)/3:
      diff*(third+(-3+6*r*r+2*b*r)*C-fourbr*E)*invbr/3;
  moment=(2*pi*(r>b)+lam)/3;
  dr=contained?4*r*root*E:2*r*diff*C*invbr;
  db=contained?4*r*root*(E-2*C)/3:2*r*diff*(C-2*E)*invbr/3;
}
KOALA_INLINE double segment(double angle) {
  if(angle>=.1)return angle-sin(angle)*cos(angle);
  double s=angle*angle,p=-4096./6227020800.;
  p=1024./39916800.+s*p;p=-256./362880.+s*p;p=64./5040.+s*p;
  p=-16./120.+s*p;p=4./6.+s*p;
  return angle*angle*angle*p;
}
KOALA_INLINE Parts partials(double u1,double u2,double raw_r,double raw_z) {
  double r=fabs(raw_r),z=fabs(raw_z);
  if(r>=1+z)return {-1.,0.,0.,0.,0.};
  if(z>=1+r || r==0.)return {0.,0.,0.,0.,0.};
  double kite=0.,area=pi*r*r,k0=pi;
  if(z>fabs(1-r) && z<1+r) {
    double a=minimum(r,z),b=maximum(r,z),c=1.;
    double nb=minimum(b,c);c=maximum(b,c);b=nb;
    double na=minimum(a,b);b=maximum(a,b);a=na;
    kite=sqrt(maximum(0.,(a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))));
    k0=atan2(kite,z*z+(r-1)*(r+1));
    double product=(1+r-z)*(z+r-1)*(z+1-r)*(z+1+r);
    double chord=sqrt(maximum(product,0.));
    double first=atan2(chord,z*z+(1-r)*(1+r));
    double second=atan2(chord,z*z+(r-1)*(r+1));
    area=segment(first)+r*r*segment(second);
  }
  double s2=2*r*r*(r*r+2*z*z-1)*k0+.5*kite*(1-5*r*r-z*z);
  double m1,m1r,m1z;mu_moment(z,r,m1,m1r,m1z);
  double norm=pi*(1-u1/3-u2/6);
  double occulted=(1-u1-1.5*u2)*area+(u1+2*u2)*m1+.25*u2*s2;
  double value=-occulted/norm;
  double ar=2*r*k0,az=z>0.?-kite/z:0.;
  double sine=sin(k0),cosine=cos(k0);
  double m2r=2*r*((1-r*r-z*z)*k0+2*z*r*sine);
  double m2z=-2*r*((1-r*r-z*z)*sine+z*r*(k0+sine*cosine));
  double dr=-((1-u1-u2)*ar+(u1+2*u2)*m1r-u2*m2r)/norm;
  double dz=-((1-u1-u2)*az+(u1+2*u2)*m1z-u2*m2z)/norm;
  double du1=(area-m1+value*pi/3)/norm;
  double du2=(1.5*area-2*m1-.25*s2+value*pi/6)/norm;
  return {value,raw_r<0.?-dr:dr,raw_z<0.?-dz:dz,du1,du2};
}
// K trend columns; out holds log probability, four transit, K trend and one jitter term.
template<int K>
KOALA_INLINE void cadence(const double* theta,double z,double zdot,double y,double var,
                       bool valid,const double* design,bool front,double* out) {
  for(int k=0;k<K+6;++k)out[k]=0.;
  if(!valid)return;
  Parts p=front?partials(theta[1],theta[2],theta[0],z):Parts{0.,0.,0.,0.,0.};
  double flux=1+p.value,trend=0.;
  for(int k=0;k<K;++k)trend+=design[k]*theta[4+k];
  double residual=y-flux*trend,jitter=theta[4+K],totalvar=var+jitter*jitter;
  double sigma=sqrt(totalvar),standard=residual/sigma;
  // Match numpyro Normal.log_prob, retaining the original arithmetic.
  out[0]=-.5*standard*standard-log(sqrt(2*pi)*sigma);
  double score=residual/totalvar,transit=score*trend;
  out[1]=transit*p.dr;out[2]=transit*p.du1;out[3]=transit*p.du2;
  out[4]=transit*p.dz*zdot;
  for(int k=0;k<K;++k)out[5+k]=score*flux*design[k];
  out[5+K]=jitter*(residual*residual/totalvar-1)/totalvar;
}
} // namespace koala_quad

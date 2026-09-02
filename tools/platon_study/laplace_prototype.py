#!/usr/bin/env python
"""Float64 exact-JAX PLATON likelihood benchmark and Laplace-NUTS prototype.

This never writes in the PLATON checkout. The saved RetrievalResult supplies
the data, priors, defaults, and a representative set of starting points.
"""
import argparse, json, pickle, time
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("result"); ap.add_argument("--output",required=True)
    ap.add_argument("--startag",required=True); ap.add_argument("--include-opacities",nargs="*")
    ap.add_argument("--warmup",type=int,default=200); ap.add_argument("--samples",type=int,default=300); ap.add_argument("--starts",type=int,default=3)
    a=ap.parse_args()
    import jax; jax.config.update("jax_enable_x64",True); import jax.numpy as jnp
    import numpyro; from numpyro.infer import MCMC,NUTS
    from platon.experimental import _jax_forward_model as fm
    from platon.experimental._jax_forward_model import prepare_jax_data, make_scalar_lnlike
    from platon.experimental.combined_retriever import TransitDepthCalculator
    fm._FM_DTYPE=jnp.float64
    with open(a.result,"rb") as f:r=pickle.load(f)
    fi=r.fit_info; names=list(fi.fit_param_names); n=len(names)
    tc=TransitDepthCalculator(include_condensation=True,method="xsec",startag=a.startag,
        **({"include_opacities":a.include_opacities} if a.include_opacities else {}))
    tc.change_wavelength_bins(r.transit_bins)
    defaults_arr=np.array([fi.all_params[k].best_guess for k in names]); defaults=fi._interpret_param_array(defaults_arr)
    scalar_keys=(int,float,np.floating,np.integer,bool)
    all_defaults={k:(float(v) if isinstance(v,scalar_keys) else v) for k,v in defaults.items() if v is not None}
    tprep=time.perf_counter(); data=prepare_jax_data(tc.atm,tc.atm.abundance_getter,r.transit_bins,
        defaults.get("T_star"),defaults.get("T_spot"),defaults.get("spot_cov_frac"),False,fi,len(r.transit_depths))
    ll=make_scalar_lnlike(names,all_defaults,data,jnp.asarray(r.transit_depths,jnp.float64),jnp.asarray(r.transit_errors,jnp.float64))
    kinds=[]; lo=[]; hi=[]; loc=[]; scale=[]
    for name in names:
        p=fi.all_params[name]
        if hasattr(p,"low_lim"): kinds.append(0); lo.append(p.low_lim); hi.append(p.high_lim); loc.append(0); scale.append(1)
        else: kinds.append(1); lo.append(0); hi.append(0); loc.append(p.best_guess); scale.append(p.std)
    kinds=jnp.array(kinds); lo=jnp.array(lo); hi=jnp.array(hi); loc=jnp.array(loc); scale=jnp.array(scale)
    def transform(z):
        s=jax.nn.sigmoid(z); return jnp.where(kinds==0,lo+(hi-lo)*s,loc+scale*z)
    def potential(z):
        # logistic density is the Uniform-prior Jacobian; Gaussian coords are standardized.
        lp=jnp.sum(jnp.where(kinds==0,jax.nn.log_sigmoid(z)+jax.nn.log_sigmoid(-z),-0.5*z*z))
        return -(ll(transform(z))+lp)
    vg=jax.jit(jax.value_and_grad(potential)); post=np.asarray(r.equal_samples)
    def to_z(x):
        u=np.clip((x-np.asarray(lo))/(np.asarray(hi)-np.asarray(lo)),1e-8,1-1e-8)
        return np.where(np.asarray(kinds)==0,np.log(u)-np.log1p(-u),(x-np.asarray(loc))/np.asarray(scale))
    z0=to_z(post[np.argmax(np.asarray(r.logp))]); v,g=vg(jnp.asarray(z0)); jax.block_until_ready(g); compile_s=time.perf_counter()-tprep
    reps=20; t=time.perf_counter()
    for _ in range(reps): v=potential(jnp.asarray(z0)); jax.block_until_ready(v)
    like_ms=1000*(time.perf_counter()-t)/reps
    t=time.perf_counter()
    for _ in range(reps): v,g=vg(jnp.asarray(z0)); jax.block_until_ready(g)
    grad_ms=1000*(time.perf_counter()-t)/reps
    from scipy.optimize import minimize
    starts=np.linspace(0,len(post)-1,a.starts,dtype=int); maps=[]
    for ix in starts:
        zz=to_z(post[ix]); calls=0; t=time.perf_counter()
        def fg(q):
            nonlocal calls; calls+=1; v,g=vg(jnp.asarray(q)); return float(v),np.asarray(g,float)
        opt=minimize(fg,zz,jac=True,method="L-BFGS-B",options={"maxiter":300,"ftol":1e-10,"gtol":1e-5})
        maps.append(dict(z=np.asarray(opt.x),potential=float(opt.fun),success=bool(opt.success),iterations=int(opt.nit),gradient_evals=calls,seconds=time.perf_counter()-t,message=str(opt.message)))
    best=min(maps,key=lambda q:q["potential"]); zmap=jnp.asarray(best["z"])
    th=jax.jit(jax.hessian(potential)); t=time.perf_counter(); H=np.asarray(th(zmap)); jax.block_until_ready(jnp.asarray(H)); hess_s=time.perf_counter()-t
    H=(H+H.T)/2; ev,Q=np.linalg.eigh(H); floor=max(ev.max()*1e-8,1e-8); repaired=np.maximum(ev,floor); precision=(Q*repaired)@Q.T
    kernel=NUTS(potential_fn=potential,inverse_mass_matrix=jnp.asarray(precision),adapt_mass_matrix=False,target_accept_prob=.9,max_tree_depth=8)
    m=MCMC(kernel,num_warmup=a.warmup,num_samples=a.samples,progress_bar=False); t=time.perf_counter(); m.run(jax.random.PRNGKey(7),init_params=zmap,extra_fields=("num_steps","diverging")); sample_s=time.perf_counter()-t
    zs=np.asarray(m.get_samples()); xs=np.asarray(jax.vmap(transform)(jnp.asarray(zs))); extra=m.get_extra_fields(); ess=np.asarray(numpyro.diagnostics.effective_sample_size(zs[None,...]))
    ref_med=np.median(post,0); ref_sd=np.std(post,0,ddof=1); med=np.median(xs,0); sd=np.std(xs,0,ddof=1)
    result=dict(result=a.result,n_parameters=n,n_data=len(r.transit_depths),dtype=str(xs.dtype),compile_and_prepare_seconds=compile_s,
      likelihood_ms=like_ms,value_and_gradient_ms=grad_ms,maps=[{k:v for k,v in q.items() if k!="z"} for q in maps],map_pairwise_distances=[],
      hessian_seconds=hess_s,hessian_raw_eigen_min=float(ev.min()),hessian_raw_eigen_max=float(ev.max()),hessian_repaired_condition=float(repaired.max()/repaired.min()),
      warmup=a.warmup,samples=a.samples,sampling_seconds=sample_s,divergences=int(np.sum(extra["diverging"])),mean_num_steps=float(np.mean(extra["num_steps"])),
      min_ess=float(np.min(ess)),median_ess=float(np.median(ess)),gradient_evals_per_min_ess=float((a.warmup+a.samples)*np.mean(extra["num_steps"])/max(np.min(ess),1e-12)),
      comparison=[dict(name=names[i],median_shift_sigma=float((med[i]-ref_med[i])/ref_sd[i]),sigma_ratio=float(sd[i]/ref_sd[i])) for i in range(n)])
    for i in range(len(maps)):
      for j in range(i): result["map_pairwise_distances"].append(float(np.linalg.norm(maps[i]["z"]-maps[j]["z"])))
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()

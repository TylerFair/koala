#!/usr/bin/env python
"""Read-only diagnostics for saved PLATON RetrievalResult posteriors."""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("results", nargs="+"); ap.add_argument("--output", required=True); a=ap.parse_args()
    out={}
    for fn in a.results:
        with open(fn,"rb") as f: r=pickle.load(f)
        x=np.asarray(r.equal_samples,float); names=list(r.fit_info.fit_param_names)
        # Work in standardized physical coordinates so scale alone cannot dominate.
        sd=x.std(0,ddof=1); z=(x-x.mean(0))/np.where(sd>0,sd,1)
        cov=np.cov(z,rowvar=False); eig=np.linalg.eigvalsh(cov)
        per=[]
        for j,n in enumerate(names):
            zz=z[:,j:j+1]
            g1=GaussianMixture(1,random_state=0).fit(zz); g2=GaussianMixture(2,random_state=0).fit(zz)
            means=sorted(g2.means_[:,0]); weights=g2.weights_[np.argsort(g2.means_[:,0])]
            sep=(means[1]-means[0])/np.sqrt(np.mean(g2.covariances_.reshape(2)))
            per.append(dict(name=n,median=float(np.median(x[:,j])),sigma=float(sd[j]),skew=float(stats.skew(x[:,j])),
                excess_kurtosis=float(stats.kurtosis(x[:,j])),bic_improvement_2_minus_1=float(g1.bic(zz)-g2.bic(zz)),
                mixture_separation_sigma=float(sep),mixture_weights=[float(v) for v in weights]))
        ij=np.unravel_index(np.argmax(np.abs(cov-np.eye(len(names)))),cov.shape)
        out[Path(fn).parent.name]=dict(file=str(fn),n_samples=len(x),n_parameters=len(names),names=names,
            covariance_condition=float(eig[-1]/max(eig[0],np.finfo(float).tiny)),max_abs_correlation=float(abs(cov[ij])),
            max_correlation_pair=[names[ij[0]],names[ij[1]]],forward_model_eval_count=getattr(r,"forward_model_eval_count",None),
            forward_model_eval_seconds=getattr(r,"forward_model_eval_seconds",None),parameters=per)
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__": main()

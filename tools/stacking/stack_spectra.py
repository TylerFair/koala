#!/usr/bin/env python3
"""Score completed stacking variants and write spectra and diagnostics."""
from __future__ import annotations
import argparse, io, json, pickle, re, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
import numpy as np
import yaml
from models.stacking import (pointwise_loglik, psis_loo, pseudo_bma_plus_weights,
    aligned_stack_posteriors, stacking_weights, write_stacked_spectrum)
from tools.spectro_stage_inputs import load_stage_inputs

def _load_chunks(output_dir):
    families = {}
    for path in (output_dir / "chunks").glob("*chunk_*_*.pkl"):
        match = re.search(r"_chunk_(\d+)_(\d+)\.pkl$", path.name)
        if match:
            prefix = re.sub(r"_chunk_\d+_\d+\.pkl$", "", path.name)
            families.setdefault(prefix, []).append((int(match[1]), int(match[2]), path))
    candidates = [sorted(x) for x in families.values() if min(y[0] for y in x) == 0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint family under {output_dir}")
    chunks = max(candidates, key=lambda x: max(y[1] for y in x))
    if any(chunks[i][1] != chunks[i+1][0] for i in range(len(chunks)-1)):
        raise ValueError(f"Non-contiguous chunks under {output_dir}")
    payloads = []
    for _, _, path in chunks:
        with path.open("rb") as stream: payload = pickle.load(stream)
        # ESS-selective fallback checkpoints include routing metadata beside
        # the posterior mapping.
        payloads.append(payload.get("samples", payload))
    return {key: np.concatenate([np.asarray(x[key]) for x in payloads], axis=1)
            for key in payloads[0]}

def _stage(result_dir, stage_kind):
    paths = sorted((result_dir / "stage_inputs").glob(f"*{stage_kind}_inputs.pkl"))
    if len(paths) != 1: raise FileNotFoundError(f"Expected one stage dump; found {len(paths)}")
    return load_stage_inputs(paths[0])

def _write_loglik_batched(path, samples, stage, batch_size=25):
    """Evaluate draws in bounded-memory batches and stream them into one NPZ."""
    n_draw = next(iter(samples.values())).shape[0]
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True) as archive:
        for start in range(0, n_draw, batch_size):
            stop = min(start + batch_size, n_draw)
            batch = {key: value[start:stop] for key, value in samples.items()}
            values = pointwise_loglik(batch, stage).astype(np.float32)
            payload = io.BytesIO()
            np.lib.format.write_array(payload, values, allow_pickle=False)
            archive.writestr(f"loglik_{start:06d}_{stop:06d}.npy", payload.getvalue())

def _score_loglik_cache(path):
    """Run PSIS a channel at a time without materializing the full cache."""
    with np.load(path) as cached:
        keys = sorted(cached.files)
        if keys == ["loglik"]:
            values = np.asarray(cached["loglik"], dtype=np.float32)
        else:
            shape = (sum(cached[key].shape[0] for key in keys),) + cached[keys[0]].shape[1:]
            values = np.empty(shape, dtype=np.float32)
            start = 0
            for key in keys:
                batch = cached[key]
                values[start:start + batch.shape[0]] = batch
                start += batch.shape[0]
    elpd = np.empty(values.shape[1:], dtype=np.float64)
    khat = np.empty(values.shape[1:], dtype=np.float64)
    for channel in range(values.shape[1]):
            elpd_channel, _, khat_channel = psis_loo(
                np.asarray(values[:, channel:channel + 1, :], dtype=np.float64))
            elpd[channel] = elpd_channel[0]
            khat[channel] = khat_channel[0]
    return elpd, khat

def _plot(path, wave, names, depths, aligned_summary, absolute_summary,
          weights, pbma, khat):
    fig, axes = plt.subplots(4, 1, figsize=(12,14), sharex=True,
        gridspec_kw={"height_ratios":[2.2,1.4,1.6,1.2]})
    for name, values in zip(names, depths):
        q16, med, q84 = np.percentile(values, [16,50,84], axis=0)
        axes[0].errorbar(wave, med*1e6, [(med-q16)*1e6,(q84-med)*1e6], fmt=".", alpha=.5, label=name)
    axes[0].errorbar(wave, aligned_summary["median"]*1e6,
        [aligned_summary["depth_err_lo"]*1e6,aligned_summary["depth_err_hi"]*1e6], fmt="ko", ms=3, label="aligned stacked")
    axes[0].set_ylabel("Transit depth (ppm)"); axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(wave, absolute_summary["median"]*1e6, color="tab:red", label="absolute stacked")
    axes[1].fill_between(wave,(absolute_summary["median"]-absolute_summary["depth_err_lo"])*1e6,
        (absolute_summary["median"]+absolute_summary["depth_err_hi"])*1e6,color="tab:red",alpha=.2)
    axes[1].plot(wave, aligned_summary["median"]*1e6, color="k", label="offset-aligned stacked")
    axes[1].fill_between(wave,(aligned_summary["median"]-aligned_summary["depth_err_lo"])*1e6,
        (aligned_summary["median"]+aligned_summary["depth_err_hi"])*1e6,color="k",alpha=.2)
    axes[1].set_ylabel("Stacked depth (ppm)"); axes[1].legend()
    colors=plt.cm.tab10(np.linspace(0,1,len(names)))
    for color,name,value,value_pbma in zip(colors,names,weights,pbma):
        axes[2].plot(wave,value,color=color,label=f"{name} stacking")
        axes[2].plot(wave,value_pbma,color=color,ls="--",alpha=.75,label=f"{name} pseudo-BMA+")
    axes[2].set_ylabel("Model weight"); axes[2].set_ylim(0,1); axes[2].legend(fontsize=7,ncol=2)
    axes[3].plot(wave,np.max(khat,axis=(0,2)),"o-",label="max k-hat")
    axes[3].axhline(.7,color="r",ls="--",lw=1)
    axes[3].plot(wave,aligned_summary["disagreement"],"s-",label="aligned disagreement")
    axes[3].plot(wave,absolute_summary["disagreement"],"^-",alpha=.6,label="absolute disagreement")
    axes[3].set_ylabel("Diagnostic"); axes[3].set_xlabel("Wavelength (micron)"); axes[3].legend()
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("matrix_spec")
    parser.add_argument("--output",default="acceleration_reports/stacking")
    parser.add_argument("--queue-start",type=int,default=300); parser.add_argument("--n-out",type=int,default=20000)
    parser.add_argument("--stage",choices=("low_resolution","high_resolution"),default="high_resolution")
    parser.add_argument("--label",default="wasp39_nrs1_reference")
    args=parser.parse_args()
    with Path(args.matrix_spec).open() as stream: spec=yaml.safe_load(stream)
    names=[x["name"] for x in spec["variants"]]; dataset=spec["dataset"]
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    depths=[]; elpds=[]; khats=[]; wave=None
    for index,name in enumerate(names):
        fit=Path("/scratch/midway3/tfairnington/accel_stacking")/dataset/name
        queue_number=int(spec["variants"][index].get("queue_number", args.queue_start+index))
        result=Path("/scratch/midway3/tfairnington/accel_gpu_results")/f"{queue_number}_stacking_{name}"
        samples=_load_chunks(fit); stage=_stage(result,args.stage); llpath=output/f"{args.label}_{name}_pointwise_loglik.npz"
        if not llpath.exists():
            _write_loglik_batched(llpath, samples, stage)
        elpd_i,khat=_score_loglik_cache(llpath); elpds.append(elpd_i); khats.append(khat)
        depths.append(np.asarray(samples["rors"])[...,0]**2)
        current=np.asarray(stage.meta["wavelength"])
        if wave is None: wave=current
        elif not np.allclose(wave,current): raise ValueError("Variant wavelength grids differ")
    elpds=np.asarray(elpds); khats=np.asarray(khats)
    weights=np.column_stack([stacking_weights(elpds[:,ch,:]) for ch in range(elpds.shape[1])])
    pbma=np.column_stack([pseudo_bma_plus_weights(elpds[:,ch,:],rng=800+ch) for ch in range(elpds.shape[1])])
    stacked=aligned_stack_posteriors(depths,weights,args.n_out,rng=20260902)
    write_stacked_spectrum(output/f"{args.label}_stacked.csv",wave,stacked["aligned_summary"],names,depths,weights,
        absolute_summary=stacked["absolute_summary"],offsets=stacked["offsets"],
        offset_uncertainties=stacked["offset_uncertainties"])
    _plot(output/f"{args.label}_stacking.png",wave,names,depths,stacked["aligned_summary"],
          stacked["absolute_summary"],weights,pbma,khats)
    np.savez_compressed(output/f"{args.label}_stacking_arrays.npz",wavelength=wave,
        stacking_weights=weights,pseudo_bma_plus_weights=pbma,khat=khats,
        aligned_mixture=stacked["aligned_mixture"].astype(np.float32),
        absolute_mixture=stacked["absolute_mixture"].astype(np.float32),offsets=stacked["offsets"],
        offset_uncertainties=stacked["offset_uncertainties"],average_offset=stacked["average_offset"])
    diag={"models":names,"stacking_weights":weights.tolist(),"pseudo_bma_plus_weights":pbma.tolist(),
        "khat_max_by_model_channel":np.max(khats,axis=2).tolist(),"achromatic_offsets":stacked["offsets"].tolist(),
        "achromatic_offset_uncertainties":stacked["offset_uncertainties"].tolist(),
        "weighted_average_offset":stacked["average_offset"],
        "laplace_bma":"unavailable unless each run emits white-light MAP/Hessian/log-joint"}
    (output/f"{args.label}_diagnostics.json").write_text(json.dumps(diag,indent=2))
if __name__ == "__main__": main()

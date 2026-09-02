#!/usr/bin/env python3
"""Run full-pipeline A/B/C processes with isolated outputs and compile logs."""
import argparse, json, os, subprocess, time
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',required=True); p.add_argument('--tag',required=True)
    p.add_argument('--mode',choices=['soss','g395h'],required=True)
    a=p.parse_args(); base=yaml.safe_load(Path(a.base).read_text())
    root=Path('/scratch/midway3/tfairnington')
    cache=root/'jax_cache'/f"speed_p2_{a.tag}_cold"
    results=[]
    for label,box in [('A',False),('B',True),('C',True)]:
        cfg=yaml.safe_load(yaml.safe_dump(base)); flags=cfg.setdefault('flags',{})
        flags.update(random_seed=555, whitelight_mass_matrix='laplace',
          whitelight_laplace_warmup=200, whitelight_laplace_target_accept=0.9,
          whitelight_laplace_max_tree_depth=10,
          whitelight_laplace_hessian_method='finite_difference',
          whitelight_min_ess=400, whitelight_max_divergences=0,
          whitelight_max_extra_blocks=3, spectro_sampler='independent_nuts',
          spectro_mass_matrix='laplace', spectro_jitter_prior='lognormal',
          spectro_min_depth_ess=400, compile_box=box,
          jax_compilation_cache_dir=str(cache), save_whitelight_trace=True,
          whitelight_num_samples=1000, need_lowres=True, vmap_chunk=40)
        cfg['host_device']='gpu'; cfg['output_dir']=f"{a.tag}_SPEED_{label}"
        config=Path('configs_speed')/f"{a.tag}_SPEED_{label}.yaml"
        config.parent.mkdir(exist_ok=True); config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        log=Path('acceleration_reports/diag_compile_box')/f"{a.tag}_{label}.log"
        log.parent.mkdir(exist_ok=True)
        env=dict(os.environ); env['JAX_LOG_COMPILES']='1'; env['JAX_ENABLE_X64']='1'
        started=time.perf_counter()
        with log.open('w') as stream:
            proc=subprocess.run(['python','fit_jwst.py','--config',str(config)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        wall=time.perf_counter()-started; text=log.read_text(errors='ignore')
        out=root/cfg['output_dir']
        diag=out/'whitelight_mcmc_diagnostics.json'
        wl=json.loads(diag.read_text()) if diag.exists() else {}
        results.append(dict(label=label,compile_box=box,returncode=proc.returncode,
          process_wall_seconds=wall,compile_log_events=text.count('Compiling '),
          whitelight_sampling_seconds=wl.get('sampling_wall_seconds'),
          whitelight_preparation_seconds=wl.get('laplace_preparation_wall_seconds'),
          output_dir=str(out),log=str(log)))
        if proc.returncode: break
    path=Path('acceleration_reports/diag_compile_box')/f"{a.tag}_matrix.json"
    path.write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))

if __name__=='__main__': main()

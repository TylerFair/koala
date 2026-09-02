#!/usr/bin/env python3
import os,subprocess,time,yaml,json
from pathlib import Path
base=yaml.safe_load(Path('configs_accel/HAT-P-65_nrs1_prism_jaxoplanet_accel_dump.yaml').read_text())
f=base.setdefault('flags',{}); f.update(random_seed=555,whitelight_mass_matrix='laplace',whitelight_laplace_warmup=200,whitelight_laplace_target_accept=0.99,whitelight_min_ess=400,whitelight_max_divergences=0,whitelight_max_extra_blocks=3,spectro_sampler='independent_nuts',spectro_mass_matrix='laplace',spectro_jitter_prior='lognormal',spectro_min_depth_ess=400,vmap_chunk=21,chunk_mode='serial',compile_box=True,jax_compilation_cache_dir='/scratch/midway3/tfairnington/jax_cache/speed_p2_prism_serial')
base['host_device']='gpu'; base['output_dir']='HAT-P-65_PRISM_P2_SPEED_SERIAL'
p=Path('configs_speed/HAT-P-65_PRISM_P2_SPEED_SERIAL.yaml'); p.write_text(yaml.safe_dump(base,sort_keys=False))
t=time.perf_counter(); rc=subprocess.run(['python','fit_jwst.py','--config',str(p)]).returncode
Path('acceleration_reports/diag_compile_box/prism_serial_wall.json').write_text(json.dumps({'returncode':rc,'wall_seconds':time.perf_counter()-t},indent=2))
raise SystemExit(rc)

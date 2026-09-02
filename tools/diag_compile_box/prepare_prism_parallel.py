#!/usr/bin/env python3
import shutil,yaml
from pathlib import Path
src=Path('/scratch/midway3/tfairnington/HAT-P-65_PRISM_P2_SPEED_SERIAL')
dst=Path('/scratch/midway3/tfairnington/HAT-P-65_PRISM_P2_SPEED_PARALLEL')
if dst.exists(): raise SystemExit(f'refusing existing destination {dst}')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('chunks','*R50*','3[0-9]_*'))
base=yaml.safe_load(Path('configs_speed/HAT-P-65_PRISM_P2_SPEED_SERIAL.yaml').read_text())
base['output_dir']='HAT-P-65_PRISM_P2_SPEED_PARALLEL'; f=base['flags']; f['analysis_stage']='highres'; f['chunk_mode']='parallel'; f['chunk_parallel_job_count']=2
for i in (0,1):
 c=yaml.safe_load(yaml.safe_dump(base)); c['flags']['chunk_parallel_job_index']=i
 Path(f'configs_speed/HAT-P-65_PRISM_P2_SPEED_PARALLEL_{i}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))

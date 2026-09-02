#!/usr/bin/env python3
"""Create, but never replace, prioritized immutable parity queue scripts."""
from pathlib import Path
import yaml

PROJECT=Path('/project/ekempton/tfairnington/JWST')
CFG=PROJECT/'configs_fiducial_stellarinformed'
STAGED=Path('/scratch/midway3/tfairnington/accel_parity_staged')
OUT=PROJECT/'acceleration_reports/gpu_queue/templates'

def priority(path, cfg):
    instrument=str(cfg.get('instrument','')).upper(); name=path.name
    if 'soss_order1' in name: return (1,name)
    if 'G395H' in instrument and str(cfg.get('nrs','')).lower()=='nrs1': return (2,name)
    if 'G395H' in instrument: return (3,name)
    return (4,name)

items=[]
for path in CFG.glob('*_config.yaml'):
    cfg=yaml.safe_load(path.read_text()); run=Path('/cds2/ekempton/tfairnington/STELLARINFORMED')/str(cfg.get('output_dir',''))
    fits=Path(cfg.get('path','.'))/str(cfg.get('input_dir',''))/str(cfg.get('fits_file',''))
    staged=STAGED/f'{path.stem}_reference.pkl'
    if run.is_dir() and fits.is_file() and staged.is_file(): items.append((path,cfg,staged))
items.sort(key=lambda x:priority(x[0],x[1]))
for index,(path,cfg,staged) in enumerate(items,210):
    if path.name=='HAT-P-12_soss_order1_config.yaml': continue
    target=OUT/f'{index:03d}_parity_{path.stem}.sh'
    text=f'''#!/bin/bash
set -u
cd {PROJECT}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python tools/campaign/run_parity_dataset.py \\
  --config {path} \\
  --staged-reference {staged} \\
  --run-tag 20260902a --candidates A,B,C
'''
    try:
        with target.open('x') as stream: stream.write(text)
    except FileExistsError:
        continue
    target.chmod(0o775); print(target)

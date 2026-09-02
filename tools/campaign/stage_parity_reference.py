#!/usr/bin/env python3
"""Stage one immutable saved high-resolution reference for GPU parity runs."""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, yaml
PROJECT=Path('/project/ekempton/tfairnington/JWST'); sys.path.insert(0,str(PROJECT))
from tools.campaign.run_dataset import concatenate_reference_chunks, _read_wavelength_csv

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); saved=Path('/cds2/ekempton/tfairnington/STELLARINFORMED')/cfg['output_dir']
candidates=[]
for f in saved.glob('*_wavelengths*.csv'):
    w=_read_wavelength_csv(f); candidates.append((len(w),f,w))
if not candidates: raise FileNotFoundError(f'No Rreference wavelength file in {saved}')
n,path,w=max(candidates,key=lambda x:x[0]); samples,manifest=concatenate_reference_chunks(saved,n)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('xb') as stream: pickle.dump({'samples':samples,'wavelengths':np.asarray(w),'wavelength_file':str(path),'manifest':manifest},stream,pickle.HIGHEST_PROTOCOL)
print(out)

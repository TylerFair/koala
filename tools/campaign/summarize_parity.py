#!/usr/bin/env python3
"""Summarize immutable parity result JSON files as Markdown."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--output',required=True); a=p.parse_args()
rows=[]
for path in Path(a.root).glob('**/result_final.json'):
 d=json.loads(path.read_text())
 for c in d.get('candidates',[]):
  s=c.get('spectrum',{}); e=c.get('depth_ess',{}); f=c.get('fidelity',{})
  rows.append((d['dataset'],c['label'],c['status'],c['fit']['wall_seconds'],e.get('median'),s.get('weighted_mean_offset_ppm'),s.get('slope_ppm_per_um'),s.get('rms_channel_median_difference_ppm'),s.get('sigma_ratio_median'),s.get('sigma_ratio_p05'),s.get('sigma_ratio_p95'),f.get('gate_pass_fraction')))
lines=['| Dataset | Candidate | Status | Wall min | depth ESS med | offset ppm | slope ppm/um | RMS ppm | sigma ratio med [5,95] | gate pass |','|---|---|---|---:|---:|---:|---:|---:|---|---:|']
fmt=lambda x:'—' if x is None else f'{x:.3g}'
for r in rows: lines.append(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]/60:.1f} | {fmt(r[4])} | {fmt(r[5])} | {fmt(r[6])} | {fmt(r[7])} | {fmt(r[8])} [{fmt(r[9])},{fmt(r[10])}] | {fmt(r[11])} |')
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:f.write('\n'.join(lines)+'\n')
print(out)

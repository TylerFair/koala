# GPU job queue (orchestrator-run dispatcher, V100 16GB allocation)

Workers cannot call Slurm.  To run something on the GPU, write an executable
bash script into `acceleration_reports/gpu_queue/pending/<name>.sh`.  The
dispatcher (a login-node loop owned by the orchestrator) runs scripts one at a
time, oldest first, on the allocated V100 node with:

    cd /project/ekempton/tfairnington/JWST
    module load cuda/12.9
    bash <name>.sh   (PATH already has the jaxoplanet conda env python first)

Do NOT set JAX_PLATFORMS in the script (or set it to gpu).  Everything the
script prints (stdout+stderr) goes to `gpu_queue/done/<name>.out`; when it
finishes, `gpu_queue/done/<name>.exit` holds the exit code and
`gpu_queue/done/<name>.sh` is the script that ran.  While it runs, it sits in
`gpu_queue/running/`.  Scripts are killed after 90 minutes.  Keep each script
self-contained and write any large outputs to
/scratch/midway3/tfairnington/accel_gpu_results/<name>/ .  Poll for the
`.exit` file; do not busy-wait faster than every 20 s.  One GPU: do not
queue more than ~3 scripts at once, and prefer short (<30 min) runs.

## Update (17:30): two GPUs serve this queue
A second dispatcher now also pulls from `pending/` and runs on a **Quadro RTX 6000**
(1/32-rate FP64, roughly 2.5-4x slower than the V100 for these float64 kernels).
Scripts run there get a `done/<name>.gpu` file naming the GPU; scripts without a
`.gpu` file ran on the Tesla V100.  Never compare wall times across the two GPUs:
for a speedup claim, run baseline and candidate on the same GPU (check the `.gpu`
file, or put both runs in one script).  For accuracy-only or CPU-bound work either
GPU is fine.

## Update (19:15): a second V100 joined the queue
A third dispatcher runs on another Tesla V100 (job 57128921).  Its runs get a
`.gpu` file that says "Tesla V100-PCIE-16GB (... second V100)".  V100-to-V100
timings are comparable; RTX 6000 timings are not.  The original V100 (no .gpu
file) expires at about 21:00.

## Update (22:10): fresh single V100 (job 57150049)
All earlier allocations are gone.  One dispatcher now runs on a Tesla V100
(job 57150049); its `.gpu` files say "third V100".  Only one GPU serves the
queue, so runs are serialized.

## Update (20:55): second V100 (job 57157918, "fourth V100" marker) added; two GPUs serve the queue again.

## Update (2026-09-02 00:43): one Tesla V100 (job 57218349, "fifth V100" marker). All earlier allocations are gone; runs are serialized.
## Update (00:54): second V100 (job 57220760, "sixth V100" marker) added; two GPUs serve the queue.
## Rule tonight (01:29): NEVER delete, move, or overwrite existing files; new outputs to new paths (see acceleration_reports/OVERNIGHT_MANIFEST.md).
## Update (01:41): fourth allocation (any GPU type; its .gpu marker says GPU9 — treat as RTX unless the .out shows a V100) (job 57227133, "seventh V100" marker) added; three GPUs serve the queue. Rule tonight: NEVER delete or overwrite existing files (acceleration_reports/OVERNIGHT_MANIFEST.md).
## Update (02:06): third V100 (job 57224760, "seventh V100" marker) added; three GPUs serve the queue. Rule tonight: NEVER delete or overwrite existing files (acceleration_reports/OVERNIGHT_MANIFEST.md).
## Update (07:32): fourth allocation (any GPU type; its .gpu marker says GPU9 — treat as RTX unless the .out shows a V100) (job 57336378, "seventh V100" marker) added; three GPUs serve the queue. Rule tonight: NEVER delete or overwrite existing files (acceleration_reports/OVERNIGHT_MANIFEST.md).
## Update (10:48): fourth allocation (any GPU type; its .gpu marker says GPU9 — treat as RTX unless the .out shows a V100) (job 57396794, "seventh V100" marker) added; three GPUs serve the queue. Rule tonight: NEVER delete or overwrite existing files (acceleration_reports/OVERNIGHT_MANIFEST.md).
## Update (11:07): fourth allocation (any GPU type; its .gpu marker says GPU9 — treat as RTX unless the .out shows a V100) (job 57401379, "seventh V100" marker) added; three GPUs serve the queue. Rule tonight: NEVER delete or overwrite existing files (acceleration_reports/OVERNIGHT_MANIFEST.md).
## Update (11:33): fourth allocation (any GPU type; its .gpu marker says GPU9 — treat as RTX unless the .out shows a V100) (job 57401380, "seventh V100" marker) added; three GPUs serve the queue. Rule tonight: NEVER delete or overwrite existing files (acceleration_reports/OVERNIGHT_MANIFEST.md).

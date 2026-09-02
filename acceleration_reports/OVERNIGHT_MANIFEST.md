# Overnight run manifest (2026-09-02, 01:40 -> ~09:40)

## Standing rules for every worker and for the orchestrator
1. **Never delete, move, truncate, or overwrite-in-place any existing file**,
   anywhere (repo, /scratch, /cds2).  New outputs go to NEW paths; edits to
   source files are additive patches; if a file must be replaced, write a
   new file with a new name and say so in the report.  `rm`, `mv`,
   `shutil.rmtree`, `os.remove`, truncating opens on existing paths, and
   `scancel` of jobs you did not create are forbidden.
2. Reference data: /cds2/ekempton/tfairnington/STELLARINFORMED/ (read-only;
   not visible on GPU nodes; stage copies to /scratch/midway3/tfairnington/
   under a new directory) and /scratch/midway3/tfairnington/ (copy, never
   delete).
3. Fidelity target: match the saved STELLARINFORMED transmission spectra
   (depths and errors) within Monte-Carlo scatter, unbiasedly, using the
   calibrated gates in acceleration_reports/reference.md plus the ppm-level
   mean-offset/slope test in ORCHESTRATOR_SUMMARY.md.  Only after parity is
   demonstrated do speed changes count.
4. All GPU work through acceleration_reports/gpu_queue (README at the
   bottom lists the live GPUs).  No Slurm commands from workers.

## Work packages
- P1 parity campaign v2 (median white-light handoff restored): candidates
  A = Laplace NUTS, B = Laplace HMC-8, C = Laplace-IS v3, vs saved spectra.
- P2 white-light preconditioning (in progress) -> padded compilation box
  (fixed lane width for low/high-res, cadence bucketing, persistent JAX
  compilation cache) with exact-potential equality tests.
- P3 Sing et al. 2026 (arXiv:2609.00263) limb-darkening offset prior,
  opt-in, side-by-side, Laplace-IS compatible; note for later testing.
- P4 Laplace-IS fallback padding (compile once) and parallel PRISM chunks.

## Progress log (append-only)
- 2026-09-02 01:26:26 manifest created; workers P1/P3 launching; P2 running
- 2026-09-02 01:30:01 launched parity worker (P1) and sing-prior worker (P3); white-light worker (P2) still running; third V100 requested (detached)
- 2026-09-02 01:52:36 status: WL worker validating (SOSS 63 steps/draw, G395H 7, PRISM running); parity jobs queued; sing second pass started; 2 V100 active, 2 allocations pending
- 2026-09-02 02:10:16 CDT: parity WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 02:15:53 CDT: parity HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT/result_final.json
- 2026-09-02 02:24:12 PARITY vs saved STELLARINFORMED (Laplace-IS, full pipeline, median handoff): HAT-P-12 SOSS +3.5 ppm / -1.2 ppm/um / rms 8.3 ppm / sigma-ratio 1.007 / 95.8% pass; WASP-52 G395H -1.2 ppm / -2.3 ppm/um / rms 8.7 ppm / sigma-ratio 0.99 / 97.5% pass. A/B retries queued (serializer bug fixed). WL Laplace: SOSS/G395H pass (<=0.08 sigma), PRISM fails (divergences).
- 2026-09-02 02:31:16 WL Laplace report in (SOSS 1.2x incl. prep, G395H 3.1x, PRISM TA0.99 divergence-free 48 s vs 1104 s, needs 3-seed validation); launched P2 speed package (WL ESS gate, compile box, fallback padding, parallel PRISM)
- 2026-09-02 02:34 P2 task 1: queued two additional PRISM white-light Laplace TA=0.99 seeds in new output paths; existing seed 559 was divergence-free and the third adaptive reference remains in progress.
- 2026-09-02 02:43 P2 task 2/4 implementation: added white-light retained-draw ESS/divergence continuation using the warmed state (6 CPU tests pass) and fixed Laplace-IS fallback compilation width to the parent lane width; automatic Laplace-to-adaptive fallback and spectroscopic selective extension remain under test.
- 2026-09-02 02:49 P2 task 3 partial: added opt-in `compile_box` persistent JAX cache configuration (`/scratch/midway3/tfairnington/jax_cache`, one-second threshold); cadence/lane padding equality work remains pending.
- 2026-09-02 02:56 P2 task 5 status: verified existing parallel routing test coverage; did not queue the two-GPU PRISM run while priority parity jobs and the worker's PRISM validation occupied the queue. Interim report written to acceleration_reports/speed_p2.md with incomplete items explicitly identified.
- 2026-09-02 03:06 P2 second pass task 2: implemented automatic white-light adaptive fallback and selective spectroscopic depth-ESS Laplace-NUTS fallback; syntax and 20-test white-light/independent-NUTS suite passed.
- 2026-09-02 03:07 P2 second pass task 3: exact masked cadence padding passes real SOSS and G395H potential/gradient equality at 1e-10 (max differences 1.82e-12 and 9.09e-13); GPU cold/warm timing remains queued behind priority work.
- 2026-09-02 03:14 P2 third pass task 1 complete: PRISM WL TA=0.99 passed 3-seed pooled gate vs 3 adaptive references; zero divergences each, max physical science shift 0.090 sigma (b proxy 1.81 ppm); one seed automatically extended to 4000 draws.
- 2026-09-02 03:15 P2 third pass tasks 3/4 closed as documented failures: no valid compile-box timing because replay CLI bypasses padding wrapper; no valid two-GPU equality launch because sliced replay renumbers RNG lanes and no new serial pipeline checkpoint tree existed. No GPU scripts from P2 remain pending.
- 2026-09-02 03:22 P2 fourth pass: PRISM WL 3x3 pooled gate harvested (zero candidate divergences; max physical shift 0.090 sigma). Queued full-pipeline SOSS and G395H A/B/C matrices with isolated outputs/cache and process compile logs (two P2 scripts pending).
- 2026-09-02 07:55 P2 sixth pass: harvested V100 SOSS/G395H matrices; compile-box warm whole walls 651.9/527.2 s but A-vs-box spectra were not roundoff identical (max depth differences 10.4/14.7 ppm), so identity gate fails. Queued manifest-safe full-pipeline PRISM serial run.
- 2026-09-02 08:10 P2 seventh pass: harvested PRISM serial V100 wall 3366.6 s (WL/low/high artifact intervals 238.3/718.6/2393.1 s); queued two pipeline-native global-index parallel halves and entered blocking wait.
- 2026-09-02 06:49 P2 eighth-pass audit: original parallel index0 computed only low chunk and index1 computed nothing; invalid for high-stage comparison. Queued corrected new-tree HIGH halves with need_lowres=false; both pending behind shared priority work. Final measured flag/wall summary appended without claiming unavailable production comparisons.
- 2026-09-02 02:42:00 PARITY: HAT-P-11 SOSS A: -0.0 ppm / -0.3 ppm/um / rms 1.3 ppm / sigma 1.002 / 100% pass / 912 s end-to-end; B 99.2%; HAT-P-12 SOSS A retry +0.6 ppm / 99.3%. Campaign feeder automated (2 pending, stops 09:00). Sing candidate-A demo ran (220); P2 building WL ESS gate + compile box.
- 2026-09-02 02:49:18 CDT: parity HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT completed; candidates=A,B; result=/scratch/midway3/tfairnington/accel_parity/20260902b/HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT/result_final.json
- 2026-09-02 02:53:40 CDT: parity HAT-P-11_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-11_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 03:04:27 CDT: parity WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B; result=/scratch/midway3/tfairnington/accel_parity/20260902b/WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 03:07:22 Sing second pass complete (auto gray-offset stage, tests, GPU demo A + Laplace-IS); P2 second pass: WL adaptive fallback + spectro depth-ESS fallback + exact compile-box masking (potential/grad equal to 1e-12) done; GPU timings pending; resuming P2
- 2026-09-02 03:07:54 SING prior demo (HAT-P-12 SOSS): free-LD calibration failed its own gate (ESS<10, 2 div) -> tabulated Stagger offsets used; spectra A vs Laplace-IS agree with each other but differ from saved power-2 stellar-informed by +86 ppm mean, -23 ppm/um slope, 39 ppm scatter after removing mean+slope (apples-to-oranges; LD law + prior). Left for later testing (needs longer calibration fit).
- 2026-09-02 03:18:22 CDT: parity HAT-P-26_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-26_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 03:19:23 WL PRISM Laplace TA=0.99 passes 3-seed gate (0 divergences, shifts <=0.09 sigma, 48-69 s MCMC + 52 s prep vs ~1100 s adaptive). Recommended per-mode WL flags: SOSS/G395H TA 0.9, PRISM 0.99; global default stays adaptive. P2 fourth pass: end-to-end compile-box timing via fit_jwst.py + chunk_mode parallel PRISM.
- 2026-09-02 03:28:02 parity session died (upstream capacity, 3.1M tokens); fresh parity_v2 session launched to feed/validate/report; queue still running 202/213/214/215
- 2026-09-02 03:37:43 CDT: parity HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 03:38:47 CDT: PARITY-V2 completed HAT-P-11_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 03:38:47 CDT: PARITY-V2 completed HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT; candidates=A,B,C
- 2026-09-02 03:38:47 CDT: PARITY-V2 completed HAT-P-26_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 03:38:47 CDT: PARITY-V2 completed HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 03:38:47 CDT: PARITY-V2 completed WASP-107_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT; candidates=A
- 2026-09-02 03:38:47 CDT: PARITY-V2 completed WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,C
- 2026-09-02 03:50:23 CDT: parity WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=B; result=/scratch/midway3/tfairnington/accel_parity/20260902c/WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 03:55:59 CDT: PARITY-V2 completed NGTS-2_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT_LINEAR_DISCONTINUITY; candidates=A
- 2026-09-02 03:59:30 CDT: parity WASP-107_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-107_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT/result_final.json
- 2026-09-02 04:07:29 CDT: PARITY-V2 completed WASP-121_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_QUADRATIC; candidates=A
- 2026-09-02 04:34:37 P2 compile-box matrix (V100, full pipeline): warm-cache process speedup only 1.29x SOSS / 1.32x G395H; padded box changes MCMC realization (8-15 ppm max channel diff, MC-level, not bias) -> compile_box stays opt-in/unrecommended. End-to-end walls today: SOSS 652-839 s, G395H 487-695 s (WL stage 150-300 s incl ~1400 compile events). PRISM serial/parallel test queued.
- 2026-09-02 04:39:29 CDT: parity WASP-121_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_QUADRATIC completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-121_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_QUADRATIC/result_final.json
- 2026-09-02 04:50:11 CDT: PARITY-V2 completed WASP-127_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 04:55:12 CDT: PARITY-V2 completed WASP-17_SOSS_ORDER1_V2_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 05:00:13 CDT: PARITY-V2 completed WASP-166_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 05:03:32 CDT: parity HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=D; result=/scratch/midway3/tfairnington/accel_parity/20260902d/HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 05:04:33 CDT: parity WASP-127_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-127_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 05:12:36 CDT: parity WASP-17_SOSS_ORDER1_V2_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-17_SOSS_ORDER1_V2_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 05:16:03 HAT-P-30 flag resolved: production control D shows the same 61 ppm scatter vs saved run (INPUT-DIFF); A/B/C CONTROL-PASS vs D. WASP-17, WASP-121, WASP-107, NGTS-2 pass. PRISM serial speed run started.
- 2026-09-02 05:18:45 CDT: PARITY-V2 completed WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 05:28:48 CDT: PARITY-V2 completed WASP-52_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_2SPOT; candidates=A
- 2026-09-02 05:33:48 CDT: parity WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 05:34:51 CDT: parity WASP-166_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-166_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 05:48:07 CDT: parity WASP-52_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_2SPOT completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-52_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_2SPOT/result_final.json
- 2026-09-02 05:56:48 CDT: PARITY-V2 completed WASP-69_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 06:04:59 PRISM full pipeline (HAT-P-65, accelerated flags, V100): 3367 s whole process (WL 238 s, low 719 s, high 2393 s for 106 ch) vs production ~5-6 h; parallel halves queued
- 2026-09-02 06:06:49 CDT: PARITY-V2 completed WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR; candidates=A
- 2026-09-02 06:06:49 CDT: PARITY-V2 completed WASP-94_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 06:11:50 CDT: PARITY-V2 completed WASP-96_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 06:21:39 CDT: parity WASP-69_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-69_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 06:25:08 CDT: parity WASP-96_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-96_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 06:25:34 CDT: parity WASP-94_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-94_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 06:25:57 WASP-63 SOSS (explinear trend) candidate A FLAG: 84 divergences, 93.7% pass, offset -0.15 ppm; explinear correlations (c-v-A-log_tau) likely need target 0.99 like PRISM; B/C pending. Parity tally: 42 PASS, 3 CONTROL-PASS, 1 INPUT-DIFF, 1 FLAG. PRISM parallel half 0 running.
- 2026-09-02 06:49:05 CDT: parity WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR/result_final.json
- 2026-09-02 06:49:39 P2 eighth pass: first PRISM 'halves' were invalid (low-res only); corrected high-res halves queued. Final recommended flag set written to speed_p2.md (Laplace WL + Laplace NUTS spectro + ESS gates; compile_box unrecommended).
- 2026-09-02 06:49:52 WASP-63 explinear: A/B/C all FLAG (divergences 84/37, C 48 fallbacks); production control D expected from parity session. Tally 44 PASS / 3 CONTROL-PASS / 1 INPUT-DIFF / 3 FLAG (all WASP-63).
- 2026-09-02 06:55:35 CDT: PARITY-V2 completed HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR; candidates=A
- 2026-09-02 07:05:38 CDT: PARITY-V2 completed HAT-P-11_G395H_NRS2_V1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 07:11:07 4th FLAG: HAT-P-11 G395H V2 (explinear) A: 10 divergences, 91.8% pass, bias -0.8 ppm. Pattern = explinear trend needs higher target accept (as PRISM). Launching explinear_fix session (candidate A99: target 0.99, depth 10).
- 2026-09-02 07:27 CDT explinear_fix: added immutable-dump A99 replay (TA 0.99, depth 10, warmup 200) and queued WASP-63 as 242; still pending behind four unchanged running parity jobs after 13 min, so HAT-P-11 was not queued, D remained pending, and no production auto-rule was enabled. Report: acceleration_reports/explinear_fix.md.
- 2026-09-02 08:11 CDT explinear_fix continuation: WASP-63 A99 harvested from result files: 0 divergences, 94.26% calibrated rows (strict gate false), -0.227 ppm offset, -0.137 ppm/um slope, 5.051 ppm RMS, depth ESS min/p05/median 241/346/908. HAT-P-11 V2 A99 queued as 243 but remained oldest pending after 23 min behind five running jobs; production auto-rule withheld pending its result and because WASP-63 aggregate gate remains false.
- 2026-09-02 07:15:42 CDT: PARITY-V2 completed HAT-P-11_G395H_NRS2_V2_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A
- 2026-09-02 07:20:48 CDT: PARITY-V2 completed HAT-P-11_G395H_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR; candidates=A
- 2026-09-02 07:22:06 5th FLAG: HAT-P-11 G395H NRS1 V1 (explinear) A: 10 div, 93.8% pass, bias 0.04 ppm; ALL flags are explinear datasets -> A99 (target 0.99, depth 10) queued via explinear_fix session. Note explinear G395H candidate walls 27-57 min (ESS-fallback reruns), vs 12 min for linear-trend G395H.
- 2026-09-02 07:33:09 queue reprioritised (A99, WASP-63 control, PRISM halves ahead of further parity); 5th GPU requested
- 2026-09-02 07:33:54 CDT: parity HAT-P-11_G395H_NRS2_V1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-11_G395H_NRS2_V1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 07:34:35 note: a 'touch' on 242_explinear_wasp63_a99.sh raced the dispatcher and created an empty duplicate that ran for 1 s (exit 0) on V100g; the real A99 run is on GPU10 (started 07:32:59); its done/242.out log was truncated by the duplicate but the result files under /scratch are unaffected.
- 2026-09-02 07:47:16 CDT: parity HAT-P-11_G395H_NRS2_V2_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-11_G395H_NRS2_V2_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 08:02:20 CDT: parity HAT-P-26_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-26_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 08:19:10 HAT-P-11 G395H V2 A99 replay (243) failed: parity run 227 timed out so result_final.json is absent; not retried before 09:00. PRISM high-res parallel half 0 done (even global chunks), half 1 running. WASP-63 A99: 0 divergences, 94.3% gate, bias -0.23 ppm; control D running.
- 2026-09-02 08:22:11 CDT: parity HAT-P-26_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-26_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 08:25:41 CDT: parity WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR completed; candidates=D; result=/scratch/midway3/tfairnington/accel_parity/20260902e/WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR/result_final.json
- 2026-09-02 08:29:33 CDT: parity HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 08:29:48 WASP-63 resolved: production control D fails the saved gate identically (93.97%) -> INPUT-DIFF; A/B/C match D (offsets 0.17-0.81 ppm) -> CONTROL-PASS. Remaining flags: HAT-P-11 G395H explinear V1/V2 (runs timed out at 90 min; A99 replay not possible without result_final).
- 2026-09-02 08:44:48 CDT: parity HAT-P-30_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-30_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 08:48 CDT P2 ninth pass: corrected PRISM halves computed even/odd R50 global chunks (V100; high intervals 1489/2642 s), CPU combine 0.333 s. Equality vs serial failed because each worker regenerated a different white-light handoff/fingerprint; critical path 2642 s vs serial high 2393 s (0.906x). Final recommendations appended to speed_p2.md; no GPU work pending.
- 2026-09-02 08:49:25 PRISM two-GPU split: halves regenerated their own white-light geometry (different fingerprints) -> combined samples not equal to serial; critical path 2642 s vs 2393 s serial (0.91x, chunk imbalance). Needs a shared immutable handoff before parallel workers launch; unrecommended for now. P2 final flag set written to speed_p2.md.
- 2026-09-02 08:51:10 CDT: parity HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT/result_final.json
- 2026-09-02 09:04:28 CDT: parity HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 09:06:18 CDT: parity HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 09:08:09 CDT: parity HAT-P-12_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-12_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 09:09:57 CDT: parity HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR completed; candidates=A,B,C; result=/scratch/midway3/tfairnington/accel_parity/20260902a/HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR/result_final.json
- 2026-09-02 09:15:00 CDT: parity HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT completed; candidates=D; result=/scratch/midway3/tfairnington/accel_parity/20260902f/HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT/result_final.json
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-12_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT; candidates=A,B,C,D
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-26_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-26_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-30_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:18:21 CDT: PARITY-V2 completed HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR; candidates=A,B,C
- 2026-09-02 09:19:43 FINAL parity tally: 31 datasets, 89 candidate rows: 71 PASS, 6 CONTROL-PASS, 2 INPUT-DIFF, 10 FLAG (explinear HAT-P-11 G395H divergences at target 0.95 [A99 untested there]; SOSS order-2 slope flags are a metric artefact of the 0.25-um baseline: the production control itself shows +13 ppm/um, pass fractions 99-100%). Last parity job (HAT-P-65 PRISM) still running; GPU allocations expire 10:45-11:30. No files deleted or overwritten.
- 2026-09-02 10:48:31 day 2: user chose exact samplers (HMC-8/NUTS) as production path, Laplace-IS opt-in; launching explinear_spectroscopic (fixed timescale) and Sing calibration fix; new GPU requested
- 2026-09-02 11:18:51 CDT: parity HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC completed; candidates=A; result=/scratch/midway3/tfairnington/accel_parity/20260902_explinspec_retry/HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC/result_final.json
- 2026-09-02 11:28:27 CDT: parity WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC completed; candidates=A; result=/scratch/midway3/tfairnington/accel_parity/20260902_explinspec_v2/WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC/result_final.json
- 2026-09-02 11:46:43 validation summary: dashed reference assumed ESS=N; regenerating with ESS-based expectation; c shift -0.018 sigma attributed to jitter-prior change (0.2 ppm baseline, depth shift -0.001 sigma)
- 2026-09-02 11:55:56 CDT: parity HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC completed; candidates=A; result=/scratch/midway3/tfairnington/accel_parity/20260902_explinspec_v3/HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR_EXPLINSPEC/result_final.json
- 2026-09-02 15:09–15:26 CDT: queue 272_harmonica_pooled_gate (GPU13 job 57401380). Two extra joint-NUTS seeds for WASP-94 Harmonica R20, pooled 3-seed reference + calibrated gates written to /scratch/midway3/tfairnington/accel_gpu_results/272_harmonica_pooled_gate/. Laplace NUTS TA0.95 and TA0.99 pass 96/96 calibrated rows (addendum in harmonica_accel.md, noise floor copied to harmonica_pooled_noise_floor.txt). No files deleted.
- 2026-09-02 16:41–18:10 CDT: diagnosed wide-LD Maxted boundary/random-init failure and ran queues 293–294; corrected Maxted narrowly failed (1 divergence, c2 ESS min 397), linear failed (69 divergences), so coefficient defaults remain; details appended to defaults_and_ld_reparam.md.
- 2026-09-02 15:1x CDT: repository initialised with git (branch main, commit 2c6ea24) with a .gitignore excluding data/products/logs; nothing removed from disk.
- 2026-09-02: Added the 15-page Sphinx/MyST/Furo user documentation for the JWST light-curve fitter; warning-as-error HTML build passed and 14 focused CPU tests passed; see acceleration_reports/docs_build.md.
- 2026-09-02 16:xx CDT: production Laplace defaults, exact selective NUTS/HMC/adaptive swap, fixed explinear timescale, and opt-in exact LD transforms implemented; CPU 36/36 pass; GPU 283 and 290-292 exit 0, G395H 282 completed products but dispatcher exit 127; see defaults_and_ld_reparam.md.
- 2026-09-02: Expanded the public documentation to 3,232 MyST lines across 17 built pages, added concepts/FAQ and nine existing example figures; clean `sphinx-build -E -W` passed; details appended to acceleration_reports/docs_build.md.
2026-09-02 — Added and validated per-channel PSIS-LOO stacking prototype on WASP-39 NRS1 R20 (4 LD variants; 0/9100 k-hat flags; outputs/report in acceleration_reports/stacking/).
- 2026-09-02 18:10 CDT: wide-LD follow-up complete; queues 293–294 finished with no pending jobs, neither Maxted nor linear met the zero-divergence/LD-ESS gate, and coefficient defaults remain (defaults_and_ld_reparam.md).
- 2026-09-02 CDT: inverse-CDF `latent_gaussian` LD implemented and tested; GPU 294/295 exited 0 but failed wide-LD divergence/ESS/parity gates, so it remains opt-in and coefficient defaults remain; see defaults_and_ld_reparam.md.

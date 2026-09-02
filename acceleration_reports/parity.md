# Overnight spectroscopic parity campaign

Last refreshed: 2026-09-02 09:18:48 CDT.

Candidates: A = Laplace-preconditioned NUTS; B = Laplace-preconditioned fixed HMC-8; C = Laplace importance sampling v3. All comparisons use the saved STELLARINFORMED posterior for the identical dataset.

## Per-dataset validation

| Dataset | Cand. | Wall min | depth ESS med/min | offset ppm | slope ppm/um | RMS ppm | sigma med [5%,95%] | pass | div / max k / fallback | Verdict |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HAT-P-11_G395H_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR | A | 56.6 | 2.27e+03/1.11e+03 | 0.0389 | 1.35 | 1.69 | 1 [0.935,1.08] | 93.79% | 10 / — / 0 | FLAG |
| HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR | A | 27.5 | 1.45e+03/461 | -0.771 | -1.89 | 2.63 | 1 [0.941,1.06] | 91.83% | 10 / — / 0 | FLAG |
| HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR | B | 35.0 | 2.06e+03/444 | -0.0995 | -1.87 | 2.5 | 0.992 [0.938,1.07] | 87.42% | 4 / — / 0 | FLAG |
| HAT-P-11_G395H_NRS2_V1_STELLARINFORMEDLD_POWER2_LINEAR | A | 21.4 | 2.45e+03/1.4e+03 | 0.624 | 0.37 | 5.57 | 0.995 [0.935,1.06] | 99.49% | 4 / — / 0 | PASS |
| HAT-P-11_G395H_NRS2_V1_STELLARINFORMEDLD_POWER2_LINEAR | B | 19.4 | 6.6e+03/1.1e+03 | 0.523 | -0.642 | 5.43 | 0.999 [0.924,1.06] | 99.49% | 3 / — / 0 | PASS |
| HAT-P-11_G395H_NRS2_V1_STELLARINFORMEDLD_POWER2_LINEAR | C | 11.1 | 955/695 | 0.193 | -0.326 | 6.02 | 1 [0.946,1.06] | 99.11% | 0 / 0.367 / 0 | PASS |
| HAT-P-11_G395H_NRS2_V2_STELLARINFORMEDLD_POWER2_LINEAR | A | 25.7 | 2.42e+03/1.64e+03 | 0.475 | 1.34 | 6.14 | 1 [0.935,1.07] | 99.24% | 1 / — / 0 | PASS |
| HAT-P-11_G395H_NRS2_V2_STELLARINFORMEDLD_POWER2_LINEAR | B | 17.5 | 6.6e+03/1.09e+03 | 0.365 | 1.72 | 6.13 | 1 [0.931,1.07] | 99.36% | 1 / — / 0 | PASS |
| HAT-P-11_G395H_NRS2_V2_STELLARINFORMEDLD_POWER2_LINEAR | C | 14.9 | 957/690 | 0.146 | 1.39 | 7.13 | 1 [0.948,1.08] | 98.09% | 0 / 0.594 / 0 | PASS |
| HAT-P-11_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 15.2 | 3.17e+03/1.99e+03 | -0.0117 | -0.311 | 1.34 | 1 [0.928,1.07] | 100% | 0 / — / 0 | PASS |
| HAT-P-11_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 14.7 | 1.38e+03/495 | -0.121 | -0.19 | 2.28 | 1 [0.944,1.08] | 99.15% | 0 / — / 0 | PASS |
| HAT-P-11_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 13.2 | 963/632 | -0.327 | -0.0494 | 1.6 | 1 [0.943,1.07] | 99.15% | 0 / 0.547 / 0 | PASS |
| HAT-P-12_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | A | 19.1 | 2.26e+03/1.4e+03 | -1.3 | 0.135 | 9.63 | 0.997 [0.937,1.07] | 99.36% | 12 / — / 0 | PASS |
| HAT-P-12_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | B | 15.6 | 6.6e+03/1.13e+03 | -1.33 | -0.284 | 8.74 | 0.995 [0.923,1.08] | 99.52% | 3 / — / 0 | PASS |
| HAT-P-12_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | C | 11.1 | 944/607 | -1.05 | -0.535 | 11.6 | 0.998 [0.932,1.07] | 98.41% | 0 / 0.787 / 2 | PASS |
| HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | A | 13.4 | 3.13e+03/2.01e+03 | 0.649 | 0.849 | 6.18 | 1 [0.922,1.07] | 99.27% | 0 / — / 0 | PASS |
| HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | B | 12.1 | 3.02e+03/674 | 1.59 | 1.95 | 7.56 | 1.01 [0.936,1.08] | 99.39% | 0 / — / 0 | PASS |
| HAT-P-12_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | C | 13.0 | 922/257 | 3.49 | -1.17 | 8.26 | 1.01 [0.928,1.06] | 95.76% | 0 / 0.71 / 2 | PASS |
| HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT | A | 10.8 | 3.12e+03/2.23e+03 | 2.38 | 6.49 | 6.04 | 0.996 [0.93,1.07] | 100% | 0 / — / 0 | FLAG |
| HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT | B | 7.4 | 2.1e+03/819 | -4.41 | 21.9 | 8.91 | 0.993 [0.937,1.04] | 99.23% | 0 / — / 0 | FLAG |
| HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT | C | 7.1 | 951/785 | 0.055 | 5.35 | 7.71 | 0.997 [0.944,1.06] | 99.61% | 0 / 0.424 / 0 | FLAG |
| HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT | D | 10.4 | 1.77e+03/1.2e+03 | 0.553 | 13.4 | 6.92 | 0.999 [0.929,1.04] | 99.61% | 0 / — / 0 | FLAG |
| HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | A | 16.9 | 2.22e+03/1.47e+03 | 0.156 | -0.198 | 11.9 | 0.993 [0.931,1.07] | 99.2% | 14 / — / 0 | PASS |
| HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | B | 9.9 | 6.6e+03/1.3e+03 | -0.678 | -0.00455 | 11.2 | 1 [0.916,1.08] | 99.68% | 5 / — / 0 | PASS |
| HAT-P-18_G395M_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | C | 8.1 | 947/650 | -0.289 | 0.865 | 14.8 | 0.998 [0.935,1.07] | 97.92% | 0 / 0.552 / 0 | PASS |
| HAT-P-26_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | A | 11.5 | 2.15e+03/1.45e+03 | -0.699 | -0.233 | 4.52 | 0.992 [0.93,1.06] | 99.02% | 8 / — / 0 | PASS |
| HAT-P-26_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | B | 8.2 | 6.6e+03/1.76e+03 | -0.514 | -0.0762 | 4.27 | 1 [0.916,1.08] | 98.77% | 1 / — / 0 | PASS |
| HAT-P-26_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | C | 7.2 | 948/748 | -0.56 | -0.889 | 5.68 | 1 [0.932,1.07] | 98.04% | 0 / 0.614 / 0 | PASS |
| HAT-P-26_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR | A | 13.0 | 2.21e+03/1.33e+03 | -1.29 | -0.989 | 9.8 | 1 [0.941,1.08] | 97.71% | 4 / — / 0 | PASS |
| HAT-P-26_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR | B | 8.9 | 6.6e+03/1.35e+03 | -1.57 | -2.18 | 9.52 | 1 [0.918,1.09] | 98.6% | 0 / — / 0 | PASS |
| HAT-P-26_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR | C | 6.4 | 936/613 | -1.89 | -0.358 | 10.8 | 1.01 [0.938,1.08] | 96.31% | 0 / 0.611 / 0 | PASS |
| HAT-P-26_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 11.9 | 3.04e+03/2.01e+03 | 0.974 | 0.262 | 4.91 | 0.996 [0.93,1.05] | 99.72% | 0 / — / 0 | PASS |
| HAT-P-26_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 8.0 | 1.49e+03/500 | -2.06 | 1.62 | 5.72 | 0.994 [0.916,1.07] | 99.15% | 0 / — / 0 | PASS |
| HAT-P-26_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 9.0 | 949/461 | -1.02 | 0.787 | 4.95 | 0.991 [0.944,1.06] | 98.16% | 0 / 0.812 / 1 | PASS |
| HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | A | 10.8 | 3.05e+03/2.08e+03 | -3.71 | 21.6 | 6.45 | 1 [0.943,1.05] | 100% | 0 / — / 0 | FLAG |
| HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | B | 8.6 | 1.35e+03/575 | -2.75 | 3.01 | 5.74 | 1 [0.941,1.07] | 99.1% | 0 / — / 0 | PASS |
| HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | C | 8.0 | 966/747 | -1.52 | 13.2 | 6.79 | 1 [0.925,1.08] | 99.1% | 0 / 0.157 / 0 | FLAG |
| HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | A | 11.7 | 2.19e+03/1.53e+03 | -0.038 | -0.614 | 3.55 | 0.991 [0.933,1.05] | 100% | 3 / — / 0 | PASS |
| HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | B | 7.9 | 6.6e+03/1.98e+03 | 0.0741 | -0.049 | 2.95 | 0.999 [0.904,1.06] | 100% | 1 / — / 0 | PASS |
| HAT-P-30_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | C | 7.5 | 934/730 | 0.376 | 0.537 | 5.66 | 0.999 [0.932,1.06] | 98.77% | 0 / 0.655 / 0 | PASS |
| HAT-P-30_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR | A | 13.6 | 2.25e+03/1.53e+03 | -0.777 | -1.28 | 8.9 | 1 [0.94,1.08] | 98.22% | 4 / — / 0 | PASS |
| HAT-P-30_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR | B | 8.9 | 6.6e+03/1.56e+03 | -1.06 | -3.38 | 8.41 | 1 [0.913,1.07] | 98.98% | 4 / — / 0 | PASS |
| HAT-P-30_G395H_NRS2_STELLARINFORMEDLD_POWER2_LINEAR | C | 6.7 | 934/596 | -1.64 | -0.684 | 9.38 | 1 [0.934,1.07] | 97.84% | 0 / 0.641 / 0 | PASS |
| HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 14.5 | 3.15e+03/2.06e+03 | -0.817 | -1.34 | 61.8 | 1.01 [0.923,1.45] | 79.1% | 0 / — / 0 | CONTROL-PASS |
| HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 10.2 | 1.42e+03/512 | -0.728 | -0.826 | 62 | 1.01 [0.931,1.5] | 78.67% | 0 / — / 0 | CONTROL-PASS |
| HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 8.5 | 966/519 | -0.596 | -1.31 | 63.1 | 1.01 [0.914,1.45] | 78.81% | 0 / 0.587 / 0 | CONTROL-PASS |
| HAT-P-30_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | D | 23.9 | 1.9e+03/995 | -1.4 | -0.0295 | 61.1 | 1.01 [0.928,1.5] | 78.53% | 0 / — / 0 | INPUT-DIFF |
| HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | A | 8.7 | 3.22e+03/1.99e+03 | -0.115 | -3.66 | 3.29 | 1.01 [0.944,1.08] | 100% | 0 / — / 0 | PASS |
| HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | B | 6.5 | 1.1e+03/575 | 1.01 | -9.28 | 5.26 | 1.01 [0.928,1.08] | 100% | 0 / — / 0 | FLAG |
| HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR | C | 6.2 | 951/656 | 0.398 | -4.33 | 5.61 | 1 [0.952,1.09] | 98.2% | 0 / 0.133 / 0 | PASS |
| NGTS-2_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT_LINEAR_DISCONTINUITY | A | 43.2 | 3.02e+03/2.06e+03 | -3.67 | 1.45 | 5.97 | 1 [0.929,1.07] | 95.91% | 0 / — / 0 | PASS |
| NGTS-2_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT_LINEAR_DISCONTINUITY | B | 37.3 | 1.98e+03/509 | -3.18 | 0.832 | 5.47 | 0.996 [0.931,1.07] | 97.9% | 0 / — / 0 | PASS |
| WASP-107_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | A | 16.8 | 2.82e+03/1.9e+03 | -2.53 | 0.908 | 4.22 | 1 [0.942,1.06] | 99.52% | 0 / — / 0 | PASS |
| WASP-107_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | B | 11.6 | 2.43e+03/588 | -2.45 | 1.63 | 4.48 | 1.01 [0.945,1.08] | 98.31% | 0 / — / 0 | PASS |
| WASP-107_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_SPOT | C | 12.5 | 944/485 | -1.67 | 0.569 | 5.29 | 1.01 [0.948,1.06] | 97.82% | 0 / 0.665 / 0 | PASS |
| WASP-121_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_QUADRATIC | A | 21.5 | 2.26e+03/209 | -0.0478 | 0.465 | 10.5 | 1 [0.943,1.07] | 98.37% | 50 / — / 0 | PASS |
| WASP-121_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_QUADRATIC | B | 15.4 | 1.97e+03/270 | -0.172 | 1.75 | 11 | 1.01 [0.934,1.07] | 98.12% | 176 / — / 0 | PASS |
| WASP-121_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_QUADRATIC | C | 22.2 | 916/83.5 | -0.28 | 1.96 | 11.3 | 0.996 [0.934,1.07] | 97.24% | 0 / 0.822 / 6 | PASS |
| WASP-127_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 17.7 | 3.19e+03/2.05e+03 | -0.669 | 0.201 | 3.01 | 0.997 [0.925,1.07] | 100% | 0 / — / 0 | PASS |
| WASP-127_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 9.7 | 1.07e+03/579 | -0.558 | 0.514 | 4 | 1 [0.926,1.08] | 99.44% | 0 / — / 0 | PASS |
| WASP-127_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 9.0 | 962/701 | -0.414 | 0.231 | 3.57 | 0.996 [0.94,1.07] | 99.15% | 0 / 0.401 / 0 | PASS |
| WASP-166_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 26.7 | 3.17e+03/1.74e+03 | -1.49 | 0.918 | 3.17 | 0.994 [0.943,1.08] | 98.31% | 0 / — / 0 | PASS |
| WASP-166_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 23.3 | 952/504 | -1.82 | 1.47 | 4.91 | 1 [0.929,1.08] | 96.61% | 0 / — / 0 | PASS |
| WASP-166_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 16.3 | 962/700 | -0.653 | 0.402 | 4.44 | 0.999 [0.934,1.07] | 99.44% | 0 / 0.124 / 0 | PASS |
| WASP-17_SOSS_ORDER1_V2_STELLARINFORMEDLD_POWER2_LINEAR | A | 12.8 | 3.12e+03/1.87e+03 | -0.858 | 0.331 | 5.93 | 0.996 [0.936,1.08] | 99.69% | 0 / — / 0 | PASS |
| WASP-17_SOSS_ORDER1_V2_STELLARINFORMEDLD_POWER2_LINEAR | B | 9.4 | 1.25e+03/407 | -0.656 | -0.309 | 7.83 | 0.994 [0.942,1.07] | 98.92% | 0 / — / 0 | PASS |
| WASP-17_SOSS_ORDER1_V2_STELLARINFORMEDLD_POWER2_LINEAR | C | 8.5 | 983/712 | 0.629 | -0.374 | 6.36 | 1 [0.952,1.06] | 99.23% | 0 / 0.367 / 0 | PASS |
| WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 11.9 | 3.14e+03/2.06e+03 | -1.18 | 0.663 | 5.42 | 0.996 [0.931,1.07] | 100% | 0 / — / 0 | PASS |
| WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 8.1 | 1.4e+03/519 | -0.537 | -0.478 | 6.36 | 0.996 [0.922,1.07] | 99.58% | 0 / — / 0 | PASS |
| WASP-39_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 9.2 | 963/654 | 0.297 | -0.219 | 6.8 | 0.999 [0.93,1.07] | 98.45% | 0 / 0.754 / 1 | PASS |
| WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | A | 12.6 | 2.2e+03/1.51e+03 | -1.19 | 1.18 | 6.76 | 0.993 [0.923,1.06] | 98.28% | 3 / — / 0 | PASS |
| WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | B | 12.6 | 6.6e+03/1.61e+03 | -1.02 | -1.75 | 5.76 | 0.985 [0.915,1.1] | 98.53% | 3 / — / 0 | PASS |
| WASP-52_G395H_NRS1_STELLARINFORMEDLD_POWER2_LINEAR | C | 8.8 | 958/731 | -1.16 | -2.28 | 8.75 | 0.992 [0.934,1.07] | 97.55% | 0 / 0.591 / 0 | PASS |
| WASP-52_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_2SPOT | A | 15.0 | 2.6e+03/1.54e+03 | 1.34 | -0.117 | 8.61 | 1 [0.948,1.06] | 99.68% | 0 / — / 0 | PASS |
| WASP-52_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_2SPOT | B | 10.8 | 1.7e+03/533 | 3.59 | -1.8 | 8.92 | 1.01 [0.942,1.05] | 99.58% | 0 / — / 0 | PASS |
| WASP-52_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_2SPOT | C | 9.5 | 926/470 | 4.33 | -1.21 | 11.5 | 0.995 [0.945,1.06] | 98.83% | 0 / 0.539 / 0 | PASS |
| WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR | A | 28.0 | 1.14e+03/379 | -0.151 | 0.464 | 6.54 | 1.01 [0.929,1.06] | 93.69% | 84 / — / 0 | CONTROL-PASS |
| WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR | B | 22.0 | 1.41e+03/20 | 0.197 | -0.243 | 7.14 | 0.999 [0.926,1.08] | 93.41% | 37 / — / 0 | CONTROL-PASS |
| WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR | C | 25.1 | 587/121 | 0.48 | 0.138 | 8.32 | 0.996 [0.889,1.08] | 88.7% | 0 / 0.944 / 48 | CONTROL-PASS |
| WASP-63_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_EXPLINEAR | D | 39.8 | 753/301 | -0.303 | 0.665 | 5.6 | 0.995 [0.926,1.07] | 93.97% | 0 / — / 0 | INPUT-DIFF |
| WASP-69_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 19.4 | 3.16e+03/1.84e+03 | -0.242 | 0.171 | 2.68 | 1 [0.939,1.08] | 100% | 0 / — / 0 | PASS |
| WASP-69_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 15.1 | 1.26e+03/490 | -0.11 | -0.224 | 2.53 | 0.996 [0.938,1.08] | 100% | 0 / — / 0 | PASS |
| WASP-69_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 12.2 | 955/668 | -0.325 | 0.189 | 2.98 | 1 [0.942,1.06] | 99.44% | 0 / 0.521 / 0 | PASS |
| WASP-94_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 15.7 | 3.15e+03/2e+03 | 0.358 | -0.178 | 3.35 | 1.01 [0.933,1.08] | 99.72% | 0 / — / 0 | PASS |
| WASP-94_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 11.3 | 1e+03/487 | 0.75 | -0.793 | 4.31 | 0.998 [0.926,1.09] | 99.15% | 0 / — / 0 | PASS |
| WASP-94_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 10.4 | 967/679 | 0.298 | -0.007 | 4.24 | 1.01 [0.939,1.07] | 99.15% | 0 / 0.31 / 0 | PASS |
| WASP-96_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | A | 11.1 | 3.03e+03/2.03e+03 | 0.895 | 0.929 | 11.5 | 0.997 [0.922,1.08] | 99.44% | 0 / — / 0 | PASS |
| WASP-96_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | B | 6.6 | 1.37e+03/563 | -0.866 | 0.803 | 14.1 | 0.998 [0.907,1.08] | 99.15% | 0 / — / 0 | PASS |
| WASP-96_SOSS_ORDER1_STELLARINFORMEDLD_POWER2_LINEAR | C | 7.7 | 959/622 | 0.764 | 1.16 | 14.9 | 0.994 [0.933,1.07] | 97.18% | 0 / 0.668 / 0 | PASS |

## Aggregate distributions

- **A (Laplace NUTS, n=30):** median [range] offset -0.197 [-3.71,2.38] ppm; slope 0.296 [-3.66,21.6] ppm/um; RMS 5.95 [1.34,61.8] ppm; sigma ratio 1 [0.991,1.01]; pass 99.4% [79.1,100]% .
- **B (Laplace HMC-8, n=29):** median [range] offset -0.558 [-4.41,3.59] ppm; slope -0.19 [-9.28,21.9] ppm/um; RMS 5.76 [2.28,62] ppm; sigma ratio 1 [0.985,1.01]; pass 99.15% [78.67,100]% .
- **C (Laplace-IS v3, n=27):** median [range] offset -0.289 [-1.89,4.33] ppm; slope -0.007 [-4.33,13.2] ppm/um; RMS 6.8 [1.6,63.1] ppm; sigma ratio 1 [0.991,1.01]; pass 98.2% [78.81,99.61]% .
- **D (production joint NUTS control, n=3):** median [range] offset -0.303 [-1.4,0.553] ppm; slope 0.665 [-0.0295,13.4] ppm/um; RMS 6.92 [5.6,61.1] ppm; sigma ratio 0.999 [0.995,1.01]; pass 93.97% [78.53,99.61]% .

## Verdict

Accelerated samplers match the regenerated production control: **no / incomplete**. Flagged: HAT-P-11_G395H_NRS1_V1_STELLARINFORMEDLD_POWER2_EXPLINEAR/A, HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR/A, HAT-P-11_G395H_NRS1_V2_STELLARINFORMEDLD_POWER2_EXPLINEAR/B, HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT/A, HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT/B, HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT/C, HAT-P-12_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_SPOT/D, HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR/A, HAT-P-26_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR/C, HAT-P-30_SOSS_ORDER2_STELLARINFORMEDLD_POWER2_LINEAR/B.

This is a fidelity campaign, not a controlled speed comparison: wall times are reported, but no speedup is claimed because baselines were not rerun on identical GPU/input pairs.

## Queue and datasets not run

Unsubmitted templates: none.

A template counted as submitted may still be pending, running, failed, or incomplete. Completed result datasets discovered: 30; completed candidate comparisons: 89.

## What was built or changed

- `tools/campaign/monitor_parity_v2.py`: progressive result validator and report generator.
- `tools/campaign/run_parity_dataset.py`: added candidate D (production joint NUTS, log-uniform jitter, 1000/1000).
- `acceleration_reports/gpu_queue/{done,pending}/240-242_parity_*_joint_control.sh`: immutable production-control runs for each flagged dataset.
- `acceleration_reports/parity.md`: this progressively refreshed campaign report.
- `fit_jwst.py`: inherited additive safeguard strips HMC-only options before selective NUTS fallback; no source edit was needed in this takeover.

## Reproduction

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 0-15 \
  /home/tfairnington/miniconda3/envs/jaxoplanet/bin/python -m pytest \
  tests/test_spectro_safety_guards.py tests/test_independent_hmc.py \
  tests/test_independent_runner_reuse.py -x -q

/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/campaign/feed_parity_queue.py --first 213 --max-pending 2 \
  --poll-seconds 30 --stop 09:00

/home/tfairnington/miniconda3/envs/jaxoplanet/bin/python \
  tools/campaign/monitor_parity_v2.py --root /scratch/midway3/tfairnington/accel_parity --queue acceleration_reports/gpu_queue \
  --report acceleration_reports/parity.md --stop 09:30
```

GPU dataset commands are preserved verbatim in `acceleration_reports/gpu_queue/done/*_parity_*.sh`; stdout, exit code, and GPU identity are adjacent `.out`, `.exit`, and `.gpu` files.

## Failures and open risks

- Early A/B serializer failures are retained in immutable `20260902a` results; successful retries supersede them candidate-by-candidate.
- Candidate B's low-depth-ESS lanes can selectively fall back to NUTS. The HMC-only-option forwarding bug is fixed and its focused CPU suite passed 17 tests; GPU retry 205 provides end-to-end confirmation.
- Kepler-12 PRISM reached the queue's 90-minute limit (exit 124) before producing a final result. It was not resubmitted; its partial immutable outputs remain available for diagnosis.
- HAT-P-30 SOSS is resolved as a saved-run/input-revision discrepancy, not a sampler failure. Production-control D (joint NUTS, log-uniform jitter, 1000/1000) also differs from saved by 61.1 ppm RMS, 78.5% pass, and sigma-ratio p95 1.497. A/B/C versus D have only 6.36/7.39/7.30 ppm RMS, offsets 0.65/0.71/0.85 ppm, slopes -1.40/-0.89/-1.69 ppm/um, and median sigma ratios 0.997/1.002/0.997. Saved and regenerated products have byte-identical white-light and spectroscopic masks, R20/reference channel grids, and spectroscopy-data pickle; both use stellar-informed power-2 LD. White-light geometry shifts by at most 0.075 saved sigma. The remaining identifiable difference is pipeline revision/RNG: saved products date 2026-05-09 and lack current artifact/config manifests, while regenerated products use the current 2026-09-02 pipeline. The exact historical code revision is unrecoverable from the saved directory.
- WASP-63 SOSS is likewise a saved-run/control discrepancy: production D itself passes only 93.97% against saved (5.60 ppm RMS), while A/B/C versus D have offsets 0.17/0.53/0.81 ppm, slopes -0.16/-0.86/-0.57 ppm/um, RMS 7.53/6.19/8.39 ppm, and median sigma ratios 1.002/0.999/0.989. The accelerated samplers therefore pass the regenerated production control; candidate B's depth-ESS minimum of 20 remains a caution for that single lane.
- HAT-P-12 SOSS order 2 remains FLAGGED/inconclusive. Against saved, A/B/C/D slopes are +6.49/+21.86/+5.35/+13.40 ppm/um (all offsets within 4.5 ppm and pass fractions 99.2-100%). Against regenerated production D, A/B/C slopes are -7.29/+9.15/-6.76 ppm/um with RMS 6.91/9.99/9.28 ppm; these exceed the predeclared slope threshold, so the accelerated candidates do not earn a control-pass classification. The very short order-2 wavelength lever arm makes slope Monte-Carlo-sensitive, but the stated gate is applied without post-hoc relaxation.
- Flags discovered at/after the 09:00 submission cutoff remain unresolved: HAT-P-26 SOSS order 2 (A/C slope), HAT-P-30 SOSS order 2 (B slope), and partial HAT-P-11 G395H NRS1 explinear V1/V2 (pass fraction and divergences). No post-cutoff D controls were submitted. HAT-P-65 PRISM was still running at final harvest; Kepler-12 PRISM timed out, so these datasets have no completed parity result.
- Dataset-level parity does not establish a speedup, and a single Monte-Carlo realization can conceal small biases; flagged thresholds and diagnostics must be reviewed for every new result.

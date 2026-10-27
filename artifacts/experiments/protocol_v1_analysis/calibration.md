# Calibration

ECE = sum over 10 equal-width bins of (n_b/N)|acc_b-conf_b|; temperature fit on val NLL (L-BFGS).

## cross_subject
- fold0: T=2.109, raw ece 11.78pp nll 0.910 brier 0.327 -> cal ece 2.51pp
  rejection c100: cov 100.0 acc 79.3; c90: cov 90.0 acc 83.1; c80: cov 79.8 acc 86.7; c70: cov 69.6 acc 90.2
- fold1: T=1.588, raw ece 6.75pp nll 0.600 brier 0.264 -> cal ece 1.94pp
  rejection c100: cov 100.0 acc 82.3; c90: cov 90.0 acc 86.7; c80: cov 79.4 acc 90.3; c70: cov 68.9 acc 93.0
- fold2: T=1.495, raw ece 9.50pp nll 0.762 brier 0.318 -> cal ece 2.31pp
  rejection c100: cov 100.0 acc 78.7; c90: cov 91.2 acc 82.9; c80: cov 81.4 acc 86.7; c70: cov 71.4 acc 89.8
- fold3: T=1.500, raw ece 6.78pp nll 0.602 brier 0.275 -> cal ece 1.27pp
  rejection c100: cov 100.0 acc 81.5; c90: cov 94.3 acc 84.0; c80: cov 86.9 acc 86.8; c70: cov 78.0 acc 89.8
- fold4: T=1.578, raw ece 13.02pp nll 1.074 brier 0.382 -> cal ece 6.98pp
  rejection c100: cov 100.0 acc 76.0; c90: cov 90.7 acc 79.4; c80: cov 81.1 acc 81.8; c70: cov 72.7 acc 84.3
## cross_day
- s01: T=0.857, raw ece 1.89pp nll 0.255 brier 0.121 -> cal ece 3.08pp
  rejection c100: cov 98.4 acc 92.6; c90: cov 73.1 acc 98.9; c80: cov 60.2 acc 99.6; c70: cov 51.6 acc 99.7
- s02: T=1.481, raw ece 10.26pp nll 0.925 brier 0.360 -> cal ece 5.41pp
  rejection c100: cov 100.0 acc 77.0; c90: cov 75.8 acc 85.3; c80: cov 54.7 acc 92.0; c70: cov 42.2 acc 94.7
- s03: T=0.950, raw ece 6.25pp nll 0.753 brier 0.316 -> cal ece 5.84pp
  rejection c100: cov 99.7 acc 79.6; c90: cov 60.2 acc 88.9; c80: cov 40.8 acc 90.4; c70: cov 29.6 acc 91.0
- s04: T=0.816, raw ece 6.44pp nll 0.823 brier 0.314 -> cal ece 9.44pp
  rejection c100: cov 99.8 acc 79.3; c90: cov 74.4 acc 89.2; c80: cov 56.1 acc 91.4; c70: cov 45.6 acc 91.9
- s05: T=1.025, raw ece 15.00pp nll 1.686 brier 0.400 -> cal ece 14.78pp
  rejection c100: cov 98.3 acc 77.1; c90: cov 67.8 acc 83.7; c80: cov 52.6 acc 82.1; c70: cov 41.0 acc 78.2
- s06: T=0.679, raw ece 2.17pp nll 0.281 brier 0.131 -> cal ece 2.94pp
  rejection c100: cov 100.0 acc 91.6; c90: cov 90.0 acc 95.2; c80: cov 79.5 acc 97.6; c70: cov 67.6 acc 98.6
- s07: T=0.945, raw ece 1.91pp nll 0.381 brier 0.197 -> cal ece 2.65pp
  rejection c100: cov 99.8 acc 86.7; c90: cov 73.9 acc 94.7; c80: cov 57.3 acc 97.6; c70: cov 44.6 acc 98.7
- s08: T=0.834, raw ece 3.72pp nll 0.342 brier 0.154 -> cal ece 3.32pp
  rejection c100: cov 100.0 acc 90.8; c90: cov 94.0 acc 93.6; c80: cov 88.7 acc 94.4; c70: cov 67.6 acc 96.9
- s09: T=1.044, raw ece 2.76pp nll 0.255 brier 0.104 -> cal ece 3.33pp
  rejection c100: cov 100.0 acc 93.5; c90: cov 87.3 acc 97.6; c80: cov 74.4 acc 99.1; c70: cov 65.4 acc 99.6
- s10: T=1.275, raw ece 2.48pp nll 0.337 brier 0.136 -> cal ece 2.27pp
  rejection c100: cov 100.0 acc 91.7; c90: cov 94.0 acc 94.0; c80: cov 89.1 acc 95.1; c70: cov 81.9 acc 96.3
- s11: T=0.729, raw ece 6.57pp nll 0.668 brier 0.282 -> cal ece 10.21pp
  rejection c100: cov 99.8 acc 82.2; c90: cov 59.5 acc 93.4; c80: cov 45.3 acc 96.1; c70: cov 34.2 acc 97.5
- s12: T=0.891, raw ece 3.32pp nll 0.480 brier 0.169 -> cal ece 3.37pp
  rejection c100: cov 99.6 acc 89.6; c90: cov 76.1 acc 95.4; c80: cov 61.3 acc 95.5; c70: cov 49.7 acc 95.3
- s13: T=0.397, raw ece 6.85pp nll 0.679 brier 0.264 -> cal ece 13.60pp
  rejection c100: cov 99.2 acc 83.4; c90: cov 64.0 acc 94.3; c80: cov 46.9 acc 97.0; c70: cov 33.8 acc 98.2
- s14: T=1.365, raw ece 2.22pp nll 0.341 brier 0.145 -> cal ece 8.05pp
  rejection c100: cov 99.7 acc 91.3; c90: cov 94.6 acc 93.0; c80: cov 90.6 acc 94.0; c70: cov 80.9 acc 96.1
- s15: T=0.968, raw ece 3.85pp nll 0.467 brier 0.193 -> cal ece 3.99pp
  rejection c100: cov 100.0 acc 87.4; c90: cov 76.2 acc 94.8; c80: cov 58.7 acc 97.3; c70: cov 46.6 acc 98.1
- s16: T=0.572, raw ece 1.42pp nll 0.081 brier 0.037 -> cal ece 0.95pp
  rejection c100: cov 100.0 acc 97.9; c90: cov 84.6 acc 99.7; c80: cov 67.6 acc 100.0; c70: cov 50.2 acc 100.0
- s17: T=0.867, raw ece 0.79pp nll 0.291 brier 0.146 -> cal ece 2.06pp
  rejection c100: cov 96.5 acc 91.8; c90: cov 51.8 acc 99.5; c80: cov 39.7 acc 99.8; c70: cov 32.7 acc 99.9
- s18: T=0.140, raw ece 5.15pp nll 0.433 brier 0.173 -> cal ece 9.22pp
  rejection c100: cov 98.9 acc 90.0; c90: cov 77.7 acc 95.0; c80: cov 58.7 acc 95.0; c70: cov 42.0 acc 95.7
- s19: T=0.852, raw ece 0.54pp nll 0.230 brier 0.109 -> cal ece 1.81pp
  rejection c100: cov 99.5 acc 93.0; c90: cov 67.1 acc 99.5; c80: cov 50.8 acc 99.8; c70: cov 37.8 acc 99.9
- s20: T=1.299, raw ece 1.93pp nll 0.399 brier 0.206 -> cal ece 8.23pp
  rejection c100: cov 93.9 acc 88.4; c90: cov 44.5 acc 99.0; c80: cov 28.0 acc 99.7; c70: cov 21.0 acc 100.0
- s21: T=1.086, raw ece 2.34pp nll 0.464 brier 0.195 -> cal ece 4.06pp
  rejection c100: cov 100.0 acc 87.3; c90: cov 76.7 acc 95.4; c80: cov 61.0 acc 98.2; c70: cov 48.9 acc 98.8
- s22: T=0.951, raw ece 5.94pp nll 0.554 brier 0.233 -> cal ece 6.64pp
  rejection c100: cov 99.9 acc 84.2; c90: cov 59.3 acc 98.3; c80: cov 46.2 acc 99.2; c70: cov 36.6 acc 99.7
- s23: T=0.322, raw ece 2.68pp nll 0.191 brier 0.094 -> cal ece 4.20pp
  rejection c100: cov 99.0 acc 94.4; c90: cov 66.8 acc 99.1; c80: cov 53.3 acc 99.6; c70: cov 41.4 acc 99.9
- s24: T=0.996, raw ece 4.57pp nll 0.439 brier 0.217 -> cal ece 4.60pp
  rejection c100: cov 94.0 acc 88.5; c90: cov 53.1 acc 96.5; c80: cov 43.7 acc 97.9; c70: cov 32.4 acc 98.4
- s25: T=0.873, raw ece 2.55pp nll 0.239 brier 0.116 -> cal ece 0.84pp
  rejection c100: cov 99.6 acc 92.7; c90: cov 62.8 acc 99.4; c80: cov 41.0 acc 100.0; c70: cov 32.3 acc 100.0
- s26: T=0.807, raw ece 1.35pp nll 0.441 brier 0.226 -> cal ece 4.26pp
  rejection c100: cov 98.5 acc 84.8; c90: cov 75.5 acc 93.2; c80: cov 59.7 acc 97.0; c70: cov 48.5 acc 98.8
- s27: T=0.990, raw ece 4.22pp nll 0.400 brier 0.176 -> cal ece 4.41pp
  rejection c100: cov 100.0 acc 89.3; c90: cov 78.8 acc 94.1; c80: cov 66.0 acc 95.8; c70: cov 56.8 acc 96.6
- s28: T=1.036, raw ece 3.74pp nll 0.155 brier 0.072 -> cal ece 4.20pp
  rejection c100: cov 99.4 acc 96.0; c90: cov 63.2 acc 99.7; c80: cov 42.4 acc 99.9; c70: cov 30.0 acc 100.0
- s29: T=0.648, raw ece 2.03pp nll 0.181 brier 0.091 -> cal ece 2.00pp
  rejection c100: cov 100.0 acc 94.0; c90: cov 79.4 acc 99.1; c80: cov 66.4 acc 99.7; c70: cov 56.4 acc 100.0
- s30: T=0.623, raw ece 3.16pp nll 0.300 brier 0.123 -> cal ece 5.17pp
  rejection c100: cov 99.1 acc 93.0; c90: cov 76.7 acc 96.8; c80: cov 64.1 acc 97.3; c70: cov 51.3 acc 97.9
- s31: T=0.805, raw ece 7.74pp nll 0.637 brier 0.250 -> cal ece 9.81pp
  rejection c100: cov 97.5 acc 85.2; c90: cov 63.7 acc 94.1; c80: cov 49.5 acc 94.8; c70: cov 39.9 acc 95.1
- s32: T=1.189, raw ece 4.96pp nll 0.612 brier 0.182 -> cal ece 3.24pp
  rejection c100: cov 97.7 acc 90.1; c90: cov 70.0 acc 95.3; c80: cov 58.8 acc 95.7; c70: cov 49.2 acc 95.8
- s33: T=1.088, raw ece 1.58pp nll 0.204 brier 0.105 -> cal ece 1.16pp
  rejection c100: cov 100.0 acc 93.3; c90: cov 77.8 acc 98.6; c80: cov 68.5 acc 98.9; c70: cov 53.9 acc 99.5
- s34: T=0.650, raw ece 1.16pp nll 0.123 brier 0.062 -> cal ece 2.20pp
  rejection c100: cov 100.0 acc 96.3; c90: cov 78.3 acc 99.6; c80: cov 62.2 acc 99.8; c70: cov 50.6 acc 99.8
- s35: T=0.611, raw ece 2.15pp nll 0.143 brier 0.047 -> cal ece 1.12pp
  rejection c100: cov 99.7 acc 97.4; c90: cov 87.9 acc 99.7; c80: cov 71.3 acc 99.9; c70: cov 58.6 acc 100.0
- s36: T=0.725, raw ece 2.44pp nll 0.114 brier 0.051 -> cal ece 0.39pp
  rejection c100: cov 99.6 acc 96.8; c90: cov 58.1 acc 100.0; c80: cov 37.5 acc 100.0; c70: cov 28.9 acc 100.0
- s37: T=0.487, raw ece 2.75pp nll 0.501 brier 0.248 -> cal ece 10.93pp
  rejection c100: cov 96.7 acc 83.9; c90: cov 56.2 acc 97.5; c80: cov 42.6 acc 99.2; c70: cov 33.3 acc 99.9
- s38: T=1.099, raw ece 4.93pp nll 0.482 brier 0.244 -> cal ece 3.44pp
  rejection c100: cov 99.8 acc 83.4; c90: cov 85.7 acc 89.1; c80: cov 64.3 acc 94.3; c70: cov 44.5 acc 97.7
- s39: T=0.778, raw ece 2.01pp nll 0.063 brier 0.028 -> cal ece 0.53pp
  rejection c100: cov 100.0 acc 98.4; c90: cov 88.1 acc 99.9; c80: cov 74.6 acc 100.0; c70: cov 59.1 acc 100.0
- s40: T=1.291, raw ece 6.35pp nll 0.596 brier 0.241 -> cal ece 2.29pp
  rejection c100: cov 98.2 acc 85.4; c90: cov 65.5 acc 95.7; c80: cov 52.2 acc 97.4; c70: cov 42.4 acc 98.2
- s41: T=0.910, raw ece 5.21pp nll 0.528 brier 0.234 -> cal ece 6.34pp
  rejection c100: cov 96.6 acc 86.1; c90: cov 61.4 acc 96.7; c80: cov 46.6 acc 98.4; c70: cov 36.1 acc 99.3
- s42: T=0.974, raw ece 1.55pp nll 0.328 brier 0.156 -> cal ece 1.72pp
  rejection c100: cov 99.9 acc 89.5; c90: cov 81.5 acc 96.3; c80: cov 67.9 acc 98.6; c70: cov 55.4 acc 99.4
- s43: T=1.306, raw ece 1.07pp nll 0.200 brier 0.091 -> cal ece 3.65pp
  rejection c100: cov 96.0 acc 95.7; c90: cov 58.6 acc 99.5; c80: cov 44.7 acc 99.6; c70: cov 36.8 acc 99.6
## within_subject
- s01_d1: T=0.866, raw ece 2.42pp nll 0.621 brier 0.218 -> cal ece 4.74pp
  rejection c100: cov 99.6 acc 85.9; c90: cov 80.0 acc 93.8; c80: cov 70.8 acc 95.3; c70: cov 64.2 acc 95.9
- s01_d2: T=0.573, raw ece 3.18pp nll 0.295 brier 0.141 -> cal ece 4.86pp
  rejection c100: cov 99.8 acc 90.5; c90: cov 84.8 acc 95.8; c80: cov 73.6 acc 97.4; c70: cov 67.3 acc 98.3
- s01_d3: T=0.567, raw ece 7.51pp nll 0.220 brier 0.082 -> cal ece 2.00pp
  rejection c100: cov 98.1 acc 97.2; c90: cov 83.9 acc 98.1; c80: cov 69.8 acc 98.6; c70: cov 58.7 acc 98.6
- s02_d1: T=1.508, raw ece 7.04pp nll 1.040 brier 0.355 -> cal ece 6.77pp
  rejection c100: cov 100.0 acc 77.5; c90: cov 91.3 acc 82.1; c80: cov 78.7 acc 85.8; c70: cov 66.6 acc 88.4
- s02_d2: T=0.611, raw ece 9.66pp nll 0.275 brier 0.120 -> cal ece 1.40pp
  rejection c100: cov 100.0 acc 93.4; c90: cov 91.8 acc 96.1; c80: cov 83.5 acc 97.9; c70: cov 75.6 acc 98.9
- s02_d3: T=0.610, raw ece 12.47pp nll 0.239 brier 0.110 -> cal ece 4.10pp
  rejection c100: cov 99.9 acc 94.7; c90: cov 89.3 acc 98.4; c80: cov 77.3 acc 100.0; c70: cov 65.1 acc 100.0
- s03_d1: T=0.713, raw ece 5.09pp nll 0.265 brier 0.101 -> cal ece 0.70pp
  rejection c100: cov 99.8 acc 94.2; c90: cov 79.6 acc 98.8; c80: cov 67.1 acc 99.0; c70: cov 57.6 acc 99.0
- s03_d2: T=0.579, raw ece 4.78pp nll 0.160 brier 0.067 -> cal ece 1.06pp
  rejection c100: cov 99.7 acc 96.3; c90: cov 86.4 acc 99.3; c80: cov 71.3 acc 99.4; c70: cov 60.4 acc 99.7
- s03_d3: T=0.735, raw ece 12.69pp nll 0.443 brier 0.198 -> cal ece 4.21pp
  rejection c100: cov 99.9 acc 89.1; c90: cov 87.0 acc 93.1; c80: cov 72.8 acc 96.3; c70: cov 64.1 acc 97.8
- s04_d1: T=0.520, raw ece 5.25pp nll 0.279 brier 0.135 -> cal ece 3.53pp
  rejection c100: cov 100.0 acc 91.3; c90: cov 88.4 acc 95.6; c80: cov 79.0 acc 97.3; c70: cov 67.9 acc 98.3
- s04_d2: T=1.002, raw ece 4.00pp nll 0.320 brier 0.108 -> cal ece 4.06pp
  rejection c100: cov 100.0 acc 93.2; c90: cov 89.3 acc 97.6; c80: cov 79.6 acc 98.9; c70: cov 69.3 acc 99.4
- s04_d3: T=0.864, raw ece 2.98pp nll 0.369 brier 0.157 -> cal ece 0.98pp
  rejection c100: cov 99.2 acc 90.2; c90: cov 86.2 acc 95.8; c80: cov 73.1 acc 97.9; c70: cov 60.9 acc 99.4
- s05_d1: T=0.330, raw ece 2.39pp nll 0.167 brier 0.068 -> cal ece 2.90pp
  rejection c100: cov 99.7 acc 96.5; c90: cov 88.1 acc 98.3; c80: cov 79.0 acc 98.4; c70: cov 68.7 acc 98.8
- s05_d2: T=0.769, raw ece 2.63pp nll 0.204 brier 0.098 -> cal ece 2.10pp
  rejection c100: cov 99.2 acc 94.2; c90: cov 82.9 acc 98.9; c80: cov 66.7 acc 100.0; c70: cov 52.9 acc 100.0
- s05_d3: T=0.197, raw ece 8.09pp nll 0.285 brier 0.125 -> cal ece 4.79pp
  rejection c100: cov 100.0 acc 94.2; c90: cov 89.5 acc 96.4; c80: cov 77.0 acc 96.9; c70: cov 65.5 acc 97.1
- s06_d1: T=0.782, raw ece 5.25pp nll 0.315 brier 0.157 -> cal ece 1.79pp
  rejection c100: cov 100.0 acc 90.2; c90: cov 92.2 acc 93.1; c80: cov 84.8 acc 94.5; c70: cov 75.3 acc 96.5
- s06_d2: T=0.849, raw ece 3.51pp nll 0.297 brier 0.138 -> cal ece 2.15pp
  rejection c100: cov 99.2 acc 91.4; c90: cov 88.3 acc 95.4; c80: cov 79.4 acc 97.9; c70: cov 68.1 acc 99.3
- s06_d3: T=0.566, raw ece 4.91pp nll 0.413 brier 0.176 -> cal ece 5.89pp
  rejection c100: cov 99.8 acc 89.8; c90: cov 93.9 acc 91.8; c80: cov 85.7 acc 93.4; c70: cov 77.8 acc 94.1
- s07_d1: T=0.765, raw ece 2.14pp nll 0.310 brier 0.154 -> cal ece 4.19pp
  rejection c100: cov 99.9 acc 89.9; c90: cov 86.2 acc 94.8; c80: cov 76.4 acc 97.2; c70: cov 65.3 acc 98.3
- s07_d2: T=0.448, raw ece 2.88pp nll 0.089 brier 0.038 -> cal ece 1.54pp
  rejection c100: cov 100.0 acc 97.5; c90: cov 88.8 acc 99.6; c80: cov 82.7 acc 99.7; c70: cov 72.5 acc 99.8
- s07_d3: T=0.619, raw ece 9.97pp nll 0.220 brier 0.087 -> cal ece 1.68pp
  rejection c100: cov 100.0 acc 95.3; c90: cov 91.8 acc 98.3; c80: cov 78.3 acc 99.6; c70: cov 71.3 acc 99.9
- s08_d1: T=1.291, raw ece 7.76pp nll 1.108 brier 0.379 -> cal ece 7.40pp
  rejection c100: cov 99.9 acc 74.9; c90: cov 87.0 acc 80.6; c80: cov 76.8 acc 85.1; c70: cov 68.2 acc 87.3
- s08_d2: T=0.603, raw ece 7.24pp nll 0.118 brier 0.046 -> cal ece 2.05pp
  rejection c100: cov 99.7 acc 98.5; c90: cov 90.3 acc 99.9; c80: cov 85.4 acc 99.9; c70: cov 76.3 acc 100.0
- s08_d3: T=1.038, raw ece 2.37pp nll 0.395 brier 0.142 -> cal ece 2.69pp
  rejection c100: cov 99.8 acc 91.3; c90: cov 88.8 acc 95.4; c80: cov 79.0 acc 97.8; c70: cov 69.0 acc 98.4
- s09_d1: T=0.618, raw ece 4.46pp nll 0.292 brier 0.138 -> cal ece 3.72pp
  rejection c100: cov 100.0 acc 89.7; c90: cov 71.0 acc 99.6; c80: cov 67.0 acc 99.7; c70: cov 56.2 acc 99.9
- s09_d2: T=1.061, raw ece 3.36pp nll 0.569 brier 0.198 -> cal ece 2.80pp
  rejection c100: cov 99.8 acc 87.9; c90: cov 92.0 acc 91.2; c80: cov 83.6 acc 92.8; c70: cov 75.8 acc 94.0
- s09_d3: T=0.260, raw ece 2.98pp nll 0.095 brier 0.043 -> cal ece 1.70pp
  rejection c100: cov 98.9 acc 98.0; c90: cov 91.5 acc 99.6; c80: cov 86.6 acc 99.8; c70: cov 79.8 acc 100.0
- s10_d1: T=0.715, raw ece 8.80pp nll 1.162 brier 0.466 -> cal ece 4.96pp
  rejection c100: cov 99.9 acc 69.4; c90: cov 86.1 acc 74.0; c80: cov 76.8 acc 76.7; c70: cov 65.7 acc 79.0
- s10_d2: T=0.456, raw ece 2.06pp nll 0.305 brier 0.133 -> cal ece 5.21pp
  rejection c100: cov 98.7 acc 92.5; c90: cov 80.8 acc 96.8; c80: cov 64.9 acc 97.9; c70: cov 59.6 acc 99.6
- s10_d3: T=0.533, raw ece 5.65pp nll 0.132 brier 0.056 -> cal ece 0.64pp
  rejection c100: cov 99.9 acc 97.1; c90: cov 86.4 acc 99.3; c80: cov 72.3 acc 99.5; c70: cov 60.5 acc 99.9
- s11_d1: T=0.812, raw ece 5.80pp nll 0.252 brier 0.105 -> cal ece 3.72pp
  rejection c100: cov 100.0 acc 92.4; c90: cov 90.0 acc 98.1; c80: cov 81.8 acc 99.8; c70: cov 72.4 acc 100.0
- s11_d2: T=0.437, raw ece 4.61pp nll 0.220 brier 0.107 -> cal ece 3.25pp
  rejection c100: cov 99.8 acc 93.1; c90: cov 84.3 acc 98.1; c80: cov 72.2 acc 99.4; c70: cov 63.5 acc 99.5
- s11_d3: T=0.326, raw ece 3.36pp nll 0.078 brier 0.033 -> cal ece 1.23pp
  rejection c100: cov 99.6 acc 98.2; c90: cov 89.1 acc 99.9; c80: cov 80.3 acc 100.0; c70: cov 69.4 acc 100.0
- s12_d1: T=0.497, raw ece 2.11pp nll 0.167 brier 0.073 -> cal ece 2.31pp
  rejection c100: cov 99.8 acc 95.5; c90: cov 87.7 acc 98.9; c80: cov 80.1 acc 99.5; c70: cov 74.9 acc 99.4
- s12_d2: T=0.735, raw ece 3.72pp nll 0.293 brier 0.148 -> cal ece 3.35pp
  rejection c100: cov 99.7 acc 91.0; c90: cov 85.3 acc 94.9; c80: cov 69.2 acc 97.2; c70: cov 58.7 acc 98.6
- s12_d3: T=0.432, raw ece 13.33pp nll 0.221 brier 0.087 -> cal ece 0.60pp
  rejection c100: cov 100.0 acc 96.8; c90: cov 93.9 acc 98.4; c80: cov 83.1 acc 99.9; c70: cov 69.2 acc 99.9
- s13_d1: T=0.805, raw ece 2.60pp nll 0.423 brier 0.173 -> cal ece 3.69pp
  rejection c100: cov 100.0 acc 87.7; c90: cov 93.3 acc 90.7; c80: cov 80.7 acc 96.2; c70: cov 69.7 acc 97.8
- s13_d2: T=0.353, raw ece 3.56pp nll 0.351 brier 0.166 -> cal ece 8.05pp
  rejection c100: cov 100.0 acc 88.2; c90: cov 81.9 acc 96.0; c80: cov 73.3 acc 98.3; c70: cov 65.9 acc 99.5
- s13_d3: T=0.405, raw ece 4.41pp nll 0.222 brier 0.099 -> cal ece 3.72pp
  rejection c100: cov 99.9 acc 93.4; c90: cov 85.5 acc 98.3; c80: cov 78.6 acc 99.5; c70: cov 72.0 acc 99.6
- s14_d1: T=2.222, raw ece 27.11pp nll 2.482 brier 0.632 -> cal ece 14.66pp
  rejection c100: cov 99.9 acc 64.8; c90: cov 89.7 acc 69.5; c80: cov 81.4 acc 70.0; c70: cov 70.3 acc 70.1
- s14_d2: T=0.862, raw ece 3.87pp nll 0.615 brier 0.232 -> cal ece 5.58pp
  rejection c100: cov 99.4 acc 86.1; c90: cov 88.2 acc 90.3; c80: cov 79.7 acc 92.1; c70: cov 67.8 acc 93.9
- s14_d3: T=0.607, raw ece 3.01pp nll 0.266 brier 0.107 -> cal ece 2.72pp
  rejection c100: cov 100.0 acc 93.7; c90: cov 91.1 acc 96.1; c80: cov 84.4 acc 97.7; c70: cov 73.2 acc 97.9
- s15_d1: T=0.735, raw ece 3.18pp nll 0.529 brier 0.214 -> cal ece 6.97pp
  rejection c100: cov 99.8 acc 85.0; c90: cov 91.1 acc 90.2; c80: cov 85.0 acc 92.4; c70: cov 74.5 acc 95.9
- s15_d2: T=0.800, raw ece 4.68pp nll 0.268 brier 0.117 -> cal ece 1.44pp
  rejection c100: cov 100.0 acc 92.4; c90: cov 93.0 acc 95.8; c80: cov 82.8 acc 97.9; c70: cov 69.2 acc 99.8
- s15_d3: T=0.838, raw ece 2.11pp nll 0.383 brier 0.190 -> cal ece 2.39pp
  rejection c100: cov 100.0 acc 86.0; c90: cov 86.6 acc 92.2; c80: cov 66.6 acc 98.1; c70: cov 54.3 acc 99.5
- s16_d1: T=0.571, raw ece 6.44pp nll 0.129 brier 0.048 -> cal ece 1.20pp
  rejection c100: cov 100.0 acc 97.9; c90: cov 89.5 acc 99.6; c80: cov 80.8 acc 99.8; c70: cov 72.0 acc 99.9
- s16_d2: T=0.050, raw ece 1.77pp nll 0.172 brier 0.083 -> cal ece 5.50pp
  rejection c100: cov 99.8 acc 94.5; c90: cov 95.0 acc 97.0; c80: cov 95.0 acc 97.0; c70: cov 95.0 acc 97.0
- s16_d3: T=0.308, raw ece 1.96pp nll 0.246 brier 0.122 -> cal ece 6.69pp
  rejection c100: cov 99.5 acc 90.6; c90: cov 79.8 acc 98.8; c80: cov 74.7 acc 99.6; c70: cov 67.7 acc 99.8
- s17_d1: T=0.730, raw ece 2.38pp nll 0.279 brier 0.134 -> cal ece 3.26pp
  rejection c100: cov 99.9 acc 90.7; c90: cov 91.3 acc 94.7; c80: cov 78.5 acc 98.2; c70: cov 67.2 acc 99.4
- s17_d2: T=0.389, raw ece 5.19pp nll 0.138 brier 0.053 -> cal ece 1.68pp
  rejection c100: cov 99.6 acc 97.1; c90: cov 88.5 acc 99.6; c80: cov 80.0 acc 99.8; c70: cov 74.5 acc 99.8
- s17_d3: T=0.200, raw ece 5.72pp nll 0.293 brier 0.129 -> cal ece 7.15pp
  rejection c100: cov 99.5 acc 91.5; c90: cov 86.4 acc 97.2; c80: cov 77.7 acc 99.0; c70: cov 66.3 acc 99.6
- s18_d1: T=0.050, raw ece 5.37pp nll 0.160 brier 0.071 -> cal ece 4.15pp
  rejection c100: cov 100.0 acc 95.8; c90: cov 86.4 acc 99.3; c80: cov 86.4 acc 99.3; c70: cov 86.4 acc 99.3
- s18_d2: T=0.482, raw ece 1.68pp nll 0.073 brier 0.037 -> cal ece 1.97pp
  rejection c100: cov 99.8 acc 97.5; c90: cov 89.3 acc 100.0; c80: cov 81.5 acc 100.0; c70: cov 70.7 acc 100.0
- s18_d3: T=0.619, raw ece 3.91pp nll 0.190 brier 0.078 -> cal ece 1.29pp
  rejection c100: cov 100.0 acc 95.3; c90: cov 88.7 acc 98.5; c80: cov 81.3 acc 99.1; c70: cov 72.1 acc 99.4
- s19_d1: T=1.001, raw ece 2.59pp nll 0.352 brier 0.172 -> cal ece 2.58pp
  rejection c100: cov 100.0 acc 88.2; c90: cov 86.4 acc 94.1; c80: cov 75.9 acc 95.8; c70: cov 68.7 acc 96.1
- s19_d2: T=0.569, raw ece 7.12pp nll 0.265 brier 0.124 -> cal ece 1.59pp
  rejection c100: cov 98.2 acc 93.2; c90: cov 78.9 acc 98.8; c80: cov 69.0 acc 99.4; c70: cov 59.2 acc 100.0
- s19_d3: T=0.607, raw ece 3.45pp nll 0.329 brier 0.151 -> cal ece 4.03pp
  rejection c100: cov 98.1 acc 91.2; c90: cov 81.4 acc 96.1; c80: cov 71.1 acc 98.0; c70: cov 63.4 acc 98.1
- s20_d1: T=0.516, raw ece 2.47pp nll 0.169 brier 0.079 -> cal ece 3.19pp
  rejection c100: cov 99.6 acc 95.2; c90: cov 89.7 acc 98.0; c80: cov 79.8 acc 99.6; c70: cov 71.9 acc 99.9
- s20_d2: T=0.373, raw ece 4.48pp nll 0.117 brier 0.050 -> cal ece 1.31pp
  rejection c100: cov 99.9 acc 97.3; c90: cov 83.8 acc 99.9; c80: cov 75.1 acc 100.0; c70: cov 68.2 acc 100.0
- s20_d3: T=0.548, raw ece 6.56pp nll 0.351 brier 0.178 -> cal ece 4.24pp
  rejection c100: cov 99.4 acc 87.7; c90: cov 84.4 acc 93.1; c80: cov 70.8 acc 97.5; c70: cov 60.5 acc 99.0
- s21_d1: T=0.359, raw ece 4.15pp nll 0.361 brier 0.158 -> cal ece 6.61pp
  rejection c100: cov 100.0 acc 89.7; c90: cov 92.3 acc 93.1; c80: cov 82.7 acc 95.3; c70: cov 73.1 acc 96.4
- s21_d2: T=1.210, raw ece 3.85pp nll 0.620 brier 0.245 -> cal ece 3.36pp
  rejection c100: cov 99.8 acc 83.4; c90: cov 86.8 acc 90.5; c80: cov 73.9 acc 95.0; c70: cov 64.5 acc 97.1
- s21_d3: T=0.805, raw ece 4.43pp nll 0.181 brier 0.080 -> cal ece 1.51pp
  rejection c100: cov 99.6 acc 95.1; c90: cov 90.9 acc 98.2; c80: cov 83.9 acc 98.9; c70: cov 73.9 acc 99.9
- s22_d1: T=0.516, raw ece 3.42pp nll 0.098 brier 0.035 -> cal ece 0.66pp
  rejection c100: cov 99.7 acc 98.4; c90: cov 87.7 acc 99.8; c80: cov 78.7 acc 99.9; c70: cov 69.6 acc 99.9
- s22_d2: T=0.449, raw ece 2.01pp nll 0.112 brier 0.052 -> cal ece 2.52pp
  rejection c100: cov 100.0 acc 96.5; c90: cov 90.2 acc 99.4; c80: cov 81.9 acc 100.0; c70: cov 75.1 acc 100.0
- s22_d3: T=0.986, raw ece 2.36pp nll 0.174 brier 0.083 -> cal ece 2.28pp
  rejection c100: cov 99.9 acc 94.5; c90: cov 89.7 acc 97.8; c80: cov 81.0 acc 99.3; c70: cov 73.0 acc 99.8
- s23_d1: T=0.277, raw ece 4.70pp nll 0.168 brier 0.080 -> cal ece 3.31pp
  rejection c100: cov 99.6 acc 95.4; c90: cov 83.9 acc 99.0; c80: cov 74.0 acc 99.4; c70: cov 65.5 acc 99.7
- s23_d2: T=0.383, raw ece 8.02pp nll 0.142 brier 0.055 -> cal ece 0.63pp
  rejection c100: cov 99.7 acc 97.4; c90: cov 92.7 acc 99.5; c80: cov 80.8 acc 100.0; c70: cov 69.2 acc 100.0
- s23_d3: T=0.277, raw ece 7.54pp nll 0.505 brier 0.230 -> cal ece 10.79pp
  rejection c100: cov 99.9 acc 84.8; c90: cov 78.7 acc 93.1; c80: cov 68.1 acc 96.8; c70: cov 61.8 acc 98.4
- s24_d1: T=0.568, raw ece 4.17pp nll 0.096 brier 0.027 -> cal ece 0.44pp
  rejection c100: cov 99.7 acc 99.0; c90: cov 97.0 acc 99.5; c80: cov 91.3 acc 99.5; c70: cov 84.8 acc 99.6
- s24_d2: T=0.050, raw ece 4.03pp nll 0.099 brier 0.036 -> cal ece 1.84pp
  rejection c100: cov 99.2 acc 98.4; c90: cov 93.0 acc 99.5; c80: cov 93.0 acc 99.5; c70: cov 93.0 acc 99.5
- s24_d3: T=0.050, raw ece 3.76pp nll 0.073 brier 0.026 -> cal ece 1.31pp
  rejection c100: cov 99.5 acc 98.9; c90: cov 93.7 acc 99.7; c80: cov 93.7 acc 99.7; c70: cov 93.7 acc 99.7
- s25_d1: T=0.627, raw ece 3.17pp nll 0.087 brier 0.039 -> cal ece 2.07pp
  rejection c100: cov 99.7 acc 97.0; c90: cov 95.1 acc 99.5; c80: cov 89.0 acc 100.0; c70: cov 80.7 acc 100.0
- s25_d2: T=0.406, raw ece 3.79pp nll 0.127 brier 0.054 -> cal ece 1.96pp
  rejection c100: cov 99.9 acc 97.0; c90: cov 92.1 acc 99.1; c80: cov 85.5 acc 99.8; c70: cov 79.7 acc 99.9
- s25_d3: T=0.340, raw ece 9.86pp nll 0.157 brier 0.057 -> cal ece 0.93pp
  rejection c100: cov 100.0 acc 98.2; c90: cov 85.1 acc 100.0; c80: cov 76.1 acc 100.0; c70: cov 66.4 acc 100.0
- s26_d1: T=0.404, raw ece 5.02pp nll 0.220 brier 0.103 -> cal ece 3.26pp
  rejection c100: cov 98.8 acc 93.5; c90: cov 76.9 acc 99.5; c80: cov 66.8 acc 99.9; c70: cov 56.6 acc 99.9
- s26_d2: T=0.329, raw ece 2.93pp nll 0.215 brier 0.094 -> cal ece 4.51pp
  rejection c100: cov 99.5 acc 94.3; c90: cov 89.2 acc 98.1; c80: cov 83.9 acc 99.0; c70: cov 77.7 acc 99.2
- s26_d3: T=0.554, raw ece 6.29pp nll 0.671 brier 0.261 -> cal ece 12.51pp
  rejection c100: cov 99.9 acc 82.4; c90: cov 79.2 acc 92.7; c80: cov 70.2 acc 95.4; c70: cov 63.3 acc 97.3
- s27_d1: T=0.203, raw ece 3.63pp nll 0.183 brier 0.087 -> cal ece 4.33pp
  rejection c100: cov 99.9 acc 94.7; c90: cov 88.0 acc 97.4; c80: cov 78.9 acc 99.1; c70: cov 69.7 acc 99.4
- s27_d2: T=1.020, raw ece 3.84pp nll 0.252 brier 0.097 -> cal ece 3.94pp
  rejection c100: cov 98.1 acc 94.3; c90: cov 82.1 acc 98.6; c80: cov 70.8 acc 98.6; c70: cov 60.3 acc 98.5
- s27_d3: T=0.285, raw ece 7.54pp nll 0.132 brier 0.051 -> cal ece 1.36pp
  rejection c100: cov 99.3 acc 98.1; c90: cov 92.4 acc 99.2; c80: cov 83.5 acc 100.0; c70: cov 78.3 acc 100.0
- s28_d1: T=0.050, raw ece 3.19pp nll 0.237 brier 0.080 -> cal ece 4.99pp
  rejection c100: cov 99.5 acc 95.2; c90: cov 89.9 acc 98.8; c80: cov 89.9 acc 98.8; c70: cov 89.9 acc 98.8
- s28_d2: T=0.050, raw ece 3.87pp nll 0.054 brier 0.016 -> cal ece 0.92pp
  rejection c100: cov 99.1 acc 99.6; c90: cov 94.6 acc 100.0; c80: cov 94.6 acc 100.0; c70: cov 94.6 acc 100.0
- s28_d3: T=0.630, raw ece 1.63pp nll 0.024 brier 0.007 -> cal ece 0.32pp
  rejection c100: cov 99.7 acc 99.8; c90: cov 94.8 acc 100.0; c80: cov 88.1 acc 100.0; c70: cov 82.0 acc 100.0
- s29_d1: T=0.661, raw ece 4.67pp nll 0.289 brier 0.104 -> cal ece 1.78pp
  rejection c100: cov 98.7 acc 95.1; c90: cov 81.8 acc 98.8; c80: cov 76.1 acc 99.2; c70: cov 70.7 acc 99.4
- s29_d2: T=0.370, raw ece 3.15pp nll 0.075 brier 0.030 -> cal ece 1.02pp
  rejection c100: cov 99.9 acc 98.4; c90: cov 92.6 acc 99.5; c80: cov 84.5 acc 99.9; c70: cov 74.8 acc 100.0
- s29_d3: T=0.050, raw ece 3.32pp nll 0.114 brier 0.047 -> cal ece 2.70pp
  rejection c100: cov 100.0 acc 97.3; c90: cov 90.7 acc 99.4; c80: cov 90.7 acc 99.4; c70: cov 90.7 acc 99.4
- s30_d1: T=0.285, raw ece 9.60pp nll 0.213 brier 0.097 -> cal ece 2.30pp
  rejection c100: cov 99.7 acc 94.4; c90: cov 89.2 acc 98.5; c80: cov 81.8 acc 99.5; c70: cov 65.5 acc 100.0
- s30_d2: T=0.759, raw ece 2.31pp nll 0.122 brier 0.063 -> cal ece 1.28pp
  rejection c100: cov 98.2 acc 96.1; c90: cov 88.2 acc 99.9; c80: cov 83.5 acc 100.0; c70: cov 77.6 acc 100.0
- s30_d3: T=0.302, raw ece 4.53pp nll 0.116 brier 0.034 -> cal ece 1.08pp
  rejection c100: cov 99.9 acc 98.4; c90: cov 92.4 acc 99.8; c80: cov 82.7 acc 99.8; c70: cov 69.5 acc 99.9
- s31_d1: T=0.297, raw ece 7.18pp nll 0.202 brier 0.083 -> cal ece 2.87pp
  rejection c100: cov 99.7 acc 95.4; c90: cov 90.1 acc 98.9; c80: cov 80.7 acc 99.6; c70: cov 71.9 acc 99.9
- s31_d2: T=0.906, raw ece 2.68pp nll 0.198 brier 0.084 -> cal ece 2.24pp
  rejection c100: cov 99.3 acc 94.8; c90: cov 83.1 acc 99.7; c80: cov 77.9 acc 99.7; c70: cov 72.1 acc 99.8
- s31_d3: T=0.491, raw ece 1.97pp nll 0.168 brier 0.065 -> cal ece 2.23pp
  rejection c100: cov 99.9 acc 96.6; c90: cov 87.9 acc 98.2; c80: cov 79.5 acc 98.4; c70: cov 69.6 acc 98.8
- s32_d1: T=0.960, raw ece 2.49pp nll 0.222 brier 0.080 -> cal ece 1.82pp
  rejection c100: cov 99.7 acc 95.9; c90: cov 93.3 acc 97.1; c80: cov 86.9 acc 97.3; c70: cov 78.5 acc 98.4
- s32_d2: T=0.050, raw ece 11.85pp nll 0.157 brier 0.048 -> cal ece 1.20pp
  rejection c100: cov 99.0 acc 99.2; c90: cov 85.8 acc 100.0; c80: cov 84.7 acc 100.0; c70: cov 84.7 acc 100.0
- s32_d3: T=0.647, raw ece 8.56pp nll 0.573 brier 0.210 -> cal ece 4.58pp
  rejection c100: cov 99.0 acc 85.0; c90: cov 81.8 acc 96.1; c80: cov 70.8 acc 98.8; c70: cov 60.0 acc 99.3
- s33_d1: T=0.930, raw ece 5.70pp nll 0.311 brier 0.125 -> cal ece 4.25pp
  rejection c100: cov 97.1 acc 93.9; c90: cov 84.2 acc 98.3; c80: cov 79.9 acc 99.3; c70: cov 71.7 acc 99.6
- s33_d2: T=0.582, raw ece 1.68pp nll 0.257 brier 0.108 -> cal ece 5.25pp
  rejection c100: cov 99.8 acc 94.3; c90: cov 89.8 acc 95.7; c80: cov 78.7 acc 96.7; c70: cov 67.6 acc 96.8
- s33_d3: T=1.106, raw ece 0.87pp nll 0.316 brier 0.155 -> cal ece 3.94pp
  rejection c100: cov 99.8 acc 89.7; c90: cov 82.9 acc 95.9; c80: cov 66.6 acc 99.9; c70: cov 59.2 acc 100.0
- s34_d1: T=0.289, raw ece 1.03pp nll 0.227 brier 0.125 -> cal ece 6.98pp
  rejection c100: cov 100.0 acc 91.8; c90: cov 88.9 acc 95.3; c80: cov 80.4 acc 97.5; c70: cov 69.1 acc 99.3
- s34_d2: T=0.050, raw ece 2.43pp nll 0.129 brier 0.064 -> cal ece 3.96pp
  rejection c100: cov 99.4 acc 96.1; c90: cov 91.6 acc 98.5; c80: cov 91.6 acc 98.5; c70: cov 91.6 acc 98.5
- s34_d3: T=0.522, raw ece 2.35pp nll 0.148 brier 0.071 -> cal ece 2.28pp
  rejection c100: cov 99.3 acc 95.7; c90: cov 87.4 acc 99.2; c80: cov 78.1 acc 100.0; c70: cov 72.7 acc 100.0
- s35_d1: T=0.773, raw ece 3.83pp nll 0.227 brier 0.112 -> cal ece 1.35pp
  rejection c100: cov 100.0 acc 92.4; c90: cov 93.6 acc 95.4; c80: cov 83.2 acc 98.0; c70: cov 74.5 acc 99.4
- s35_d2: T=0.676, raw ece 3.92pp nll 0.119 brier 0.051 -> cal ece 1.00pp
  rejection c100: cov 98.7 acc 97.3; c90: cov 79.2 acc 100.0; c80: cov 66.0 acc 100.0; c70: cov 58.0 acc 100.0
- s35_d3: T=0.244, raw ece 10.60pp nll 0.433 brier 0.154 -> cal ece 6.78pp
  rejection c100: cov 99.6 acc 91.2; c90: cov 81.8 acc 97.4; c80: cov 69.1 acc 99.3; c70: cov 58.0 acc 99.7
- s36_d1: T=0.533, raw ece 3.95pp nll 0.106 brier 0.042 -> cal ece 1.62pp
  rejection c100: cov 100.0 acc 98.1; c90: cov 93.0 acc 98.9; c80: cov 88.4 acc 99.3; c70: cov 82.8 acc 99.7
- s36_d2: T=0.618, raw ece 3.13pp nll 0.170 brier 0.060 -> cal ece 1.50pp
  rejection c100: cov 100.0 acc 96.1; c90: cov 87.5 acc 99.2; c80: cov 78.8 acc 99.1; c70: cov 64.5 acc 99.0
- s36_d3: T=0.050, raw ece 3.62pp nll 0.128 brier 0.058 -> cal ece 3.51pp
  rejection c100: cov 99.8 acc 96.5; c90: cov 90.4 acc 99.3; c80: cov 90.4 acc 99.3; c70: cov 90.4 acc 99.3
- s37_d1: T=0.353, raw ece 4.80pp nll 0.269 brier 0.128 -> cal ece 4.61pp
  rejection c100: cov 98.7 acc 92.1; c90: cov 78.0 acc 99.0; c80: cov 64.6 acc 99.7; c70: cov 57.0 acc 100.0
- s37_d2: T=0.501, raw ece 3.50pp nll 0.143 brier 0.066 -> cal ece 2.36pp
  rejection c100: cov 99.6 acc 96.4; c90: cov 86.3 acc 99.1; c80: cov 75.6 acc 99.6; c70: cov 67.3 acc 99.5
- s37_d3: T=0.575, raw ece 7.47pp nll 0.158 brier 0.065 -> cal ece 1.07pp
  rejection c100: cov 99.6 acc 96.9; c90: cov 88.9 acc 99.2; c80: cov 74.9 acc 100.0; c70: cov 61.3 acc 100.0
- s38_d1: T=1.671, raw ece 13.48pp nll 1.400 brier 0.436 -> cal ece 5.45pp
  rejection c100: cov 99.7 acc 70.5; c90: cov 82.1 acc 80.3; c80: cov 70.6 acc 85.6; c70: cov 63.4 acc 88.1
- s38_d2: T=0.649, raw ece 1.49pp nll 0.096 brier 0.052 -> cal ece 1.98pp
  rejection c100: cov 100.0 acc 96.4; c90: cov 92.9 acc 98.8; c80: cov 84.7 acc 99.6; c70: cov 76.8 acc 99.9
- s38_d3: T=0.687, raw ece 4.24pp nll 0.067 brier 0.022 -> cal ece 1.14pp
  rejection c100: cov 99.9 acc 99.1; c90: cov 92.9 acc 99.9; c80: cov 84.9 acc 100.0; c70: cov 72.9 acc 100.0
- s39_d1: T=0.224, raw ece 14.25pp nll 0.220 brier 0.080 -> cal ece 0.99pp
  rejection c100: cov 99.9 acc 97.6; c90: cov 93.3 acc 99.6; c80: cov 87.1 acc 99.7; c70: cov 74.9 acc 99.9
- s39_d2: T=1.169, raw ece 6.16pp nll 0.526 brier 0.224 -> cal ece 7.45pp
  rejection c100: cov 99.4 acc 87.4; c90: cov 72.7 acc 91.2; c80: cov 58.8 acc 91.3; c70: cov 52.2 acc 91.1
- s39_d3: T=0.918, raw ece 3.28pp nll 0.274 brier 0.123 -> cal ece 3.17pp
  rejection c100: cov 99.3 acc 92.1; c90: cov 94.5 acc 94.8; c80: cov 87.7 acc 97.5; c70: cov 73.9 acc 98.8
- s40_d1: T=1.096, raw ece 3.35pp nll 0.425 brier 0.167 -> cal ece 4.96pp
  rejection c100: cov 98.7 acc 90.8; c90: cov 83.7 acc 94.4; c80: cov 71.7 acc 94.7; c70: cov 63.0 acc 94.4
- s40_d2: T=0.788, raw ece 1.95pp nll 0.274 brier 0.091 -> cal ece 2.24pp
  rejection c100: cov 100.0 acc 94.7; c90: cov 82.4 acc 98.0; c80: cov 71.4 acc 98.1; c70: cov 59.8 acc 98.3
- s40_d3: T=0.802, raw ece 4.43pp nll 0.278 brier 0.099 -> cal ece 3.32pp
  rejection c100: cov 99.8 acc 94.9; c90: cov 90.1 acc 96.9; c80: cov 82.4 acc 97.3; c70: cov 74.4 acc 97.3
- s41_d1: T=0.832, raw ece 5.79pp nll 0.177 brier 0.080 -> cal ece 2.65pp
  rejection c100: cov 99.9 acc 96.3; c90: cov 86.7 acc 97.3; c80: cov 76.1 acc 98.0; c70: cov 67.6 acc 98.6
- s41_d2: T=0.849, raw ece 5.74pp nll 0.158 brier 0.072 -> cal ece 2.99pp
  rejection c100: cov 100.0 acc 95.7; c90: cov 91.7 acc 98.0; c80: cov 80.9 acc 99.2; c70: cov 73.1 acc 99.8
- s41_d3: T=0.582, raw ece 10.41pp nll 0.506 brier 0.243 -> cal ece 4.08pp
  rejection c100: cov 99.1 acc 83.9; c90: cov 81.6 acc 92.0; c80: cov 73.9 acc 93.4; c70: cov 59.6 acc 97.5
- s42_d1: T=0.561, raw ece 3.68pp nll 0.316 brier 0.132 -> cal ece 4.84pp
  rejection c100: cov 98.6 acc 91.2; c90: cov 86.7 acc 97.2; c80: cov 79.7 acc 98.8; c70: cov 75.1 acc 99.3
- s42_d2: T=0.679, raw ece 3.62pp nll 0.223 brier 0.092 -> cal ece 1.45pp
  rejection c100: cov 100.0 acc 94.5; c90: cov 92.6 acc 97.2; c80: cov 83.4 acc 98.7; c70: cov 73.1 acc 99.8
- s42_d3: T=0.662, raw ece 2.15pp nll 0.156 brier 0.067 -> cal ece 1.90pp
  rejection c100: cov 99.8 acc 95.9; c90: cov 94.1 acc 97.8; c80: cov 87.4 acc 98.9; c70: cov 78.5 acc 98.9
- s43_d1: T=0.389, raw ece 4.16pp nll 0.276 brier 0.104 -> cal ece 3.85pp
  rejection c100: cov 99.2 acc 94.2; c90: cov 81.3 acc 97.6; c80: cov 73.6 acc 97.4; c70: cov 64.5 acc 97.3
- s43_d2: T=0.365, raw ece 10.11pp nll 0.210 brier 0.081 -> cal ece 2.38pp
  rejection c100: cov 99.9 acc 96.5; c90: cov 95.2 acc 97.5; c80: cov 87.1 acc 98.3; c70: cov 77.7 acc 98.7
- s43_d3: T=0.469, raw ece 4.70pp nll 0.129 brier 0.053 -> cal ece 1.57pp
  rejection c100: cov 100.0 acc 97.1; c90: cov 85.6 acc 99.4; c80: cov 77.8 acc 99.7; c70: cov 67.1 acc 99.7
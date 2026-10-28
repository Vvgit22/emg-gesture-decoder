# Calibration

ECE = sum over 10 equal-width bins of (n_b/N)|acc_b-conf_b|; temperature fit on val NLL (L-BFGS).

## cross_subject
- fold0: T=1.469, raw ece 10.73pp nll 0.868 brier 0.360 -> cal ece 3.89pp
  rejection c100: cov 100.0 acc 76.2; c90: cov 91.0 acc 80.1; c80: cov 80.9 acc 83.8; c70: cov 69.9 acc 86.6
- fold1: T=1.845, raw ece 8.79pp nll 0.697 brier 0.261 -> cal ece 3.22pp
  rejection c100: cov 100.0 acc 83.4; c90: cov 90.1 acc 87.2; c80: cov 80.1 acc 90.3; c70: cov 70.8 acc 93.4
- fold2: T=2.012, raw ece 10.06pp nll 0.826 brier 0.294 -> cal ece 3.35pp
  rejection c100: cov 100.0 acc 81.5; c90: cov 89.8 acc 85.0; c80: cov 80.1 acc 88.5; c70: cov 70.2 acc 91.6
- fold3: T=1.991, raw ece 8.02pp nll 0.589 brier 0.239 -> cal ece 2.46pp
  rejection c100: cov 100.0 acc 84.6; c90: cov 94.2 acc 87.2; c80: cov 87.1 acc 89.7; c70: cov 79.0 acc 92.1
- fold4: T=1.409, raw ece 13.46pp nll 1.033 brier 0.397 -> cal ece 7.25pp
  rejection c100: cov 100.0 acc 74.2; c90: cov 91.3 acc 78.2; c80: cov 81.9 acc 82.0; c70: cov 72.3 acc 85.5
## within_subject
- s01_d1: T=0.861, raw ece 4.31pp nll 0.677 brier 0.249 -> cal ece 6.09pp
  rejection c100: cov 99.8 acc 83.8; c90: cov 86.4 acc 90.0; c80: cov 76.3 acc 91.9; c70: cov 69.2 acc 93.4
- s01_d2: T=0.629, raw ece 4.79pp nll 0.226 brier 0.101 -> cal ece 2.34pp
  rejection c100: cov 99.9 acc 93.5; c90: cov 85.0 acc 97.9; c80: cov 74.9 acc 98.4; c70: cov 66.8 acc 99.1
- s01_d3: T=0.718, raw ece 3.68pp nll 0.247 brier 0.094 -> cal ece 1.87pp
  rejection c100: cov 98.3 acc 95.6; c90: cov 79.6 acc 98.4; c80: cov 68.6 acc 98.5; c70: cov 61.3 acc 98.8
- s02_d1: T=1.245, raw ece 6.28pp nll 0.971 brier 0.384 -> cal ece 2.44pp
  rejection c100: cov 99.8 acc 73.7; c90: cov 89.7 acc 78.4; c80: cov 78.3 acc 82.5; c70: cov 64.7 acc 88.7
- s02_d2: T=0.699, raw ece 2.21pp nll 0.204 brier 0.095 -> cal ece 2.92pp
  rejection c100: cov 99.7 acc 93.7; c90: cov 90.5 acc 97.2; c80: cov 83.4 acc 98.1; c70: cov 76.1 acc 98.5
- s02_d3: T=0.841, raw ece 4.21pp nll 0.162 brier 0.081 -> cal ece 1.48pp
  rejection c100: cov 100.0 acc 94.6; c90: cov 89.6 acc 98.0; c80: cov 76.0 acc 99.7; c70: cov 65.5 acc 100.0
- s03_d1: T=0.672, raw ece 5.00pp nll 0.183 brier 0.063 -> cal ece 0.71pp
  rejection c100: cov 99.3 acc 96.7; c90: cov 81.0 acc 99.4; c80: cov 68.6 acc 99.4; c70: cov 60.4 acc 99.3
- s03_d2: T=0.601, raw ece 5.48pp nll 0.233 brier 0.099 -> cal ece 2.18pp
  rejection c100: cov 98.7 acc 94.5; c90: cov 83.4 acc 98.5; c80: cov 71.6 acc 98.6; c70: cov 57.7 acc 99.3
- s03_d3: T=0.841, raw ece 8.80pp nll 0.345 brier 0.150 -> cal ece 4.68pp
  rejection c100: cov 98.7 acc 92.5; c90: cov 87.3 acc 95.4; c80: cov 73.8 acc 96.7; c70: cov 63.1 acc 97.3
- s04_d1: T=0.623, raw ece 3.35pp nll 0.293 brier 0.124 -> cal ece 2.46pp
  rejection c100: cov 100.0 acc 92.2; c90: cov 89.0 acc 96.0; c80: cov 79.5 acc 97.7; c70: cov 68.3 acc 98.6
- s04_d2: T=0.863, raw ece 5.03pp nll 0.321 brier 0.123 -> cal ece 3.64pp
  rejection c100: cov 99.3 acc 92.0; c90: cov 88.2 acc 97.5; c80: cov 76.9 acc 99.2; c70: cov 70.3 acc 99.4
- s04_d3: T=0.770, raw ece 4.15pp nll 0.373 brier 0.153 -> cal ece 2.18pp
  rejection c100: cov 99.9 acc 90.5; c90: cov 92.3 acc 93.4; c80: cov 84.0 acc 96.6; c70: cov 71.1 acc 98.8
- s05_d1: T=0.406, raw ece 2.02pp nll 0.179 brier 0.075 -> cal ece 3.14pp
  rejection c100: cov 99.2 acc 95.7; c90: cov 85.5 acc 98.5; c80: cov 76.1 acc 99.0; c70: cov 69.2 acc 99.3
- s05_d2: T=0.577, raw ece 2.43pp nll 0.225 brier 0.108 -> cal ece 3.91pp
  rejection c100: cov 100.0 acc 92.5; c90: cov 74.3 acc 99.7; c80: cov 61.0 acc 99.9; c70: cov 49.0 acc 100.0
- s05_d3: T=0.329, raw ece 10.95pp nll 0.287 brier 0.122 -> cal ece 3.01pp
  rejection c100: cov 99.9 acc 95.0; c90: cov 90.5 acc 96.8; c80: cov 82.0 acc 97.1; c70: cov 73.4 acc 97.7
- s06_d1: T=0.976, raw ece 2.04pp nll 0.258 brier 0.130 -> cal ece 1.60pp
  rejection c100: cov 100.0 acc 90.5; c90: cov 90.5 acc 95.4; c80: cov 83.6 acc 97.2; c70: cov 76.1 acc 97.9
- s06_d2: T=0.806, raw ece 8.49pp nll 0.381 brier 0.187 -> cal ece 3.41pp
  rejection c100: cov 99.7 acc 88.5; c90: cov 88.6 acc 92.2; c80: cov 77.3 acc 95.2; c70: cov 64.2 acc 97.3
- s06_d3: T=0.676, raw ece 3.00pp nll 0.380 brier 0.151 -> cal ece 4.70pp
  rejection c100: cov 100.0 acc 90.9; c90: cov 92.7 acc 93.7; c80: cov 83.9 acc 94.5; c70: cov 75.4 acc 95.3
- s07_d1: T=0.711, raw ece 2.64pp nll 0.326 brier 0.147 -> cal ece 3.41pp
  rejection c100: cov 99.7 acc 90.4; c90: cov 86.6 acc 95.5; c80: cov 73.0 acc 97.8; c70: cov 61.5 acc 98.6
- s07_d2: T=0.457, raw ece 2.95pp nll 0.094 brier 0.041 -> cal ece 1.85pp
  rejection c100: cov 100.0 acc 97.9; c90: cov 90.1 acc 99.4; c80: cov 84.5 acc 99.7; c70: cov 79.3 acc 99.9
- s07_d3: T=0.761, raw ece 6.50pp nll 0.160 brier 0.072 -> cal ece 2.35pp
  rejection c100: cov 100.0 acc 95.9; c90: cov 89.2 acc 98.8; c80: cov 77.9 acc 99.7; c70: cov 71.2 acc 99.8
- s08_d1: T=1.338, raw ece 9.26pp nll 1.102 brier 0.384 -> cal ece 6.69pp
  rejection c100: cov 99.6 acc 75.6; c90: cov 87.0 acc 81.2; c80: cov 77.2 acc 85.7; c70: cov 68.7 acc 88.0
- s08_d2: T=0.406, raw ece 11.67pp nll 0.209 brier 0.087 -> cal ece 0.94pp
  rejection c100: cov 100.0 acc 96.1; c90: cov 91.7 acc 98.9; c80: cov 78.3 acc 99.9; c70: cov 70.8 acc 100.0
- s08_d3: T=0.961, raw ece 2.67pp nll 0.383 brier 0.138 -> cal ece 1.81pp
  rejection c100: cov 100.0 acc 91.3; c90: cov 88.1 acc 95.7; c80: cov 79.9 acc 96.5; c70: cov 68.0 acc 98.1
- s09_d1: T=0.634, raw ece 2.00pp nll 0.277 brier 0.139 -> cal ece 4.79pp
  rejection c100: cov 99.9 acc 89.8; c90: cov 72.7 acc 99.4; c80: cov 66.5 acc 99.6; c70: cov 60.8 acc 99.6
- s09_d2: T=0.851, raw ece 4.20pp nll 0.813 brier 0.308 -> cal ece 4.63pp
  rejection c100: cov 99.7 acc 80.2; c90: cov 92.8 acc 83.5; c80: cov 84.5 acc 87.1; c70: cov 75.9 acc 88.9
- s09_d3: T=0.177, raw ece 7.04pp nll 0.098 brier 0.031 -> cal ece 0.46pp
  rejection c100: cov 100.0 acc 99.2; c90: cov 87.6 acc 100.0; c80: cov 82.2 acc 100.0; c70: cov 77.1 acc 100.0
- s10_d1: T=1.070, raw ece 6.19pp nll 1.051 brier 0.361 -> cal ece 4.97pp
  rejection c100: cov 99.9 acc 73.7; c90: cov 85.5 acc 82.2; c80: cov 65.9 acc 92.9; c70: cov 54.2 acc 95.7
- s10_d2: T=0.371, raw ece 2.95pp nll 0.291 brier 0.118 -> cal ece 4.93pp
  rejection c100: cov 99.7 acc 93.0; c90: cov 79.7 acc 97.5; c80: cov 71.4 acc 98.4; c70: cov 65.5 acc 98.5
- s10_d3: T=0.078, raw ece 8.14pp nll 0.195 brier 0.082 -> cal ece 3.34pp
  rejection c100: cov 100.0 acc 96.2; c90: cov 83.3 acc 98.7; c80: cov 72.4 acc 99.1; c70: cov 62.1 acc 99.2
- s11_d1: T=0.846, raw ece 4.29pp nll 0.228 brier 0.102 -> cal ece 2.11pp
  rejection c100: cov 99.7 acc 93.4; c90: cov 88.1 acc 98.1; c80: cov 80.7 acc 99.7; c70: cov 73.0 acc 100.0
- s11_d2: T=0.594, raw ece 3.25pp nll 0.205 brier 0.092 -> cal ece 2.37pp
  rejection c100: cov 99.7 acc 94.4; c90: cov 88.9 acc 98.0; c80: cov 78.8 acc 99.3; c70: cov 69.5 acc 99.6
- s11_d3: T=0.402, raw ece 4.50pp nll 0.116 brier 0.049 -> cal ece 0.87pp
  rejection c100: cov 99.8 acc 97.4; c90: cov 87.3 acc 99.7; c80: cov 78.2 acc 100.0; c70: cov 70.5 acc 100.0
- s12_d1: T=0.622, raw ece 2.72pp nll 0.153 brier 0.057 -> cal ece 0.99pp
  rejection c100: cov 99.5 acc 97.0; c90: cov 91.4 acc 98.7; c80: cov 81.6 acc 99.3; c70: cov 73.4 acc 99.7
- s12_d2: T=0.319, raw ece 6.84pp nll 0.694 brier 0.337 -> cal ece 16.87pp
  rejection c100: cov 99.4 acc 74.3; c90: cov 85.5 acc 80.4; c80: cov 72.8 acc 85.5; c70: cov 62.9 acc 90.0
- s12_d3: T=0.515, raw ece 10.51pp nll 0.270 brier 0.120 -> cal ece 2.16pp
  rejection c100: cov 100.0 acc 93.7; c90: cov 94.2 acc 95.6; c80: cov 85.9 acc 97.0; c70: cov 71.9 acc 98.6
- s13_d1: T=0.962, raw ece 2.39pp nll 0.437 brier 0.150 -> cal ece 2.28pp
  rejection c100: cov 100.0 acc 89.8; c90: cov 89.8 acc 94.5; c80: cov 81.5 acc 97.1; c70: cov 74.0 acc 98.1
- s13_d2: T=0.371, raw ece 3.67pp nll 0.216 brier 0.117 -> cal ece 6.10pp
  rejection c100: cov 99.2 acc 92.0; c90: cov 84.5 acc 98.2; c80: cov 76.1 acc 99.6; c70: cov 66.0 acc 100.0
- s13_d3: T=0.427, raw ece 5.36pp nll 0.167 brier 0.072 -> cal ece 2.20pp
  rejection c100: cov 99.4 acc 95.9; c90: cov 87.0 acc 99.6; c80: cov 80.7 acc 99.8; c70: cov 71.9 acc 100.0
- s14_d1: T=1.569, raw ece 18.68pp nll 1.666 brier 0.566 -> cal ece 4.87pp
  rejection c100: cov 99.7 acc 64.7; c90: cov 88.0 acc 70.8; c80: cov 81.8 acc 73.1; c70: cov 73.3 acc 74.8
- s14_d2: T=0.884, raw ece 2.16pp nll 0.577 brier 0.213 -> cal ece 3.97pp
  rejection c100: cov 99.4 acc 86.6; c90: cov 88.5 acc 91.5; c80: cov 77.2 acc 94.2; c70: cov 63.9 acc 95.3
- s14_d3: T=0.636, raw ece 1.02pp nll 0.279 brier 0.095 -> cal ece 3.36pp
  rejection c100: cov 100.0 acc 93.9; c90: cov 94.8 acc 96.4; c80: cov 87.2 acc 97.6; c70: cov 70.9 acc 98.5
- s15_d1: T=0.947, raw ece 2.02pp nll 0.456 brier 0.163 -> cal ece 2.01pp
  rejection c100: cov 100.0 acc 88.8; c90: cov 90.0 acc 94.3; c80: cov 81.3 acc 96.7; c70: cov 74.2 acc 97.1
- s15_d2: T=0.768, raw ece 4.42pp nll 0.227 brier 0.100 -> cal ece 1.29pp
  rejection c100: cov 99.9 acc 93.6; c90: cov 94.3 acc 95.9; c80: cov 83.7 acc 98.7; c70: cov 72.3 acc 99.7
- s15_d3: T=0.892, raw ece 1.64pp nll 0.341 brier 0.166 -> cal ece 1.89pp
  rejection c100: cov 99.6 acc 88.3; c90: cov 88.1 acc 93.6; c80: cov 73.9 acc 97.7; c70: cov 60.7 acc 99.0
- s16_d1: T=0.486, raw ece 3.02pp nll 0.147 brier 0.076 -> cal ece 2.80pp
  rejection c100: cov 99.9 acc 95.2; c90: cov 88.5 acc 98.8; c80: cov 76.6 acc 99.8; c70: cov 67.6 acc 99.9
- s16_d2: T=0.609, raw ece 2.69pp nll 0.243 brier 0.105 -> cal ece 5.01pp
  rejection c100: cov 99.4 acc 94.1; c90: cov 89.4 acc 96.6; c80: cov 82.2 acc 97.5; c70: cov 72.0 acc 98.2
- s16_d3: T=0.358, raw ece 3.83pp nll 0.220 brier 0.101 -> cal ece 3.98pp
  rejection c100: cov 98.2 acc 93.7; c90: cov 83.1 acc 99.3; c80: cov 73.4 acc 100.0; c70: cov 65.2 acc 100.0
- s17_d1: T=0.701, raw ece 2.24pp nll 0.222 brier 0.126 -> cal ece 4.18pp
  rejection c100: cov 100.0 acc 90.7; c90: cov 89.0 acc 94.9; c80: cov 78.1 acc 98.5; c70: cov 69.5 acc 99.4
- s17_d2: T=0.352, raw ece 6.13pp nll 0.116 brier 0.041 -> cal ece 0.88pp
  rejection c100: cov 99.7 acc 98.1; c90: cov 93.8 acc 99.6; c80: cov 87.6 acc 99.8; c70: cov 79.9 acc 99.8
- s17_d3: T=0.370, raw ece 1.81pp nll 0.291 brier 0.135 -> cal ece 6.73pp
  rejection c100: cov 99.2 acc 91.4; c90: cov 87.6 acc 95.4; c80: cov 78.9 acc 96.4; c70: cov 66.4 acc 97.8
- s18_d1: T=0.120, raw ece 6.02pp nll 0.188 brier 0.088 -> cal ece 4.48pp
  rejection c100: cov 99.4 acc 94.9; c90: cov 79.5 acc 99.4; c80: cov 72.4 acc 99.8; c70: cov 64.5 acc 100.0
- s18_d2: T=0.378, raw ece 1.81pp nll 0.062 brier 0.029 -> cal ece 1.41pp
  rejection c100: cov 99.7 acc 98.1; c90: cov 89.6 acc 99.9; c80: cov 80.5 acc 100.0; c70: cov 70.9 acc 100.0
- s18_d3: T=0.531, raw ece 5.63pp nll 0.218 brier 0.089 -> cal ece 1.90pp
  rejection c100: cov 99.8 acc 95.0; c90: cov 88.8 acc 98.6; c80: cov 79.7 acc 99.3; c70: cov 70.8 acc 99.3
- s19_d1: T=0.878, raw ece 2.34pp nll 0.281 brier 0.140 -> cal ece 2.52pp
  rejection c100: cov 100.0 acc 90.5; c90: cov 93.1 acc 93.6; c80: cov 82.3 acc 97.0; c70: cov 74.1 acc 98.4
- s19_d2: T=0.465, raw ece 5.48pp nll 0.181 brier 0.082 -> cal ece 2.16pp
  rejection c100: cov 99.8 acc 94.6; c90: cov 83.9 acc 99.6; c80: cov 74.2 acc 99.9; c70: cov 66.0 acc 100.0
- s19_d3: T=0.748, raw ece 2.36pp nll 0.284 brier 0.136 -> cal ece 3.13pp
  rejection c100: cov 97.7 acc 92.8; c90: cov 83.1 acc 96.2; c80: cov 72.9 acc 97.5; c70: cov 59.4 acc 98.2
- s20_d1: T=0.473, raw ece 5.39pp nll 0.236 brier 0.097 -> cal ece 2.42pp
  rejection c100: cov 99.7 acc 94.8; c90: cov 91.9 acc 97.3; c80: cov 83.2 acc 98.2; c70: cov 74.1 acc 98.9
- s20_d2: T=0.231, raw ece 11.68pp nll 0.288 brier 0.145 -> cal ece 3.16pp
  rejection c100: cov 99.8 acc 92.9; c90: cov 88.3 acc 95.9; c80: cov 75.7 acc 98.0; c70: cov 64.5 acc 99.0
- s20_d3: T=1.024, raw ece 3.36pp nll 0.206 brier 0.113 -> cal ece 3.69pp
  rejection c100: cov 99.7 acc 92.0; c90: cov 75.3 acc 99.2; c80: cov 69.3 acc 99.8; c70: cov 61.3 acc 100.0
- s21_d1: T=0.454, raw ece 3.17pp nll 0.292 brier 0.115 -> cal ece 4.85pp
  rejection c100: cov 100.0 acc 93.6; c90: cov 95.0 acc 95.0; c80: cov 88.8 acc 96.2; c70: cov 81.1 acc 96.6
- s21_d2: T=1.147, raw ece 4.73pp nll 0.638 brier 0.248 -> cal ece 4.25pp
  rejection c100: cov 99.7 acc 83.2; c90: cov 88.2 acc 88.6; c80: cov 75.5 acc 93.9; c70: cov 65.3 acc 96.8
- s21_d3: T=0.716, raw ece 6.44pp nll 0.218 brier 0.093 -> cal ece 2.06pp
  rejection c100: cov 100.0 acc 94.1; c90: cov 90.7 acc 98.4; c80: cov 82.9 acc 99.4; c70: cov 74.9 acc 99.6
- s22_d1: T=0.472, raw ece 1.79pp nll 0.148 brier 0.069 -> cal ece 2.88pp
  rejection c100: cov 100.0 acc 95.4; c90: cov 89.8 acc 98.5; c80: cov 80.4 acc 99.8; c70: cov 71.8 acc 99.9
- s22_d2: T=0.200, raw ece 3.75pp nll 0.116 brier 0.053 -> cal ece 3.62pp
  rejection c100: cov 99.9 acc 96.0; c90: cov 87.0 acc 99.9; c80: cov 80.0 acc 100.0; c70: cov 73.6 acc 100.0
- s22_d3: T=1.051, raw ece 2.72pp nll 0.162 brier 0.068 -> cal ece 3.47pp
  rejection c100: cov 99.5 acc 95.9; c90: cov 91.9 acc 98.2; c80: cov 81.6 acc 99.4; c70: cov 72.2 acc 99.8
- s23_d1: T=0.299, raw ece 2.10pp nll 0.160 brier 0.084 -> cal ece 4.36pp
  rejection c100: cov 99.9 acc 94.1; c90: cov 84.7 acc 99.0; c80: cov 72.9 acc 99.7; c70: cov 64.6 acc 99.9
- s23_d2: T=0.280, raw ece 14.76pp nll 0.247 brier 0.102 -> cal ece 0.73pp
  rejection c100: cov 99.2 acc 96.8; c90: cov 90.0 acc 99.0; c80: cov 81.8 acc 99.8; c70: cov 68.0 acc 100.0
- s23_d3: T=0.319, raw ece 5.69pp nll 0.297 brier 0.131 -> cal ece 5.41pp
  rejection c100: cov 99.2 acc 92.1; c90: cov 78.3 acc 98.5; c80: cov 66.5 acc 99.2; c70: cov 53.9 acc 99.7
- s24_d1: T=0.366, raw ece 5.70pp nll 0.120 brier 0.038 -> cal ece 0.78pp
  rejection c100: cov 100.0 acc 98.2; c90: cov 96.2 acc 99.3; c80: cov 91.3 acc 99.4; c70: cov 83.7 acc 99.4
- s24_d2: T=0.050, raw ece 3.80pp nll 0.082 brier 0.029 -> cal ece 1.17pp
  rejection c100: cov 99.8 acc 98.9; c90: cov 94.9 acc 99.6; c80: cov 94.9 acc 99.6; c70: cov 94.9 acc 99.6
- s24_d3: T=0.050, raw ece 5.29pp nll 0.066 brier 0.016 -> cal ece 0.33pp
  rejection c100: cov 99.2 acc 99.7; c90: cov 94.6 acc 100.0; c80: cov 94.6 acc 100.0; c70: cov 94.6 acc 100.0
- s25_d1: T=0.499, raw ece 4.69pp nll 0.153 brier 0.050 -> cal ece 1.88pp
  rejection c100: cov 100.0 acc 96.7; c90: cov 95.2 acc 99.6; c80: cov 91.1 acc 99.9; c70: cov 87.2 acc 100.0
- s25_d2: T=0.472, raw ece 7.04pp nll 0.158 brier 0.059 -> cal ece 1.13pp
  rejection c100: cov 99.7 acc 97.1; c90: cov 91.5 acc 99.9; c80: cov 83.9 acc 100.0; c70: cov 76.7 acc 100.0
- s25_d3: T=0.338, raw ece 5.85pp nll 0.078 brier 0.024 -> cal ece 0.41pp
  rejection c100: cov 99.6 acc 99.5; c90: cov 87.3 acc 100.0; c80: cov 75.3 acc 100.0; c70: cov 64.1 acc 100.0
- s26_d1: T=0.337, raw ece 8.98pp nll 0.372 brier 0.177 -> cal ece 6.83pp
  rejection c100: cov 99.5 acc 88.0; c90: cov 76.7 acc 96.6; c80: cov 64.3 acc 99.2; c70: cov 51.8 acc 99.7
- s26_d2: T=0.344, raw ece 2.12pp nll 0.292 brier 0.127 -> cal ece 6.44pp
  rejection c100: cov 100.0 acc 91.8; c90: cov 89.6 acc 96.2; c80: cov 80.4 acc 98.2; c70: cov 73.5 acc 98.6
- s26_d3: T=0.451, raw ece 3.25pp nll 0.697 brier 0.288 -> cal ece 12.22pp
  rejection c100: cov 98.3 acc 82.1; c90: cov 81.2 acc 88.0; c80: cov 71.3 acc 90.6; c70: cov 63.5 acc 92.5
- s27_d1: T=0.431, raw ece 3.58pp nll 0.140 brier 0.066 -> cal ece 2.16pp
  rejection c100: cov 99.9 acc 96.1; c90: cov 88.5 acc 98.8; c80: cov 81.1 acc 99.1; c70: cov 72.9 acc 99.4
- s27_d2: T=0.795, raw ece 3.68pp nll 0.275 brier 0.114 -> cal ece 1.32pp
  rejection c100: cov 99.2 acc 93.1; c90: cov 85.9 acc 97.4; c80: cov 69.5 acc 98.3; c70: cov 62.7 acc 98.4
- s27_d3: T=0.298, raw ece 4.05pp nll 0.099 brier 0.043 -> cal ece 1.80pp
  rejection c100: cov 99.8 acc 97.4; c90: cov 92.6 acc 99.7; c80: cov 88.9 acc 99.8; c70: cov 81.4 acc 100.0
- s28_d1: T=0.186, raw ece 10.06pp nll 0.237 brier 0.085 -> cal ece 4.02pp
  rejection c100: cov 98.3 acc 95.9; c90: cov 81.5 acc 99.5; c80: cov 72.4 acc 99.8; c70: cov 66.1 acc 99.7
- s28_d2: T=0.050, raw ece 3.47pp nll 0.040 brier 0.010 -> cal ece 0.03pp
  rejection c100: cov 99.7 acc 100.0; c90: cov 96.4 acc 100.0; c80: cov 96.4 acc 100.0; c70: cov 96.4 acc 100.0
- s28_d3: T=0.356, raw ece 1.61pp nll 0.017 brier 0.003 -> cal ece 0.09pp
  rejection c100: cov 99.9 acc 100.0; c90: cov 95.4 acc 100.0; c80: cov 88.0 acc 100.0; c70: cov 80.0 acc 100.0
- s29_d1: T=0.607, raw ece 7.64pp nll 0.341 brier 0.125 -> cal ece 1.67pp
  rejection c100: cov 97.5 acc 94.4; c90: cov 83.9 acc 97.6; c80: cov 75.2 acc 99.1; c70: cov 68.5 acc 99.1
- s29_d2: T=0.379, raw ece 4.95pp nll 0.119 brier 0.042 -> cal ece 1.16pp
  rejection c100: cov 99.7 acc 97.9; c90: cov 88.9 acc 99.9; c80: cov 82.0 acc 100.0; c70: cov 66.3 acc 100.0
- s29_d3: T=0.289, raw ece 1.69pp nll 0.100 brier 0.041 -> cal ece 2.06pp
  rejection c100: cov 99.2 acc 97.8; c90: cov 89.3 acc 99.3; c80: cov 77.4 acc 99.3; c70: cov 68.8 acc 99.5
- s30_d1: T=0.314, raw ece 12.23pp nll 0.292 brier 0.137 -> cal ece 3.47pp
  rejection c100: cov 99.7 acc 91.7; c90: cov 86.4 acc 97.7; c80: cov 75.0 acc 99.3; c70: cov 55.4 acc 99.4
- s30_d2: T=0.713, raw ece 2.07pp nll 0.064 brier 0.029 -> cal ece 0.93pp
  rejection c100: cov 99.9 acc 98.0; c90: cov 89.7 acc 100.0; c80: cov 85.9 acc 100.0; c70: cov 79.7 acc 100.0
- s30_d3: T=0.358, raw ece 4.78pp nll 0.150 brier 0.055 -> cal ece 2.27pp
  rejection c100: cov 99.8 acc 96.0; c90: cov 92.9 acc 99.4; c80: cov 85.7 acc 100.0; c70: cov 76.2 acc 100.0
- s31_d1: T=0.485, raw ece 2.92pp nll 0.098 brier 0.042 -> cal ece 1.36pp
  rejection c100: cov 99.6 acc 97.7; c90: cov 85.5 acc 100.0; c80: cov 76.4 acc 100.0; c70: cov 68.3 acc 100.0
- s31_d2: T=0.889, raw ece 6.14pp nll 0.130 brier 0.049 -> cal ece 4.60pp
  rejection c100: cov 99.3 acc 98.4; c90: cov 81.1 acc 99.8; c80: cov 74.3 acc 99.8; c70: cov 68.9 acc 99.8
- s31_d3: T=0.583, raw ece 4.88pp nll 0.125 brier 0.045 -> cal ece 0.62pp
  rejection c100: cov 99.0 acc 98.1; c90: cov 87.6 acc 99.3; c80: cov 78.0 acc 99.4; c70: cov 67.2 acc 99.2
- s32_d1: T=0.857, raw ece 2.67pp nll 0.221 brier 0.088 -> cal ece 2.30pp
  rejection c100: cov 99.9 acc 95.2; c90: cov 92.9 acc 96.7; c80: cov 87.6 acc 96.5; c70: cov 81.8 acc 96.3
- s32_d2: T=0.100, raw ece 4.13pp nll 0.103 brier 0.046 -> cal ece 2.58pp
  rejection c100: cov 99.8 acc 97.2; c90: cov 82.5 acc 100.0; c80: cov 70.3 acc 100.0; c70: cov 70.3 acc 100.0
- s32_d3: T=0.582, raw ece 10.65pp nll 0.613 brier 0.220 -> cal ece 3.87pp
  rejection c100: cov 98.3 acc 85.6; c90: cov 84.4 acc 94.7; c80: cov 75.2 acc 97.4; c70: cov 66.6 acc 98.7
- s33_d1: T=0.903, raw ece 4.26pp nll 0.307 brier 0.120 -> cal ece 3.25pp
  rejection c100: cov 99.8 acc 90.8; c90: cov 88.8 acc 97.4; c80: cov 80.4 acc 99.2; c70: cov 72.1 acc 99.3
- s33_d2: T=0.749, raw ece 1.75pp nll 0.250 brier 0.095 -> cal ece 3.26pp
  rejection c100: cov 99.0 acc 94.7; c90: cov 90.5 acc 96.8; c80: cov 77.6 acc 96.8; c70: cov 65.4 acc 96.7
- s33_d3: T=1.056, raw ece 3.43pp nll 0.289 brier 0.139 -> cal ece 4.16pp
  rejection c100: cov 99.6 acc 90.3; c90: cov 86.3 acc 96.5; c80: cov 69.7 acc 99.8; c70: cov 58.0 acc 100.0
- s34_d1: T=0.838, raw ece 4.17pp nll 0.241 brier 0.138 -> cal ece 5.37pp
  rejection c100: cov 99.2 acc 90.8; c90: cov 86.7 acc 95.7; c80: cov 80.3 acc 98.3; c70: cov 75.6 acc 99.2
- s34_d2: T=0.050, raw ece 2.16pp nll 0.100 brier 0.046 -> cal ece 2.80pp
  rejection c100: cov 100.0 acc 97.2; c90: cov 92.4 acc 98.7; c80: cov 92.4 acc 98.7; c70: cov 92.4 acc 98.7
- s34_d3: T=0.336, raw ece 2.90pp nll 0.112 brier 0.053 -> cal ece 2.05pp
  rejection c100: cov 99.8 acc 96.9; c90: cov 86.6 acc 99.7; c80: cov 77.9 acc 100.0; c70: cov 70.3 acc 100.0
- s35_d1: T=0.908, raw ece 3.39pp nll 0.167 brier 0.072 -> cal ece 2.15pp
  rejection c100: cov 100.0 acc 95.7; c90: cov 91.8 acc 98.2; c80: cov 83.8 acc 99.0; c70: cov 73.4 acc 99.8
- s35_d2: T=0.631, raw ece 2.94pp nll 0.133 brier 0.058 -> cal ece 1.09pp
  rejection c100: cov 99.1 acc 96.7; c90: cov 82.2 acc 100.0; c80: cov 71.9 acc 100.0; c70: cov 63.7 acc 100.0
- s35_d3: T=0.338, raw ece 6.43pp nll 0.426 brier 0.163 -> cal ece 7.57pp
  rejection c100: cov 98.4 acc 89.8; c90: cov 81.3 acc 97.0; c80: cov 69.1 acc 99.5; c70: cov 59.8 acc 99.6
- s36_d1: T=0.511, raw ece 5.49pp nll 0.137 brier 0.054 -> cal ece 1.64pp
  rejection c100: cov 99.8 acc 96.8; c90: cov 94.9 acc 98.6; c80: cov 90.3 acc 98.9; c70: cov 84.2 acc 99.5
- s36_d2: T=0.860, raw ece 1.97pp nll 0.160 brier 0.062 -> cal ece 1.38pp
  rejection c100: cov 99.6 acc 95.6; c90: cov 85.0 acc 99.4; c80: cov 71.6 acc 99.3; c70: cov 63.0 acc 99.2
- s36_d3: T=0.050, raw ece 2.74pp nll 0.145 brier 0.068 -> cal ece 4.44pp
  rejection c100: cov 99.5 acc 95.8; c90: cov 93.3 acc 98.4; c80: cov 93.3 acc 98.4; c70: cov 93.3 acc 98.4
- s37_d1: T=0.204, raw ece 3.76pp nll 0.297 brier 0.138 -> cal ece 8.78pp
  rejection c100: cov 99.7 acc 89.4; c90: cov 79.2 acc 99.2; c80: cov 70.2 acc 99.6; c70: cov 62.4 acc 99.9
- s37_d2: T=0.412, raw ece 2.91pp nll 0.119 brier 0.056 -> cal ece 1.81pp
  rejection c100: cov 99.5 acc 96.6; c90: cov 85.0 acc 99.4; c80: cov 73.2 acc 100.0; c70: cov 64.5 acc 100.0
- s37_d3: T=0.512, raw ece 6.66pp nll 0.164 brier 0.064 -> cal ece 1.26pp
  rejection c100: cov 99.3 acc 96.3; c90: cov 92.4 acc 98.9; c80: cov 74.6 acc 100.0; c70: cov 62.4 acc 100.0
- s38_d1: T=1.319, raw ece 6.35pp nll 1.370 brier 0.438 -> cal ece 9.19pp
  rejection c100: cov 99.2 acc 68.0; c90: cov 83.4 acc 75.7; c80: cov 69.3 acc 85.0; c70: cov 57.6 acc 90.1
- s38_d2: T=0.442, raw ece 2.37pp nll 0.137 brier 0.075 -> cal ece 3.82pp
  rejection c100: cov 99.4 acc 94.4; c90: cov 89.7 acc 98.6; c80: cov 83.6 acc 99.5; c70: cov 78.2 acc 100.0
- s38_d3: T=0.530, raw ece 4.16pp nll 0.160 brier 0.067 -> cal ece 1.55pp
  rejection c100: cov 99.2 acc 95.9; c90: cov 90.8 acc 99.6; c80: cov 80.4 acc 99.9; c70: cov 70.9 acc 100.0
- s39_d1: T=0.484, raw ece 6.70pp nll 0.106 brier 0.034 -> cal ece 0.54pp
  rejection c100: cov 99.8 acc 98.8; c90: cov 93.0 acc 99.8; c80: cov 83.6 acc 99.8; c70: cov 74.5 acc 99.8
- s39_d2: T=0.923, raw ece 6.08pp nll 0.621 brier 0.265 -> cal ece 6.58pp
  rejection c100: cov 99.2 acc 83.7; c90: cov 81.6 acc 89.3; c80: cov 71.0 acc 91.5; c70: cov 61.9 acc 92.9
- s39_d3: T=0.754, raw ece 4.22pp nll 0.241 brier 0.098 -> cal ece 2.89pp
  rejection c100: cov 100.0 acc 93.3; c90: cov 93.3 acc 97.2; c80: cov 89.2 acc 99.2; c70: cov 83.1 acc 99.9
- s40_d1: T=0.677, raw ece 7.59pp nll 0.345 brier 0.151 -> cal ece 2.85pp
  rejection c100: cov 99.8 acc 92.3; c90: cov 90.7 acc 94.3; c80: cov 83.8 acc 95.4; c70: cov 73.8 acc 95.4
- s40_d2: T=0.618, raw ece 1.86pp nll 0.304 brier 0.100 -> cal ece 2.92pp
  rejection c100: cov 99.8 acc 93.8; c90: cov 85.2 acc 97.3; c80: cov 75.0 acc 98.2; c70: cov 63.5 acc 98.3
- s40_d3: T=0.702, raw ece 4.77pp nll 0.302 brier 0.122 -> cal ece 1.44pp
  rejection c100: cov 100.0 acc 92.3; c90: cov 90.9 acc 96.1; c80: cov 79.3 acc 97.8; c70: cov 68.2 acc 98.0
- s41_d1: T=0.783, raw ece 2.29pp nll 0.144 brier 0.069 -> cal ece 1.02pp
  rejection c100: cov 100.0 acc 95.8; c90: cov 92.9 acc 97.7; c80: cov 86.1 acc 98.5; c70: cov 76.1 acc 98.9
- s41_d2: T=0.966, raw ece 3.88pp nll 0.148 brier 0.072 -> cal ece 3.80pp
  rejection c100: cov 99.7 acc 94.7; c90: cov 89.6 acc 98.8; c80: cov 80.1 acc 100.0; c70: cov 69.7 acc 100.0
- s41_d3: T=0.754, raw ece 4.53pp nll 0.231 brier 0.099 -> cal ece 1.95pp
  rejection c100: cov 99.7 acc 93.2; c90: cov 87.5 acc 98.9; c80: cov 72.1 acc 99.9; c70: cov 61.4 acc 99.9
- s42_d1: T=0.503, raw ece 8.14pp nll 0.275 brier 0.112 -> cal ece 2.40pp
  rejection c100: cov 98.6 acc 93.5; c90: cov 86.9 acc 99.2; c80: cov 79.9 acc 99.5; c70: cov 73.8 acc 99.9
- s42_d2: T=0.620, raw ece 3.96pp nll 0.229 brier 0.090 -> cal ece 2.07pp
  rejection c100: cov 100.0 acc 93.4; c90: cov 90.5 acc 99.3; c80: cov 81.3 acc 100.0; c70: cov 69.2 acc 100.0
- s42_d3: T=0.569, raw ece 3.26pp nll 0.149 brier 0.064 -> cal ece 1.92pp
  rejection c100: cov 99.8 acc 96.6; c90: cov 93.3 acc 98.2; c80: cov 87.0 acc 98.8; c70: cov 77.6 acc 99.1
- s43_d1: T=0.355, raw ece 4.36pp nll 0.304 brier 0.102 -> cal ece 3.78pp
  rejection c100: cov 99.2 acc 95.0; c90: cov 84.7 acc 97.0; c80: cov 72.6 acc 97.2; c70: cov 62.3 acc 96.9
- s43_d2: T=0.366, raw ece 5.32pp nll 0.169 brier 0.069 -> cal ece 2.71pp
  rejection c100: cov 99.9 acc 96.2; c90: cov 93.5 acc 97.9; c80: cov 84.4 acc 98.8; c70: cov 74.5 acc 99.3
- s43_d3: T=0.429, raw ece 2.41pp nll 0.094 brier 0.043 -> cal ece 1.90pp
  rejection c100: cov 100.0 acc 97.2; c90: cov 85.0 acc 99.8; c80: cov 77.6 acc 99.8; c70: cov 69.7 acc 100.0
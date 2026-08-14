# PoC 실행 요약 — CTU-UHB 5분 세그먼트 Cat28 인자 추출

- 레코드 552건 → 5분 세그먼트 6,400개 (결측 >30% 제외 1098개)
- Cat28 인자 28개 × 6,400세그먼트 계산 시간: **31.0초** (단일 CPU 코어)
- 세그먼트당 평균 4.8 ms

## 인자별 요약 (pH 라벨 기준 레코드 단위 평균의 방향성 점검)

| 인자 | 정상(pH≥7.20) | 비정상(pH<7.20) |
|---|---:|---:|
| fhr_mean | 135.132 | 135.449 |
| fhr_min | 116.426 | 114.381 |
| fhr_max | 149.383 | 151.728 |
| fhr_sd | 8.383 | 9.766 |
| stv | 0.088 | 0.101 |
| brady_frac | 0.078 | 0.104 |
| tachy_frac | 0.070 | 0.102 |
| n_decel | 0.520 | 0.623 |
| decel_max_depth | 13.734 | 16.595 |
| decel_time_frac | 0.067 | 0.086 |
| n_accel | 0.204 | 0.268 |
| toco_mean | 23.164 | 23.546 |
| toco_max | 45.591 | 46.743 |
| toco_sd | 9.982 | 10.522 |
| n_contractions | 1.173 | 1.144 |
| ft_corr0 | -0.099 | -0.073 |
| ft_corr_min | -0.471 | -0.464 |
| ft_corr_min_lag_s | 5.027 | 6.073 |
| figo_baseline | 136.153 | 136.698 |
| figo_baseline_var | 1.510 | 1.654 |
| n_early_decel | 0.395 | 0.481 |
| n_late_decel | 0.126 | 0.142 |
| n_variable_decel | 0.456 | 0.522 |
| n_severe_decel | 0.000 | 0.000 |
| n_prolonged_decel | 0.000 | 0.000 |
| hist_width | 32.957 | 37.347 |
| hist_median | 136.086 | 136.520 |
| hist_mode | 137.504 | 137.900 |

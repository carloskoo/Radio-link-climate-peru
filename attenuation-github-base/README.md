# Attenuation Modeling Module

This module contains the scripts, input data, processed results, and figures used to compute atmospheric attenuation in point-to-point radio links located in representative regions of Peru.

The workflow applies ITU-R propagation models to quantify the contribution of gaseous attenuation, rain attenuation, and total atmospheric attenuation as a function of frequency, region, and climatological severity.

---

## Objective

The objective of this module is to reproduce the atmospheric attenuation analysis used in the thesis:

**Evaluation of the influence of climatological factors on point-to-point radio link attenuation in regions of Peru.**

The analysis considers representative coastal, Andean, and rainforest environments and evaluates attenuation over microwave frequency bands relevant to terrestrial radio-link planning.

---

## Methodological basis

The attenuation modeling follows two main ITU-R recommendations:

- **ITU-R P.676-13**: attenuation by atmospheric gases.
- **ITU-R P.838-3**: specific attenuation model for rain.

The total atmospheric attenuation is computed as:

```text
γ_total = γ_gas + γ_rain
Donde: 
γ_gas   = gaseous attenuation from ITU-R P.676
γ_rain  = rain attenuation from ITU-R P.838
γ_total = total atmospheric attenuation

attenuation-github-base/
├── data/
│   ├── input/
│   │   ├── phase5_gaseous_results.csv
│   │   ├── phase6_rain_rates.csv
│   │   ├── phase6_k_alpha_p838.csv
│   │   └── phase6_rain_results.csv
│   │
│   └── processed/
│       ├── phase5_gaseous_summary_p50_p95.csv
│       ├── phase7_total_attenuation_results.csv
│       └── attenuation_mechanisms_combined.csv
│
├── figures/
│   ├── fig_gaseous_attenuation_q1.png
│   ├── fig_rain_attenuation_q1.png
│   ├── fig_total_attenuation_q1.png
│   └── fig_comparative_mechanisms_q1.png
│
├── metadata/
│   └── f_GHz_pol_k_alpha.csv
│
├── scripts/
│   ├── README.md
│   ├── phase5_gaseous_attenuation_clean.py
│   ├── phase6_rain_attenuation_clean.py
│   ├── phase7_total_attenuation_clean.py
│   └── phase8_rain_dominance_ratio.py
│
└── README.md

## Reproducibility

All attenuation results reported in the thesis can be reproduced directly from the datasets and scripts available in this repository.

Workflow:

1. Gaseous attenuation (ITU-R P.676)
2. Rain attenuation (ITU-R P.838)
3. Total attenuation
4. Rain dominance analysis

The repository contains all intermediate datasets, metadata files, processing scripts, and publication-quality figures used in the study.

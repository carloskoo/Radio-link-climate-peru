# Atmospheric Attenuation Modeling for Microwave Radio Links in Peru

This repository contains the datasets, scripts, and figures used to analyze the impact of atmospheric attenuation on microwave radio links across representative Peruvian regions (coastal, Andean, and rainforest environments).

The study evaluates the contribution of **gaseous attenuation**, **rain attenuation**, and their combined effect on total propagation losses, following ITU-R recommendations.

---

## 📌 Objectives

- Quantify atmospheric attenuation as a function of frequency
- Evaluate regional variability under different climatological conditions (p50 and p95)
- Identify the transition frequency where rain attenuation becomes dominant
- Provide engineering insights for climate-aware microwave link design

---

## 📂 Repository Structure

```text
attenuation-github-base/
│
├── data/
│   ├── input/              # Raw input datasets
│   ├── processed/          # Computed results
│
├── figures/                # Publication-ready figures
│
├── metadata/               # ITU-R coefficients and auxiliary data
│
├── scripts/                # Python scripts for modeling
│
└── README.md

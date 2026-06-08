# Attenuation modeling module

This module contains the scripts, input data, processed results, and figures used to compute atmospheric attenuation in point-to-point radio links.

The workflow includes:

1. Gaseous attenuation using ITU-R P.676.
2. Rain attenuation using ITU-R P.838.
3. Total atmospheric attenuation as:

γ_total = γ_gas + γ_rain

## Structure

```text
attenuation-github-base/
├── data/
│   ├── input/
│   └── processed/
├── figures/
├── metadata/
├── scripts/
└── README.md

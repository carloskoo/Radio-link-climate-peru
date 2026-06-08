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

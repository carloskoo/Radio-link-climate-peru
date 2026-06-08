# Scripts directory

This directory contains the Python scripts used to compute atmospheric attenuation for representative radio links in Peru.

---

## Workflow

The attenuation analysis follows four sequential phases:

1. Gaseous attenuation (ITU-R P.676)
2. Rain attenuation (ITU-R P.838)
3. Total atmospheric attenuation
4. Rain dominance analysis

---

## phase5_gaseous_attenuation_clean.py

Purpose:

Compute gaseous attenuation statistics from previously generated attenuation datasets.

Input:

```text
data/input/phase5_gaseous_results.csv
```

Outputs:

```text
data/processed/phase5_gaseous_summary_p50_p95.csv
figures/fig_gaseous_attenuation_q1.png
```

Model:

```text
ITU-R P.676-13
```

---

## phase6_rain_attenuation_clean.py

Purpose:

Compute rain attenuation using rain-rate information and ITU-R P.838 coefficients.

Inputs:

```text
data/input/phase6_rain_rates.csv
data/input/phase6_k_alpha_p838.csv
```

Outputs:

```text
data/input/phase6_rain_results.csv
figures/fig_rain_attenuation_q1.png
```

Model:

```text
ITU-R P.838-3
```

---

## phase7_total_attenuation_clean.py

Purpose:

Combine gaseous attenuation and rain attenuation to estimate total atmospheric attenuation.

Inputs:

```text
data/processed/phase5_gaseous_summary_p50_p95.csv
data/input/phase6_rain_results.csv
```

Outputs:

```text
data/processed/phase7_total_attenuation_results.csv
data/processed/attenuation_mechanisms_combined.csv
figures/fig_total_attenuation_q1.png
```

Equation:

```text
γ_total = γ_gas + γ_rain
```

---

## phase8_rain_dominance_ratio.py

Purpose:

Evaluate the relative contribution of rain attenuation to total atmospheric attenuation.

Input:

```text
data/processed/phase7_total_attenuation_results.csv
```

Output:

```text
figures/fig_rain_dominance_ratio_q1.png
```

---

## Recommended execution order

```bash
python scripts/phase5_gaseous_attenuation_clean.py

python scripts/phase6_rain_attenuation_clean.py

python scripts/phase7_total_attenuation_clean.py

python scripts/phase8_rain_dominance_ratio.py
```

---

## References

- ITU-R P.676-13: Attenuation by atmospheric gases.
- ITU-R P.838-3: Specific attenuation model for rain.
- ERA5 / ERA5-Land (Copernicus Climate Data Store).
- NASA POWER.

"""
Script: compute_metrics_all.py

Description:
Computes consistency metrics between NASA POWER and ERA5/ERA5-Land
from paired daily observations stored in daily_pairs.csv.

Inputs:
- daily_pairs.csv

Outputs:
- metrics_all.csv

Expected columns in daily_pairs.csv:
- site
- var
- power
- era5

Computed metrics:
- n
- r
- bias
- RMSE
- MAE
- slope
- intercept
- mean_POWER
- mean_ERA5_used

Notes:
- Surface pressure (SP) is bias-corrected before computing metrics.
- Precipitation (PRECI / PRECIP) is evaluated in log10 scale.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================
# CONFIGURATION
# =============================
INPUT_FILE = "daily_pairs.csv"
OUTPUT_FILE = "metrics_all.csv"

BIAS_CORR_VARS = {"SP"}
LOG10_VARS = {"PRECI", "PRECIP"}


# =============================
# HELPER FUNCTIONS
# =============================
def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - x) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(y - x)))


def linear_fit(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return 1.0, 0.0
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def prepare_xy(df_var: pd.DataFrame, var: str):
    x = df_var["power"].astype(float).to_numpy()
    y = df_var["era5"].astype(float).to_numpy()

    # Log10 transformation for precipitation
    if var.upper() in LOG10_VARS:
        x = np.log10(np.maximum(x, 0) + 1e-3)
        y = np.log10(np.maximum(y, 0) + 1e-3)

    # Bias correction for SP
    if var.upper() in BIAS_CORR_VARS:
        bias_applied = float(np.mean(y - x))
        y = y - bias_applied
    else:
        bias_applied = 0.0

    return x, y, bias_applied


# =============================
# MAIN
# =============================
def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)

    required_cols = {"site", "var", "power", "era5"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in daily_pairs.csv: {missing}")

    rows = []

    for (site, var), g in df.groupby(["site", "var"]):
        x, y, bias_applied = prepare_xy(g, var)

        r = pearson_r(x, y)
        bias = float(np.mean(y - x))
        e_rmse = rmse(x, y)
        e_mae = mae(x, y)
        slope, intercept = linear_fit(x, y)

        rows.append({
            "site": site,
            "variable": var.upper(),
            "n": int(len(g)),
            "r": r,
            "bias": bias,
            "RMSE": e_rmse,
            "MAE": e_mae,
            "slope": slope,
            "intercept": intercept,
            "mean_POWER": float(np.mean(x)),
            "mean_ERA5_used": float(np.mean(y)),
            "bias_applied": bias_applied,
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_path, index=False, float_format="%.6f")

    print("✔ metrics_all.csv generated successfully")
    print(f" - {output_path.resolve()}")


if __name__ == "__main__":
    main()

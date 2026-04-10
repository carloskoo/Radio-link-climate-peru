"""
Script: plot_summary_metrics.py

Description:
Generates summary figures (mean correlation and mean RMSE)
between ERA5/ERA5-Land and NASA POWER datasets using metrics_all.csv.

Inputs:
- metrics_all.csv

Outputs:
- figures/mean_correlation.png
- figures/mean_rmse.png

Expected columns in metrics_all.csv:
- variable
- r
- RMSE

Variables:
SP = Surface Pressure
RH = Relative Humidity
PRECIP = Precipitation
T = Temperature
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# CONFIGURATION
# =============================
INPUT_FILE = "metrics_all.csv"
OUTPUT_DIR = "figures"

# Logical order for paper-style presentation
ORDER = ["SP", "PRECIP", "RH", "T"]


# =============================
# MAIN
# =============================
def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)

    # Normalize variable naming
    if "variable" not in df.columns:
        raise ValueError("The file metrics_all.csv must contain a 'variable' column.")
    if "r" not in df.columns:
        raise ValueError("The file metrics_all.csv must contain an 'r' column.")
    if "RMSE" not in df.columns:
        raise ValueError("The file metrics_all.csv must contain an 'RMSE' column.")

    df["variable"] = df["variable"].astype(str).str.upper()

    # Keep only expected variables
    df = df[df["variable"].isin(ORDER)].copy()

    # ---------- Mean correlation ----------
    r_mean = df.groupby("variable")["r"].mean().reindex(ORDER)

    plt.style.use("default")
    plt.figure(figsize=(6, 4))
    plt.bar(r_mean.index, r_mean.values)

    plt.ylabel("r (average)")
    plt.title("Mean correlation (ERA5 vs NASA POWER)")
    plt.ylim(0, 1)

    for i, v in enumerate(r_mean.values):
        if pd.notna(v):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path / "mean_correlation.png", dpi=300)
    plt.close()

    # ---------- Mean RMSE ----------
    rmse_mean = df.groupby("variable")["RMSE"].mean().reindex(ORDER)

    plt.figure(figsize=(6, 4))
    plt.bar(rmse_mean.index, rmse_mean.values)

    plt.ylabel("RMSE (average)")
    plt.title("Mean RMSE (ERA5 vs NASA POWER)")

    y_offset = max(rmse_mean.values) * 0.03 if rmse_mean.notna().any() else 0.2
    for i, v in enumerate(rmse_mean.values):
        if pd.notna(v):
            plt.text(i, v + y_offset, f"{v:.2f}", ha="center")

    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path / "mean_rmse.png", dpi=300)
    plt.close()

    print("✔ Figures generated successfully:")
    print(f" - {(output_path / 'mean_correlation.png').resolve()}")
    print(f" - {(output_path / 'mean_rmse.png').resolve()}")


if __name__ == "__main__":
    main()

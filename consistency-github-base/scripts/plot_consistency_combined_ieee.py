"""
Script: plot_consistency_combined_ieee.py

Description:
Generates a publication-style combined consistency figure between
ERA5/ERA5-Land and NASA POWER using paired daily observations.

Figure layout:
- Top row: Relative Humidity (RH)
- Bottom row: Precipitation (PRECI, log10 scale)
- Columns: Coastal, Andean, Rainforest

Inputs:
- daily_pairs.csv

Outputs:
- figures/consistency_combined_ieee.png

Expected columns in daily_pairs.csv:
- site
- var
- power
- era5

Representative sites used:
- Coastal    -> L09-A
- Andean     -> L03-A
- Rainforest -> L10-A
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================
# CONFIGURATION
# =============================
INPUT_FILE = "daily_pairs.csv"
OUTPUT_DIR = "figures"
OUTPUT_FILE = "consistency_combined_ieee.png"

REPRESENTATIVE_SITES = {
    "Coastal": "L09-A",
    "Andean": "L03-A",
    "Rainforest": "L10-A",
}

PLOT_STYLE = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
}


# =============================
# HELPER FUNCTIONS
# =============================
def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - x) ** 2)))


def linear_fit(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return 1.0, 0.0
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def prepare_xy(df_var: pd.DataFrame, log10: bool = False):
    x = df_var["power"].astype(float).to_numpy()
    y = df_var["era5"].astype(float).to_numpy()

    if log10:
        x = np.log10(np.maximum(x, 0) + 1e-3)
        y = np.log10(np.maximum(y, 0) + 1e-3)

    return x, y


def plot_panel(ax, df_var, site_label, y_label=None, log10=False, show_xlabel=False):
    x, y = prepare_xy(df_var, log10=log10)

    r = pearson_r(x, y)
    e = rmse(x, y)
    m, b = linear_fit(x, y)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad

    # Points
    ax.scatter(x, y, s=28)

    # 1:1 line
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)

    # Regression
    xx = np.linspace(lo, hi, 100)
    ax.plot(xx, m * xx + b, color="red", lw=1.0)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(site_label)

    ax.text(
        0.03,
        0.95,
        f"r={r:.2f}\nRMSE={e:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    if y_label is not None:
        ax.set_ylabel(y_label)

    if show_xlabel:
        ax.set_xlabel("POWER")


# =============================
# MAIN
# =============================
def main():
    plt.rcParams.update(PLOT_STYLE)

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)

    required_cols = {"site", "var", "power", "era5"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in daily_pairs.csv: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)

    # -------- Top row: RH --------
    for j, (region, site) in enumerate(REPRESENTATIVE_SITES.items()):
        d = df[(df["site"] == site) & (df["var"].str.upper() == "RH")].copy()

        plot_panel(
            ax=axes[0, j],
            df_var=d,
            site_label=f"{site} ({region})",
            y_label="Relative Humidity (%)" if j == 0 else None,
            log10=False,
            show_xlabel=False,
        )

    # -------- Bottom row: PRECI --------
    for j, (region, site) in enumerate(REPRESENTATIVE_SITES.items()):
        d = df[(df["site"] == site) & (df["var"].str.upper().isin(["PRECI", "PRECIP"]))].copy()

        plot_panel(
            ax=axes[1, j],
            df_var=d,
            site_label=f"{site} ({region})",
            y_label="Precipitation (log$_{10}$ mm/day)" if j == 0 else None,
            log10=True,
            show_xlabel=True,
        )

    # Global legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=5, label="Observed"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.0, label="1:1"),
        Line2D([0], [0], color="red", lw=1.0, label="Regression"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)

    out_file = output_path / OUTPUT_FILE
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("✔ Figure generated successfully:")
    print(f" - {out_file.resolve()}")


if __name__ == "__main__":
    main()

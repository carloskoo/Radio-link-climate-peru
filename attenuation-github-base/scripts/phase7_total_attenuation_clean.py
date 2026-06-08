
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REGION_ORDER = ["costa", "sierra", "selva"]
REGION_LABELS = {
    "costa": "Coastal",
    "sierra": "Andean",
    "selva": "Rainforest",
}
REGION_COLORS = {
    "costa": "tab:blue",
    "sierra": "tab:red",
    "selva": "tab:purple",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine gaseous and rain attenuation results to compute total atmospheric attenuation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing phase5_gaseous_results.csv and phase6_rain_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where outputs will be saved",
    )
    parser.add_argument(
        "--gas-file",
        type=str,
        default="phase5_gaseous_results.csv",
        help="CSV file with gaseous attenuation results",
    )
    parser.add_argument(
        "--rain-file",
        type=str,
        default="phase6_rain_results.csv",
        help="CSV file with rain attenuation results",
    )
    return parser.parse_args()


def load_gaseous_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    freq_col = "f_ghz" if "f_ghz" in df.columns else "f_GHz"
    required = {"region", "percentile", freq_col, "gamma_dB_per_km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gaseous-results CSV: {missing}")

    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")
    df["f_GHz"] = pd.to_numeric(df[freq_col], errors="coerce")
    df["gamma_dB_per_km"] = pd.to_numeric(df["gamma_dB_per_km"], errors="coerce")

    df = df.dropna(subset=["region", "percentile", "f_GHz", "gamma_dB_per_km"]).copy()
    df["percentile"] = df["percentile"].astype(int)

    # Aggregate repeated samples per region, percentile, and frequency.
    df = (
        df.groupby(["region", "percentile", "f_GHz"], as_index=False)["gamma_dB_per_km"]
        .mean()
    )
    return df


def load_rain_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = {"region", "percentile", "f_GHz", "gammaR_dB_per_km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in rain-results CSV: {missing}")

    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")
    df["f_GHz"] = pd.to_numeric(df["f_GHz"], errors="coerce")
    df["gammaR_dB_per_km"] = pd.to_numeric(df["gammaR_dB_per_km"], errors="coerce")

    df = df.dropna(subset=["region", "percentile", "f_GHz", "gammaR_dB_per_km"]).copy()
    df["percentile"] = df["percentile"].astype(int)

    df = (
        df.groupby(["region", "percentile", "f_GHz"], as_index=False)["gammaR_dB_per_km"]
        .mean()
    )
    return df


def compute_total_attenuation(gaseous_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    merged = gaseous_df.merge(
        rain_df,
        on=["region", "percentile", "f_GHz"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping region/percentile/frequency combinations were found.")

    merged["gamma_total_dB_per_km"] = (
        merged["gamma_dB_per_km"] + merged["gammaR_dB_per_km"]
    )
    return merged.sort_values(["region", "percentile", "f_GHz"])


def plot_total_attenuation(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)

    for region in REGION_ORDER:
        if region not in df["region"].unique():
            continue

        for percentile in [50, 95]:
            sub = df[
                (df["region"] == region) & (df["percentile"] == percentile)
            ].sort_values("f_GHz")
            if sub.empty:
                continue

            linestyle = "-" if percentile == 50 else "--"
            label = f"{REGION_LABELS[region]} p{percentile}"
            ax.plot(
                sub["f_GHz"],
                sub["gamma_total_dB_per_km"],
                color=REGION_COLORS[region],
                linestyle=linestyle,
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$\gamma_{\mathrm{total}}$ (dB/km)")
    ax.set_title("Total atmospheric attenuation")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(ncol=2, frameon=True)

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gaseous_path = input_dir / args.gas_file
    rain_path = input_dir / args.rain_file

    gaseous_df = load_gaseous_results(gaseous_path)
    rain_df = load_rain_results(rain_path)
    total_df = compute_total_attenuation(gaseous_df, rain_df)

    result_csv = output_dir / "phase7_total_attenuation_results.csv"
    figure_png = output_dir / "fig_7_1_total_attenuation.png"

    total_df.to_csv(result_csv, index=False)
    plot_total_attenuation(total_df, figure_png)

    print("[OK] Outputs generated:")
    print(f" - {result_csv}")
    print(f" - {figure_png}")


if __name__ == "__main__":
    main()

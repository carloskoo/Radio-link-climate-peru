from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REGION_ORDER = ["coast", "highlands", "rainforest"]
REGION_LABELS = {
    "coast": "Coastal",
    "highlands": "Andean",
    "rainforest": "Rainforest",
    "costa": "Coastal",
    "sierra": "Andean",
    "selva": "Rainforest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot gaseous attenuation from phase5_gaseous_results.csv."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("phase5_gaseous_results.csv"),
        help="CSV file containing gaseous attenuation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where outputs will be saved",
    )
    return parser.parse_args()


def normalize_region(region: str) -> str:
    mapping = {
        "costa": "coast",
        "sierra": "highlands",
        "selva": "rainforest",
        "coast": "coast",
        "highlands": "highlands",
        "rainforest": "rainforest",
    }
    return mapping.get(str(region).strip().lower(), str(region).strip().lower())


def load_and_aggregate(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"region", "percentile", "f_ghz", "gamma_db_per_km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gaseous-results CSV: {missing}")

    df["region"] = df["region"].apply(normalize_region)
    df["percentile"] = pd.to_numeric(df["percentile"], errors="coerce")
    df["f_ghz"] = pd.to_numeric(df["f_ghz"], errors="coerce")
    df["gamma_db_per_km"] = pd.to_numeric(df["gamma_db_per_km"], errors="coerce")

    df = df.dropna(subset=["region", "percentile", "f_ghz", "gamma_db_per_km"]).copy()

    summary = (
        df[df["percentile"].isin([50, 95])]
        .groupby(["region", "percentile", "f_ghz"], as_index=False)["gamma_db_per_km"]
        .mean()
        .sort_values(["region", "percentile", "f_ghz"])
    )

    if summary.empty:
        raise ValueError("No aggregated gaseous attenuation results were generated")

    return summary


def ordered_regions(regions: list[str]) -> list[str]:
    ordered = [r for r in REGION_ORDER if r in regions]
    ordered += [r for r in regions if r not in ordered]
    return ordered


def plot_gaseous_attenuation(summary: pd.DataFrame, out_path: Path) -> None:
    regions = ordered_regions(list(summary["region"].unique()))

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)

    for region in regions:
        for percentile in [50, 95]:
            sub = summary[
                (summary["region"] == region) & (summary["percentile"] == percentile)
            ].sort_values("f_ghz")
            if sub.empty:
                continue

            linestyle = "-" if percentile == 50 else "--"
            label = f"{REGION_LABELS.get(region, region.title())} p{percentile}"
            ax.plot(
                sub["f_ghz"],
                sub["gamma_db_per_km"],
                linestyle=linestyle,
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$\gamma(f)$ (dB/km)")
    ax.set_title("Gaseous attenuation")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(ncol=2, frameon=True)

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_and_aggregate(args.input_csv)
    summary_path = args.output_dir / "phase5_gaseous_summary_p50_p95.csv"
    fig_path = args.output_dir / "fig_5_1_gaseous_attenuation.png"

    summary.to_csv(summary_path, index=False)
    plot_gaseous_attenuation(summary, fig_path)

    print("[OK] Outputs generated:")
    print(f" - {summary_path}")
    print(f" - {fig_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency_pipeline_github.py

Pipeline reproducible para evaluar la consistencia entre NASA POWER y ERA5/ERA5-Land
a partir de series horarias en formato largo. El script:
    1) lee CSVs horarios POWER y ERA5,
    2) agrega a escala diaria con control de cobertura,
    3) construye daily_pairs.csv,
    4) calcula métricas de consistencia por sitio y variable,
    5) genera rankings,
    6) crea figuras de dispersión por variable y región,
    7) crea una figura combinada (RH + PRECI) tipo paper.

Formato largo esperado en los CSVs de entrada:
    datetime, site, var, value

Variables recomendadas:
    SP     -> presión superficial (agregado diario: mean)
    T      -> temperatura (agregado diario: mean)
    RH     -> humedad relativa (agregado diario: mean)
    PRECI  -> precipitación (agregado diario: sum)

Ejemplo de uso:
python consistency_pipeline_github.py \
    --power power_hourly.csv \
    --era5 era5_hourly.csv \
    --locations sites_locations_AB.csv \
    --regions sites_regions_AB.csv \
    --outdir report_out \
    --year 2024

Autor: OpenAI / generado para integración en GitHub
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------- Configuración general -----------------------------

AGG_MAP_DEFAULT = {
    "SP": "mean",
    "T": "mean",
    "RH": "mean",
    "PRECI": "sum",
}

UNITS_MAP = {
    "SP": "hPa",
    "T": "°C",
    "RH": "%",
    "PRECI": "mm/day",
}

BIAS_CORR_VARS = {"SP"}     # Se recomienda corrección de sesgo para SP
LOG10_VARS = {"PRECI"}      # Precipitación en escala log10 para visualización

REPRESENTATIVE_BY_REGION_DEFAULT = {
    "Coastal": "L09-A",
    "Andean": "L03-A",
    "Rainforest": "L10-A",
}

PLOT_STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}

# ----------------------------- Funciones utilitarias -----------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 1.0, 0.0
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# ----------------------------- Lectura y preparación -----------------------------

def read_hourly_csv(path: str, datetime_col: str, site_col: str, var_col: str, value_col: str) -> pd.DataFrame:
    """Lee un CSV horario en formato largo y normaliza columnas."""
    df = pd.read_csv(path)
    rename_map = {
        datetime_col: "datetime",
        site_col: "site",
        var_col: "var",
        value_col: "value",
    }
    df = df.rename(columns=rename_map)
    required = ["datetime", "site", "var", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["site"] = df["site"].astype(str)
    df["var"] = df["var"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["datetime", "site", "var", "value"]).copy()
    return df[["datetime", "site", "var", "value"]]

def aggregate_daily(df: pd.DataFrame, agg_map: Dict[str, str]) -> pd.DataFrame:
    """Agrega de horario a diario con método por variable."""
    df = df.copy()
    df["date"] = df["datetime"].dt.floor("D")

    rows = []
    for (site, var, date), g in df.groupby(["site", "var", "date"]):
        method = agg_map.get(var, "mean")
        if method == "sum":
            val = g["value"].sum()
        elif method == "max":
            val = g["value"].max()
        elif method == "min":
            val = g["value"].min()
        else:
            val = g["value"].mean()

        n_hours = int(g["value"].count())
        rows.append({
            "site": site,
            "var": var,
            "date": date,
            "value": float(val),
            "n_hours": n_hours,
            "coverage": n_hours / 24.0,
        })

    out = pd.DataFrame(rows)
    return out

def build_daily_pairs(power_daily: pd.DataFrame, era5_daily: pd.DataFrame, min_coverage: float = 0.8) -> pd.DataFrame:
    """Empareja POWER y ERA5 a escala diaria con filtro de cobertura."""
    p = power_daily.rename(columns={
        "value": "power",
        "n_hours": "n_hours_power",
        "coverage": "coverage_power",
    })
    e = era5_daily.rename(columns={
        "value": "era5",
        "n_hours": "n_hours_era5",
        "coverage": "coverage_era5",
    })
    pairs = pd.merge(p, e, on=["site", "var", "date"], how="inner")
    pairs = pairs[
        (pairs["coverage_power"] >= min_coverage) &
        (pairs["coverage_era5"] >= min_coverage)
    ].copy()
    pairs = pairs.rename(columns={"date": "date_utc"})
    return pairs

def attach_optional_mapping(df: pd.DataFrame, mapping_path: str | None, on: str = "site") -> pd.DataFrame:
    """Adjunta tablas auxiliares (locations, regions)."""
    if not mapping_path:
        return df
    m = pd.read_csv(mapping_path)
    if on not in m.columns:
        return df
    return df.merge(m, on=on, how="left")

# ----------------------------- Métricas de consistencia -----------------------------

def prepare_xy(d: pd.DataFrame, var: str, bias_corr: bool = False, use_log10: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Prepara pares x/y para métricas y gráficos."""
    x = d["power"].astype(float).to_numpy()
    y = d["era5"].astype(float).to_numpy()

    if use_log10:
        x = np.log10(np.maximum(x, 0) + 1e-3)
        y = np.log10(np.maximum(y, 0) + 1e-3)

    bias_applied = 0.0
    if bias_corr:
        bias_applied = float(np.mean(y - x))
        y = y - bias_applied

    return x, y, bias_applied

def compute_metrics_for_group(d: pd.DataFrame, var: str, bias_corr: bool = False, use_log10: bool = False) -> Dict[str, float]:
    """Calcula métricas para un sitio-variable."""
    x, y, bias_applied = prepare_xy(d, var=var, bias_corr=bias_corr, use_log10=use_log10)
    m, b = linear_fit(x, y)

    return {
        "n": int(len(d)),
        "r": pearson_r(x, y),
        "bias": float(np.mean(y - x)),
        "RMSE": rmse(y, x),
        "MAE": mae(y, x),
        "slope": m,
        "intercept": b,
        "mean_POWER": float(np.mean(x)),
        "mean_ERA5_used": float(np.mean(y)),
        "bias_applied": bias_applied,
    }

def build_metrics_all(daily_pairs: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas para todas las variables y sitios."""
    rows = []
    for (site, var), g in daily_pairs.groupby(["site", "var"]):
        cfg_bias = var in BIAS_CORR_VARS
        cfg_log = var in LOG10_VARS
        metrics = compute_metrics_for_group(g, var=var, bias_corr=cfg_bias, use_log10=cfg_log)
        row = {"site": site, "var": var}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)

def write_rankings(metrics_all: pd.DataFrame, outdir: Path) -> None:
    """Escribe resumen rápido en markdown con top-3 por r y RMSE."""
    tops_r = []
    tops_rmse = []
    for var in metrics_all["var"].dropna().unique():
        d = metrics_all[metrics_all["var"] == var].copy()
        tops_r.append(d.sort_values("r", ascending=False).head(3))
        tops_rmse.append(d.sort_values("RMSE", ascending=True).head(3))

    md = ["# Resumen rápido", "", "## Top-3 por correlación (r)"]
    md.append(pd.concat(tops_r).to_markdown(index=False))
    md.extend(["", "## Top-3 por RMSE"])
    md.append(pd.concat(tops_rmse).to_markdown(index=False))

    (outdir / "summary_rankings.md").write_text("\n".join(md), encoding="utf-8")

# ----------------------------- Selección de sitios -----------------------------

def representative_sites_from_regions(regions_df: pd.DataFrame) -> Dict[str, str]:
    """
    Devuelve un diccionario region -> site usando nombres esperados.
    Si la tabla de regiones no sirve, cae al default.
    """
    if regions_df is None or regions_df.empty:
        return REPRESENTATIVE_BY_REGION_DEFAULT.copy()

    possible_region_cols = [c for c in regions_df.columns if c.lower() in {"region", "climatic_region"}]
    if not possible_region_cols or "site" not in regions_df.columns:
        return REPRESENTATIVE_BY_REGION_DEFAULT.copy()

    region_col = possible_region_cols[0]
    reps = {}
    for region_name in ["Coastal", "Andean", "Rainforest"]:
        m = regions_df[regions_df[region_col].astype(str).str.lower() == region_name.lower()]
        if not m.empty:
            reps[region_name] = str(m.iloc[0]["site"])

    for k, v in REPRESENTATIVE_BY_REGION_DEFAULT.items():
        reps.setdefault(k, v)
    return reps

# ----------------------------- Gráficos -----------------------------

def scatter_panel(ax, d: pd.DataFrame, site: str, region_label: str, var: str, ylab: str, use_log10: bool, bias_corr: bool, show_xlabel: bool, show_ylabel: bool) -> None:
    """Dibuja un panel de dispersión con 1:1 y regresión."""
    x, y, _ = prepare_xy(d, var=var, bias_corr=bias_corr, use_log10=use_log10)
    r = pearson_r(x, y)
    e = rmse(y, x)
    m, b = linear_fit(x, y)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad

    ax.scatter(x, y, s=30)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    xx = np.linspace(lo, hi, 100)
    ax.plot(xx, m * xx + b, color="red", lw=1.0)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(f"{site} ({region_label})")
    ax.text(
        0.03, 0.95, f"r={r:.2f}\nRMSE={e:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    if show_ylabel:
        ax.set_ylabel(ylab)
    if show_xlabel:
        ax.set_xlabel("POWER")

def make_combined_figure(daily_pairs: pd.DataFrame, representative_sites: Dict[str, str], outdir: Path) -> Path:
    """Genera figura combinada Q1: RH (fila superior) + PRECI (fila inferior)."""
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)

    regions_order = ["Coastal", "Andean", "Rainforest"]
    row_cfg = [
        ("RH", "Relative Humidity (%)", False, False),
        ("PRECI", "Precipitation (log$_{10}$ mm/day)", True, False),
    ]

    for i, (var, ylab, use_log10, bias_corr) in enumerate(row_cfg):
        for j, region in enumerate(regions_order):
            site = representative_sites[region]
            d = daily_pairs[(daily_pairs["site"] == site) & (daily_pairs["var"] == var)].copy()
            scatter_panel(
                axes[i, j], d, site, region, var, ylab, use_log10, bias_corr,
                show_xlabel=(i == 1), show_ylabel=(j == 0)
            )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", label="Observed", markersize=5),
        Line2D([0], [0], color="black", linestyle="--", label="1:1", lw=1.0),
        Line2D([0], [0], color="red", label="Regression", lw=1.0),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    out = outdir / "consistency_combined_ieee.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

def make_three_region_plot(daily_pairs: pd.DataFrame, representative_sites: Dict[str, str], var: str, outdir: Path) -> Path:
    """Genera figura 1x3 para una variable específica."""
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)

    use_log10 = var in LOG10_VARS
    bias_corr = var in BIAS_CORR_VARS
    ylab = {
        "SP": "Surface Pressure (hPa)",
        "T": "Temperature (°C)",
        "RH": "Relative Humidity (%)",
        "PRECI": "Precipitation (log$_{10}$ mm/day)" if use_log10 else "Precipitation (mm/day)",
    }[var]

    for ax, region in zip(axes, ["Coastal", "Andean", "Rainforest"]):
        site = representative_sites[region]
        d = daily_pairs[(daily_pairs["site"] == site) & (daily_pairs["var"] == var)].copy()
        scatter_panel(
            ax, d, site, region, var, ylab, use_log10, bias_corr,
            show_xlabel=True, show_ylabel=(ax is axes[0])
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", label="Observed", markersize=5),
        Line2D([0], [0], color="black", linestyle="--", label="1:1", lw=1.0),
        Line2D([0], [0], color="red", label="Regression", lw=1.0),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    out = outdir / f"{var.lower()}_consistency_3regions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

# ----------------------------- CLI principal -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pipeline completo para cálculo de consistencia POWER vs ERA5/ERA5-Land"
    )
    ap.add_argument("--power", required=True, help="CSV horario POWER en formato largo")
    ap.add_argument("--era5", required=True, help="CSV horario ERA5 en formato largo")
    ap.add_argument("--datetime-col", default="datetime")
    ap.add_argument("--site-col", default="site")
    ap.add_argument("--var-col", default="var")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--agg", default=None, help='Mapa opcional: "SP=mean,T=mean,RH=mean,PRECI=sum"')
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--locations", default=None, help="CSV opcional con site,location")
    ap.add_argument("--regions", default=None, help="CSV opcional con site,region")
    ap.add_argument("--year", type=int, default=None, help="Filtra por año si aplica")
    ap.add_argument("--outdir", default="report_out", help="Carpeta base de salida")
    return ap.parse_args()

def build_agg_map(arg: str | None) -> Dict[str, str]:
    agg = AGG_MAP_DEFAULT.copy()
    if not arg:
        return agg
    for kv in [p.strip() for p in arg.split(",") if p.strip()]:
        k, v = [t.strip() for t in kv.split("=", 1)]
        agg[k] = v
    return agg

def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))
    figs_dir = ensure_dir(outdir / "figs")

    agg_map = build_agg_map(args.agg)

    power = read_hourly_csv(args.power, args.datetime_col, args.site_col, args.var_col, args.value_col)
    era5 = read_hourly_csv(args.era5, args.datetime_col, args.site_col, args.var_col, args.value_col)

    if args.year is not None:
        power = power[power["datetime"].dt.year == int(args.year)].copy()
        era5 = era5[era5["datetime"].dt.year == int(args.year)].copy()

    power_daily = aggregate_daily(power, agg_map)
    era5_daily = aggregate_daily(era5, agg_map)
    daily_pairs = build_daily_pairs(power_daily, era5_daily, min_coverage=args.min_coverage)

    daily_pairs = attach_optional_mapping(daily_pairs, args.locations)
    daily_pairs = attach_optional_mapping(daily_pairs, args.regions)

    daily_pairs.to_csv(outdir / "daily_pairs.csv", index=False, float_format="%.6f")
    print(f"[OK] {outdir / 'daily_pairs.csv'}")

    metrics_all = build_metrics_all(daily_pairs)
    metrics_all.to_csv(outdir / "metrics_all.csv", index=False, float_format="%.6f")
    print(f"[OK] {outdir / 'metrics_all.csv'}")

    write_rankings(metrics_all, outdir)
    print(f"[OK] {outdir / 'summary_rankings.md'}")

    regions_df = pd.read_csv(args.regions) if args.regions else pd.DataFrame()
    representative_sites = representative_sites_from_regions(regions_df)

    # Figuras individuales por variable
    for var in ["SP", "T", "RH", "PRECI"]:
        f = make_three_region_plot(daily_pairs, representative_sites, var, figs_dir)
        print(f"[OK] {f}")

    # Figura combinada principal
    f_comb = make_combined_figure(daily_pairs, representative_sites, figs_dir)
    print(f"[OK] {f_comb}")

    print("[DONE] Pipeline completado.")

if __name__ == "__main__":
    main()

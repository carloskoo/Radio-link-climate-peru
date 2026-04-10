import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path("..") / "data" / "processed"

TOTAL_CSV = BASE_DIR / "phase7_total_attenuation_results.csv"

OUT_DIR = Path("..") / "figures"
OUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(TOTAL_CSV)

df.columns = [c.lower().strip() for c in df.columns]

# Normalizar regiones
df["region"] = df["region"].replace({
    "costa": "Coastal",
    "sierra": "Andean",
    "selva": "Rainforest"
})

# Solo p95 (clave para diseño)
df = df[df["percentile"] == 95]

# =========================
# RAIN DOMINANCE RATIO
# =========================
df["rain_ratio"] = df["gammar_db_per_km"] / df["gamma_total_db_per_km"]

# =========================
# FIGURA Q1
# =========================
regions = ["Coastal", "Andean", "Rainforest"]

plt.figure(figsize=(9,5))

for region in regions:
    sub = df[df["region"] == region].sort_values("f_ghz")

    plt.plot(
        sub["f_ghz"],
        sub["rain_ratio"],
        linewidth=2,
        label=region
    )

# Línea de transición crítica (75%)
plt.axhline(y=0.75, linestyle="--", linewidth=1.5)

plt.xlabel("Frequency (GHz)")
plt.ylabel(r"$C_{rain} = \gamma_R / \gamma_{total}$")
plt.title("Rain dominance ratio vs frequency (p95)")

plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(frameon=True)

plt.ylim(0,1)

plt.savefig(OUT_DIR / "fig_rain_dominance_ratio_q1.png", dpi=600, bbox_inches="tight")
plt.show()

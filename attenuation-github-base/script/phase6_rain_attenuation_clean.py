import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================
BASE_DIR = Path(".")
RAIN_CSV = BASE_DIR / "phase6_rain_rates.csv"
KALPHA_CSV = BASE_DIR / "phase6_k_alpha_p838.csv"

OUT_DIR = BASE_DIR
OUT_DIR.mkdir(exist_ok=True)

# =========================
# CARGA DE DATOS
# =========================
rain = pd.read_csv(RAIN_CSV)
kalpha = pd.read_csv(KALPHA_CSV)

# Normalizar nombres
rain.columns = [c.lower().strip() for c in rain.columns]
kalpha.columns = [c.lower().strip() for c in kalpha.columns]

# =========================
# FUNCIÓN ITU-R P.838-3
# =========================
def gamma_rain(R, k, alpha):
    return k * (R ** alpha)

# =========================
# PREPARAR DATASET
# =========================
results = []

for _, row in rain.iterrows():
    region = row["region"].lower()
    percentile = row["percentile"]
    R = row["rain_rate_mm_h"]

    for _, krow in kalpha.iterrows():
        f = krow["f_ghz"]
        pol = krow["pol"]
        k = krow["k"]
        alpha = krow["alpha"]

        gamma = gamma_rain(R, k, alpha)

        results.append({
            "region": region,
            "percentile": percentile,
            "f_ghz": f,
            "gammaR_dB_per_km": gamma
        })

df = pd.DataFrame(results)

# =========================
# LIMPIEZA
# =========================
df["region"] = df["region"].replace({
    "costa": "Coastal",
    "sierra": "Andean",
    "selva": "Rainforest"
})

df = df[df["percentile"].isin([50, 95])]
df = df.groupby(["region", "percentile", "f_ghz"], as_index=False).mean()

# =========================
# GUARDAR CSV
# =========================
csv_out = OUT_DIR / "phase6_rain_results_clean.csv"
df.to_csv(csv_out, index=False)

# =========================
# FIGURA (ESTILO Q1)
# =========================
regions = ["Coastal", "Andean", "Rainforest"]

plt.figure(figsize=(9,5))

for region in regions:
    for p in [50, 95]:
        sub = df[(df["region"] == region) & (df["percentile"] == p)].sort_values("f_ghz")

        linestyle = "-" if p == 50 else "--"

        plt.plot(
            sub["f_ghz"],
            sub["gammaR_dB_per_km"],
            linestyle=linestyle,
            linewidth=2,
            label=f"{region} p{p}"
        )

plt.xlabel("Frequency (GHz)")
plt.ylabel(r"$\gamma_R(f)$ (dB/km)")
plt.title("Rain attenuation")

plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend(ncol=2, frameon=True)

fig_out = OUT_DIR / "fig_rain_attenuation_q1.png"
plt.savefig(fig_out, dpi=600, bbox_inches="tight")
plt.show()

print("OK generado:")
print(csv_out)
print(fig_out)

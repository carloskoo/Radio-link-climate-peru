import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =============================
# CONFIG
# =============================
INPUT_FILE = "../metrics_all.csv"
OUTPUT_DIR = "../figures"

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(INPUT_FILE)

# Asegurar nombres consistentes
df['variable'] = df['variable'].str.upper()

# Orden IEEE típico
order = ['SP', 'RH', 'PRECIP', 'T']

# =============================
# 1. CORRELATION (r promedio)
# =============================
r_mean = df.groupby('variable')['r'].mean().reindex(order)

plt.figure(figsize=(6,4))
bars = plt.bar(r_mean.index, r_mean.values)

# Estilo limpio tipo paper
plt.ylabel("r (average)")
plt.title("Mean correlation (ERA5 vs NASA POWER)")

# Valores encima
for i, v in enumerate(r_mean.values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mean_correlation.png", dpi=300)
plt.close()


# =============================
# 2. RMSE promedio
# =============================
rmse_mean = df.groupby('variable')['RMSE'].mean().reindex(order)

plt.figure(figsize=(6,4))
bars = plt.bar(rmse_mean.index, rmse_mean.values)

plt.ylabel("RMSE (average)")
plt.title("Mean RMSE (ERA5 vs NASA POWER)")

for i, v in enumerate(rmse_mean.values):
    plt.text(i, v + 0.2, f"{v:.2f}", ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mean_rmse.png", dpi=300)
plt.close()

print("✔ Figures generated:")
print(" - mean_correlation.png")
print(" - mean_rmse.png")

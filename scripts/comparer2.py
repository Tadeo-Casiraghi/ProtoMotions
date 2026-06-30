import os
import glob
import re
import numpy as np
import pandas as pd

# ======================================================
# CONFIGURACIÓN
# ======================================================

data_folder = "data2"
output_file = "results_summary.csv"

GROUND_FREQ = 4800
GROUND_POS = 32
GROUND_VEL = 8

columns = [
    "Pistoning (suspension_slide)",
    "Angle_X",
    "Angle_Y",
    "Angle_Z"
]

# ======================================================
# CARGAR GROUND TRUTH
# ======================================================

gt_filename = f"movement_freq{GROUND_FREQ}hz_pos{GROUND_POS}_vel{GROUND_VEL}.csv"
gt_path = os.path.join(data_folder, gt_filename)

df_gt = pd.read_csv(gt_path)

# conservar únicamente las columnas necesarias
df_gt = df_gt[["Time"] + columns]

print("Ground Truth:", gt_filename)

# ======================================================
# RECORRER TODOS LOS CSV
# ======================================================

pattern = os.path.join(data_folder, "movement_freq*hz_pos*_vel*.csv")

results = []

for file in glob.glob(pattern):

    filename = os.path.basename(file)

    # saltar el ground truth
    if filename == gt_filename:
        continue

    m = re.search(r"freq(\d+)hz_pos(\d+)_vel(\d+)", filename)

    if m is None:
        continue

    freq = int(m.group(1))
    pos = int(m.group(2))
    vel = int(m.group(3))

    df = pd.read_csv(file)

    df = df[["Time"] + columns + ["Mean_Wall_Time_Sec"]]

    # merge únicamente en tiempos comunes
    merged = pd.merge(
        df,
        df_gt,
        on="Time",
        suffixes=("_test", "_gt")
    )

    if len(merged) == 0:
        print("No hubo coincidencias:", filename)
        continue

    row = {
        "Frequency": freq,
        "PositionIterations": pos,
        "VelocityIterations": vel,
        "WallTime": merged["Mean_Wall_Time_Sec"].iloc[0]
    }

    # --------------------------------------------------
    # RMSE por variable
    # --------------------------------------------------

    normalized_errors = []
    eps = 1e-12

    for col in columns:

        rmse = np.sqrt(
            np.mean(
                (merged[f"{col}_test"] - merged[f"{col}_gt"]) ** 2
            )
        )

        row[f"RMSE_{col}"] = rmse

        gt_std = merged[f"{col}_gt"].std()

        if gt_std > eps:
            normalized_errors.append(rmse / gt_std)

    if normalized_errors:
        row["RMSE_Global"] = np.sqrt(np.mean(np.square(normalized_errors)))
    else:
        row["RMSE_Global"] = np.nan

    results.append(row)
# ======================================================
# Agregar el Ground Truth
# ======================================================

results.append({
    "Frequency":GROUND_FREQ,
    "PositionIterations":GROUND_POS,
    "VelocityIterations":GROUND_VEL,
    "WallTime":df_gt["Time"].shape[0],   # luego se reemplaza
    "RMSE_Pistoning (suspension_slide)":0,
    "RMSE_Angle_X":0,
    "RMSE_Angle_Y":0,
    "RMSE_Angle_Z":0,
    "RMSE_Global":0
})

# poner el tiempo real del GT
gt_wall = pd.read_csv(gt_path)["Mean_Wall_Time_Sec"].iloc[0]
results[-1]["WallTime"] = gt_wall

# ======================================================
# GUARDAR
# ======================================================

df_results = pd.DataFrame(results)

df_results = df_results.sort_values(
    ["Frequency", "PositionIterations", "VelocityIterations"]
)

df_results.to_csv(output_file, index=False)

print()
print(f"Resumen guardado en {output_file}")
print(df_results.head())
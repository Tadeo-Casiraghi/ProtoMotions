import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
folder_path = "data"        # Carpeta donde están tus archivos CSV
output_folder = "figs"      # Carpeta donde se guardarán los gráficos
kps = [1000, 2000]
ground_truth_freq = 1200

# Columnas que vamos a analizar
columns_to_compare = {
    "Posición": "Motor Position",
    "Velocidad": "Motor Velocity",
    "Torque Aplicado": "Motor Applied Torque",
    "Posicion Deseada": "Target Motor Position"
}

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Lista para guardar todos los resultados calculados
data_compiled = []

# --- PROCESAMIENTO DE DATOS ---
for kp in kps:
    # 1. Cargar el Ground Truth (1200 Hz) para el KP actual
    gt_file = os.path.join(folder_path, f"kp{kp}_freq{ground_truth_freq}hz.csv")
    
    if not os.path.exists(gt_file):
        print(f"⚠️ No se encontró el archivo Ground Truth: {gt_file}")
        continue
        
    df_gt_raw = pd.read_csv(gt_file)

    # Lógica exacta para aislar el transitorio del Ground Truth
    dt_gt = df_gt_raw["Time"].iloc[1] - df_gt_raw["Time"].iloc[0]
    # Tiempo justo antes de que ocurra el step
    step_time_gt = df_gt_raw.loc[df_gt_raw["Target Motor Position"] > 0, "Time"].iloc[0] - dt_gt
    
    # Recortar ventana y normalizar tiempo en el Ground Truth
    df_gt = df_gt_raw[df_gt_raw["Time"] >= step_time_gt].copy()
    df_gt["Time"] = (df_gt["Time"] - step_time_gt).round(6)
    df_gt = df_gt[df_gt["Time"] <= 0.1]

    # 2. Buscar todos los archivos de este KP para identificar las otras frecuencias
    search_pattern = os.path.join(folder_path, f"kp{kp}_freq*hz.csv")
    all_files = glob.glob(search_pattern)
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            freq_str = filename.split("freq")[1].split("hz")[0]
            freq = int(freq_str)
        except (IndexError, ValueError):
            continue
            
        if freq == ground_truth_freq or freq == 120:
            continue
        
        # Cargar archivo de prueba
        df_test_raw = pd.read_csv(file_path)

        dt_test = df_test_raw["Time"].iloc[1] - df_test_raw["Time"].iloc[0]
        step_time_test = df_test_raw.loc[df_test_raw["Target Motor Position"] > 0, "Time"].iloc[0] - dt_test
        
        # Recortar ventana inicial
        df_test = df_test_raw[df_test_raw["Time"] >= step_time_test].copy()
        
        # Normalizar a t=0
        df_test["Time"] = df_test["Time"] - step_time_test
        
        # --- COMPENSACIÓN MATEMÁTICA DEL DESFASE ---
        desfase_compensacion = (1.0 / freq) - (1.0 / ground_truth_freq)
        df_test["Time"] -= desfase_compensacion
        
        # Redondear para el merge y filtrar la ventana exacta de 0.1s
        df_test["Time"] = df_test["Time"].round(6)
        df_test = df_test[(df_test["Time"] >= 0.0) & (df_test["Time"] <= 0.1)]
        
        # Alinear los datos mediante un INNER JOIN basado en el tiempo corregido
        df_merged = pd.merge(df_test, df_gt, on='Time', suffixes=('_test', '_gt'))
        
        if df_merged.empty:
            print(f"⚠️ No hubo coincidencias de tiempo entre {freq}Hz y 1200Hz para KP {kp} tras la compensación")
            continue
            
        # Calcular el % de Error (NRMSE) para cada variable en la ventana corregida
        row_result = {'KP': kp, 'Frequency': freq}
        for label, col_name in columns_to_compare.items():
            # 1. Error cuadrático medio (RMSE)
            error_cuadratico = (df_merged[f"{col_name}_test"] - df_merged[f"{col_name}_gt"]) ** 2
            rmse = np.sqrt(error_cuadratico.mean())
            
            # 2. Rango dinámico del Ground Truth en la ventana de 0.1s
            gt_max = df_merged[f"{col_name}_gt"].max()
            gt_min = df_merged[f"{col_name}_gt"].min()
            gt_range = gt_max - gt_min
            
            if gt_range == 0:
                gt_range = 1.0  # Evita divisiones por cero si la señal no se mueve
                
            # 3. Mapear a porcentaje
            nrmse_percentage = (rmse / gt_range) * 100
            print(f"KP={kp}, Freq={freq}Hz, {label}: RMSE={rmse:.4f}, GT Range={gt_range:.4f}, NRMSE%={nrmse_percentage:.2f}%")
            row_result[label] = nrmse_percentage
            
        data_compiled.append(row_result)
        
    # Forzar el error 0% en la frecuencia de 1200Hz para la convergencia visual
    perfect_row = {'KP': kp, 'Frequency': ground_truth_freq}
    for label in columns_to_compare:
        perfect_row[label] = 0.0
    data_compiled.append(perfect_row)

# Convertir resultados a un DataFrame limpio
df_results = pd.DataFrame(data_compiled).sort_values(by=['KP', 'Frequency'])


# --- GRAFICADO Y GUARDADO ---
colors = {1000: '#1f77b4', 2000: '#ff7f0e'} # Azul y Naranja

for label, col_name in columns_to_compare.items():
    plt.figure(figsize=(8, 5))
    
    for kp in kps:
        df_kp = df_results[df_results['KP'] == kp]
        
        plt.plot(
            df_kp['Frequency'], 
            df_kp[label], 
            marker='o', 
            linestyle='-', 
            linewidth=2, 
            color=colors[kp],
            label=f"KP = {kp}"
        )
        
    plt.title(f"Porcentaje de Error en {label} vs Frecuencia (Ventana: 0.1s)", fontsize=12, fontweight='bold')
    plt.xlabel("Frecuencia (Hz)", fontsize=10)
    plt.ylabel("Error Relativo NRMSE (%)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.ylim(bottom=0)  # El piso visual siempre arranca en 0%
    
    sanitized_label = label.lower().replace(" ", "_")
    file_name = f"error_porcentaje_{sanitized_label}.png"
    save_path = os.path.join(output_folder, file_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Gráfico de porcentaje guardado en: {save_path}")
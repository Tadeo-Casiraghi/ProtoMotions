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
    "Posicion Deseado": "Target Motor Position"
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

    # Lógica exacta para el Ground Truth (df2 en tu script original)
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
        
        # Cargar archivo de prueba (df1 en tu script original)
        df_test_raw = pd.read_csv(file_path)

        dt_test = df_test_raw["Time"].iloc[1] - df_test_raw["Time"].iloc[0]
        step_time_test = df_test_raw.loc[df_test_raw["Target Motor Position"] > 0, "Time"].iloc[0] - dt_test
        
        # Recortar ventana inicial
        df_test = df_test_raw[df_test_raw["Time"] >= step_time_test].copy()
        
        # Normalizar a t=0
        df_test["Time"] = df_test["Time"] - step_time_test
        
        # --- TU COMPENSACIÓN MATEMÁTICA ---
        # Tu ecuación: (1200/freq - 1) * (1/1200) simplifica a: (1/freq) - (1/1200)
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
            
        # Calcular el RMSE para cada variable en la ventana de 0.1 segundos
        row_result = {'KP': kp, 'Frequency': freq}
        for label, col_name in columns_to_compare.items():
            error_cuadratico = (df_merged[f"{col_name}_test"] - df_merged[f"{col_name}_gt"]) ** 2
            rmse = np.sqrt(error_cuadratico.mean())
            row_result[label] = rmse
            
        data_compiled.append(row_result)
        
    # Forzar el error 0 en la frecuencia de 1200Hz para la convergencia visual
    perfect_row = {'KP': kp, 'Frequency': ground_truth_freq}
    for label in columns_to_compare:
        perfect_row[label] = 0.0
    data_compiled.append(perfect_row)

# Convertir resultados a un DataFrame limpio
df_results = pd.DataFrame(data_compiled).sort_values(by=['KP', 'Frequency'])


# --- GRAFICADO Y GUARDADO ---
colors = {1000: '#1f77b4', 2000: '#ff7f0e'} # Azul y Naranja

# Iteramos sobre cada variable para generar un archivo independiente
for label, col_name in columns_to_compare.items():
    # Crear una nueva figura para este gráfico específico
    plt.figure(figsize=(8, 5))
    
    for kp in kps:
        # Filtrar datos por KP
        df_kp = df_results[df_results['KP'] == kp]
        
        # Graficar Línea + Marcadores
        plt.plot(
            df_kp['Frequency'], 
            df_kp[label], 
            marker='o', 
            linestyle='-', 
            linewidth=2, 
            color=colors[kp],
            label=f"KP = {kp}"
        )
        
    plt.title(f"Error en {label} vs Frecuencia (Ground Truth: 1200 Hz)", fontsize=12, fontweight='bold')
    plt.xlabel("Frecuencia (Hz)", fontsize=10)
    plt.ylabel("Error (RMSE)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    
    # Definir el nombre del archivo (ej: "error_torque_aplicado.png")
    sanitized_label = label.lower().replace(" ", "_")
    file_name = f"error_{sanitized_label}.png"
    save_path = os.path.join(output_folder, file_name)
    
    # Guardar la imagen con alta resolución (300 DPI) y ajustando los márgenes
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Limpiar y cerrar la figura actual de la memoria antes de pasar a la siguiente
    plt.close()
    
    print(f"💾 Gráfico guardado exitosamente en: {save_path}")
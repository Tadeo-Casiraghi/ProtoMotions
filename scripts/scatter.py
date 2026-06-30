import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Cargar resultados
# -----------------------

df = pd.read_csv("results_summary.csv")

gt_time = df.loc[
    (df["Frequency"]==360) &
    (df["PositionIterations"]==16) &
    (df["VelocityIterations"]==8),
    "WallTime"
].iloc[0]

df["Speedup"] = gt_time / df["WallTime"]

# -----------------------
# Configuración gráfica
# -----------------------

markers = {
    2: "o",
    4: "s",
    8: "^",
    16: "D",
    32: "P"
}

freqs = sorted(df["Frequency"].unique())

cmap = plt.cm.viridis
colors = {
    f: cmap(i/(len(freqs)-1))
    for i, f in enumerate(freqs)
}

# tamaño según vel iterations
sizes = {
    1:40,
    2:70,
    4:110,
    8:170
}

fig, ax = plt.subplots(figsize=(10,8))

# -----------------------
# Scatter
# -----------------------

for _, row in df.iterrows():

    ax.scatter(
        row["Speedup"],
        row["RMSE_Global"],
        marker=markers[row["PositionIterations"]],
        color=colors[row["Frequency"]],
        s=sizes[row["VelocityIterations"]],
        edgecolor="k",
        linewidth=0.5,
        alpha=0.8
    )

    # comentar esta línea si quedan demasiadas etiquetas
    ax.annotate(
        f'{row["Frequency"]}-{row["PositionIterations"]}-{row["VelocityIterations"]}',
        (row["Speedup"]+0.01, row["RMSE_Global"]),
        fontsize=6,
        alpha=0.7
    )

# -----------------------
# Etiquetas
# -----------------------

ax.set_xlabel("Speedup")
ax.set_ylabel("Global RMSE")
ax.set_title("Accuracy vs Computational Cost")

ax.grid(alpha=0.3)

# -----------------------
# Leyenda de formas
# -----------------------

shape_handles = []

for pos, marker in markers.items():
    shape_handles.append(
        plt.Line2D(
            [],
            [],
            color="black",
            marker=marker,
            linestyle="",
            markersize=8,
            label=f"Pos {pos}"
        )
    )

legend1 = plt.legend(
    handles=shape_handles,
    title="Position Iterations",
    loc="upper right"
)

ax.add_artist(legend1)

# -----------------------
# Leyenda de tamaños
# -----------------------

size_handles = []

for vel, s in sizes.items():
    size_handles.append(
        ax.scatter([], [], s=s, color="gray", label=f"Vel {vel}")
    )

ax.legend(
    handles=size_handles,
    title="Velocity Iterations",
    loc="lower right"
)

# -----------------------
# Barra de colores
# -----------------------

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(
        vmin=min(freqs),
        vmax=max(freqs)
    )
)

sm.set_array([])

fig.colorbar(sm, ax=ax, label="Frequency [Hz]")

plt.tight_layout()
plt.savefig("scatter.png", dpi=600)
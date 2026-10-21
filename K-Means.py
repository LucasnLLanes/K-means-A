# ============================================================
# Clustering de Jurisdicciones Argentinas con K-Means
# Autor: Lucas Nahuel Llanes
# Materia: Aprendizaje Automático
# ISFT N° 197 — 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paleta púrpura / coral / turquesa
PURPURA  = "#5e35b1"
CORAL    = "#ff6f61"
TURQUESA = "#26c6da"
TEXTO    = "#2d2d3a"

# ------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------
registros = [
    {"jurisdicción": "Ciudad Autónoma de Buenos Aires", "viviendas": 1391258},
    {"jurisdicción": "Buenos Aires", "viviendas": 5970702},
    {"jurisdicción": "Catamarca", "viviendas": 131978},
    {"jurisdicción": "Chaco", "viviendas": 368728},
    {"jurisdicción": "Chubut", "viviendas": 213317},
    {"jurisdicción": "Córdoba", "viviendas": 1378237},
    {"jurisdicción": "Corrientes", "viviendas": 370958},
    {"jurisdicción": "Entre Ríos", "viviendas": 494473},
    {"jurisdicción": "Formosa", "viviendas": 194689},
    {"jurisdicción": "Jujuy", "viviendas": 238141},
    {"jurisdicción": "La Pampa", "viviendas": 140879},
    {"jurisdicción": "La Rioja", "viviendas": 124149},
    {"jurisdicción": "Mendoza", "viviendas": 639467},
    {"jurisdicción": "Misiones", "viviendas": 420101},
    {"jurisdicción": "Neuquén", "viviendas": 254545},
    {"jurisdicción": "Río Negro", "viviendas": 276371},
    {"jurisdicción": "Salta", "viviendas": 404504},
    {"jurisdicción": "San Juan", "viviendas": 241436},
    {"jurisdicción": "San Luis", "viviendas": 182886},
    {"jurisdicción": "Santa Cruz", "viviendas": 118047},
    {"jurisdicción": "Santa Fe", "viviendas": 1273460},
    {"jurisdicción": "Santiago del Estero", "viviendas": 311361},
    {"jurisdicción": "Tierra del Fuego, Antártida e Islas del Atlántico Sur", "viviendas": 65535},
    {"jurisdicción": "Tucumán", "viviendas": 493794},
]

df = pd.DataFrame(registros)
print("Vista previa:")
print(df.head(), "\n")

# ------------------------------------------------------------
# 2. Estandarización
# ------------------------------------------------------------
X = df[["viviendas"]].values
escalador = StandardScaler()
X_esc = escalador.fit_transform(X)

# ------------------------------------------------------------
# 3. Método del codo
# ------------------------------------------------------------
inercias = []
valores_k = range(1, 11)

for k in valores_k:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_esc)
    inercias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(valores_k, inercias, marker="D", color=PURPURA,
         linewidth=2.5, markersize=8,
         markerfacecolor=CORAL, markeredgecolor=TEXTO)
plt.title("Método del Codo — Elección de K", color=PURPURA, fontsize=13)
plt.xlabel("Cantidad de Clusters (K)")
plt.ylabel("Inercia")
plt.xticks(valores_k)
plt.grid(True, alpha=0.25, linestyle="--")
plt.savefig("grafico_codo.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 4. Modelo final K=3
# ------------------------------------------------------------
modelo = KMeans(n_clusters=3, random_state=42)
df["cluster"] = modelo.fit_predict(X_esc)

print("\nJurisdicciones ordenadas:")
print(df.sort_values("viviendas")[["jurisdicción", "viviendas", "cluster"]].to_string(index=False))

# ------------------------------------------------------------
# 5. Visualización
# ------------------------------------------------------------
paleta = {0: PURPURA, 1: CORAL, 2: TURQUESA}

plt.figure(figsize=(11, 6))
for c in range(3):
    grupo = df[df["cluster"] == c]
    plt.scatter(grupo["jurisdicción"], grupo["viviendas"],
                label=f"Grupo {c}", color=paleta[c],
                s=90, edgecolor=TEXTO, linewidth=0.6, alpha=0.9)

plt.xticks(rotation=90)
plt.xlabel("Jurisdicción")
plt.ylabel("Cantidad de viviendas")
plt.title("Segmentación de Viviendas — K-Means", color=PURPURA, fontsize=13)
plt.legend(frameon=True, edgecolor=PURPURA)
plt.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig("grafico_segmentacion.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 6. Resumen
# ------------------------------------------------------------
print("\n=== RESUMEN DE CLUSTERS ===")
for c in range(3):
    grupo = df[df["cluster"] == c]
    print(f"\nGrupo {c} ({paleta[c]}): {len(grupo)} jurisdicciones")
    print(f"  Rango: {grupo['viviendas'].min():,} – {grupo['viviendas'].max():,}")
    print(f"  Integrantes: {', '.join(grupo['jurisdicción'].tolist())}")

print("\nListo.")
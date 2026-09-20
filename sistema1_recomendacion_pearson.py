import pandas as pd
import numpy as np

RUTA = r"C:\Users\ALIENWARE\Desktop\SextoSemestre\BI-PROYECTO\Sistemas de Recomendacion\Pearson.csv"
TOP_N = 5
MIN_TEMPORADAS_COMUNES = 8  # con menos puntos, Pearson es matematicamente inestable

# ETAPA 1: Carga
df = pd.read_csv(RUTA, sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "OPS"], encoding="utf-8-sig")

# ETAPA 2: Anio calendario -> Temporada N de carrera (para comparar entre epocas)
df = df.sort_values(["Id_Jugador", "Anio"])
df["Temporada_Numero"] = df.groupby("Id_Jugador").cumcount() + 1

# ETAPA 3: Matriz Jugador x Temporada_Numero
pivot = df.pivot_table(index="Nombre_Completo", columns="Temporada_Numero", values="OPS")

# ETAPA 4: Correlacion de Pearson entre todos los jugadores (pairwise)
matriz_corr = pivot.T.corr(min_periods=MIN_TEMPORADAS_COMUNES)

# ETAPA 5: Top-N automatico para cada jugador
filas = []
for jugador_base in matriz_corr.index:
    correlaciones = matriz_corr[jugador_base].drop(jugador_base).dropna()
    if correlaciones.empty:
        continue
    top = correlaciones.sort_values(ascending=False).head(TOP_N)
    for ranking, (comparable, r) in enumerate(top.items(), start=1):
        filas.append({"Jugador_Base": jugador_base, "Ranking": ranking,
                       "Jugador_Comparable": comparable, "Correlacion_Pearson": round(r, 4)})

resultado = pd.DataFrame(filas)
resultado.to_csv("salida_pearson.csv", index=False, sep=";", decimal=",")
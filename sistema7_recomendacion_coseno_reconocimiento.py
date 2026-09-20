"""
Sistema 7 -- FactReconocimiento (repite el metodo del sistema 2: Similitud
de Coseno), consumiendo datos de FactReconocimiento unidos a DimJugador,
DimTiempo y DimTipoReconocimiento (ver extraer_reconocimiento.py).

Por que Coseno: igual que en sistema2 (perfil de 6 atributos ofensivos
estandarizados), cada jugador aqui se representa como un vector de cuantas
veces gano cada TIPO de reconocimiento a lo largo de su carrera (Rookie of
the Year, All-Star, MVP, etc.). No es una serie temporal como en Pearson,
es un perfil de composicion en un momento dado (fin de carrera), que es
exactamente la forma de dato para la que esta pensada la similitud de
coseno: encontrar jugadores con un patron de reconocimientos parecido,
respondiendo la pregunta de negocio sobre reconocimientos vs. trayectoria
de rendimiento.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Reconocimiento.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Tipo_Reconocimiento", "Cantidad",
           "Total_Reconocimientos"], encoding="utf-8-sig")

# ETAPA 1: matriz Jugador x Tipo_Reconocimiento (celdas = Cantidad ganada)
matriz = df.pivot_table(index=["Id_Jugador", "Nombre_Completo"],
                         columns="Tipo_Reconocimiento", values="Cantidad",
                         aggfunc="sum", fill_value=0)

# ETAPA 2: estandarizar cada tipo de reconocimiento (Z-score) y calcular
# similitud de coseno entre todos los jugadores
X_std = StandardScaler().fit_transform(matriz.values)
sim_matrix = cosine_similarity(X_std)
np.fill_diagonal(sim_matrix, -1)

nombres = matriz.index.get_level_values("Nombre_Completo").to_numpy()

TOP_N = 5
filas = []
for idx in range(len(matriz)):
    sims = sim_matrix[idx]
    top_idx = np.argpartition(sims, -TOP_N)[-TOP_N:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    for ranking, j in enumerate(top_idx, start=1):
        filas.append({"Jugador_Base": nombres[idx], "Ranking": ranking,
                       "Jugador_Comparable": nombres[j],
                       "Similitud_Coseno": round(float(sims[j]), 4)})

resultado = pd.DataFrame(filas)
resultado.to_csv("salida_coseno_reconocimiento.csv", index=False, sep=";", decimal=",")

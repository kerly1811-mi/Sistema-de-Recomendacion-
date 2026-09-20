import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("similitud coseno.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Turnos_Al_Bate", "Imparables",
           "Cuadrangulares", "Carreras_Impulsadas", "Bases_Por_Bolas", "OPS_Postemporada"],
    encoding="utf-8-sig", na_values=["NULL"])

# Excluir perfiles no calculables (sin turnos al bate reales en postemporada)
df = df.dropna(subset=["OPS_Postemporada"]).reset_index(drop=True)

atributos = ["Turnos_Al_Bate", "Imparables", "Cuadrangulares",
             "Carreras_Impulsadas", "Bases_Por_Bolas", "OPS_Postemporada"]
X_std = StandardScaler().fit_transform(df[atributos].values)

sim_matrix = cosine_similarity(X_std)
np.fill_diagonal(sim_matrix, -1)

TOP_N = 5
filas = []
for idx in range(len(df)):
    sims = sim_matrix[idx]
    top_idx = np.argpartition(sims, -TOP_N)[-TOP_N:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    for ranking, j in enumerate(top_idx, start=1):
        filas.append({"Jugador_Base": df.loc[idx, "Nombre_Completo"], "Ranking": ranking,
                       "Jugador_Comparable": df.loc[j, "Nombre_Completo"],
                       "Similitud_Coseno": round(float(sims[j]), 4)})

resultado = pd.DataFrame(filas)
resultado.to_csv("salida_coseno_postemporada.csv", index=False, sep=";", decimal=",")
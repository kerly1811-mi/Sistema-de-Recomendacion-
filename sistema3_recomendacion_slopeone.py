import pandas as pd
import numpy as np

df = pd.read_csv("Slope One.csv", sep=";", decimal=".", header=None,
    names=["Id_Equipo", "Nombre_Equipo", "Anio", "Salario_Promedio_Equipo", "Porcentaje_Victorias"],
    encoding="utf-8-sig")

# Item 1 = percentil de salario dentro del mismo anio (misma escala 0-1 que Item 2)
df["Salario_Relativo"] = df.groupby("Anio")["Salario_Promedio_Equipo"].rank(pct=True)

# b = diferencia media entre Item2 e Item1 (formula f(x) = x + b de la diapositiva)
item1 = df["Salario_Relativo"].values
item2 = df["Porcentaje_Victorias"].values
b = float(np.mean(item2 - item1))

# Prediccion para cada equipo-temporada
df["Porcentaje_Victorias_Predicho"] = df["Salario_Relativo"] + b
df["Error_Prediccion"] = df["Porcentaje_Victorias"] - df["Porcentaje_Victorias_Predicho"]

df.to_csv("salida_slopeone.csv", index=False, sep=";", decimal=",")
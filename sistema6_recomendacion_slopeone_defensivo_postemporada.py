"""
Sistema 6 -- FactRendimientoDefensivo x FactRendimientoDefensivoPostemporada
(repite el metodo del sistema 3: Slope One), consumiendo el drill-across
generado por extraer_defensivo_postemporada.py (que a su vez consume
DimJugador para el nombre de cada jugador).

Por que Slope One: igual que en sistema3 (Item1 = salario relativo del
equipo, Item2 = % de victorias del equipo), aqui Item1 = % de fildeo en
temporada regular e Item2 = % de fildeo en postemporada, del MISMO jugador.
Slope One ajusta f(x) = x + b para predecir el Item2 a partir del Item1 y
detectar, por el signo y tamano del error, que jugadores mejoran, mantienen
o empeoran su eficiencia defensiva bajo la presion de playoffs -- la
pregunta de negocio de la carta de diseno sobre el efecto de la
postemporada en el rendimiento.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("DefensivoPostemporada.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Porcentaje_Fildeo_Regular",
           "Porcentaje_Fildeo_Postemporada"], encoding="utf-8-sig")

# Item1 = % de fildeo regular, Item2 = % de fildeo en postemporada (ya en la
# misma escala 0-1, no requiere percentil como en sistema3)
item1 = df["Porcentaje_Fildeo_Regular"].values
item2 = df["Porcentaje_Fildeo_Postemporada"].values
b = float(np.mean(item2 - item1))

df["Porcentaje_Fildeo_Postemporada_Predicho"] = df["Porcentaje_Fildeo_Regular"] + b
df["Error_Prediccion"] = df["Porcentaje_Fildeo_Postemporada"] - df["Porcentaje_Fildeo_Postemporada_Predicho"]

df.to_csv("salida_slopeone_defensivo_postemporada.csv", index=False, sep=";", decimal=",")

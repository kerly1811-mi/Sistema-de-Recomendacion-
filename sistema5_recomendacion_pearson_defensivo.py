"""
Sistema 5 -- FactRendimientoDefensivo (repite el metodo del sistema 1:
Correlacion de Pearson), consumiendo datos de FactRendimientoDefensivo
unidos a DimJugador, DimTiempo y DimPosicion (ver extraer_defensivo.py).

Por que Pearson: igual que en sistema1 (OPS por temporada de carrera),
FactRendimientoDefensivo tiene una SERIE por jugador a lo largo de las
temporadas (Porcentaje_Fildeo). Pearson permite encontrar jugadores cuya
CURVA de eficiencia defensiva a lo largo de la carrera tiene una forma
parecida (ambos mejoran con la experiencia, ambos son estables, ambos
declinan), respondiendo la pregunta de negocio de la carta de diseno sobre
como varia la eficiencia defensiva con la edad/experiencia del jugador.
"""
import pandas as pd
import numpy as np

RUTA = "Defensivo.csv"
TOP_N = 5
MIN_TEMPORADAS_COMUNES = 8

# ETAPA 1: Carga (generado por extraer_defensivo.py desde FactRendimientoDefensivo
# + DimJugador + DimTiempo + DimPosicion)
df = pd.read_csv(RUTA, sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "Porcentaje_Fildeo",
           "Posicion_Principal", "Jugadas_Totales"], encoding="utf-8-sig")

# ETAPA 2: Anio calendario -> Temporada N de carrera
df = df.sort_values(["Id_Jugador", "Anio"])
df["Temporada_Numero"] = df.groupby("Id_Jugador").cumcount() + 1

# ETAPA 3: Matriz Jugador x Temporada_Numero
pivot = df.pivot_table(index="Nombre_Completo", columns="Temporada_Numero",
                        values="Porcentaje_Fildeo")

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
resultado.to_csv("salida_pearson_defensivo.csv", index=False, sep=";", decimal=",")

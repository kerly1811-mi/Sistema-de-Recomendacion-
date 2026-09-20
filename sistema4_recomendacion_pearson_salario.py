"""
Sistema 4 -- FactSalario (repite el metodo del sistema 1: Correlacion de
Pearson), consumiendo datos de FactSalario unidos a DimJugador y DimTiempo
(ver extraer_salario.py).

Por que Pearson y no Coseno o Slope One para FactSalario:
- Coseno (sistema 2) compara un PERFIL de varios atributos en un mismo
  instante (postemporada de un solo anio); FactSalario no tiene esa forma,
  tiene una SERIE por jugador a lo largo de temporadas.
- Slope One (sistema 3) resuelve una relacion bivariante agregada a nivel
  equipo-temporada (salario vs victorias); no sirve para comparar jugadores
  entre si.
- Pearson (sistema 1) ya esta pensado exactamente para esto: comparar la
  FORMA de una trayectoria de un jugador a lo largo de su carrera con la de
  otro. Se reutiliza el mismo enfoque que en sistema1 (OPS por temporada de
  carrera) mas el Indice_Eficiencia_Salarial por temporada de carrera, para
  encontrar jugadores con una evolucion de eficiencia salarial parecida
  (ambos suben, ambos caen, ambos son estables, etc.), que es exactamente la
  pregunta de negocio de la carta de diseno (seccion 2, pregunta 4).
"""
import pandas as pd
import numpy as np

RUTA = "Salario.csv"
TOP_N = 5
MIN_TEMPORADAS_COMUNES = 5  # el indice tiene mas huecos que el OPS, se relaja el minimo

# ETAPA 1: Carga (generado por extraer_salario.py desde FactSalario + DimJugador + DimTiempo)
df = pd.read_csv(RUTA, sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "Indice_Eficiencia_Salarial",
           "Salario_Anual", "Edad_Jugador", "Anios_Experiencia"], encoding="utf-8-sig")

# ETAPA 2: Anio calendario -> Temporada N de carrera (para comparar entre epocas,
# igual que en sistema1)
df = df.sort_values(["Id_Jugador", "Anio"])
df["Temporada_Numero"] = df.groupby("Id_Jugador").cumcount() + 1

# ETAPA 3: Matriz Jugador x Temporada_Numero (valores = indice de eficiencia salarial)
pivot = df.pivot_table(index="Nombre_Completo", columns="Temporada_Numero",
                        values="Indice_Eficiencia_Salarial")

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
resultado.to_csv("salida_pearson_salario.csv", index=False, sep=";", decimal=",")

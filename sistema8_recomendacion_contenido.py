"""
Sistema 8 - Recomendacion Basada en Contenido
FactRendimientoDefensivo + DimPosicion + DimTiempo

A diferencia de los sistemas 1-7 (que comparan trayectorias o perfiles
numericos completos con Pearson o similitud coseno), este sistema NO
calcula ninguna correlacion ni distancia entre vectores. Es un
recomendador basado en contenido clasico: construye para cada jugador
una "ficha" de atributos descriptivos (contenido) y recomienda a los
jugadores cuya ficha comparte mas atributos con la del jugador base,
exactamente como recomendar peliculas por genero/decada/reparto en
lugar de por el comportamiento de otros usuarios.

Atributos de contenido usados (todos categoricos, derivados de datos
reales, ninguno inventado):
  1. Posicion_Principal  -> ya viene en Defensivo.csv (DimPosicion)
  2. Decada_Principal    -> se deriva agrupando el Anio en decadas
                             (rol descriptivo: en que epoca jugo)
  3. Nivel_Fildeo        -> se deriva discretizando el promedio de
                             carrera de Porcentaje_Fildeo en 4 niveles
                             (Elite/Alto/Medio/Bajo) calculados con
                             cuartiles reales de la propia base, no con
                             umbrales arbitrarios

La recomendacion para un jugador es: de todos los jugadores que
comparten su misma posicion principal, ordenar por cuantos atributos
de contenido adicionales (decada, nivel) tambien coinciden, es decir
un puntaje de coincidencia de contenido (Jaccard sobre 3 etiquetas),
no una metrica de correlacion.
"""
import pandas as pd
import numpy as np

RUTA = "Defensivo.csv"

df = pd.read_csv(RUTA, sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "Porcentaje_Fildeo",
           "Posicion_Principal", "Jugadas_Totales"], encoding="utf-8-sig")

# ------------------------------------------------------------------
# 1. Ficha de contenido por jugador (se agrega toda la carrera a una
#    sola fila por jugador: esto es el "perfil de contenido" del item)
# ------------------------------------------------------------------


def decada_de(anio):
    return f"{(anio // 10) * 10}s"


df["Decada"] = df["Anio"].apply(decada_de)

ficha = df.groupby(["Id_Jugador", "Nombre_Completo"]).apply(
    lambda g: pd.Series({
        "Posicion_Principal": g.loc[g["Jugadas_Totales"].idxmax(), "Posicion_Principal"],
        "Decada_Principal": g.groupby("Decada")["Jugadas_Totales"].sum().idxmax(),
        "Fildeo_Promedio_Carrera": np.average(g["Porcentaje_Fildeo"], weights=g["Jugadas_Totales"]),
        "Jugadas_Totales_Carrera": g["Jugadas_Totales"].sum(),
        "Temporadas_Jugadas": g["Anio"].nunique(),
    }), include_groups=False
).reset_index()

# Solo se conservan jugadores con carrera minimamente representativa
# (se filtra la informacion, tal como exige el punto 4 de la guia):
# menos de 3 temporadas o menos de 300 jugadas totales no alcanza para
# describir de forma confiable un "estilo" de jugador.
ficha = ficha[(ficha["Temporadas_Jugadas"] >= 3) & (ficha["Jugadas_Totales_Carrera"] >= 300)].reset_index(drop=True)

# Nivel de fildeo discretizado en 4 niveles por CUARTILES REALES,
# calculados por separado dentro de cada posicion (un Catcher y un
# Pitcher no fildean en la misma escala, asi que "Elite" significa
# "de las mejores de su propia posicion", no un numero fijo global).
etiquetas = ["Bajo", "Medio", "Alto", "Elite"]


def etiquetar_nivel(serie):
    try:
        return pd.qcut(serie, 4, labels=etiquetas, duplicates="drop")
    except ValueError:
        return pd.Series(["Medio"] * len(serie), index=serie.index)


ficha["Nivel_Fildeo"] = (
    ficha.groupby("Posicion_Principal")["Fildeo_Promedio_Carrera"]
    .transform(etiquetar_nivel)
    .astype(str)
)
ficha = ficha.reset_index(drop=True)

# ------------------------------------------------------------------
# 2. Recomendacion basada en contenido: para cada jugador, se buscan
#    SOLO jugadores de su misma posicion (filtro obligatorio, porque el
#    contenido "posicion" define el tipo de item) y se ordenan por
#    cuantas etiquetas de contenido adicionales (decada, nivel)
#    tambien coinciden.
# ------------------------------------------------------------------

resultados = []
por_posicion = {pos: g for pos, g in ficha.groupby("Posicion_Principal")}

for pos, grupo in por_posicion.items():
    grupo = grupo.reset_index(drop=True)
    for i, base in grupo.iterrows():
        coincidencias = grupo.apply(
            lambda fila: (fila["Decada_Principal"] == base["Decada_Principal"]) +
                         (fila["Nivel_Fildeo"] == base["Nivel_Fildeo"]),
            axis=1
        )
        puntaje = (1 + coincidencias) / 3.0  # posicion siempre coincide (1) + hasta 2 etiquetas mas, sobre 3 en total
        candidatos = grupo.assign(Puntaje_Contenido=puntaje)
        candidatos = candidatos[candidatos["Id_Jugador"] != base["Id_Jugador"]]
        candidatos = candidatos.sort_values(
            ["Puntaje_Contenido", "Jugadas_Totales_Carrera"], ascending=[False, False]
        ).head(5)
        for rank, (_, comp) in enumerate(candidatos.iterrows(), start=1):
            resultados.append({
                "Jugador_Base": base["Nombre_Completo"],
                "Ranking": rank,
                "Jugador_Comparable": comp["Nombre_Completo"],
                "Puntaje_Contenido": round(comp["Puntaje_Contenido"], 4),
            })

salida = pd.DataFrame(resultados)
salida.to_csv("salida_contenido.csv", index=False, header=False, sep=";", decimal=",", encoding="utf-8-sig")
ficha.to_csv("fichas_contenido.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")

print(f"salida_contenido.csv generado: {len(salida)} filas ({ficha.shape[0]} jugadores con ficha de contenido)")
print(ficha["Posicion_Principal"].value_counts())
print(salida.head(10))
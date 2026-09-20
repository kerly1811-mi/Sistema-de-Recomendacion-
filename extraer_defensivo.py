"""
Extrae de LahmanDW_Kim (FactRendimientoDefensivo + DimJugador + DimTiempo +
DimPosicion) el insumo para el sistema 5 (Pearson sobre porcentaje de fildeo).

Un jugador puede tener varios tramos y varias posiciones en el mismo anio
(cambio de equipo o de posicion a mitad de temporada); se agregan a nivel
jugador-anio ponderando el porcentaje de fildeo por el volumen real de
jugadas (Outs_Realizados + Asistencias + Errores), y se conserva la posicion
mas jugada ese anio (DimPosicion) solo como dato informativo.
"""
import pyodbc
import pandas as pd
import numpy as np

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;DATABASE=LahmanDW_Kim;"
    "Trusted_Connection=yes;TrustServerCertificate=yes;"
)

QUERY = """
SELECT
    dj.Id_Jugador,
    CONCAT(dj.Nombre, ' ', dj.Apellido) AS Nombre_Completo,
    dt.Anio,
    dp.Nombre_Posicion_Descriptivo AS Posicion,
    fd.Outs_Realizados,
    fd.Asistencias,
    fd.Errores,
    fd.Jugadas_Dobles,
    fd.Porcentaje_Fildeo_Calculado
FROM dbo.FactRendimientoDefensivo fd
JOIN dbo.DimJugador  dj ON dj.Id_Jugador  = fd.Id_Jugador
JOIN dbo.DimTiempo   dt ON dt.Id_Tiempo   = fd.Id_Tiempo
JOIN dbo.DimPosicion dp ON dp.Id_Posicion = fd.Id_Posicion
WHERE fd.Porcentaje_Fildeo_Calculado IS NOT NULL
ORDER BY dj.Id_Jugador, dt.Anio;
"""

conn = pyodbc.connect(CONN_STR)
df = pd.read_sql(QUERY, conn)
conn.close()

df["Jugadas_Totales"] = df["Outs_Realizados"] + df["Asistencias"] + df["Errores"]


def agregar_anio(g):
    peso = g["Jugadas_Totales"].to_numpy()
    valor = g["Porcentaje_Fildeo_Calculado"].astype(float).to_numpy()
    pct_ponderado = np.average(valor, weights=peso) if peso.sum() > 0 else valor.mean()
    posicion_principal = g.loc[g["Jugadas_Totales"].idxmax(), "Posicion"]
    return pd.Series({
        "Porcentaje_Fildeo": round(pct_ponderado, 4),
        "Posicion_Principal": posicion_principal,
        "Jugadas_Totales": int(peso.sum()),
    })


agregado = (
    df.groupby(["Id_Jugador", "Nombre_Completo", "Anio"])
    .apply(agregar_anio, include_groups=False)
    .reset_index()
)
agregado = agregado.sort_values(["Id_Jugador", "Anio"])

agregado.to_csv(
    "Defensivo.csv", index=False, header=False, sep=";", decimal=",", encoding="utf-8-sig"
)
print(f"Defensivo.csv generado: {len(agregado)} filas, {agregado['Id_Jugador'].nunique()} jugadores")

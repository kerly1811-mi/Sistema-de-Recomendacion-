"""
Extrae de LahmanDW_Kim el insumo para el sistema 6 (Slope One: fildeo
regular vs fildeo en postemporada).

Es un drill-across explicito entre FactRendimientoDefensivo y
FactRendimientoDefensivoPostemporada (permitido por la carta de diseno,
seccion 11: "los cruces se resuelven agregando primero y relacionando
despues"; ninguna tabla de hechos se une a otra directamente en el DW, la
union ocurre aqui, en la capa analitica de Python, sobre datos ya
agregados). Cada tabla se agrega primero a un solo valor por jugador
(promedio de carrera de temporada regular y de postemporada), y luego se
relacionan por Id_Jugador.
"""
import pyodbc
import pandas as pd

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;DATABASE=LahmanDW_Kim;"
    "Trusted_Connection=yes;TrustServerCertificate=yes;"
)

QUERY_REGULAR = """
SELECT dj.Id_Jugador, CONCAT(dj.Nombre, ' ', dj.Apellido) AS Nombre_Completo,
       fd.Porcentaje_Fildeo_Calculado AS Porcentaje_Fildeo,
       (fd.Outs_Realizados + fd.Asistencias + fd.Errores) AS Jugadas
FROM dbo.FactRendimientoDefensivo fd
JOIN dbo.DimJugador dj ON dj.Id_Jugador = fd.Id_Jugador
WHERE fd.Porcentaje_Fildeo_Calculado IS NOT NULL;
"""

QUERY_POST = """
SELECT dj.Id_Jugador, CONCAT(dj.Nombre, ' ', dj.Apellido) AS Nombre_Completo,
       fp.Porcentaje_Fildeo_Postemporada AS Porcentaje_Fildeo_Post,
       (fp.Putouts + fp.Asistencias + fp.Errores) AS Jugadas_Post
FROM dbo.FactRendimientoDefensivoPostemporada fp
JOIN dbo.DimJugador dj ON dj.Id_Jugador = fp.Id_Jugador
WHERE fp.Porcentaje_Fildeo_Postemporada IS NOT NULL;
"""

conn = pyodbc.connect(CONN_STR)
regular = pd.read_sql(QUERY_REGULAR, conn)
post = pd.read_sql(QUERY_POST, conn)
conn.close()


def promedio_ponderado(g, col_valor, col_peso):
    peso = g[col_peso].to_numpy()
    valor = g[col_valor].astype(float).to_numpy()
    if peso.sum() == 0:
        return valor.mean()
    return (valor * peso).sum() / peso.sum()


agg_regular = regular.groupby(["Id_Jugador", "Nombre_Completo"]).apply(
    lambda g: promedio_ponderado(g, "Porcentaje_Fildeo", "Jugadas"), include_groups=False
).reset_index(name="Porcentaje_Fildeo_Regular")

agg_post = post.groupby(["Id_Jugador", "Nombre_Completo"]).apply(
    lambda g: promedio_ponderado(g, "Porcentaje_Fildeo_Post", "Jugadas_Post"), include_groups=False
).reset_index(name="Porcentaje_Fildeo_Postemporada")

union = pd.merge(agg_regular, agg_post[["Id_Jugador", "Porcentaje_Fildeo_Postemporada"]],
                  on="Id_Jugador", how="inner")
union["Porcentaje_Fildeo_Regular"] = union["Porcentaje_Fildeo_Regular"].round(4)
union["Porcentaje_Fildeo_Postemporada"] = union["Porcentaje_Fildeo_Postemporada"].round(4)
union = union.sort_values("Id_Jugador")

union.to_csv("DefensivoPostemporada.csv", index=False, header=False, sep=";", decimal=",",
             encoding="utf-8-sig")
print(f"DefensivoPostemporada.csv generado: {len(union)} jugadores con ambos datos")

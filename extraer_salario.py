"""
Extrae de LahmanDW_Kim (FactSalario + dimensiones DimJugador y DimTiempo) el
insumo para el sistema 4 (Pearson sobre eficiencia salarial).

Se usa Indice_Eficiencia_Salarial (OPS normalizado / salario en millones,
calculado ya en el ETL) en vez del salario crudo: el salario crudo solo
refleja inflacion y antiguedad de la liga, mientras que el indice es la
metrica de negocio real (rendimiento por dolar invertido).

Varios jugadores tienen mas de un tramo (stint) en el mismo anio porque
cambiaron de equipo a mitad de temporada (grano jugador-temporada-equipo-
tramo de FactSalario); se promedian los tramos para obtener un valor por
jugador-temporada, que es el grano que necesita la matriz de Pearson.
"""
import pyodbc
import pandas as pd

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
    fs.Indice_Eficiencia_Salarial,
    fs.Salario_Anual,
    fs.Edad_Jugador,
    fs.Anios_Experiencia
FROM dbo.FactSalario fs
JOIN dbo.DimJugador dj ON dj.Id_Jugador = fs.Id_Jugador
JOIN dbo.DimTiempo  dt ON dt.Id_Tiempo  = fs.Id_Tiempo
WHERE fs.Indice_Eficiencia_Salarial IS NOT NULL
ORDER BY dj.Id_Jugador, dt.Anio;
"""

conn = pyodbc.connect(CONN_STR)
df = pd.read_sql(QUERY, conn)
conn.close()

# Colapsa tramos (mismo jugador-anio con varios equipos) a un solo registro
df = df.groupby(["Id_Jugador", "Nombre_Completo", "Anio"], as_index=False).agg(
    Indice_Eficiencia_Salarial=("Indice_Eficiencia_Salarial", "mean"),
    Salario_Anual=("Salario_Anual", "sum"),
    Edad_Jugador=("Edad_Jugador", "max"),
    Anios_Experiencia=("Anios_Experiencia", "max"),
)

df = df.sort_values(["Id_Jugador", "Anio"])

df.to_csv(
    "Salario.csv", index=False, header=False, sep=";", decimal=",", encoding="utf-8-sig"
)
print(f"Salario.csv generado: {len(df)} filas, {df['Id_Jugador'].nunique()} jugadores")

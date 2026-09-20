"""
Extrae de LahmanDW_Kim (FactReconocimiento + DimJugador + DimTiempo +
DimTipoReconocimiento) el insumo para el sistema 7 (Similitud de Coseno
sobre el perfil de reconocimientos acumulados por jugador).

Se agrega la Cantidad de cada tipo de reconocimiento por jugador a lo largo
de TODA su carrera (no por temporada), porque lo que se compara aqui es el
PERFIL de carrera completo (cuantas veces gano cada tipo de premio /
seleccion All-Star), no una evolucion temporada a temporada.
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
    tr.Nombre_Reconocimiento_Descriptivo AS Tipo_Reconocimiento,
    fr.Cantidad,
    MIN(dt.Anio) OVER (PARTITION BY fr.Id_Jugador) AS Primer_Anio,
    MAX(dt.Anio) OVER (PARTITION BY fr.Id_Jugador) AS Ultimo_Anio
FROM dbo.FactReconocimiento fr
JOIN dbo.DimJugador dj           ON dj.Id_Jugador = fr.Id_Jugador
JOIN dbo.DimTiempo dt            ON dt.Id_Tiempo = fr.Id_Tiempo
JOIN dbo.DimTipoReconocimiento tr ON tr.Id_TipoReconocimiento = fr.Id_TipoReconocimiento;
"""

conn = pyodbc.connect(CONN_STR)
df = pd.read_sql(QUERY, conn)
conn.close()

# ETAPA 1: agregar a nivel jugador-tipo (suma de Cantidad en toda la carrera)
perfil = df.groupby(["Id_Jugador", "Nombre_Completo", "Tipo_Reconocimiento"],
                     as_index=False)["Cantidad"].sum()

# ETAPA 2: solo tipos con presencia real (>=5 jugadores), para no tener un
# vector de 85 columnas casi todas en cero por reconocimientos unicos/raros
tipos_frecuentes = perfil.groupby("Tipo_Reconocimiento")["Id_Jugador"].nunique()
tipos_frecuentes = tipos_frecuentes[tipos_frecuentes >= 5].index
perfil = perfil[perfil["Tipo_Reconocimiento"].isin(tipos_frecuentes)]

total_por_jugador = perfil.groupby(["Id_Jugador", "Nombre_Completo"], as_index=False)["Cantidad"].sum()
total_por_jugador = total_por_jugador.rename(columns={"Cantidad": "Total_Reconocimientos"})

perfil = perfil.merge(total_por_jugador, on=["Id_Jugador", "Nombre_Completo"])
perfil = perfil.sort_values(["Id_Jugador", "Tipo_Reconocimiento"])

perfil.to_csv("Reconocimiento.csv", index=False, header=False, sep=";", decimal=",",
              encoding="utf-8-sig")
print(f"Reconocimiento.csv generado: {perfil['Id_Jugador'].nunique()} jugadores, "
      f"{len(tipos_frecuentes)} tipos de reconocimiento frecuentes")

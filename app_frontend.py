import unicodedata
from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def normalizar(texto):
    """Quita tildes, pasa a minusculas y recorta espacios, para poder
    buscar sin importar mayusculas/acentos ni coincidencia exacta."""
    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


# ============================================================
# CARGA DE DATOS Y CALCULO (se hace una vez, al arrancar el servidor)
# ============================================================

# ---- 1. Pearson ----
pearson_df = pd.read_csv("Pearson.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "OPS"], encoding="utf-8-sig")
pearson_df = pearson_df.sort_values(["Id_Jugador", "Anio"])
pearson_df["Temporada_Numero"] = pearson_df.groupby("Id_Jugador").cumcount() + 1

# Varios jugadores distintos comparten el mismo nombre (83 casos): se usa
# Id_Jugador como clave real y se agrega "(ID ...)" solo a los nombres
# duplicados, para no mezclar carreras de personas distintas.
_id_a_nombre_pearson = pearson_df.drop_duplicates("Id_Jugador").set_index("Id_Jugador")["Nombre_Completo"]
_dup_pearson = _id_a_nombre_pearson[_id_a_nombre_pearson.duplicated(keep=False)]
pearson_display = {
    idj: (f"{nom} (ID {idj})" if idj in _dup_pearson.index else nom)
    for idj, nom in _id_a_nombre_pearson.items()
}

pearson_pivot = pearson_df.pivot_table(index="Id_Jugador", columns="Temporada_Numero", values="OPS")
pearson_corr = pearson_pivot.T.corr(min_periods=8)

pearson_lista = sorted(pearson_display.items(), key=lambda x: x[1])  # [(id, nombre_mostrado), ...]
pearson_norm_a_id = {normalizar(nom): idj for idj, nom in pearson_display.items()}

# ---- 2. Coseno ----
coseno_df = pd.read_csv("similitud coseno.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Turnos_Al_Bate", "Imparables",
           "Cuadrangulares", "Carreras_Impulsadas", "Bases_Por_Bolas", "OPS_Postemporada"],
    encoding="utf-8-sig", na_values=["NULL"]).dropna(subset=["OPS_Postemporada", "Nombre_Completo"]).reset_index(drop=True)
coseno_df["Nombre_Completo"] = coseno_df["Nombre_Completo"].astype(str)

_dup_coseno = coseno_df["Nombre_Completo"][coseno_df["Nombre_Completo"].duplicated(keep=False)].unique()
coseno_df["Nombre_Mostrado"] = coseno_df.apply(
    lambda r: f'{r["Nombre_Completo"]} (ID {r["Id_Jugador"]})' if r["Nombre_Completo"] in _dup_coseno else r["Nombre_Completo"],
    axis=1,
)

atributos_coseno = ["Turnos_Al_Bate", "Imparables", "Cuadrangulares", "Carreras_Impulsadas", "Bases_Por_Bolas", "OPS_Postemporada"]
X_std = StandardScaler().fit_transform(coseno_df[atributos_coseno].values)
coseno_sim = cosine_similarity(X_std)
np.fill_diagonal(coseno_sim, -1)

coseno_lista = sorted(coseno_df["Nombre_Mostrado"].tolist())
coseno_norm_a_idx = {normalizar(n): i for i, n in enumerate(coseno_df["Nombre_Mostrado"])}

# ---- 3. Slope One ----
slope_df = pd.read_csv("Slope One.csv", sep=";", decimal=".", header=None,
    names=["Id_Equipo", "Nombre_Equipo", "Anio", "Salario_Promedio_Equipo", "Porcentaje_Victorias"],
    encoding="utf-8-sig")
slope_df["Salario_Relativo"] = slope_df.groupby("Anio")["Salario_Promedio_Equipo"].rank(pct=True)
b = float((slope_df["Porcentaje_Victorias"] - slope_df["Salario_Relativo"]).mean())
slope_df["Porcentaje_Victorias_Predicho"] = slope_df["Salario_Relativo"] + b
slope_df["Error"] = slope_df["Porcentaje_Victorias"] - slope_df["Porcentaje_Victorias_Predicho"]

slope_equipos = sorted(slope_df["Nombre_Equipo"].unique().tolist())
slope_norm_a_nombre = {normalizar(e): e for e in slope_equipos}

# ---- 4. Salario (Pearson, FactSalario) ----
salario_df = pd.read_csv("Salario.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "Indice_Eficiencia_Salarial",
           "Salario_Anual", "Edad_Jugador", "Anios_Experiencia"], encoding="utf-8-sig")
salario_df = salario_df.sort_values(["Id_Jugador", "Anio"])
salario_df["Temporada_Numero"] = salario_df.groupby("Id_Jugador").cumcount() + 1

_id_a_nombre_salario = salario_df.drop_duplicates("Id_Jugador").set_index("Id_Jugador")["Nombre_Completo"]
_dup_salario = _id_a_nombre_salario[_id_a_nombre_salario.duplicated(keep=False)]
salario_display = {
    idj: (f"{nom} (ID {idj})" if idj in _dup_salario.index else nom)
    for idj, nom in _id_a_nombre_salario.items()
}

salario_pivot = salario_df.pivot_table(index="Id_Jugador", columns="Temporada_Numero",
                                        values="Indice_Eficiencia_Salarial")
salario_corr = salario_pivot.T.corr(min_periods=5)

salario_lista = sorted(salario_display.items(), key=lambda x: x[1])  # [(id, nombre_mostrado), ...]
salario_norm_a_id = {normalizar(nom): idj for idj, nom in salario_display.items()}

# ---- 5. Defensivo (Pearson, FactRendimientoDefensivo) ----
defensivo_df = pd.read_csv("Defensivo.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "Porcentaje_Fildeo",
           "Posicion_Principal", "Jugadas_Totales"], encoding="utf-8-sig")
defensivo_df = defensivo_df.sort_values(["Id_Jugador", "Anio"])
defensivo_df["Temporada_Numero"] = defensivo_df.groupby("Id_Jugador").cumcount() + 1

_id_a_nombre_defensivo = defensivo_df.drop_duplicates("Id_Jugador").set_index("Id_Jugador")["Nombre_Completo"]
_dup_defensivo = _id_a_nombre_defensivo[_id_a_nombre_defensivo.duplicated(keep=False)]
defensivo_display = {
    idj: (f"{nom} (ID {idj})" if idj in _dup_defensivo.index else nom)
    for idj, nom in _id_a_nombre_defensivo.items()
}

defensivo_pivot = defensivo_df.pivot_table(index="Id_Jugador", columns="Temporada_Numero",
                                            values="Porcentaje_Fildeo")
defensivo_corr = defensivo_pivot.T.corr(min_periods=8)

defensivo_lista = sorted(defensivo_display.items(), key=lambda x: x[1])
defensivo_norm_a_id = {normalizar(nom): idj for idj, nom in defensivo_display.items()}

_posicion_principal_defensivo = (
    defensivo_df.sort_values("Jugadas_Totales", ascending=False)
    .drop_duplicates("Id_Jugador")
    .set_index("Id_Jugador")["Posicion_Principal"]
)

# ---- 6. Defensivo Postemporada (Slope One, FactRendimientoDefensivo x FactRendimientoDefensivoPostemporada) ----
defpost_df = pd.read_csv("DefensivoPostemporada.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Porcentaje_Fildeo_Regular",
           "Porcentaje_Fildeo_Postemporada"], encoding="utf-8-sig")
_b_defpost = float((defpost_df["Porcentaje_Fildeo_Postemporada"] - defpost_df["Porcentaje_Fildeo_Regular"]).mean())
defpost_df["Porcentaje_Fildeo_Postemporada_Predicho"] = defpost_df["Porcentaje_Fildeo_Regular"] + _b_defpost
defpost_df["Error"] = defpost_df["Porcentaje_Fildeo_Postemporada"] - defpost_df["Porcentaje_Fildeo_Postemporada_Predicho"]

_dup_defpost = defpost_df["Nombre_Completo"][defpost_df["Nombre_Completo"].duplicated(keep=False)].unique()
defpost_df["Nombre_Mostrado"] = defpost_df.apply(
    lambda r: f'{r["Nombre_Completo"]} (ID {r["Id_Jugador"]})' if r["Nombre_Completo"] in _dup_defpost else r["Nombre_Completo"],
    axis=1,
)
defpost_lista = sorted(defpost_df["Nombre_Mostrado"].tolist())
defpost_norm_a_nombre = {normalizar(n): n for n in defpost_lista}

# ---- 7. Reconocimiento (Coseno, FactReconocimiento) ----
reconocimiento_df = pd.read_csv("Reconocimiento.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Tipo_Reconocimiento", "Cantidad",
           "Total_Reconocimientos"], encoding="utf-8-sig")

reconocimiento_matriz = reconocimiento_df.pivot_table(
    index=["Id_Jugador", "Nombre_Completo"], columns="Tipo_Reconocimiento",
    values="Cantidad", aggfunc="sum", fill_value=0,
)
_reconocimiento_X = StandardScaler().fit_transform(reconocimiento_matriz.values)
reconocimiento_sim = cosine_similarity(_reconocimiento_X)
np.fill_diagonal(reconocimiento_sim, -1)

_dup_reconocimiento = reconocimiento_matriz.index.get_level_values("Nombre_Completo")
_dup_reconocimiento = pd.Series(_dup_reconocimiento)[pd.Series(_dup_reconocimiento).duplicated(keep=False)].unique()
reconocimiento_nombres = [
    f"{nom} (ID {idj})" if nom in _dup_reconocimiento else nom
    for idj, nom in reconocimiento_matriz.index
]
reconocimiento_lista = sorted(reconocimiento_nombres)
reconocimiento_norm_a_idx = {normalizar(n): i for i, n in enumerate(reconocimiento_nombres)}
reconocimiento_tipos_por_jugador = {
    i: reconocimiento_matriz.iloc[i][reconocimiento_matriz.iloc[i] > 0].to_dict()
    for i in range(len(reconocimiento_matriz))
}


# ---- 8. Contenido (Recomendacion Basada en Contenido, FactRendimientoDefensivo) ----
# A diferencia de los sistemas 1-7, aqui no se calcula ninguna correlacion
# ni similitud coseno entre vectores numericos. Se construye, para cada
# jugador, una ficha de atributos de contenido (Posicion_Principal,
# Decada_Principal, Nivel_Fildeo) y se recomienda por coincidencia de
# esas etiquetas, tal como un sistema de contenido recomienda peliculas
# por genero/decada en vez de por el comportamiento de otros usuarios.
contenido_base_df = pd.read_csv("Defensivo.csv", sep=";", decimal=",", header=None,
    names=["Id_Jugador", "Nombre_Completo", "Anio", "Porcentaje_Fildeo",
           "Posicion_Principal", "Jugadas_Totales"], encoding="utf-8-sig")
contenido_base_df["Decada"] = (contenido_base_df["Anio"] // 10 * 10).astype(str) + "s"


def _ficha_jugador(g):
    return pd.Series({
        "Posicion_Principal": g.loc[g["Jugadas_Totales"].idxmax(), "Posicion_Principal"],
        "Decada_Principal": g.groupby("Decada")["Jugadas_Totales"].sum().idxmax(),
        "Fildeo_Promedio_Carrera": np.average(g["Porcentaje_Fildeo"], weights=g["Jugadas_Totales"]),
        "Jugadas_Totales_Carrera": g["Jugadas_Totales"].sum(),
        "Temporadas_Jugadas": g["Anio"].nunique(),
    })


contenido_ficha = contenido_base_df.groupby(["Id_Jugador", "Nombre_Completo"]).apply(
    _ficha_jugador, include_groups=False
).reset_index()

# Se descartan carreras demasiado cortas para describir un "estilo"
# confiable (filtro de datos, igual que el minimo de temporadas de los
# sistemas 1, 4 y 5).
contenido_ficha = contenido_ficha[
    (contenido_ficha["Temporadas_Jugadas"] >= 3) & (contenido_ficha["Jugadas_Totales_Carrera"] >= 300)
].reset_index(drop=True)

_niveles = ["Bajo", "Medio", "Alto", "Elite"]


def _etiquetar_nivel(serie):
    try:
        return pd.qcut(serie, 4, labels=_niveles, duplicates="drop")
    except ValueError:
        return pd.Series(["Medio"] * len(serie), index=serie.index)


contenido_ficha["Nivel_Fildeo"] = (
    contenido_ficha.groupby("Posicion_Principal")["Fildeo_Promedio_Carrera"]
    .transform(_etiquetar_nivel).astype(str)
)

_dup_contenido = contenido_ficha["Nombre_Completo"][contenido_ficha["Nombre_Completo"].duplicated(keep=False)].unique()
contenido_ficha["Nombre_Mostrado"] = contenido_ficha.apply(
    lambda r: f'{r["Nombre_Completo"]} (ID {r["Id_Jugador"]})' if r["Nombre_Completo"] in _dup_contenido else r["Nombre_Completo"],
    axis=1,
)
contenido_lista = sorted(contenido_ficha["Nombre_Mostrado"].tolist())
contenido_norm_a_idx = {normalizar(n): i for i, n in enumerate(contenido_ficha["Nombre_Mostrado"])}


def recomendar_contenido(idx, top=5):
    base = contenido_ficha.iloc[idx]
    grupo = contenido_ficha[contenido_ficha["Posicion_Principal"] == base["Posicion_Principal"]]
    grupo = grupo[grupo["Id_Jugador"] != base["Id_Jugador"]].copy()
    coincidencias = (
        (grupo["Decada_Principal"] == base["Decada_Principal"]).astype(int) +
        (grupo["Nivel_Fildeo"] == base["Nivel_Fildeo"]).astype(int)
    )
    grupo["Puntaje_Contenido"] = (1 + coincidencias) / 3.0  # posicion (1) + hasta 2 etiquetas mas, sobre 3
    grupo = grupo.sort_values(["Puntaje_Contenido", "Jugadas_Totales_Carrera"], ascending=[False, False])
    return grupo.head(top)


def buscar(texto, norm_a_clave, opciones_mostradas):
    """Busca por coincidencia exacta (normalizada) y, si no hay,
    devuelve sugerencias por coincidencia parcial (contiene el texto)."""
    if not texto:
        return None, []
    t = normalizar(texto)
    if t in norm_a_clave:
        return norm_a_clave[t], []
    sugerencias = [o for o in opciones_mostradas if t in normalizar(o)][:8]
    return None, sugerencias


# ============================================================
# PLANTILLA HTML (estilo cuidado, sin librerias externas)
# ============================================================

# Cada sistema pertenece a una familia de metodo (para el color del tag y
# del borde de la tarjeta en el inicio).
METODOS = {
    "pearson":  {"color": "#1a56b0", "fondo": "#e8f0fe", "nombre": "Pearson"},
    "coseno":   {"color": "#6b21a8", "fondo": "#f3e8fd", "nombre": "Coseno"},
    "slopeone": {"color": "#0f7a4a", "fondo": "#e6f7ee", "nombre": "Slope One"},
    "contenido": {"color": "#b45309", "fondo": "#fef3e2", "nombre": "Contenido"},
}


def tag_metodo(metodo):
    m = METODOS[metodo]
    return f'<span class="tag" style="color:{m["color"]};background:{m["fondo"]}">{m["nombre"]}</span>'


def pagina(contenido, activo=""):
    def link(href, texto, clave):
        cls = "activo" if clave == activo else ""
        return f'<a class="{cls}" href="{href}">{texto}</a>'

    return f"""
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Sistemas de Recomendacion - Lahman</title>
      <style>
        * {{ box-sizing: border-box; }}
        body {{
          font-family: 'Segoe UI', Arial, sans-serif;
          max-width: 960px; margin: 0 auto; padding: 0 20px 50px;
          background: #f4f6f9; color: #1c2530;
        }}
        header {{
          background: linear-gradient(135deg, #16324f, #1d4a73);
          color: white; margin: 0 -20px 25px; padding: 22px 20px 16px;
        }}
        header h1 {{ margin: 0 0 4px; font-size: 20px; }}
        header p {{ margin: 0; opacity: .85; font-size: 13px; }}
        nav {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 4px 4px; }}
        nav a {{
          color: #cfe0f2; text-decoration: none; font-size: 12.8px;
          padding: 5px 10px; border-radius: 14px; border: 1px solid transparent;
        }}
        nav a.activo {{ background: rgba(255,255,255,.14); color: white; border-color: rgba(255,255,255,.25); }}
        nav a:hover {{ background: rgba(255,255,255,.09); color: white; }}
        h2 {{ font-size: 19px; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .subtitulo {{ color: #55606e; font-size: 13.5px; margin-top: 0; }}
        .tarjeta {{
          background: white; border: 1px solid #e1e6ec; border-radius: 10px;
          padding: 18px 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(20,30,50,.05);
        }}
        form {{ display: flex; gap: 8px; margin: 4px 0 0; }}
        input {{
          flex: 1; padding: 9px 12px; font-size: 14px;
          border: 1px solid #c7d0da; border-radius: 6px;
        }}
        input:focus {{ outline: none; border-color: #1f6feb; box-shadow: 0 0 0 3px rgba(31,111,235,.15); }}
        button {{
          padding: 9px 18px; font-size: 14px; border: none; border-radius: 6px;
          background: #1f6feb; color: white; cursor: pointer; font-weight: 600;
        }}
        button:hover {{ background: #185ec4; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13.5px; }}
        td, th {{ border-bottom: 1px solid #e6eaef; padding: 8px 10px; text-align: left; }}
        th {{ background: #eef2f7; color: #33404d; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
        tr:hover td {{ background: #f7fafd; }}
        .aviso {{ background: #fff6e5; border: 1px solid #f0d999; color: #7a5b00; padding: 10px 14px; border-radius: 6px; font-size: 13.5px; }}
        .error {{ background: #fdeaea; border: 1px solid #f0b8b8; color: #8a1f1f; padding: 10px 14px; border-radius: 6px; font-size: 13.5px; }}
        .sugerencias a {{ display: inline-block; background: #eef2f7; padding: 4px 10px; border-radius: 14px;
          margin: 3px 4px 0 0; font-size: 12.5px; color: #16324f; text-decoration: none; }}
        .sugerencias a:hover {{ background: #dbe6f2; }}
        .perfil {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 0; list-style: none; margin: 8px 0 0; }}
        .perfil li {{ background: #eef2f7; padding: 6px 12px; border-radius: 6px; font-size: 12.5px; }}
        .badge {{ display: inline-block; background: #eaf3ff; color: #1656a3; padding: 2px 8px; border-radius: 10px; font-weight: 600; }}
        .tag {{ display: inline-block; font-size: 11px; font-weight: 700; text-transform: uppercase;
          letter-spacing: .04em; padding: 3px 10px; border-radius: 12px; }}

        .grid-inicio {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-top: 14px; }}
        .card-sistema {{
          display: block; background: white; border: 1px solid #e1e6ec; border-left: 4px solid #c7d0da;
          border-radius: 8px; padding: 14px 16px; text-decoration: none; color: inherit;
          transition: transform .12s ease, box-shadow .12s ease;
        }}
        .card-sistema:hover {{ transform: translateY(-2px); box-shadow: 0 4px 10px rgba(20,30,50,.08); }}
        .card-sistema .num {{ font-size: 11px; color: #8b95a1; font-weight: 700; }}
        .card-sistema .titulo {{ font-size: 14.5px; font-weight: 600; margin: 3px 0 6px; }}
        .card-sistema .desc {{ font-size: 12.5px; color: #5b6674; line-height: 1.4; }}

        footer {{ color: #8b95a1; font-size: 11.5px; margin-top: 30px; }}

        @media (max-width: 560px) {{
          form {{ flex-direction: column; }}
          button {{ width: 100%; }}
        }}
      </style>
    </head>
    <body>
      <header>
        <h1>Sistemas de Recomendación </h1>
        
        <nav>
          {link("/", "Inicio", "inicio")}
          {link("/pearson", "1. Rend. Ofensivo", "pearson")}
          {link("/coseno", "2. Rend. Ofensivo Post.", "coseno")}
          {link("/slopeone", "3. Resultado Equipo", "slopeone")}
          {link("/salario", "4. Salario", "salario")}
          {link("/defensivo", "5. Defensivo", "defensivo")}
          {link("/defensivo_postemporada", "6. Defensivo Post.", "defpost")}
          {link("/reconocimiento", "7. Reconocimiento", "reconocimiento")}
          {link("/contenido", "8. Contenido", "contenido")}
        </nav>
      </header>
      {contenido}
      <footer>Datos: Lahman Baseball Database &middot; Modelo dimensional Kimball</footer>
    </body>
    </html>
    """


def tabla_recomendaciones(filas, col1, col2):
    if not filas:
        return '<p class="aviso">Sin comparables suficientes para este caso.</p>'
    filas_html = "".join(
        f'<tr><td>{i+1}</td><td>{nombre}</td><td><span class="badge">{valor:.4f}</span></td></tr>'
        for i, (nombre, valor) in enumerate(filas)
    )
    return f"""
    <table>
      <tr><th>#</th><th>{col1}</th><th>{col2}</th></tr>
      {filas_html}
    </table>
    """


def caja_sugerencias(sugerencias, ruta, parametro):
    if not sugerencias:
        return '<p class="error">No se encontro ninguna coincidencia.</p>'
    links = "".join(f'<a href="{ruta}?{parametro}={s}">{s}</a>' for s in sugerencias)
    return f'<p>Quizas quisiste decir:</p><div class="sugerencias">{links}</div>'


def tarjeta_inicio(href, numero, metodo, titulo, desc):
    m = METODOS[metodo]
    return f"""
    <a class="card-sistema" href="{href}" style="border-left-color:{m['color']}">
      <div class="num">SISTEMA {numero} &middot; {m['nombre'].upper()}</div>
      <div class="titulo">{titulo}</div>
      <div class="desc">{desc}</div>
    </a>
    """


@app.route("/")
def home():
    tarjetas = "".join([
        tarjeta_inicio("/pearson", 1, "pearson", "Rendimiento Ofensivo",
            "FactRendimientoOfensivo: correlacion de trayectoria de carrera (OPS por temporada)."),
        tarjeta_inicio("/coseno", 2, "coseno", "Rendimiento Ofensivo Postemporada",
            "FactRendimientoOfensivoPostemporada: similitud de perfil ofensivo en playoffs."),
        tarjeta_inicio("/slopeone", 3, "slopeone", "Resultado de Equipo",
            "FactResultadoEquipo: prediccion de % de victorias a partir del salario relativo."),
        tarjeta_inicio("/salario", 4, "pearson", "Salario",
            "FactSalario: correlacion de trayectoria de eficiencia salarial."),
        tarjeta_inicio("/defensivo", 5, "pearson", "Defensivo",
            "FactRendimientoDefensivo: correlacion de trayectoria de porcentaje de fildeo."),
        tarjeta_inicio("/defensivo_postemporada", 6, "slopeone", "Defensivo Postemporada",
            "FactRendimientoDefensivo x FactRendimientoDefensivoPostemporada: prediccion del fildeo en playoffs."),
        tarjeta_inicio("/reconocimiento", 7, "coseno", "Reconocimiento",
            "FactReconocimiento: similitud de perfil de premios y selecciones All-Star acumulados."),
        tarjeta_inicio("/contenido", 8, "contenido", "Recomendacion por Contenido",
            "FactRendimientoDefensivo: recomendacion por posicion, decada y nivel de fildeo (sin correlacion ni coseno)."),
    ])
    return pagina(f"""
        <div class="tarjeta">
          <h2>Bienvenida</h2>
          <p class="subtitulo">Elige un sistema. Ninguno usa datos precalculados: cada busqueda corre el algoritmo sobre el CSV completo, en el momento.</p>
          <div class="grid-inicio">{tarjetas}</div>
        </div>
    """, "inicio")


@app.route("/pearson")
def pearson():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        idj, sugerencias = buscar(jugador, pearson_norm_a_id, [n for _, n in pearson_lista])
        if idj is not None:
            nombre_mostrado = pearson_display[idj]
            correlaciones = pearson_corr[idj].drop(idj).dropna().sort_values(ascending=False).head(5)
            filas = [(pearson_display[j], v) for j, v in correlaciones.items()]
            if filas:
                resultado_html = f"<h3>Top {len(filas)} comparables a {nombre_mostrado}</h3>" + tabla_recomendaciones(filas, "Jugador", "Correlacion Pearson")
            else:
                resultado_html = f'<p class="aviso">{nombre_mostrado} no tiene suficientes temporadas en comun (minimo 8) con otro jugador para calcular la correlacion.</p>'
        else:
            resultado_html = caja_sugerencias(sugerencias, "/pearson", "jugador")

    opciones = "".join(f"<option value='{n}'>" for _, n in pearson_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 1 &mdash; Rendimiento Ofensivo {tag_metodo("pearson")}</h2>
          <p class="subtitulo">FactRendimientoOfensivo: correlacion de trayectoria de OPS por temporada de carrera (minimo 8 temporadas en comun).</p>
          <form method="get">
              <input list="lista-pearson" name="jugador" placeholder="Escribe un jugador... (ej. Hank Aaron)" value="{jugador}">
              <datalist id="lista-pearson">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un nombre y presiona Buscar.</p>"}</div>
    """, "pearson")


@app.route("/coseno")
def coseno():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        idx, sugerencias = buscar(jugador, coseno_norm_a_idx, coseno_lista)
        if idx is not None:
            nombre_mostrado = coseno_df.loc[idx, "Nombre_Mostrado"]
            sims = coseno_sim[idx]
            top_idx = np.argsort(sims)[::-1][:5]
            filas = [(coseno_df.loc[j, "Nombre_Mostrado"], sims[j]) for j in top_idx]
            perfil = coseno_df.loc[idx, atributos_coseno].to_dict()
            perfil_html = "".join(f"<li>{k.replace('_', ' ')}: {v:.2f}</li>" for k, v in perfil.items())
            resultado_html = f"""
            <h3>Perfil de {nombre_mostrado} (postemporada)</h3>
            <ul class="perfil">{perfil_html}</ul>
            <h3>Top 5 comparables</h3>
            {tabla_recomendaciones(filas, "Jugador", "Similitud Coseno")}
            """
        else:
            resultado_html = caja_sugerencias(sugerencias, "/coseno", "jugador")

    opciones = "".join(f"<option value='{n}'>" for n in coseno_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 2 &mdash; Rendimiento Ofensivo Postemporada {tag_metodo("coseno")}</h2>
          <p class="subtitulo">FactRendimientoOfensivoPostemporada: perfil de 6 atributos estandarizados (Z-score).</p>
          <form method="get">
              <input list="lista-coseno" name="jugador" placeholder="Escribe un jugador... (ej. Babe Ruth)" value="{jugador}">
              <datalist id="lista-coseno">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un nombre y presiona Buscar.</p>"}</div>
    """, "coseno")


@app.route("/slopeone")
def slopeone():
    equipo = request.args.get("equipo", "").strip()
    resultado_html = ""
    if equipo:
        nombre_real, sugerencias = buscar(equipo, slope_norm_a_nombre, slope_equipos)
        if nombre_real is not None:
            filas_equipo = slope_df[slope_df["Nombre_Equipo"] == nombre_real].sort_values("Anio")
            filas_html = "".join(
                f"<tr><td>{int(r.Anio)}</td><td>{r.Salario_Relativo*100:.1f}%</td>"
                f"<td>{r.Porcentaje_Victorias*100:.1f}%</td><td>{r.Porcentaje_Victorias_Predicho*100:.1f}%</td>"
                f"<td>{r.Error:+.4f}</td></tr>"
                for r in filas_equipo.itertuples()
            )
            resultado_html = f"""
            <h3>{nombre_real} &mdash; todas las temporadas (b = {b:.4f})</h3>
            <table>
              <tr><th>Anio</th><th>Percentil Salario</th><th>% Victorias Real</th><th>% Victorias Predicho</th><th>Error</th></tr>
              {filas_html}
            </table>
            """
        else:
            resultado_html = caja_sugerencias(sugerencias, "/slopeone", "equipo")

    opciones = "".join(f"<option value='{e}'>" for e in slope_equipos)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 3 &mdash; Resultado de Equipo {tag_metodo("slopeone")}</h2>
          <p class="subtitulo">FactResultadoEquipo: f(x) = x + b, prediciendo % de victorias desde el percentil de salario.</p>
          <form method="get">
              <input list="lista-equipos" name="equipo" placeholder="Escribe un equipo... (ej. New York Yankees)" value="{equipo}">
              <datalist id="lista-equipos">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un equipo y presiona Buscar.</p>"}</div>
    """, "slopeone")


@app.route("/salario")
def salario():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        idj, sugerencias = buscar(jugador, salario_norm_a_id, [n for _, n in salario_lista])
        if idj is not None:
            nombre_mostrado = salario_display[idj]
            correlaciones = salario_corr[idj].drop(idj).dropna().sort_values(ascending=False).head(5)
            filas = [(salario_display[j], v) for j, v in correlaciones.items()]
            if filas:
                resultado_html = f"<h3>Top {len(filas)} comparables a {nombre_mostrado} (eficiencia salarial)</h3>" + tabla_recomendaciones(filas, "Jugador", "Correlacion Pearson")
            else:
                resultado_html = f'<p class="aviso">{nombre_mostrado} no tiene suficientes temporadas en comun (minimo 5) con otro jugador para calcular la correlacion.</p>'
        else:
            resultado_html = caja_sugerencias(sugerencias, "/salario", "jugador")

    opciones = "".join(f"<option value='{n}'>" for _, n in salario_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 4 &mdash; Salario {tag_metodo("pearson")}</h2>
          <p class="subtitulo">FactSalario + DimJugador + DimTiempo: correlacion de trayectoria del Indice_Eficiencia_Salarial por temporada de carrera (minimo 5 temporadas en comun). Reutiliza el metodo de Pearson del sistema 1: en vez de comparar OPS, compara la forma en que evoluciona la eficiencia salarial de un jugador a lo largo de su carrera.</p>
          <form method="get">
              <input list="lista-salario" name="jugador" placeholder="Escribe un jugador... (ej. Alex Rodriguez)" value="{jugador}">
              <datalist id="lista-salario">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un nombre y presiona Buscar.</p>"}</div>
    """, "salario")


@app.route("/defensivo")
def defensivo():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        idj, sugerencias = buscar(jugador, defensivo_norm_a_id, [n for _, n in defensivo_lista])
        if idj is not None:
            nombre_mostrado = defensivo_display[idj]
            correlaciones = defensivo_corr[idj].drop(idj).dropna().sort_values(ascending=False).head(5)
            filas = [(defensivo_display[j], v) for j, v in correlaciones.items()]
            posicion = _posicion_principal_defensivo.get(idj, "?")
            if filas:
                resultado_html = f"<h3>Top {len(filas)} comparables a {nombre_mostrado} <span class='badge'>{posicion}</span></h3>" + tabla_recomendaciones(filas, "Jugador", "Correlacion Pearson")
            else:
                resultado_html = f'<p class="aviso">{nombre_mostrado} no tiene suficientes temporadas en comun (minimo 8) con otro jugador para calcular la correlacion.</p>'
        else:
            resultado_html = caja_sugerencias(sugerencias, "/defensivo", "jugador")

    opciones = "".join(f"<option value='{n}'>" for _, n in defensivo_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 5 &mdash; Defensivo {tag_metodo("pearson")}</h2>
          <p class="subtitulo">FactRendimientoDefensivo + DimJugador + DimTiempo + DimPosicion: correlacion de trayectoria del porcentaje de fildeo por temporada de carrera (minimo 8 temporadas en comun).</p>
          <form method="get">
              <input list="lista-defensivo" name="jugador" placeholder="Escribe un jugador... (ej. Ozzie Smith)" value="{jugador}">
              <datalist id="lista-defensivo">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un nombre y presiona Buscar.</p>"}</div>
    """, "defensivo")


@app.route("/defensivo_postemporada")
def defensivo_postemporada():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        nombre_real, sugerencias = buscar(jugador, defpost_norm_a_nombre, defpost_lista)
        if nombre_real is not None:
            fila = defpost_df[defpost_df["Nombre_Mostrado"] == nombre_real].iloc[0]
            signo = "mejora" if fila.Error > 0 else ("empeora" if fila.Error < 0 else "sin cambio")
            resultado_html = f"""
            <h3>{nombre_real} &mdash; fildeo regular vs. postemporada (b = {_b_defpost:+.4f})</h3>
            <table>
              <tr><th>% Fildeo Regular</th><th>% Fildeo Postemporada</th><th>Predicho</th><th>Error</th><th>Efecto</th></tr>
              <tr>
                <td>{fila.Porcentaje_Fildeo_Regular*100:.2f}%</td>
                <td>{fila.Porcentaje_Fildeo_Postemporada*100:.2f}%</td>
                <td>{fila.Porcentaje_Fildeo_Postemporada_Predicho*100:.2f}%</td>
                <td>{fila.Error:+.4f}</td>
                <td><span class="badge">{signo}</span></td>
              </tr>
            </table>
            """
        else:
            resultado_html = caja_sugerencias(sugerencias, "/defensivo_postemporada", "jugador")

    opciones = "".join(f"<option value='{n}'>" for n in defpost_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 6 &mdash; Defensivo Postemporada {tag_metodo("slopeone")}</h2>
          <p class="subtitulo">FactRendimientoDefensivo x FactRendimientoDefensivoPostemporada (drill-across): f(x) = x + b, prediciendo el % de fildeo en playoffs a partir del % de fildeo en temporada regular del mismo jugador.</p>
          <form method="get">
              <input list="lista-defpost" name="jugador" placeholder="Escribe un jugador... (ej. Derek Jeter)" value="{jugador}">
              <datalist id="lista-defpost">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un jugador y presiona Buscar.</p>"}</div>
    """, "defpost")


@app.route("/reconocimiento")
def reconocimiento():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        idx, sugerencias = buscar(jugador, reconocimiento_norm_a_idx, reconocimiento_lista)
        if idx is not None:
            nombre_mostrado = reconocimiento_nombres[idx]
            sims = reconocimiento_sim[idx]
            top_idx = np.argsort(sims)[::-1][:5]
            filas = [(reconocimiento_nombres[j], sims[j]) for j in top_idx]
            perfil = reconocimiento_tipos_por_jugador[idx]
            perfil_html = "".join(f"<li>{k}: {int(v)}</li>" for k, v in sorted(perfil.items(), key=lambda kv: -kv[1]))
            if not perfil_html:
                perfil_html = "<li>Sin reconocimientos frecuentes registrados</li>"
            resultado_html = f"""
            <h3>Perfil de reconocimientos de {nombre_mostrado}</h3>
            <ul class="perfil">{perfil_html}</ul>
            <h3>Top 5 comparables</h3>
            {tabla_recomendaciones(filas, "Jugador", "Similitud Coseno")}
            """
        else:
            resultado_html = caja_sugerencias(sugerencias, "/reconocimiento", "jugador")

    opciones = "".join(f"<option value='{n}'>" for n in reconocimiento_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 7 &mdash; Reconocimiento {tag_metodo("coseno")}</h2>
          <p class="subtitulo">FactReconocimiento + DimJugador + DimTiempo + DimTipoReconocimiento: perfil de tipos de reconocimiento (premios, All-Star) acumulados en toda la carrera, estandarizado (Z-score).</p>
          <form method="get">
              <input list="lista-reconocimiento" name="jugador" placeholder="Escribe un jugador... (ej. Cal Ripken)" value="{jugador}">
              <datalist id="lista-reconocimiento">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un nombre y presiona Buscar.</p>"}</div>
    """, "reconocimiento")


@app.route("/contenido")
def contenido():
    jugador = request.args.get("jugador", "").strip()
    resultado_html = ""
    if jugador:
        idx, sugerencias = buscar(jugador, contenido_norm_a_idx, contenido_lista)
        if idx is not None:
            base = contenido_ficha.iloc[idx]
            recomendados = recomendar_contenido(idx)
            filas = [(r["Nombre_Mostrado"], r["Puntaje_Contenido"]) for _, r in recomendados.iterrows()]
            ficha_html = f"""
            <ul class="perfil">
              <li>Posicion: {base['Posicion_Principal']}</li>
              <li>Decada principal: {base['Decada_Principal']}</li>
              <li>Nivel de fildeo: {base['Nivel_Fildeo']} (dentro de su posicion)</li>
              <li>Fildeo promedio de carrera: {base['Fildeo_Promedio_Carrera']:.4f}</li>
            </ul>
            """
            if filas:
                resultado_html = f"""
                <h3>Ficha de contenido de {base['Nombre_Mostrado']}</h3>
                {ficha_html}
                <h3>Top {len(filas)} comparables por contenido</h3>
                {tabla_recomendaciones(filas, "Jugador", "Puntaje de Contenido")}
                """
            else:
                resultado_html = f'<p class="aviso">No hay otros jugadores de la posicion {base["Posicion_Principal"]} con ficha de contenido suficiente para comparar.</p>'
        else:
            resultado_html = caja_sugerencias(sugerencias, "/contenido", "jugador")

    opciones = "".join(f"<option value='{n}'>" for n in contenido_lista)
    return pagina(f"""
        <div class="tarjeta">
          <h2>Sistema 8 &mdash; Recomendacion Basada en Contenido {tag_metodo("contenido")}</h2>
          <p class="subtitulo">FactRendimientoDefensivo + DimPosicion + DimTiempo: a diferencia de los sistemas 1 a 7, aqui no se calcula correlacion de Pearson ni similitud coseno entre vectores numericos. Cada jugador recibe una ficha de contenido (Posicion_Principal, Decada_Principal y Nivel_Fildeo, este ultimo discretizado por cuartiles dentro de su propia posicion) y se recomienda a los jugadores que comparten mas etiquetas de esa ficha, igual que un sistema de contenido recomienda por genero o decada en vez de por comportamiento de otros usuarios. Se exige un minimo de 3 temporadas y 300 jugadas de carrera para construir la ficha.</p>
          <form method="get">
              <input list="lista-contenido" name="jugador" placeholder="Escribe un jugador... (ej. Ozzie Smith)" value="{jugador}">
              <datalist id="lista-contenido">{opciones}</datalist>
              <button type="submit">Buscar</button>
          </form>
        </div>
        <div class="tarjeta">{resultado_html if resultado_html else "<p class='subtitulo'>Escribe un nombre y presiona Buscar.</p>"}</div>
    """, "contenido")


if __name__ == "__main__":
    app.run(debug=False)
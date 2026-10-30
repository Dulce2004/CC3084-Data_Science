"""
Diagrama del flujo, desde la lectura de las fuentes hasta la consulta de las
estimaciones en el tablero.

Se genera por script para que la figura del informe corresponda siempre al
pipeline implementado y no haya que redibujarla a mano:

    python 06_mlops/src/diagrama.py

Escribe reports/figuras/lab11_flujo.png.

Paleta: la misma del tablero del Laboratorio 10 (documento de diseño, sección 4),
con un significado fijo por color:
  azul     etapas del flujo
  gris     fuentes y destinos de información
  naranja  decisión y caminos de rechazo
  verde    los tres puntos de control: validación, evaluación y autorización
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
SALIDA = ROOT / "reports" / "figuras" / "lab11_flujo.png"

AZUL = "#2889E2"
AZUL_OSCURO = "#1F3A5F"
GRIS = "#8A94A6"
NARANJA = "#C4571A"
VERDE = "#2A7D4F"
PANEL = "#EAF0F7"
CREMA = "#FDF2E9"
TEXTO = "#1B2430"
SECUNDARIO = "#3C4654"

FIGURA = (8.2, 10.6)

# Columna central del flujo y columnas laterales.
X0, X1 = 0.250, 0.750           # caja central
XL0, XL1 = 0.000, 0.200         # columna izquierda: caminos de rechazo
XR = 0.768                      # columna derecha: puntos de control

ALTO_CAJA = 0.052
SEPARACION = 0.0455


class Lienzo:
    """Apila las etapas del flujo de arriba hacia abajo."""

    def __init__(self, ax, y_inicial=0.968):
        self.ax = ax
        self.y = y_inicial

    # ------------------------------------------------------------- piezas
    def etapa(self, titulo, detalle="", color=AZUL, relleno="white", grosor=1.6,
              alto=ALTO_CAJA):
        """Caja de ancho completo en la columna central. Devuelve sus anclajes."""
        y = self.y - alto
        self.ax.add_patch(FancyBboxPatch(
            (X0, y), X1 - X0, alto, boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=grosor, edgecolor=color, facecolor=relleno, zorder=2))
        cx = (X0 + X1) / 2
        if detalle:
            self.ax.text(cx, y + alto * 0.68, titulo, ha="center", va="center",
                         fontsize=9.2, fontweight="bold", color=TEXTO, zorder=3)
            self.ax.text(cx, y + alto * 0.27, detalle, ha="center", va="center",
                         fontsize=7.1, color=SECUNDARIO, zorder=3, linespacing=1.4)
        else:
            self.ax.text(cx, y + alto / 2, titulo, ha="center", va="center",
                         fontsize=9.2, fontweight="bold", color=TEXTO, zorder=3)
        ancla = {"arriba": (cx, y + alto), "abajo": (cx, y),
                 "izq": (X0, y + alto / 2), "der": (X1, y + alto / 2),
                 "centro_y": y + alto / 2}
        self.y = y
        return ancla

    def decision(self, titulo, alto=0.072, ancho=0.34):
        y = self.y - alto
        cx, cy = (X0 + X1) / 2, y + alto / 2
        self.ax.add_patch(Polygon(
            [(cx, cy + alto / 2), (cx + ancho / 2, cy), (cx, cy - alto / 2),
             (cx - ancho / 2, cy)], closed=True, linewidth=1.9,
            edgecolor=NARANJA, facecolor=CREMA, zorder=2))
        self.ax.text(cx, cy, titulo, ha="center", va="center", fontsize=8.8,
                     fontweight="bold", color=TEXTO, zorder=3)
        self.y = y
        return {"arriba": (cx, cy + alto / 2), "abajo": (cx, cy - alto / 2),
                "izq": (cx - ancho / 2, cy), "der": (cx + ancho / 2, cy),
                "centro_y": cy}

    def hueco(self, cantidad=SEPARACION):
        self.y -= cantidad

    # ------------------------------------------------------------- laterales
    def lateral(self, centro_y, titulo, detalle, color=NARANJA, relleno=CREMA,
                alto=0.072):
        """Caja en la columna izquierda, centrada frente a una etapa."""
        y = centro_y - alto / 2
        self.ax.add_patch(FancyBboxPatch(
            (XL0, y), XL1 - XL0, alto, boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=1.6, edgecolor=color, facecolor=relleno, zorder=2))
        cx = (XL0 + XL1) / 2
        self.ax.text(cx, y + alto * 0.74, titulo, ha="center", va="center",
                     fontsize=8.2, fontweight="bold", color=TEXTO, zorder=3)
        self.ax.text(cx, y + alto * 0.32, detalle, ha="center", va="center",
                     fontsize=7.0, color=SECUNDARIO, zorder=3, linespacing=1.4)
        return {"der": (XL1, centro_y)}

    def control(self, centro_y, texto):
        """Marca, a la derecha, uno de los tres puntos de control del enunciado."""
        self.ax.text(XR, centro_y, texto, ha="left", va="center", fontsize=8.0,
                     color=VERDE, fontweight="bold", zorder=3, linespacing=1.45)

    # ------------------------------------------------------------- flechas
    def flecha(self, p0, p1, color=AZUL_OSCURO, etiqueta="", alinear="izquierda",
               estilo="-|>"):
        self.ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=estilo, mutation_scale=12, linewidth=1.3,
            color=color, zorder=1, shrinkA=0, shrinkB=0))
        if not etiqueta:
            return
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        if abs(p1[0] - p0[0]) < 1e-6:          # vertical: etiqueta al lado
            self.ax.text(mx + 0.008, my, etiqueta, ha="left", va="center",
                         fontsize=7.4, color=color, style="italic", zorder=3)
        else:                                   # horizontal: etiqueta encima
            self.ax.text(mx, my + 0.011, etiqueta, ha="center", va="bottom",
                         fontsize=7.4, color=color, style="italic", zorder=3,
                         linespacing=1.3)


def dibujar() -> Path:
    fig, ax = plt.subplots(figsize=FIGURA)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.07, 1)
    ax.axis("off")
    L = Lienzo(ax)

    fuente = L.etapa("Fuente institucional mensual",
                     "Base de Migración 2009 – jun 2026, hoja «Datos»",
                     color=GRIS, relleno=PANEL)

    L.hueco()
    lectura = L.etapa("1. Lectura",
                      "huella SHA-256 de la fuente y de la configuración")
    L.flecha(fuente["abajo"], lectura["arriba"])

    L.hueco()
    validacion = L.etapa(
        "2. Validación — seis controles",
        "columnas y tipos · fechas y cobertura · unicidad\n"
        "faltantes · valores y denominadores · regla metodológica",
        color=VERDE, grosor=2.1, alto=0.062)
    L.flecha(lectura["abajo"], validacion["arriba"])
    L.control(validacion["centro_y"], "aquí se validan\nlos datos")

    # El rótulo del camino de rechazo va dentro de la caja: el hueco entre columnas
    # no da para una etiqueta sobre la flecha sin encimarse.
    detenida = L.lateral(validacion["centro_y"], "Ejecución detenida",
                         "si falla un control crítico,\nno se reemplazan los\n"
                         "resultados publicados", alto=0.080)
    L.flecha(validacion["izq"], detenida["der"], color=NARANJA)

    L.hueco()
    preparacion = L.etapa("3. Preparación de la serie del destino",
                          "V(m) = D(m) + f · A(m); un mes sin cobertura no es cero")
    L.flecha(validacion["abajo"], preparacion["arriba"], etiqueta="sin error crítico")

    L.hueco()
    ajuste = L.etapa("4. Ajuste del procedimiento",
                     "parámetros solo con ene-2009 a dic-2025")
    L.flecha(preparacion["abajo"], ajuste["arriba"])

    L.hueco()
    evaluacion = L.etapa("5. Evaluación",
                         "meses ocultos ene–jun 2026; MAE y RMSE contra referencia",
                         color=VERDE, grosor=2.1)
    L.flecha(ajuste["abajo"], evaluacion["arriba"])
    L.control(evaluacion["centro_y"], "aquí se evalúa\nel procedimiento")

    L.hueco()
    almacenamiento = L.etapa("6. Almacenamiento de la ejecución",
                             "data/resultados/<run_id>/ · bitácora e historial")
    L.flecha(evaluacion["abajo"], almacenamiento["arriba"])

    L.hueco()
    regla = L.decision("Regla de aceptación")
    L.flecha(almacenamiento["abajo"], regla["arriba"])
    L.control(regla["centro_y"], "aquí se autoriza\nla publicación")

    rechazada = L.lateral(regla["centro_y"], "Versión rechazada",
                          "queda el expediente;\nsigue vigente la versión\naprobada anterior",
                          alto=0.080)
    L.flecha(regla["izq"], rechazada["der"], color=NARANJA, etiqueta="rechaza")

    L.hueco()
    publicacion = L.etapa("7. Publicación de la versión aprobada",
                          "data/aprobado/<run_id>/ y version_actual.json")
    L.flecha(regla["abajo"], publicacion["arriba"], etiqueta="aprueba")

    restauracion = L.lateral(publicacion["centro_y"], "Restauración",
                             "--restaurar <run_id>\ndeja vigente una\nversión anterior",
                             color=AZUL, relleno=PANEL, alto=0.080)
    L.flecha(publicacion["izq"], restauracion["der"], color=AZUL, estilo="<|-|>")

    L.hueco()
    tablero = L.etapa("Tablero del Observatorio (Lab. 10)",
                      "consulta la versión aprobada vigente y su actualización",
                      color=GRIS, relleno=PANEL)
    L.flecha(publicacion["abajo"], tablero["arriba"])

    ax.text(0.5, L.y - 0.042,
            "Un solo comando ejecuta las siete etapas:\n"
            "python 06_mlops/src/pipeline.py --config 06_mlops/config/pipeline.yaml",
            ha="center", va="center", fontsize=8.2, color=SECUNDARIO,
            style="italic", linespacing=1.5)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=220, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    plt.close(fig)
    return SALIDA


if __name__ == "__main__":
    print(f"escrito: {dibujar()}")

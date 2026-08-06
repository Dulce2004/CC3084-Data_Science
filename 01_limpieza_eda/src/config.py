"""Rutas y constantes compartidas del laboratorio."""
from pathlib import Path

# Raíz del repositorio (dos niveles arriba de este archivo)
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figuras"

RAW_XLSX = RAW_DIR / "Base_Migracion_2009-2026jun.xlsx"
CLEAN_CSV = PROCESSED_DIR / "migracion_limpia.csv"

# Proporción temporal entrenamiento/prueba (instrucción del lab: ~70/30)
TRAIN_FRAC = 0.70

# Paleta categórica accesible (validada para daltonismo), orden fijo.
# Fuente: principios del skill dataviz (orden categórico fijo, no cíclico).
PALETTE = {
    "azul":    "#3b7dd8",
    "naranja": "#e8833a",
    "verde":   "#4aa564",
    "rojo":    "#d1495b",
    "morado":  "#8a5fbf",
    "cafe":    "#9c6b4e",
    "gris":    "#7a7a7a",
}
PALETTE_LIST = list(PALETTE.values())


def aplicar_estilo():
    """Aplica un estilo matplotlib limpio y consistente para todo el lab."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (10, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#9a9a9a",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#222222",
        "text.color": "#333333",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "font.size": 10,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })
    from cycler import cycler
    mpl.rcParams["axes.prop_cycle"] = cycler(color=PALETTE_LIST)


def miles(x, pos=None):
    """Formateador de eje: separador de miles."""
    return f"{x:,.0f}"

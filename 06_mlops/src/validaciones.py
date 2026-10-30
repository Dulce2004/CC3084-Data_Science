"""
Etapa 2 — controles automáticos de calidad.

Seis controles sobre la fuente cruda, antes de preparar la serie y antes de
publicar. Cada control declara qué verifica, cuándo genera una advertencia y
cuándo detiene la ejecución; el nivel (crítico o advertencia) se configura en
`controles` del archivo de configuración, no en el código.

Regla transversal: un período del rango de cobertura sin información NO se
convierte en cero. El control 2 lo reporta como mes sin cobertura y, si supera
el umbral, detiene el flujo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OK, ADV, ERROR = "ok", "advertencia", "error"


def _entero(s: pd.Series) -> bool:
    n = pd.to_numeric(s, errors="coerce")
    return bool(n.notna().all() and np.all(np.mod(n.dropna(), 1) == 0))


def _numerica(s: pd.Series) -> bool:
    return bool(pd.to_numeric(s, errors="coerce").notna().all())


# Comprobadores del tipo declarado en la configuración.
TIPOS = {"entero": _entero, "numerica": _numerica, "texto": lambda s: True}

# Columnas de granularidad diaria: si aparecen, la serie mensual no debe
# construirse sumando presencias (regla metodológica del proyecto).
COLUMNAS_DIARIAS = {"Día", "Dia", "Día cod", "Dia cod", "Fecha exacta", "day"}


@dataclass
class Resultado:
    """Resultado de un control, tal como se escribe en el reporte."""
    id: str
    verifica: str
    nivel: str
    estado: str = OK
    detalle: str = ""
    filas_afectadas: int = 0

    @property
    def detiene(self) -> bool:
        """Solo un control crítico en estado error detiene la ejecución."""
        return self.estado == ERROR and self.nivel == "critico"

    def como_fila(self) -> dict:
        return {
            "control": self.id,
            "verifica": self.verifica,
            "nivel": self.nivel,
            "estado": self.estado,
            "detiene_ejecucion": self.detiene,
            "filas_afectadas": self.filas_afectadas,
            "detalle": self.detalle,
        }


@dataclass
class Reporte:
    """Conjunto de resultados de los seis controles."""
    resultados: list = field(default_factory=list)

    def agregar(self, r: Resultado) -> Resultado:
        self.resultados.append(r)
        return r

    @property
    def detiene(self) -> bool:
        return any(r.detiene for r in self.resultados)

    @property
    def errores(self) -> list:
        return [r for r in self.resultados if r.estado == ERROR]

    @property
    def advertencias(self) -> list:
        return [r for r in self.resultados if r.estado == ADV]

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame([r.como_fila() for r in self.resultados])

    def resumen(self) -> dict:
        return {
            "n_controles": len(self.resultados),
            "n_ok": sum(r.estado == OK for r in self.resultados),
            "n_advertencias": len(self.advertencias),
            "n_errores": len(self.errores),
            "detiene": self.detiene,
            "controles_en_error": [r.id for r in self.errores],
        }


def _fecha_mensual(df: pd.DataFrame) -> pd.Series:
    """Fecha del primer día del mes a partir de Año y Mes cod (NaT si es inválida)."""
    return pd.to_datetime(
        dict(year=pd.to_numeric(df.get("Año"), errors="coerce"),
             month=pd.to_numeric(df.get("Mes cod"), errors="coerce"),
             day=1),
        errors="coerce")


# ====================================================================
# Control 1 — columnas y tipos
# ====================================================================
def control_columnas_tipos(df: pd.DataFrame, cfg) -> Resultado:
    """
    Verifica: que estén todas las columnas indispensables y que cada una admita
    el tipo declarado en la configuración.
    Advierte: si la fuente trae columnas no declaradas (cambio de estructura).
    Detiene: si falta una columna indispensable o una no admite su tipo.
    """
    r = Resultado("1. columnas_tipos",
                  "columnas indispensables presentes y con tipo compatible",
                  cfg.nivel_control("columnas_tipos"))
    requeridas = cfg.get("fuentes", "migracion", "columnas_requeridas")

    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        r.estado = ERROR
        r.detalle = "columnas indispensables ausentes: " + ", ".join(faltantes)
        return r

    incompatibles = []
    for col, tipo in requeridas.items():
        comprobar = TIPOS.get(str(tipo))
        if comprobar is None:
            incompatibles.append(f"{col} (tipo declarado desconocido: {tipo})")
        elif not comprobar(df[col]):
            malos = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
            incompatibles.append(f"{col} (declarada {tipo}, {malos} valores no convertibles)")
    if incompatibles:
        r.estado = ERROR
        r.filas_afectadas = len(df)
        r.detalle = "columnas con tipo incompatible: " + "; ".join(incompatibles)
        return r

    conocidas = set(requeridas) | set(
        cfg.get("fuentes", "migracion", "columnas_opcionales", defecto=[])) | {"fecha"}
    extra = [str(c) for c in df.columns if c not in conocidas]
    r.detalle = f"{len(requeridas)} columnas indispensables verificadas"
    if extra:
        r.estado = ADV
        r.detalle += "; columnas no declaradas en la configuración: " + ", ".join(extra)
    return r


# ====================================================================
# Control 2 — fechas y cobertura
# ====================================================================
def control_fechas_cobertura(df: pd.DataFrame, cfg) -> Resultado:
    """
    Verifica: que Año y Mes cod formen fechas válidas y que todos los meses del
    rango de cobertura configurado traigan información.
    Advierte: si la fuente trae meses fuera del rango de cobertura, o si los
    meses sin cobertura no superan el umbral.
    Detiene: si hay fechas inválidas, o si los meses sin cobertura superan
    `controles.umbrales.max_meses_sin_cobertura`.
    Nunca rellena con cero un mes sin cobertura.
    """
    r = Resultado("2. fechas_cobertura",
                  "fechas válidas y cobertura completa del rango configurado",
                  cfg.nivel_control("fechas_cobertura"))

    fecha = _fecha_mensual(df)
    invalidas = int(fecha.isna().sum())
    if invalidas:
        r.estado = ERROR
        r.filas_afectadas = invalidas
        r.detalle = f"{invalidas} filas con Año/Mes cod que no forman una fecha válida"
        return r

    presentes = set(fecha.dt.to_period("M"))
    cobertura = cfg.meses_cobertura().to_period("M")
    sin_cobertura = [str(m) for m in cobertura if m not in presentes]
    fuera = sorted(str(m) for m in presentes if m not in set(cobertura))
    maximo = int(cfg.umbral("max_meses_sin_cobertura", defecto=0))

    r.detalle = (f"{len(presentes)} meses con información; cobertura configurada "
                 f"{cobertura[0]} a {cobertura[-1]} ({len(cobertura)} meses)")
    if len(sin_cobertura) > maximo:
        r.estado = ERROR
        r.filas_afectadas = len(sin_cobertura)
        r.detalle = (f"{len(sin_cobertura)} meses del rango de cobertura sin información "
                     f"(máximo permitido {maximo}): " + ", ".join(sin_cobertura[:12])
                     + (" …" if len(sin_cobertura) > 12 else "")
                     + ". No se rellenan con cero.")
        return r
    if sin_cobertura:
        r.estado = ADV
        r.detalle += "; meses sin cobertura dentro del umbral: " + ", ".join(sin_cobertura)
    if fuera:
        r.estado = ADV
        r.detalle += "; meses fuera del rango de cobertura: " + ", ".join(fuera[:6])
    return r


# ====================================================================
# Control 3 — unicidad
# ====================================================================
def control_unicidad(df: pd.DataFrame, cfg) -> Resultado:
    """
    Verifica: que no haya filas repetidas según la clave que define la
    granularidad de la fuente (`fuentes.migracion.clave`).
    Advierte: si hay duplicados pero su proporción no supera
    `controles.umbrales.max_prop_duplicados`.
    Detiene: si la proporción de duplicados supera ese umbral, o si ninguna
    columna de la clave está en la fuente.
    """
    r = Resultado("3. unicidad",
                  "sin filas repetidas según la clave de la granularidad declarada",
                  cfg.nivel_control("unicidad"))
    declarada = cfg.get("fuentes", "migracion", "clave")
    clave = [c for c in declarada if c in df.columns]
    if not clave:
        r.estado = ERROR
        r.detalle = "ninguna columna de la clave configurada está en la fuente: " + \
                    ", ".join(declarada)
        return r

    dup = df.duplicated(subset=clave, keep=False)
    n = int(dup.sum())
    prop = n / len(df) if len(df) else 0.0
    maximo = float(cfg.umbral("max_prop_duplicados", defecto=0.0))
    r.detalle = f"clave: {' + '.join(clave)}; {n} filas duplicadas ({prop:.4%})"
    if n == 0:
        return r
    ejemplos = (df.loc[dup, clave].astype(str).agg(" | ".join, axis=1)
                  .drop_duplicates().head(3).tolist())
    r.filas_afectadas = n
    r.detalle += "; ejemplos: " + " // ".join(ejemplos)
    r.estado = ERROR if prop > maximo else ADV
    return r


# ====================================================================
# Control 4 — valores faltantes
# ====================================================================
def control_valores_faltantes(df: pd.DataFrame, cfg) -> Resultado:
    """
    Verifica: que no falten valores en las columnas necesarias para estimar, y
    distingue las ausencias permitidas (`fuentes.migracion.nulos_permitidos`).
    Advierte: si una columna de ausencia permitida supera
    `controles.umbrales.max_prop_nulos_permitidos`.
    Detiene: si falta cualquier valor en una columna indispensable.
    """
    r = Resultado("4. valores_faltantes",
                  "sin ausencias en las columnas necesarias para estimar",
                  cfg.nivel_control("valores_faltantes"))
    requeridas = cfg.get("fuentes", "migracion", "columnas_requeridas")
    permitidas = set(cfg.get("fuentes", "migracion", "nulos_permitidos", defecto=[]))
    maximo = float(cfg.umbral("max_prop_nulos_permitidos", defecto=1.0))

    bloquean, avisan, filas = [], [], 0
    for col in requeridas:
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n == 0:
            continue
        prop = n / len(df)
        if col in permitidas:
            if prop > maximo:
                avisan.append(f"{col}: {n} ({prop:.2%})")
        else:
            bloquean.append(f"{col}: {n} ({prop:.2%})")
            filas += n

    if bloquean:
        r.estado = ERROR
        r.filas_afectadas = filas
        r.detalle = "ausencias que impiden estimar: " + "; ".join(bloquean)
        return r
    r.detalle = "sin ausencias en columnas indispensables"
    if avisan:
        r.estado = ADV
        r.detalle += f"; ausencias permitidas por encima del {maximo:.0%}: " + "; ".join(avisan)
    return r


# ====================================================================
# Control 5 — valores y denominadores
# ====================================================================
def control_valores_denominadores(df: pd.DataFrame, cfg) -> Resultado:
    """
    Verifica: que los conteos de viajeros no sean negativos ni no finitos, y que
    los denominadores del procedimiento no sean cero. El denominador del
    procedimiento es la base del factor de reparto: los turistas que ingresan por
    el puesto configurado en `procedimiento.puesto_reparto`.
    Advierte: si un denominador es cero en un mes fuera del período evaluado.
    Detiene: si hay conteos negativos o no finitos, o si el denominador es cero o
    está ausente en algún mes del período de evaluación.
    """
    r = Resultado("5. valores_denominadores",
                  "conteos no negativos y finitos; denominadores distintos de cero",
                  cfg.nivel_control("valores_denominadores"))

    v = pd.to_numeric(df.get("Viajero"), errors="coerce")
    arr = v.to_numpy(dtype="float64", na_value=np.nan)
    negativos = int(np.nansum(arr < 0))
    no_finitos = int((~np.isfinite(arr)).sum())
    if negativos or no_finitos:
        r.estado = ERROR
        r.filas_afectadas = negativos + no_finitos
        r.detalle = (f"{negativos} conteos negativos y {no_finitos} valores no finitos "
                     "en 'Viajero'")
        return r

    puesto = cfg.get("procedimiento", "puesto_reparto")
    tipos = list(cfg.get("procedimiento", "tipos_reparto"))
    base = (df.assign(_fecha=_fecha_mensual(df), _v=arr)
              .loc[lambda d: d["Frontera"].eq(puesto) & d["Tipo de Viajero"].isin(tipos)]
              .groupby("_fecha")["_v"].sum())

    evaluacion = cfg.meses_evaluacion()
    ceros = base[base <= 0]
    ceros_eval = [m for m in ceros.index if m in set(evaluacion)]
    faltan_eval = [m for m in evaluacion if m not in base.index]

    r.detalle = (f"sin conteos negativos ni no finitos; denominador del reparto "
                 f"({puesto}, {'/'.join(tipos)}) con {len(base)} meses")
    if ceros_eval or faltan_eval:
        r.estado = ERROR
        r.filas_afectadas = len(ceros_eval) + len(faltan_eval)
        meses = [str(pd.Timestamp(m).date()) for m in list(ceros_eval) + list(faltan_eval)]
        r.detalle = ("denominador del factor de reparto igual a cero o ausente en meses del "
                     "período de evaluación: " + ", ".join(meses))
        return r
    if len(ceros):
        r.estado = ADV
        r.detalle += (f"; {len(ceros)} meses con denominador cero fuera del período evaluado: "
                      + ", ".join(str(pd.Timestamp(m).date()) for m in ceros.index[:6]))
    return r


# ====================================================================
# Control 6 — regla metodológica del proyecto
# ====================================================================
def control_regla_metodologica(df: pd.DataFrame, cfg) -> Resultado:
    """
    Verifica tres reglas del proyecto a la vez:
      (a) la serie mensual proviene de la fuente mensual: la fuente declara Año y
          Mes cod, no trae columna de granularidad diaria y ningún año supera los
          12 meses distintos, de modo que la serie no se obtiene sumando
          presencias diarias;
      (b) no se duplican visitantes: los puestos del destino y el puesto base del
          reparto son conjuntos disjuntos, y ambos existen en la fuente;
      (c) el factor de reparto está en (0, 1].
    Advierte: si el factor de reparto está fuera del rango de sensibilidad
    declarado en `procedimiento.factor_reparto_rango`.
    Detiene: si falla cualquiera de las tres reglas.
    """
    r = Resultado("6. regla_metodologica",
                  "serie mensual de fuente mensual, sin duplicar visitantes y factor "
                  "de reparto en rango",
                  cfg.nivel_control("regla_metodologica"))

    # (a) la serie mensual proviene de la fuente mensual
    diarias = [str(c) for c in df.columns if str(c) in COLUMNAS_DIARIAS]
    if diarias:
        r.estado = ERROR
        r.detalle = ("la fuente trae granularidad diaria (" + ", ".join(diarias) +
                     "): la serie mensual no puede construirse sumando presencias diarias")
        return r
    meses_por_anio = df.groupby("Año")["Mes cod"].nunique()
    exceso = meses_por_anio[meses_por_anio > 12]
    if len(exceso):
        r.estado = ERROR
        r.filas_afectadas = int(len(exceso))
        r.detalle = ("años con más de 12 meses distintos (indicio de agregación diaria): "
                     + ", ".join(f"{a}={n}" for a, n in exceso.items()))
        return r

    # (b) sin duplicar visitantes
    puestos_destino = set(cfg.get("procedimiento", "puestos_destino"))
    puesto_reparto = cfg.get("procedimiento", "puesto_reparto")
    if puesto_reparto in puestos_destino:
        r.estado = ERROR
        r.detalle = (f"el puesto base del reparto ({puesto_reparto}) también figura entre los "
                     "puestos del destino: sumar ambos contaría dos veces al mismo visitante")
        return r
    en_fuente = set(df["Frontera"].dropna().astype(str).unique())
    ausentes = sorted(puestos_destino - en_fuente)
    if ausentes:
        r.estado = ERROR
        r.filas_afectadas = len(ausentes)
        r.detalle = ("puestos de entrada del destino que no existen en la fuente: "
                     + ", ".join(ausentes))
        return r
    if puesto_reparto not in en_fuente:
        r.estado = ERROR
        r.detalle = f"el puesto base del reparto ({puesto_reparto}) no existe en la fuente"
        return r

    # (c) factor de reparto
    f = float(cfg.get("procedimiento", "factor_reparto"))
    if not 0 < f <= 1:
        r.estado = ERROR
        r.detalle = f"factor de reparto fuera de (0, 1]: {f}"
        return r

    r.detalle = (f"fuente mensual (Año + Mes cod, máx. {int(meses_por_anio.max())} meses por "
                 f"año); {len(puestos_destino)} puestos del destino disjuntos de "
                 f"{puesto_reparto}; factor de reparto {f}")
    rango = cfg.get("procedimiento", "factor_reparto_rango")
    if rango and not (float(rango[0]) <= f <= float(rango[1])):
        r.estado = ADV
        r.detalle += f"; fuera del rango de sensibilidad declarado {list(rango)}"
    return r


CONTROLES = [
    control_columnas_tipos,
    control_fechas_cobertura,
    control_unicidad,
    control_valores_faltantes,
    control_valores_denominadores,
    control_regla_metodologica,
]


def ejecutar(df: pd.DataFrame, cfg, bitacora=None) -> Reporte:
    """
    Corre los seis controles en orden y devuelve el reporte completo.

    No interrumpe en el primer error: corre todos para que el reporte describa el
    estado completo de la fuente. Quien decide detenerse es el pipeline, leyendo
    `Reporte.detiene`. Si un control lanza una excepción, su resultado queda en
    error: un control roto nunca aprueba una versión.
    """
    reporte = Reporte()
    for control in CONTROLES:
        nombre = control.__name__.replace("control_", "")
        try:
            r = control(df, cfg)
        except Exception as exc:
            r = Resultado(f"?. {nombre}", "el control no pudo ejecutarse",
                          cfg.nivel_control(nombre), ERROR, f"excepción: {exc!r}")
        reporte.agregar(r)
        if bitacora is not None:
            mensaje = f"control {r.id}: {r.estado} — {r.detalle}"
            if r.estado == ERROR:
                bitacora.error(mensaje)
            elif r.estado == ADV:
                bitacora.advertencia(mensaje)
            else:
                bitacora.info(mensaje)
    return reporte

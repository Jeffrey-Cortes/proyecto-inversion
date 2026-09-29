"""
Motor de backtest compartido por todas las estrategias: convierte una
señal 0/1 en curva de valor del portafolio y en una fila de métricas, con
la misma disciplina anti look-ahead bias en todos los casos (la señal
calculada al cierre del día t se ejecuta en el retorno del día t+1).
"""
import numpy as np
import pandas as pd

from . import metrics as m


def recortar_periodo(df: pd.DataFrame, señal: pd.Series, fecha_inicio: str,
                      fecha_fin: str | None = None) -> pd.DataFrame:
    """Adjunta la señal al DataFrame, recorta al periodo de análisis y
    descarta el warmup (filas donde la señal todavía es NaN porque el
    indicador no tenía suficiente historia). `fecha_fin` es opcional —
    útil para acotar un benchmark a la misma ventana exacta de un holdout
    (ver Fase 3, Paso 3)."""
    df = df.copy()
    df["senal"] = señal.values
    df = df[df["fecha"] >= fecha_inicio]
    if fecha_fin is not None:
        df = df[df["fecha"] <= fecha_fin]
    df = df.dropna(subset=["senal"])
    return df.reset_index(drop=True)


def simular(df: pd.DataFrame, capital_inicial: float = 10_000.0,
            costo_bps: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """df debe traer 'fecha', 'cierre' y 'senal' (0/1, ya recortado con
    `recortar_periodo`). Devuelve el df con columnas de retorno/posición
    añadidas, y la curva de valor del portafolio.

    `costo_bps`: costo por operación en puntos básicos (100 bps = 1%),
    modelando spread/slippage/comisión — se resta del retorno cada vez que
    `posicion` cambia. Por defecto 0 (no rompe Paso 1/2, que no lo
    modelaron); la comparación final del Paso 3 lo usa explícitamente
    ("comparación justa... después de comisiones", CLAUDE.md)."""
    df = df.copy()
    df["retorno_diario"] = df["cierre"].pct_change()
    df["posicion"] = df["senal"].shift(1).fillna(0)
    df["retorno_estrategia"] = df["posicion"] * df["retorno_diario"]
    df.loc[df.index[0], ["retorno_diario", "retorno_estrategia"]] = 0.0

    if costo_bps:
        cambia_posicion = df["posicion"].diff().abs().fillna(0) > 0
        df.loc[cambia_posicion, "retorno_estrategia"] -= costo_bps / 10_000

    valor = capital_inicial * (1 + df["retorno_estrategia"]).cumprod()
    return df, valor


def simular_dca(df: pd.DataFrame, aporte: float, frecuencia: str = "ME",
                 costo_bps: float = 0.0, vender_en_señal_negativa: bool = False) -> tuple[pd.DataFrame, list]:
    """DCA con señal de timing: en cada fecha de aportación (la primera
    fila de trading de cada mes/semana), se agrega `aporte` a una reserva
    en efectivo (0% de rendimiento, mismo criterio que en Paso 2). Si la
    posición (señal del día anterior, mismo criterio anti look-ahead bias
    que `simular`) es 1, se invierte TODA la reserva acumulada — no solo
    la aportación de ese periodo, para no dejar efectivo ocioso
    indefinidamente una vez que la señal vuelve a ser positiva.

    `vender_en_señal_negativa` (default False, preserva el comportamiento
    original de la Fase 3 Paso 4): si es False, esta función **solo
    compra** — controla si el dinero NUEVO entra o espera, pero las
    unidades ya compradas nunca se venden aunque la señal baje a 0
    después. Si es True, además **vende todas las unidades** en cuanto la
    posición pasa a 0 (mismo día que se detecta el cambio, no solo en
    fechas de aportación) — simétrico con `simular` (Paso 2/3), donde
    salir de la señal sí protege el capital ya invertido, no solo el
    nuevo.

    df debe traer 'fecha', 'cierre' y 'senal' (0/1, ya recortado con
    `recortar_periodo`). `frecuencia`: cualquier alias de `pandas.Grouper`
    ("ME" = fin de mes, "W" = semanal).

    Devuelve el df con columnas de valor/retorno añadidas, y la lista de
    flujos de caja (aportaciones, negativas) para calcular IRR con
    `metrics.irr_periodica` — el valor final se sigue viendo en
    `df["valor"].iloc[-1]`."""
    df = df.copy().reset_index(drop=True)
    df["posicion"] = df["senal"].shift(1).fillna(0)

    periodo = df["fecha"].dt.to_period("M" if frecuencia == "ME" else frecuencia)
    es_dia_aporte = df["fecha"] == df.groupby(periodo)["fecha"].transform("min")

    unidades = 0.0
    reserva = 0.0
    precio_prev = None
    serie_unidades = np.empty(len(df))
    serie_reserva = np.empty(len(df))
    serie_retorno = np.empty(len(df))
    flujos = []

    for i, row in df.iterrows():
        precio = row["cierre"]
        valor_previo = unidades * (precio_prev or precio) + reserva
        serie_retorno[i] = (
            unidades * (precio - precio_prev) / valor_previo
            if precio_prev is not None and valor_previo > 0 else 0.0
        )

        if es_dia_aporte.iloc[i]:
            reserva += aporte
            flujos.append(-aporte)

        if row["posicion"] == 1 and reserva > 0:
            costo = reserva * costo_bps / 10_000
            unidades += (reserva - costo) / precio
            reserva = 0.0
        elif vender_en_señal_negativa and row["posicion"] == 0 and unidades > 0:
            producto = unidades * precio
            costo = producto * costo_bps / 10_000
            reserva += producto - costo
            unidades = 0.0

        serie_unidades[i] = unidades
        serie_reserva[i] = reserva
        precio_prev = precio

    df["unidades"] = serie_unidades
    df["reserva"] = serie_reserva
    df["retorno_estrategia"] = serie_retorno
    df["valor"] = df["unidades"] * df["cierre"] + df["reserva"]

    flujos[-1] += df["valor"].iloc[-1]
    return df, flujos


def resumen_dca(df: pd.DataFrame, flujos: list, aporte: float, nombre: str,
                 periodos_por_año: int) -> dict:
    """Una fila de métricas para DCA con timing — usa IRR (money-weighted)
    en vez de CAGR, porque el capital entra en momentos distintos (mismo
    criterio que el benchmark de DCA del Paso 1). `flujos` y `aporte` son
    los que devuelve `simular_dca` (cada entrada de `flujos` corresponde a
    una aportación de tamaño `aporte`, salvo la última que además suma el
    valor final liquidado)."""
    return {
        "Escenario": nombre,
        "Total aportado (USD)": aporte * len(flujos),
        "Valor final (USD)": df["valor"].iloc[-1],
        "IRR anualizado": m.irr_periodica(flujos, periodos_por_año),
        "Drawdown máximo": m.max_drawdown(df["valor"]),
        "Sharpe (rf=0)": m.sharpe(df["retorno_estrategia"]),
        "Sortino (rf=0)": m.sortino(df["retorno_estrategia"]),
        "Operaciones": int((df["posicion"].diff().abs() == 1).sum()),
        "% tiempo invertido": df["posicion"].mean(),
    }


def resumen(df: pd.DataFrame, valor: pd.Series, capital_inicial: float, nombre: str) -> dict:
    """Una fila de métricas para la tabla comparativa (mismo formato en
    todas las estrategias y en el benchmark)."""
    fechas = df["fecha"].values
    años = (fechas[-1] - fechas[0]) / np.timedelta64(365, "D")
    return {
        "Escenario": nombre,
        "Valor final (USD)": valor.iloc[-1],
        "CAGR": m.cagr(valor, capital_inicial, años),
        "Drawdown máximo": m.max_drawdown(valor),
        "Sharpe (rf=0)": m.sharpe(df["retorno_estrategia"]),
        "Sortino (rf=0)": m.sortino(df["retorno_estrategia"]),
        "Operaciones": int((df["senal"].diff().abs() == 1).sum()),
        "% tiempo invertido": df["posicion"].mean(),
    }

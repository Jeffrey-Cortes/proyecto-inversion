"""
Features y target para los modelos de ML de la Fase 3, Paso 3.

Todas las features se calculan solo con datos pasados (rolling/shift) —
nada de look-ahead bias. El target SÍ mira al futuro (es la etiqueta que
el modelo intenta predecir, no un insumo del modelo): en cada fecha t,
vale 1 si el retorno de los siguientes `horizonte` días de trading es
positivo.

Horizonte por defecto: 21 días (~1 mes de trading). Elegido porque el
usuario ejecuta las señales manualmente (prefiere una cadencia de revisión
mensual, no diaria) y porque el retorno diario de un ETF es casi un random
walk — señal demasiado débil para aprender sin sobreajustar (ver
CLAUDE.md, Fase 3 Paso 3).
"""
import numpy as np
import pandas as pd

from . import indicators as ind

HORIZONTE_DEFECTO = 21


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """df debe traer 'fecha', 'cierre' y 'volumen' (OHLCV de
    `data.descargar_spy_diario`). Devuelve un DataFrame de features
    alineado al índice de df, con NaN durante el warmup de cada indicador."""
    cierre = df["cierre"]
    retornos = cierre.pct_change()

    sma50 = cierre.rolling(50).mean()
    sma200 = cierre.rolling(200).mean()

    X = pd.DataFrame(index=df.index)
    X["retorno_1d"] = ind.retorno_n_dias(cierre, 1)
    X["retorno_5d"] = ind.retorno_n_dias(cierre, 5)
    X["retorno_21d"] = ind.retorno_n_dias(cierre, 21)
    X["dist_sma50"] = cierre / sma50 - 1
    X["dist_sma200"] = cierre / sma200 - 1
    X["sma50_vs_sma200"] = sma50 / sma200 - 1
    X["rsi_14"] = ind.rsi(cierre, 14)
    X["volatilidad_21d"] = retornos.rolling(21).std()
    X["volatilidad_63d"] = retornos.rolling(63).std()
    X["momentum_12m"] = ind.retorno_n_dias(cierre, 252)
    X["volumen_relativo"] = df["volumen"] / df["volumen"].rolling(20).mean() - 1
    return X


def construir_target(df: pd.DataFrame, horizonte: int = HORIZONTE_DEFECTO) -> pd.Series:
    """1 si el retorno de los siguientes `horizonte` días de trading es
    positivo, 0 si no. NaN en las últimas `horizonte` filas, donde el
    futuro necesario todavía no existe en los datos."""
    cierre = df["cierre"]
    retorno_futuro = cierre.shift(-horizonte) / cierre - 1
    target = (retorno_futuro > 0).astype(float)
    target[retorno_futuro.isna()] = np.nan
    return target


def preparar_dataset(df: pd.DataFrame, horizonte: int = HORIZONTE_DEFECTO) -> tuple[pd.DataFrame, pd.Series]:
    """Construye X, y y descarta las filas donde cualquiera de las dos
    tiene NaN (warmup de features al inicio, ventana sin futuro al final).

    Mantiene el índice original de `df` (no lo resetea) para que
    `walkforward.py` pueda alinear cada fila con su 'fecha' vía
    `df.loc[X.index, "fecha"]`."""
    X = construir_features(df)
    y = construir_target(df, horizonte)
    validas = X.notna().all(axis=1) & y.notna()
    return X[validas], y[validas]

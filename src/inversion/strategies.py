"""
Señales (reglas de entrada/salida) de la Fase 3, Paso 2. Cada función
recibe un DataFrame con columnas 'fecha' y 'cierre' (orden ascendente) y
devuelve una Serie 0/1 alineada a su índice: 1 = invertido, 0 = en
efectivo. Durante el "warmup" del indicador (mientras no hay suficiente
historia para calcularlo) la señal es NaN — así `backtest.recortar_periodo`
sabe qué filas descartar.

Todos los parámetros (50/200, 252 días, 14/30/70) son valores clásicos de
cada indicador, no ajustados sobre estos datos — por eso ninguna de estas
estrategias requiere validación walk-forward (ver CLAUDE.md, Paso 2).
"""
import numpy as np
import pandas as pd

from . import indicators as ind


def señal_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Siempre invertido. Sirve como benchmark, ejecutado con el mismo
    motor de backtest que las demás estrategias para comparar sobre
    exactamente los mismos datos y periodo."""
    return pd.Series(1.0, index=df.index)


def señal_sma(df: pd.DataFrame, corta: int = 50, larga: int = 200) -> pd.Series:
    """Cruce de medias móviles ("golden cross" / "death cross"): invertido
    cuando SMA(corta) > SMA(larga)."""
    sma_corta = df["cierre"].rolling(corta).mean()
    sma_larga = df["cierre"].rolling(larga).mean()
    señal = (sma_corta > sma_larga).astype(float)
    señal[sma_larga.isna()] = np.nan
    return señal


def señal_momentum(df: pd.DataFrame, ventana: int = 252) -> pd.Series:
    """Time-series momentum de 12 meses (Moskowitz/Ooi/Pedersen 2012):
    invertido cuando el retorno de los últimos `ventana` días es positivo."""
    retorno = ind.retorno_n_dias(df["cierre"], ventana)
    señal = (retorno > 0).astype(float)
    señal[retorno.isna()] = np.nan
    return señal


def señal_rsi(df: pd.DataFrame, ventana: int = 14,
              sobreventa: float = 30, sobrecompra: float = 70) -> pd.Series:
    """RSI(14) con suavizado de Wilder y regla con histéresis: entra en
    sobreventa (RSI cruza por debajo de `sobreventa`), sale en sobrecompra
    (RSI cruza por arriba de `sobrecompra`), mantiene posición entre
    cruces. A diferencia de SMA y Momentum (trend-following), esta es una
    regla de reversión a la media (contrarian)."""
    rsi = ind.rsi(df["cierre"], ventana)

    cruda = pd.Series(np.nan, index=df.index)
    cruda[rsi < sobreventa] = 1
    cruda[rsi > sobrecompra] = 0
    señal = cruda.ffill().fillna(0)
    señal[rsi.isna()] = np.nan
    return señal

"""
Indicadores puros de precio, sin opinión de estrategia — reusados tanto
por `strategies.py` (reglas del Paso 2: umbral fijo sobre el indicador)
como por `features.py` (Paso 3: el indicador como valor continuo, insumo
de un modelo de ML).
"""
import pandas as pd


def rsi(cierre: pd.Series, ventana: int = 14) -> pd.Series:
    """RSI con suavizado de Wilder (equivalente a una media móvil
    exponencial con alpha = 1/ventana) — el método clásico, no un promedio
    simple."""
    delta = cierre.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_ganancia = ganancia.ewm(alpha=1 / ventana, adjust=False).mean()
    avg_perdida = perdida.ewm(alpha=1 / ventana, adjust=False).mean()
    rs = avg_ganancia / avg_perdida
    return 100 - (100 / (1 + rs))


def retorno_n_dias(cierre: pd.Series, ventana: int) -> pd.Series:
    """Retorno de los últimos `ventana` días de trading (time-series
    momentum cuando ventana=252 ~ 12 meses; también sirve como feature de
    retorno rezagado con ventanas más cortas)."""
    return cierre / cierre.shift(ventana) - 1

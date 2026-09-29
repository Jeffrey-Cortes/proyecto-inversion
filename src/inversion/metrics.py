"""
Métricas de rendimiento/riesgo compartidas entre todas las estrategias de
la Fase 3 (benchmark, SMA, momentum, RSI, y los modelos de ML que vengan
después). Un solo lugar para no repetir ni desalinear estas fórmulas.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def cagr(valor_serie: pd.Series, capital_inicial: float, años: float) -> float:
    return (valor_serie.iloc[-1] / capital_inicial) ** (1 / años) - 1


def max_drawdown(valor_serie: pd.Series) -> float:
    pico = valor_serie.cummax()
    dd = (valor_serie - pico) / pico
    return dd.min()


def sharpe(retornos: pd.Series) -> float:
    # Tasa libre de riesgo = 0 (simplificación explícita, no un supuesto oculto)
    return (retornos.mean() / retornos.std()) * np.sqrt(252)


def sortino(retornos: pd.Series) -> float:
    downside = retornos[retornos < 0]
    return (retornos.mean() / downside.std()) * np.sqrt(252)


def irr_periodica(flujos: list[float], periodos_por_año: int) -> float:
    """Rendimiento ponderado por dinero (IRR), la métrica correcta cuando
    el capital entra en momentos distintos (aportaciones periódicas) en
    vez de todo de golpe. `flujos` son negativos (aportaciones) salvo el
    último, que además suma el valor final del portafolio (se asume
    liquidado en la fecha de la última aportación — misma simplificación
    usada en el benchmark de DCA del Paso 1).

    Los límites de búsqueda de la tasa periódica se calculan dinámicamente
    a partir de len(flujos): un límite fijo (ej. -0.5 a 1.0, el que
    bastaba en el Paso 1 con ~320 periodos mensuales) se desborda con
    series más largas — (1+r)**i satura a 0 o a infinito en punto flotante
    cuando i crece (ej. ~1,400 periodos con aportaciones semanales)."""
    def van(tasa, flujos):
        return sum(f / (1 + tasa) ** i for i, f in enumerate(flujos))

    exponente_max = max(len(flujos) - 1, 1)
    limite = 250 / exponente_max  # margen bajo los ~308 órdenes de magnitud de un float64
    r_min = 10 ** (-limite) - 1
    r_max = 10 ** limite - 1
    tasa_periodica = brentq(lambda r: van(r, flujos), r_min, r_max)
    return (1 + tasa_periodica) ** periodos_por_año - 1
